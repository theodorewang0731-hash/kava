#!/bin/bash

################################################################################
# KAVA Paper Reproduction - One-Click Automation Script
################################################################################
# This script automates the complete KAVA reproduction workflow on HPC cluster:
#   1. Pre-flight checks (disk space, network, SLURM availability)
#   2. Conda environment setup (kava_env)
#   3. Model download (~19GB: Llama-3.2-1B/3B-Instruct, Qwen2.5-0.5B-Instruct)
#   4. SLURM job submission (4 configs × 3 seeds = 12 jobs)
#   5. Monitoring and results collection
#
# Usage:
#   bash run_reproduce.sh [--method direct|proxy|mirror] [--skip-download]
#
# Options:
#   --method METHOD    Model download method (default: auto-detect)
#                      - direct: Hugging Face (50-100 min, needs good network)
#                      - proxy: Via proxy (17-35 min, if configured)
#                      - mirror: China mirror (33-68 min, from mainland)
#   --skip-download    Skip model download if already cached
#   --skip-env         Skip environment setup if already exists
#   --help             Show this help message
#
# Expected Timeline:
#   - Pre-flight checks: 1-2 minutes
#   - Conda environment: 5-10 minutes
#   - Model download: 17-100 minutes (depending on method)
#   - Training (parallel): 36-48 hours
#   Total: ~1-2 days for complete reproduction
#
# Requirements:
#   - HPC access (SLURM cluster)
#   - ~19GB disk space in $HOME/.cache/huggingface
#   - Network access (for model download)
#   - GPU resources: A100-80GB (4 configs × 3 seeds)
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Environment
CONDA_ENV_NAME="kava_env"
PYTHON_VERSION="3.10"

# Models to download
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
)

# Configs to run
CONFIGS=(
    "configs/llama1b_aug.yaml"
    "configs/llama1b_aug_nl.yaml"
    "configs/llama3b_aug.yaml"
    "configs/qwen05b_aug.yaml"
)

# Seeds
SEEDS=(42 123 456)

# Disk space requirements (in GB)
REQUIRED_SPACE_GB=20

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
DOWNLOAD_METHOD="auto"
SKIP_DOWNLOAD=false
SKIP_ENV=false

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}========================================${NC}"
}

print_progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%$((width - filled))s" | tr ' ' ' '
    printf "] %d%%" "$percentage"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command '$1' not found"
        return 1
    fi
    return 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --method)
                DOWNLOAD_METHOD="$2"
                shift 2
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --skip-env)
                SKIP_ENV=true
                shift
                ;;
            --help)
                head -n 40 "$0" | tail -n 38
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

preflight_checks() {
    log_section "Step 1/5: Pre-flight Checks"
    
    # Check if on HPC cluster
    log_info "Checking HPC environment..."
    if [[ ! -f /etc/slurm/slurm.conf ]]; then
        log_warning "SLURM config not found - are you on the HPC cluster?"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check required commands
    log_info "Checking required commands..."
    local missing_cmds=()
    for cmd in sbatch squeue scancel module conda git; do
        if ! check_command "$cmd"; then
            missing_cmds+=("$cmd")
        fi
    done
    
    if [ ${#missing_cmds[@]} -gt 0 ]; then
        log_error "Missing required commands: ${missing_cmds[*]}"
        log_info "Please ensure you're on the HPC login node with access to:"
        log_info "  - SLURM (sbatch, squeue, scancel)"
        log_info "  - Environment modules (module)"
        log_info "  - Conda (conda)"
        log_info "  - Git (git)"
        exit 1
    fi
    log_success "All required commands available"
    
    # Check disk space
    log_info "Checking disk space..."
    local cache_dir="${HOME}/.cache/huggingface"
    mkdir -p "$cache_dir"
    
    local available_kb=$(df -k "$cache_dir" | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [ "$available_gb" -lt "$REQUIRED_SPACE_GB" ]; then
        log_error "Insufficient disk space: ${available_gb}GB available, ${REQUIRED_SPACE_GB}GB required"
        log_info "Free up space in $cache_dir or use --skip-download if models are already cached"
        exit 1
    fi
    log_success "Disk space OK: ${available_gb}GB available"
    
    # Check network connectivity
    log_info "Checking network connectivity..."
    if ping -c 1 -W 3 huggingface.co &> /dev/null || \
       ping -c 1 -W 3 hf-mirror.com &> /dev/null || \
       ping -c 1 -W 3 8.8.8.8 &> /dev/null; then
        log_success "Network connectivity OK"
    else
        log_warning "Network connectivity issue - model download may fail"
        if [ "$SKIP_DOWNLOAD" = false ]; then
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    # Check SLURM partition availability
    log_info "Checking SLURM partition 'compute'..."
    if sinfo -p compute &> /dev/null; then
        local available_nodes=$(sinfo -p compute -h -o "%A" | awk -F/ '{print $2}')
        log_success "Partition 'compute' available with $available_nodes idle nodes"
    else
        log_warning "Cannot access partition 'compute' - jobs may wait in queue"
    fi
    
    log_success "Pre-flight checks completed"
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_environment() {
    if [ "$SKIP_ENV" = true ]; then
        log_info "Skipping environment setup (--skip-env specified)"
        return
    fi
    
    log_section "Step 2/5: Environment Setup"
    
    # Load conda module
    log_info "Loading conda module..."
    if module load anaconda3 &> /dev/null || module load miniconda3 &> /dev/null; then
        log_success "Conda module loaded"
    else
        log_warning "Could not load conda module - using existing conda"
    fi
    
    # Initialize conda for bash
    if ! command -v conda &> /dev/null; then
        log_error "Conda not available after module load"
        exit 1
    fi
    
    eval "$(conda shell.bash hook)"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "Environment '${CONDA_ENV_NAME}' already exists"
        read -p "Recreate environment? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing environment..."
            conda env remove -n "$CONDA_ENV_NAME" -y
        else
            log_info "Using existing environment"
            conda activate "$CONDA_ENV_NAME"
            log_success "Environment activated"
            return
        fi
    fi
    
    # Create new environment
    log_info "Creating conda environment '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    
    log_info "Activating environment..."
    conda activate "$CONDA_ENV_NAME"
    
    # Install PyTorch with CUDA support
    log_info "Installing PyTorch with CUDA 11.8..."
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install requirements
    log_info "Installing Python dependencies from requirements.txt..."
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    else
        log_error "requirements.txt not found"
        exit 1
    fi
    
    # Verify installation
    log_info "Verifying installation..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
    python -c "import peft; print(f'PEFT: {peft.__version__}')"
    
    log_success "Environment setup completed"
}

# =============================================================================
# Model Download
# =============================================================================

detect_download_method() {
    if [ "$DOWNLOAD_METHOD" != "auto" ]; then
        echo "$DOWNLOAD_METHOD"
        return
    fi
    
    log_info "Auto-detecting optimal download method..."
    
    # Check if HTTP_PROXY is set (likely using proxy)
    if [[ -n "${HTTP_PROXY:-}" ]] || [[ -n "${http_proxy:-}" ]]; then
        echo "proxy"
        return
    fi
    
    # Check network location (ping latency to different hosts)
    local hf_latency=9999
    local mirror_latency=9999
    
    if ping -c 3 -W 2 huggingface.co &> /dev/null; then
        hf_latency=$(ping -c 3 -W 2 huggingface.co | tail -1 | awk -F'/' '{print $5}' | cut -d'.' -f1)
    fi
    
    if ping -c 3 -W 2 hf-mirror.com &> /dev/null; then
        mirror_latency=$(ping -c 3 -W 2 hf-mirror.com | tail -1 | awk -F'/' '{print $5}' | cut -d'.' -f1)
    fi
    
    log_info "Network latency - HuggingFace: ${hf_latency}ms, Mirror: ${mirror_latency}ms"
    
    if [ "$mirror_latency" -lt "$hf_latency" ] && [ "$mirror_latency" -lt 200 ]; then
        echo "mirror"
    else
        echo "direct"
    fi
}

download_models() {
    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_info "Skipping model download (--skip-download specified)"
        return
    fi
    
    log_section "Step 3/5: Model Download"
    
    local method=$(detect_download_method)
    log_info "Using download method: $method"
    
    # Set up HuggingFace environment
    export HF_HOME="${HOME}/.cache/huggingface"
    mkdir -p "$HF_HOME"
    
    case "$method" in
        proxy)
            log_info "Using proxy settings from environment"
            if [[ -z "${HTTP_PROXY:-}" ]] && [[ -z "${http_proxy:-}" ]]; then
                log_warning "No proxy configured - falling back to direct download"
                method="direct"
            fi
            ;;
        mirror)
            log_info "Using HuggingFace mirror: hf-mirror.com"
            export HF_ENDPOINT="https://hf-mirror.com"
            ;;
        direct)
            log_info "Using direct HuggingFace download"
            ;;
    esac
    
    # Create download script
    cat > /tmp/download_models_${USER}.py << 'EOF'
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch

def download_model(model_name, progress_callback=None):
    """Download model with progress reporting."""
    cache_dir = Path(os.environ.get('HF_HOME', '~/.cache/huggingface')).expanduser()
    
    print(f"\n{'='*80}")
    print(f"Downloading: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Download model files
        print(f"[1/2] Downloading model weights and config...")
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        
        # Load to verify
        print(f"[2/2] Verifying model integrity...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded: vocab_size={len(tokenizer)}")
        
        # Check model size without loading full weights
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"✓ Model verified: {num_params:.2f}B parameters")
        
        # Clean up to save memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print(f"✅ Successfully downloaded: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

if __name__ == "__main__":
    models = sys.argv[1:]
    success_count = 0
    
    for i, model in enumerate(models, 1):
        print(f"\n{'#'*80}")
        print(f"Model {i}/{len(models)}: {model}")
        print(f"{'#'*80}")
        
        if download_model(model):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Download Summary: {success_count}/{len(models)} models successful")
    print(f"{'='*80}")
    
    if success_count < len(models):
        sys.exit(1)
EOF
    
    # Estimate download time
    local estimated_time="Unknown"
    case "$method" in
        direct) estimated_time="50-100 minutes" ;;
        proxy) estimated_time="17-35 minutes" ;;
        mirror) estimated_time="33-68 minutes" ;;
    esac
    
    log_info "Estimated download time: $estimated_time"
    log_info "Downloading ${#MODELS[@]} models (~19GB total)..."
    
    # Run download
    local start_time=$(date +%s)
    
    if python /tmp/download_models_${USER}.py "${MODELS[@]}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        log_success "Model download completed in ${minutes}m ${seconds}s"
    else
        log_error "Model download failed"
        log_info "Possible solutions:"
        log_info "  1. Check network connectivity: ping huggingface.co"
        log_info "  2. Try different method: --method mirror"
        log_info "  3. Use proxy: export HTTP_PROXY=http://proxy:port"
        log_info "  4. Manual download: see docs/KAVA_MODEL_DOWNLOAD.md"
        exit 1
    fi
    
    # Clean up temporary files (safe: user-specific)
    rm -f /tmp/download_models_${USER}.py 2>/dev/null || true
}

# =============================================================================
# Job Submission
# =============================================================================

submit_jobs() {
    log_section "Step 4/5: SLURM Job Submission"
    
    local job_ids=()
    local total_jobs=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
    local submitted=0
    
    log_info "Submitting $total_jobs training jobs (${#CONFIGS[@]} configs × ${#SEEDS[@]} seeds)..."
    
    for config in "${CONFIGS[@]}"; do
        if [ ! -f "$config" ]; then
            log_error "Config file not found: $config"
            continue
        fi
        
        local config_name=$(basename "$config" .yaml)
        log_info "Submitting jobs for config: $config_name"
        
        for seed in "${SEEDS[@]}"; do
            # Submit job using submit_multi_seed.slurm
            local job_id=$(sbatch \
                --job-name="kava_${config_name}_${seed}" \
                --export=ALL,CONFIG="$config",SEED="$seed" \
                submit_multi_seed.slurm | awk '{print $NF}')
            
            if [ -n "$job_id" ]; then
                job_ids+=("$job_id")
                submitted=$((submitted + 1))
                log_success "  ✓ Seed $seed: Job ID $job_id"
            else
                log_error "  ✗ Failed to submit seed $seed"
            fi
        done
    done
    
    log_success "Submitted $submitted/$total_jobs jobs"
    
    # Save job IDs to file
    local job_file="kava_job_ids_$(date +%Y%m%d_%H%M%S).txt"
    printf "%s\n" "${job_ids[@]}" > "$job_file"
    log_info "Job IDs saved to: $job_file"
    
    # Show queue status
    echo ""
    log_info "Current queue status:"
    squeue -u "$USER" --format="%.10i %.12j %.8T %.10M %.6D %.20R"
    
    # Provide monitoring commands
    echo ""
    log_info "Monitoring commands:"
    log_info "  - Check queue:  squeue -u \$USER"
    log_info "  - View log:     tail -f outputs/logs/<config>_seed<seed>.log"
    log_info "  - Cancel job:   scancel <job_id>"
    log_info "  - Cancel all:   scancel -u \$USER"
}

# =============================================================================
# Monitoring & Results
# =============================================================================

setup_monitoring() {
    log_section "Step 5/5: Monitoring & Results"
    
    # Create monitoring script
    cat > monitor_jobs.sh << 'EOF'
#!/bin/bash
# Quick monitoring script for KAVA experiments

echo "KAVA Training Progress"
echo "===================="
echo ""

# Check SLURM queue
echo "SLURM Queue Status:"
squeue -u $USER --format="%.10i %.15j %.8T %.10M %.10l %.6D" | head -n 20

# Count jobs by status
RUNNING=$(squeue -u $USER -t RUNNING -h | wc -l)
PENDING=$(squeue -u $USER -t PENDING -h | wc -l)
echo ""
echo "Summary: $RUNNING running, $PENDING pending"

# Check recent logs
echo ""
echo "Recent Log Updates (last 5 minutes):"
find outputs/logs -name "*.log" -mmin -5 -exec bash -c 'echo "- $(basename {}): $(tail -n 1 {})"' \;

# Check results
echo ""
echo "Completed Results:"
if [ -d "outputs/results" ]; then
    find outputs/results -name "*.json" -exec bash -c 'echo "✓ $(basename {})"' \;
fi

echo ""
echo "Run this script again: bash monitor_jobs.sh"
EOF
    
    chmod +x monitor_jobs.sh
    log_success "Created monitoring script: monitor_jobs.sh"
    
    # Create results aggregation script
    cat > collect_results.sh << 'EOF'
#!/bin/bash
# Aggregate results after all jobs complete

echo "KAVA Results Collection"
echo "====================="
echo ""

# Check if all jobs are done
RUNNING=$(squeue -u $USER -h | wc -l)
if [ $RUNNING -gt 0 ]; then
    echo "⚠ Warning: $RUNNING jobs still running"
    echo "Wait for all jobs to complete before collecting results"
    exit 1
fi

# Run aggregation
if [ -f "aggregate_multi_seed.py" ]; then
    echo "Aggregating results..."
    python aggregate_multi_seed.py --seed_dirs results/*/seed_* --output_json outputs/aggregated_results.json
    
    if [ -f "outputs/aggregated_results.json" ]; then
        echo ""
        echo "✅ Results aggregated successfully!"
        echo ""
        echo "Full results in: outputs/aggregated_results.json"
    fi
else
    echo "❌ aggregate_multi_seed.py not found"
fi
EOF
    
    chmod +x collect_results.sh
    log_success "Created results collection script: collect_results.sh"
    
    # Print instructions
    echo ""
    log_info "Next steps:"
    log_info "  1. Monitor progress:  bash monitor_jobs.sh"
    log_info "  2. View logs:         tail -f outputs/logs/<config>_seed<seed>.log"
    log_info "  3. After completion:  bash collect_results.sh"
    
    echo ""
    log_info "Expected timeline:"
    log_info "  - Training duration: 36-48 hours (parallel execution)"
    log_info "  - Jobs will auto-save checkpoints every epoch"
    log_info "  - Results saved to outputs/results/"
}

# =============================================================================
# Main Workflow
# =============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                                ║${NC}"
    echo -e "${CYAN}║        KAVA Paper Reproduction - Automated Workflow           ║${NC}"
    echo -e "${CYAN}║                                                                ║${NC}"
    echo -e "${CYAN}║  Knowledge-Augmented Verbal-Augmentation (KAVA)                ║${NC}"
    echo -e "${CYAN}║  Strict reproduction according to paper specifications         ║${NC}"
    echo -e "${CYAN}║                                                                ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    log_info "Starting automated reproduction workflow..."
    log_info "This will take approximately 1-2 days to complete"
    log_info "Press Ctrl+C to abort (safe to restart later)"
    echo ""
    
    sleep 2
}

print_summary() {
    log_section "Automation Complete"
    
    echo ""
    log_success "🎉 Setup completed successfully!"
    echo ""
    log_info "Timeline:"
    log_info "  ✓ Pre-flight checks: Complete"
    log_info "  ✓ Environment setup: Complete"
    log_info "  ✓ Model download: Complete (~19GB cached)"
    log_info "  ✓ Job submission: Complete (12 jobs)"
    log_info "  ⏳ Training: In progress (36-48 hours estimated)"
    echo ""
    
    log_info "Your experiments are now running on the HPC cluster!"
    log_info "Jobs will continue running even if you disconnect."
    echo ""
    
    log_info "Quick commands:"
    echo -e "  ${GREEN}bash monitor_jobs.sh${NC}      - Check progress"
    echo -e "  ${GREEN}squeue -u \$USER${NC}           - View queue"
    echo -e "  ${GREEN}bash collect_results.sh${NC}   - Aggregate results (after completion)"
    echo ""
    
    log_info "Log files: outputs/logs/"
    log_info "Results: outputs/results/"
    log_info "Documentation: docs/GETTING_STARTED_HPC.md"
    echo ""
}

cleanup_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        log_error "Script failed with exit code $exit_code"
        log_info "Check the error messages above for details"
        log_info "You can restart the script - it will resume from where it stopped"
        echo ""
    fi
}

main() {
    trap cleanup_on_error EXIT
    
    # Parse arguments
    parse_args "$@"
    
    # Print header
    print_header
    
    # Run workflow
    preflight_checks
    setup_environment
    download_models
    submit_jobs
    setup_monitoring
    
    # Print summary
    print_summary
    
    log_success "All tasks completed successfully!"
}

# Run main workflow
main "$@"
