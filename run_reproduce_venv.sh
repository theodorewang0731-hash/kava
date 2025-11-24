#!/bin/bash

################################################################################
# KAVA Paper Reproduction - Venv Version (No Conda Required)
################################################################################
# This script automates the complete KAVA reproduction workflow using Python venv
# instead of conda, suitable for HPC environments without conda/anaconda modules.
#
# Usage:
#   bash run_reproduce_venv.sh [--method direct|proxy|mirror] [--skip-download]
#
# Options:
#   --method METHOD    Model download method (default: auto-detect)
#                      - direct: Hugging Face (50-100 min)
#                      - proxy: Via proxy (17-35 min)
#                      - mirror: China mirror (33-68 min)
#   --skip-download    Skip model download if already cached
#   --skip-env         Skip environment setup if already exists
#   --help             Show this help message
################################################################################

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Environment
VENV_DIR="venv"

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
NC='\033[0m'

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
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN} $1${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

check_prerequisites() {
    log_section "Step 1/5: Pre-flight Checks"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    log_info "✓ Python $python_version found"
    
    # Check disk space
    local available_gb=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
    log_info "✓ Disk space: ${available_gb}GB available"
    
    if [ "$available_gb" -lt "$REQUIRED_SPACE_GB" ]; then
        log_error "Insufficient disk space. Need ${REQUIRED_SPACE_GB}GB, have ${available_gb}GB"
        exit 1
    fi
    
    # Check SLURM
    if ! command -v sbatch &> /dev/null; then
        log_error "SLURM (sbatch) not found. This script requires SLURM."
        exit 1
    fi
    log_info "✓ SLURM available"
    
    # Check required files
    local required_files=("requirements.txt" "submit_multi_seed.slurm" "train.py")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    log_info "✓ Required files present"
    
    log_success "Pre-flight checks passed"
}

# =============================================================================
# Environment Setup (venv)
# =============================================================================

setup_venv_environment() {
    if [ "$SKIP_ENV" = true ]; then
        log_info "Skipping environment setup (--skip-env specified)"
        return
    fi
    
    log_section "Step 2/5: Python Environment Setup (venv)"
    
    # Create venv if doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate venv
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment activated"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip -q
    
    # Install PyTorch with CUDA support
    log_info "Installing PyTorch with CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    
    # Install requirements
    log_info "Installing Python dependencies from requirements.txt..."
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt -q
    else
        log_error "requirements.txt not found"
        exit 1
    fi
    
    # Verify installation
    log_info "Verifying installation..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
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
    
    if [[ -n "${HTTP_PROXY:-}" ]] || [[ -n "${http_proxy:-}" ]]; then
        echo "proxy"
        return
    fi
    
    echo "mirror"
}

download_models() {
    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_info "Skipping model download (--skip-download specified)"
        return
    fi
    
    log_section "Step 3/5: Model Download"
    
    local method=$(detect_download_method)
    log_info "Using download method: $method"
    
    # Set HuggingFace endpoint based on method
    case "$method" in
        mirror)
            export HF_ENDPOINT="https://hf-mirror.com"
            log_info "Using mirror: hf-mirror.com"
            ;;
        proxy)
            log_info "Using proxy settings from environment"
            ;;
        direct)
            log_info "Using direct connection to huggingface.co"
            ;;
    esac
    
    # Download each model
    for model in "${MODELS[@]}"; do
        log_info "Downloading $model..."
        python -c "
from huggingface_hub import snapshot_download
import sys

try:
    snapshot_download(
        repo_id='$model',
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print('✓ Downloaded $model')
except Exception as e:
    print(f'✗ Failed to download $model: {e}', file=sys.stderr)
    sys.exit(1)
"
    done
    
    log_success "All models downloaded successfully"
}

# =============================================================================
# Job Submission
# =============================================================================

submit_training_jobs() {
    log_section "Step 4/5: SLURM Job Submission"
    
    local job_ids=()
    local total_jobs=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
    local job_count=0
    
    log_info "Submitting $total_jobs training jobs (${#CONFIGS[@]} configs × ${#SEEDS[@]} seeds)..."
    
    for config in "${CONFIGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_count=$((job_count + 1))
            
            # Extract config name for output naming
            local config_name=$(basename "$config" .yaml)
            local output_name="kava_${config_name}_seed${seed}"
            
            log_info "[$job_count/$total_jobs] Submitting: $config (seed=$seed)"
            
            # Submit job
            local job_id=$(sbatch \
                --job-name="$output_name" \
                --output="outputs/logs/${output_name}.out" \
                --error="outputs/logs/${output_name}.err" \
                --export=ALL,CONFIG="$config",SEED="$seed" \
                submit_multi_seed.slurm | awk '{print $4}')
            
            if [ -n "$job_id" ]; then
                job_ids+=("$job_id")
                log_success "Job submitted: $job_id"
            else
                log_error "Failed to submit job for $config (seed=$seed)"
            fi
            
            sleep 0.5  # Small delay to avoid overwhelming scheduler
        done
    done
    
    log_success "All jobs submitted: ${#job_ids[@]}/$total_jobs"
    
    # Save job IDs for monitoring
    printf "%s\n" "${job_ids[@]}" > outputs/job_ids.txt
    log_info "Job IDs saved to outputs/job_ids.txt"
}

# =============================================================================
# Generate Monitoring Scripts
# =============================================================================

generate_monitoring_scripts() {
    log_section "Step 5/5: Generate Monitoring Scripts"
    
    # Create monitor_jobs.sh
    cat > monitor_jobs.sh << 'EOF'
#!/bin/bash
# Monitor SLURM jobs for KAVA reproduction

if [ ! -f outputs/job_ids.txt ]; then
    echo "Error: outputs/job_ids.txt not found"
    exit 1
fi

echo "=== KAVA Training Job Status ==="
echo ""

mapfile -t job_ids < outputs/job_ids.txt

for job_id in "${job_ids[@]}"; do
    if squeue -j "$job_id" &> /dev/null; then
        status=$(squeue -j "$job_id" -o "%.18i %.9P %.20j %.8T %.10M %.6D" --noheader)
        echo "$status"
    else
        echo "$job_id - COMPLETED or FAILED (check logs)"
    fi
done

echo ""
echo "Total jobs: ${#job_ids[@]}"
echo "Running: $(squeue -u $USER | grep -c kava || echo 0)"
EOF
    
    chmod +x monitor_jobs.sh
    log_success "Created monitor_jobs.sh"
    
    # Create collect_results.sh
    cat > collect_results.sh << 'EOF'
#!/bin/bash
# Collect results from KAVA training runs

echo "=== Collecting KAVA Training Results ==="
echo ""

results_file="outputs/aggregated_results.csv"
echo "Config,Seed,EM,F1,Status" > "$results_file"

for log_file in outputs/logs/kava_*.out; do
    if [ -f "$log_file" ]; then
        config=$(basename "$log_file" | sed 's/kava_\(.*\)_seed[0-9]*.out/\1/')
        seed=$(basename "$log_file" | sed 's/.*seed\([0-9]*\).out/\1/')
        
        if grep -q "Final Test EM" "$log_file"; then
            em=$(grep "Final Test EM" "$log_file" | tail -1 | awk '{print $NF}')
            f1=$(grep "Final Test F1" "$log_file" | tail -1 | awk '{print $NF}')
            echo "$config,$seed,$em,$f1,COMPLETED" >> "$results_file"
        else
            echo "$config,$seed,N/A,N/A,IN_PROGRESS" >> "$results_file"
        fi
    fi
done

echo "Results saved to: $results_file"
echo ""
cat "$results_file" | column -t -s,
EOF
    
    chmod +x collect_results.sh
    log_success "Created collect_results.sh"
    
    log_info "Monitoring scripts ready:"
    log_info "  - bash monitor_jobs.sh    # Check job status"
    log_info "  - bash collect_results.sh  # Collect results after completion"
}

# =============================================================================
# Main Execution
# =============================================================================

parse_arguments() {
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
                head -n 20 "$0" | tail -n +2 | sed 's/^# //'
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

main() {
    log_section "KAVA Paper Reproduction - Automated Workflow (venv)"
    
    parse_arguments "$@"
    
    check_prerequisites
    setup_venv_environment
    download_models
    submit_training_jobs
    generate_monitoring_scripts
    
    log_section "Workflow Completed Successfully!"
    
    echo ""
    log_info "Next steps:"
    log_info "  1. Monitor jobs:    bash monitor_jobs.sh"
    log_info "  2. Check logs:      tail -f outputs/logs/kava_*.out"
    log_info "  3. After 36-48h:    bash collect_results.sh"
    echo ""
    log_success "Training jobs are now running on the cluster!"
}

main "$@"
