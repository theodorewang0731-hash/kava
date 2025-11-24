"""
Lightweight smoke test for KAVA - Verifies project structure only.
Use this on Windows or when dependencies are not installed.
For full tests, run smoke_test.py after installing requirements.txt
"""

import os
import yaml
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def log_success(msg):
    print(f"{GREEN}✓{NC} {msg}")

def log_error(msg):
    print(f"{RED}✗{NC} {msg}")

def log_info(msg):
    print(f"{BLUE}[INFO]{NC} {msg}")

def log_section(msg):
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{msg}{NC}")
    print(f"{BLUE}{'='*60}{NC}")


def test_directory_structure():
    """Test project directory structure."""
    log_section("Testing Directory Structure")
    
    required_dirs = {
        'src': 'Source code directory',
        'configs': 'Configuration files',
        'docs': 'Documentation',
    }
    
    optional_dirs = {
        'outputs': 'Output directory (created during training)',
        'outputs/logs': 'Log files',
        'outputs/results': 'Result files',
    }
    
    required_files = {
        'train.py': 'Main training script',
        'evaluate.py': 'Evaluation script',
        'inference.py': 'Inference script',
        'run_multi_seed.py': 'Multi-seed runner',
        'aggregate_multi_seed.py': 'Results aggregation',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Main documentation',
        'smoke_test.py': 'Full smoke test',
        'run_everything.sh': 'One-click automation script',
        'REPRODUCTION_CHECKLIST.md': 'Quick start checklist',
    }
    
    errors = 0
    
    # Check required directories
    for dir_name, desc in required_dirs.items():
        if Path(dir_name).exists():
            log_success(f"{dir_name}/ - {desc}")
        else:
            log_error(f"Missing: {dir_name}/ - {desc}")
            errors += 1
    
    # Check optional directories (warnings only)
    for dir_name, desc in optional_dirs.items():
        if Path(dir_name).exists():
            log_success(f"{dir_name}/ - {desc}")
        else:
            print(f"{YELLOW}⚠{NC} Optional: {dir_name}/ - {desc}")
    
    # Check required files
    for file_name, desc in required_files.items():
        if Path(file_name).exists():
            log_success(f"{file_name} - {desc}")
        else:
            log_error(f"Missing: {file_name} - {desc}")
            errors += 1
    
    return errors == 0


def test_src_modules():
    """Test source code modules."""
    log_section("Testing Source Modules")
    
    src_files = {
        'src/__init__.py': 'Package initialization',
        'src/rkv_compression.py': 'R-KV compression module',
        'src/losses.py': 'Loss functions (KAVA, CODI, KV)',
        'src/latent_reasoning.py': 'Latent reasoning module',
        'src/data_utils.py': 'Dataset utilities',
        'src/evaluation_datasets.py': 'Evaluation datasets',
        'src/utils.py': 'Helper utilities',
        'src/trainer.py': 'KAVA trainer',
    }
    
    errors = 0
    for file_name, desc in src_files.items():
        if Path(file_name).exists():
            log_success(f"{file_name} - {desc}")
        else:
            log_error(f"Missing: {file_name} - {desc}")
            errors += 1
    
    return errors == 0


def test_configs():
    """Test configuration files."""
    log_section("Testing Configuration Files")
    
    config_dir = Path("configs")
    expected_configs = {
        "llama1b_aug.yaml": "Llama-3.2-1B with augmentation",
        "llama1b_aug_nl.yaml": "Llama-3.2-1B with NL augmentation",
        "qwen05b_aug.yaml": "Qwen2.5-0.5B with augmentation",
        "llama3b_aug.yaml": "Llama-3.2-3B with augmentation",
    }
    
    errors = 0
    for config_file, desc in expected_configs.items():
        config_path = config_dir / config_file
        
        if not config_path.exists():
            log_error(f"Missing: {config_file} - {desc}")
            errors += 1
            continue
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['model', 'dataset', 'training', 'lora', 'latent', 'loss']
            missing_fields = [f for f in required_fields if f not in config]
            
            if missing_fields:
                log_error(f"{config_file}: Missing fields {missing_fields}")
                errors += 1
            else:
                model_name = config['model']['type'].split('/')[-1]
                dataset_name = config['dataset']['name'].split('/')[-1]
                log_success(f"{config_file} - {model_name} + {dataset_name}")
        
        except Exception as e:
            log_error(f"{config_file}: Parse error - {e}")
            errors += 1
    
    return errors == 0


def test_documentation():
    """Test documentation files."""
    log_section("Testing Documentation")
    
    docs = {
        'README.md': 'Main README',
        'REPRODUCTION_CHECKLIST.md': 'Quick start checklist',
        'docs/GETTING_STARTED_HPC.md': 'HPC getting started guide',
        'docs/KAVA_MODEL_DOWNLOAD.md': 'Model download instructions',
        'docs/HPC_REFERENCE.md': 'HPC reference',
        'docs/SLURM_INTERACTIVE_GUIDE.md': 'SLURM interactive guide',
    }
    
    warnings = 0
    for doc_file, desc in docs.items():
        if Path(doc_file).exists():
            size_kb = Path(doc_file).stat().st_size / 1024
            log_success(f"{doc_file} ({size_kb:.1f} KB) - {desc}")
        else:
            print(f"{YELLOW}⚠{NC} Missing: {doc_file} - {desc}")
            warnings += 1
    
    return warnings == 0


def test_slurm_scripts():
    """Test SLURM submission scripts."""
    log_section("Testing SLURM Scripts")
    
    slurm_scripts = {
        'submit_multi_seed.slurm': 'Multi-seed SLURM job script',
        'hpc_run_all.sh': 'Batch submission script',
        'run_everything.sh': 'One-click automation script',
    }
    
    errors = 0
    for script_file, desc in slurm_scripts.items():
        if Path(script_file).exists():
            log_success(f"{script_file} - {desc}")
        else:
            log_error(f"Missing: {script_file} - {desc}")
            errors += 1
    
    return errors == 0


def test_requirements():
    """Test requirements.txt."""
    log_section("Testing Dependencies")
    
    if not Path('requirements.txt').exists():
        log_error("requirements.txt not found")
        return False
    
    with open('requirements.txt', 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    log_info(f"Found {len(lines)} dependencies")
    
    critical_deps = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'peft': 'Parameter-Efficient Fine-Tuning',
        'datasets': 'Hugging Face Datasets',
        'accelerate': 'Training acceleration',
    }
    
    missing = []
    for dep, desc in critical_deps.items():
        found = any(dep in line for line in lines)
        if found:
            log_success(f"{dep} - {desc}")
        else:
            log_error(f"Missing dependency: {dep} - {desc}")
            missing.append(dep)
    
    if missing:
        log_error(f"Add these to requirements.txt: {', '.join(missing)}")
        return False
    
    return True


def test_paper_compliance():
    """Test paper reproduction compliance."""
    log_section("Testing Paper Compliance")
    
    # Check configs for paper hyperparameters
    config_dir = Path("configs")
    configs_ok = True
    
    for config_file in ["llama1b_aug.yaml", "qwen05b_aug.yaml"]:
        config_path = config_dir / config_file
        if not config_path.exists():
            continue
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        checks = [
            ('lora.r', 128, "LoRA rank"),
            ('lora.alpha', 32, "LoRA alpha"),
            ('latent.num_tokens', 24, "Latent tokens"),
            ('latent.num_iterations', 3, "Jacobi iterations"),
        ]
        
        for key_path, expected, desc in checks:
            keys = key_path.split('.')
            value = config
            for k in keys:
                value = value.get(k, None)
                if value is None:
                    break
            
            if value == expected:
                log_success(f"{config_file}: {desc} = {expected}")
            else:
                print(f"{YELLOW}⚠{NC} {config_file}: {desc} = {value} (expected {expected})")
                configs_ok = False
    
    return configs_ok


def main():
    """Run all lightweight tests."""
    print("\n" + "="*80)
    print("KAVA Implementation - Lightweight Smoke Test")
    print("="*80)
    print("\nThis test verifies project structure without loading dependencies.")
    print("For full testing, install requirements.txt and run smoke_test.py")
    print("="*80)
    
    results = {
        "Directory Structure": test_directory_structure(),
        "Source Modules": test_src_modules(),
        "Configuration Files": test_configs(),
        "Documentation": test_documentation(),
        "SLURM Scripts": test_slurm_scripts(),
        "Dependencies": test_requirements(),
        "Paper Compliance": test_paper_compliance(),
    }
    
    # Summary
    log_section("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = f"{GREEN}PASS{NC}" if result else f"{RED}FAIL{NC}"
        print(f"  {name}: {status}")
    
    print("\n" + "="*80)
    if passed == total:
        print(f"{GREEN}✅ All {total} tests passed!{NC}")
        print("="*80)
        print("\nProject structure is correct. Next steps:")
        print("  1. On HPC: bash run_reproduce.sh")
        print("  2. Locally: Install requirements.txt and run smoke_test.py")
        print("  3. Quick test: python train.py --config configs/llama1b_aug.yaml --max_train_samples 10")
        return 0
    else:
        print(f"{RED}❌ {total - passed}/{total} tests failed{NC}")
        print("="*80)
        print("\nFix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
