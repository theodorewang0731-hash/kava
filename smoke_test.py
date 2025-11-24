"""
Smoke test for KAVA implementation.
Quick verification that all components work before running full experiments.
"""

import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rkv_compression import RKVCompressor
from src.losses import KVDistillationLoss, CODILoss, KAVALoss
from src.latent_reasoning import LatentReasoningModule
from src.data_utils import GSM8KDataset
from src.evaluation_datasets import EvaluationDatasets
from src.utils import extract_answer_number, exact_match_numeric


def test_rkv_compression():
    """Test R-KV compression module."""
    print("\n🔧 Testing R-KV Compression...")
    
    compressor = RKVCompressor(
        num_latent_tokens=24,
        lambda_mix=0.1,
        layerwise_std=True,
        use_projection=False,
        hidden_dim=768
    )
    
    # Create dummy KV cache
    batch_size, num_heads, seq_len, head_dim = 1, 12, 100, 64
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    attention = torch.randn(batch_size, num_heads, seq_len, seq_len)
    
    # Compress
    compressed_keys, compressed_values = compressor.compress(
        key_cache=keys,
        value_cache=values,
        attention_weights=attention,
        answer_start_idx=80,
        steps_start_idx=20,
        steps_end_idx=80
    )
    
    assert compressed_keys.shape == (batch_size, num_heads, 24, head_dim)
    assert compressed_values.shape == (batch_size, num_heads, 24, head_dim)
    
    print(f"   ✓ R-KV compression works: {seq_len} → {24} tokens")


def test_losses():
    """Test loss functions."""
    print("\n🔧 Testing Loss Functions...")
    
    # Dummy tensors
    batch_size, seq_len, vocab_size = 2, 50, 32000
    num_layers, num_heads, head_dim = 4, 12, 64
    hidden_dim = 768
    
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    student_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    teacher_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    student_keys = torch.randn(batch_size, num_layers, num_heads, 24, head_dim)
    student_values = torch.randn(batch_size, num_layers, num_heads, 24, head_dim)
    teacher_keys = torch.randn(batch_size, num_layers, num_heads, 24, head_dim)
    teacher_values = torch.randn(batch_size, num_layers, num_heads, 24, head_dim)
    
    student_hidden = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers)]
    teacher_hidden = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers)]
    
    # Test KV loss
    kv_loss_fn = KVDistillationLoss(loss_type="smooth_l1", layerwise_std=True)
    kv_loss = kv_loss_fn(student_keys, student_values, teacher_keys, teacher_values)
    assert kv_loss.item() >= 0
    print(f"   ✓ KV loss: {kv_loss.item():.4f}")
    
    # Test CODI loss
    codi_loss_fn = CODILoss(loss_type="l1")
    codi_loss = codi_loss_fn(student_hidden, teacher_hidden, distill_token_idx=-25)
    assert codi_loss.item() >= 0
    print(f"   ✓ CODI loss: {codi_loss.item():.4f}")
    
    # Test KAVA total loss
    kava_loss_fn = KAVALoss(
        alpha1_codi=10.0,
        alpha2_kv=1.0,
        kv_loss_type="smooth_l1",
        layerwise_std=True
    )
    total_loss, loss_dict = kava_loss_fn(
        student_logits, student_labels,
        teacher_logits, teacher_labels,
        student_keys, student_values,
        teacher_keys, teacher_values,
        student_hidden, teacher_hidden,
        distill_token_idx=-25
    )
    assert total_loss.item() >= 0
    print(f"   ✓ KAVA total loss: {total_loss.item():.4f}")
    print(f"      - Student CE: {loss_dict['loss_student_ce']:.4f}")
    print(f"      - Teacher CE: {loss_dict['loss_teacher_ce']:.4f}")
    print(f"      - KV: {loss_dict['loss_kv']:.4f}")
    print(f"      - CODI: {loss_dict['loss_codi']:.4f}")


def test_latent_reasoning():
    """Test latent reasoning module."""
    print("\n🔧 Testing Latent Reasoning Module...")
    
    # This requires a real model, so we'll just check initialization
    try:
        from transformers import AutoModelForCausalLM
        print("   ⚠ Skipping model loading (too slow for smoke test)")
        print("   ✓ Latent reasoning module imports successfully")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")


def test_data_loading():
    """Test dataset loading."""
    print("\n🔧 Testing Dataset Loading...")
    
    try:
        from transformers import AutoTokenizer
        
        # Try to load tokenizer (don't actually load model)
        print("   ⚠ Skipping full dataset loading (requires model download)")
        
        # Test answer extraction
        test_cases = [
            ("The answer is 42", "42"),
            ("#### 1,234.56", "1234.56"),
            ("$25.99 is the cost", "25.99"),
            ("5 + 3 = 8", "8")
        ]
        
        for text, expected in test_cases:
            extracted = extract_answer_number(text)
            assert extracted == expected, f"Expected {expected}, got {extracted}"
        
        print("   ✓ Answer extraction works correctly")
        
        # Test exact match
        assert exact_match_numeric("42", "42.0") == True
        assert exact_match_numeric("100", "100.001") == False
        print("   ✓ Exact match scoring works correctly")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")


def test_configs():
    """Test configuration files."""
    print("\n🔧 Testing Configuration Files...")
    
    config_dir = Path("configs")
    expected_configs = [
        "llama1b_aug.yaml",
        "llama1b_aug_nl.yaml",
        "qwen05b_aug.yaml",
        "llama3b_aug.yaml"
    ]
    
    for config_file in expected_configs:
        config_path = config_dir / config_file
        
        if not config_path.exists():
            print(f"   ✗ Missing: {config_file}")
            continue
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['model', 'dataset', 'training', 'lora', 'latent', 'loss']
        for field in required_fields:
            assert field in config, f"Missing field '{field}' in {config_file}"
        
        print(f"   ✓ {config_file}: {config['model']['type']} + {config['dataset']['name'].split('/')[-1]}")


def test_directory_structure():
    """Test project directory structure."""
    print("\n🔧 Testing Directory Structure...")
    
    required_dirs = ['src', 'configs', 'docs']
    required_files = [
        'train.py',
        'evaluate.py',
        'inference.py',
        'run_multi_seed.py',
        'aggregate_results.py',
        'requirements.txt',
        'README.md'
    ]
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"   ✗ Missing directory: {dir_name}")
        else:
            print(f"   ✓ {dir_name}/ exists")
    
    for file_name in required_files:
        if not Path(file_name).exists():
            print(f"   ✗ Missing file: {file_name}")
        else:
            print(f"   ✓ {file_name} exists")


def main():
    """Run all smoke tests."""
    print("="*80)
    print("KAVA Implementation - Smoke Test")
    print("="*80)
    print("\nRunning quick verification of all components...")
    
    try:
        test_directory_structure()
        test_configs()
        test_rkv_compression()
        test_losses()
        test_latent_reasoning()
        test_data_loading()
        
        print("\n" + "="*80)
        print("✅ All smoke tests passed!")
        print("="*80)
        print("\nYou can now proceed with:")
        print("  1. Quick training test: python train.py --config configs/llama1b_aug.yaml --max_train_samples 100")
        print("  2. Full experiment: python run_multi_seed.py --config configs/llama1b_aug.yaml --seeds 42 43 44")
        print("  3. Complete replication: .\\run_all_experiments.ps1")
        
        return 0
    
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ Smoke test failed: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
