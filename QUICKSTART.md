# KAVA Quick Start Guide

## 🎯 5-Minute Setup

### Step 1: Install Dependencies

```powershell
# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### Step 2: Verify Installation

```powershell
# Quick test to ensure imports work
python -c "from src import KAVATrainer, RKVCompressor; print('✓ Installation successful')"
```

### Step 3: Run a Tiny Test

```powershell
# Test on 10 samples (should complete in 5-10 minutes)
python train.py `
    --config configs/llama1b_aug.yaml `
    --seed 42
```

## 📊 Running Experiments

### Single Model Training

**Option A: LLaMA 3.2-1B on GSM8k-AUG (Equation CoT)**
```powershell
python train.py --config configs/llama1b_aug.yaml --wandb
```

**Option B: LLaMA 3.2-1B on GSM8k-AUG-NL (Natural Language CoT)**
```powershell
python train.py --config configs/llama1b_aug_nl.yaml --wandb
```

**Option C: Qwen2.5-0.5B on GSM8k-AUG**
```powershell
python train.py --config configs/qwen05b_aug.yaml --wandb
```

**Option D: LLaMA 3.2-3B on GSM8k-AUG**
```powershell
python train.py --config configs/llama3b_aug.yaml --wandb
```

### Evaluation

After training, evaluate your model:

```powershell
python evaluate.py `
    --checkpoint checkpoints/llama-gsm8k_aug-epoch10 `
    --config configs/llama1b_aug.yaml `
    --datasets gsm8k gsm8k-hard svamp `
    --output results/my_results.yaml
```

### Full Replication (3 Seeds × 4 Configs)

```powershell
# This runs ALL experiments from the paper
.\scripts\run_all_experiments.ps1
```

⏱️ **Estimated time:** 
- LLaMA-1B: ~24 hours per run (10 epochs)
- LLaMA-3B: ~12 hours per run (5 epochs)
- Qwen-0.5B: ~20 hours per run (10 epochs)
- **Total for full replication:** ~200-300 GPU hours

## 🔍 Monitoring Training

### With Weights & Biases

```powershell
# Enable W&B logging
python train.py --config configs/llama1b_aug.yaml --wandb
```

View at: https://wandb.ai/your-username/kava-reproduction

### Key Metrics to Watch

| Metric | Expected Behavior |
|--------|-------------------|
| `loss_total` | Should decrease steadily |
| `loss_student_ce` | Should decrease (student learning) |
| `loss_teacher_ce` | Should remain stable |
| `loss_kv` | Should decrease (KV alignment improving) |
| `loss_codi` | Should decrease (hidden state alignment) |

## 🛠️ Troubleshooting

### Out of Memory

**Solution 1: Reduce batch size**
```yaml
# In config file
training:
  batch_size: 64  # Instead of 128
  gradient_accumulation_steps: 2  # Maintain effective batch size
```

**Solution 2: Use gradient checkpointing**
```python
# In trainer.py, after loading model
model.gradient_checkpointing_enable()
```

### Slow Training

**Check GPU utilization:**
```powershell
nvidia-smi -l 1
```

If GPU is underutilized:
- Increase batch size
- Reduce logging frequency
- Use torch.compile (PyTorch 2.0+)

### Dataset Download Issues

If HuggingFace download fails:

```python
# Set proxy or cache
import os
os.environ['HF_HOME'] = 'path/to/cache'
os.environ['HF_DATASETS_CACHE'] = 'path/to/datasets'
```

## 📈 Interpreting Results

### Accuracy Comparison

Expected accuracy ranges (based on paper):

| Model | GSM8k | GSM8k-Hard | SVAMP |
|-------|-------|------------|-------|
| LLaMA-1B AUG | 45-50% | 30-35% | 40-45% |
| LLaMA-1B AUG-NL | 43-48% | 28-33% | 38-43% |
| Qwen-0.5B AUG | 38-43% | 25-30% | 35-40% |
| LLaMA-3B AUG | 55-60% | 35-40% | 50-55% |

*(Adjust based on actual Table 1 values)*

### Forward Passes

Latent reasoning should use **significantly fewer forward passes** than full CoT:

- **Latent (KAVA):** T=3 iterations + answer tokens ≈ 10-20 FP
- **Full CoT:** CoT tokens + answer tokens ≈ 50-150 FP

**Speedup:** ~3-10x reduction

## 🎓 Understanding the Config Files

### Key Hyperparameters Explained

```yaml
# α₁: Weight for CODI loss (hidden state distillation)
# Higher = stronger supervision from teacher's hidden states
loss:
  alpha1_codi: 10.0  # Default for 1B models

# α₂: Weight for KV loss (KV-cache distillation)  
# Higher = stronger alignment of KV representations
loss:
  alpha2_kv: 1.0  # Default for 1B models

# λ: Balance between importance and redundancy in R-KV
# Higher = favor importance, Lower = favor diversity
rkv:
  lambda: 0.1  # Default (90% redundancy, 10% importance)

# Loss type: Choose based on model/dataset
loss:
  kv_loss_type: "smooth_l1"  # or "mse"
  layerwise_std: true  # Normalize by layer statistics
```

### When to Adjust

**Increase α₁** if:
- Student is learning too slowly
- Want stronger teacher guidance

**Increase α₂** if:
- KV alignment is poor
- Latent representations are unstable

**Decrease λ** if:
- Want more diverse latent tokens
- Redundancy removal is too aggressive

## 🚀 Advanced Usage

### Custom Dataset

```python
from src.data_utils import GSM8KDataset

# Add your own dataset
dataset = GSM8KDataset(
    dataset_name="your-org/your-dataset",
    tokenizer=tokenizer,
    cot_type="equation"
)
```

### Multi-GPU Training

```powershell
# Use accelerate
accelerate config
accelerate launch train.py --config configs/llama1b_aug.yaml
```

### Hyperparameter Sweep

```python
# Modify config and run multiple times
for alpha1 in [5, 10, 15, 20]:
    config['loss']['alpha1_codi'] = alpha1
    trainer = KAVATrainer(config)
    trainer.train()
```

## 📞 Getting Help

1. **Check CHECKLIST.md** for common issues
2. **Review paper** Section 3 and Table 6
3. **Open GitHub issue** with:
   - Config file used
   - Error message
   - GPU/system info

## ✅ Before Full Replication

Run this sanity check:

```powershell
# 1. Train for 1 epoch on small subset
python train.py --config configs/llama1b_aug.yaml --seed 42

# 2. Check losses decreased
# Expected: loss_total < 2.0 after 1 epoch

# 3. Test evaluation
python evaluate.py `
    --checkpoint checkpoints/llama-gsm8k_aug-epoch1 `
    --config configs/llama1b_aug.yaml `
    --datasets gsm8k `
    --max_samples 50

# 4. Verify accuracy > 0
# Expected: At least some correct answers (>5%)

# 5. If all checks pass, proceed with full training!
```

## 🎉 Success Criteria

Your replication is successful if:

✅ Training losses decrease consistently  
✅ Validation accuracy is within 2-3% of paper  
✅ Forward passes are 3-10x fewer than full CoT  
✅ Results are consistent across 3 seeds (std < 2%)  

Good luck! 🚀
