# KAVA Usage Examples

Practical examples for common tasks with KAVA.

## 1. Training

### Example 1.1: Basic Training

Train LLaMA 3.2-1B on GSM8k-AUG with single seed:

```bash
python train.py --config configs/llama1b_aug.yaml --seed 42
```

**Output**: Checkpoint saved to `experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10/`

### Example 1.2: Training with W&B Logging

Enable Weights & Biases tracking:

```bash
python train.py --config configs/llama1b_aug.yaml --seed 42 --wandb
```

View at: https://wandb.ai

### Example 1.3: Resume Training

Resume from checkpoint:

```bash
python train.py \
    --config configs/llama1b_aug.yaml \
    --seed 42 \
    --resume_from_checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch5
```

---

## 2. Evaluation

### Example 2.1: Evaluate on GSM8k

Evaluate single checkpoint:

```bash
python evaluate.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --datasets gsm8k \
    --output results/gsm8k.yaml
```

**Output** (`results/gsm8k.yaml`):
```yaml
gsm8k:
  accuracy: 0.8245
  avg_forward_passes: 48.2
  std_forward_passes: 12.3
  total_examples: 1319
```

### Example 2.2: Evaluate on Multiple Datasets

Evaluate on GSM8k, GSM8k-Hard, and SVAMP:

```bash
python evaluate.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --datasets gsm8k gsm8k-hard svamp \
    --output results/all_datasets.yaml
```

### Example 2.3: Compare Checkpoints

Evaluate multiple epochs:

```bash
for epoch in 5 7 10; do
    python evaluate.py \
        --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch${epoch} \
        --config configs/llama1b_aug.yaml \
        --datasets gsm8k \
        --output results/epoch${epoch}.yaml
done
```

Then compare accuracies:
```bash
cat results/epoch*.yaml | grep "accuracy:"
```

---

## 3. Inference

### Example 3.1: Interactive Testing

Launch interactive mode:

```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --mode interactive
```

**Session**:
```
Question: Janet has 3 apples and buys 2 more. How many does she have?

Answer: Janet has 3 + 2 = 5 apples.
Forward passes: 18

Question: /latent off
✓ Latent reasoning: OFF

Question: What is 12 × 7?

Answer: 84
Forward passes: 5
```

### Example 3.2: Batch Processing from File

Create `questions.txt`:
```
What is 15% of 200?
If a shirt costs $45 and is on 20% sale, what's the price?
John has 5 apples, Mary has 3 times as many. How many total?
```

Run batch inference:
```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --input_file questions.txt \
    --output_file answers.txt
```

**Output** (`answers.txt`):
```
Q: What is 15% of 200?
A: 15% of 200 is 0.15 × 200 = 30
--------------------------------------------------------------------------------
Q: If a shirt costs $45 and is on 20% sale, what's the price?
A: 20% discount is 0.2 × $45 = $9. So the price is $45 - $9 = $36
--------------------------------------------------------------------------------
...
```

### Example 3.3: Compare Latent vs Standard Generation

**With latent**:
```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --questions "What is 25% of 80?" \
    --use_latent \
    --output_file latent_answer.txt
```

**Without latent**:
```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --questions "What is 25% of 80?" \
    --no_latent \
    --output_file standard_answer.txt
```

Compare:
```bash
diff latent_answer.txt standard_answer.txt
```

---

## 4. Multi-Seed Experiments

### Example 4.1: Run Single Config with 3 Seeds

Reproduce results for LLaMA 3.2-1B + GSM8k-AUG:

```bash
python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 42 43 44 \
    --output_dir experiments
```

**What happens**:
1. Trains 3 models (seeds 42, 43, 44)
2. Evaluates each on GSM8k, GSM8k-Hard, SVAMP
3. Aggregates results with mean ± std
4. Saves to `experiments/llama1b_gsm8k-aug/summary.yaml`

**Duration**: ~6-9 hours (LLaMA 1B on A100)

### Example 4.2: Run All Paper Experiments

Execute full replication (4 configs × 3 seeds):

```powershell
.\run_all_experiments.ps1
```

**What happens**:
- 12 training runs
- 36 evaluation runs
- Aggregated statistics for each config

**Duration**: ~24-48 hours (depending on GPU)

### Example 4.3: Custom Seeds

Run with 5 seeds for higher confidence:

```bash
python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 42 43 44 45 46 \
    --output_dir experiments
```

---

## 5. Results Aggregation

### Example 5.1: View Single Experiment Summary

After running multi-seed:

```bash
cat experiments/llama1b_gsm8k-aug/summary.yaml
```

**Output**:
```yaml
gsm8k:
  accuracy_mean: 82.45
  accuracy_std: 0.73
  forward_passes_mean: 48.2
  forward_passes_std: 1.1
  n_seeds: 3
```

### Example 5.2: Create Paper-Ready Table

Aggregate all experiments:

```bash
python aggregate_results.py \
    --experiments_dir experiments \
    --output table1_results.csv
```

**Output** (`table1_results.csv`):

| Model | Training Data | GSM8k Acc (%) | GSM8k FP | GSM8k-Hard Acc (%) | SVAMP Acc (%) |
|-------|---------------|---------------|----------|--------------------|---------------|
| LLaMA 3.2-1B | GSM8k-AUG | 82.45 ± 0.73 | 48.2 | 68.91 ± 1.24 | 75.33 ± 0.89 |
| LLaMA 3.2-1B | GSM8k-AUG-NL | 84.12 ± 0.56 | 46.8 | 70.45 ± 1.01 | 77.89 ± 0.67 |
| Qwen2.5-0.5B | GSM8k-AUG | 76.34 ± 1.12 | 51.3 | 62.78 ± 1.45 | 71.23 ± 1.34 |
| LLaMA 3.2-3B | GSM8k-AUG | 86.78 ± 0.45 | 44.1 | 73.56 ± 0.89 | 80.12 ± 0.78 |

Also creates `table1_results.tex` for LaTeX.

### Example 5.3: Plot Results

Use generated CSV for plotting:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('paper_results.csv')

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.errorbar(
    range(len(df)),
    df['accuracy_mean'],
    yerr=df['accuracy_std'],
    fmt='o-',
    capsize=5
)
plt.xticks(range(len(df)), df['Model'])
plt.ylabel('Accuracy (%)')
plt.title('GSM8k Accuracy by Model')
plt.grid(True)
plt.savefig('accuracy_comparison.png')
```

---

## 6. Advanced Workflows

### Example 6.1: Train + Evaluate Pipeline

Combined script:

```bash
# Train
python train.py --config configs/llama1b_aug.yaml --seed 42

# Evaluate immediately
python evaluate.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --datasets gsm8k gsm8k-hard svamp \
    --output experiments/llama1b_gsm8k-aug/seed_42/eval_results.yaml

# Inference test
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --mode interactive
```

### Example 6.2: Parallel Multi-GPU Training

Train different configs on different GPUs:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 42 43 44

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python run_multi_seed.py \
    --config configs/llama1b_aug_nl.yaml \
    --seeds 42 43 44

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python run_multi_seed.py \
    --config configs/qwen05b_aug.yaml \
    --seeds 42 43 44
```

### Example 6.3: Hyperparameter Sweep

Test different loss weights:

```bash
# Create custom configs
cp configs/llama1b_aug.yaml configs/llama1b_aug_alpha5.yaml
# Edit alpha1: 5, alpha2: 0.5

cp configs/llama1b_aug.yaml configs/llama1b_aug_alpha15.yaml
# Edit alpha1: 15, alpha2: 1.5

# Train both
python train.py --config configs/llama1b_aug_alpha5.yaml --seed 42
python train.py --config configs/llama1b_aug_alpha15.yaml --seed 42

# Compare results
python evaluate.py \
    --checkpoint experiments/llama1b_gsm8k-aug_alpha5/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug_alpha5.yaml \
    --datasets gsm8k \
    --output results/alpha5.yaml

python evaluate.py \
    --checkpoint experiments/llama1b_gsm8k-aug_alpha15/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug_alpha15.yaml \
    --datasets gsm8k \
    --output results/alpha15.yaml
```

---

## 7. Debugging & Testing

### Example 7.1: Quick Sanity Check

Test with minimal data:

```bash
# Modify config to train on 100 examples
python train.py \
    --config configs/llama1b_aug.yaml \
    --seed 42 \
    --max_train_samples 100 \
    --max_eval_samples 20 \
    --num_epochs 1
```

Should complete in ~5-10 minutes.

### Example 7.2: Check Forward Pass Counts

Verify efficiency gains:

```python
from inference import KAVAInference

model = KAVAInference(
    checkpoint_path="experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10",
    config_path="configs/llama1b_aug.yaml"
)

question = "What is 25% of 80?"

# With latent
answer_latent, fp_latent = model.generate(
    question,
    use_latent=True,
    return_forward_count=True
)

# Without latent
answer_standard, fp_standard = model.generate(
    question,
    use_latent=False,
    return_forward_count=True
)

print(f"Latent: {fp_latent} forward passes")
print(f"Standard: {fp_standard} forward passes")
print(f"Reduction: {100*(fp_standard-fp_latent)/fp_standard:.1f}%")
```

### Example 7.3: Verify Loss Components

Check loss values during training:

```bash
# Look at W&B logs or terminal output
python train.py --config configs/llama1b_aug.yaml --seed 42 | tee training.log

# Extract loss values
grep "total_loss" training.log
grep "kv_loss" training.log
grep "codi_loss" training.log
```

---

## 8. Reproducibility Checklist

To reproduce Table 1 from the paper:

- [ ] Run `.\run_all_experiments.ps1`
- [ ] Wait 24-48 hours for completion
- [ ] Run `python aggregate_results.py --experiments_dir experiments`
- [ ] Compare `paper_results.csv` with Table 1
- [ ] Expected: ±1-2% accuracy variance due to hardware/randomness
- [ ] Forward passes should match exactly (deterministic)

---

## 9. Common Issues & Solutions

### Issue: Out of Memory

**Solution**: Reduce batch size or use smaller model:
```yaml
# In config YAML
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### Issue: Training Too Slow

**Solution**: 
1. Use mixed precision (already enabled with `bf16: true`)
2. Reduce `num_epochs` for testing
3. Use smaller model (Qwen 0.5B instead of LLaMA 3B)

### Issue: Low Accuracy

**Possible causes**:
1. Not enough training (increase epochs)
2. Learning rate too high/low (adjust in config)
3. Loss weights misconfigured (check α₁, α₂)
4. Wrong dataset format

**Debugging**:
```bash
# Check training loss curve
# Should decrease smoothly

# Check evaluation on validation set during training
# Should improve over epochs

# Try inference on known examples
python inference.py --mode interactive
```

---

## 10. Best Practices

1. **Always use multiple seeds** (at least 3) for publication results
2. **Log with W&B** for easy tracking (`--wandb` flag)
3. **Save all checkpoints** (storage is cheap, retraining is expensive)
4. **Test inference immediately** after training to verify model works
5. **Compare with paper baselines** using same evaluation protocol
6. **Document hyperparameter changes** if you deviate from Table 6
7. **Use version control** for your configs and results

---

For more details, see:
- [Multi-Seed Guide](MULTI_SEED.md)
- [Inference Guide](INFERENCE.md)
- [Quick Start](QUICKSTART.md)
