# Multi-Seed Experiments Guide

Complete guide for running statistically significant experiments with KAVA.

## Overview

The paper reports all results with **3 random seeds** and presents **mean ± std** to ensure statistical significance. This guide covers:

1. Single-seed experiments
2. Multi-seed automation
3. Results aggregation
4. Reproducing Table 1

---

## Quick Start

### Run All Paper Experiments

Execute all 4 configurations from Table 6 with 3 seeds each:

```powershell
.\run_all_experiments.ps1
```

This will:
- Train LLaMA 3.2-1B on GSM8k-AUG (3 seeds)
- Train LLaMA 3.2-1B on GSM8k-AUG-NL (3 seeds)
- Train Qwen2.5-0.5B on GSM8k-AUG (3 seeds)
- Train LLaMA 3.2-3B on GSM8k-AUG (3 seeds)

Total: **12 training runs** + **36 evaluations** (GSM8k, GSM8k-Hard, SVAMP)

**Expected runtime**: ~24-48 hours (depending on GPU)

---

## Step-by-Step Usage

### 1. Single Configuration, Multiple Seeds

Run one model configuration with 3 seeds:

```bash
python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 42 43 44 \
    --output_dir experiments
```

**What this does**:
- Trains model 3 times (one per seed)
- Evaluates each checkpoint on GSM8k, GSM8k-Hard, SVAMP
- Saves individual results to `experiments/llama1b_gsm8k-aug/seed_XX/`
- Aggregates statistics to `experiments/llama1b_gsm8k-aug/summary.yaml`

### 2. Custom Seeds

Use different random seeds:

```bash
python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 123 456 789 \
    --output_dir experiments
```

### 3. More Seeds for Robustness

Run 5 seeds instead of 3:

```bash
python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 42 43 44 45 46 \
    --output_dir experiments
```

---

## Output Structure

```
experiments/
├── llama1b_gsm8k-aug/
│   ├── seed_42/
│   │   ├── checkpoint-epoch1/
│   │   ├── results_gsm8k.yaml
│   │   ├── results_gsm8k-hard.yaml
│   │   └── results_svamp.yaml
│   ├── seed_43/
│   │   └── ...
│   ├── seed_44/
│   │   └── ...
│   ├── all_seeds_results.yaml  # Raw results from all seeds
│   ├── summary.yaml             # Aggregated mean ± std
│   └── summary.csv              # For plotting
├── llama1b_gsm8k-aug-nl/
│   └── ...
├── qwen05b_gsm8k-aug/
│   └── ...
└── llama3b_gsm8k-aug/
    └── ...
```

---

## Results Aggregation

### View Summary for One Experiment

After running `run_multi_seed.py`, check the summary:

```bash
cat experiments/llama1b_gsm8k-aug/summary.yaml
```

**Example output**:
```yaml
gsm8k:
  accuracy_mean: 82.45
  accuracy_std: 0.73
  forward_passes_mean: 48.2
  forward_passes_std: 1.1
  n_seeds: 3

gsm8k-hard:
  accuracy_mean: 68.91
  accuracy_std: 1.24
  forward_passes_mean: 52.7
  forward_passes_std: 1.9
  n_seeds: 3

svamp:
  accuracy_mean: 75.33
  accuracy_std: 0.89
  forward_passes_mean: 45.1
  forward_passes_std: 1.3
  n_seeds: 3
```

### Aggregate All Experiments

Combine results from all configurations into one table:

```bash
python aggregate_results.py \
    --experiments_dir experiments \
    --output paper_results.csv
```

**Output files**:
- `paper_results.csv`: CSV table with all results
- `paper_results.tex`: LaTeX table for paper

**Example table**:

| Model | Training Data | GSM8k Acc (%) | GSM8k FP | GSM8k-Hard Acc (%) | SVAMP Acc (%) |
|-------|---------------|---------------|----------|-------------------|---------------|
| LLaMA 3.2-1B | GSM8k-AUG | 82.45 ± 0.73 | 48.2 | 68.91 ± 1.24 | 75.33 ± 0.89 |
| LLaMA 3.2-1B | GSM8k-AUG-NL | 84.12 ± 0.56 | 46.8 | 70.45 ± 1.01 | 77.89 ± 0.67 |
| Qwen2.5-0.5B | GSM8k-AUG | 76.34 ± 1.12 | 51.3 | 62.78 ± 1.45 | 71.23 ± 1.34 |
| LLaMA 3.2-3B | GSM8k-AUG | 86.78 ± 0.45 | 44.1 | 73.56 ± 0.89 | 80.12 ± 0.78 |

---

## Reproducing Table 1

Follow these steps to reproduce **Table 1** from the paper:

### Step 1: Run All Experiments

```powershell
.\run_all_experiments.ps1
```

**Time**: 24-48 hours

### Step 2: Aggregate Results

```bash
python aggregate_results.py --experiments_dir experiments --output table1_results.csv
```

### Step 3: Compare with Paper

Open `table1_results.csv` and compare with Table 1 in the paper.

**Expected differences**:
- Small variations (±1-2%) due to hardware/randomness
- Forward pass counts should match exactly (deterministic)

---

## Advanced Usage

### Resume Failed Runs

If a seed fails, re-run just that seed:

```bash
python train.py \
    --config configs/llama1b_aug.yaml \
    --seed 43 \
    --output_dir experiments/llama1b_gsm8k-aug/seed_43
```

Then re-evaluate:

```bash
python evaluate.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_43/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --datasets gsm8k gsm8k-hard svamp \
    --output experiments/llama1b_gsm8k-aug/seed_43/results_all.yaml
```

### Run Only Training (Skip Evaluation)

Modify `run_multi_seed.py` to skip evaluation step (comment out lines 75-115).

### Run Only Evaluation

If checkpoints already exist:

```bash
python evaluate.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --datasets gsm8k gsm8k-hard svamp \
    --output experiments/llama1b_gsm8k-aug/seed_42/results.yaml
```

---

## Monitoring Progress

### Weights & Biases

If `--wandb` flag is used, track progress at: https://wandb.ai

Each seed creates a separate run with naming: `{model}_{dataset}_seed{seed}`

### Check Individual Results

While experiments are running:

```bash
# Check training progress
ls experiments/llama1b_gsm8k-aug/seed_42/

# Check if evaluation completed
cat experiments/llama1b_gsm8k-aug/seed_42/results_gsm8k.yaml
```

### Terminal Logs

Redirect output to log files:

```bash
python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 42 43 44 \
    --output_dir experiments \
    > logs/llama1b_aug.log 2>&1
```

---

## Compute Requirements

### Per Seed

| Model | Training Time | GPU Memory | Evaluation Time |
|-------|---------------|------------|-----------------|
| LLaMA 3.2-1B | ~2-3 hours | ~20GB | ~30 min |
| Qwen2.5-0.5B | ~1-2 hours | ~16GB | ~20 min |
| LLaMA 3.2-3B | ~4-6 hours | ~30GB | ~45 min |

### Full Replication

- **Total time**: 24-48 hours (4 configs × 3 seeds)
- **GPU**: 1x A100 (40GB) or 1x A6000 (48GB)
- **Storage**: ~50GB for all checkpoints

**Tip**: Run on multiple GPUs in parallel:
```bash
# GPU 0: LLaMA 1B configs
CUDA_VISIBLE_DEVICES=0 python run_multi_seed.py --config configs/llama1b_aug.yaml &

# GPU 1: Qwen config
CUDA_VISIBLE_DEVICES=1 python run_multi_seed.py --config configs/qwen05b_aug.yaml &

# GPU 2: LLaMA 3B config
CUDA_VISIBLE_DEVICES=2 python run_multi_seed.py --config configs/llama3b_aug.yaml &
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in config:

```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase from 4
```

### Issue: Dataset Loading Fails

**Solution**: Check dataset availability:

```bash
python -c "from datasets import load_dataset; load_dataset('whynlp/gsm8k-aug')"
```

If unavailable, modify `src/evaluation_datasets.py` to use local files.

### Issue: Checkpoint Not Found

**Solution**: Check if training completed:

```bash
ls experiments/llama1b_gsm8k-aug/seed_42/
```

If no checkpoint, re-run training for that seed.

---

## Statistical Significance

### Why 3 Seeds?

The paper uses **3 seeds** to:
1. Account for training randomness
2. Report confidence intervals (mean ± std)
3. Balance compute cost vs. statistical robustness

### Interpreting Results

- **std < 1%**: Highly stable across seeds
- **std 1-2%**: Normal variation
- **std > 2%**: High variance (consider more seeds)

### Comparing Methods

Use **2-std overlap rule**:
- If `mean₁ - 2×std₁ > mean₂ + 2×std₂`, then method 1 is significantly better

**Example**:
```
Method A: 82.45 ± 0.73  → Range: [81.0, 83.9]
Method B: 78.21 ± 1.12  → Range: [76.0, 80.3]

No overlap → A is significantly better
```

---

## Citation

If you use these scripts, please cite the KAVA paper:

```bibtex
@article{kava2024,
  title={Latent Reasoning via Compressed KV-Cache Distillation},
  author={...},
  journal={arXiv preprint arXiv:2510.02312},
  year={2024}
}
```

---

## Next Steps

- ✅ Run all experiments: `.\run_all_experiments.ps1`
- ✅ Aggregate results: `python aggregate_results.py`
- ✅ Compare with Table 1
- 🔄 Run ablations (see `EXPERIMENTS.md`)
- 🔄 Try your own datasets
