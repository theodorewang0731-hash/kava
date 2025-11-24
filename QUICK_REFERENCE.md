# KAVA Quick Reference Card

**一页速查：最常用的命令和配置**

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train
python train.py --config configs/llama1b_aug.yaml --seed 42

# 3. Test
python inference.py --checkpoint <path> --config configs/llama1b_aug.yaml --mode interactive
```

---

## 📁 File Quick Access

| Need | File |
|------|------|
| Training | `train.py` |
| Evaluation | `evaluate.py` |
| Testing | `inference.py` |
| Multi-seed | `run_multi_seed.py` |
| Full replication | `run_all_experiments.ps1` |
| Config (1B) | `configs/llama1b_aug.yaml` |
| Config (0.5B) | `configs/qwen05b_aug.yaml` |
| Config (3B) | `configs/llama3b_aug.yaml` |

---

## 🎯 Common Tasks

### Train Model
```bash
python train.py \
    --config configs/llama1b_aug.yaml \
    --seed 42 \
    --wandb
```

### Evaluate Model
```bash
python evaluate.py \
    --checkpoint <checkpoint_path> \
    --config configs/llama1b_aug.yaml \
    --datasets gsm8k gsm8k-hard svamp \
    --output results.yaml
```

### Interactive Testing
```bash
python inference.py \
    --checkpoint <checkpoint_path> \
    --config configs/llama1b_aug.yaml \
    --mode interactive
```

### Batch Inference
```bash
python inference.py \
    --checkpoint <checkpoint_path> \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --input_file questions.txt \
    --output_file answers.txt
```

### Multi-Seed Experiments
```bash
python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 42 43 44 \
    --output_dir experiments
```

### Aggregate Results
```bash
python aggregate_results.py \
    --experiments_dir experiments \
    --output paper_results.csv
```

### Full Replication
```powershell
.\run_all_experiments.ps1
```

---

## ⚙️ Key Hyperparameters (Table 6)

| Config | Model | α₁ | α₂ | LR | Epochs |
|--------|-------|----|----|-----|--------|
| llama1b_aug | LLaMA-1B | 10 | 1 | 8e-4 | 10 |
| llama1b_aug_nl | LLaMA-1B | 10 | 1 | 8e-4 | 10 |
| qwen05b_aug | Qwen-0.5B | 10 | 1 | 5e-4 | 10 |
| llama3b_aug | LLaMA-3B | 20 | 2 | 2e-4 | 5 |

**LoRA (All)**: r=128, α=32, dropout=0.1  
**Latent**: M=24 tokens, T=3 Jacobi iterations

---

## 📊 Expected Performance

| Model | GSM8k Acc | Forward Passes | Training Time |
|-------|-----------|----------------|---------------|
| LLaMA 3.2-1B | ~82-84% | ~48 | 2-3 hrs |
| Qwen2.5-0.5B | ~76-78% | ~51 | 1-2 hrs |
| LLaMA 3.2-3B | ~86-88% | ~44 | 4-6 hrs |

*(on A100 40GB)*

---

## 🔧 Common Issues

| Issue | Solution |
|-------|----------|
| OOM | Reduce `per_device_train_batch_size` in config |
| Dataset error | Check dataset availability on HuggingFace |
| Slow training | Use bf16, reduce logging frequency |
| Low accuracy | Increase epochs, check learning rate |
| Checkpoint not found | Verify training completed successfully |

---

## 📚 Documentation Quick Links

| Topic | Document |
|-------|----------|
| Getting started | `docs/QUICKSTART.md` |
| Multi-seed experiments | `docs/MULTI_SEED.md` |
| Inference usage | `docs/INFERENCE.md` |
| Code examples | `docs/EXAMPLES.md` |
| Paper mapping | `docs/PAPER_MAPPING.md` |
| Implementation status | `STATUS.md` |

---

## 💡 Best Practices

✅ **Always use multiple seeds** (at least 3)  
✅ **Enable W&B logging** (`--wandb`)  
✅ **Test inference immediately** after training  
✅ **Save all checkpoints** (storage is cheap)  
✅ **Document your changes** if modifying configs  

---

## 🎯 Reproducibility Checklist

- [ ] Install dependencies
- [ ] Run `python train.py --config <config> --seed 42`
- [ ] Evaluate with `python evaluate.py --checkpoint <path> --config <config>`
- [ ] Test with `python inference.py --mode interactive`
- [ ] Run multi-seed: `python run_multi_seed.py --config <config>`
- [ ] Aggregate: `python aggregate_results.py`
- [ ] Compare with Table 1 in paper

---

## 📞 Quick Help

| Question | Answer |
|----------|--------|
| First time setup? | See `docs/QUICKSTART.md` |
| How to reproduce paper? | Run `.\run_all_experiments.ps1` |
| How to test model? | `python inference.py --mode interactive` |
| Where are results? | `experiments/` directory |
| Need examples? | See `docs/EXAMPLES.md` |

---

## 🔗 Links

- **Paper**: [arXiv:2510.02312](https://arxiv.org/abs/2510.02312)
- **Code**: [GitHub repo]
- **Docs**: `docs/` folder
- **Status**: `STATUS.md`

---

**Version**: 1.0  
**Status**: ✅ Production Ready  
**Last Updated**: 2025-01

---

*Keep this card handy for quick reference!*
