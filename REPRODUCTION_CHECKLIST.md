# KAVA Paper Reproduction - Quick Start Checklist

## 🎯 Goal
Reproduce KAVA paper results on HPC cluster using **one-click automated workflow**.

---

## ✅ Pre-requisites (5 minutes)

### 1. HPC Access
- [ ] Can login to HPC cluster: `ssh your_username@hpc_login_node`
- [ ] Have SLURM access: `squeue` command works
- [ ] Can see GPU partition: `sinfo -p compute`

### 2. Disk Space
- [ ] Check available space: `df -h $HOME`
- [ ] Need **≥20GB** free space in `$HOME/.cache/huggingface`
- [ ] Can write to home directory: `touch $HOME/test && rm $HOME/test`

### 3. Network Access
- [ ] Can access internet: `ping -c 3 huggingface.co` OR `ping -c 3 hf-mirror.com`
- [ ] (Optional) Proxy configured if behind firewall
- [ ] (Optional) HuggingFace token for gated models: `export HF_TOKEN=your_token`

### 4. Code Repository
- [ ] Code uploaded to HPC: `ls ~/kava/` shows all files
- [ ] Have write permissions: `touch ~/kava/test && rm ~/kava/test`
- [ ] Git works (optional): `git --version`

---

## 🚀 One-Click Execution (1-2 days)

### Step 1: Upload and Verify
```bash
# 上传代码到 HPC
scp -r kava/ user@hpc:/home/user/

# 登录 HPC
ssh user@hpc

# 进入项目目录
cd ~/kava

# 验证部署 (推荐)
bash verify_deployment.sh
```

### Step 2: Quick Setup (Optional but Recommended)
```bash
# 快速配置环境变量和权限
bash setup_hpc.sh
```

### Step 3: Run Automated Script
```bash
# 一键启动所有任务
bash run_reproduce.sh
```

**That's it!** The script will automatically:
1. ✅ Check system requirements (1-2 min)
2. ✅ Create conda environment `kava_env` (5-10 min)
3. ✅ Download models (~19GB, 17-100 min depending on network)
4. ✅ Submit 12 SLURM jobs (4 configs × 3 seeds)
5. ✅ Configure monitoring tools

---

## 📊 Timeline Expectations

| Phase                  | Duration        | Status         |
|------------------------|-----------------|----------------|
| Pre-flight checks      | 1-2 minutes     | Automated      |
| Conda environment      | 5-10 minutes    | Automated      |
| Model download         | 17-100 minutes  | Automated*     |
| Training (parallel)    | 36-48 hours     | SLURM managed  |
| **Total**              | **~1-2 days**   | **Hands-free** |

\* *Download time depends on network:*
- **Direct (HuggingFace):** 50-100 min
- **With proxy:** 17-35 min  
- **Mirror (hf-mirror.com):** 33-68 min

---

## 🔧 Advanced Options

### Skip Model Download (if already cached)
```bash
bash run_reproduce.sh --skip-download
```

### Choose Download Method
```bash
# Use mirror (faster from China mainland)
bash run_reproduce.sh --method mirror

# Use proxy (if configured)
bash run_reproduce.sh --method proxy

# Force direct download
bash run_reproduce.sh --method direct
```

### Skip Environment Setup (if already exists)
```bash
bash run_reproduce.sh --skip-env
```

---

## 📈 Monitoring Progress

### Quick Status Check
```bash
bash monitor_jobs.sh
```

### Manual Commands
```bash
# Check SLURM queue
squeue -u $USER

# View specific log
tail -f outputs/logs/llama1b_aug_seed42.log

# Check all recent logs
ls -lt outputs/logs/ | head

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

---

## 📦 Collecting Results

### After All Jobs Complete (36-48 hours)
```bash
# Check if jobs are done
squeue -u $USER  # Should show empty

# Aggregate results
bash collect_results.sh
```

### Result Files
```
outputs/
├── results/
│   ├── llama1b_aug_seed42.json
│   ├── llama1b_aug_seed123.json
│   ├── llama1b_aug_seed456.json
│   ├── llama1b_aug_nl_seed42.json
│   ├── ... (12 files total)
│   └── aggregated_results.csv  ← Final summary
└── logs/
    ├── llama1b_aug_seed42.log
    └── ... (12 log files)
```

---

## 🎯 Paper Reproduction Verification

### Expected Results (from paper)

| Model Configuration      | GSM8K | GSM8K-Hard | SVAMP |
|-------------------------|-------|------------|-------|
| Llama-3.2-1B (aug)      | ~XX%  | ~XX%       | ~XX%  |
| Llama-3.2-1B (aug-nl)   | ~XX%  | ~XX%       | ~XX%  |
| Qwen2.5-0.5B (aug)      | ~XX%  | ~XX%       | ~XX%  |
| Llama-3.2-3B (aug)      | ~XX%  | ~XX%       | ~XX%  |

*Note: Fill in exact numbers from paper Table X.*

### Compare Your Results
```bash
# View aggregated results
cat outputs/aggregated_results.csv | column -t -s,
```

Results should be within **±1-2%** of paper values (accounting for random seed variance).

---

## ❓ Troubleshooting

### Models Not Downloading
**Symptoms:** `ConnectionError` or `TimeoutError` during download

**Solutions:**
1. Check network: `ping huggingface.co`
2. Try mirror: `bash run_reproduce.sh --method mirror`
3. Configure proxy:
   ```bash
   export HTTP_PROXY=http://proxy.server:port
   export HTTPS_PROXY=http://proxy.server:port
   bash run_reproduce.sh --method proxy
   ```
4. Manual download: See `docs/KAVA_MODEL_DOWNLOAD.md`

### Insufficient Disk Space
**Symptoms:** `OSError: [Errno 28] No space left on device`

**Solutions:**
1. Check quota: `df -h $HOME`
2. Clean cache:
   ```bash
   rm -rf $HOME/.cache/huggingface/hub/.locks
   huggingface-cli delete-cache
   ```
3. Request quota increase from HPC admin

### SLURM Jobs Pending Forever
**Symptoms:** `squeue` shows all jobs as `PENDING` for hours

**Solutions:**
1. Check reason: `squeue -u $USER --start`
2. Check partition: `sinfo -p compute`
3. Reduce concurrent jobs: Edit `hpc_run_all.sh` to submit fewer jobs
4. Contact HPC support if queue is stuck

### Environment Activation Fails
**Symptoms:** `conda: command not found` or `Environment not found`

**Solutions:**
1. Load module: `module load anaconda3` or `module load miniconda3`
2. Initialize conda: `conda init bash && source ~/.bashrc`
3. Re-run script: `bash run_reproduce.sh`

### Job Crashes (OOM or CUDA errors)
**Symptoms:** Log shows `CUDA out of memory` or `Killed`

**Solutions:**
1. Check GPU allocation: `squeue -u $USER -o "%.18i %.9P %.8j %.8T %.10M %.6D %R"`
2. Verify GPU type: Should be `a100-sxm4-80gb`
3. Reduce batch size: Edit `configs/*.yaml` → `training.batch_size: 2`
4. Contact support if GPU hardware issue

---

## 📚 Additional Documentation

- **Complete HPC Guide:** `docs/GETTING_STARTED_HPC.md`
- **Model Download Details:** `docs/KAVA_MODEL_DOWNLOAD.md`
- **SLURM Reference:** `docs/HPC_REFERENCE.md`
- **Debugging Guide:** `docs/SLURM_INTERACTIVE_GUIDE.md`
- **Main README:** `README.md`

---

## 🆘 Need Help?

### Quick Commands Cheat Sheet
```bash
# Check everything
squeue -u $USER           # Queue status
bash monitor_jobs.sh      # Progress summary
tail -f outputs/logs/*.log  # Watch all logs

# Emergency stop
scancel -u $USER          # Cancel all jobs

# Restart
bash run_reproduce.sh     # Re-runs safely (skips completed steps)
```

### Contact
- HPC Support: your_hpc_support_email
- KAVA Issues: [GitHub repository]

---

## ✨ Success Criteria

### You're Done When:
- [x] All 12 jobs completed successfully
- [x] `outputs/results/` contains 12 JSON files
- [x] `outputs/aggregated_results.csv` generated
- [x] Results match paper (within ±2%)

### Congratulations! 🎉
You've successfully reproduced the KAVA paper on your HPC cluster!

---

## 📝 Notes for Paper Reproduction

### Random Seeds
- Seed 42, 123, 456 used per paper specification
- Results averaged across 3 seeds
- Standard deviation reported in aggregated results

### Hyperparameters
All hyperparameters match paper:
- LoRA rank: 8, alpha: 16
- Learning rate: 5e-5
- Batch size: 4
- Gradient accumulation: 4
- Latent tokens: 24
- See `configs/*.yaml` for full details

### Datasets
- **GSM8K:** Grade school math problems (test set)
- **GSM8K-Hard:** Harder variants
- **SVAMP:** Math word problems
- All downloaded automatically from HuggingFace

### Models
- **Llama-3.2-1B-Instruct:** Small efficient model
- **Llama-3.2-3B-Instruct:** Larger variant
- **Qwen2.5-0.5B-Instruct:** Multilingual baseline
- All with LoRA + KAVA compression

---

**Last Updated:** 2024
**Version:** 1.0  
**Compatible with:** SLURM-based HPC clusters, CUDA 11.8+
