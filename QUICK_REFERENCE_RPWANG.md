# KAVA 项目 - HPC 快速参考（rpwang 专用）

## 📡 连接信息

```bash
# SSH 连接
ssh rpwang@10.160.22.46 -p 2223

# SFTP 连接
sftp://rpwang@10.160.22.46:2223/home/rpwang/kava%20review
```

## 📂 项目路径

```bash
# 项目主目录（⚠️ 包含空格）
/home/rpwang/kava review

# 重要：所有命令都要使用引号
cd "/home/rpwang/kava review"
```

## ⚠️ 路径空格处理

你的项目路径包含空格，所有命令必须使用引号：

```bash
# ✅ 正确
cd "/home/rpwang/kava review"
source "/home/rpwang/kava review/venv_kava/bin/activate"
ls -la "/home/rpwang/kava review/outputs"

# ❌ 错误（会导致错误）
cd /home/rpwang/kava review
source /home/rpwang/kava review/venv_kava/bin/activate
```

## 💡 推荐：重命名目录

为避免空格带来的问题，强烈建议重命名：

```bash
# 在 HPC 上执行
cd /home/rpwang
mv "kava review" kava_review
cd kava_review

# 之后就可以不用引号了
cd /home/rpwang/kava_review
```

## 🚀 快速启动（当前路径版本）

### 1. 连接到 HPC

```bash
ssh rpwang@10.160.22.46 -p 2223
```

### 2. 进入项目目录

```bash
cd "/home/rpwang/kava review"
```

### 3. 检查资源和安全性

```bash
# 运行安全检查
bash check_hpc_quota.sh

# 查看安全审查报告
cat HPC_SAFETY_AUDIT.md
```

### 4. 配置环境

```bash
# 使用简化配置脚本
bash simple_setup.sh

# 或使用完整的 venv 配置脚本
bash run_reproduce_venv.sh
```

### 5. 激活虚拟环境

```bash
# 注意路径中的引号
source "/home/rpwang/kava review/venv_kava/bin/activate"
```

### 6. 提交训练任务

```bash
# 提交所有任务
bash submit_all_jobs.sh

# 或单独提交
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm
```

### 7. 监控任务

```bash
# 查看任务状态
squeue -u rpwang

# 使用监控脚本
bash monitor_jobs.sh

# 自动监控模式
bash monitor_jobs.sh --auto
```

## 📊 常用命令

### 磁盘使用检查

```bash
# 检查 HOME 目录空间
df -h /home/rpwang

# 检查项目大小（注意引号）
du -sh "/home/rpwang/kava review"

# 检查 HuggingFace 缓存
du -sh ~/.cache/huggingface

# 检查配额（如果有）
quota -s
```

### SLURM 任务管理

```bash
# 查看你的任务
squeue -u rpwang

# 查看详细信息
squeue -u rpwang --format="%.10i %.15j %.8T %.10M %.6D %.20R"

# 取消任务
scancel <job_id>

# 取消所有任务
scancel -u rpwang

# 查看 GPU 可用性
sinfo -p compute
```

### 日志查看

```bash
# 查看最新的日志
ls -lt logs/ | head -10

# 查看特定任务日志
tail -f logs/kava_<job_id>_<array_id>.out
tail -f logs/kava_<job_id>_<array_id>.err

# 检查训练进度
grep "Epoch" logs/kava_*.out
```

### 环境管理

```bash
# 激活虚拟环境（注意引号）
source "/home/rpwang/kava review/venv_kava/bin/activate"

# 检查 Python 环境
which python
python --version

# 检查安装的包
pip list | grep -E "torch|transformers|peft"

# 安装额外的包
pip install <package_name>
```

### 清理操作

```bash
# 清理旧的 checkpoint（查看）
find "/home/rpwang/kava review/outputs" -name "checkpoint-*" -type d -mtime +30

# 清理 HuggingFace 缓存锁
rm -rf ~/.cache/huggingface/hub/.locks

# 清理 pip 缓存
pip cache purge
```

## 🔧 故障排除

### 问题 1：路径空格导致的错误

```bash
# 症状
bash: cd: /home/rpwang/kava: No such file or directory

# 解决
cd "/home/rpwang/kava review"  # 使用引号

# 或者重命名目录
cd /home/rpwang
mv "kava review" kava_review
```

### 问题 2：虚拟环境未激活

```bash
# 症状
ModuleNotFoundError: No module named 'torch'

# 解决
source "/home/rpwang/kava review/venv_kava/bin/activate"
```

### 问题 3：磁盘空间不足

```bash
# 检查使用情况
df -h /home/rpwang
du -sh "/home/rpwang/kava review"
du -sh ~/.cache/huggingface

# 清理缓存
huggingface-cli delete-cache
rm -rf ~/.cache/pip
```

### 问题 4：任务一直 PENDING

```bash
# 查看原因
squeue -u rpwang --start

# 查看分区状态
sinfo -p compute

# 减少并发任务数
scancel -u rpwang  # 取消部分任务
```

### 问题 5：离线模式模型未找到

```bash
# 检查模型缓存
ls -la /home/share/models

# 确认环境变量
echo $HF_HOME
echo $TRANSFORMERS_CACHE

# 测试模型加载
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct'); print('OK')"
```

## 📁 重要目录结构

```
/home/rpwang/kava review/           # 项目根目录（包含空格）
├── venv_kava/                      # Python 虚拟环境
├── configs/                        # 配置文件
│   ├── llama1b_aug.yaml
│   ├── llama1b_aug_nl.yaml
│   ├── llama3b_aug.yaml
│   └── qwen05b_aug.yaml
├── src/                            # 源代码
│   ├── trainer.py
│   ├── rkv_compression.py
│   └── ...
├── outputs/                        # 训练输出
│   └── <config>_multi_seed/
│       ├── seed_42/
│       ├── seed_123/
│       └── seed_456/
├── logs/                           # SLURM 日志
│   ├── kava_*.out
│   └── kava_*.err
├── train.py                        # 主训练脚本
├── evaluate.py                     # 评估脚本
├── submit_multi_seed.slurm         # SLURM 提交脚本
└── requirements.txt                # Python 依赖

/home/rpwang/.cache/huggingface/    # HuggingFace 缓存（~19GB）
/home/share/models/                 # HPC 共享模型库
```

## 🔐 安全检查清单

运行前确认：

- [ ] 连接到 HPC: `ssh rpwang@10.160.22.46 -p 2223`
- [ ] 进入项目目录: `cd "/home/rpwang/kava review"`
- [ ] 运行安全检查: `bash check_hpc_quota.sh`
- [ ] 确认磁盘空间 ≥ 30GB
- [ ] 虚拟环境已激活
- [ ] 所有命令使用引号处理路径空格
- [ ] 不会同时提交超过 15 个任务

## 📞 获取帮助

```bash
# 查看脚本帮助
bash run_reproduce_venv.sh --help
bash monitor_jobs.sh --help

# 查看安全审查报告
cat HPC_SAFETY_AUDIT.md

# 查看完整文档
ls -la docs/
cat README.md
```

## 🎯 一键启动流程（完整版）

```bash
# 1. 连接 HPC
ssh rpwang@10.160.22.46 -p 2223

# 2. 进入项目（注意引号）
cd "/home/rpwang/kava review"

# 3. 安全检查
bash check_hpc_quota.sh

# 4. 配置环境（如果还没有）
bash simple_setup.sh

# 5. 激活虚拟环境
source "/home/rpwang/kava review/venv_kava/bin/activate"

# 6. 提交训练任务
bash submit_all_jobs.sh

# 7. 监控任务
bash monitor_jobs.sh --auto

# 8. 查看进度
tail -f logs/kava_*.out
```

---

## 💡 最佳实践建议

1. **重命名目录**（强烈推荐）
   ```bash
   cd /home/rpwang
   mv "kava review" kava_review
   ```

2. **设置别名**（编辑 `~/.bashrc`）
   ```bash
   alias kava='cd "/home/rpwang/kava review"'
   alias kava-activate='source "/home/rpwang/kava review/venv_kava/bin/activate"'
   alias kava-jobs='squeue -u rpwang'
   ```

3. **定期清理**
   ```bash
   # 每周检查磁盘
   df -h /home/rpwang
   
   # 清理旧的 checkpoint
   find "/home/rpwang/kava review/outputs" -name "checkpoint-*" -mtime +30
   ```

4. **备份重要结果**
   ```bash
   # 打包结果
   tar -czf results_$(date +%Y%m%d).tar.gz "/home/rpwang/kava review/outputs"
   
   # 下载到本地（在本地执行）
   scp -P 2223 rpwang@10.160.22.46:~/results_*.tar.gz ./
   ```

---

**最后更新**: 2025-11-24  
**HPC 地址**: `10.160.22.46:2223`  
**用户**: `rpwang`  
**项目路径**: `/home/rpwang/kava review` ⚠️ 包含空格
