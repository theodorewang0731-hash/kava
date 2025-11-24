# HPC 快速启动指南（rpwang 专用）

## 🚀 正确的启动流程

### ⚠️ 常见错误
**不要运行**: `python download_from_hf.py`  
因为 HPC 已有共享模型库，无需下载！

---

## ✅ 正确步骤

### 1. 连接到 HPC
```bash
ssh rpwang@10.160.22.46 -p 2223
```

### 2. 进入项目目录
```bash
cd "/home/rpwang/kava review"
```

### 3. 检查共享模型（推荐，首次运行）
```bash
bash check_hpc_models_availability.sh
```

预期输出：
```
✓ HPC 共享模型库存在: /home/share/models
✓ 有读取权限
✓ meta-llama/Llama-3.2-1B-Instruct
✓ meta-llama/Llama-3.2-3B-Instruct
✓ Qwen/Qwen2.5-0.5B-Instruct
✓ 模型加载测试成功
```

### 4. 配置环境（首次运行）
```bash
bash simple_setup.sh
```

这会：
- ✅ 创建 Python 虚拟环境
- ✅ 安装依赖
- ✅ 自动配置使用共享模型库
- ✅ 设置离线模式

### 5. 下载数据集到本地（首次运行，推荐）
```bash
# 激活虚拟环境
source "/home/rpwang/kava review/venv_kava/bin/activate"

# 下载数据集
python datasets/download_datasets.py

# 或使用镜像加速（国内推荐）
python datasets/download_datasets.py --mirror
```

预计下载时间：10-30分钟  
占用空间：约 4-6 GB

**注意**：如果跳过此步骤，训练时会自动从 HuggingFace 加载，但速度较慢。

### 6. 激活虚拟环境
```bash
source "/home/rpwang/kava review/venv_kava/bin/activate"
```

### 7. 提交训练任务
```bash
# 提交单个配置
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm

# 或提交所有配置（12个任务）
bash submit_all_jobs.sh
```

### 8. 监控任务
```bash
# 查看任务状态
squeue -u rpwang

# 自动监控
bash monitor_jobs.sh --auto

# 查看日志
tail -f logs/kava_*.out
```

---

## 📋 环境变量（已自动配置）

SLURM 脚本中已配置：
```bash
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HUGGINGFACE_HUB_OFFLINE=1
```

**含义**：
- 使用 HPC 共享模型库 `/home/share/models`
- 启用离线模式，不尝试联网下载
- 节省时间和磁盘空间

---

## ❌ 常见错误及解决

### 错误 1: 运行了 `python download_from_hf.py`

**症状**：
```
Downloading meta-llama/Llama-3.2-1B-Instruct...
✗ Failed to download: 403 Client Error
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted
```

**原因**：
- 不需要下载！HPC 已有共享模型
- Llama 模型需要授权访问

**解决**：
```bash
# 不要运行 download_from_hf.py
# 直接使用共享模型：
bash check_hpc_models_availability.sh  # 验证模型可用
bash simple_setup.sh                   # 配置环境
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm  # 提交任务
```

### 错误 2: 路径空格问题

**症状**：
```bash
bash: cd: /home/rpwang/kava: No such file or directory
```

**解决**：使用引号
```bash
cd "/home/rpwang/kava review"
source "/home/rpwang/kava review/venv_kava/bin/activate"
```

或重命名目录：
```bash
cd /home/rpwang
mv "kava review" kava_review
cd kava_review
```

### 错误 3: 模块未找到

**症状**：
```
ModuleNotFoundError: No module named 'torch'
```

**解决**：激活虚拟环境
```bash
source "/home/rpwang/kava review/venv_kava/bin/activate"
python --version  # 验证
```

---

## 🎯 完整工作流程示例

```bash
# 1. 连接 HPC
ssh rpwang@10.160.22.46 -p 2223

# 2. 进入项目
cd "/home/rpwang/kava review"

# 3. 首次配置（只需运行一次）
bash check_hpc_models_availability.sh
bash simple_setup.sh

# 4. 下载数据集（首次运行，推荐）
source "/home/rpwang/kava review/venv_kava/bin/activate"
python datasets/download_datasets.py

# 5. 每次登录后激活环境
source "/home/rpwang/kava review/venv_kava/bin/activate"

# 6. 提交训练任务
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm

# 7. 监控
squeue -u rpwang
```

---

## 📊 任务配置说明

项目包含 4 个配置，每个配置运行 3 个随机种子：

| 配置 | 模型 | 数据集 | 任务数 |
|------|------|--------|--------|
| `llama1b_aug` | Llama-3.2-1B | gsm8k-aug | 3 |
| `llama1b_aug_nl` | Llama-3.2-1B | gsm8k-aug-nl | 3 |
| `llama3b_aug` | Llama-3.2-3B | gsm8k-aug | 3 |
| `qwen05b_aug` | Qwen2.5-0.5B | gsm8k-aug | 3 |

总计：**12 个训练任务**

单个任务资源需求：
- 1 个 A100 GPU (80GB)
- 32GB 内存
- 4 CPU 核心
- 最长 48 小时

---

## 🔍 验证检查清单

运行前确认：

- [ ] 在正确的目录：`pwd` 显示 `/home/rpwang/kava review`
- [ ] 共享模型可用：`ls -la /home/share/models` 有内容
- [ ] 虚拟环境已激活：`which python` 显示 venv 路径
- [ ] SLURM 可用：`squeue` 命令能运行
- [ ] 磁盘空间充足：`df -h /home/rpwang` > 10GB 可用

---

## 💡 有用的命令

```bash
# 查看任务详情
squeue -u rpwang --format="%.10i %.15j %.8T %.10M %.6D %.20R"

# 取消任务
scancel <job_id>

# 取消所有任务
scancel -u rpwang

# 查看日志
ls -lt logs/ | head
tail -f logs/kava_<job_id>_<array_id>.out

# 检查 GPU 分区
sinfo -p compute

# 检查磁盘使用
df -h /home/rpwang
du -sh "/home/rpwang/kava review"
```

---

## 📞 问题排查

如果遇到问题：

1. **首先运行诊断**：
   ```bash
   bash check_hpc_quota.sh
   bash check_hpc_models_availability.sh
   ```

2. **查看日志文件**：
   ```bash
   ls -lt logs/
   tail -100 logs/kava_*.err
   ```

3. **验证环境**：
   ```bash
   source "/home/rpwang/kava review/venv_kava/bin/activate"
   python -c "import torch; print(torch.__version__)"
   python -c "from transformers import AutoTokenizer; print('OK')"
   ```

4. **检查模型加载**：
   ```bash
   export HF_HOME=/home/share/models
   export HUGGINGFACE_HUB_OFFLINE=1
   python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', local_files_only=True); print('✓ 模型加载成功')"
   ```

---

**关键提示**：
- ✅ **使用共享模型**，不要下载
- ✅ **使用引号**处理路径空格
- ✅ **激活虚拟环境**再运行任务
- ✅ **提交 SLURM 任务**，不要在登录节点训练

**快速参考**：`QUICK_REFERENCE_RPWANG.md`
