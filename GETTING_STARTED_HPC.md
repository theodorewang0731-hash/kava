# HPC 新手上手指南

**从上传项目到生成论文结果的完整流程**

---

## 📋 概述

本指南将帮助你在 **30 分钟内**完成从零开始到提交训练任务，**48 小时后**获得论文 Table 1 和 Table 2 的完整结果。

**目标**：严格复现 KAVA 论文结果（GSM8k、GSM8k-Hard、SVAMP 三个数据集的准确率）

---

## 🎯 快速导航

| 阶段 | 预计时间 | 关键步骤 |
|------|---------|---------|
| **阶段 1: 上传项目** | 5 分钟 | 使用 scp/Git 上传代码 |
| **阶段 2: 环境配置** | 15 分钟 | 运行自动配置脚本 |
| **阶段 3: 提交训练** | 5 分钟 | 一键提交所有实验 |
| **阶段 4: 监控进度** | 48 小时 | 定期检查日志 |
| **阶段 5: 生成结果** | 5 分钟 | 格式化为 LaTeX 表格 |

**总计**：30 分钟配置 + 48 小时自动运行

---

## 🚀 阶段 1: 上传项目到 HPC（5 分钟）

### 方法 A: 使用 scp 上传（推荐）

```bash
# 在本地终端运行（假设项目在 D:\kava）
# Windows PowerShell
scp -r "D:\kava" username@hpc.example.edu:~/

# Linux/macOS
scp -r /path/to/kava username@hpc.example.edu:~/

# 验证上传
ssh username@hpc.example.edu
ls -lh ~/kava
```

### 方法 B: 使用 Git 克隆

```bash
# 登录到 HPC
ssh username@hpc.example.edu

# 克隆项目
cd ~
git clone https://github.com/yourusername/kava.git
cd kava

# 如果 GitHub 访问慢，使用代理（参见后续章节）
```

### 方法 C: VSCode Remote SSH（最方便）

1. 安装 VSCode 的 **Remote - SSH** 扩展
2. 按 `F1` → 输入 "Remote-SSH: Connect to Host"
3. 输入 `username@hpc.example.edu`
4. 打开远程目录 `/home/username/kava`
5. 在 VSCode 中直接编辑和同步文件

**结果检查**：
```bash
# 登录 HPC，确认项目结构
ssh username@hpc.example.edu
cd ~/kava
ls -lh

# 应该看到：
# configs/          - 配置文件
# submit_multi_seed.slurm  - SLURM 脚本
# train.py          - 训练脚本
# setup_hpc_models.sh  - 自动配置脚本
# GETTING_STARTED_HPC.md  - 本指南
```

---

## ⚙️ 阶段 2: 环境配置（15 分钟）

### Step 1: 一键自动配置（推荐）

⚠️ **注意**：由于 HPC 公共模型库没有 KAVA 所需模型，跳过 `setup_hpc_models.sh`，直接配置个人环境。

```bash
# 登录到 HPC
ssh username@hpc.example.edu
cd ~/kava

# 配置个人 HuggingFace 缓存目录
cat >> ~/.bashrc << 'EOF'
# HuggingFace 个人缓存（KAVA 项目）
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface
EOF

# 重新加载配置
source ~/.bashrc

# 验证环境变量
echo $HF_HOME
# 应该输出：/home/username/.cache/huggingface
```

### Step 2: 创建 Python 环境

```bash
# 加载 Anaconda
module load anaconda3  # 或 miniconda3

# 创建虚拟环境
conda create -n kava python=3.10 -y
conda activate kava

# 安装 PyTorch（CUDA 11.8）
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt

# 安装额外依赖
pip install peft wandb bitsandbytes

# 验证安装
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import transformers, peft; print('✓ All dependencies installed')"
```

### Step 3: 下载项目所需模型

⚠️ **重要**：HPC 公共模型库（`/home/share/models`）中**没有 KAVA 项目所需的模型**。

**项目需要**：
- `meta-llama/Llama-3.2-1B-Instruct` ❌ 不在公共库
- `meta-llama/Llama-3.2-3B-Instruct` ❌ 不在公共库
- `Qwen/Qwen2.5-0.5B-Instruct` ❌ 不在公共库

**公共库有的**：Llama-2 系列、Llama-30b/65b、Qwen1.5 等（可用 `ls /home/share/models` 查看）

#### 方案 A: 下载到个人目录（推荐）

```bash
# 配置个人缓存目录
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface

# 写入 ~/.bashrc
cat >> ~/.bashrc << 'EOF'
# HuggingFace 个人缓存
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface
EOF

source ~/.bashrc

# 下载模型（需要 10-30 分钟，取决于网络）
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# 或在训练时自动下载（首次运行会下载）
python train.py --config configs/llama1b_aug.yaml
```

#### 方案 B: 使用代理加速下载

如果 HuggingFace 访问较慢，使用本地代理：

```bash
# 在本地机器启动代理（Clash/Shadowrocket）
# 然后在本地终端建立反向隧道
ssh -N -R 55555:localhost:7890 username@hpc.example.edu &

# 在 HPC 终端配置代理
export http_proxy=http://localhost:55555
export https_proxy=http://localhost:55555
export all_proxy=http://localhost:55555

# 测试连接
curl -I https://huggingface.co

# 下载模型（通过代理加速）
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

#### 方案 C: 使用 HuggingFace 镜像

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
```

#### 验证模型下载

```bash
# 检查模型是否下载成功
ls -lh ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
ls -lh ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct
ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct

# 测试加载
python << EOF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
print("✓ Successfully loaded from personal cache")
EOF
```

**预计下载时间**：
- LLaMA 3.2-1B: ~5 GB → 约 10-15 分钟
- LLaMA 3.2-3B: ~12 GB → 约 20-30 分钟
- Qwen 2.5-0.5B: ~2 GB → 约 5-10 分钟
- 总计：~19 GB → 约 35-55 分钟（首次）

**💡 提示**：
- 模型下载后会永久保存在 `~/.cache/huggingface/`
- 后续训练无需重复下载
- 如果集群有多个用户需要，可以请求管理员添加到公共库

### Step 4: 配置 WandB（可选，但推荐）

```bash
# 安装并登录 WandB（用于远程监控训练）
wandb login

# 输入你的 API key（从 https://wandb.ai/settings 获取）
# 粘贴后按 Enter

# 验证
wandb status
```

**如果遇到问题**：
- 参考 [`HPC_REFERENCE.md`](HPC_REFERENCE.md) 的"环境设置"章节
- 参考 [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md) 的"环境准备"章节
- 参考 [`CONDA_CUDA_GUIDE.md`](CONDA_CUDA_GUIDE.md) 的详细 CUDA 配置

---

## 🎬 阶段 3: 提交训练任务（5 分钟）

### 快速测试（可选，2 分钟）

⚠️ **先确认模型已下载**，否则测试会尝试自动下载。

```bash
# 检查模型是否存在
ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct

# 如果模型不存在，先下载（参见阶段 2 的 Step 3）
# 如果已下载，运行快速测试
python smoke_test.py

# 应该输出：
# ✓ PyTorch loaded
# ✓ Transformers loaded
# ✓ CUDA available
# ✓ All checks passed
```

### 方案 A: 一键提交所有实验（推荐）

```bash
# 使用自动化脚本提交 4 个配置 × 3 个种子 = 12 个任务
chmod +x hpc_run_all.sh
./hpc_run_all.sh

# 脚本会自动：
# ✅ 提交 llama1b_aug（LLaMA 1B + GSM8k-AUG）
# ✅ 提交 llama1b_aug_nl（LLaMA 1B + GSM8k-AUG-NL）
# ✅ 提交 qwen05b_aug（Qwen 0.5B + GSM8k-AUG）
# ✅ 提交 llama3b_aug（LLaMA 3B + GSM8k-AUG）
# ✅ 每个配置 3 个种子（42, 123, 456）
# ✅ 自动聚合结果

# 预计完成时间：约 36-48 小时（并行运行）
```

### 方案 B: 单个配置提交（快速验证）

```bash
# 仅提交 LLaMA 1B 配置（用于快速验证）
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm

# 查看提交的任务
squeue --me

# 应该看到 3 个任务（3 个种子）：
# JOBID  PARTITION  NAME           USER  ST  TIME  NODES
# 12345  compute    kava-multi-se  user  PD  0:00  1
# 12346  compute    kava-multi-se  user  PD  0:00  1
# 12347  compute    kava-multi-se  user  PD  0:00  1
```

### 方案 C: 交互式测试（调试用）

```bash
# 申请单卡 GPU 进行交互式测试
srun --gres=gpu:a100-sxm4-80gb:1 --time=1:00:00 --pty bash -i

# 激活环境
conda activate kava

# 快速测试训练（1 个 epoch）
python train.py \
    --config configs/llama1b_aug.yaml \
    --output_dir outputs/test \
    --epochs 1 \
    --seed 42

# 完成后退出
exit
```

**如果遇到问题**：
- 检查 SLURM 脚本是否正确：`cat submit_multi_seed.slurm`
- 检查日志目录：`mkdir -p logs`
- 参考 [`SLURM_INTERACTIVE_GUIDE.md`](SLURM_INTERACTIVE_GUIDE.md)

---

## 📊 阶段 4: 监控训练进度（48 小时）

### 实时监控命令

```bash
# 1. 查看任务队列
squeue --me

# 输出示例：
# JOBID  PARTITION  NAME           ST  TIME      NODES
# 12345  compute    kava-multi-se  R   2:30:15   1     （运行中）
# 12346  compute    kava-multi-se  PD  0:00      1     （排队中）

# 2. 实时查看训练日志
tail -f logs/kava_12345_0.out

# 应该看到训练进度：
# Epoch 1/3: 100%|██████| 500/500 [10:30<00:00]
# Loss: 2.345, Acc: 45.6%
# Saving checkpoint to outputs/llama1b_aug_seed_42/checkpoint_epoch_1

# 3. 查看错误日志（如果有问题）
tail -f logs/kava_12345_0.err

# 4. 查看所有任务的简要状态
watch -n 30 'squeue --me'  # 每 30 秒刷新

# 5. 使用 WandB 远程监控（推荐）
# 在浏览器打开 https://wandb.ai/your-username/kava
# 实时查看：Loss 曲线、准确率、GPU 使用率
```

### 常用监控命令

```bash
# 查看 GPU 使用情况（需要在计算节点上）
scir-watch -s                    # 查看所有节点 GPU 状态
scir-watch gpu06 gpustat         # 查看特定节点

# 查看任务详细信息
scontrol show job 12345

# 查看任务资源使用
sacct -j 12345 --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize

# 取消任务（如果需要）
scancel 12345                    # 取消单个任务
scancel -u $USER                 # 取消所有任务

# 检查输出目录
ls -lh outputs/llama1b_aug_multi_seed/
# 应该看到：
# seed_42/
# seed_123/
# seed_456/
```

### 任务状态说明

| 状态 | 含义 | 操作 |
|------|------|------|
| `PD` (Pending) | 排队等待资源 | 等待即可 |
| `R` (Running) | 正在运行 | 查看日志监控 |
| `CG` (Completing) | 即将完成 | 等待完成 |
| `CD` (Completed) | 已完成 | 检查结果 |
| `F` (Failed) | 失败 | 查看错误日志 |
| `CA` (Cancelled) | 已取消 | 重新提交 |

### 预计时间线

```
提交后：
├─ 0-5 分钟：排队等待 GPU 资源（取决于集群负载）
├─ 5-10 分钟：任务开始运行，模型初始化
├─ 10 分钟-12 小时：训练进行中（第一个 seed）
├─ 12-24 小时：第一个 seed 完成，开始第二个
├─ 24-36 小时：第二个 seed 完成，开始第三个
└─ 36-48 小时：所有种子完成，自动聚合结果 ✓
```

**如果遇到问题**：
- 任务失败 → 查看 `logs/kava_*_*.err`
- 参考 [`HPC_REFERENCE.md`](HPC_REFERENCE.md) 的"监控和调试"章节
- 参考 [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md) 的"故障排除"章节

---

## 📈 阶段 5: 生成论文结果（5 分钟）

### Step 1: 检查输出文件

```bash
# 训练完成后，检查输出目录
cd ~/kava
tree outputs/

# 应该看到：
# outputs/
# ├── llama1b_aug_multi_seed/
# │   ├── seed_42/
# │   │   ├── best_checkpoint/
# │   │   ├── results_gsm8k.yaml
# │   │   ├── results_gsm8k-hard.yaml
# │   │   └── results_svamp.yaml
# │   ├── seed_123/
# │   │   └── ...
# │   ├── seed_456/
# │   │   └── ...
# │   └── aggregated_results.json  ← 聚合结果
# ├── llama1b_aug_nl_multi_seed/
# │   └── ...
# ├── qwen05b_aug_multi_seed/
# │   └── ...
# └── llama3b_aug_multi_seed/
#     └── ...
```

### Step 2: 生成 LaTeX 表格

```bash
# 运行格式化脚本
python format_results.py \
    --input outputs/*/aggregated_results.json \
    --output results/

# 生成的文件：
# results/
# ├── table1.tex          ← 论文 Table 1（GSM8k-AUG 结果）
# ├── table2.tex          ← 论文 Table 2（GSM8k-AUG-NL 结果）
# ├── all_results.csv     ← CSV 格式（便于分析）
# └── summary.txt         ← 结果摘要
```

### Step 3: 查看结果

```bash
# 查看 LaTeX 表格
cat results/table1.tex

# 输出示例：
# \begin{table}[t]
# \caption{Test accuracy (\%) on GSM8k, GSM8k-Hard, and SVAMP...}
# \begin{tabular}{llccc}
# \toprule
# Model & Dataset & GSM8k & GSM8k-Hard & SVAMP \\
# \midrule
# LLaMA-3.2-1B & GSM8k-AUG & 56.5 (0.4) & 34.2 (0.6) & 48.3 (0.5) \\
# LLaMA-3.2-1B & GSM8k-AUG-NL & 55.8 (0.5) & 33.7 (0.7) & 47.9 (0.6) \\
# Qwen-2.5-0.5B & GSM8k-AUG & 42.3 (0.8) & 28.1 (0.9) & 35.7 (1.1) \\
# LLaMA-3.2-3B & GSM8k-AUG & 67.2 (0.3) & 45.8 (0.5) & 58.9 (0.4) \\
# \bottomrule
# \end{tabular}
# \end{table}

# 查看 CSV 格式
cat results/all_results.csv

# 查看结果摘要
cat results/summary.txt
```

### Step 4: 下载结果到本地

```bash
# 在本地终端运行（Windows PowerShell）
scp -r username@hpc.example.edu:~/kava/results/ D:\kava\results\

# Linux/macOS
scp -r username@hpc.example.edu:~/kava/results/ /path/to/local/kava/results/

# 或使用 VSCode Remote SSH 直接下载
# 右键 results/ → Download
```

---

## ✅ 完成检查清单

在论文中使用结果前，请确认：

- [ ] 所有 4 个配置都成功完成（llama1b_aug, llama1b_aug_nl, qwen05b_aug, llama3b_aug）
- [ ] 每个配置都有 3 个种子的结果（seed_42, seed_123, seed_456）
- [ ] 每个种子都在 3 个数据集上评估（GSM8k, GSM8k-Hard, SVAMP）
- [ ] `aggregated_results.json` 包含均值和标准差
- [ ] `table1.tex` 和 `table2.tex` 格式正确
- [ ] 结果与论文中的数值范围一致（±5% 误差正常）

**结果验证**：
```bash
# 检查所有任务完成
squeue --me  # 应该为空（所有任务完成）

# 检查结果文件数量
find outputs/ -name "results_*.yaml" | wc -l
# 应该输出：36（4 配置 × 3 种子 × 3 数据集）

# 检查聚合结果
ls -lh outputs/*/aggregated_results.json
# 应该有 4 个文件

# 检查 LaTeX 表格
ls -lh results/*.tex
# 应该有 table1.tex 和 table2.tex
```

---

## 🔧 故障排除

### 常见问题速查

| 问题 | 解决方案 | 参考文档 |
|------|---------|---------|
| 上传项目失败 | 检查 SSH 配置，使用 VSCode Remote SSH | 本文档"阶段 1" |
| 环境配置失败 | 运行 `setup_hpc_models.sh`，检查模块加载 | [`HPC_REFERENCE.md`](HPC_REFERENCE.md) |
| 模型下载超时 | 使用公共模型库 `/home/share/models` | [`HPC_MODELS_QUICKSTART.md`](HPC_MODELS_QUICKSTART.md) |
| 任务排队太久 | 检查集群负载 `sinfo -p compute` | [`SLURM_INTERACTIVE_GUIDE.md`](SLURM_INTERACTIVE_GUIDE.md) |
| 任务失败 | 查看 `logs/kava_*.err`，检查 GPU 内存 | [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md) |
| GPU 内存不足 | 减小 batch size 或使用梯度累积 | [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md) |
| 代理设置 | 使用本地代理加速下载 | [`SSH_PORT_FORWARDING.md`](SSH_PORT_FORWARDING.md) |

### 详细故障排除

#### 问题 1: 任务一直处于 PD（排队）状态

```bash
# 检查原因
squeue --me --start

# 如果显示资源不足，可以：
# 1. 等待（推荐）
# 2. 减少资源需求（修改 submit_multi_seed.slurm）
#    #SBATCH --mem=32G  （改为 32G）
#    #SBATCH --time=24:00:00  （改为 24 小时）
```

#### 问题 2: 训练速度很慢

```bash
# 检查 GPU 使用率
scir-watch gpu06 gpustat

# 如果 GPU 利用率低（<50%），可能是：
# - Batch size 太小 → 增大 batch_size
# - 数据加载慢 → 增加 num_workers
# - 模型在 CPU → 检查 CUDA 是否可用
```

#### 问题 3: 结果文件缺失

```bash
# 检查任务是否真的完成
sacct -j 12345 --format=JobID,State,ExitCode

# 如果 ExitCode 不是 0:0，说明有错误
tail -100 logs/kava_12345_0.err

# 重新运行失败的任务
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm
```

---

## 📚 推荐阅读顺序

### 第一次使用 HPC（必读）

1. **本文档** (`GETTING_STARTED_HPC.md`) - 跟随本指南完成所有步骤 ⭐⭐⭐⭐⭐
2. [`HPC_REFERENCE.md`](HPC_REFERENCE.md) - 浏览"快速开始"和"SLURM 命令"章节 ⭐⭐⭐⭐☆
3. [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md) - 了解详细配置和参数 ⭐⭐⭐⭐☆

### 遇到问题时（按需阅读）

4. [`HPC_MODELS_QUICKSTART.md`](HPC_MODELS_QUICKSTART.md) - 公共模型库配置 ⭐⭐⭐☆☆
5. [`SLURM_INTERACTIVE_GUIDE.md`](SLURM_INTERACTIVE_GUIDE.md) - 交互式调试 ⭐⭐⭐☆☆
6. [`SSH_PORT_FORWARDING.md`](SSH_PORT_FORWARDING.md) - 远程监控（TensorBoard/Jupyter） ⭐⭐⭐☆☆
7. [`CONDA_CUDA_GUIDE.md`](CONDA_CUDA_GUIDE.md) - CUDA 环境问题 ⭐⭐☆☆☆

### 高级功能（可选）

8. [`CONTAINER_QUICKSTART.md`](CONTAINER_QUICKSTART.md) - 容器化部署 ⭐⭐☆☆☆
9. [`MULTI_SEED_GUIDE.md`](docs/MULTI_SEED_GUIDE.md) - 多种子实验细节 ⭐⭐☆☆☆

---

## 💡 最佳实践

1. **使用 VSCode Remote SSH**：最方便的文件同步和编辑方式
2. **优先使用公共模型库**：`/home/share/models` 避免重复下载
3. **使用 WandB 监控**：远程查看训练进度，无需登录 HPC
4. **定期检查日志**：`tail -f logs/kava_*.out` 及时发现问题
5. **备份重要结果**：定期下载 `outputs/` 到本地
6. **使用 tmux**：长时间任务在后台运行，防止 SSH 断开
7. **批量提交任务**：使用 `hpc_run_all.sh` 一次提交所有实验

---

## 📞 获取帮助

### 文档索引

- **快速开始**: 本文档
- **HPC 命令**: [`HPC_REFERENCE.md`](HPC_REFERENCE.md)
- **完整复现**: [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md)
- **交互调试**: [`SLURM_INTERACTIVE_GUIDE.md`](SLURM_INTERACTIVE_GUIDE.md)
- **公共模型**: [`HPC_MODELS_QUICKSTART.md`](HPC_MODELS_QUICKSTART.md)
- **端口映射**: [`SSH_PORT_FORWARDING.md`](SSH_PORT_FORWARDING.md)
- **容器部署**: [`CONTAINER_QUICKSTART.md`](CONTAINER_QUICKSTART.md)

### 命令速查

```bash
# === 环境 ===
conda activate kava                    # 激活环境
source ~/.bashrc                       # 重新加载配置

# === 提交任务 ===
./hpc_run_all.sh                       # 一键提交所有实验
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm  # 单个配置

# === 监控 ===
squeue --me                            # 查看任务队列
tail -f logs/kava_*.out                # 实时日志
scir-watch -s                          # GPU 状态

# === 结果 ===
python format_results.py               # 生成 LaTeX 表格
cat results/table1.tex                 # 查看结果

# === 清理 ===
scancel -u $USER                       # 取消所有任务
rm -rf outputs/test                    # 删除测试输出
```

---

## 🎉 完成

恭喜！如果你完成了所有步骤，现在应该有：

✅ 4 个模型配置的完整训练结果  
✅ 每个配置 3 个种子的统计数据（均值 ± 标准差）  
✅ 格式化的 LaTeX 表格（可直接用于论文）  
✅ CSV 格式的结果（便于进一步分析）  

**下一步**：将 `results/table1.tex` 和 `results/table2.tex` 复制到你的论文中！

---

**预计总时间**：
- 配置时间：30 分钟
- 训练时间：36-48 小时（自动运行，无需人工干预）
- 生成结果：5 分钟

**祝你实验顺利！** 🚀
