# KAVA 论文复现指南

**严格按照原始论文配置复现 KAVA 实验结果**

本指南提供从零开始复现 KAVA 论文 Table 1 和 Table 2 的完整步骤。

---

## 📋 目录

1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [完整复现流程](#完整复现流程)
4. [结果验证](#结果验证)
5. [故障排除](#故障排除)

---

## 🔧 环境准备

### 系统要求

- **GPU**: NVIDIA GPU with 24GB+ VRAM (推荐 A100/H100)
- **CUDA**: 11.8+ 
- **Python**: 3.9+
- **OS**: Linux/Windows with PowerShell

---

### 方案 A: HPC 集群环境（推荐）

#### Step 1: 加载 CUDA Module

```bash
# 1. 初始化 module 命令
. /usr/share/modules/init/bash  # 如果使用 zsh，改为 zsh

# 2. 验证 module 可用
module -V

# 3. 添加模块路径
module use --append /home/share/modules/modulefiles

# 4. 查看可用 CUDA 版本
module avail cuda

# 5. 加载 CUDA（选择 11.8 或更高版本）
module load cuda/11.8.0  # 或 cuda/12.1.1

# 6. 验证 CUDA
nvcc -V
nvidia-smi
```

#### Step 2: 创建 Python 环境

**方法 1: 使用 HPC 系统 CUDA（推荐用于集群）**

```bash
# 1. 加载 Anaconda（如果 HPC 提供）
module load anaconda3  # 或 miniconda3

# 2. 创建虚拟环境
conda create -n kava python=3.10 -y
conda activate kava

# 3. 安装 PyTorch（匹配系统 CUDA 版本）
# 对于 CUDA 11.8
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 对于 CUDA 12.1
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装额外依赖
pip install peft wandb bitsandbytes

# 6. 配置 HPC 公共模型库（重要！）
# HPC 已预下载常用模型至共享位置，无需等待下载
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models

# 写入 ~/.bashrc 以永久生效
echo 'export HF_HOME=/home/share/models' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/home/share/models' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/home/share/models' >> ~/.bashrc
source ~/.bashrc

# 验证模型可用性
ls /home/share/models/models--meta-llama--Llama-3.2-1B-Instruct
ls /home/share/models/models--Qwen--Qwen2.5-0.5B-Instruct

# 7. 验证安装
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

---

**方法 2: 使用 Conda 安装 CUDA（推荐用于个人开发）**

这种方法将 CUDA 安装在 Conda 环境内，无需依赖系统 CUDA，更灵活。

```bash
# 1. 创建虚拟环境并同时安装 CUDA
conda create -n kava python=3.10 cudatoolkit=11.8 -c nvidia -y
conda activate kava

# 或者在已有环境中安装 CUDA
conda activate kava
conda install cudatoolkit=11.8 -c nvidia

# 查看可用的 CUDA 版本
conda search cudatoolkit -c nvidia

# 2. 安装 PyTorch（自动匹配 Conda CUDA）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. 安装项目依赖
pip install -r requirements.txt
pip install peft wandb bitsandbytes

# 4. 验证安装
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
nvcc -V  # 应显示 Conda 安装的 CUDA 版本
```

**设置环境变量（解决编译问题）**

如果编译扩展（如 apex、deepspeed）时找不到 nvcc，需要设置环境变量：

```bash
# 1. 查找 Conda 环境路径
conda env list
# 输出示例: kava  /home/username/.conda/envs/kava

# 2. 设置 CUDA 路径
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 3. 创建 lib64 符号链接（解决动态链接库错误）
cd $CONDA_PREFIX
ln -s lib lib64

# 4. 验证
echo $CUDA_HOME
nvcc -V
```

**永久设置（添加到激活脚本）**

```bash
# 方法 1: 使用 conda env config
conda activate kava
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 方法 2: 添加到 ~/.bashrc（全局生效）
cat >> ~/.bashrc << 'EOF'
# KAVA CUDA Environment
if [[ "$CONDA_DEFAULT_ENV" == "kava" ]]; then
    export CUDA_HOME=$CONDA_PREFIX
    export CUDA_PATH=$CONDA_PREFIX
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi
EOF

# 重新激活环境测试
conda deactivate
conda activate kava
echo $CUDA_HOME
```

**对比两种方法**

| 特性 | 系统 CUDA (module) | Conda CUDA |
|------|-------------------|------------|
| **适用场景** | HPC 集群 | 个人工作站 |
| **权限要求** | 无需管理员 | 无需管理员 |
| **版本切换** | `module swap` | `conda activate` |
| **多版本共存** | ✅ 系统级 | ✅ 环境级 |
| **编译支持** | ✅ 原生 | ⚠️ 需配置 |
| **磁盘占用** | 共享 | 每环境 ~3GB |
| **推荐度** | ⭐⭐⭐⭐⭐ (HPC) | ⭐⭐⭐⭐⭐ (本地) |

---

#### Step 3: 创建 SLURM 提交脚本

创建 `submit_kava.sh`：

```bash
#!/bin/bash
#SBATCH --job-name=kava-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/kava_%j.out
#SBATCH --error=logs/kava_%j.err

# 加载 CUDA
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/11.8.0

# 加载 Anaconda
module load anaconda3

# 激活环境
source activate kava

# 验证 GPU
nvidia-smi
echo "CUDA Version: $(nvcc -V)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.version.cuda)')"

# 运行训练
python train.py \
    --config configs/llama1b_aug.yaml \
    --output_dir $SLURM_SUBMIT_DIR/outputs/seed_${SLURM_JOB_ID} \
    --seed 42 \
    --use_wandb

echo "Job completed at $(date)"
```

提交任务：

```bash
# 创建日志目录
mkdir -p logs

# 提交单个任务
sbatch submit_kava.sh

# 提交多个种子（批量）
for seed in 42 123 456; do
    sbatch --export=SEED=$seed submit_kava.sh
done

# 查看任务状态
squeue -u $USER
```

---

### 方案 B: 本地环境（个人工作站）

```bash
# 1. 创建虚拟环境（推荐）
conda create -n kava python=3.10
conda activate kava

# 2. 安装 PyTorch（根据您的 CUDA 版本）
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
pip install -r requirements.txt

# 4. 安装额外依赖
pip install peft wandb

# 5. 验证安装
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## ⚡ 快速开始

### 最小可运行示例（5 分钟）

```bash
# 1. 快速验证环境
python smoke_test.py

# 2. 单次训练测试（使用小数据集）
python train.py --config configs/llama1b_aug.yaml --quick_test

# 3. 单次评估测试
python evaluate.py --checkpoint checkpoints/test --dataset gsm8k --quick_test
```

---

## 🎯 完整复现流程

### 复现论文 Table 1: Test Accuracy

#### **方案 A: HPC 自动化运行（推荐）**

```bash
# 1. 赋予执行权限
chmod +x hpc_run_all.sh submit_multi_seed.slurm submit_aggregate.slurm

# 2. 运行所有配置
./hpc_run_all.sh

# 或运行指定配置
./hpc_run_all.sh llama1b_aug
./hpc_run_all.sh llama1b_aug qwen05b_aug

# 3. 监控任务
watch -n 10 'squeue -u $USER'
tail -f logs/kava_*.out

# 4. 任务完成后查看结果
cat outputs/llama1b_aug_multi_seed/aggregated_results.yaml
```

**脚本功能**：
- ✅ 自动提交 3 个种子的训练任务（SLURM 数组作业）
- ✅ 自动提交依赖的聚合任务
- ✅ 并行运行多个配置
- ✅ 自动错误处理和日志记录

**预计时间**: 
- 单个配置（3 种子）：约 36-48 小时（并行运行）
- 全部 4 个配置：约 36-48 小时（可并行）

---

#### **方案 B: HPC 手动提交（更灵活）**

##### **单个配置提交**

```bash
# 1. 提交训练任务（3 个种子并行）
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm

# 2. 查看任务状态
squeue -u $USER

# 3. 等待完成后提交聚合
sbatch --export=CONFIG=llama1b_aug submit_aggregate.slurm

# 4. 查看结果
cat outputs/llama1b_aug_multi_seed/aggregated_results.yaml
```

##### **批量提交多个配置**

```bash
# 提交所有配置
for config in llama1b_aug llama1b_aug_nl qwen05b_aug llama3b_aug; do
    echo "Submitting $config..."
    sbatch --export=CONFIG=$config submit_multi_seed.slurm
done

# 查看所有任务
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
```

---

#### **方案 C: 本地 PowerShell 自动化运行**

#### **方案 C: 本地 PowerShell 自动化运行**

```powershell
# Windows PowerShell
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456

# 等价的 Python 命令
python run_multi_seed.py --config llama1b_aug --seeds 42 123 456
```

这个脚本会自动：
1. ✅ 使用 3 个种子训练模型（42, 123, 456）
2. ✅ 在 3 个数据集上评估（GSM8k, GSM8k-Hard, SVAMP）
3. ✅ 聚合结果并计算均值±标准差
4. ✅ 生成 JSON/YAML 结果文件

**预计时间**: 每个模型约 10-15 小时（取决于 GPU）

---

#### **方案 D: 手动分步运行**

##### **Step 1: 训练模型（3 个种子）**

```bash
# Seed 1
python train.py \
    --config configs/llama1b_aug.yaml \
    --output_dir outputs/llama1b_aug/seed_42 \
    --seed 42 \
    --use_wandb

# Seed 2
python train.py \
    --config configs/llama1b_aug.yaml \
    --output_dir outputs/llama1b_aug/seed_123 \
    --seed 123 \
    --use_wandb

# Seed 3
python train.py \
    --config configs/llama1b_aug.yaml \
    --output_dir outputs/llama1b_aug/seed_456 \
    --seed 456 \
    --use_wandb
```

##### **Step 2: 评估模型（每个种子 × 3 个数据集）**

```bash
# Seed 42
for dataset in gsm8k gsm8k-hard svamp; do
    python evaluate.py \
        --checkpoint_dir outputs/llama1b_aug/seed_42/best_checkpoint \
        --eval_dataset $dataset \
        --output outputs/llama1b_aug/seed_42/results_${dataset}.yaml
done

# Seed 123
for dataset in gsm8k gsm8k-hard svamp; do
    python evaluate.py \
        --checkpoint_dir outputs/llama1b_aug/seed_123/best_checkpoint \
        --eval_dataset $dataset \
        --output outputs/llama1b_aug/seed_123/results_${dataset}.yaml
done

# Seed 456
for dataset in gsm8k gsm8k-hard svamp; do
    python evaluate.py \
        --checkpoint_dir outputs/llama1b_aug/seed_456/best_checkpoint \
        --eval_dataset $dataset \
        --output outputs/llama1b_aug/seed_456/results_${dataset}.yaml
done
```

##### **Step 3: 聚合结果**

```bash
python aggregate_multi_seed.py \
    --seed_dirs \
        outputs/llama1b_aug/seed_42 \
        outputs/llama1b_aug/seed_123 \
        outputs/llama1b_aug/seed_456 \
    --datasets gsm8k gsm8k-hard svamp \
    --model_name "KAVA-LLaMA-3.2-1B" \
    --output_json outputs/llama1b_aug/aggregated_results.json \
    --output_yaml outputs/llama1b_aug/aggregated_results.yaml
```

##### **Step 4: 生成 LaTeX 表格**

```bash
python format_results.py \
    --input_dir outputs/ \
    --output_latex paper_tables.tex \
    --output_csv results.csv
```

**输出文件**:
- `paper_tables.tex` - 可直接插入论文的 LaTeX 表格
- `results.csv` - Excel 可读的结果文件

---

### 复现所有模型配置（论文 Table 6）

#### **6 个模型配置 × 3 个种子 = 18 次训练**

```bash
# LLaMA-3.2-1B
.\run_multi_seed.ps1 -Config llama1b_aug      # GSM8k-AUG
.\run_multi_seed.ps1 -Config llama1b_aug_nl   # GSM8k-AUG-NL

# Qwen2.5-0.5B
.\run_multi_seed.ps1 -Config qwen05b_aug      # GSM8k-AUG

# LLaMA-3.2-3B
.\run_multi_seed.ps1 -Config llama3b_aug      # GSM8k-AUG
```

**预计总时间**: 约 150-200 小时（可并行）

---

## 📊 结果验证

### 检查训练日志

```bash
# 查看训练损失曲线
cat outputs/llama1b_aug/seed_42/train.log

# 或使用 WandB
wandb login
# 访问: https://wandb.ai/your-username/kava-reproduction
```

### 检查聚合结果

```bash
# YAML 格式（易读）
cat outputs/llama1b_aug/aggregated_results.yaml

# JSON 格式（程序可读）
python -m json.tool outputs/llama1b_aug/aggregated_results.json
```

**预期输出示例**:
```yaml
llama1b_aug:
  num_seeds: 3
  datasets:
    gsm8k:
      accuracy_mean: 0.565
      accuracy_std: 0.004
      forward_passes_mean: 28.3
      forward_passes_std: 0.8
    gsm8k-hard:
      accuracy_mean: 0.342
      accuracy_std: 0.006
```

### 对比论文结果

打开生成的 `paper_tables.tex`，对比 Table 1 数值：

| Model | Dataset | GSM8k | GSM8k-Hard | SVAMP |
|-------|---------|-------|------------|-------|
| **论文** | LLaMA-1B-AUG | 56.5 (0.4) | 34.2 (0.6) | 48.3 (0.5) |
| **您的** | LLaMA-1B-AUG | ? | ? | ? |

**允许误差**: ±2% 属于正常范围

---

## 🛠️ 故障排除

### HPC 专用问题

#### 问题 1: Module 命令不可用

**症状**:
```
bash: module: command not found
```

**解决方案**:
```bash
# 初始化 module（根据您的 shell）
. /usr/share/modules/init/bash   # bash
. /usr/share/modules/init/zsh    # zsh
. /usr/share/modules/init/fish   # fish

# 添加到 ~/.bashrc 永久生效
echo '. /usr/share/modules/init/bash' >> ~/.bashrc
echo 'module use --append /home/share/modules/modulefiles' >> ~/.bashrc
```

---

#### 问题 2: SLURM 任务一直处于 PENDING 状态

**症状**:
```
JOBID   PARTITION   STATE   TIME   REASON
12345   gpu         PD      0:00   (Resources)
```

**解决方案**:
```bash
# 查看任务详情
scontrol show job <JOB_ID>

# 查看队列状态
sinfo -p gpu

# 常见原因：
# 1. GPU 资源不足 - 减少请求的 GPU 数量或等待
# 2. 内存不足 - 减少 --mem 参数
# 3. 超过配额 - 检查 sacctmgr show user $USER

# 取消并重新提交
scancel <JOB_ID>
sbatch --gres=gpu:1 submit_multi_seed.slurm  # 减少 GPU 需求
```

---

#### 问题 3: CUDA 版本不匹配

**症状**:
```
RuntimeError: CUDA version mismatch. PyTorch compiled with CUDA 11.8 but system has CUDA 12.1
```

**解决方案**:
```bash
# 方法 1: 加载匹配的 CUDA 版本
module load cuda/11.8.0

# 方法 2: 重新安装匹配的 PyTorch
pip uninstall torch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
nvcc -V
```

---

#### 问题 4: 任务被 OOM Killer 杀死

**症状**:
```
slurmstepd: error: Detected X oom-kill event(s) in StepId=12345.batch
```

**解决方案**:
```bash
# 1. 增加内存请求
#SBATCH --mem=128G  # 从 64G 增加到 128G

# 2. 或减少批次大小
# 编辑 configs/llama1b_aug.yaml
training:
  batch_size: 64  # 从 128 降低
  gradient_accumulation_steps: 2

# 3. 启用混合精度
system:
  mixed_precision: "bf16"  # 使用 bfloat16 减少内存
```

---

#### 问题 5: 多个 Conda 环境冲突

**症状**:
```
CondaError: Environment kava is not activated
```

**解决方案**:
```bash
# 确保正确激活
conda deactivate  # 先退出当前环境
conda activate kava

# 或在 SLURM 脚本中使用完整路径
source /home/$USER/.conda/envs/kava/bin/activate

# 验证
which python
python --version
```

---

#### 问题 6: Conda CUDA 编译错误 "nvcc not found"

**症状**:
```
RuntimeError: Ninja is required to load C++ extensions
OSError: CUDA_HOME environment variable is not set
```

**解决方案**:
```bash
# 1. 检查 Conda 环境路径
conda activate kava
echo $CONDA_PREFIX
# 输出: /home/username/.conda/envs/kava

# 2. 设置 CUDA 环境变量
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

# 3. 验证 nvcc 可用
which nvcc
nvcc -V

# 4. 如果仍然失败，创建 lib64 链接
cd $CONDA_PREFIX
ln -s lib lib64
ls -la | grep lib64  # 验证链接

# 5. 重新安装需要编译的包
pip install --no-cache-dir --force-reinstall apex
pip install --no-cache-dir --force-reinstall deepspeed
```

---

#### 问题 7: 动态链接库错误 "cannot open shared object file"

**症状**:
```
OSError: libcudart.so.11.8: cannot open shared object file: No such file or directory
ImportError: libcublas.so.11: cannot open shared object file
```

**解决方案**:
```bash
# 方法 1: 创建 lib64 符号链接
cd $CONDA_PREFIX
ln -s lib lib64

# 方法 2: 更新 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH  # 如果有

# 方法 3: 检查库文件是否存在
ls $CONDA_PREFIX/lib/libcudart*
ls $CONDA_PREFIX/lib/libcublas*

# 如果库不存在，重新安装 CUDA
conda install cudatoolkit=11.8 -c nvidia --force-reinstall

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

---

#### 问题 8: DeepSpeed FusedAdam JIT 编译失败

**症状**:
```
RuntimeError: Error building extension 'fused_adam'
FileNotFoundError: Could not find module 'cudart64_110.dll'
```

**解决方案**:
```bash
# 1. 确保 lib64 链接存在
cd $CONDA_PREFIX
ln -s lib lib64

# 2. 设置完整的 CUDA 环境变量
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH

# 3. 清除 DeepSpeed 缓存
rm -rf ~/.cache/torch_extensions/*

# 4. 重新运行
python train.py --config configs/llama1b_aug.yaml

# 或禁用 FusedAdam（使用标准优化器）
# 编辑 train.py，将 FusedAdam 替换为 torch.optim.AdamW
```

---

### 通用问题

#### 问题 1: CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解决方案**:
```yaml
# 编辑 configs/llama1b_aug.yaml
training:
  batch_size: 64  # 从 128 降低到 64
  gradient_accumulation_steps: 2  # 从 1 增加到 2
```

---

### 问题 2: 导入错误 "No module named 'peft'"

**症状**:
```
ModuleNotFoundError: No module named 'peft'
```

**解决方案**:
```bash
pip install peft==0.7.0
pip install bitsandbytes  # 如果使用量化
```

---

### 问题 3: HuggingFace 下载超时

**症状**:
```
ConnectionError: HTTPSConnectionPool timeout
```

**解决方案**:

**HPC 集群用户（推荐）**:
```bash
# 方法 1: 使用 HPC 公共模型库（最快）
# HPC 已预下载常用模型至 /home/share/models，无需等待
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models

# 验证模型是否可用
ls /home/share/models/models--meta-llama--Llama-3.2-1B-Instruct
ls /home/share/models/models--Qwen--Qwen2.5-0.5B-Instruct

# 在训练脚本中自动使用
python train.py --config configs/llama1b_aug.yaml  # 自动从共享路径加载
```

**本地/个人开发**:
```bash
# 方法 2: 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法 3: 预先下载模型
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

# 方法 4: 使用本地模型路径
# 编辑 configs/llama1b_aug.yaml
model:
  name: "/path/to/local/Llama-3.2-1B-Instruct"
```

**说明**: HPC 公共模型库位于 `/home/share/models`，包含从 HuggingFace 下载的常用开源模型，持续更新。建议 HPC 用户优先使用此方式，可显著减少下载时间。

---

### 问题 4: PowerShell 脚本执行策略错误

**症状**:
```
无法加载文件 run_multi_seed.ps1，因为在此系统上禁止运行脚本
```

**解决方案**:
```powershell
# 临时允许
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 或直接运行
powershell -ExecutionPolicy Bypass -File .\run_multi_seed.ps1
```

---

### 问题 5: 数据集加载失败

**症状**:
```
FileNotFoundError: Dataset 'whynlp/gsm8k-aug' not found
```

**解决方案**:
```bash
# 手动下载数据集
python -c "from datasets import load_dataset; load_dataset('whynlp/gsm8k-aug')"

# 或使用本地数据集
# 编辑 configs/llama1b_aug.yaml
dataset:
  name: "/path/to/local/gsm8k-aug"
```

---

## 📈 高级功能

### 使用增量解码加速推理（3-5x）

```bash
# inference.py 已自动启用
python inference.py \
    --checkpoint outputs/llama1b_aug/seed_42/best_checkpoint \
    --config configs/llama1b_aug.yaml \
    --mode interactive
```

### 性能基准测试

```bash
# 对比增量解码 vs 原始解码
python benchmark_incremental_decoding.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max_new_tokens 50
```

### 自定义配置

```bash
# 使用 model_configs.py 管理配置
python model_configs.py --list --verbose
python model_configs.py --compare llama1b_aug llama3b_aug
python model_configs.py --export configs_backup/ --format yaml
```

---

## 📚 配置文件说明

### 论文 Table 6 对应关系

| 配置文件 | 模型 | 数据集 | lr | α₁ | α₂ | Loss |
|---------|------|--------|----|----|-------|------|
| `llama1b_aug.yaml` | LLaMA-1B | AUG | 8e-4 | 10 | 1 | smooth_l1 |
| `llama1b_aug_nl.yaml` | LLaMA-1B | AUG-NL | 8e-4 | 10 | 1 | smooth_l1 |
| `qwen05b_aug.yaml` | Qwen-0.5B | AUG | 5e-4 | 10 | 1 | mse |
| `llama3b_aug.yaml` | LLaMA-3B | AUG | 2e-4 | 20 | 2 | smooth_l1 |

### 关键超参数

```yaml
latent:
  num_tokens: 24      # M (潜在 token 数量)
  num_iterations: 3   # T (Jacobi 迭代次数)

loss:
  alpha1_codi: 10.0   # α₁ (CODI 损失权重)
  alpha2_kv: 1.0      # α₂ (KV 蒸馏权重)

rkv:
  lambda: 0.1         # λ (重要性 vs 冗余平衡)

lora:
  r: 128              # LoRA 秩
  alpha: 32           # LoRA alpha
```

---

## 🎓 预期输出示例

### 训练完成后的目录结构

```
outputs/
├── llama1b_aug_multi_seed/
│   ├── seed_42/
│   │   ├── best_checkpoint/
│   │   ├── results_gsm8k.yaml
│   │   ├── results_gsm8k-hard.yaml
│   │   └── results_svamp.yaml
│   ├── seed_123/
│   │   └── ...
│   ├── seed_456/
│   │   └── ...
│   └── aggregated_results.json
├── llama1b_aug_nl_multi_seed/
│   └── ...
└── ...
```

### LaTeX 表格预览

```latex
\begin{table}[t]
\caption{Test accuracy (\%) on GSM8k, GSM8k-Hard, and SVAMP...}
\begin{tabular}{llccc}
\toprule
Model & Dataset & GSM8k & GSM8k-Hard & SVAMP \\
\midrule
LLaMA-3.2-1B & GSM8k-AUG & 56.5 (0.4) & 34.2 (0.6) & 48.3 (0.5) \\
LLaMA-3.2-1B & GSM8k-AUG-NL & 55.8 (0.5) & 33.7 (0.7) & 47.9 (0.6) \\
...
\bottomrule
\end{tabular}
\end{table}
```

---

## 🌐 远程监控和开发

### TensorBoard 可视化

#### 在 HPC 上启动 TensorBoard

```bash
# 方法 1: 交互式会话
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:1 --time=2:00:00 --pty bash -i
conda activate kava
tensorboard --logdir outputs/llama1b_aug_seed_42/logs --port 6006 --bind_all

# 方法 2: 后台运行
nohup tensorboard --logdir outputs/llama1b_aug_seed_42/logs --port 6006 --bind_all > tensorboard.log 2>&1 &
```

#### 本地访问

```bash
# 在本地终端建立 SSH 隧道（将 HPC 的 6006 映射到本地 6006）
ssh -N -L 6006:gpu10:6006 username@hpc.example.edu

# 或使用其他本地端口
ssh -N -L 22222:gpu10:6006 username@hpc.example.edu

# 浏览器打开
# http://localhost:6006
# 或 http://localhost:22222
```

#### VSCode 自动端口转发（推荐）

1. 使用 VSCode Remote SSH 连接到 HPC
2. 在远程终端启动 TensorBoard
3. VSCode 自动检测并提示"转发端口 6006"
4. 点击通知，自动在本地浏览器打开

---

### Jupyter Notebook 开发

#### 启动 Jupyter

```bash
# ⚠️ 必须使用支持 SSH 的节点：gpu10-gpu14
srun -w gpu12 --gres=gpu:a100-sxm4-80gb:1 --time=4:00:00 --pty bash -i
conda activate kava

# 启动 Jupyter Notebook
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# 或 JupyterLab
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# 记录 token：http://gpu12:8888/?token=abc123...
```

#### 本地访问

```bash
# 在本地终端建立隧道
ssh -L 8888:gpu12:8888 username@hpc.example.edu

# 浏览器打开（使用记录的 token）
# http://localhost:8888/?token=abc123...
```

---

### 使用本地代理加速 HPC 下载

如果 HPC 访问 HuggingFace/GitHub 缓慢，可以使用本地代理加速。

#### Step 1: 配置本地代理

```bash
# Clash for Windows
1. 打开 Clash
2. 启用 "Allow LAN" 选项
3. 记住端口（默认 7890）

# Shadowrocket (macOS)
# 默认端口 1089
```

#### Step 2: 建立反向隧道

```bash
# 在本地终端运行（将本地 7890 映射到 HPC 的 55555）
ssh -N -R 55555:localhost:7890 username@hpc.example.edu

# 后台运行
ssh -N -R 55555:localhost:7890 username@hpc.example.edu &
```

#### Step 3: 在 HPC 使用代理

```bash
# 在 HPC 终端配置代理
export http_proxy=http://localhost:55555
export https_proxy=http://localhost:55555
export all_proxy=http://localhost:55555

# 测试连接
curl -I https://www.google.com
curl https://huggingface.co

# 下载模型（通过本地代理加速）
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

# 如果端口冲突，使用其他端口（如 56789）
ssh -N -R 56789:localhost:7890 username@hpc.example.edu
export all_proxy=http://localhost:56789
```

**详细文档**: 参见 `HPC_REFERENCE.md` 的"SSH 端口映射"章节。

---

## 🔗 相关文档

- **HPC 快速参考**: `HPC_REFERENCE.md` - HPC 集群完整指南（含端口映射）
- **交互式使用**: `SLURM_INTERACTIVE_GUIDE.md` - 交互式开发和调试
- **快速参考**: `QUICK_REFERENCE.md` - 常用命令速查
- **项目清单**: `PROJECT_INVENTORY.md` - 所有文件说明
- **论文映射**: `PAPER_MAPPING.md` - 代码与论文对应关系
- **检查清单**: `CHECKLIST.md` - 复现前检查项

---

## 📝 引用

如果本代码对您的研究有帮助，请引用原始论文：

```bibtex
@article{kava2024,
  title={KAVA: Key-value Augmented Distillation for Efficient Reasoning},
  author={...},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

---

## 💡 提示

1. **首次运行建议**: 先用 `smoke_test.py` 验证环境
2. **训练监控**: 使用 WandB 实时监控损失和指标
3. **并行训练**: 可在多个 GPU 上同时运行不同种子
4. **中间检查点**: 每个 epoch 都会保存，可随时恢复训练
5. **磁盘空间**: 每个模型约需 5-10GB 空间

---

## 📧 支持

遇到问题？请检查：
1. ✅ **故障排除**章节
2. ✅ 项目 Issues（如果是开源项目）
3. ✅ 论文原文的实现细节

---

**祝您复现顺利！** 🚀
