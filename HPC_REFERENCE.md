# HPC 快速参考

**KAVA 在 HPC 集群上的常用命令和工作流程**

---

## 🚀 快速开始

### 1. 环境设置（仅需一次）

**方法 A: 使用系统 Module CUDA**

```bash
# 初始化 module
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles

# 添加到 ~/.bashrc
cat >> ~/.bashrc << 'EOF'
# KAVA Environment
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
alias load-kava='module load cuda/11.8.0 anaconda3 && conda activate kava'
EOF

source ~/.bashrc
```

**方法 B: 使用 Conda 安装 CUDA（更灵活）**

```bash
# 一键创建环境
conda create -n kava python=3.10 \
    cudatoolkit=11.8 \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y

conda activate kava

# 配置环境变量（自动激活）
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 创建 lib64 链接
cd $CONDA_PREFIX && ln -s lib lib64

# 安装依赖
pip install -r requirements.txt
pip install peft wandb bitsandbytes

# 配置 HPC 公共模型库
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
echo 'export HF_HOME=/home/share/models' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/home/share/models' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/home/share/models' >> ~/.bashrc

# 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
nvcc -V
ls /home/share/models/models--meta-llama--Llama-3.2-1B-Instruct

# 添加别名
echo "alias load-kava='conda activate kava'" >> ~/.bashrc
```

---

### 2. 加载环境

```bash
load-kava  # 使用别名
# 或
module load cuda/11.8.0 anaconda3
conda activate kava

# HPC 公共模型库环境变量（如果未写入 ~/.bashrc）
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
```

### 3. 提交任务

```bash
# 单个配置
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm

# 批量提交
./hpc_run_all.sh llama1b_aug qwen05b_aug
```

---

## � HPC 公共模型库

### 简介

HPC 集群维护了一个共享模型库，位于 `/home/share/models`，包含从 HuggingFace 下载的常用开源模型。

**优势**：
- ✅ **快速启动**：无需等待模型下载，立即开始训练
- ✅ **节省空间**：多用户共享，无需每人下载
- ✅ **持续更新**：管理员定期更新最新模型
- ✅ **稳定可靠**：避免网络超时问题

---

### 配置方法

#### 方法 1: 永久配置（推荐）

```bash
# 添加到 ~/.bashrc（仅需执行一次）
cat >> ~/.bashrc << 'EOF'
# HuggingFace 公共模型库
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
EOF

# 立即生效
source ~/.bashrc
```

#### 方法 2: 在 SLURM 脚本中配置

```bash
# 在 submit_*.slurm 脚本中添加
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models

# 或写入 Conda 环境激活脚本
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/hf_models.sh << 'EOF'
#!/bin/bash
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/hf_models.sh
```

---

### 可用模型列表

```bash
# 查看所有可用模型
ls -lh /home/share/models/models--*

# 查看 KAVA 项目所需模型
ls -lh /home/share/models/models--meta-llama--Llama-3.2-1B-Instruct
ls -lh /home/share/models/models--meta-llama--Llama-3.2-3B-Instruct
ls -lh /home/share/models/models--Qwen--Qwen2.5-0.5B-Instruct

# 查看模型详情
tree -L 2 /home/share/models/models--meta-llama--Llama-3.2-1B-Instruct
```

**常用模型**：
- `meta-llama/Llama-3.2-1B-Instruct` - LLaMA 1B（KAVA 论文）
- `meta-llama/Llama-3.2-3B-Instruct` - LLaMA 3B（KAVA 论文）
- `Qwen/Qwen2.5-0.5B-Instruct` - Qwen 0.5B（KAVA 论文）

---

### 使用示例

#### 在 Python 代码中使用

```python
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 方法 1: 环境变量已配置（推荐）
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# 自动从 /home/share/models 加载

# 方法 2: 显式指定缓存目录
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    cache_dir="/home/share/models"
)

# 方法 3: 直接使用本地路径
model = AutoModelForCausalLM.from_pretrained(
    "/home/share/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/..."
)
```

#### 在训练脚本中使用

```bash
# 配置环境变量后直接运行
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models

# 训练（自动从共享路径加载）
python train.py --config configs/llama1b_aug.yaml

# 推理
python inference.py --model_path outputs/llama1b_aug_seed_42
```

---

### 验证配置

```bash
# 检查环境变量
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

# 验证模型可访问
python -c "
import os
print('HF_HOME:', os.environ.get('HF_HOME'))
print('Model exists:', os.path.exists('/home/share/models/models--meta-llama--Llama-3.2-1B-Instruct'))
"

# 测试加载模型（不会下载）
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
print('✅ Successfully loaded from shared cache')
"
```

---

### 故障排除

**问题 1: 仍然尝试下载模型**

```bash
# 症状：看到 "Downloading model..." 提示

# 解决：确认环境变量已设置
echo $HF_HOME  # 应该输出 /home/share/models

# 如果为空，重新配置
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
```

**问题 2: 权限拒绝**

```bash
# 症状：Permission denied

# 解决：确认路径可访问
ls -ld /home/share/models  # 应该显示 drwxr-xr-x

# 如果无权限，联系管理员
```

**问题 3: 模型不存在**

```bash
# 症状：Model not found in /home/share/models

# 解决：检查模型是否已下载
ls /home/share/models/models--*/

# 如果模型不存在，请求管理员添加
# 或临时使用 HF_ENDPOINT 镜像下载到个人目录
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/.cache/huggingface
```

---

## �📋 SLURM 常用命令

### SLURM 架构说明

```
登录跳板机 → 提交作业 → SLURM 调度器 → 分配到 GPU 节点
```

**重要**: 
- ⚠️ 大部分节点禁用 SSH（防止资源抢占）
- ✅ 开放 SSH 的节点：`gpu10~gpu14`
- ⚠️ 禁止使用 `sleep 7 day` 等方式抢占资源

---

### 集群状态查询

#### sinfo - 查看集群状态

```bash
# 基本查看
sinfo

# 详细信息（推荐）
# 节点名称 | 状态 | CPU(已分配/可用/其他/总) | CPU负载 | 可用内存 | 总内存 | GPU
sinfo -N -o "%5N  %5t  %13C  %8O  %8e  %7m  %G"

# 查看特定分区
sinfo -p gpu
sinfo -p compute

# 查看节点详情
scontrol show node gpu06
```

**节点状态说明**：

| 状态 | 说明 | 是否可用 |
|------|------|---------|
| `idle` | 节点空闲 | ✅ 可提交 |
| `mix` | 资源部分分配 | ✅ 可提交 |
| `alloc` | 资源完全分配 | ❌ 等待 |
| `down` | 节点下线 | ❌ 不可用 |
| `drain` | 节点故障 | ❌ 不可用 |
| `drng` | 故障但作业继续 | ⚠️ 不建议 |
| `comp` | 正在清理 | ⚠️ 等待 |

#### scir-watch - 查看 GPU 状态（推荐）

```bash
# 查看所有节点 GPU 状态
scir-watch -s

# 输出：GPU名称 | 费用 | 空闲卡数 | 所在节点

# 查看特定节点的 GPU 使用情况
scir-watch gpu06 gpustat
scir-watch gpu10 gpustat
```

---

### 任务提交

#### srun - 交互式作业（实时执行）

```bash
# 基本用法
srun <命令>

# 在计算节点运行命令
srun nvidia-smi
srun python --version

# 申请 GPU 并启动交互式 Shell
srun --gres=gpu:a100-sxm4-80gb:1 --pty bash -i

# 申请 4 卡 A100 80GB
srun --gres=gpu:a100-sxm4-80gb:4 --pty bash -i

# 指定节点（gpu10-gpu14 支持 SSH）
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:2 --pty bash -i

# 完整参数示例
srun -p compute \                    # 分区
     -N 1 \                          # 节点数
     -w gpu12 \                      # 指定节点
     --gres=gpu:a100-sxm4-80gb:4 \   # GPU 类型和数量
     --mem=128G \                    # 内存
     --cpus-per-task=16 \            # CPU 数量
     --time=2:00:00 \                # 时间限制
     --pty bash -i                   # 交互式 Shell
```

**GPU 类型说明**：
- `gpu:a100-sxm4-80gb:N` - A100 80GB（N 为数量）
- `gpu:a100-pcie-40gb:N` - A100 40GB
- `gpu:v100:N` - V100
- 查看可用类型：`sinfo -o "%G"` 或 `scir-watch -s`

#### sbatch - 批量作业（脚本提交）

```bash
# 基本提交
sbatch run.sh

# 带参数提交
sbatch --export=CONFIG=llama1b_aug,SEED=42 run.sh

# 数组作业（批量运行）
sbatch --array=0-9 run.sh         # 10 个任务
sbatch --array=0-2%1 run.sh       # 3 个任务，每次只运行 1 个

# 依赖关系
JOB1=$(sbatch --parsable train.sh)
sbatch --dependency=afterok:$JOB1 eval.sh
```

**标准 SLURM 脚本模板**：

```bash
#!/bin/bash
#SBATCH -J kava-train              # 作业名
#SBATCH -o logs/train_%j.out       # stdout 输出（%j = job ID）
#SBATCH -e logs/train_%j.err       # stderr 输出
#SBATCH -p compute                 # 分区
#SBATCH -N 1                       # 节点数
#SBATCH -n 1                       # 任务数
#SBATCH --cpus-per-task=8          # 每任务 CPU 数
#SBATCH --mem=64G                  # 内存
#SBATCH -t 48:00:00                # 时间限制（48小时）
#SBATCH --gres=gpu:a100-sxm4-80gb:1  # GPU 资源
# #SBATCH -w gpu10                 # 指定节点（可选）

# 创建日志目录
mkdir -p logs

# 加载环境
. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate kava

# 验证环境
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
nvidia-smi

# 运行程序
python train.py --config configs/llama1b_aug.yaml

echo "Job finished at $(date)"
```

#### salloc - 资源预分配

```bash
# 预分配资源
salloc --gres=gpu:a100-sxm4-80gb:4 --time=2:00:00

# 在分配的资源上运行命令
srun <命令>

# 释放资源
exit
```

---

### 任务监控

#### squeue - 查看作业队列

```bash
# 查看所有作业
squeue

# 仅查看自己的作业（推荐）
squeue --me
squeue -u $USER

# 详细输出格式
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
# 输出：JobID | 分区 | 作业名 | 状态 | 运行时间 | 节点数 | 原因

# 持续监控（每 10 秒刷新）
watch -n 10 'squeue --me'

# 查看特定作业
squeue -j <JOB_ID>

# 按状态过滤
squeue --me --state=RUNNING
squeue --me --state=PENDING
```

**作业状态**：`PD`(等待) | `R`(运行) | `CG`(完成中) | `CD`(已完成) | `F`(失败) | `CA`(取消)

#### 查看作业详情

```bash
# 详细信息
scontrol show job <JOB_ID>

# 查看输出
tail -f logs/train_<JOB_ID>.out
tail -f logs/train_<JOB_ID>.err

# 监控 GPU 使用
scir-watch gpu06 gpustat
```

---
squeue -u $USER

# 详细输出
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"

# 持续监控
watch -n 10 'squeue -u $USER'

# 查看特定任务
scontrol show job <JOB_ID>

# 查看数组任务
squeue -j <ARRAY_JOB_ID>
```

### 任务管理

```bash
# 取消任务
scancel <JOB_ID>

# 取消所有任务
scancel -u $USER

# 取消特定名称的任务
scancel -n kava-train

# 取消数组任务的特定子任务
scancel <ARRAY_JOB_ID>_<INDEX>

# 暂停任务
scontrol hold <JOB_ID>

# 恢复任务
scontrol release <JOB_ID>
```

### 历史查询

```bash
# 查看已完成任务
sacct -u $USER

# 详细信息
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

# 最近 24 小时
sacct -S $(date -d '1 day ago' +%Y-%m-%d) -u $USER

# 特定时间范围
sacct -S 2024-01-01 -E 2024-01-31 -u $USER
```

### 资源查询

```bash
# 查看分区信息
sinfo
sinfo -p gpu

# 查看节点状态
sinfo -N
scontrol show node <NODE_NAME>

# 查看配额
sacctmgr show user $USER
sacctmgr show association user=$USER
```

---

## 🔧 Module 命令

### 基本操作

```bash
# 查看可用模块
module avail

# 搜索模块
module avail cuda
module spider pytorch

# 加载模块
module load cuda/11.8.0
module load anaconda3

# 卸载模块
module unload cuda

# 切换版本
module swap cuda/11.8.0 cuda/12.1.1

# 查看已加载模块
module list

# 清除所有模块
module purge
```

### CUDA 模块

```bash
# 查看可用 CUDA 版本
module avail cuda

# 加载指定版本
module load cuda/11.8.0

# 验证
nvcc -V
nvidia-smi

# 查看模块详情
module show cuda/11.8.0
```

---

## 📊 监控和调试

### 实时监控

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控任务输出
tail -f logs/kava_*.out

# 监控错误日志
tail -f logs/kava_*.err

# 同时监控多个文件
tail -f logs/kava_{12345,12346,12347}.out
```

### 性能分析

```bash
# 查看任务资源使用
sacct -j <JOB_ID> --format=JobID,JobName,MaxRSS,MaxVMSize,Elapsed

# 查看 GPU 使用历史
ssh <NODE_NAME>  # 登录到计算节点
nvidia-smi dmon -i 0 -s puc  # 监控 GPU 0
```

### 调试任务

```bash
# 交互式会话（调试用）
srun --pty --gres=gpu:1 --mem=32G bash

# 在交互会话中测试
python train.py --config configs/llama1b_aug.yaml --quick_test

# 查看任务标准输出
cat logs/kava_<JOB_ID>.out

# 查看任务错误输出
cat logs/kava_<JOB_ID>.err

# 搜索错误
grep -i error logs/kava_*.err
grep -i "out of memory" logs/kava_*.err
```

---

## 📁 文件管理

### 数据传输

```bash
# 从本地上传到 HPC
scp -r kava/ username@hpc.example.edu:~/

# 从 HPC 下载到本地
scp -r username@hpc.example.edu:~/kava/outputs/ ./

# 使用 rsync（增量同步）
rsync -avz --progress kava/ username@hpc.example.edu:~/kava/
rsync -avz --progress username@hpc.example.edu:~/kava/outputs/ ./outputs/
```

### 磁盘配额

```bash
# 查看配额
quota -s

# 查看目录大小
du -sh outputs/
du -h --max-depth=1 outputs/

# 清理空间
# 删除旧的检查点
find outputs/ -name "checkpoint-*" -type d -mtime +30 -exec rm -rf {} +

# 压缩结果
tar -czf outputs_backup.tar.gz outputs/
```

---

## 🔄 工作流程示例

### 完整实验流程

```bash
# 1. 登录 HPC
ssh username@hpc.example.edu

# 2. 进入项目目录
cd ~/kava

# 3. 加载环境
load-kava

# 4. 提交训练任务
./hpc_run_all.sh llama1b_aug

# 5. 监控任务
watch -n 10 'squeue -u $USER'

# 6. 查看日志
tail -f logs/kava_*.out

# 7. 任务完成后查看结果
cat outputs/llama1b_aug_multi_seed/aggregated_results.yaml

# 8. 生成表格
python format_results.py --input_dir outputs/

# 9. 下载结果到本地
exit  # 退出 HPC
scp -r username@hpc.example.edu:~/kava/outputs/ ./
scp username@hpc.example.edu:~/kava/kava_tables.tex ./
```

### 批量提交多个配置

```bash
# 方法 1: 使用脚本
./hpc_run_all.sh llama1b_aug llama1b_aug_nl qwen05b_aug llama3b_aug

# 方法 2: 循环提交
for config in llama1b_aug llama1b_aug_nl qwen05b_aug llama3b_aug; do
    echo "Submitting $config"
    sbatch --export=CONFIG=$config submit_multi_seed.slurm
    sleep 1
done

# 方法 3: 任务链
JOB1=$(sbatch --parsable --export=CONFIG=llama1b_aug submit_multi_seed.slurm)
JOB2=$(sbatch --parsable --export=CONFIG=llama1b_aug_nl submit_multi_seed.slurm)
JOB3=$(sbatch --parsable --export=CONFIG=qwen05b_aug submit_multi_seed.slurm)

# 等待所有任务完成后聚合
sbatch --dependency=afterok:$JOB1:$JOB2:$JOB3 submit_aggregate_all.slurm
```

---

## ⚙️ 配置优化

### 资源请求优化

```bash
# LLaMA-1B (小模型)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# LLaMA-3B (中等模型)
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00

# Qwen-0.5B (最小模型)
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
```

### 并行策略

```bash
# 策略 1: 数据并行（单机多卡）
#SBATCH --gres=gpu:4
python -m torch.distributed.launch --nproc_per_node=4 train.py

# 策略 2: 数组作业（多个独立任务）
#SBATCH --array=0-11  # 4 configs × 3 seeds

# 策略 3: 管道并行（大模型）
#SBATCH --gres=gpu:2
# 在代码中使用 model.parallelize()
```

---

## 📞 获取帮助

```bash
# SLURM 帮助
man sbatch
man squeue
man sacct

# Module 帮助
module help
module show <MODULE>

# 联系管理员
# 查看集群公告
cat /etc/motd

# 提交支持工单
# support@hpc.example.edu
```

---

## 🌐 SSH 端口映射

### 正向映射 (Local Port Forwarding)

将远程端口映射到本地，方便访问远程服务（如 TensorBoard、Jupyter）。

#### 基本用法

```bash
# 语法
ssh -L [本地端口]:localhost:[远程端口] [远程主机]

# 示例：映射 HPC TensorBoard (6006) 到本地 22222
ssh -L 22222:localhost:6006 hpc

# 然后在本地浏览器访问
# http://localhost:22222
```

#### 后台运行（不登录）

```bash
# 使用 -N 选项仅做端口映射，不打开交互式 shell
ssh -N -L 22222:localhost:6006 hpc

# 后台运行
ssh -N -L 22222:localhost:6006 hpc &

# 或使用 nohup
nohup ssh -N -L 22222:localhost:6006 hpc > /dev/null 2>&1 &
```

#### 常见应用场景

```bash
# 1. TensorBoard
ssh -N -L 6006:localhost:6006 hpc
# 本地访问: http://localhost:6006

# 2. Jupyter Notebook (假设远程运行在 8888)
ssh -N -L 8888:localhost:8888 hpc
# 本地访问: http://localhost:8888

# 3. Jupyter 映射到不同本地端口
ssh -N -L 9999:localhost:8888 hpc
# 本地访问: http://localhost:9999

# 4. WandB Local Server
ssh -N -L 8080:localhost:8080 hpc
# 本地访问: http://localhost:8080

# 5. VS Code Server
ssh -N -L 8000:localhost:8000 hpc
```

#### 多端口映射

```bash
# 同时映射多个端口
ssh -N \
    -L 6006:localhost:6006 \
    -L 8888:localhost:8888 \
    -L 8080:localhost:8080 \
    hpc
```

#### VSCode Remote SSH 自动映射

VSCode 的 Remote SSH 插件具有**自动端口转发**功能：

1. 在远程运行服务（如 `tensorboard --logdir runs --port 6006`）
2. VSCode 自动检测并提示转发端口
3. 点击通知中的"转发端口"或在"端口"面板手动添加
4. 自动映射到本地相同端口

---

### 反向映射 (Remote Port Forwarding)

将本地端口映射到远程，让远程访问本地服务（如代理、数据库）。

#### 基本用法

```bash
# 语法
ssh -R [远程端口]:localhost:[本地端口] [远程主机]

# 示例：将本地 Clash (7890) 映射到 HPC 的 55555
ssh -R 55555:localhost:7890 hpc
```

#### 使用本地代理加速 HPC 下载

**场景**: HPC 访问 HuggingFace/GitHub 缓慢，使用本地代理加速。

##### Step 1: 配置本地代理

```bash
# Clash for Windows
1. 打开 Clash
2. 启用 "Allow LAN" 选项
3. 记住端口号（默认 7890）

# Shadowrocket (macOS)
# 默认端口 1089
```

##### Step 2: 建立反向隧道

```bash
# 映射本地 Clash (7890) 到 HPC 的 55555
ssh -R 55555:localhost:7890 hpc

# 或后台运行
ssh -N -R 55555:localhost:7890 hpc &

# 使用 Shadowrocket
ssh -N -R 55555:localhost:1089 hpc &
```

##### Step 3: 在 HPC 配置代理

```bash
# 在 HPC 终端设置代理环境变量
export http_proxy=http://localhost:55555
export https_proxy=http://localhost:55555
export all_proxy=http://localhost:55555

# 测试代理连接
curl -I https://www.google.com
curl https://huggingface.co

# 下载 Google 主页测试
wget https://www.google.com -O google.html
cat google.html

# 如果成功，可以看到 Google HTML 内容
```

##### Step 4: 永久配置（可选）

```bash
# 写入 ~/.bashrc（每次登录自动生效）
cat >> ~/.bashrc << 'EOF'
# 代理配置 (需要本地先建立 SSH 反向隧道)
# ssh -N -R 55555:localhost:7890 hpc
export http_proxy=http://localhost:55555
export https_proxy=http://localhost:55555
export all_proxy=http://localhost:55555
EOF

source ~/.bashrc
```

##### Step 5: 在 SLURM 作业中使用

```bash
# 在 submit_*.slurm 脚本中添加
export http_proxy=http://localhost:55555
export https_proxy=http://localhost:55555
export all_proxy=http://localhost:55555

# 然后正常运行
python train.py --config configs/llama1b_aug.yaml
```

#### 故障排除

**问题 1: 端口已占用**

```bash
# 症状
bind: Address already in use
channel_setup_fwd_listener_tcpip: cannot listen to port: 55555

# 解决：使用其他端口（建议 50000-65535）
ssh -N -R 56789:localhost:7890 hpc
export all_proxy=http://localhost:56789
```

**问题 2: 连接被拒绝**

```bash
# 症状
curl: (7) Failed to connect to localhost port 55555: Connection refused

# 解决：确认 SSH 隧道仍在运行
ps aux | grep "ssh -R"

# 重新建立隧道
ssh -N -R 55555:localhost:7890 hpc &
```

**问题 3: 本地代理未启用 LAN**

```bash
# 症状
channel 2: open failed: connect failed: Connection refused

# 解决：在 Clash 中启用 "Allow LAN"
1. 打开 Clash
2. General → Allow LAN → 开启
3. 重新建立 SSH 隧道
```

**问题 4: 隧道意外断开**

```bash
# 使用 autossh 自动重连（本地安装）
# Linux/macOS
autossh -M 0 -N -R 55555:localhost:7890 hpc

# Windows (PowerShell)
# 创建重连脚本 keep_tunnel.ps1
while ($true) {
    ssh -N -R 55555:localhost:7890 hpc
    Start-Sleep -Seconds 5
}
```

#### 安全建议

```bash
# 1. 使用非特权端口 (>1024)
ssh -R 55555:localhost:7890 hpc  # ✅ 推荐
ssh -R 80:localhost:7890 hpc     # ❌ 需要 root

# 2. 限制绑定地址（仅允许本地连接）
ssh -R localhost:55555:localhost:7890 hpc

# 3. 使用完毕后清理环境变量
unset http_proxy https_proxy all_proxy

# 4. 不要在公共脚本中硬编码代理
```

#### 高级用法

```bash
# 1. 动态端口转发（SOCKS5 代理）
ssh -D 1080 hpc
# 然后配置应用使用 SOCKS5: localhost:1080

# 2. 跳板机转发
ssh -J jumphost -L 6006:localhost:6006 compute-node

# 3. 多级转发
# 本地 → 跳板机 → 计算节点
ssh -L 6006:compute-node:6006 jumphost

# 4. 配置文件简化命令
# ~/.ssh/config
Host hpc-tunnel
    HostName hpc.example.edu
    User username
    LocalForward 6006 localhost:6006
    LocalForward 8888 localhost:8888

# 使用
ssh hpc-tunnel
```

---

## � 容器化部署

HPC 支持两种容器技术：**Enroot**（推荐）和 **Docker**。

### 为什么使用容器？

- ✅ **环境一致性**：避免依赖冲突
- ✅ **快速部署**：预装所有依赖
- ✅ **版本隔离**：不同项目使用不同环境
- ✅ **易于分享**：导出镜像给团队使用
- ✅ **GPU 支持**：容器内直接访问 GPU

---

## 🚀 Enroot 容器（推荐）

Enroot 是 NVIDIA 开发的轻量级容器运行时，专为 HPC 设计，与 SLURM 深度集成。

### 1. 导入 Docker 镜像

```bash
# 基本语法
enroot import docker://<IMAGE_NAME>

# 示例：导入 PyTorch 镜像
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 如果 DockerHub 访问有问题，使用国内镜像
enroot import docker://dockerpull.org/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 使用代理加速下载
export all_proxy=http://localhost:55555
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 导入完成后，会在当前目录生成 .sqsh 镜像文件
# pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh
```

**常用镜像**：
```bash
# PyTorch 官方镜像
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
enroot import docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# TensorFlow 官方镜像
enroot import docker://tensorflow/tensorflow:2.14.0-gpu

# NVIDIA CUDA 基础镜像
enroot import docker://nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 自定义镜像（从私有仓库）
enroot import docker://myregistry.com/myproject:latest
```

### 2. 创建容器

```bash
# 基本语法
enroot create --name <CONTAINER_NAME> <SQSH_PATH>

# 示例：从 .sqsh 文件创建容器
enroot create --name torch251 pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh

# 容器创建后会保存在 ~/.local/share/enroot/
```

### 3. 容器内执行命令

```bash
# 基本语法
enroot start <CONTAINER_NAME> <COMMAND>

# 示例：检查 GPU
enroot start torch251 nvidia-smi

# 示例：运行 Python 脚本
enroot start torch251 python train.py

# 示例：交互式 Shell
enroot start torch251 bash

# 示例：测试 PyTorch CUDA
enroot start torch251 python -c "import torch; print(torch.cuda.is_available())"
```

**输出示例**：
```
Thu Jan 17 10:30:45 2025       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    52W / 400W |      0MiB / 81920MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 4. 目录挂载

Enroot 默认**不挂载任何目录**，需要使用 `--mount` 参数。

```bash
# 基本语法
enroot start --mount <SRC>:<DST> <CONTAINER_NAME> <COMMAND>

# 示例：挂载公共模型库
enroot start --mount /home/share/models:/models:ro torch251 ls /models

# 示例：挂载项目目录（读写）
enroot start --mount /home/username/kava:/workspace torch251 bash

# 示例：多个挂载
enroot start \
    --mount /home/share/models:/models:ro \
    --mount /home/username/kava:/workspace \
    --mount /home/username/data:/data:ro \
    torch251 python /workspace/train.py

# :ro 表示只读，:rw 表示读写（默认）
```

**验证挂载**：
```bash
# 在容器内查看挂载的目录
enroot start --mount /home/share/models:/models:ro torch251 ls -lh /models

# 输出：
# drwxr-xr-x 15 user group 4.0K Jan 17 10:00 models--meta-llama--Llama-3.2-1B-Instruct
# drwxr-xr-x 12 user group 4.0K Jan 17 10:00 models--Qwen--Qwen2.5-0.5B-Instruct
```

### 5. 与 SLURM 集成（推荐）

Enroot 与 SLURM 深度集成，可直接在 `sbatch` 脚本中使用。

#### SLURM + Enroot 脚本示例

```bash
#!/usr/bin/bash
#SBATCH --job-name=kava-enroot
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/kava_%j.out
#SBATCH --error=logs/kava_%j.err

# ========== Enroot 容器配置 ==========
#SBATCH --container-writable                              # 容器内可写
#SBATCH --container-mount-home                            # 挂载家目录
#SBATCH --container-mounts /home/share/models:/models:ro  # 挂载公共模型库
#SBATCH --container-image torch251                        # 容器名称或 .sqsh 路径

# ========== 以下命令在容器内执行 ==========

# 检测 GPU
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "NUM_GPUS: $NUM_GPUS"

# 验证 PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 配置 HuggingFace 使用挂载的模型库
export HF_HOME=/models
export TRANSFORMERS_CACHE=/models
export HF_DATASETS_CACHE=/models

# 训练
cd /workspace  # 假设挂载了项目目录
python train.py --config configs/llama1b_aug.yaml
```

#### 提交作业

```bash
# 准备镜像（仅首次）
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
enroot create --name torch251 pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh

# 提交作业
sbatch submit_enroot.slurm

# 查看作业
squeue --me
tail -f logs/kava_*.out
```

#### 使用 .sqsh 文件路径

```bash
# 如果不想创建命名容器，可以直接使用 .sqsh 路径
#SBATCH --container-image /home/username/pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh
```

### 6. Enroot 常用命令

```bash
# 列出所有容器
enroot list

# 删除容器
enroot remove torch251

# 导出容器为 .sqsh
enroot export --output mycontainer.sqsh mycontainer

# 从本地 .sqsh 创建容器
enroot create --name newcontainer mycontainer.sqsh

# 查看容器信息
enroot inspect torch251

# 清理未使用的镜像
enroot remove --all
```

### 7. KAVA 项目容器化部署

#### Step 1: 创建自定义 Dockerfile

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install peft wandb bitsandbytes

# 复制项目代码
COPY . .

# 默认命令
CMD ["bash"]
```

#### Step 2: 本地构建并导出（可选）

```bash
# 在本地机器构建 Docker 镜像
docker build -t kava:latest .

# 导出为 tar
docker save kava:latest -o kava-latest.tar

# 上传到 HPC
scp kava-latest.tar username@hpc.example.edu:~/

# 在 HPC 上导入
enroot import docker://kava-latest.tar
enroot create --name kava kava+latest.sqsh
```

#### Step 3: 或直接在 HPC 导入基础镜像

```bash
# 导入 PyTorch 基础镜像
enroot import docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
enroot create --name kava-base pytorch+pytorch+2.1.0-cuda12.1-cudnn8-devel.sqsh

# 进入容器安装依赖
enroot start --mount $PWD:/workspace --writable kava-base bash

# 在容器内
cd /workspace
pip install -r requirements.txt
pip install peft wandb bitsandbytes
exit

# 导出修改后的容器
enroot export --output kava-ready.sqsh kava-base
enroot create --name kava kava-ready.sqsh
```

#### Step 4: 使用容器训练

```bash
# 方法 1: 直接运行
enroot start \
    --mount $PWD:/workspace \
    --mount /home/share/models:/models:ro \
    kava python /workspace/train.py --config /workspace/configs/llama1b_aug.yaml

# 方法 2: SLURM 提交
sbatch --container-image kava submit_multi_seed.slurm
```

---

## 🐋 Docker 容器（高级）

HPC 支持 **rootless Docker**（无需 root 权限），适合需要导出镜像的场景。

**推荐**: 如果不需要导出镜像，优先使用 Enroot！

### 1. 初次配置（仅需一次）

#### 方法 1: 自动配置（推荐）

```bash
# 登录到计算节点（gpu10-gpu14）
srun -w gpu10 --pty bash

# 运行配置脚本
dockerd-rootless-setuptool.sh install

# 如果成功，会输出类似：
# [INFO] Installed dockerd-rootless-setuptool.sh
# [INFO] Make sure the following environment variables are set:
#   export PATH=/usr/bin:$PATH
#   export DOCKER_HOST=unix:///tmp/$(id -u)/docker/run/docker.sock
```

#### 方法 2: 手动配置

```bash
# 1. 创建运行目录
mkdir -p /tmp/$(id -u)/docker/run

# 2. 设置环境变量
export XDG_RUNTIME_DIR=/tmp/$(id -u)/docker/run
export DOCKER_HOST=unix:///tmp/$(id -u)/docker/run/docker.sock

# 3. 写入 ~/.bashrc
cat >> ~/.bashrc << 'EOF'
# Docker rootless
export XDG_RUNTIME_DIR=/tmp/$(id -u)/docker/run
export DOCKER_HOST=unix:///tmp/$(id -u)/docker/run/docker.sock
EOF

# 4. 启动 Docker 服务
PATH=/usr/bin:/sbin:/usr/sbin:$PATH dockerd-rootless.sh &

# 等待服务启动（约 10 秒）
sleep 10
```

### 2. 配置数据目录（避免权限问题）

```bash
# 创建配置文件
mkdir -p ~/.config/docker
cat > ~/.config/docker/daemon.json << EOF
{
  "data-root": "/tmp/$(id -u)/docker-data"
}
EOF

# 创建数据目录
mkdir -p /tmp/$(id -u)/docker-data

# 重启 Docker 服务
systemctl --user restart docker
```

### 3. 验证安装

```bash
# 检查服务状态
systemctl --user status docker

# 应该看到：
# ● docker.service - Docker Application Container Engine (Rootless)
#      Loaded: loaded
#      Active: active (running)

# 测试运行容器
docker run hello-world

# 测试 GPU 容器
docker run --rm --gpus 0 pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel nvidia-smi
```

### 4. 使用 Docker

```bash
# 拉取镜像
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 列出镜像
docker images

# 运行容器（交互式）
docker run -it --rm --gpus all \
    -v $PWD:/workspace \
    -v /home/share/models:/models:ro \
    pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime bash

# 运行训练
docker run --rm --gpus all \
    -v $PWD:/workspace \
    -v /home/share/models:/models:ro \
    -e HF_HOME=/models \
    pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
    python /workspace/train.py --config /workspace/configs/llama1b_aug.yaml

# 后台运行
docker run -d --gpus all \
    -v $PWD:/workspace \
    --name kava-train \
    pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
    python /workspace/train.py --config /workspace/configs/llama1b_aug.yaml

# 查看日志
docker logs -f kava-train

# 停止容器
docker stop kava-train
```

### 5. 构建自定义镜像

```bash
# 使用之前的 Dockerfile
docker build -t kava:latest .

# 运行自定义镜像
docker run -it --rm --gpus all -v $PWD:/workspace kava:latest
```

### 6. 导出和分享镜像

```bash
# 导出为 tar
docker save kava:latest -o kava-latest.tar

# 分享给团队
scp kava-latest.tar teammate@hpc.example.edu:~/

# 导入镜像
docker load -i kava-latest.tar

# 或推送到 Docker Hub
docker tag kava:latest myusername/kava:latest
docker push myusername/kava:latest
```

### 7. 故障排除

**问题 1: 权限错误**

```bash
# 症状
Got permission denied while trying to connect to the Docker daemon socket

# 解决：检查环境变量
echo $DOCKER_HOST  # 应该输出 unix:///tmp/.../docker.sock

# 重新设置
export DOCKER_HOST=unix:///tmp/$(id -u)/docker/run/docker.sock
```

**问题 2: 服务未运行**

```bash
# 症状
Cannot connect to the Docker daemon

# 解决：启动服务
PATH=/usr/bin:/sbin:/usr/sbin:$PATH dockerd-rootless.sh &

# 或使用 systemd
systemctl --user start docker
```

**问题 3: GPU 不可用**

```bash
# 症状
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

# 解决：使用正确的 GPU 参数
docker run --gpus 0 ...        # 使用 GPU 0
docker run --gpus all ...      # 使用所有 GPU
docker run --gpus '"device=0,1"' ...  # 使用 GPU 0 和 1
```

---

## 📊 Enroot vs Docker 对比

| 特性 | Enroot | Docker |
|------|--------|--------|
| **性能** | ⭐⭐⭐⭐⭐ 更快 | ⭐⭐⭐⭐☆ 略慢 |
| **SLURM 集成** | ⭐⭐⭐⭐⭐ 原生支持 | ⭐⭐⭐☆☆ 需手动 |
| **易用性** | ⭐⭐⭐⭐☆ 简单 | ⭐⭐⭐⭐⭐ 更成熟 |
| **镜像构建** | ⭐⭐⭐☆☆ 需 Docker | ⭐⭐⭐⭐⭐ 原生 |
| **镜像分享** | ⭐⭐⭐☆☆ 需导出 | ⭐⭐⭐⭐⭐ Docker Hub |
| **HPC 优化** | ⭐⭐⭐⭐⭐ 专为 HPC 设计 | ⭐⭐⭐☆☆ 通用 |
| **推荐场景** | HPC 训练、批量作业 | 镜像开发、导出分享 |

**推荐策略**：
- ✅ **Enroot**: 日常训练、SLURM 作业、团队共享镜像
- ✅ **Docker**: 镜像开发、调试、推送到 Docker Hub

**混合使用**：
```bash
# 1. 用 Docker 构建镜像
docker build -t kava:latest .

# 2. 导出并转换为 Enroot
docker save kava:latest | enroot import docker://kava:latest -
enroot create --name kava kava+latest.sqsh

# 3. 在 SLURM 中使用 Enroot
sbatch --container-image kava submit_multi_seed.slurm
```

---

## �🔗 相关文档

- **完整指南**: `REPRODUCTION_GUIDE.md`
- **快速参考**: `QUICK_REFERENCE.md`
- **项目清单**: `PROJECT_INVENTORY.md`
- **交互式使用**: `SLURM_INTERACTIVE_GUIDE.md`
- **Enroot 官方文档**: https://github.com/NVIDIA/enroot
- **Enroot + SLURM**: https://github.com/NVIDIA/pyxis

---

## 💡 最佳实践

1. **使用数组作业**: 批量运行多个种子，自动并行
2. **设置依赖关系**: 自动化工作流程，无需手动等待
3. **定期检查日志**: 及早发现问题
4. **合理估算时间**: 避免任务被过早终止
5. **备份重要数据**: 定期下载检查点和结果
6. **使用 WandB**: 远程监控训练进度
7. **压缩存储**: 节省磁盘配额
8. **清理临时文件**: 定期删除不需要的检查点

---

**快速联系方式**
- 技术问题：查看 Issues
- HPC 支持：联系集群管理员
