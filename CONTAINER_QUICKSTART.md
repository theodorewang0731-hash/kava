# 容器化部署快速指南

**在 HPC 上使用 Enroot/Docker 容器运行 KAVA**

---

## 🎯 为什么使用容器？

| 优势 | 说明 |
|------|------|
| ✅ **环境一致性** | 避免"在我机器上能跑"问题 |
| ✅ **依赖隔离** | 不同项目使用不同 Python/CUDA 版本 |
| ✅ **快速部署** | 预装所有依赖，秒级启动 |
| ✅ **易于分享** | 导出镜像给团队，一次构建处处运行 |
| ✅ **GPU 支持** | 容器内直接访问 GPU，性能无损 |

---

## 🚀 Enroot 快速开始（推荐）

### 1 分钟快速部署

```bash
# 1. 导入 PyTorch 镜像（仅首次，约 2-5 分钟）
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 2. 创建容器（仅首次，约 10 秒）
enroot create --name kava-torch pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh

# 3. 运行训练（立即开始）
enroot start \
    --mount $PWD:/workspace \
    --mount /home/share/models:/models:ro \
    kava-torch python /workspace/train.py --config /workspace/configs/llama1b_aug.yaml
```

### SLURM 批量作业（3 个种子）

```bash
# 1. 准备容器（仅首次）
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
enroot create --name kava-torch pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh

# 2. 提交作业
sbatch --export=CONFIG=llama1b_aug submit_enroot.slurm

# 3. 监控
squeue --me
tail -f logs/kava_enroot_*.out
```

---

## 📦 推荐镜像

| 镜像 | 适用场景 | 导入命令 |
|------|---------|---------|
| **PyTorch 2.5.1 + CUDA 12.1** | KAVA 训练（推荐） | `enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` |
| **PyTorch 2.1.0 + CUDA 12.1** | 兼容性更好 | `enroot import docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` |
| **TensorFlow 2.14 + GPU** | TensorFlow 项目 | `enroot import docker://tensorflow/tensorflow:2.14.0-gpu` |
| **NVIDIA CUDA 12.1** | 自定义环境 | `enroot import docker://nvidia/cuda:12.1.0-runtime-ubuntu22.04` |

---

## 🛠️ 完整工作流程

### Step 1: 导入镜像

```bash
# 方法 1: 从 Docker Hub（国内可能较慢）
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 方法 2: 使用国内镜像（推荐）
enroot import docker://dockerpull.org/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 方法 3: 使用代理加速
export all_proxy=http://localhost:55555
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 导入后会生成 .sqsh 文件
ls -lh pytorch+pytorch+*.sqsh
```

### Step 2: 创建容器

```bash
# 从 .sqsh 创建命名容器
enroot create --name kava-torch pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh

# 验证容器
enroot list
# 输出：kava-torch
```

### Step 3: 测试容器

```bash
# 测试 GPU
enroot start kava-torch nvidia-smi

# 测试 PyTorch
enroot start kava-torch python -c "import torch; print(torch.cuda.is_available())"

# 交互式 Shell
enroot start kava-torch bash
```

### Step 4: 安装依赖（首次）

```bash
# 进入容器并挂载项目
enroot start --mount $PWD:/workspace kava-torch bash

# 在容器内
cd /workspace
pip install -r requirements.txt
pip install peft wandb bitsandbytes

# 退出
exit
```

### Step 5: 运行训练

#### 方法 A: 命令行直接运行

```bash
enroot start \
    --mount $PWD:/workspace \
    --mount /home/share/models:/models:ro \
    kava-torch python /workspace/train.py \
        --config /workspace/configs/llama1b_aug.yaml \
        --output_dir /workspace/outputs/llama1b_aug_seed_42 \
        --seed 42
```

#### 方法 B: SLURM 提交（推荐）

**编辑 submit_enroot.slurm**:
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

# Enroot 配置
#SBATCH --container-writable
#SBATCH --container-mount-home
#SBATCH --container-mounts /home/share/models:/models:ro
#SBATCH --container-image kava-torch  # 或 .sqsh 文件路径

# 配置 HuggingFace
export HF_HOME=/models
export TRANSFORMERS_CACHE=/models
export HF_DATASETS_CACHE=/models

# 运行训练
cd $HOME/kava
python train.py --config configs/llama1b_aug.yaml --use_wandb
```

**提交作业**:
```bash
sbatch submit_enroot.slurm
```

---

## 🐋 Docker 使用（可选）

### 初次配置

```bash
# 1. 登录到计算节点
srun -w gpu10 --pty bash

# 2. 配置 rootless Docker
dockerd-rootless-setuptool.sh install

# 3. 配置数据目录
mkdir -p ~/.config/docker
cat > ~/.config/docker/daemon.json << EOF
{
  "data-root": "/tmp/$(id -u)/docker-data"
}
EOF

# 4. 启动服务
systemctl --user start docker

# 5. 验证
docker run hello-world
docker run --rm --gpus 0 pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel nvidia-smi
```

### 使用 Docker 训练

```bash
# 拉取镜像
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 运行训练
docker run --rm --gpus all \
    -v $PWD:/workspace \
    -v /home/share/models:/models:ro \
    -e HF_HOME=/models \
    -e TRANSFORMERS_CACHE=/models \
    pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
    python /workspace/train.py --config /workspace/configs/llama1b_aug.yaml
```

---

## 📊 Enroot vs Docker

| 特性 | Enroot | Docker |
|------|--------|--------|
| **推荐度** | ⭐⭐⭐⭐⭐ HPC 首选 | ⭐⭐⭐⭐☆ 镜像开发 |
| **SLURM 集成** | ✅ 原生支持 | ❌ 需手动 |
| **性能** | ✅ 更快 | ✅ 良好 |
| **镜像构建** | ❌ 需 Docker | ✅ 原生 |
| **适用场景** | 日常训练 | 镜像开发 |

**推荐策略**: 用 Docker 构建镜像 → 转为 Enroot 在 HPC 使用

---

## 🔧 常见问题

### Q1: 容器内找不到模型

```bash
# 症状
FileNotFoundError: Model 'meta-llama/Llama-3.2-1B-Instruct' not found

# 解决：挂载公共模型库并设置环境变量
enroot start \
    --mount /home/share/models:/models:ro \
    kava-torch bash

# 在容器内
export HF_HOME=/models
export TRANSFORMERS_CACHE=/models
```

### Q2: 依赖未安装

```bash
# 症状
ModuleNotFoundError: No module named 'peft'

# 解决：在容器内安装
enroot start --mount $PWD:/workspace kava-torch bash
cd /workspace
pip install -r requirements.txt
pip install peft wandb bitsandbytes
```

### Q3: 容器内无法写入

```bash
# 症状
PermissionError: [Errno 13] Permission denied

# 解决：使用 --writable 或 SLURM 的 --container-writable
enroot start --writable --mount $PWD:/workspace kava-torch bash

# 或在 SLURM 脚本中
#SBATCH --container-writable
```

### Q4: GPU 不可用

```bash
# 症状
torch.cuda.is_available() = False

# 检查：
# 1. 节点是否有 GPU
nvidia-smi

# 2. SLURM 是否分配 GPU
echo $CUDA_VISIBLE_DEVICES

# 3. 容器内是否识别
enroot start kava-torch nvidia-smi
```

### Q5: 镜像太大下载慢

```bash
# 解决：使用代理
export all_proxy=http://localhost:55555
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 或使用国内镜像
enroot import docker://dockerpull.org/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 或选择更小的 runtime 镜像（不含编译工具）
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime  # 约 5GB
# 而不是 devel 镜像（约 10GB）
```

---

## 💡 最佳实践

### 1. 选择合适的镜像标签

```bash
# ✅ 推荐：runtime（更小，适合训练）
pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# ⚠️ 可选：devel（更大，包含编译工具）
pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# 选择原则：
# - 仅训练/推理 → runtime
# - 需要编译扩展 → devel
```

### 2. 挂载目录规划

```bash
# 推荐的挂载策略
enroot start \
    --mount $HOME/kava:/workspace \              # 项目代码（读写）
    --mount /home/share/models:/models:ro \      # 公共模型（只读）
    --mount /home/username/data:/data:ro \       # 数据集（只读）
    --mount /home/username/outputs:/outputs \    # 输出目录（读写）
    kava-torch bash

# :ro = 只读，:rw = 读写（默认）
```

### 3. 使用 SLURM 容器参数

```bash
# ✅ 推荐：使用 SLURM 的容器参数（更简洁）
#SBATCH --container-image kava-torch
#SBATCH --container-mount-home
#SBATCH --container-mounts /home/share/models:/models:ro
#SBATCH --container-writable

# ❌ 避免：在脚本中手动调用 enroot start（复杂）
```

### 4. 预安装依赖

```bash
# 方法 1: 构建自定义镜像（推荐）
# 编写 Dockerfile，预装所有依赖
docker build -t kava:latest .
docker save kava:latest | enroot import docker://kava:latest -

# 方法 2: 修改现有容器并导出
enroot start --writable --mount $PWD:/workspace kava-torch bash
# 在容器内安装依赖
pip install -r /workspace/requirements.txt
exit
enroot export --output kava-ready.sqsh kava-torch
enroot create --name kava-ready kava-ready.sqsh
```

### 5. 清理未使用的容器

```bash
# 列出所有容器
enroot list

# 删除不需要的容器
enroot remove old-container

# 清理磁盘空间
rm -f *.sqsh  # 删除 .sqsh 文件
```

---

## 🔗 相关文档

- **HPC 完整参考**: [HPC_REFERENCE.md](HPC_REFERENCE.md) - 容器化部署详细章节
- **复现指南**: [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)
- **SLURM 交互式**: [SLURM_INTERACTIVE_GUIDE.md](SLURM_INTERACTIVE_GUIDE.md)

**官方文档**:
- [Enroot 基本用法](https://github.com/NVIDIA/enroot)
- [Enroot + SLURM (Pyxis)](https://github.com/NVIDIA/pyxis)
- [Docker Rootless](https://docs.docker.com/engine/security/rootless/)

---

## 📞 快速命令备忘

```bash
# === Enroot ===
# 导入镜像
enroot import docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 创建容器
enroot create --name kava-torch pytorch+*.sqsh

# 运行命令
enroot start kava-torch nvidia-smi

# 挂载目录运行
enroot start --mount $PWD:/workspace kava-torch python /workspace/train.py

# 列出容器
enroot list

# 删除容器
enroot remove kava-torch

# === SLURM + Enroot ===
# 提交作业
sbatch --container-image kava-torch submit_enroot.slurm

# 查看作业
squeue --me

# 取消作业
scancel <JOB_ID>

# === Docker ===
# 启动服务
systemctl --user start docker

# 拉取镜像
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 运行容器
docker run --rm --gpus all -v $PWD:/workspace pytorch/pytorch:2.5.1 python train.py

# 列出镜像
docker images

# 列出容器
docker ps -a
```

---

**提示**: Enroot 是 HPC 的最佳选择，与 SLURM 无缝集成，性能优异！
