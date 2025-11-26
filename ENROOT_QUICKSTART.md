# Enroot 容器快速启动指南

## 🎯 概述

本指南帮助您使用 Enroot 容器在 HPC 上运行 KAVA 训练，**完全绕过 conda/venv 环境问题**。

### 为什么使用 Enroot？

- ✅ **环境隔离**：不依赖系统 Python 环境，避免版本冲突
- ✅ **开箱即用**：PyTorch 官方镜像，包含完整 CUDA 环境
- ✅ **网络加速**：使用国内镜像源（dockerpull.org）
- ✅ **资源优化**：时间设置为 4 小时，避免余额不足问题

---

## 📋 4 步快速启动

### 步骤 1：导入容器镜像（登录节点）

在 HPC 登录节点运行以下命令：

```bash
# 进入项目目录
cd ~/kava  # 或您的项目路径

# 运行自动设置脚本
bash setup_enroot_container.sh
```

**脚本功能：**
- 自动导入 PyTorch 2.5.1 + CUDA 12.1 镜像
- 验证镜像文件完整性
- 检查共享模型目录
- 提供后续操作指引

**预期输出：**
```
✓ Enroot 容器环境设置完成！
镜像文件: pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh
大小: ~4-6 GB
```

---

### 步骤 2：提交训练任务

镜像导入成功后，提交训练作业：

```bash
# 提交 Llama-3.2-1B 训练任务
sbatch --export=CONFIG=llama1b_aug submit_enroot.slurm
```

**可用配置：**
- `llama1b_aug` - Llama-3.2-1B（推荐首次测试）
- `llama3b_aug` - Llama-3.2-3B
- `phi3_aug` - Phi-3.5-mini
- `qwen2_aug` - Qwen2.5-1.5B

**任务特性：**
- 并行运行 3 个随机种子（42, 123, 456）
- 每个任务限时 4 小时
- 自动挂载共享模型库（/home/share/models）
- 容器启动时自动安装依赖

---

### 步骤 3：监控任务状态

#### 查看任务队列

```bash
squeue -u $USER
```

**状态说明：**
- `PD` (Pending) - 等待资源分配
- `R` (Running) - 正在运行
- `CG` (Completing) - 即将完成

#### 查看实时日志

```bash
# 找到最新的日志文件
ls -lt logs/kava_enroot_*.out | head -3

# 查看日志（替换为实际文件名）
tail -f logs/kava_enroot_<JOB_ID>_<ARRAY_ID>.out
```

**关键日志阶段：**
1. 容器启动和环境验证
2. 依赖安装（首次运行需要几分钟）
3. 训练进度（loss、learning rate、步数）
4. 评估结果（GSM8K、GSM8K-Hard、SVAMP）

---

### 步骤 4：检查结果

训练完成后，查看输出：

```bash
# 查看目录结构
tree outputs/llama1b_aug_multi_seed -L 2

# 查看特定种子的结果
cat outputs/llama1b_aug_multi_seed/seed_42/results_gsm8k.yaml
```

**预期输出结构：**
```
outputs/llama1b_aug_multi_seed/
├── seed_42/
│   ├── best_checkpoint/
│   ├── results_gsm8k.yaml
│   ├── results_gsm8k-hard.yaml
│   └── results_svamp.yaml
├── seed_123/
└── seed_456/
```

---

## 🔧 常见问题

### Q1: 镜像导入失败怎么办？

**可能原因：**
- 网络连接不稳定
- enroot 未加载

**解决方案：**
```bash
# 加载 enroot 模块
module load enroot

# 手动导入镜像
enroot import "docker://dockerpull.org/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
```

---

### Q2: 任务一直处于 Pending 状态？

**可能原因：**
- 资源繁忙
- 账户余额不足

**检查方法：**
```bash
# 查看详细状态
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# 查看账户余额
sacctmgr show assoc user=$USER format=user,account,qos
```

**解决方案：**
- 等待资源释放
- 联系管理员充值或调整 QoS

---

### Q3: 训练中断或失败？

**查看日志：**
```bash
# 查看标准输出
cat logs/kava_enroot_<JOB_ID>_<ARRAY_ID>.out

# 查看错误输出
cat logs/kava_enroot_<JOB_ID>_<ARRAY_ID>.err
```

**常见错误：**

#### 错误 1：CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**解决：** 减少 batch size 或使用更小的模型

#### 错误 2：模型未找到
```
OSError: [Errno 2] No such file or directory: '/models/...'
```
**解决：** 检查 `/home/share/models` 是否可访问

#### 错误 3：依赖安装失败
```
ERROR: Could not install packages due to an EnvironmentError
```
**解决：** 检查网络连接，或在容器配置中使用镜像源

---

### Q4: 如何取消正在运行的任务？

```bash
# 取消特定任务
scancel <JOB_ID>

# 取消用户的所有任务
scancel -u $USER

# 取消特定任务的某个 array 任务
scancel <JOB_ID>_<ARRAY_ID>
```

---

## 📊 性能优化建议

### 1. 资源配置调整

如果需要调整资源，编辑 `submit_enroot.slurm`：

```bash
#SBATCH --gres=gpu:a100-sxm4-80gb:1  # GPU 类型和数量
#SBATCH --cpus-per-task=8            # CPU 核心数
#SBATCH --mem=64G                     # 内存
#SBATCH --time=04:00:00              # 时间限制
```

### 2. 批量测试多个模型

创建批量提交脚本：

```bash
#!/bin/bash
# batch_submit.sh

CONFIGS=("llama1b_aug" "llama3b_aug" "phi3_aug" "qwen2_aug")

for config in "${CONFIGS[@]}"; do
    echo "Submitting $config..."
    sbatch --export=CONFIG=$config submit_enroot.slurm
    sleep 2  # 避免同时提交
done
```

运行：
```bash
bash batch_submit.sh
```

### 3. 使用 Weights & Biases 追踪

如果需要实验追踪，在训练命令中保留 `--use_wandb`：

```bash
# 在容器内设置 wandb
export WANDB_API_KEY="your_api_key"
```

---

## 🔍 调试技巧

### 交互式容器测试

如果需要调试，可以启动交互式容器会话：

```bash
# 请求交互式节点
srun --partition=compute \
     --gres=gpu:a100-sxm4-80gb:1 \
     --cpus-per-task=8 \
     --mem=64G \
     --time=01:00:00 \
     --pty bash

# 在交互节点上启动容器
enroot start \
    --mount /home/share/models:/models:ro \
    --mount $HOME:$HOME \
    pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh

# 在容器内测试
python -c "import torch; print(torch.cuda.is_available())"
```

### 查看容器内环境

```bash
# 在提交的任务日志中查看这些信息
grep "PyTorch Info:" logs/kava_enroot_*.out -A 5
grep "GPU Info:" logs/kava_enroot_*.out -A 5
```

---

## 📝 文件清单

本方案涉及的关键文件：

| 文件 | 用途 |
|------|------|
| `setup_enroot_container.sh` | 自动导入镜像脚本 |
| `submit_enroot.slurm` | Slurm 任务提交脚本 |
| `pytorch+...sqsh` | 容器镜像文件（导入后生成） |
| `logs/kava_enroot_*.out` | 任务输出日志 |
| `outputs/*/seed_*/` | 训练结果目录 |

---

## 🆘 获取帮助

如果遇到问题：

1. **查看日志**：`cat logs/kava_enroot_*.err`
2. **检查队列**：`squeue -u $USER`
3. **查看资源**：`sinfo -p compute`
4. **联系管理员**：提供 Job ID 和错误日志

---

## ✅ 成功标志

当看到以下输出时，表示训练成功：

```
✅ Training completed successfully
✅ gsm8k evaluation completed
✅ gsm8k-hard evaluation completed
✅ svamp evaluation completed
========================================
Job completed at <timestamp>
Results saved to: outputs/llama1b_aug_multi_seed/seed_42
========================================
```

---

**祝训练顺利！🚀**
