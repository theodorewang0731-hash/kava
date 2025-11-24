# SLURM 交互式使用指南

**在 HPC 集群上使用 srun 进行交互式开发和调试**

---

## 🎯 交互式 vs 批处理

| 特性 | srun（交互式） | sbatch（批处理） |
|------|---------------|-----------------|
| **使用场景** | 调试、测试、开发 | 正式训练、批量实验 |
| **执行方式** | 阻塞终端，实时输出 | 后台运行，输出到文件 |
| **资源占用** | 立即分配 | 排队等待 |
| **适合时长** | < 2 小时 | 数小时至数天 |
| **灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐☆☆ |
| **稳定性** | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ |

---

## 🚀 快速开始

### 1. 申请单卡 GPU 节点

```bash
# 最简单的方式
srun --gres=gpu:a100-sxm4-80gb:1 --pty bash -i

# 完整参数
srun -p compute \
     --gres=gpu:a100-sxm4-80gb:1 \
     --mem=32G \
     --cpus-per-task=8 \
     --time=2:00:00 \
     --pty bash -i
```

### 2. 在计算节点上工作

```bash
# 成功后会看到提示符变化
# [username@login-node]$ → [username@gpu06]$

# 2. 加载环境
conda activate kava

# 配置 HPC 公共模型库（可选，如果未写入 ~/.bashrc）
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models

# 验证 GPU
nvidia-smi

# 验证模型可用
ls /home/share/models/models--meta-llama--Llama-3.2-1B-Instruct

# 运行程序
python train.py --config configs/llama1b_aug.yaml

# 完成后退出
exit
```

---

## 📋 常用交互式命令

### 申请不同规格的资源

```bash
# 单卡调试
srun --gres=gpu:a100-sxm4-80gb:1 --mem=32G --time=1:00:00 --pty bash -i

# 双卡测试
srun --gres=gpu:a100-sxm4-80gb:2 --mem=64G --time=2:00:00 --pty bash -i

# 4 卡训练
srun --gres=gpu:a100-sxm4-80gb:4 --mem=128G --time=4:00:00 --pty bash -i

# 指定节点（gpu10-gpu14 支持 SSH）
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:2 --pty bash -i
srun -w gpu12 --gres=gpu:a100-sxm4-80gb:4 --pty bash -i
```

### 快速测试命令

```bash
# 不需要交互式 Shell，直接运行命令
srun --gres=gpu:a100-sxm4-80gb:1 nvidia-smi
srun --gres=gpu:a100-sxm4-80gb:1 python --version
srun --gres=gpu:a100-sxm4-80gb:1 python smoke_test.py
```

---

## 🔧 使用 tmux 多窗口操作

由于大部分节点禁用 SSH，如需多个终端，在 srun 后启动 tmux：

```bash
# 1. 申请资源并启动 tmux
srun -w gpu12 --gres=gpu:a100-sxm4-80gb:2 --time=4:00:00 --pty bash -c "tmux new-session -s kava"

# 2. 在 tmux 中操作
# Ctrl+B, C     - 创建新窗口
# Ctrl+B, N     - 切换到下一个窗口
# Ctrl+B, P     - 切换到上一个窗口
# Ctrl+B, %     - 垂直分屏
# Ctrl+B, "     - 水平分屏
# Ctrl+B, D     - 退出 tmux（保持运行）

# 3. 重新连接（同一节点）
tmux attach -t kava
```

**tmux 快速参考**：

```bash
# 创建新会话
tmux new -s kava

# 列出会话
tmux ls

# 连接到会话
tmux attach -t kava

# 杀死会话
tmux kill-session -t kava

# 在窗口中
Ctrl+B, C      # 新窗口
Ctrl+B, N      # 下一个窗口
Ctrl+B, 0-9    # 切换到窗口 N
Ctrl+B, %      # 垂直分屏
Ctrl+B, "      # 水平分屏
Ctrl+B, 方向键  # 切换面板
Ctrl+B, D      # 退出（保持运行）
Ctrl+D         # 关闭当前面板/窗口
```

---

## 🎓 典型工作流程

### 场景 1: 快速调试代码

```bash
# 1. 申请 1 卡，短时间
srun --gres=gpu:a100-sxm4-80gb:1 --time=30:00 --pty bash -i

# 2. 加载环境
conda activate kava

# 3. 快速测试
python smoke_test.py
python train.py --config configs/llama1b_aug.yaml --quick_test

# 4. 退出
exit
```

### 场景 2: 交互式训练（小数据集）

```bash
# 1. 申请 2 卡，2 小时
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:2 --time=2:00:00 --pty bash -i

# 2. 启动 tmux（可选）
tmux new -s kava

# 3. 加载环境
conda activate kava

# 4. 运行训练（可以实时看输出）
python train.py --config configs/llama1b_aug.yaml --epochs 1

# 5. 新窗口监控（Ctrl+B, C）
watch -n 1 nvidia-smi

# 6. 完成后退出 tmux（Ctrl+B, D）
exit
```

### 场景 3: 测试多GPU训练

```bash
# 1. 申请 4 卡
srun --gres=gpu:a100-sxm4-80gb:4 --mem=128G --time=1:00:00 --pty bash -i

# 2. 测试数据并行
conda activate kava
python -m torch.distributed.launch --nproc_per_node=4 train.py

# 3. 验证所有 GPU 都在使用
nvidia-smi

# 4. 退出
exit
```

### 场景 4: Jupyter Notebook 开发

**⚠️ 注意**: 由于大部分节点禁用 SSH，需要使用支持 SSH 的节点（gpu10-gpu14）。

#### 方法 1: 手动端口映射（传统方法）

```bash
# 1. 申请支持 SSH 的节点
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:1 --time=4:00:00 --pty bash -i

# 2. 加载环境并启动 Jupyter
conda activate kava
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# 3. 记录 token 和节点名称
# 输出示例: 
#   http://gpu10:8888/?token=abc123def456...
# 记住: 节点=gpu10, 端口=8888, token=abc123def456

# 4. 在本地新终端建立 SSH 隧道
# Windows PowerShell:
ssh -L 8888:gpu10:8888 username@hpc.example.edu

# Linux/macOS:
ssh -L 8888:gpu10:8888 username@hpc.example.edu

# 5. 在本地浏览器打开
# http://localhost:8888/?token=abc123def456

# 6. 完成后停止 Jupyter
# 在 HPC 终端按 Ctrl+C 两次
```

#### 方法 2: VSCode Remote SSH（推荐）

VSCode 自动处理端口转发，更简单！

```bash
# 1. 在 VSCode 中通过 Remote SSH 连接到 HPC

# 2. 在 VSCode 终端申请资源
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:1 --time=4:00:00 --pty bash -i

# 3. 启动 Jupyter
conda activate kava
jupyter notebook --no-browser --port=8888

# 4. VSCode 会自动检测并提示 "Forward Port 8888"
#    点击通知或在"端口"面板手动添加

# 5. 直接在本地浏览器打开
#    http://localhost:8888/?token=abc123def456

# VSCode 自动处理所有端口转发！
```

#### 方法 3: JupyterLab（更强大）

```bash
# 1. 安装 JupyterLab
pip install jupyterlab

# 2. 申请资源并启动
srun -w gpu12 --gres=gpu:a100-sxm4-80gb:2 --time=4:00:00 --pty bash -i
conda activate kava
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# 3. 端口映射（同上）
ssh -L 8888:gpu12:8888 username@hpc.example.edu

# 4. 本地访问
# http://localhost:8888/lab?token=abc123def456
```

#### 常见问题

**端口冲突**:
```bash
# 如果 8888 被占用，使用其他端口
jupyter notebook --no-browser --port=9999 --ip=0.0.0.0

# 本地映射时也要改端口
ssh -L 9999:gpu10:9999 username@hpc.example.edu
```

**节点 SSH 限制**:
```bash
# ❌ 错误：gpu06 不支持 SSH
srun -w gpu06 --gres=gpu:a100-sxm4-80gb:1 --pty bash -i
# 无法建立隧道：ssh: connect to host gpu06 port 22: Connection refused

# ✅ 正确：使用 gpu10-gpu14
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:1 --pty bash -i
srun -w gpu12 --gres=gpu:a100-sxm4-80gb:1 --pty bash -i
```

**后台运行**:
```bash
# 1. 使用 nohup 后台运行
nohup jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 > jupyter.log 2>&1 &

# 2. 查看 token
cat jupyter.log | grep token

# 3. 停止 Jupyter
pkill -f jupyter
```

### 场景 5: TensorBoard 监控

```bash
# 1. 申请资源（支持 SSH 的节点）
srun -w gpu11 --gres=gpu:a100-sxm4-80gb:1 --time=2:00:00 --pty bash -i

# 2. 启动 TensorBoard
conda activate kava
tensorboard --logdir outputs/llama1b_aug_seed_42/logs --port 6006 --bind_all

# 3. 在本地建立隧道
# 新终端运行
ssh -N -L 6006:gpu11:6006 username@hpc.example.edu

# 4. 本地浏览器访问
# http://localhost:6006

# 5. 或使用 VSCode 自动端口转发
# VSCode 会自动检测 6006 端口并提示转发
```

---

## ⚠️ 注意事项

### 1. 资源使用

```bash
# ❌ 错误：长时间占用资源不使用
srun --gres=gpu:a100-sxm4-80gb:4 --time=24:00:00 --pty bash -i
# 然后 sleep 或不操作

# ✅ 正确：用完立即退出
srun --gres=gpu:a100-sxm4-80gb:1 --time=1:00:00 --pty bash -i
# 运行程序
python train.py
# 完成后立即退出
exit
```

**重要**：
- ⚠️ 程序结束后尽快 `exit` 释放资源
- ⚠️ 不要使用 `sleep` 抢占资源
- ⚠️ 大部分节点禁用 SSH（gpu10-gpu14 除外）

### 2. 时间限制

```bash
# 根据实际需要设置时间
--time=0:30:00   # 30 分钟（调试）
--time=2:00:00   # 2 小时（测试）
--time=4:00:00   # 4 小时（小规模训练）

# 超时任务会被强制终止
# 建议留 10% 余量
```

### 3. 节点选择

```bash
# 如果需要 SSH 连接（多终端），必须指定 gpu10-gpu14
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:1 --pty bash -i
srun -w gpu11 --gres=gpu:a100-sxm4-80gb:1 --pty bash -i

# 其他节点禁用 SSH
srun -w gpu06 --gres=gpu:a100-sxm4-80gb:1 --pty bash -i  # ❌ 无法 SSH
```

---

## 🔍 监控和调试

### 实时监控 GPU

```bash
# 在交互式会话中
watch -n 1 nvidia-smi

# 查看特定 GPU
nvidia-smi -i 0

# 持续监控
nvidia-smi dmon -i 0 -s puc
```

### 检查资源使用

```bash
# 查看当前作业信息
squeue --me

# 查看节点状态
sinfo -N -o "%5N  %5t  %13C  %8O  %8e  %7m  %G"

# 查看 GPU 状态
scir-watch -s
scir-watch gpu10 gpustat
```

### 调试失败的作业

```bash
# 查看标准输出
tail -f logs/kava_*.out

# 查看错误输出
tail -f logs/kava_*.err

# 搜索错误
grep -i error logs/kava_*.err
grep -i "out of memory" logs/kava_*.err
```

---

## 💡 最佳实践

1. **开发阶段**：
   - ✅ 使用 `srun` 交互式调试
   - ✅ 申请少量资源（1 卡，短时间）
   - ✅ 快速迭代，频繁测试

2. **训练阶段**：
   - ✅ 切换到 `sbatch` 批处理
   - ✅ 使用数组作业运行多个种子
   - ✅ 设置合理的时间限制

3. **资源管理**：
   - ✅ 用完立即释放
   - ✅ 使用 tmux 管理多窗口
   - ✅ 监控 GPU 使用率

4. **错误处理**：
   - ✅ 先在小数据集测试
   - ✅ 保存检查点，支持恢复
   - ✅ 查看日志诊断问题

---

## 🔗 相关文档

- **HPC 快速参考**: `HPC_REFERENCE.md`
- **完整复现指南**: `REPRODUCTION_GUIDE.md`
- **SLURM 脚本示例**: `submit_multi_seed.slurm`

---

## 📞 获取帮助

```bash
# SLURM 帮助
man srun
man sbatch
man squeue

# 查看分区信息
sinfo

# 查看可用 GPU
scir-watch -s

# 联系管理员
# 查看集群公告: cat /etc/motd
```

---

**快速命令备忘**

```bash
# 申请资源
srun --gres=gpu:a100-sxm4-80gb:1 --pty bash -i

# 查看队列
squeue --me

# 取消作业
scancel <JOB_ID>

# 查看 GPU
scir-watch -s

# 监控节点
scir-watch gpu10 gpustat
```
