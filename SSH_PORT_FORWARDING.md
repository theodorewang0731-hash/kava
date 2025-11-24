# SSH 端口映射快速参考

**HPC 集群远程服务访问指南**

---

## 📖 概述

SSH 端口映射允许你在本地访问 HPC 远程服务（TensorBoard、Jupyter）或让 HPC 使用本地代理加速下载。

| 映射类型 | 用途 | 命令格式 |
|---------|------|---------|
| **正向映射** (`-L`) | 本地访问远程服务 | `ssh -L 本地端口:远程主机:远程端口 HPC` |
| **反向映射** (`-R`) | 远程访问本地服务 | `ssh -R 远程端口:localhost:本地端口 HPC` |

---

## 🎯 正向映射 (Local → Remote)

### 常用场景速查

```bash
# TensorBoard (6006)
ssh -N -L 6006:gpu10:6006 username@hpc.example.edu
# 访问: http://localhost:6006

# Jupyter Notebook (8888)
ssh -N -L 8888:gpu12:8888 username@hpc.example.edu
# 访问: http://localhost:8888/?token=...

# 自定义本地端口
ssh -N -L 22222:gpu10:6006 username@hpc.example.edu
# 访问: http://localhost:22222

# 多端口同时映射
ssh -N \
    -L 6006:gpu10:6006 \
    -L 8888:gpu12:8888 \
    username@hpc.example.edu
```

### TensorBoard 完整流程

```bash
# === HPC 终端 ===
# 1. 申请资源（gpu10-gpu14 支持 SSH）
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:1 --time=2:00:00 --pty bash -i

# 2. 启动 TensorBoard
conda activate kava
tensorboard --logdir outputs/llama1b_aug_seed_42/logs --port 6006 --bind_all

# === 本地终端 ===
# 3. 建立隧道（新终端）
ssh -N -L 6006:gpu10:6006 username@hpc.example.edu

# === 本地浏览器 ===
# 4. 访问 TensorBoard
# http://localhost:6006
```

### Jupyter Notebook 完整流程

```bash
# === HPC 终端 ===
# 1. 申请支持 SSH 的节点
srun -w gpu12 --gres=gpu:a100-sxm4-80gb:1 --time=4:00:00 --pty bash -i

# 2. 启动 Jupyter
conda activate kava
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# 3. 记录输出的 token
# http://gpu12:8888/?token=abc123def456...

# === 本地终端 ===
# 4. 建立隧道（新终端）
ssh -L 8888:gpu12:8888 username@hpc.example.edu

# === 本地浏览器 ===
# 5. 访问 Jupyter（使用记录的 token）
# http://localhost:8888/?token=abc123def456
```

### VSCode 自动端口转发（推荐！）

```bash
# 1. VSCode 安装 Remote SSH 扩展
# 2. 连接到 HPC
# 3. 在远程终端启动服务（TensorBoard/Jupyter）

# VSCode 会自动检测端口并提示 "Forward Port"
# 点击通知 → 自动在本地浏览器打开
# 无需手动 ssh -L 命令！
```

---

## 🔄 反向映射 (Remote → Local)

### 常用场景：本地代理加速 HPC 下载

```bash
# === 本地（开启 Clash/代理）===
# 1. Clash: 启用 "Allow LAN"，端口 7890
# 2. Shadowrocket: 端口 1089

# === 本地终端 ===
# 3. 建立反向隧道（将本地 7890 映射到 HPC 的 55555）
ssh -N -R 55555:localhost:7890 username@hpc.example.edu

# 后台运行
ssh -N -R 55555:localhost:7890 username@hpc.example.edu &

# === HPC 终端 ===
# 4. 配置代理
export http_proxy=http://localhost:55555
export https_proxy=http://localhost:55555
export all_proxy=http://localhost:55555

# 5. 测试
curl -I https://www.google.com

# 6. 加速下载
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
git clone https://github.com/user/repo.git
pip install transformers
```

### 在 SLURM 脚本中使用代理

```bash
#!/bin/bash
#SBATCH --job-name=kava-train
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=48:00:00

# 配置代理（需要先在本地建立反向隧道）
export http_proxy=http://localhost:55555
export https_proxy=http://localhost:55555
export all_proxy=http://localhost:55555

# 激活环境
conda activate kava

# 训练（通过代理下载模型）
python train.py --config configs/llama1b_aug.yaml
```

---

## 🛠️ 常用命令模板

### 仅端口映射（不登录）

```bash
# -N 选项：不执行远程命令，仅转发端口
ssh -N -L 6006:gpu10:6006 username@hpc.example.edu

# 后台运行（&）
ssh -N -L 6006:gpu10:6006 username@hpc.example.edu &

# 查看后台任务
jobs
ps aux | grep "ssh -N"

# 停止后台任务
fg  # 拉到前台
Ctrl+C  # 停止

# 或直接 kill
kill %1  # 停止 job 1
```

### 多端口映射

```bash
# 方法 1: 多个 -L 选项
ssh -N \
    -L 6006:gpu10:6006 \
    -L 8888:gpu12:8888 \
    -L 8080:gpu10:8080 \
    username@hpc.example.edu

# 方法 2: 多个命令
ssh -N -L 6006:gpu10:6006 username@hpc.example.edu &
ssh -N -L 8888:gpu12:8888 username@hpc.example.edu &
```

### SSH 配置文件简化

```bash
# ~/.ssh/config
Host hpc
    HostName hpc.example.edu
    User username
    
Host hpc-tunnel
    HostName hpc.example.edu
    User username
    LocalForward 6006 gpu10:6006
    LocalForward 8888 gpu12:8888

# 使用
ssh hpc-tunnel  # 自动建立所有映射
```

---

## ⚠️ 常见问题

### 问题 1: 端口已被占用

```bash
# 症状
bind: Address already in use

# 解决：使用其他端口
ssh -N -L 7777:gpu10:6006 username@hpc.example.edu
# 访问: http://localhost:7777

# 或关闭占用端口的程序
lsof -ti:6006 | xargs kill -9  # Linux/macOS
netstat -ano | findstr :6006   # Windows (查找 PID)
taskkill /PID <PID> /F         # Windows (关闭进程)
```

### 问题 2: 远程端口冲突（反向映射）

```bash
# 症状
channel_setup_fwd_listener_tcpip: cannot listen to port: 55555

# 解决：使用其他端口（建议 50000-65535）
ssh -N -R 56789:localhost:7890 username@hpc.example.edu
export all_proxy=http://localhost:56789
```

### 问题 3: 无法连接到远程节点

```bash
# 症状
ssh: connect to host gpu06 port 22: Connection refused

# 原因：大部分节点禁用 SSH
# 解决：使用 gpu10-gpu14
srun -w gpu10 --gres=gpu:a100-sxm4-80gb:1 --pty bash -i
srun -w gpu11 --gres=gpu:a100-sxm4-80gb:1 --pty bash -i
```

### 问题 4: 代理连接失败

```bash
# 症状
curl: (7) Failed to connect to localhost port 55555

# 检查清单：
# 1. SSH 反向隧道是否运行？
ps aux | grep "ssh -R"

# 2. 本地代理是否启用 "Allow LAN"？
# Clash → General → Allow LAN → 开启

# 3. 重新建立隧道
ssh -N -R 55555:localhost:7890 username@hpc.example.edu &

# 4. 验证代理
curl -x http://localhost:55555 https://www.google.com
```

### 问题 5: 隧道自动断开

```bash
# 使用 autossh 自动重连（本地安装）
# Linux/macOS
brew install autossh  # 或 apt install autossh
autossh -M 0 -N -L 6006:gpu10:6006 username@hpc.example.edu

# Windows PowerShell 重连脚本
# keep_tunnel.ps1
while ($true) {
    Write-Host "Establishing SSH tunnel..."
    ssh -N -L 6006:gpu10:6006 username@hpc.example.edu
    Write-Host "Connection lost, reconnecting in 5s..."
    Start-Sleep -Seconds 5
}

# 运行
powershell -ExecutionPolicy Bypass -File keep_tunnel.ps1
```

---

## 📊 端口分配建议

| 服务 | 默认端口 | 建议本地端口 | 命令 |
|------|---------|-------------|------|
| TensorBoard | 6006 | 6006 或 22222 | `ssh -N -L 6006:gpu10:6006 hpc` |
| Jupyter | 8888 | 8888 或 9999 | `ssh -N -L 8888:gpu12:8888 hpc` |
| JupyterLab | 8888 | 8889 | `ssh -N -L 8889:gpu10:8888 hpc` |
| WandB Local | 8080 | 8080 | `ssh -N -L 8080:gpu10:8080 hpc` |
| VS Code Server | 8000 | 8000 | `ssh -N -L 8000:gpu10:8000 hpc` |
| Clash (反向) | 55555 | 7890 (本地) | `ssh -N -R 55555:localhost:7890 hpc` |

**避免使用的端口**：
- `< 1024`: 需要 root 权限
- `22`: SSH 服务
- `80/443`: HTTP/HTTPS
- `3306`: MySQL
- `5432`: PostgreSQL

**推荐端口范围**：
- 正向映射本地: `6000-9999`
- 反向映射远程: `50000-65535`

---

## 💡 最佳实践

### 1. 使用 VSCode Remote SSH

✅ **推荐**: VSCode 自动处理端口转发，无需手动命令

```bash
# 步骤：
1. 安装 Remote SSH 扩展
2. 连接到 HPC
3. 启动远程服务（Jupyter/TensorBoard）
4. VSCode 自动检测并提示转发
5. 点击通知，自动打开浏览器
```

### 2. 后台运行长时间映射

```bash
# 使用 nohup 防止意外关闭
nohup ssh -N -L 6006:gpu10:6006 username@hpc.example.edu > /dev/null 2>&1 &

# 记录 PID
echo $! > tunnel.pid

# 停止时
kill $(cat tunnel.pid)
```

### 3. 配置文件管理

```bash
# ~/.ssh/config
Host hpc
    HostName hpc.example.edu
    User username
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host hpc-tb
    HostName hpc.example.edu
    User username
    LocalForward 6006 gpu10:6006

Host hpc-jupyter
    HostName hpc.example.edu
    User username
    LocalForward 8888 gpu12:8888

# 使用
ssh hpc-tb       # 自动映射 TensorBoard
ssh hpc-jupyter  # 自动映射 Jupyter
```

### 4. 安全建议

```bash
# ✅ 推荐
ssh -N -L 6006:localhost:6006 hpc           # 仅本地访问
ssh -N -R localhost:55555:localhost:7890 hpc # 限制绑定地址

# ❌ 避免
ssh -N -L 0.0.0.0:6006:localhost:6006 hpc   # 允许所有 IP 访问（安全风险）
```

---

## 🔗 相关文档

- **HPC 完整指南**: [HPC_REFERENCE.md](HPC_REFERENCE.md)
- **交互式开发**: [SLURM_INTERACTIVE_GUIDE.md](SLURM_INTERACTIVE_GUIDE.md)
- **复现指南**: [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)

---

## 📞 快速帮助

```bash
# SSH 隧道帮助
man ssh
ssh -h

# 查看当前 SSH 连接
ss -tnp | grep ssh      # Linux
netstat -an | grep 22   # Windows

# 测试端口连通性
nc -zv localhost 6006           # Linux/macOS
Test-NetConnection localhost -Port 6006  # Windows PowerShell

# 查看端口占用
lsof -i:6006                    # Linux/macOS
netstat -ano | findstr :6006    # Windows
```

---

**提示**: 如果遇到问题，优先尝试 VSCode Remote SSH 的自动端口转发功能！
