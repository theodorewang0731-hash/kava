# HPC 共享环境安全审查报告

## 🔒 审查日期
2025-11-24

## 📍 HPC 环境信息
- **HPC 地址**: `10.160.22.46:2223`
- **用户名**: `rpwang`
- **项目路径**: `/home/rpwang/kava review` ⚠️ 路径包含空格
- **连接方式**: `ssh rpwang@10.160.22.46 -p 2223`
- **SFTP**: `sftp://rpwang@10.160.22.46:2223/home/rpwang/kava%20review`

## ✅ 安全性评估结果

### 1. **高危操作检查**

#### ❌ 未发现的危险操作：
- ✓ 无 `sudo` 命令
- ✓ 无系统重启 (`reboot`/`shutdown`)
- ✓ 无全局进程终止 (`killall`/`pkill -9`)
- ✓ 无系统目录修改 (`/etc`, `/usr`, `/opt`, `/var`)
- ✓ 无 `chmod 777` 权限滥用

#### ⚠️ 需要注意的操作：
1. **临时文件使用** (`/tmp/download_models_${USER}.py`)
   - ✅ 安全：使用 `${USER}` 变量隔离不同用户
   - ✅ 清理：脚本结束后自动删除
   
2. **用户缓存清理** (`rm -rf ~/.cache/...`)
   - ✅ 安全：仅限用户自己的 HOME 目录
   - ⚠️ 提示：仅在文档中作为故障排除建议，不会自动执行

### 2. **文件系统安全**

#### ✅ 所有操作限制在用户目录：
```bash
/home/rpwang/.cache/huggingface         # HuggingFace 缓存
/home/rpwang/kava review                # 项目目录（注意：路径包含空格）
/home/rpwang/kava review/outputs        # 输出目录
/home/rpwang/kava review/logs           # 日志目录
/home/rpwang/kava review/venv_kava      # 虚拟环境
```

**⚠️ 重要提示：路径包含空格**
- 项目路径：`/home/rpwang/kava review`（包含空格）
- 所有脚本已针对空格路径进行防护
- 建议重命名为 `/home/rpwang/kava_review` 避免潜在问题

#### ✅ 无跨用户影响：
- 不修改其他用户文件
- 不访问共享系统目录
- 不修改全局环境变量（仅在当前会话）

### 3. **资源使用安全**

#### ✅ SLURM 资源限制：
```bash
#SBATCH --nodes=1           # 单节点
#SBATCH --ntasks=1          # 单任务
#SBATCH --cpus-per-task=8   # 8 CPU核心
#SBATCH --mem=64G           # 64GB内存
#SBATCH --gres=gpu:1        # 1个GPU
#SBATCH --time=48:00:00     # 最长48小时
```

#### ✅ 磁盘使用检查：
- 脚本在开始前检查可用空间（需要 20GB）
- 超出配额会提前退出，不会耗尽磁盘

#### ✅ 进程隔离：
- 所有训练任务通过 SLURM 提交
- 使用用户独立的工作目录
- GPU 资源由 SLURM 自动分配和隔离

### 4. **网络安全**

#### ✅ 仅下载操作：
- 从 HuggingFace 下载模型（公开资源）
- 无上传用户数据
- 无监听端口（除非用户手动启动 Jupyter）

#### ⚠️ 可选的 Jupyter 端口转发：
- 在文档 `SSH_PORT_FORWARDING.md` 中提到
- 用户手动启动，非自动运行
- 仅监听 localhost，需 SSH 隧道访问

### 5. **环境隔离**

#### ✅ Python 虚拟环境：
```bash
# 使用 venv 或 conda 创建隔离环境
python3 -m venv venv_kava
# 或
conda create -n kava_env python=3.10
```

#### ✅ 不影响系统 Python：
- 所有包安装在虚拟环境中
- 不需要 `sudo pip install`
- 不修改系统 Python 路径

---

## 🛡️ 安全保证

### 对 HPC 管理员的保证：

1. **资源使用规范**
   - ✅ 所有计算任务通过 SLURM 调度
   - ✅ 明确指定资源需求（CPU、内存、GPU、时间）
   - ✅ 遵守 HPC 使用规范

2. **文件系统隔离**
   - ✅ 所有操作限制在用户 HOME 目录 (`/home/rpwang`)
   - ✅ 不访问其他用户数据
   - ✅ 不修改系统配置

3. **进程管理**
   - ✅ 所有训练任务通过 SLURM 管理
   - ✅ 任务结束自动清理
   - ✅ 无后台守护进程

4. **安全编码实践**
   - ✅ 使用 `set -euo pipefail` 防止错误传播
   - ✅ 变量引号保护防止路径注入
   - ✅ 临时文件使用用户名隔离
   - ✅ 脚本结束自动清理临时文件

---

## ⚠️ 用户注意事项

### 需要用户注意的地方：

1. **磁盘配额**
   ```bash
   # 检查磁盘使用（你的实际目录）
   df -h /home/rpwang
   quota -s  # 如果 HPC 有配额系统
   du -sh /home/rpwang/.cache/huggingface  # 查看缓存大小
   du -sh "/home/rpwang/kava review"       # 查看项目大小（注意引号）
   ```
   - 模型缓存: ~19GB (`/home/rpwang/.cache/huggingface`)
   - 训练输出: ~5-10GB (`/home/rpwang/kava review/outputs`)
   - 建议保留: 30GB 空闲空间
   
   **⚠️ 路径空格注意**: 使用引号包裹路径 `"/home/rpwang/kava review"`

2. **SLURM 作业数量**
   ```bash
   # 检查运行中的任务
   squeue -u $USER
   ```
   - 默认提交 12 个任务（4 配置 × 3 种子）
   - 根据 HPC 策略调整并发数

3. **网络下载**
   ```bash
   # 如果 HPC 限制外网访问
   bash run_reproduce.sh --method mirror  # 使用镜像
   bash run_reproduce.sh --skip-download  # 跳过下载
   ```

4. **清理旧数据**
   ```bash
   # 定期清理旧 checkpoint（在你的项目目录下）
   cd "/home/rpwang/kava review"  # 使用引号处理空格
   find outputs/ -name "checkpoint-*" -type d -mtime +30  # 查看
   # 手动删除（不会自动执行）
   
   # 清理 HuggingFace 缓存锁文件
   rm -rf ~/.cache/huggingface/hub/.locks
   ```

---

## 📋 安全检查清单

在运行项目前，确认：

- [ ] 有足够磁盘空间（≥20GB）
- [ ] 了解 HPC 的资源限制策略
- [ ] 不会同时提交过多任务
- [ ] 项目目录在自己的 HOME 下 (`/home/rpwang/kava review`)
- [ ] 虚拟环境已激活
- [ ] 已阅读 HPC 使用规范
- [ ] ⚠️ **已注意路径包含空格** - 所有命令使用引号

### ⚠️ 路径空格重要提示

你的项目路径 `/home/rpwang/kava review` 包含空格。在命令行操作时：

```bash
# ✅ 正确 - 使用引号
cd "/home/rpwang/kava review"
source "/home/rpwang/kava review/venv_kava/bin/activate"

# ❌ 错误 - 不使用引号
cd /home/rpwang/kava review  # 会被解析为两个参数

# 💡 推荐 - 重命名目录避免问题
mv "/home/rpwang/kava review" /home/rpwang/kava_review
```

---

## 🔧 推荐的安全配置

### 1. 设置磁盘配额警告

在 `~/.bashrc` 添加：
```bash
# 检查磁盘使用情况
DISK_USAGE=$(df -h $HOME | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "⚠️ 警告: 磁盘使用率 ${DISK_USAGE}%"
fi
```

### 2. 限制并发任务数

编辑 `submit_all_jobs.sh`，添加延迟：
```bash
for config in "${CONFIGS[@]}"; do
    sbatch --export=CONFIG=$config submit_multi_seed.slurm
    sleep 5  # 间隔5秒提交
done
```

### 3. 自动清理旧日志

创建清理脚本：
```bash
# 保留最近30天的日志
find logs/ -name "*.out" -mtime +30 -delete
find logs/ -name "*.err" -mtime +30 -delete
```

---

## 🎯 总结

### ✅ 安全性评估：**通过**

该项目代码：
- ✅ 无系统级危险操作
- ✅ 遵守 HPC 使用规范
- ✅ 资源使用透明可控
- ✅ 文件操作限制在用户目录
- ✅ 通过 SLURM 正确管理计算资源

### 建议：
1. 阅读所在 HPC 的使用规范
2. 根据 HPC 政策调整并发任务数
3. 定期清理不需要的输出和缓存
4. 监控磁盘使用情况

---

## 📞 联系信息

如果 HPC 管理员有任何疑问，可以：
1. 查看本安全审查报告
2. 检查 `SLURM` 脚本的资源配置
3. 查看 GitHub 仓库: https://github.com/theodorewang0731-hash/kava

所有代码公开透明，欢迎审查！
