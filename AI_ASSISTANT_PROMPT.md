# KAVA 项目部署提示词 - 给 HPC AI 助手

> 将下面的内容复制给 HPC 中的 ChatGPT-4/o1 助手

---

## 📋 任务说明

你好！我需要在 HPC 集群上部署并运行 KAVA 项目（论文复现）。这是一个深度学习项目，需要：
1. 配置 Python 环境
2. 下载大语言模型（~19GB）
3. 提交 SLURM 训练任务
4. 监控进度并收集结果

项目已经上传到当前目录，包含完整的自动化脚本。

---

## 🎯 核心需求

**目标：** 严格按照论文规格复现 KAVA 实验结果

**配置：**
- 4 个模型配置（Llama-3.2-1B/3B, Qwen2.5-0.5B）
- 3 个随机种子（42, 123, 456）
- 共 12 个训练任务（4 × 3）
- 3 个评估数据集（GSM8K, GSM8K-Hard, SVAMP）

**资源需求：**
- GPU: A100-80GB（每任务 1 个）
- 磁盘: ~20GB（模型缓存）
- 时间: 36-48 小时（并行训练）

---

## ✅ 执行步骤（请按顺序执行）

### 第 1 步：验证部署环境

请运行以下命令验证所有文件和环境：

```bash
cd ~/kava  # 或项目所在目录
bash verify_deployment.sh
```

**期望输出：** 所有检查都显示 ✓（绿色对勾）

**如果有错误：** 请告诉我具体的错误信息，我会提供解决方案。

---

### 第 2 步：快速环境配置

运行配置脚本设置 HuggingFace 缓存和权限：

```bash
bash setup_hpc.sh
```

**这会自动完成：**
- 配置 `$HOME/.cache/huggingface` 缓存目录
- 创建必要的输出目录
- 设置脚本执行权限
- 验证 SLURM 环境

**注意：** 脚本会修改 `~/.bashrc`，配置完成后请运行：
```bash
source ~/.bashrc
```

---

### 第 3 步：一键启动训练

现在运行主启动脚本：

```bash
bash start.sh
```

**或者，如果你能判断网络情况，使用优化选项：**

```bash
# 如果从中国大陆访问（推荐使用镜像）
bash start.sh --method mirror

# 如果配置了代理
bash start.sh --method proxy

# 如果已经下载过模型
bash start.sh --skip-download
```

**这个脚本会自动完成所有工作：**
1. ✅ 重新验证环境
2. ✅ 创建 conda 环境（kava_env, Python 3.10）
3. ✅ 安装依赖（torch, transformers, peft 等）
4. ✅ 下载模型（3 个模型，~19GB）
5. ✅ 提交 12 个 SLURM 训练任务
6. ✅ 创建监控脚本

**预计时间：**
- 环境配置: 5-10 分钟
- 模型下载: 17-100 分钟（取决于网络）
- 训练: 36-48 小时（自动，无需等待）

---

### 第 4 步：监控训练进度

训练提交后，使用以下命令监控：

```bash
# 快速查看所有任务状态
bash monitor_jobs.sh

# 或手动检查 SLURM 队列
squeue -u $USER

# 查看特定任务的实时日志
tail -f outputs/logs/llama1b_aug_seed42.log

# 查看所有日志（最近更新的）
ls -lt outputs/logs/ | head -10
```

**正常状态：**
- 12 个任务应该显示为 `RUNNING` 或 `PENDING`
- 日志文件应该定期更新（每分钟都有新内容）
- GPU 利用率应该在 80-100%

---

### 第 5 步：收集结果（训练完成后）

等待所有任务完成（`squeue -u $USER` 显示为空），然后：

```bash
# 聚合所有结果
bash collect_results.sh

# 查看汇总结果
cat outputs/aggregated_results.csv | column -t -s,
```

**期望输出：** 包含 GSM8K、GSM8K-Hard、SVAMP 的准确率，格式化的表格

---

## 🔍 关键信息

### 模型下载说明

**重要：** HPC 的 `/home/share/models` 公共库**没有 KAVA 所需的模型**！

需要下载的模型：
1. `meta-llama/Llama-3.2-1B-Instruct` (~5GB)
2. `meta-llama/Llama-3.2-3B-Instruct` (~6GB)
3. `Qwen/Qwen2.5-0.5B-Instruct` (~1GB)

下载到：`$HOME/.cache/huggingface`

**下载时间估算：**
- 直连 HuggingFace: 50-100 分钟
- 使用代理: 17-35 分钟
- 使用镜像 (hf-mirror.com): 33-68 分钟

**如果下载失败：** 请查看 `docs/KAVA_MODEL_DOWNLOAD.md` 获取手动下载方案

---

### SLURM 任务配置

每个任务的资源分配：
- **分区：** compute
- **节点：** 1
- **GPU：** 1 × A100-80GB
- **CPU：** 8 核
- **内存：** 64GB
- **时间限制：** 48 小时

**任务命名：** `kava_<config>_<seed>`
- 例如：`kava_llama1b_aug_42`, `kava_qwen05b_aug_123`

---

## 🆘 常见问题处理

### 问题 1: 磁盘空间不足

**症状：** `Insufficient disk space: XXgb available`

**解决：**
```bash
# 检查配额
df -h $HOME

# 清理 HuggingFace 缓存
rm -rf $HOME/.cache/huggingface/hub/.locks
huggingface-cli delete-cache

# 如果还不够，请联系管理员申请增加配额
```

---

### 问题 2: 模型下载超时

**症状：** `ConnectionError` 或 `TimeoutError`

**解决：**
```bash
# 方法 1: 使用镜像站
bash start.sh --method mirror

# 方法 2: 配置代理（如果有）
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
bash start.sh --method proxy

# 方法 3: 手动下载（参见 docs/KAVA_MODEL_DOWNLOAD.md）
```

---

### 问题 3: SLURM 任务一直排队

**症状：** `squeue` 显示所有任务都是 `PENDING`

**检查：**
```bash
# 查看排队原因
squeue -u $USER --start

# 检查分区状态
sinfo -p compute

# 查看节点可用性
sinfo -p compute -N -l
```

**可能原因：**
- 集群繁忙（等待资源释放）
- 配额限制（联系管理员）
- 配置错误（检查 submit_multi_seed.slurm）

---

### 问题 4: 训练失败（OOM）

**症状：** 日志显示 `CUDA out of memory`

**解决：**
```bash
# 降低 batch size（编辑配置文件）
nano configs/llama1b_aug.yaml
# 将 batch_size: 4 改为 batch_size: 2

# 重新提交
bash run_reproduce.sh --skip-env --skip-download
```

---

## 📊 预期结果

### 训练完成标志
- ✅ `squeue -u $USER` 显示为空（无运行中任务）
- ✅ `outputs/results/` 包含 12 个 JSON 文件
- ✅ `outputs/aggregated_results.csv` 已生成

### 结果格式示例
```csv
config,seed,gsm8k_acc,gsm8k_hard_acc,svamp_acc,avg_passes
llama1b_aug,42,75.2,62.8,78.5,6.2
llama1b_aug,123,74.8,63.1,77.9,6.3
llama1b_aug,456,75.5,62.5,78.2,6.1
...
```

### 论文对比
结果应在论文报告值的 **±2%** 范围内（考虑随机性）

---

## 📁 重要文件位置

```
~/kava/
├── start.sh                      ← 一键启动脚本
├── verify_deployment.sh          ← 验证脚本
├── setup_hpc.sh                  ← 配置脚本
├── run_reproduce.sh              ← 核心自动化脚本
├── monitor_jobs.sh               ← 监控脚本（自动生成）
├── collect_results.sh            ← 结果收集（自动生成）
├── outputs/
│   ├── logs/                     ← SLURM 日志
│   ├── results/                  ← 评估结果
│   └── checkpoints/              ← 模型检查点
└── docs/
    ├── HPC_LINUX_READY.md        ← Linux 部署完整文档
    ├── GETTING_STARTED_HPC.md    ← HPC 入门指南
    └── KAVA_MODEL_DOWNLOAD.md    ← 模型下载详解
```

---

## 🎯 AI 助手的执行清单

请按照以下清单执行，并在每步完成后告诉我结果：

- [ ] **步骤 1:** 运行 `bash verify_deployment.sh`
  - 报告：是否所有检查都通过？如有错误，具体是什么？

- [ ] **步骤 2:** 运行 `bash setup_hpc.sh`
  - 报告：是否成功配置？

- [ ] **步骤 3:** 运行 `bash start.sh`（或带选项）
  - 报告：是否成功启动？看到多少个任务提交？

- [ ] **步骤 4:** 运行 `squeue -u $USER`
  - 报告：有多少任务在运行/排队？

- [ ] **步骤 5:** 监控日志 `tail -f outputs/logs/*.log`
  - 报告：训练是否正常进行？有无错误？

- [ ] **步骤 6:** 等待完成后运行 `bash collect_results.sh`
  - 报告：结果是否符合预期？

---

## 💡 给 AI 助手的建议

1. **按顺序执行：** 不要跳过验证和配置步骤
2. **报告详细信息：** 每步执行后，复制完整的输出给我
3. **处理错误：** 如果某步失败，停下来告诉我错误信息
4. **监控进度：** 定期检查 `squeue` 和日志文件
5. **保持耐心：** 模型下载和训练需要较长时间（1-2天）

---

## 🚀 开始吧！

请从**步骤 1**开始执行，并告诉我每一步的结果。我会根据你的反馈提供进一步的指导。

如果遇到任何问题，请提供：
1. 具体的错误消息
2. 相关的日志输出
3. 当前的工作目录和文件列表

让我们开始这个项目！💪
