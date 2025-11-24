# KAVA 一键运行指南

## 🚀 快速开始

### 最简单的方式（推荐）

```bash
# 在 HPC 登录节点运行
cd "/home/rpwang/kava review"
bash quick_start.sh
```

就这样！脚本会自动完成：
- ✅ 下载模型和数据集
- ✅ 验证资源完整性
- ✅ 更新配置文件
- ✅ 提交所有训练任务（12个）
- ✅ 监控训练进度
- ✅ 收集和打包结果

## 📋 三种使用方式

### 方式 1：标准运行（推荐新手）

```bash
bash quick_start.sh
```

**特点**：
- 使用 HF-Mirror 镜像加速下载
- 下载所有模型和数据集
- 自动完成全流程
- 适合第一次运行

### 方式 2：跳过下载（资源已存在）

```bash
bash quick_start.sh --skip-download
```

**特点**：
- 跳过下载步骤
- 适合资源已下载或使用共享存储
- 直接开始训练

### 方式 3：完全自定义

```bash
bash run_everything.sh
```

**可自定义的选项**（编辑脚本开头）：
```bash
USE_HF_MIRROR=true      # 是否使用镜像
SKIP_DOWNLOAD=false     # 是否跳过下载
UPLOAD_TO_HF=true       # 是否上传结果
HF_REPO="user/repo"     # HuggingFace 仓库
```

## 📊 运行后做什么

### 1. 查看结果摘要

```bash
# 摘要文件会自动生成
cat all_results_*/SUMMARY.txt
```

### 2. 分析实验结果

```bash
# 生成详细的结果报告
python analyze_results.py
```

这会生成：
- `results_summary.csv` - CSV 格式的汇总表
- `RESULTS_REPORT.md` - Markdown 格式的详细报告

### 3. 下载结果到本地

```bash
# 在本地机器运行
scp user@hpc:"/home/rpwang/kava review/kava_results_*.tar.gz" .

# 解压
tar -xzf kava_results_*.tar.gz
```

## 🔧 高级选项

### 只运行特定配置

如果你只想训练某个配置（如 Qwen-0.5B），可以直接提交：

```bash
# 提交单个配置，3 个随机种子
sbatch --export=CONFIG=qwen05b_aug --array=0,1,2 submit_multi_seed.slurm
```

### 自定义训练配置

编辑配置文件：

```bash
# 修改 Qwen 配置
vim configs/qwen05b_aug.yaml

# 常见修改：
# - training.epochs: 训练轮数
# - training.batch_size: 批次大小
# - lora.r: LoRA 秩
# - latent.num_tokens: Latent tokens 数量
```

### 使用共享存储的模型

如果 HPC 已有共享模型：

```bash
# 1. 查找共享模型
bash check_hpc_datasets.sh

# 2. 手动更新配置文件
vim configs/llama1b_aug.yaml

# 修改 model.name 为共享路径
model:
  name: "/home/share/models/Llama-3.2-1B-Instruct"
```

## 🐛 故障排查

### 问题 1: 下载失败

```
ERROR: Failed to download model
```

**解决**：
```bash
# 检查网络
curl -I https://huggingface.co

# 使用镜像重试
HF_ENDPOINT=https://hf-mirror.com bash quick_start.sh

# 或手动下载
bash download_models_only.sh
bash download_datasets_only.sh
```

### 问题 2: 任务提交失败

```
ERROR: Submitted batch job failed
```

**解决**：
```bash
# 检查 SLURM 配置
sinfo  # 查看可用节点
squeue -u $USER  # 查看任务队列

# 检查 SLURM 脚本
cat submit_multi_seed.slurm

# 测试单个任务
sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm
```

### 问题 3: 训练 OOM (内存不足)

```
ERROR: CUDA out of memory
```

**解决**：
```bash
# 减少批次大小
vim configs/qwen05b_aug.yaml
# 修改 training.batch_size: 128 → 64

# 或使用梯度累积
# 修改 training.gradient_accumulation_steps: 1 → 2
```

### 问题 4: 任务卡住不动

```bash
# 检查任务状态
squeue -j <job_id>

# 查看实时日志
tail -f logs/kava_<job_id>_<array_id>.out

# 取消任务
scancel <job_id>
```

## 📁 文件结构

运行后会生成以下文件：

```
kava review/
├── models/                          # 下载的模型
│   ├── Llama-3.2-1B-Instruct/
│   ├── Llama-3.2-3B-Instruct/
│   └── Qwen2.5-0.5B-Instruct/
├── datasets/                        # 下载的数据集
│   ├── gsm8k-aug/
│   ├── gsm8k-aug-nl/
│   └── gsm8k/
├── results/                         # 训练结果
│   ├── llama1b_aug_seed42/
│   ├── llama1b_aug_seed123/
│   └── ...
├── logs/                            # SLURM 日志
│   ├── kava_20110_0.out
│   └── ...
├── all_results_YYYYMMDD_HHMMSS/     # 收集的结果
│   ├── SUMMARY.txt
│   └── logs/
├── kava_results_*.tar.gz            # 打包的结果
├── results_summary.csv              # CSV 汇总
├── RESULTS_REPORT.md                # Markdown 报告
└── .job_ids.txt                     # Job IDs 记录
```

## ⏱️ 预计时间

基于 HPC 配置（A100-80GB × 1）：

| 阶段 | 时间 |
|------|------|
| 下载模型 | 10-30 分钟 |
| 下载数据集 | 5-15 分钟 |
| 任务提交 | < 1 分钟 |
| 单个任务训练 | 2-6 小时 |
| 12 个任务（并行） | 36-48 小时 |
| 结果收集 | < 5 分钟 |

**总计**：约 2-3 天（大部分时间是训练）

## 🎯 检查清单

运行前确认：

- [ ] 在 HPC 登录节点（不是计算节点）
- [ ] 已安装所有依赖（PyTorch, transformers, peft, datasets）
- [ ] 有足够的磁盘空间（至少 50GB）
- [ ] 有足够的 GPU 时间配额
- [ ] （可选）已登录 HuggingFace：`huggingface-cli login`

运行后验证：

- [ ] 所有 12 个任务已提交
- [ ] 至少 1 个任务开始运行
- [ ] 日志文件正在生成
- [ ] 没有 OOM 或其他错误

## 📞 获取帮助

查看帮助信息：

```bash
bash quick_start.sh --help
```

查看详细日志：

```bash
# 实时查看最新日志
tail -f logs/kava_*.out

# 搜索错误
grep -r "ERROR" logs/

# 查看完整日志
cat logs/kava_<job_id>_<array_id>.out
```

## 🎉 成功标志

运行成功的标志：

```
✓ 所有任务已提交，共 12 个任务
✓ 所有任务已完成
✓ 结果已打包: kava_results_*.tar.gz
✓ 统计: 完成 12, 失败 0
🎉 所有任务完成！
```

然后你会看到：
- `RESULTS_REPORT.md` 包含详细的实验结果
- `results_summary.csv` 包含汇总表格
- 可以下载 `.tar.gz` 文件到本地分析

## 🔗 相关文档

- [HPC 下载指南](HPC_DOWNLOAD_GUIDE.md) - 详细的下载说明
- [SLURM 提交指南](submit_all_jobs.sh) - 任务提交脚本
- [结果分析工具](analyze_results.py) - 结果分析脚本
- [R-KV 修复说明](RKV_PADDING_FIX.md) - Padding tokens 处理
