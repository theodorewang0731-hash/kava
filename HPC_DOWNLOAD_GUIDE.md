# HPC 模型和数据集下载指南

## 📋 概述

HPC 集群的**计算节点通常没有网络访问**，但**登录节点有网络**。因此需要：
1. 在登录节点下载模型和数据集
2. 下载到用户目录或共享存储
3. 计算节点通过本地路径或缓存访问

## 🚀 方法一：使用 Python 脚本（推荐）

### 1. 安装依赖

```bash
# 在登录节点
pip install huggingface_hub
```

### 2. 登录 HuggingFace（如果需要）

```bash
# LLaMA 模型需要授权，先在 https://huggingface.co/meta-llama 申请
huggingface-cli login
# 输入你的 HuggingFace token
```

### 3. 下载模型和数据集

#### 选项 A：全部下载
```bash
# 直连（国外或有代理）
python download_from_hf.py

# 使用镜像（国内推荐）
HF_ENDPOINT=https://hf-mirror.com python download_from_hf.py
```

#### 选项 B：只下载模型
```bash
# 直连
python download_from_hf.py --models-only

# 使用镜像
HF_ENDPOINT=https://hf-mirror.com python download_from_hf.py --models-only
```

#### 选项 C：只下载数据集
```bash
# 直连
python download_from_hf.py --datasets-only

# 使用镜像
HF_ENDPOINT=https://hf-mirror.com python download_from_hf.py --datasets-only
```

### 4. 下载内容

**模型** (下载到 `./models/`):
- `Llama-3.2-1B-Instruct` (~2.5 GB)
- `Llama-3.2-3B-Instruct` (~6 GB)
- `Qwen2.5-0.5B-Instruct` (~1 GB)

**数据集** (下载到 `./datasets/`):
- `gsm8k-aug` (~385K 样本，equation-only CoT)
- `gsm8k-aug-nl` (~385K 样本，natural language CoT)
- `gsm8k` (~7.5K 训练 + 1.3K 测试样本)

## 🔧 方法二：使用 Shell 脚本

### 下载模型
```bash
# 直连
bash download_models_only.sh

# 使用镜像
HF_ENDPOINT=https://hf-mirror.com bash download_models_only.sh
```

### 下载数据集
```bash
# 直连
bash download_datasets_only.sh

# 使用镜像
HF_ENDPOINT=https://hf-mirror.com bash download_datasets_only.sh
```

## 📦 方法三：手动使用 huggingface-cli

### 下载模型
```bash
# 下载 LLaMA-1B
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
    --local-dir ./models/Llama-3.2-1B-Instruct \
    --local-dir-use-symlinks False

# 下载 LLaMA-3B
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --local-dir ./models/Llama-3.2-3B-Instruct \
    --local-dir-use-symlinks False

# 下载 Qwen-0.5B
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir ./models/Qwen2.5-0.5B-Instruct \
    --local-dir-use-symlinks False
```

### 下载数据集
```bash
# 下载 gsm8k-aug
huggingface-cli download whynlp/gsm8k-aug \
    --repo-type dataset \
    --local-dir ./datasets/gsm8k-aug \
    --local-dir-use-symlinks False

# 下载 gsm8k-aug-nl
huggingface-cli download whynlp/gsm8k-aug-nl \
    --repo-type dataset \
    --local-dir ./datasets/gsm8k-aug-nl \
    --local-dir-use-symlinks False

# 下载 gsm8k
huggingface-cli download gsm8k \
    --repo-type dataset \
    --local-dir ./datasets/gsm8k \
    --local-dir-use-symlinks False
```

### 使用镜像加速
```bash
# 设置环境变量后再运行上述命令
export HF_ENDPOINT=https://hf-mirror.com
```

## 🔍 方法四：检查 HPC 共享存储

HPC 可能已经提供了共享模型/数据集：

```bash
# 运行检查脚本
bash check_hpc_datasets.sh

# 手动搜索
find /home/share -name "*llama*" -o -name "*qwen*" -o -name "*gsm8k*" 2>/dev/null
```

如果找到了共享资源，可以直接使用，无需下载！

## 📝 更新配置文件

下载完成后，更新配置文件中的路径：

### 如果下载到本地目录
```yaml
# configs/llama1b_aug.yaml
model:
  name: "./models/Llama-3.2-1B-Instruct"  # 相对路径
  # 或绝对路径: "/home/rpwang/kava review/models/Llama-3.2-1B-Instruct"

dataset:
  name: "./datasets/gsm8k-aug"
```

### 如果使用 HPC 共享存储
```yaml
# configs/llama1b_aug.yaml
model:
  name: "/home/share/models/Llama-3.2-1B-Instruct"  # HPC 共享路径

dataset:
  name: "/home/share/datasets/gsm8k-aug"
```

### 如果使用 HuggingFace 缓存
```yaml
# configs/llama1b_aug.yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"  # 保持 repo_id
  # 代码会自动从 ~/.cache/huggingface/ 加载

dataset:
  name: "whynlp/gsm8k-aug"
```

## ⚠️ 常见问题

### 1. LLaMA 模型 403 错误
```
ERROR: Access denied (403)
```
**解决方法**:
- 访问 https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- 点击 "Request Access" 申请权限
- 等待批准（通常几分钟到几小时）
- 运行 `huggingface-cli login` 登录

### 2. 下载速度慢
```
Downloading: 0%|          | 0.00/2.5G [00:00<?, ?B/s]
```
**解决方法**:
- 使用镜像: `HF_ENDPOINT=https://hf-mirror.com`
- 使用代理: `export https_proxy=http://proxy:port`
- 断点续传: 脚本支持 `resume_download=True`

### 3. 磁盘空间不足
```
ERROR: No space left on device
```
**解决方法**:
```bash
# 检查磁盘使用
df -h

# 清理 HuggingFace 缓存
rm -rf ~/.cache/huggingface/hub/*

# 使用其他目录
export HF_HOME=/path/to/large/disk/.cache/huggingface
```

### 4. 计算节点无法访问
```
ERROR: Network is unreachable
```
**解决方法**:
- 确保在**登录节点**下载，不是计算节点
- 检查下载路径是否可被计算节点访问
- 使用绝对路径或设置 `local_files_only=True`

## 📊 下载检查清单

下载完成后，验证文件完整性：

```bash
# 检查模型文件
ls -lh models/Llama-3.2-1B-Instruct/
# 应该包含:
# - config.json
# - tokenizer.json
# - model-*.safetensors 或 pytorch_model.bin

# 检查数据集文件
ls -lh datasets/gsm8k-aug/
# 应该包含:
# - dataset_info.json 或 README.md
# - train.parquet 或 data/ 目录

# 验证可加载
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./models/Llama-3.2-1B-Instruct', local_files_only=True)
print('✓ 模型可加载')
"

python -c "
from datasets import load_from_disk
dataset = load_from_disk('./datasets/gsm8k-aug')
print('✓ 数据集可加载')
print(f'样本数: {len(dataset)}')
"
```

## 🚀 下一步

下载并验证完成后：

```bash
# 1. 测试单个训练任务
sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm

# 2. 提交全部训练任务
bash submit_all_jobs.sh

# 3. 监控任务状态
bash monitor_jobs.sh --auto
```

## 📚 参考资料

- [HuggingFace Hub 文档](https://huggingface.co/docs/huggingface_hub)
- [HF-Mirror 镜像站](https://hf-mirror.com/)
- [datasets 库文档](https://huggingface.co/docs/datasets)
