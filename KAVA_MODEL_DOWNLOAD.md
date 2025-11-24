# KAVA 模型下载指南

**HPC 公共模型库不包含 KAVA 所需模型的解决方案**

---

## ⚠️ 重要发现

经过检查 HPC 公共模型库（`/home/share/models`），发现**没有 KAVA 项目所需的特定模型**。

### KAVA 需要的模型

| 模型 | HuggingFace ID | 在公共库？ | 大小 |
|------|---------------|-----------|------|
| LLaMA 3.2-1B | `meta-llama/Llama-3.2-1B-Instruct` | ❌ 否 | ~5 GB |
| LLaMA 3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | ❌ 否 | ~12 GB |
| Qwen 2.5-0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | ❌ 否 | ~2 GB |

**总下载大小**：~19 GB

### HPC 公共库实际包含的模型

通过 `ls /home/share/models` 查看，公共库包含：

**Llama 系列**（但不是 Llama-3.2）：
- `llama-7b`
- `Llama-2-7b`, `Llama-2-13b`, `Llama-2-70b`
- `Llama-30b`, `llama-65b`

**Code 系列**：
- `CodeLlama-7b/13b/34b/70b-hf/Instruct/Python`

**Qwen 系列**（但不是 Qwen2.5-0.5B）：
- `Qwen1.5-MoE-A2.7B`

**其他模型**：
- `Mistral-7B`, `Mixtral-8x7B`
- `phi-1/2`, `gemma-2b/7b`
- `deepseek-coder`, `deepseek-llm`
- `WizardCoder`, `WizardLM`
- `vicuna`, `stable-code`

---

## 🚀 解决方案

### 方案 A: 下载到个人目录（推荐）

这是最直接的方案，将模型下载到你的个人缓存目录。

#### Step 1: 配置环境变量

```bash
# 配置个人 HuggingFace 缓存
cat >> ~/.bashrc << 'EOF'
# HuggingFace 个人缓存（KAVA 项目）
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface
EOF

# 立即生效
source ~/.bashrc

# 验证
echo $HF_HOME
# 输出：/home/username/.cache/huggingface
```

#### Step 2: 下载模型（3 种方法）

**方法 1: 直接下载（如果网络好）**

```bash
# 安装 huggingface-cli
pip install -U huggingface-hub

# 下载模型（约 35-55 分钟）
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# 显示进度
# Fetching 15 files:   60%|██████    | 9/15 [05:23<03:35, 35.89s/it]
```

**方法 2: 使用代理加速（推荐）**

```bash
# === 在本地机器 ===
# 1. 启动 Clash/Shadowrocket，启用 "Allow LAN"
# 2. 建立反向隧道
ssh -N -R 55555:localhost:7890 username@hpc.example.edu &

# === 在 HPC 终端 ===
# 3. 配置代理
export http_proxy=http://localhost:55555
export https_proxy=http://localhost:55555
export all_proxy=http://localhost:55555

# 4. 测试连接
curl -I https://huggingface.co
# HTTP/2 200

# 5. 下载模型（通过代理加速）
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

**方法 3: 使用 HuggingFace 镜像**

```bash
# 配置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

#### Step 3: 验证下载

```bash
# 检查下载的模型
ls -lh ~/.cache/huggingface/hub/

# 应该看到：
# models--meta-llama--Llama-3.2-1B-Instruct/
# models--meta-llama--Llama-3.2-3B-Instruct/
# models--Qwen--Qwen2.5-0.5B-Instruct/

# 测试加载
python << EOF
from transformers import AutoTokenizer

# 测试 LLaMA 1B
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
print("✓ LLaMA 3.2-1B loaded successfully")

# 测试 LLaMA 3B
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
print("✓ LLaMA 3.2-3B loaded successfully")

# 测试 Qwen
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print("✓ Qwen 2.5-0.5B loaded successfully")
EOF
```

---

### 方案 B: 自动下载（训练时）

如果你不想手动下载，可以在首次训练时自动下载：

```bash
# 配置环境变量（同上）
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface

# 首次运行训练时会自动下载模型
python train.py --config configs/llama1b_aug.yaml

# 输出：
# Downloading model meta-llama/Llama-3.2-1B-Instruct...
# Fetching 15 files:  100%|██████████| 15/15 [10:23<00:00]
# Model downloaded to ~/.cache/huggingface/hub/
# Starting training...
```

**优点**：
- ✅ 无需手动操作
- ✅ 按需下载

**缺点**：
- ❌ 占用训练任务时间
- ❌ 可能导致任务超时（如果下载太慢）
- ❌ 多个任务会重复下载（如果同时启动）

---

### 方案 C: 请求管理员添加（多用户）

如果你的团队有多人需要这些模型，可以请求管理员添加到公共库：

```bash
# 给管理员的邮件模板
主题：请求添加模型到 HPC 公共库

您好，

我们的研究项目（KAVA）需要使用以下模型：
1. meta-llama/Llama-3.2-1B-Instruct (~5 GB)
2. meta-llama/Llama-3.2-3B-Instruct (~12 GB)
3. Qwen/Qwen2.5-0.5B-Instruct (~2 GB)

这些模型目前不在 /home/share/models 中。
如果能添加到公共库，将节省所有用户的下载时间和存储空间。

下载命令：
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

感谢！
```

---

## 📊 下载时间估算

| 网络环境 | LLaMA 1B (5GB) | LLaMA 3B (12GB) | Qwen 0.5B (2GB) | 总计 (19GB) |
|---------|----------------|-----------------|-----------------|-------------|
| 直连 HuggingFace | 15-30 分钟 | 30-60 分钟 | 5-10 分钟 | 50-100 分钟 |
| 使用代理 | 5-10 分钟 | 10-20 分钟 | 2-5 分钟 | 17-35 分钟 |
| HF 镜像 | 10-20 分钟 | 20-40 分钟 | 3-8 分钟 | 33-68 分钟 |

**推荐**：使用代理加速（方案 A 方法 2），最快 **17-35 分钟**完成所有下载。

---

## 🔍 验证下载完整性

```bash
# 检查文件数量
find ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct -type f | wc -l
# 应该有 15-20 个文件

# 检查总大小
du -sh ~/.cache/huggingface/hub/models--*

# 输出示例：
# 5.2G    models--meta-llama--Llama-3.2-1B-Instruct
# 12.8G   models--meta-llama--Llama-3.2-3B-Instruct
# 2.1G    models--Qwen--Qwen2.5-0.5B-Instruct

# 测试加载速度
python << EOF
import time
from transformers import AutoModelForCausalLM

start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
elapsed = time.time() - start
print(f"✓ Model loaded in {elapsed:.2f}s")
# 预期：5-15 秒
EOF
```

---

## 💡 最佳实践

### 1. 使用交互式会话下载

```bash
# 申请 GPU 节点（虽然下载不需要 GPU，但避免占用登录节点）
srun --time=2:00:00 --mem=16G --pty bash -i

# 配置环境
conda activate kava
export HF_HOME=$HOME/.cache/huggingface

# 下载模型
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# 完成后退出
exit
```

### 2. 后台下载（避免 SSH 断开）

```bash
# 使用 nohup 后台下载
nohup huggingface-cli download meta-llama/Llama-3.2-1B-Instruct > download_llama1b.log 2>&1 &
nohup huggingface-cli download meta-llama/Llama-3.2-3B-Instruct > download_llama3b.log 2>&1 &
nohup huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct > download_qwen05b.log 2>&1 &

# 查看进度
tail -f download_llama1b.log

# 查看进程
ps aux | grep huggingface-cli
```

### 3. 磁盘空间管理

```bash
# 检查磁盘配额
quota -s

# 检查 HuggingFace 缓存大小
du -sh ~/.cache/huggingface

# 如果空间不足，清理旧模型
rm -rf ~/.cache/huggingface/hub/models--old-model-name

# 或软链接到其他目录（如果有大容量存储）
mkdir -p /scratch/username/huggingface
mv ~/.cache/huggingface /scratch/username/
ln -s /scratch/username/huggingface ~/.cache/huggingface
```

---

## 🚨 故障排除

### 问题 1: 下载中断

```bash
# 症状：下载到一半断开
# ConnectionError: HTTPSConnectionPool

# 解决：重新运行下载命令，会自动续传
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

# 或使用 --resume-download
huggingface-cli download --resume-download meta-llama/Llama-3.2-1B-Instruct
```

### 问题 2: 磁盘配额不足

```bash
# 症状：No space left on device

# 解决：使用 scratch 目录
export HF_HOME=/scratch/$USER/huggingface
mkdir -p $HF_HOME
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
```

### 问题 3: 权限错误

```bash
# 症状：Permission denied

# 解决：检查目录权限
ls -ld ~/.cache/huggingface
chmod 755 ~/.cache/huggingface
```

### 问题 4: 模型加载失败

```bash
# 症状：OSError: Model not found

# 检查环境变量
echo $HF_HOME
echo $TRANSFORMERS_CACHE

# 检查模型文件
ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/

# 如果文件不完整，删除并重新下载
rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
```

---

## 📚 相关文档

- [GETTING_STARTED_HPC.md](GETTING_STARTED_HPC.md) - HPC 完整上手指南
- [HPC_REFERENCE.md](HPC_REFERENCE.md) - HPC 命令参考
- [SSH_PORT_FORWARDING.md](SSH_PORT_FORWARDING.md) - 代理配置详解
- [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) - 完整复现指南

---

## 🎯 快速命令备忘

```bash
# 配置环境
export HF_HOME=$HOME/.cache/huggingface
echo 'export HF_HOME=$HOME/.cache/huggingface' >> ~/.bashrc

# 下载模型（直连）
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# 下载模型（使用代理）
export all_proxy=http://localhost:55555
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

# 验证
ls ~/.cache/huggingface/hub/
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')"
```

---

**预计下载时间**：17-100 分钟（取决于网络）  
**磁盘空间需求**：~19 GB  
**一次性操作**：模型下载后永久保存，后续训练无需重复下载
