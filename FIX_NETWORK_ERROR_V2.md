# 🔧 修复 "Network is unreachable" 问题 - 正确版本

## 📋 问题根源分析

### ❌ 之前的误解
我之前认为需要修改配置文件，把 `meta-llama/Llama-3.2-1B-Instruct` 改成本地路径 `/home/share/models/Llama-3.2-1B-Instruct`。

### ✅ 真正的问题
**HPC 共享库里的模型是按照 HuggingFace 标准格式存储的**：
```
/home/share/models/
├── models--meta-llama--Llama-3.2-1B-Instruct/
│   └── snapshots/<hash>/
│       ├── config.json
│       ├── model.safetensors
│       └── ...
├── models--Qwen--Qwen2.5-0.5B-Instruct/
│   └── snapshots/<hash>/
│       └── ...
```

当设置 `HF_HOME=/home/share/models` 时，transformers **应该能找到这些模型**，但代码中的 `from_pretrained()` **默认会先尝试联网**检查更新，即使本地已有完整模型。

---

## ✅ 正确的解决方案（代码级修复）

### 修改 1: `src/trainer.py` - 添加 `local_files_only` 参数

**问题代码**：
```python
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)
```

**修复后**：
```python
# 检查是否使用离线模式
use_local = os.path.exists(model_name) or os.environ.get('HUGGINGFACE_HUB_OFFLINE') == '1'

self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    local_files_only=use_local  # ✅ 关键参数
)
```

### 修改 2: `evaluate.py` - 同样的修复

已在评估脚本中添加相同的 `local_files_only` 参数。

### 修改 3: `submit_multi_seed.slurm` - 环境变量保持不变

SLURM 脚本中的环境变量**已经正确**：
```bash
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 🔍 工作原理

### HuggingFace 模型查找逻辑

当调用 `from_pretrained("meta-llama/Llama-3.2-1B-Instruct")` 时：

1. **默认行为**（`local_files_only=False`）：
   ```
   ① 尝试联网到 huggingface.co 检查最新版本
   ② 如果网络失败 → 抛出 "Network is unreachable"
   ③ 即使本地缓存存在也不会使用（因为第①步就失败了）
   ```

2. **离线模式**（`local_files_only=True`）：
   ```
   ① 直接查找本地缓存：$HF_HOME/models--meta-llama--Llama-3.2-1B-Instruct/
   ② 如果找到 → 直接加载
   ③ 如果没找到 → 立即报错（不尝试联网）
   ```

3. **智能判断**（我们的方案）：
   ```python
   # 如果设置了 HUGGINGFACE_HUB_OFFLINE=1，使用离线模式
   use_local = os.environ.get('HUGGINGFACE_HUB_OFFLINE') == '1'
   
   # 或者如果 model_name 是本地路径（如 /home/share/models/xxx），也用离线
   use_local = os.path.exists(model_name) or use_local
   ```

---

## 📊 配置文件无需修改

**保持原样**（使用标准 repo ID）：
```yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"  # ✅ 标准格式
```

**不要改成**（这样反而不对）：
```yaml
model:
  name: "/home/share/models/Llama-3.2-1B-Instruct"  # ❌ 错误
```

**原因**：
- HPC 共享库使用 HuggingFace 标准缓存格式（`models--<org>--<model>/snapshots/<hash>/`）
- transformers 通过 repo ID 自动解析路径
- 直接指定路径会跳过缓存机制，可能找不到文件

---

## 🚀 测试步骤

### 步骤 1: 验证代码修复

```bash
cd "/home/rpwang/kava review"
source venv/bin/activate

# 设置环境变量
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 快速测试加载
python -c "
import os
os.environ['HF_HOME'] = '/home/share/models'
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'

from transformers import AutoConfig

# 测试加载配置（不加载整个模型，速度快）
config = AutoConfig.from_pretrained(
    'meta-llama/Llama-3.2-1B-Instruct',
    local_files_only=True
)
print(f'✓ 成功加载 Llama-1B 配置')
print(f'  模型类型: {config.model_type}')
print(f'  隐藏层: {config.hidden_size}')
"
```

**预期输出**：
```
✓ 成功加载 Llama-1B 配置
  模型类型: llama
  隐藏层: 2048
```

### 步骤 2: 单任务测试

```bash
# 提交最小任务
sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm

# 等待 2-3 分钟查看日志
tail -n 100 outputs/logs/kava_qwen05b_aug_*.out
```

**成功标志**：
```
Loading base model...
Model: Qwen/Qwen2.5-0.5B-Instruct
Loading mode: Local/Offline              ← ✅ 应该显示这个
✓ Model loaded successfully
```

**不应出现**：
```
Network is unreachable
Cannot connect to huggingface.co
```

### 步骤 3: 提交所有任务

```bash
bash submit_all_jobs.sh
bash monitor_jobs.sh --auto
```

---

## 🐛 常见问题

### Q1: 为什么不直接改配置文件路径？

**A**: HPC 的模型存储格式是：
```
/home/share/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<hash>/
```

如果配置文件写 `/home/share/models/Llama-3.2-1B-Instruct`，transformers 会去找：
```
/home/share/models/Llama-3.2-1B-Instruct/config.json  ← ❌ 不存在
```

正确的做法是设置 `HF_HOME` + 使用标准 repo ID，transformers 会自动解析成：
```
/home/share/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<hash>/config.json  ← ✅ 存在
```

### Q2: `local_files_only` 和 `HUGGINGFACE_HUB_OFFLINE` 有什么区别？

**A**: 
- `HUGGINGFACE_HUB_OFFLINE=1`：环境变量，全局设置
- `local_files_only=True`：参数，单次调用设置

我们的代码会读取环境变量，自动决定是否传 `local_files_only=True`。

### Q3: 如果 HPC 的模型格式不是标准缓存格式怎么办？

**A**: 运行这个命令检查：
```bash
ls -la /home/share/models/

# 如果看到 models--meta-llama--Llama-3.2-1B-Instruct/ → 标准格式 ✅
# 如果看到 Llama-3.2-1B-Instruct/config.json → 直接格式（需要改配置）
```

如果是直接格式，那么**之前的方案是对的**（改配置文件用本地路径）。

---

## 📞 验证 HPC 模型格式

请 AI 助手运行这个命令确认：

```bash
# 检查模型存储格式
echo "=== HPC 模型库格式检查 ==="
ls -la /home/share/models/ | grep -i llama | head -5
echo ""
ls -la /home/share/models/ | grep -i qwen | head -5

# 如果看到 models--xxx 格式
if ls /home/share/models/models--* 2>/dev/null | grep -q .; then
    echo "✓ 使用 HuggingFace 标准缓存格式"
    echo "  → 配置文件应使用: meta-llama/Llama-3.2-1B-Instruct"
    echo "  → 代码已修复: local_files_only=True"
else
    echo "✗ 使用直接目录格式"
    echo "  → 配置文件应使用: /home/share/models/Llama-3.2-1B-Instruct"
fi
```

---

## ✨ 修复总结

| 文件 | 修改内容 | 原因 |
|------|----------|------|
| `src/trainer.py` | 添加 `local_files_only=use_local` | 避免联网检查 |
| `evaluate.py` | 添加 `local_files_only=use_local` | 避免联网检查 |
| `configs/*.yaml` | **保持原样**（repo ID） | 配合 HF_HOME 使用 |
| `submit_multi_seed.slurm` | **已正确**（环境变量） | 无需修改 |

**核心原理**：
```python
# 当 HUGGINGFACE_HUB_OFFLINE=1 时
from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",  # repo ID
    local_files_only=True                  # 强制离线
)

# transformers 自动查找：
# $HF_HOME/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<hash>/
```

这样既保持了配置文件的标准性，又解决了网络访问问题！
