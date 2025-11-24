# 🔧 修复 "Network is unreachable" 问题

## 📋 问题诊断

**症状**：
- 所有任务快速失败（几秒内）
- `squeue` 显示 0 个运行中任务
- 日志错误：`[Errno 101] Network is unreachable`
- transformers 尝试从 HuggingFace Hub 下载模型但失败

**根本原因**：
1. ❌ 配置文件使用 HF repo ID（如 `meta-llama/Llama-3.2-1B-Instruct`）
2. ❌ transformers 尝试联网获取元数据
3. ❌ 计算节点无外网访问
4. ❌ 本地缓存布局不符合 transformers 预期

---

## ✅ 解决方案（已自动修复）

### 修改 1：配置文件使用本地路径

**修改前**（使用 HF repo ID）：
```yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
```

**修改后**（使用本地路径）：
```yaml
model:
  name: "/home/share/models/Llama-3.2-1B-Instruct"  # ✅ 本地路径
```

**已修改的文件**：
- ✅ `configs/llama1b_aug.yaml`
- ✅ `configs/llama1b_aug_nl.yaml`
- ✅ `configs/llama3b_aug.yaml`
- ✅ `configs/qwen05b_aug.yaml`

### 修改 2：SLURM 脚本强制离线模式

在 `submit_multi_seed.slurm` 中添加：
```bash
# 强制离线模式 - 避免网络访问
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

**优点**：
- ✅ 避免任何网络访问尝试
- ✅ 加载速度更快（直接读取本地文件）
- ✅ 不依赖缓存布局
- ✅ 错误信息更明确（立即失败而非长时间重试）

---

## 🔍 验证步骤（建议先运行）

### 步骤 1: 运行诊断脚本

```bash
cd "/home/rpwang/kava review"
source venv/bin/activate

# 设置环境变量（与 SLURM 脚本一致）
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 运行诊断
python quick_model_test.py
```

**预期输出**：
```
✅ 推荐方案: 在配置文件中使用本地路径

修改 configs/*.yaml 中的 model.name 为:
  - llama1b: /home/share/models/Llama-3.2-1B-Instruct
  - llama3b: /home/share/models/Llama-3.2-3B-Instruct
  - qwen05b: /home/share/models/Qwen2.5-0.5B-Instruct

这样可以:
  ✓ 避免网络访问
  ✓ 加载速度更快
  ✓ 不依赖缓存布局
```

### 步骤 2: 单任务测试（推荐）

在重新提交全部任务前，先测试一个任务：

```bash
# 测试最小的模型（Qwen 0.5B）
sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm
```

**检查日志**：
```bash
# 等待 1-2 分钟后检查
tail -f outputs/logs/kava_qwen05b_aug_*.out
tail -f outputs/logs/kava_qwen05b_aug_*.err
```

**成功标志**：
- ✅ 日志显示 "Loading model from /home/share/models/..."
- ✅ 无 "Network is unreachable" 错误
- ✅ 无 "Cannot connect to huggingface.co" 错误
- ✅ 训练开始（显示 epoch 0, step 0 等）

**失败标志**：
- ❌ 仍有网络错误 → 检查环境变量是否正确设置
- ❌ "FileNotFoundError" → 检查共享库路径和文件完整性
- ❌ "ImportError" → 检查 venv 是否正确激活

---

## 🚀 重新提交所有任务

### 清理旧日志（可选）

```bash
cd "/home/rpwang/kava review"

# 备份旧日志
mkdir -p outputs/logs_backup_$(date +%Y%m%d_%H%M%S)
mv outputs/logs/*.out outputs/logs/*.err outputs/logs_backup_* 2>/dev/null || true

# 或直接删除
rm -f outputs/logs/kava_*.out outputs/logs/kava_*.err
```

### 提交所有任务

```bash
cd "/home/rpwang/kava review"
bash submit_all_jobs.sh
```

**预期输出**：
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KAVA 训练任务批量提交
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/3] 验证 HPC 共享模型库...
✓ Llama-3.2-1B-Instruct 已找到
✓ Llama-3.2-3B-Instruct 已找到
✓ Qwen2.5-0.5B-Instruct 已找到

[2/3] 提交训练任务...
提交配置: llama1b_aug
  任务 ID: 20110 (3 个子任务: 种子 42, 123, 456)
提交配置: llama1b_aug_nl
  任务 ID: 20111 (3 个子任务: 种子 42, 123, 456)
提交配置: llama3b_aug
  任务 ID: 20112 (3 个子任务: 种子 42, 123, 456)
提交配置: qwen05b_aug
  任务 ID: 20113 (3 个子任务: 种子 42, 123, 456)

总计: 4 个主任务，12 个子任务

[3/3] 生成辅助脚本...
✓ monitor_jobs.sh
✓ collect_results.sh
```

---

## 📊 监控任务

### 自动刷新监控（推荐）

```bash
bash monitor_jobs.sh --auto
```

每 30 秒自动更新，显示：
- 任务状态统计
- 进度百分比
- 最新日志片段
- GPU 使用情况

### 手动检查

```bash
# 查看队列
squeue --me

# 查看任务历史
sacct -j 20110,20111,20112,20113 --format=JobID,JobName,State,ExitCode,Start,Elapsed

# 查看最新日志
tail -f outputs/logs/kava_*.out

# 查看错误日志
tail -f outputs/logs/kava_*.err
```

---

## 🐛 常见问题

### Q1: 仍然出现 "Network is unreachable"

**检查**：
```bash
# 1. 检查配置文件是否已更新
grep "name:" configs/*.yaml

# 应该看到本地路径：
# configs/llama1b_aug.yaml:  name: "/home/share/models/Llama-3.2-1B-Instruct"
# configs/qwen05b_aug.yaml:  name: "/home/share/models/Qwen2.5-0.5B-Instruct"

# 2. 检查 SLURM 脚本环境变量
grep "OFFLINE" submit_multi_seed.slurm

# 应该看到：
# export HUGGINGFACE_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
```

### Q2: "FileNotFoundError: config.json not found"

**原因**：共享模型库中模型文件不完整

**检查**：
```bash
ls -lh /home/share/models/Llama-3.2-1B-Instruct/
ls -lh /home/share/models/Qwen2.5-0.5B-Instruct/

# 必需文件：
# - config.json
# - tokenizer.json 或 tokenizer_config.json
# - *.safetensors 或 *.bin (模型权重)
```

**解决**：联系 HPC 管理员补充完整模型文件

### Q3: 数据集下载失败

**症状**：`datasets` 库尝试下载 GSM8K 数据集

**解决**：数据集可以联网下载（在登录节点预下载）：
```bash
# 在登录节点运行（有网络）
cd "/home/rpwang/kava review"
source venv/bin/activate

python -c "
from datasets import load_dataset
# 预下载数据集到个人缓存
dataset = load_dataset('whynlp/gsm8k-aug')
print('✓ GSM8K-AUG 数据集已缓存')
"
```

---

## 📈 预期结果

**时间线**：
- 0-5 分钟：任务进入 PENDING 状态
- 5-30 分钟：任务开始 RUNNING，模型加载完成
- 每个 epoch：1-4 小时（取决于模型大小）
- 总训练时间：12-36 小时（Qwen 最快，Llama-3B 最慢）

**成功指标**（日志中应该看到）：
```
✓ Loading model from /home/share/models/...
✓ Model loaded successfully
✓ Training started
✓ Epoch 0 | Step 0 | Loss: ...
✓ Validation EM: ... | F1: ...
```

**失败指标**（不应该看到）：
```
✗ Network is unreachable
✗ Cannot connect to huggingface.co
✗ Repository not found
✗ 401/403 Client Error
```

---

## 📞 需要帮助？

如果问题仍未解决，请提供：
1. `sacct` 输出
2. 最新的 `.out` 和 `.err` 日志文件内容
3. `quick_model_test.py` 的输出

---

## ✨ 修复总结

| 问题 | 修复 | 文件 |
|------|------|------|
| 使用 HF repo ID | 改为本地路径 | `configs/*.yaml` |
| 尝试联网访问 | 强制离线模式 | `submit_multi_seed.slurm` |
| 缺少诊断工具 | 添加验证脚本 | `quick_model_test.py` |

**核心改变**：
```yaml
# 改前
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"

# 改后  
model:
  name: "/home/share/models/Llama-3.2-1B-Instruct"
```

```bash
# 新增环境变量
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

这两个改动确保：
✅ transformers 直接从本地加载  
✅ 不尝试任何网络访问  
✅ 快速失败（如果文件不存在）  
✅ 与 HPC 环境完全兼容
