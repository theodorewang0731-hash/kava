# 给 HPC AI 助手的简洁指令

## 🎯 快速执行（推荐方案）

### 方案一：先验证后提交（最稳妥）

```bash
# 步骤 1: 验证修复
cd "/home/rpwang/kava review"
bash INSTRUCTIONS_FOR_AI.sh
# 选择 'y' 运行验证测试

# 步骤 2: 单任务测试
# 选择 'y' 提交单任务测试
# 等待 2-3 分钟查看日志

# 步骤 3: 确认成功后提交全部
bash submit_all_jobs.sh
```

### 方案二：一步到位（快速但需要确认）

```bash
cd "/home/rpwang/kava review"
bash submit_all_jobs.sh
```

---

## 📋 已完成的修复

### ✅ 修改 1: 配置文件使用本地路径

所有配置文件已更新：
```yaml
# 改前: "meta-llama/Llama-3.2-1B-Instruct"
# 改后: "/home/share/models/Llama-3.2-1B-Instruct"
```

**文件列表**：
- `configs/llama1b_aug.yaml`
- `configs/llama1b_aug_nl.yaml`
- `configs/llama3b_aug.yaml`
- `configs/qwen05b_aug.yaml`

### ✅ 修改 2: SLURM 脚本强制离线

`submit_multi_seed.slurm` 已添加：
```bash
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

### ✅ 新增工具

1. **quick_model_test.py** - 诊断脚本
   - 验证模型是否能从本地加载
   - 测试 3 种加载方式
   - 给出明确建议

2. **FIX_NETWORK_ERROR.md** - 完整文档
   - 问题分析
   - 修复步骤
   - 常见问题 FAQ
   - 预期结果

3. **INSTRUCTIONS_FOR_AI.sh** - 交互式执行脚本
   - 引导式操作
   - 验证 → 测试 → 提交
   - 自动检查日志

---

## 🔍 验证命令（推荐先运行）

```bash
cd "/home/rpwang/kava review"
source venv/bin/activate

# 设置环境（与 SLURM 一致）
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
  ✓ 避免网络访问
  ✓ 加载速度更快
  ✓ 不依赖缓存布局
```

---

## 🚀 提交任务

### 选择 A: 单任务测试（推荐）

```bash
# 测试 Qwen 0.5B（最小最快）
sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm

# 等待 2-3 分钟
squeue --me

# 查看日志（应该看到模型加载成功）
tail -n 100 outputs/logs/kava_qwen05b_aug_*.out
```

### 选择 B: 提交全部（验证通过后）

```bash
bash submit_all_jobs.sh
bash monitor_jobs.sh --auto
```

---

## 📊 监控和日志

```bash
# 自动刷新监控（每 30 秒）
bash monitor_jobs.sh --auto

# 查看队列
squeue --me

# 查看任务历史
sacct -u $USER -S today

# 查看最新日志
ls -lht outputs/logs/ | head -20
tail -f outputs/logs/kava_*.out
```

---

## ✅ 成功标志

日志应该显示：
```
✓ Loading model from /home/share/models/Llama-3.2-1B-Instruct
✓ Model loaded successfully
✓ Training started
✓ Epoch 0 | Step 0 | Loss: 2.xxx
```

**不应该看到**：
```
✗ Network is unreachable
✗ Cannot connect to huggingface.co
✗ [Errno 101]
```

---

## 🐛 如果仍失败

检查清单：
```bash
# 1. 配置文件是否使用本地路径
grep "name:" configs/*.yaml

# 应该看到：/home/share/models/...

# 2. SLURM 脚本是否设置离线
grep "OFFLINE" submit_multi_seed.slurm

# 应该看到：HUGGINGFACE_HUB_OFFLINE=1

# 3. 共享模型是否完整
ls -lh /home/share/models/Llama-3.2-1B-Instruct/
ls -lh /home/share/models/Qwen2.5-0.5B-Instruct/

# 应该有：config.json, tokenizer.json, *.safetensors
```

---

## 📞 报告格式（如需帮助）

如果问题未解决，请提供：

1. **验证脚本输出**：
```bash
python quick_model_test.py > test_output.txt 2>&1
cat test_output.txt
```

2. **任务状态**：
```bash
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed
```

3. **最新日志**（前后 100 行）：
```bash
tail -n 100 outputs/logs/kava_*.out
tail -n 100 outputs/logs/kava_*.err
```

---

## 核心改变总结

| 问题 | 原因 | 修复 |
|------|------|------|
| Network unreachable | 使用 HF repo ID | 改用本地路径 |
| 尝试联网 | transformers 默认行为 | 强制离线模式 |
| 失败过快 | 网络超时重试 | 立即从本地加载 |

**关键修改**：
```diff
# configs/*.yaml
- name: "meta-llama/Llama-3.2-1B-Instruct"
+ name: "/home/share/models/Llama-3.2-1B-Instruct"

# submit_multi_seed.slurm
+ export HUGGINGFACE_HUB_OFFLINE=1
+ export TRANSFORMERS_OFFLINE=1
```

这确保 transformers 只从本地加载，不尝试任何网络访问！
