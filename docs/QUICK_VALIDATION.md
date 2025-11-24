# 🚀 快速验证指南

在运行完整实验前，先进行快速验证确保所有组件正常工作。

---

## Step 1: 烟雾测试 (2 分钟)

验证所有核心组件：

```bash
python smoke_test.py
```

**预期输出**：
```
✅ All smoke tests passed!

You can now proceed with:
  1. Quick training test
  2. Full experiment
  3. Complete replication
```

如果失败，检查：
- 依赖安装：`pip install -r requirements.txt`
- 配置文件存在：`ls configs/`
- Python 版本：≥ 3.8

---

## Step 2: 快速训练测试 (5-10 分钟)

在少量数据上快速测试训练流程：

```bash
python train.py \
    --config configs/llama1b_aug.yaml \
    --seed 42 \
    --max_train_samples 100 \
    --max_eval_samples 20 \
    --num_epochs 1 \
    --output_dir experiments/quick_test
```

**预期输出**：
- 训练进度条运行无错误
- 损失值下降（loss < 5.0）
- 保存 checkpoint 到 `experiments/quick_test/`

**常见问题**：
- **OOM**：减少 batch size 到 1
- **数据集加载失败**：检查网络连接
- **模型下载慢**：设置 HF_ENDPOINT

---

## Step 3: 快速评估测试 (2-3 分钟)

测试评估流程（需要 checkpoint）：

```bash
python evaluate.py \
    --checkpoint experiments/quick_test/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --datasets gsm8k \
    --max_samples 20 \
    --output experiments/quick_test/eval_results.yaml
```

**预期输出**：
```
=== Evaluating on gsm8k ===
Progress: 100%|███████████| 20/20 [00:45<00:00]
Accuracy: 25.00% (5/20)
Avg Forward Passes: 48.2

Results saved to:
  - experiments/quick_test/eval_results.yaml
  - experiments/quick_test/eval_results.json
```

**注意**：快速测试的准确率较低是正常的（只训练了 100 样本）。

---

## Step 4: 推理测试 (1 分钟)

测试交互式推理：

```bash
python inference.py \
    --checkpoint experiments/quick_test/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --mode interactive
```

**交互示例**：
```
Question: What is 5 + 3?

Generating answer...

Answer: 5 + 3 = 8
Forward passes: 18

Question: /quit
```

---

## Step 5: 单种子完整实验 (2-3 小时)

验证完整训练流程：

```bash
python train.py \
    --config configs/llama1b_aug.yaml \
    --seed 42 \
    --wandb
```

**监控**：
- 训练时间：~2-3 小时 (LLaMA 1B on A100)
- 最终损失：< 1.0
- 保存 checkpoint

然后评估：

```bash
python evaluate.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch10 \
    --config configs/llama1b_aug.yaml \
    --datasets gsm8k gsm8k-hard svamp \
    --output experiments/llama1b_gsm8k-aug/seed_42/results.yaml
```

**预期结果**：
- GSM8k 准确率：~80-84%
- 前向传播数：~45-50

---

## Step 6: 多种子实验 (6-9 小时)

运行 3 个种子获得统计显著性：

```powershell
.\run_multi_seed_enhanced.ps1 -Config llama1b_aug -Seeds 42,43,44 -UseWandB
```

**或使用 Python 版本**：
```bash
python run_multi_seed.py \
    --config configs/llama1b_aug.yaml \
    --seeds 42 43 44 \
    --output_dir experiments
```

**预期输出**：
```
=== FINAL RESULTS ===
Dataset         Accuracy (%)              Forward Passes
gsm8k           82.45 ± 0.73              48.2 ± 1.1
gsm8k-hard      68.91 ± 1.24              52.7 ± 1.9
svamp           75.33 ± 0.89              45.1 ± 1.3

Results based on 3 seeds
```

---

## Step 7: 完整复现 (24-48 小时)

运行所有 4 个配置：

```powershell
.\run_all_experiments.ps1
```

这将运行：
- LLaMA 3.2-1B + GSM8k-AUG (3 seeds)
- LLaMA 3.2-1B + GSM8k-AUG-NL (3 seeds)
- Qwen2.5-0.5B + GSM8k-AUG (3 seeds)
- LLaMA 3.2-3B + GSM8k-AUG (3 seeds)

聚合结果：

```bash
python aggregate_results.py \
    --experiments_dir experiments \
    --output table1_results.csv
```

---

## 验证清单

完成每个步骤后打勾：

- [ ] **Step 1**: 烟雾测试通过
- [ ] **Step 2**: 快速训练完成（无错误）
- [ ] **Step 3**: 快速评估完成（有结果）
- [ ] **Step 4**: 推理正常工作
- [ ] **Step 5**: 单种子完整实验达到预期准确率
- [ ] **Step 6**: 多种子统计结果合理（std < 2%）
- [ ] **Step 7**: 完整复现与论文 Table 1 对齐（±1-2%）

---

## 常见问题

### 训练速度慢

**优化方案**：
```yaml
# 在 config YAML 中调整
training:
  per_device_train_batch_size: 4  # 增大 batch size
  gradient_accumulation_steps: 4  # 减少累积步数
  
system:
  mixed_precision: bf16  # 确保开启混合精度
```

### OOM 错误

**解决方案**：
```yaml
training:
  per_device_train_batch_size: 1  # 减小 batch size
  gradient_accumulation_steps: 16  # 增大累积步数保持有效 batch size
```

### 准确率低于预期

**检查项**：
1. 训练是否收敛（loss < 1.0）
2. 学习率是否合适（检查 W&B 曲线）
3. 数据集是否正确加载
4. 评估格式是否正确（答案提取）

### 数据集加载失败

**替代方案**：
```python
# 使用本地缓存
export HF_DATASETS_OFFLINE=1

# 或手动下载数据集
wget https://huggingface.co/datasets/whynlp/gsm8k-aug
```

---

## 下一步

完成验证后，您可以：

1. **调整超参数**：修改 configs/ 中的配置
2. **添加消融实验**：测试不同 loss 权重
3. **扩展到其他模型**：LLaMA 7B, Mistral 等
4. **评估其他数据集**：MATH, AQuA-RAT 等

参考文档：
- `docs/EXAMPLES.md` - 更多使用示例
- `docs/MULTI_SEED.md` - 多种子实验详细指南
- `docs/INFERENCE.md` - 推理使用指南
