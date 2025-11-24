# KAVA 项目脚本总览

本文档提供所有脚本的快速参考和使用示例。

## 核心训练脚本

### 1. `train.py` - 主训练脚本

训练 KAVA 模型的主要入口。

```bash
# 基础训练
python train.py --config configs/llama1b_aug.yaml

# 快速测试（100 样本）
python train.py --config configs/llama1b_aug.yaml --max_train_samples 100 --num_epochs 1

# 自定义输出目录和种子
python train.py --config configs/llama1b_aug.yaml --output_dir outputs/my_exp --seed 42

# 启用 W&B 日志
python train.py --config configs/llama1b_aug.yaml --use_wandb
```

**关键参数**:
- `--config`: 配置文件路径（必需）
- `--output_dir`: 输出目录（默认：`outputs`）
- `--seed`: 随机种子（默认：42）
- `--max_train_samples`: 限制训练样本数（用于快速测试）
- `--num_epochs`: 训练轮数（覆盖配置文件）
- `--use_wandb`: 启用 Weights & Biases 日志

## 评估脚本

### 2. `evaluate.py` - 模型评估

评估训练好的模型在测试集上的性能。

```bash
# 评估 GSM8k
python evaluate.py --checkpoint_dir outputs/best_checkpoint \
                   --eval_dataset gsm8k \
                   --output outputs/results_gsm8k.yaml

# 评估 GSM8k-Hard
python evaluate.py --checkpoint_dir outputs/best_checkpoint \
                   --eval_dataset gsm8k-hard \
                   --output outputs/results_hard.yaml

# 评估 SVAMP
python evaluate.py --checkpoint_dir outputs/best_checkpoint \
                   --eval_dataset svamp \
                   --output outputs/results_svamp.yaml

# 限制样本数（快速测试）
python evaluate.py --checkpoint_dir outputs/best_checkpoint \
                   --eval_dataset gsm8k \
                   --max_samples 100
```

**输出格式**:
```yaml
dataset: gsm8k
accuracy: 0.5234
exact_match: 0.5234
total_samples: 1319
correct: 690
avg_forward_passes: 6.2
forward_passes_std: 1.3
```

**关键参数**:
- `--checkpoint_dir`: 模型检查点目录（必需）
- `--eval_dataset`: 评估数据集（gsm8k, gsm8k-hard, svamp）
- `--output`: 结果输出路径（支持 .yaml 和 .json）
- `--max_samples`: 限制评估样本数
- `--seed`: 随机种子

### 3. `inference.py` - 交互式推理

单样本推理和批量预测。

```bash
# 交互式推理
python inference.py --checkpoint_dir outputs/best_checkpoint --interactive

# 单问题推理
python inference.py --checkpoint_dir outputs/best_checkpoint \
                   --question "If John has 5 apples and buys 3 more, how many does he have?"

# 批量推理
python inference.py --checkpoint_dir outputs/best_checkpoint \
                   --input_file questions.txt \
                   --output_file answers.txt
```

**关键参数**:
- `--checkpoint_dir`: 模型检查点目录（必需）
- `--interactive`: 交互式模式
- `--question`: 单个问题（非交互式）
- `--input_file`: 输入文件（每行一个问题）
- `--output_file`: 输出文件
- `--num_latent_tokens`: Latent tokens 数量（默认：24）
- `--num_iterations`: Jacobi 迭代次数（默认：3）

## 实验自动化

### 4. `run_multi_seed.ps1` - 多种子实验（推荐）

**PowerShell 自动化脚本**，包含训练、评估和聚合。

```powershell
# 基础用法
.\run_multi_seed.ps1 -Config llama1b_aug

# 自定义种子
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456

# 指定输出目录
.\run_multi_seed.ps1 -Config llama1b_aug -OutputDir "my_experiments"

# 启用 W&B
.\run_multi_seed.ps1 -Config llama1b_aug -UseWandB

# 跳过训练（仅评估）
.\run_multi_seed.ps1 -Config llama1b_aug -SkipTraining

# 跳过评估（仅训练）
.\run_multi_seed.ps1 -Config llama1b_aug -SkipEvaluation
```

**可用配置**:
- `llama1b_aug`: LLaMA-3.2-1B + GSM8k-AUG
- `llama1b_aug_nl`: LLaMA-3.2-1B + GSM8k-AUG-NL
- `qwen05b_aug`: Qwen2.5-0.5B + GSM8k-AUG
- `llama3b_aug`: LLaMA-3.2-3B + GSM8k-AUG

**输出结构**:
```
outputs/llama1b_aug_multi_seed/
├── seed_42/
│   ├── best_checkpoint/
│   ├── results_gsm8k.json
│   ├── results_gsm8k-hard.json
│   └── results_svamp.json
├── seed_123/
├── seed_456/
├── aggregated_results.json
└── aggregated_results.yaml
```

### 5. `run_multi_seed.py` - Python 版本多种子脚本

等效的 Python 脚本（如果无法使用 PowerShell）。

```bash
# 基础用法
python run_multi_seed.py --config configs/llama1b_aug.yaml --seeds 42 123 456

# 指定输出目录
python run_multi_seed.py --config configs/llama1b_aug.yaml \
                         --seeds 42 123 456 \
                         --output_dir my_experiments

# 跳过评估
python run_multi_seed.py --config configs/llama1b_aug.yaml \
                         --seeds 42 123 456 \
                         --skip_eval
```

## 结果聚合

### 6. `aggregate_multi_seed.py` - 统计聚合

聚合多个种子的结果，计算 mean ± std。

```bash
# 基础用法
python aggregate_multi_seed.py \
    --seed_dirs outputs/seed_42 outputs/seed_123 outputs/seed_456 \
    --datasets gsm8k gsm8k-hard svamp \
    --model_name "KAVA-LLaMA-1B"

# 保存到文件
python aggregate_multi_seed.py \
    --seed_dirs outputs/seed_42 outputs/seed_123 outputs/seed_456 \
    --datasets gsm8k gsm8k-hard svamp \
    --model_name "KAVA-LLaMA-1B" \
    --output_json results.json \
    --output_yaml results.yaml
```

**输出示例**:
```
================================================================================
Dataset              Accuracy (mean±std)            Forward Passes (mean±std)     
================================================================================
gsm8k                52.34 ± 0.89                   6.2 ± 0.3                     
gsm8k-hard           31.67 ± 1.25                   8.5 ± 0.4                     
svamp                58.91 ± 1.02                   5.8 ± 0.2                     
================================================================================
```

**LaTeX 输出**:
```latex
\begin{tabular}{l|cc|cc|cc}
\hline
Method & \multicolumn{2}{c|}{GSM8k} & \multicolumn{2}{c|}{GSM8k-Hard} & \multicolumn{2}{c}{SVAMP} \\
 & Acc & FP & Acc & FP & Acc & FP \\
\hline
KAVA-LLaMA-1B & 52.3 (0.9) & 6.2 (0.3) & 31.7 (1.3) & 8.5 (0.4) & 58.9 (1.0) & 5.8 (0.2) \\
\hline
\end{tabular}
```

## 测试和验证

### 7. `smoke_test.py` - 烟雾测试

快速验证所有组件是否正常工作（**运行任何实验前必须执行**）。

```bash
python smoke_test.py
```

**测试项目**:
1. ✓ R-KV 压缩算法
2. ✓ 损失函数（KV, CODI, KAVA）
3. ✓ Latent Reasoning 模块
4. ✓ 数据加载和答案提取
5. ✓ 配置文件完整性
6. ✓ 目录结构

**预期输出**:
```
============================================================
Test Results: 6 passed, 0 failed
============================================================
✅ All smoke tests PASSED!
```

**运行时间**: ~2 分钟（无需 GPU）

## 工作流推荐

### 快速验证流程（15 分钟）

```bash
# 1. 烟雾测试（2 分钟）
python smoke_test.py

# 2. 快速训练测试（10 分钟）
python train.py --config configs/llama1b_aug.yaml --max_train_samples 100 --num_epochs 1

# 3. 快速评估（3 分钟）
python evaluate.py --checkpoint_dir outputs/best_checkpoint --eval_dataset gsm8k --max_samples 100
```

### 单种子完整实验（3 小时）

```bash
# 1. 训练（2.5 小时）
python train.py --config configs/llama1b_aug.yaml --seed 42

# 2. 评估 3 个数据集（30 分钟）
python evaluate.py --checkpoint_dir outputs/best_checkpoint --eval_dataset gsm8k
python evaluate.py --checkpoint_dir outputs/best_checkpoint --eval_dataset gsm8k-hard
python evaluate.py --checkpoint_dir outputs/best_checkpoint --eval_dataset svamp
```

### 多种子统计实验（9 小时）

```powershell
# 一键运行所有步骤
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456
```

等价于：
```bash
# 训练 3 个种子
python train.py --config configs/llama1b_aug.yaml --seed 42 --output_dir outputs/seed_42
python train.py --config configs/llama1b_aug.yaml --seed 123 --output_dir outputs/seed_123
python train.py --config configs/llama1b_aug.yaml --seed 456 --output_dir outputs/seed_456

# 评估每个种子（3 个数据集）
for seed in 42 123 456; do
    python evaluate.py --checkpoint_dir outputs/seed_$seed/best_checkpoint --eval_dataset gsm8k --output outputs/seed_$seed/results_gsm8k.yaml
    python evaluate.py --checkpoint_dir outputs/seed_$seed/best_checkpoint --eval_dataset gsm8k-hard --output outputs/seed_$seed/results_hard.yaml
    python evaluate.py --checkpoint_dir outputs/seed_$seed/best_checkpoint --eval_dataset svamp --output outputs/seed_$seed/results_svamp.yaml
done

# 聚合结果
python aggregate_multi_seed.py --seed_dirs outputs/seed_42 outputs/seed_123 outputs/seed_456
```

### 完整论文复现（48 小时）

```powershell
# LLaMA-3.2-1B (两个数据集)
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456
.\run_multi_seed.ps1 -Config llama1b_aug_nl -Seeds 42,123,456

# Qwen2.5-0.5B
.\run_multi_seed.ps1 -Config qwen05b_aug -Seeds 42,123,456

# LLaMA-3.2-3B
.\run_multi_seed.ps1 -Config llama3b_aug -Seeds 42,123,456
```

## 常见问题排查

### 内存不足

```bash
# 降低 batch size（修改配置文件或命令行）
python train.py --config configs/llama1b_aug.yaml --batch_size 2

# 启用梯度累积
python train.py --config configs/llama1b_aug.yaml --gradient_accumulation_steps 4
```

### CUDA 错误

```bash
# 清空 CUDA 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 检查 GPU 可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 模型下载慢

```bash
# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用本地缓存
export HF_HOME=/path/to/cache
```

### 评估结果不一致

```bash
# 确保使用相同的种子
python evaluate.py --checkpoint_dir outputs/best_checkpoint --eval_dataset gsm8k --seed 42

# 检查 forward pass 统计是否稳定
# 如果 avg_forward_passes 变化很大，可能是生成参数问题
```

## 脚本依赖关系

```
smoke_test.py (独立，无依赖)
    ↓
train.py (生成 checkpoints)
    ↓
evaluate.py (生成 metrics)
    ↓
aggregate_multi_seed.py (聚合统计)

或使用自动化脚本：
run_multi_seed.ps1 (调用上述所有脚本)
```

## 相关文档

- **[QUICK_VALIDATION.md](./QUICK_VALIDATION.md)** - 7 步验证指南（2 分钟到 48 小时）
- **[MULTI_SEED_GUIDE.md](./MULTI_SEED_GUIDE.md)** - 详细的多种子实验说明
- **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** - 训练超参数和技巧
- **[EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md)** - 评估方法和指标解释
- **[README.md](../README.md)** - 项目总览

## 快速参考表

| 任务                     | 命令                                                                 | 时间   |
|--------------------------|----------------------------------------------------------------------|--------|
| 烟雾测试                 | `python smoke_test.py`                                               | 2 分钟 |
| 快速训练                 | `python train.py --config configs/llama1b_aug.yaml --max_train_samples 100` | 10 分钟 |
| 完整训练                 | `python train.py --config configs/llama1b_aug.yaml`                  | 2.5 小时 |
| 单数据集评估             | `python evaluate.py --checkpoint_dir outputs/best_checkpoint --eval_dataset gsm8k` | 10 分钟 |
| 多种子实验（自动）       | `.\run_multi_seed.ps1 -Config llama1b_aug`                           | 9 小时 |
| 交互式推理               | `python inference.py --checkpoint_dir outputs/best_checkpoint --interactive` | 即时   |
| 聚合结果                 | `python aggregate_multi_seed.py --seed_dirs outputs/seed_*`          | 1 分钟 |

---

**提示**: 始终先运行 `python smoke_test.py` 确保环境配置正确！
