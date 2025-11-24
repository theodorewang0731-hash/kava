# 多种子实验指南

本指南说明如何运行多种子实验以获得统计显著的结果。

## 快速开始

### 1. 运行烟雾测试（2 分钟）

在运行完整实验前，先验证所有组件：

```powershell
python smoke_test.py
```

预期输出：
```
✅ All smoke tests passed!
```

### 2. 运行多种子实验

使用 PowerShell 脚本自动化整个流程：

```powershell
# 基础用法：使用默认配置和种子
.\run_multi_seed.ps1 -Config llama1b_aug

# 自定义种子
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456

# 指定输出目录
.\run_multi_seed.ps1 -Config llama1b_aug -OutputDir "./my_experiments"

# 启用 W&B 日志
.\run_multi_seed.ps1 -Config llama1b_aug -UseWandB
```

### 3. 查看结果

脚本会自动生成聚合结果：

```powershell
# 查看汇总表格
cat outputs/llama1b_aug_multi_seed/aggregated_results.yaml

# 查看 JSON 格式（便于程序解析）
cat outputs/llama1b_aug_multi_seed/aggregated_results.json
```

## 可用配置

| 配置名称          | 模型              | 数据集        | 说明                    |
|-------------------|-------------------|---------------|-------------------------|
| `llama1b_aug`     | LLaMA-3.2-1B      | GSM8k-AUG     | 标准 CoT 格式           |
| `llama1b_aug_nl`  | LLaMA-3.2-1B      | GSM8k-AUG-NL  | 自然语言 CoT            |
| `qwen05b_aug`     | Qwen2.5-0.5B      | GSM8k-AUG     | 轻量级模型              |
| `llama3b_aug`     | LLaMA-3.2-3B      | GSM8k-AUG     | 大型模型（需要更多显存）|

## 输出结构

```
outputs/llama1b_aug_multi_seed/
├── seed_42/
│   ├── best_checkpoint/          # 最佳模型检查点
│   ├── results_gsm8k.json        # GSM8k 评估结果
│   ├── results_gsm8k-hard.json   # GSM8k-Hard 评估结果
│   └── results_svamp.json        # SVAMP 评估结果
├── seed_123/
│   └── ...
├── seed_456/
│   └── ...
├── aggregated_results.json       # 聚合统计（JSON）
└── aggregated_results.yaml       # 聚合统计（YAML）
```

## 手动聚合结果

如果需要手动聚合结果：

```bash
python aggregate_multi_seed.py \
    --seed_dirs outputs/llama1b_aug_multi_seed/seed_42 \
                outputs/llama1b_aug_multi_seed/seed_123 \
                outputs/llama1b_aug_multi_seed/seed_456 \
    --datasets gsm8k gsm8k-hard svamp \
    --model_name "KAVA-LLaMA-1B" \
    --output_json results.json \
    --output_yaml results.yaml
```

## 高级用法

### 跳过训练（仅评估）

如果已经训练完成，只想重新评估：

```powershell
.\run_multi_seed.ps1 -Config llama1b_aug -SkipTraining
```

### 跳过评估（仅训练）

如果只想训练模型：

```powershell
.\run_multi_seed.ps1 -Config llama1b_aug -SkipEvaluation
```

### 单独运行每个步骤

也可以手动运行每个步骤：

```bash
# 1. 训练单个种子
python train.py --config configs/llama1b_aug.yaml --output_dir outputs/seed_42 --seed 42

# 2. 评估单个种子
python evaluate.py --checkpoint_dir outputs/seed_42/best_checkpoint \
                   --eval_dataset gsm8k \
                   --output outputs/seed_42/results_gsm8k.yaml

# 3. 聚合结果
python aggregate_multi_seed.py --seed_dirs outputs/seed_42 outputs/seed_123 outputs/seed_456
```

## 预期运行时间

基于 A100 40GB GPU：

- **单个种子训练**: 2-3 小时（LLaMA-1B, 10 epochs）
- **单个数据集评估**: 5-10 分钟（~1,300 samples）
- **3 种子完整实验**: 6-9 小时

## 结果解读

聚合脚本会输出两种格式的表格：

### 1. 汇总表格（控制台）

```
================================================================================
Dataset              Accuracy (mean±std)            Forward Passes (mean±std)     
================================================================================
gsm8k                52.34 ± 0.89                   6.2 ± 0.3                     
gsm8k-hard           31.67 ± 1.25                   8.5 ± 0.4                     
svamp                58.91 ± 1.02                   5.8 ± 0.2                     
================================================================================
```

### 2. LaTeX 表格

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

直接复制到论文中！

## 常见问题

### Q: 为什么需要多种子？

A: 单次运行可能受随机初始化影响，多种子可提供统计置信度。论文中通常报告 mean ± std。

### Q: 3 个种子够吗？

A: 对于初步验证够用。如果要发表论文，建议使用 5+ 种子。

### Q: 如何选择种子值？

A: 使用不同数量级的值（如 42, 123, 456）可以更好地覆盖随机性空间。

### Q: 内存不足怎么办？

A: 
- 降低 batch size（修改 YAML 配置）
- 使用梯度累积（gradient_accumulation_steps）
- 尝试较小模型（qwen05b_aug）

### Q: 如何并行运行多个种子？

A: 如果有多张 GPU，可以同时启动多个脚本：

```powershell
# Terminal 1
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42

# Terminal 2
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 123

# Terminal 3
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 456
```

## 论文复现清单

完整复现 Table 6 的所有实验：

```powershell
# LLaMA-3.2-1B (两个数据集)
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456
.\run_multi_seed.ps1 -Config llama1b_aug_nl -Seeds 42,123,456

# Qwen2.5-0.5B
.\run_multi_seed.ps1 -Config qwen05b_aug -Seeds 42,123,456

# LLaMA-3.2-3B
.\run_multi_seed.ps1 -Config llama3b_aug -Seeds 42,123,456
```

预计总时间：**24-48 小时**（使用单个 A100 GPU）

## 相关文档

- [QUICK_VALIDATION.md](./QUICK_VALIDATION.md) - 快速验证指南（2 分钟到 48 小时）
- [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) - 详细评估说明
- [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) - 训练超参数调优
- [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) - 实现细节和设计决策
