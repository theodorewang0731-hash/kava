# KAVA 论文参数合规性报告

**论文**: Latent Reasoning via Compressed KV-Cache Distillation (arXiv:2510.02312v1)  
**验证日期**: 2025年11月20日  
**状态**: ✅ 完全符合

---

## 📋 Table 6 超参数完整对照

### 1. LoRA 配置（所有模型/数据集统一）

| 参数 | 论文值 | 实现值 | 状态 |
|------|--------|--------|------|
| **rank (r)** | 128 | 128 | ✅ |
| **alpha (α)** | 32 | 32 | ✅ |
| **dropout** | 0.1 | 0.1 | ✅ |
| **target_modules** | q/k/v/o_proj | q/k/v/o_proj | ✅ |

**说明**: 所有模型只微调 LoRA 模块，backbone 冻结。

---

### 2. Latent CoT (PCCoT) 参数（所有配置统一）

| 参数 | 论文值 | 实现值 | 状态 |
|------|--------|--------|------|
| **M (latent tokens)** | 24 | 24 | ✅ |
| **T (Jacobi iterations)** | 3 | 3 | ✅ |

**说明**: 24 个 latent token 并行生成，Jacobi 迭代 3 次。

---

### 3. LLaMA3.2-1B-Instruct 参数

#### 3.1 LLaMA-1B + GSM8k-AUG (Equation-only)

| 参数 | 论文 Table 6 | 配置文件 | 状态 |
|------|--------------|----------|------|
| **α₁ (CODI)** | 10.0 | 10.0 | ✅ |
| **α₂ (KV)** | 1.0 | 1.0 | ✅ |
| **KV loss** | Smooth L1 | smooth_l1 | ✅ |
| **Layer-wise std** | True | true | ✅ |
| **R-KV λ** | 0.1 | 0.1 | ✅ |
| **Use Projection** | True | true | ✅ |
| **Learning rate** | 8e-4 | 8.0e-4 | ✅ |
| **LR scheduler** | Cosine | cosine | ✅ |
| **Optimizer** | AdamW | adamw | ✅ |
| **Batch size** | 128 | 128 | ✅ |
| **Weight decay** | 0.1 | 0.1 | ✅ |
| **Gradient clipping** | 2.0 | 2.0 | ✅ |
| **Epochs** | 10 | 10 | ✅ |

**配置文件**: `configs/llama1b_aug.yaml`

#### 3.2 LLaMA-1B + GSM8k-AUG-NL (Natural Language)

| 参数 | 论文 Table 6 | 配置文件 | 状态 |
|------|--------------|----------|------|
| **α₁ (CODI)** | 10.0 | 10.0 | ✅ |
| **α₂ (KV)** | 1.0 | 1.0 | ✅ |
| **KV loss** | MSE | mse | ✅ |
| **Layer-wise std** | True | true | ✅ |
| **R-KV λ** | 0.1 | 0.1 | ✅ |
| **Use Projection** | True | true | ✅ |
| **Learning rate** | 8e-4 | 8.0e-4 | ✅ |
| **Weight decay** | 0.1 | 0.1 | ✅ |
| **Epochs** | 10 | 10 | ✅ |

**配置文件**: `configs/llama1b_aug_nl.yaml`

---

### 4. LLaMA3.2-3B-Instruct 参数

#### 4.1 LLaMA-3B + GSM8k-AUG

| 参数 | 论文 Table 6 | 配置文件 | 状态 |
|------|--------------|----------|------|
| **α₁ (CODI)** | 20.0 | 20.0 | ✅ |
| **α₂ (KV)** | 2.0 | 2.0 | ✅ |
| **KV loss** | Smooth L1 | smooth_l1 | ✅ |
| **Layer-wise std** | False | false | ✅ |
| **R-KV λ** | 0.1 | 0.1 | ✅ |
| **Use Projection** | True | true | ✅ |
| **Learning rate** | 2e-4 | 2.0e-4 | ✅ |
| **Batch size** | 128 | 128 | ✅ |
| **Weight decay** | 0.1 | 0.1 | ✅ |
| **Gradient clipping** | 2.0 | 2.0 | ✅ |
| **Epochs** | 5 | 5 | ✅ |

**配置文件**: `configs/llama3b_aug.yaml`

#### 4.2 LLaMA-3B + GSM8k-AUG-NL

| 参数 | 论文 Table 6 | 配置文件 | 状态 |
|------|--------------|----------|------|
| **α₁ (CODI)** | 20.0 | 20.0 | ✅ |
| **α₂ (KV)** | 2.0 | 2.0 | ✅ |
| **KV loss** | Smooth L1 | smooth_l1 | ✅ |
| **Layer-wise std** | False | false | ✅ |
| **R-KV λ** | 0.0 | 0.0 | ✅ |
| **Use Projection** | False | false | ✅ |
| **Learning rate** | 2e-4 | 2.0e-4 | ✅ |
| **Weight decay** | 0.1 | 0.1 | ✅ |
| **Epochs** | 5 | 5 | ✅ |

**配置文件**: `configs/llama3b_aug_nl.yaml`

---

### 5. Qwen2.5-0.5B-Instruct 参数

#### 5.1 Qwen-0.5B + GSM8k-AUG

| 参数 | 论文 Table 6 | 配置文件 | 状态 |
|------|--------------|----------|------|
| **α₁ (CODI)** | 10.0 | 10.0 | ✅ |
| **α₂ (KV)** | 1.0 | 1.0 | ✅ |
| **KV loss** | MSE | mse | ✅ |
| **Layer-wise std** | False | false | ✅ |
| **R-KV λ** | 0.1 | 0.1 | ✅ |
| **Use Projection** | True | true | ✅ |
| **Learning rate** | 5e-4 | 5.0e-4 | ✅ |
| **Batch size** | 128 | 128 | ✅ |
| **Weight decay** | 0.01 | 0.01 | ✅ |
| **Gradient clipping** | 2.0 | 2.0 | ✅ |
| **Epochs** | 10 | 10 | ✅ |

**配置文件**: `configs/qwen05b_aug.yaml`

#### 5.2 Qwen-0.5B + GSM8k-AUG-NL

| 参数 | 论文 Table 6 | 配置文件 | 状态 |
|------|--------------|----------|------|
| **α₁ (CODI)** | 10.0 | 10.0 | ✅ |
| **α₂ (KV)** | 1.0 | 1.0 | ✅ |
| **KV loss** | MSE | mse | ✅ |
| **Layer-wise std** | True | true | ✅ |
| **R-KV λ** | 0.1 | 0.1 | ✅ |
| **Use Projection** | True | true | ✅ |
| **Learning rate** | 8e-4 | 8.0e-4 | ✅ |
| **Batch size** | 128 | 128 | ✅ |
| **Weight decay** | 0.1 | 0.1 | ✅ |
| **Gradient clipping** | 2.0 | 2.0 | ✅ |
| **Epochs** | 10 | 10 | ✅ |

**配置文件**: `configs/qwen05b_aug_nl.yaml`

---

## 📊 数据集参数

### 训练数据集

| 数据集 | 样本数 | CoT 类型 | 状态 |
|--------|--------|----------|------|
| **GSM8k-AUG** | ~385k | Equation-only | ✅ |
| **GSM8k-AUG-NL** | ~385k | Natural Language | ✅ |

**说明**: 两个数据集均由 GPT-4 从 GSM8k 扩展生成。

### 评测数据集

| 数据集 | 类型 | 状态 |
|--------|------|------|
| **GSM8k** | In-distribution | ✅ |
| **GSM8k-Hard** | Zero-shot | ✅ |
| **SVAMP** | Zero-shot | ✅ |

---

## 🔬 实验设置参数

| 参数 | 论文要求 | 实现 | 状态 |
|------|----------|------|------|
| **随机种子数** | 3 | 3 (42, 123, 456) | ✅ |
| **报告格式** | Mean ± Std | Mean ± Std | ✅ |
| **评估指标** | Accuracy + Forward passes | Accuracy + Forward passes | ✅ |
| **Ablation 基准模型** | LLaMA-1B | LLaMA-1B | ✅ |

---

## 🎯 配置文件清单

| 模型 | 数据集 | 配置文件 | 状态 |
|------|--------|----------|------|
| LLaMA-1B | AUG | `configs/llama1b_aug.yaml` | ✅ |
| LLaMA-1B | AUG-NL | `configs/llama1b_aug_nl.yaml` | ✅ |
| LLaMA-3B | AUG | `configs/llama3b_aug.yaml` | ✅ |
| LLaMA-3B | AUG-NL | `configs/llama3b_aug_nl.yaml` | ✅ |
| Qwen-0.5B | AUG | `configs/qwen05b_aug.yaml` | ✅ |
| Qwen-0.5B | AUG-NL | `configs/qwen05b_aug_nl.yaml` | ✅ |

**总计**: 6 个配置文件

---

## 🚀 运行计划

### 完整实验矩阵

- **模型数**: 3 (LLaMA-1B, LLaMA-3B, Qwen-0.5B)
- **数据集数**: 2 (AUG, AUG-NL)
- **配置数**: 3 × 2 = 6
- **每配置种子数**: 3
- **总训练任务数**: 6 × 3 = **18 个任务**

### 预计资源需求

| 模型 | 每 epoch 时间 | Epochs | 单次训练时间 | 3种子总时间 |
|------|--------------|--------|--------------|-------------|
| LLaMA-1B | ~30分钟 | 10 | ~5小时 | ~15小时 |
| LLaMA-3B | ~60分钟 | 5 | ~5小时 | ~15小时 |
| Qwen-0.5B | ~20分钟 | 10 | ~3.5小时 | ~10.5小时 |

**总预计时间**: ~40-48 小时（并行提交到 SLURM）

---

## ✅ 合规性检查清单

- [x] LoRA 参数完全一致 (r=128, α=32)
- [x] Latent CoT 参数完全一致 (M=24, T=3)
- [x] 损失函数权重完全一致 (α₁, α₂)
- [x] KV 损失类型完全一致 (Smooth L1 / MSE)
- [x] Layer-wise 标准化设置完全一致
- [x] R-KV λ 参数完全一致
- [x] Learning rate 完全一致
- [x] Optimizer 完全一致 (AdamW)
- [x] Batch size 完全一致 (128)
- [x] Weight decay 完全一致
- [x] Gradient clipping 完全一致 (2.0)
- [x] Epochs 完全一致
- [x] 数据集配置完全一致
- [x] 评测数据集完全一致
- [x] 随机种子设置完全一致 (3 seeds)

---

## 📝 关键差异说明

**无差异** - 所有参数严格遵循论文 Table 6 规范。

---

## 🔍 验证方法

```bash
# 验证所有配置文件参数
python smoke_test_lite.py

# 查看具体配置
cat configs/llama1b_aug.yaml
cat configs/llama1b_aug_nl.yaml
cat configs/llama3b_aug.yaml
cat configs/llama3b_aug_nl.yaml
cat configs/qwen05b_aug.yaml
cat configs/qwen05b_aug_nl.yaml
```

---

## 📚 论文引用

```bibtex
@article{shen2025kava,
  title={Latent Reasoning via Compressed KV-Cache Distillation},
  author={Shen and Wu},
  journal={arXiv preprint arXiv:2510.02312},
  year={2025}
}
```

---

**验证人**: GitHub Copilot  
**验证日期**: 2025年11月20日  
**合规状态**: ✅ 100% 符合论文规范
