# KAVA: Latent Reasoning via Compressed KV-Cache Distillation

[![Implementation](https://img.shields.io/badge/Implementation-Complete-brightgreen)](STATUS.md)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2510.02312-blue)](https://arxiv.org/abs/2510.02312)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](requirements.txt)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Strict Paper Implementation for Reproducibility**

This repository implements KAVA (Latent Reasoning via Compressed KV-Cache Distillation) following the exact specifications from the paper, including all hyperparameters from Table 6.

**🎉 Status**: ✅ Implementation complete with multi-seed automation, interactive inference, and comprehensive documentation.

## 📄 Paper Reference

**Title:** Latent Reasoning via Compressed KV-Cache Distillation  
**Authors:** Shen & Wu (2025)  
**ArXiv:** [2510.02312](https://arxiv.org/abs/2510.02312)

This implementation reproduces:
- **Table 1:** Accuracy on GSM8k, GSM8k-Hard, SVAMP
- **Table 2:** Average forward passes
- **Table 6:** All hyperparameter configurations

## 🎯 Key Features

✅ **Exact Paper Configuration**
- LLaMA 3.2-1B/3B-Instruct models
- Qwen2.5-0.5B-Instruct model
- GSM8k-AUG (equation-only CoT)
- GSM8k-AUG-NL (natural language CoT)

✅ **Core Components Implemented**
- **R-KV Compression:** Importance + Redundancy scoring
- **KV Distillation Loss:** Smooth L1 / MSE with layer-wise normalization
- **CODI Loss:** Hidden state distillation
- **PCCoT Latent Reasoning:** 24 tokens, 3 Jacobi iterations
- **LoRA Fine-tuning:** r=128, α=32

✅ **All Table 6 Hyperparameters**
- Loss weights (α₁, α₂)
- Learning rates (2e-4 to 8e-4)
- Loss types (Smooth L1 vs MSE)
- Layer-wise std normalization flags
- Projection layer flags

## 🚀 Quick Start

### ⚡ 最简单方式：Linux HPC 一键启动（推荐）

```bash
# 1. 上传代码到 HPC
scp -r kava/ user@hpc:/home/user/

# 2. SSH 登录并启动
ssh user@hpc
cd ~/kava
bash start.sh  # 自动验证+配置+下载+训练
```

**完成！** 仅需 3 步，脚本会自动处理所有事情。

**详细选项：**
```bash
bash start.sh --verify-only   # 仅验证环境
bash start.sh --method mirror # 使用中国镜像下载
bash start.sh --skip-download # 跳过模型下载（已缓存）
```

**🤖 使用 HPC AI 助手？** 查看这些引导文档：
- **[AI_ASSISTANT_PROMPT.md](AI_ASSISTANT_PROMPT.md)** - 完整的 AI 助手提示词
- **[PROMPT_FOR_AI.txt](PROMPT_FOR_AI.txt)** - 快速提示词（可直接复制）
- **[CONVERSATION_GUIDE.md](CONVERSATION_GUIDE.md)** - 分步对话脚本

---

### 🎯 一键复现（HPC 集群推荐）

**最快方式：** 只需一条命令即可完成所有配置和训练！

```bash
# 1. 上传代码到 HPC
scp -r kava/ user@hpc:/home/user/

# 2. 登录并运行自动化脚本
ssh user@hpc
cd ~/kava
bash run_reproduce.sh  # 自动完成：环境+模型下载+作业提交
```

**详细说明：** 参见 **[REPRODUCTION_CHECKLIST.md](REPRODUCTION_CHECKLIST.md)** 获取完整清单

---

### 📚 HPC 部署文档（按需阅读）

**如果你想了解细节或遇到问题**，参考以下文档：

1. **[REPRODUCTION_CHECKLIST.md](REPRODUCTION_CHECKLIST.md)** - 快速启动清单（5分钟）
2. **[GETTING_STARTED_HPC.md](docs/GETTING_STARTED_HPC.md)** - 完整 HPC 指南（30分钟）
3. **[KAVA_MODEL_DOWNLOAD.md](docs/KAVA_MODEL_DOWNLOAD.md)** - 模型下载详解（17-100分钟）

⚠️ **重要**：HPC 公共模型库**没有 KAVA 所需模型**（Llama-3.2, Qwen2.5）  
✅ **解决方案**：`run_reproduce.sh` 会自动下载到 `$HOME/.cache/huggingface`（~19GB）

---

### 🆕 新手必读：HPC 上手指南

**如果你是第一次在 HPC 上运行本项目**，请直接阅读：

👉 **[GETTING_STARTED_HPC.md](docs/GETTING_STARTED_HPC.md)** 👈

这个指南将带你完成：
1. ✅ 上传项目到 HPC（5 分钟）
2. ✅ 一键自动配置环境（15 分钟）
3. ✅ **自动下载所需模型**（17-100 分钟）
4. ✅ 提交训练任务（5 分钟）
5. ✅ 监控进度并生成论文结果（48 小时自动运行）

**总计**：30 分钟配置 + 模型下载 + 48 小时训练 → 得到论文 Table 1 & 2 结果！

---

### HPC Cluster Setup (手动方式)

如果需要手动配置而非使用 `run_reproduce.sh`：

**快速配置个人环境**:

```bash
# 配置个人 HuggingFace 缓存
cat >> ~/.bashrc << 'EOF'
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface
EOF

source ~/.bashrc

# 下载模型（17-100 分钟，使用代理可加速）
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

**提交训练任务**:

```bash
# 方法 1: 一键提交所有实验（4 配置 × 3 种子）
./hpc_run_all.sh

# 方法 2: 单个配置提交
sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm
```

**详细文档**：
- 🌟 新手指南: [GETTING_STARTED_HPC.md](GETTING_STARTED_HPC.md)
- 📦 模型下载: [KAVA_MODEL_DOWNLOAD.md](KAVA_MODEL_DOWNLOAD.md) ⚠️ **必读**
- 📖 HPC 参考: [HPC_REFERENCE.md](HPC_REFERENCE.md)
- 🔬 完整复现: [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)

---

### Installation

```powershell
# Install dependencies
pip install -r requirements.txt

# Verify installation (2 minutes, no GPU needed)
python smoke_test.py
```

### Multi-Seed Experiment (Recommended)

The fastest way to get statistical results:

```powershell
# One-line multi-seed training + evaluation + aggregation
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456
```

This will:
1. Train 3 models with different seeds (6-8 hours)
2. Evaluate on GSM8k, GSM8k-Hard, SVAMP (30 minutes)
3. Aggregate results with mean ± std statistics (1 minute)

See [Multi-Seed Guide](docs/MULTI_SEED_GUIDE.md) for details.

### Single Experiment

Train LLaMA 3.2-1B on GSM8k-AUG:

```powershell
python train.py --config configs/llama1b_aug.yaml --use_wandb
```

Evaluate trained model:

```powershell
python evaluate.py `
    --checkpoint_dir outputs/best_checkpoint `
    --eval_dataset gsm8k `
    --output results/gsm8k.yaml
```

### Interactive Inference

Test your trained model:

```powershell
python inference.py `
    --checkpoint_dir outputs/best_checkpoint `
    --interactive
```

**Example**:
```
Question: What is 25% of 80?
Answer: 25% of 80 is 20.
Forward passes: 6.2
```

See [Scripts Overview](docs/SCRIPTS_OVERVIEW.md) for all available commands.

### Full Paper Replication

Run all experiments (3 seeds × 4 configurations):

```powershell
# All Table 6 configurations
.\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456
.\run_multi_seed.ps1 -Config llama1b_aug_nl -Seeds 42,123,456
.\run_multi_seed.ps1 -Config qwen05b_aug -Seeds 42,123,456
.\run_multi_seed.ps1 -Config llama3b_aug -Seeds 42,123,456
```

Expected time: **24-48 hours** on A100 40GB GPU.

This will:
1. Train LLaMA 3.2-1B on GSM8k-AUG (3 seeds)
2. Train LLaMA 3.2-1B on GSM8k-AUG-NL (3 seeds)
3. Train Qwen2.5-0.5B on GSM8k-AUG (3 seeds)
4. Train LLaMA 3.2-3B on GSM8k-AUG (3 seeds)
5. Evaluate all checkpoints
6. Aggregate results with mean ± std

See [Multi-Seed Guide](docs/MULTI_SEED.md) for detailed instructions.

## 📊 Configuration Files

All configs strictly follow **Table 6** from the paper:

| Config | Model | Dataset | α₁ | α₂ | KV Loss | Layer-wise std | LR | Epochs |
|--------|-------|---------|----|----|---------|----------------|-----|--------|
| `llama1b_aug.yaml` | LLaMA-1B | AUG | 10 | 1 | Smooth L1 | ✓ | 8e-4 | 10 |
| `llama1b_aug_nl.yaml` | LLaMA-1B | AUG-NL | 10 | 1 | MSE | ✓ | 8e-4 | 10 |
| `qwen05b_aug.yaml` | Qwen-0.5B | AUG | 10 | 1 | MSE | ✗ | 5e-4 | 10 |
| `llama3b_aug.yaml` | LLaMA-3B | AUG | 20 | 2 | Smooth L1 | ✗ | 2e-4 | 5 |

## 🏗️ Architecture

### Directory Structure

```
kava review/
├── configs/              # Exact Table 6 configurations
│   ├── llama1b_aug.yaml
│   ├── llama1b_aug_nl.yaml
│   ├── qwen05b_aug.yaml
│   └── llama3b_aug.yaml
├── src/
│   ├── rkv_compression.py       # R-KV algorithm (Section 3.2)
│   ├── losses.py                # KV + CODI losses (Section 3)
│   ├── latent_reasoning.py      # PCCoT with Jacobi iterations
│   ├── data_utils.py            # GSM8k-AUG dataset loading
│   ├── evaluation_datasets.py   # GSM8k-Hard, SVAMP loaders
│   └── trainer.py               # Main training loop
├── docs/
│   ├── MULTI_SEED.md        # Multi-seed experiments guide
│   ├── INFERENCE.md         # Inference usage guide
│   ├── QUICKSTART.md        # Quick start tutorial
│   └── PAPER_MAPPING.md     # Paper section → code mapping
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation script
├── inference.py             # Interactive/batch inference
├── run_multi_seed.py        # Multi-seed automation
├── aggregate_results.py     # Results aggregation
├── run_all_experiments.ps1  # Full replication script
└── requirements.txt
```

### Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train.py` | Single training run | One config, one seed |
| `evaluate.py` | Evaluate checkpoint on GSM8k/GSM8k-Hard/SVAMP | After training |
| `inference.py` | Interactive testing or batch inference | Test trained model |
| `run_multi_seed.py` | Automated multi-seed training & eval | Statistical significance |
| `aggregate_results.py` | Combine results into paper tables | After all experiments |
| `run_all_experiments.ps1` | Run all 4 configs × 3 seeds | Full replication |

### Key Modules

**1. R-KV Compression (`rkv_compression.py`)**
```python
# Importance score: I = (1/N_A) Σ_j A_{j,i}
importance = compute_importance_score(attention, answer_idx, steps_idx)

# Redundancy score: R = softmax(-avg_cosine_similarity)
redundancy = compute_redundancy_score(keys)

# Mixed score: S = λ*I + (1-λ)*R
score = lambda * importance + (1 - lambda) * redundancy

# Select top-M tokens
compressed_kv = select_top_tokens(score, M=24)
```

**2. KV Distillation Loss (`losses.py`)**
```python
# L_KV = (1/2M) * (||K_t - K_s||_p + ||V_t - V_s||_p)
loss_kv = kv_distillation(
    teacher_kv_compressed,
    student_kv_latent,
    loss_type="smooth_l1",  # or "mse"
    layerwise_std=True
)
```

**3. KAVA Total Loss**
```python
# L_KAVA = -log p(A|Z,Q) - log p(A,C|Q) + α₁*L_CODI + α₂*L_KV
loss = (
    student_ce_loss +
    teacher_ce_loss +
    alpha1 * codi_loss +
    alpha2 * kv_loss
)
```

## 📈 Evaluation Metrics

### Accuracy
- **Exact Match (EM):** Extract numerical answer, compare with ground truth
- Report: mean ± std over 3 seeds

### Efficiency
- **Forward Passes:** Count forward passes during inference
  - Latent reasoning: T iterations (3) + answer tokens
  - vs Full CoT: Much higher due to long reasoning chains

## 🔬 Datasets

Following **Appendix B** of the paper:

| Dataset | HuggingFace Path | Train | Val | Test | CoT Type |
|---------|------------------|-------|-----|------|----------|
| GSM8k-AUG | `whynlp/gsm8k-aug` | 385,620 | 500 | 1,319 | Equation-only |
| GSM8k-AUG-NL | `whynlp/gsm8k-aug-nl` | 385,620 | 500 | 1,319 | Natural language |

Evaluation datasets:
- GSM8k test (original)
- GSM8k-Hard (Gao et al.)
- SVAMP (Patel et al.)

## ⚙️ Hyperparameters (Table 6)

### LoRA Configuration (All Models)
```yaml
r: 128
alpha: 32
dropout: 0.1
target_modules: [q_proj, k_proj, v_proj, o_proj]
```

### Latent Reasoning (All Models)
```yaml
num_tokens: 24  # M
num_iterations: 3  # T (Jacobi)
```

### Training Configuration

**LLaMA 3.2-1B + AUG:**
- Learning rate: 8e-4
- Batch size: 128
- Weight decay: 0.1
- KV loss: Smooth L1
- Layer-wise std: True
- α₁=10, α₂=1, λ=0.1
- Epochs: 10

**LLaMA 3.2-3B + AUG:**
- Learning rate: 2e-4 (lower for larger model)
- α₁=20, α₂=2 (stronger regularization)
- Epochs: 5 (fewer needed)

*(See configs/ for complete settings)*

## 🔍 Implementation Notes

### What's Strictly Following the Paper

✅ All hyperparameters from Table 6  
✅ R-KV compression algorithm (Section 3.2)  
✅ Loss formulations (Section 3)  
✅ Dataset sizes and sources (Appendix B)  
✅ Evaluation protocol (Section 4)  

### Engineering Choices (Not Specified in Paper)

⚠️ Exact HuggingFace checkpoint names (paper only mentions model families)  
⚠️ Prompt templates (paper says "follow CODI/PCCoT" but no exact strings)  
⚠️ Batch processing details  

These are common to all reproductions and don't affect core methodology.

## 📝 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{shen2025kava,
  title={Latent Reasoning via Compressed KV-Cache Distillation},
  author={Shen and Wu},
  journal={arXiv preprint arXiv:2510.02312},
  year={2025}
}
```

## 📚 Documentation

### 🌟 新手指南（必读）

| 文档 | 说明 | 预计时间 | 推荐度 |
|------|------|---------|--------|
| **[GETTING_STARTED_HPC.md](GETTING_STARTED_HPC.md)** | **HPC 完整上手指南** | 30 分钟 | ⭐⭐⭐⭐⭐ |
| **[KAVA_MODEL_DOWNLOAD.md](KAVA_MODEL_DOWNLOAD.md)** | **模型下载指南** ⚠️ 必读 | 17-100 分钟 | ⭐⭐⭐⭐⭐ |
| [HPC_REFERENCE.md](HPC_REFERENCE.md) | HPC 命令速查 | 按需查阅 | ⭐⭐⭐⭐⭐ |
| [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) | 完整复现流程 | 1 小时 | ⭐⭐⭐⭐☆ |

⚠️ **重要说明**：HPC 公共模型库（`/home/share/models`）中没有 KAVA 所需的 LLaMA 3.2 和 Qwen 2.5 模型，需要下载到个人目录。详见 [KAVA_MODEL_DOWNLOAD.md](KAVA_MODEL_DOWNLOAD.md)。

### 📖 HPC 部署文档

| 文档 | 说明 | 适用场景 |
|------|------|---------|
| [HPC_MODELS_QUICKSTART.md](HPC_MODELS_QUICKSTART.md) | 公共模型库配置 | 加速模型下载 |
| [SLURM_INTERACTIVE_GUIDE.md](SLURM_INTERACTIVE_GUIDE.md) | 交互式调试 | 代码调试、快速测试 |
| [SSH_PORT_FORWARDING.md](SSH_PORT_FORWARDING.md) | 端口映射 | TensorBoard、Jupyter |
| [CONTAINER_QUICKSTART.md](CONTAINER_QUICKSTART.md) | 容器化部署 | Enroot/Docker |
| [CONDA_CUDA_GUIDE.md](CONDA_CUDA_GUIDE.md) | CUDA 环境配置 | 依赖问题排查 |

### 🎓 技术文档

#### Getting Started
- **[Scripts Overview](docs/SCRIPTS_OVERVIEW.md)**: All commands and workflows ⭐ **Start here**
- **[Quick Validation Guide](docs/QUICK_VALIDATION.md)**: 7-step validation (2 min to 48 hrs)
- **[Multi-Seed Guide](docs/MULTI_SEED_GUIDE.md)**: Statistical experiments
- **[Inference Guide](docs/INFERENCE.md)**: Interactive and batch inference

#### Technical Details
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Hyperparameters and optimization
- **[Evaluation Guide](docs/EVALUATION_GUIDE.md)**: Metrics and datasets
- **[Implementation Notes](docs/IMPLEMENTATION_NOTES.md)**: Design decisions
- **[Paper Mapping](docs/PAPER_MAPPING.md)**: Paper sections → code

### 🗺️ 文档路线图

```
你是谁？
├─ 🆕 第一次使用 HPC
│   └─> 阅读 GETTING_STARTED_HPC.md
│       └─> 30 分钟配置 → 48 小时得到论文结果 ✓
│
├─ 🔧 遇到环境问题
│   ├─> HPC_REFERENCE.md（命令速查）
│   ├─> CONDA_CUDA_GUIDE.md（CUDA 问题）
│   └─> HPC_MODELS_QUICKSTART.md（模型下载）
│
├─ 🐛 需要调试代码
│   ├─> SLURM_INTERACTIVE_GUIDE.md（交互式会话）
│   └─> SSH_PORT_FORWARDING.md（远程监控）
│
├─ 📊 想要深入了解
│   ├─> REPRODUCTION_GUIDE.md（完整流程）
│   ├─> docs/TRAINING_GUIDE.md（训练细节）
│   └─> docs/PAPER_MAPPING.md（代码对应）
│
└─ 🚢 容器化部署
    └─> CONTAINER_QUICKSTART.md（Enroot/Docker）
```

### 📋 快速命令备忘

```bash
# === 新手一键启动 ===
./setup_hpc_models.sh        # 配置环境
./hpc_run_all.sh              # 提交所有实验

# === 监控 ===
squeue --me                   # 查看任务
tail -f logs/kava_*.out       # 实时日志

# === 结果 ===
python format_results.py      # 生成 LaTeX 表格
cat results/table1.tex        # 查看结果
```

---

## 📚 Documentation (English)### Reference
- **[Status Report](STATUS.md)**: Implementation completeness
- **[Checklist](docs/CHECKLIST.md)**: Feature tracking

## 🤝 Contributing

This is a paper replication project. Contributions should:
- Maintain strict adherence to paper specifications
- Add missing evaluation datasets (GSM8k-Hard, SVAMP)
- Improve engineering efficiency without changing methodology
- Fix bugs in implementation

## 📧 Issues

If you find discrepancies between this implementation and the paper:
1. Check if it's in Table 6 or explicitly stated in the paper
2. Open an issue with paper section reference
3. Engineering choices (not in paper) are open for optimization

## 🙏 Acknowledgments

- Original KAVA paper authors (Shen & Wu)
- PCCoT paper (Wu et al., 2025)
- CODI paper (for hidden state distillation baseline)
- R-KV compression algorithm

---

**Status:** ✅ **Implementation Complete** with production-ready tools

**Latest Features:**
- ✅ Multi-seed automation with PowerShell (`run_multi_seed.ps1`)
- ✅ Statistical aggregation with LaTeX output (`aggregate_multi_seed.py`)
- ✅ Smoke test suite for quick validation (`smoke_test.py`)
- ✅ Enhanced answer extraction with 4-strategy matching
- ✅ Dual-format output (JSON + YAML) for all results
- ✅ Interactive and batch inference modes
- ✅ GSM8k-Hard and SVAMP dataset support
- ✅ Comprehensive documentation (15+ guides, ~80 pages)

**Quick Links:**
- 📖 [Scripts Overview](docs/SCRIPTS_OVERVIEW.md) - All commands in one place
- ⚡ [Quick Validation](docs/QUICK_VALIDATION.md) - 2 min smoke test → 48 hr full replication
- 🎲 [Multi-Seed Guide](docs/MULTI_SEED_GUIDE.md) - Statistical experiments

**License:** MIT (implementation only; paper content © authors)
