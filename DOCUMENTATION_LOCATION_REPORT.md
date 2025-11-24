# 文档位置说明

## 📋 问题分析

你提到的 4 个文档文件**实际上存在于项目根目录**，而非 `docs/` 子目录。这导致了路径引用不一致的问题。

---

## ✅ 文档实际位置

| 文档名称 | 预期位置（README引用） | 实际位置 | 状态 |
|---------|---------------------|---------|------|
| `GETTING_STARTED_HPC.md` | `docs/GETTING_STARTED_HPC.md` | **根目录** ✅ | ⚠️ 路径不一致 |
| `KAVA_MODEL_DOWNLOAD.md` | `docs/KAVA_MODEL_DOWNLOAD.md` | **根目录** ✅ | ⚠️ 路径不一致 |
| `HPC_REFERENCE.md` | `docs/HPC_REFERENCE.md` | **根目录** ✅ | ⚠️ 路径不一致 |
| `SLURM_INTERACTIVE_GUIDE.md` | `docs/SLURM_INTERACTIVE_GUIDE.md` | **根目录** ✅ | ⚠️ 路径不一致 |

---

## 📁 当前项目结构

### 根目录文档（应移至 docs/）

```
h:/kava/kava review/
├── GETTING_STARTED_HPC.md          ← 691 行，HPC 新手上手指南
├── KAVA_MODEL_DOWNLOAD.md          ← 405 行，模型下载详解
├── HPC_REFERENCE.md                ← 1635 行，HPC 命令速查
├── SLURM_INTERACTIVE_GUIDE.md      ← 495 行，交互式调试指南
├── ... (其他 40+ 个 .md 文件)
└── docs/
    ├── COMPLETION.md
    ├── EXAMPLES.md
    ├── FINAL_ITERATION_REPORT.md
    ├── PROJECT_INVENTORY.md
    ├── MULTI_SEED_GUIDE.md
    ├── MULTI_SEED.md
    ├── INFERENCE.md
    ├── SCRIPTS_OVERVIEW.md
    └── QUICK_VALIDATION.md
```

---

## 🔍 文档内容概览

### 1. `GETTING_STARTED_HPC.md`（691 行）

**目标读者**：HPC 新手用户  
**预计时间**：30 分钟配置 + 48 小时运行  
**关键内容**：
- 阶段 1: 上传项目到 HPC（5 分钟）
- 阶段 2: 环境配置（15 分钟）
- 阶段 3: 提交训练（5 分钟）
- 阶段 4: 监控进度（48 小时）
- 阶段 5: 生成结果（5 分钟）

**核心流程**：
```bash
# 1. 上传项目
scp -r kava/ your_username@hpc.example.edu:~/

# 2. 自动配置
bash setup_hpc.sh

# 3. 提交所有实验
bash run_reproduce.sh
```

---

### 2. `KAVA_MODEL_DOWNLOAD.md`（405 行）

**目标读者**：需要下载模型的用户  
**预计时间**：17-100 分钟（取决于网络）  
**核心问题**：HPC 公共模型库（`/home/share/models`）**不包含** KAVA 所需模型

**所需模型**：
- LLaMA 3.2-1B (~5 GB)
- LLaMA 3.2-3B (~12 GB)
- Qwen 2.5-0.5B (~2 GB)
- **总计 ~19 GB**

**下载方法**：
1. **方法 A**：在 HPC 节点上直接下载（需要外网访问）
2. **方法 B**：本地下载后上传到 HPC（适合网络受限环境）
3. **方法 C**：使用代理服务器下载

**关键命令**：
```bash
# 自动下载所有模型
python download_from_hf.py

# 或使用脚本
bash download_models_only.sh
```

---

### 3. `HPC_REFERENCE.md`（1635 行）

**目标读者**：HPC 用户（所有技能水平）  
**文档类型**：命令速查表 + 工作流参考  
**关键章节**：

#### 3.1 环境设置
- Module 系统使用
- Conda 环境激活
- CUDA 版本选择

#### 3.2 SLURM 命令
```bash
# 提交任务
sbatch submit_multi_seed.slurm

# 查看队列
squeue -u $USER

# 取消任务
scancel <job_id>

# 查看日志
tail -f logs/llama1b_aug_seed42.log
```

#### 3.3 资源管理
- GPU 类型选择（A100/V100）
- 内存配置
- CPU 核心数
- 时间限制

#### 3.4 数据管理
- 数据集位置
- 模型缓存
- 输出目录结构

#### 3.5 故障排查
- 常见错误及解决方案
- 日志分析技巧
- 性能优化建议

---

### 4. `SLURM_INTERACTIVE_GUIDE.md`（495 行）

**目标读者**：需要调试代码的用户  
**使用场景**：开发、测试、快速验证  
**核心对比**：

| 特性 | srun（交互式） | sbatch（批处理） |
|------|---------------|-----------------|
| **使用场景** | 调试、测试、开发 | 正式训练、批量实验 |
| **执行方式** | 阻塞终端，实时输出 | 后台运行，输出到文件 |
| **资源占用** | 立即分配 | 排队等待 |
| **适合时长** | < 2 小时 | 数小时至数天 |

**快速开始**：
```bash
# 申请单卡 GPU 节点
srun --gres=gpu:a100-sxm4-80gb:1 --pty bash -i

# 在交互式环境中运行
conda activate kava
python train.py --config configs/llama1b_aug.yaml
```

**典型工作流**：
1. **开发阶段**：使用 `srun` 交互式调试
2. **验证通过**：切换到 `sbatch` 批处理运行
3. **大规模实验**：使用批处理脚本提交多个任务

---

## 🔧 路径不一致问题

### 问题来源

多个文件中引用了这些文档，但路径不一致：

#### README.md 中的引用（指向 docs/）
```markdown
2. **[GETTING_STARTED_HPC.md](docs/GETTING_STARTED_HPC.md)** - 完整 HPC 指南
3. **[KAVA_MODEL_DOWNLOAD.md](docs/KAVA_MODEL_DOWNLOAD.md)** - 模型下载详解
```

#### smoke_test_lite.py 中的检查（也指向 docs/）
```python
required_docs = {
    'docs/GETTING_STARTED_HPC.md': 'HPC getting started guide',
    'docs/KAVA_MODEL_DOWNLOAD.md': 'Model download instructions',
    'docs/HPC_REFERENCE.md': 'HPC reference',
    'docs/SLURM_INTERACTIVE_GUIDE.md': 'SLURM interactive guide',
}
```

#### 但文件实际在根目录
```
h:/kava/kava review/GETTING_STARTED_HPC.md          ✅ 存在
h:/kava/kava review/KAVA_MODEL_DOWNLOAD.md          ✅ 存在
h:/kava/kava review/HPC_REFERENCE.md                ✅ 存在
h:/kava/kava review/SLURM_INTERACTIVE_GUIDE.md      ✅ 存在
```

---

## 💡 推荐解决方案

### 方案 A：移动文件到 docs/（推荐）

**优点**：
- 符合标准项目结构
- 与 README 引用一致
- 便于文档管理

**操作**：
```bash
# 移动 4 个文档到 docs/
mv GETTING_STARTED_HPC.md docs/
mv KAVA_MODEL_DOWNLOAD.md docs/
mv HPC_REFERENCE.md docs/
mv SLURM_INTERACTIVE_GUIDE.md docs/
```

**需要更新的引用**：
- `start.sh` (line 203)
- `setup_hpc.sh` (line 152, 157)
- `run_reproduce.sh` (line 490, 703)
- `SETUP_COMPLETE.md` (多处)
- 其他脚本中的相对路径引用

---

### 方案 B：修改所有引用指向根目录

**优点**：
- 不需要移动文件
- 保持现有结构

**缺点**：
- 根目录已有 40+ 个 .md 文件，比较混乱
- 不符合标准项目结构惯例

**操作**：
```bash
# 更新 README.md 中的所有路径引用
# 从: docs/GETTING_STARTED_HPC.md
# 改为: GETTING_STARTED_HPC.md

# 更新 smoke_test_lite.py
# 从: 'docs/GETTING_STARTED_HPC.md'
# 改为: 'GETTING_STARTED_HPC.md'
```

---

## 📊 影响范围分析

### 受影响的文件（需要路径修正）

**优先级 1（高）**：
1. `README.md` - 主入口文档，多处引用
2. `smoke_test_lite.py` - 自动化测试会失败
3. `SETUP_COMPLETE.md` - 用户查看的设置完成指南

**优先级 2（中）**：
4. `start.sh` - 启动脚本的提示信息
5. `setup_hpc.sh` - HPC 配置脚本
6. `run_reproduce.sh` - 复现脚本

**优先级 3（低）**：
7. 其他零散的交叉引用

---

## ✅ 验证方法

### 检查路径是否正确
```bash
# 方法 1: 检查文件是否存在
ls -lh docs/GETTING_STARTED_HPC.md
ls -lh GETTING_STARTED_HPC.md

# 方法 2: 运行 smoke test
python smoke_test_lite.py

# 方法 3: 搜索所有引用
grep -r "GETTING_STARTED_HPC" --include="*.md" --include="*.sh" --include="*.py"
```

### 验证链接有效性
```bash
# 在 Markdown 中测试链接
# 如果路径错误，VS Code 会显示灰色链接
```

---

## 📝 建议行动

### 立即执行（修复测试）
```bash
# 临时解决：在 docs/ 创建符号链接
ln -s ../GETTING_STARTED_HPC.md docs/GETTING_STARTED_HPC.md
ln -s ../KAVA_MODEL_DOWNLOAD.md docs/KAVA_MODEL_DOWNLOAD.md
ln -s ../HPC_REFERENCE.md docs/HPC_REFERENCE.md
ln -s ../SLURM_INTERACTIVE_GUIDE.md docs/SLURM_INTERACTIVE_GUIDE.md
```

### 长期方案（结构优化）
```bash
# 1. 移动文件
mv GETTING_STARTED_HPC.md docs/
mv KAVA_MODEL_DOWNLOAD.md docs/
mv HPC_REFERENCE.md docs/
mv SLURM_INTERACTIVE_GUIDE.md docs/

# 2. 更新所有引用（需要逐一检查和修改）
# 3. 运行验证
python smoke_test_lite.py
```

---

## 🎯 结论

**文档本身是完整的**，只是位置与引用不匹配。推荐采用**方案 A**（移动文件到 docs/），这样既符合标准项目结构，也能保证所有引用的一致性。

**当前状态**：
- ✅ 文档内容完整（共 3426 行高质量文档）
- ⚠️ 路径引用不一致
- ❌ smoke_test_lite.py 会报告文档缺失

**修复优先级**：
1. 🔴 高优先级：修复 `smoke_test_lite.py` 的路径检查
2. 🟡 中优先级：统一 README.md 中的所有路径引用
3. 🟢 低优先级：清理根目录，整理文档到 docs/

---

## 📚 补充：完整文档清单

### HPC 相关文档（4 个主要 + 6 个补充）

#### 主要文档（你提到的）
1. ✅ `GETTING_STARTED_HPC.md` (691 行) - HPC 新手指南
2. ✅ `KAVA_MODEL_DOWNLOAD.md` (405 行) - 模型下载详解
3. ✅ `HPC_REFERENCE.md` (1635 行) - HPC 命令速查
4. ✅ `SLURM_INTERACTIVE_GUIDE.md` (495 行) - 交互式调试

#### 补充文档（根目录）
5. ✅ `HPC_DOWNLOAD_GUIDE.md` - 下载指南（简化版）
6. ✅ `HPC_LINUX_READY.md` - Linux 环境就绪检查
7. ✅ `HPC_MODELS_QUICKSTART.md` - 模型快速开始
8. ✅ `CONDA_CUDA_GUIDE.md` - Conda + CUDA 配置
9. ✅ `SSH_PORT_FORWARDING.md` - SSH 端口转发
10. ✅ `SETUP_COMPLETE.md` - 设置完成总结

### docs/ 子目录（9 个文档）
- `COMPLETION.md`
- `EXAMPLES.md`
- `FINAL_ITERATION_REPORT.md`
- `PROJECT_INVENTORY.md`
- `MULTI_SEED_GUIDE.md`
- `MULTI_SEED.md`
- `INFERENCE.md`
- `SCRIPTS_OVERVIEW.md`
- `QUICK_VALIDATION.md`

**文档总计**：50+ 个 .md 文件，超过 20,000 行文档

---

## 🔗 相关文件

- 问题来源：`smoke_test_lite.py:167-170`
- 主要引用：`README.md:101-102, 113, 159-161, 441-443`
- 脚本引用：`setup_hpc.sh:152,157`, `run_reproduce.sh:490,703`, `start.sh:203`
