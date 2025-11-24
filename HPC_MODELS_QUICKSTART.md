# HPC 公共模型库使用说明

## 📍 模型库位置

```
/home/share/models
```

HPC 集群管理员维护的共享 HuggingFace 模型缓存，包含常用开源模型，持续更新。

---

## ✨ 优势

- ✅ **零等待**：无需下载，立即开始训练
- ✅ **节省空间**：多用户共享，单个模型仅存储一次
- ✅ **稳定可靠**：避免网络超时和下载失败
- ✅ **持续更新**：管理员定期添加最新模型

---

## 🔧 快速配置

### 方法 1: 自动配置脚本（推荐）

```bash
# 一键配置
chmod +x setup_hpc_models.sh
./setup_hpc_models.sh

# 重新加载
source ~/.bashrc
```

### 方法 2: 手动配置

```bash
# 添加到 ~/.bashrc
cat >> ~/.bashrc << 'EOF'
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
EOF

# 立即生效
source ~/.bashrc
```

### 方法 3: 仅当前会话

```bash
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
```

---

## 📦 KAVA 项目可用模型

配置后，以下模型可直接使用：

```bash
# 检查模型是否存在
ls /home/share/models/models--meta-llama--Llama-3.2-1B-Instruct
ls /home/share/models/models--meta-llama--Llama-3.2-3B-Instruct
ls /home/share/models/models--Qwen--Qwen2.5-0.5B-Instruct
```

✅ **LLaMA 3.2-1B** (`meta-llama/Llama-3.2-1B-Instruct`)  
✅ **LLaMA 3.2-3B** (`meta-llama/Llama-3.2-3B-Instruct`)  
✅ **Qwen 2.5-0.5B** (`Qwen/Qwen2.5-0.5B-Instruct`)

---

## 💻 使用示例

### Python 代码

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 直接使用模型名称（自动从共享缓存加载）
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# 不需要指定 cache_dir，环境变量已配置
```

### 训练脚本

```bash
# 配置环境变量后直接运行
python train.py --config configs/llama1b_aug.yaml

# 模型自动从 /home/share/models 加载，无需下载
```

### SLURM 脚本

```bash
#!/bin/bash
#SBATCH --job-name=kava
#SBATCH --gres=gpu:a100-sxm4-80gb:1

# 配置模型库
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models

# 运行训练（自动使用共享模型）
python train.py --config configs/llama1b_aug.yaml
```

---

## ✅ 验证配置

### 检查环境变量

```bash
echo $HF_HOME
echo $TRANSFORMERS_CACHE
echo $HF_DATASETS_CACHE

# 应该输出: /home/share/models
```

### 测试 Python 加载

```bash
python -c "
import os
print('HF_HOME:', os.environ.get('HF_HOME'))
print('Path exists:', os.path.exists('/home/share/models'))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
print('✅ Successfully loaded from shared cache')
"
```

### 运行快速测试

```bash
# 使用共享模型运行 smoke test
python smoke_test.py

# 查看是否从共享路径加载
# 日志应显示: Loading model from /home/share/models/...
```

---

## 🔍 查看可用模型

```bash
# 列出所有模型
ls -1 /home/share/models/models--*

# 查看模型详情
ls -lh /home/share/models/models--meta-llama--Llama-3.2-1B-Instruct

# 统计模型数量
ls -1 /home/share/models/models--* | wc -l

# 查看模型总大小
du -sh /home/share/models
```

---

## ⚠️ 故障排除

### 问题 1: 仍然尝试下载模型

**症状**：看到 "Downloading model..." 提示

**解决**：
```bash
# 检查环境变量
echo $HF_HOME

# 如果为空，重新配置
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
```

### 问题 2: 权限拒绝

**症状**：`Permission denied: /home/share/models`

**解决**：
```bash
# 检查目录权限
ls -ld /home/share/models

# 如果无权限，联系管理员
```

### 问题 3: 模型不存在

**症状**：`Model not found in /home/share/models`

**解决**：
```bash
# 检查模型列表
ls /home/share/models/models--*/

# 如果模型确实不存在，有两个选择：

# 选项 A: 请求管理员添加模型
# 发送邮件给 HPC 管理员，说明需要的模型

# 选项 B: 下载到个人目录（临时方案）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/.cache/huggingface
python train.py --config configs/llama1b_aug.yaml
```

### 问题 4: Conda 环境未自动加载

**症状**：每次激活环境都需要重新设置

**解决**：
```bash
# 配置 Conda 激活脚本
conda activate kava
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/hf_models.sh << 'EOF'
#!/bin/bash
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/hf_models.sh

# 重新激活验证
conda deactivate && conda activate kava
echo $HF_HOME  # 应该输出 /home/share/models
```

---

## 📚 相关文档

- **完整复现指南**: [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)
- **HPC 命令参考**: [HPC_REFERENCE.md](HPC_REFERENCE.md)
- **交互式使用**: [SLURM_INTERACTIVE_GUIDE.md](SLURM_INTERACTIVE_GUIDE.md)
- **配置脚本**: `setup_hpc_models.sh`

---

## 💡 最佳实践

1. **首次使用**：运行 `setup_hpc_models.sh` 自动配置
2. **验证配置**：每次登录检查 `echo $HF_HOME`
3. **SLURM 脚本**：在脚本开头设置环境变量
4. **日志检查**：训练时查看是否从 `/home/share/models` 加载
5. **定期更新**：关注管理员公告，了解新增模型

---

## 📞 获取帮助

- **检查配置**：`python -c "import os; print(os.environ.get('HF_HOME'))"`
- **验证模型**：`ls /home/share/models/models--meta-llama*`
- **联系管理员**：如需添加新模型或遇到权限问题

---

**更新日期**: 2025-01-17  
**维护者**: KAVA Project Team
