# Conda CUDA 安装快速参考

**在 Conda 环境中安装和管理 CUDA 的完整指南**

---

## 🚀 快速开始

### 一键安装（推荐）

```bash
# 创建环境 + CUDA + PyTorch 一步完成
conda create -n kava python=3.10 \
    cudatoolkit=11.8 \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y

conda activate kava

# 安装项目依赖
pip install -r requirements.txt
pip install peft wandb bitsandbytes

# 配置环境变量
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
cd $CONDA_PREFIX && ln -s lib lib64

# 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📦 详细安装步骤

### Step 1: 查找可用的 CUDA 版本

```bash
# 搜索 cudatoolkit
conda search cudatoolkit -c nvidia

# 常见版本
# - cudatoolkit=11.3
# - cudatoolkit=11.7
# - cudatoolkit=11.8
# - cudatoolkit=12.1
```

### Step 2: 创建环境并安装 CUDA

```bash
# 方法 1: 创建时安装
conda create -n kava python=3.10 cudatoolkit=11.8 -c nvidia -y

# 方法 2: 在现有环境中安装
conda create -n kava python=3.10 -y
conda activate kava
conda install cudatoolkit=11.8 -c nvidia
```

### Step 3: 安装 PyTorch

```bash
# 确保匹配 CUDA 版本
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 或使用 pip（备选）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: 配置环境变量

```bash
# 查找 Conda 环境路径
conda env list
# 输出: kava  /home/username/.conda/envs/kava

# 设置临时变量（当前会话）
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 创建 lib64 链接
cd $CONDA_PREFIX
ln -s lib lib64
```

### Step 5: 永久配置（自动激活）

```bash
# 创建激活脚本
conda activate kava
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# 激活时设置变量
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH
EOF

# 停用时清除变量
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << 'EOF'
#!/bin/bash
unset CUDA_HOME
unset CUDA_PATH
EOF

# 赋予执行权限
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# 测试
conda deactivate
conda activate kava
echo $CUDA_HOME  # 应显示 Conda 环境路径
```

---

## ✅ 验证安装

### 基本验证

```bash
# 激活环境
conda activate kava

# 检查 Python
python --version

# 检查 CUDA
nvcc -V
which nvcc

# 检查环境变量
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 检查库文件
ls $CONDA_PREFIX/lib/libcudart*
ls $CONDA_PREFIX/lib/libcublas*

# 检查 lib64 链接
ls -la $CONDA_PREFIX | grep lib64
```

### PyTorch 验证

```bash
# 完整验证脚本
python << 'EOF'
import torch
import sys

print("=" * 60)
print("PyTorch CUDA Verification")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 测试张量运算
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU tensor operation successful")
else:
    print("✗ CUDA not available!")
    
print("=" * 60)
EOF
```

### 编译测试（可选）

```bash
# 测试 JIT 编译
python << 'EOF'
import torch
from torch.utils.cpp_extension import load_inline

# 简单的 CUDA kernel
cuda_source = """
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

cpp_source = """
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::zeros_like(a);
    int n = a.numel();
    add_kernel<<<(n+255)/256, 256>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        n
    );
    return c;
}
"""

try:
    module = load_inline(
        name='test_cuda',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=['add'],
        verbose=True
    )
    print("✓ CUDA JIT compilation successful")
except Exception as e:
    print(f"✗ CUDA JIT compilation failed: {e}")
EOF
```

---

## 🔧 常见问题修复

### 问题 1: nvcc 找不到

```bash
# 检查
which nvcc

# 如果没有输出
export PATH=$CONDA_PREFIX/bin:$PATH
which nvcc  # 应该找到了

# 永久修复
echo 'export PATH=$CONDA_PREFIX/bin:$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### 问题 2: 库文件找不到

```bash
# 症状: libcudart.so.11.8: cannot open shared object file

# 检查库文件
ls $CONDA_PREFIX/lib/libcudart*

# 如果存在但找不到
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 创建 lib64 链接
cd $CONDA_PREFIX
ln -s lib lib64

# 验证
ldd $(python -c "import torch; print(torch.__file__)") | grep cuda
```

### 问题 3: CUDA_HOME 未设置

```bash
# 临时设置
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX

# 验证
echo $CUDA_HOME
ls $CUDA_HOME/bin/nvcc

# 永久设置（见 Step 5）
```

### 问题 4: DeepSpeed 编译失败

```bash
# 清除缓存
rm -rf ~/.cache/torch_extensions/*
rm -rf /tmp/torch_extensions/*

# 设置完整环境
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
cd $CONDA_PREFIX && ln -s lib lib64

# 重新安装 DeepSpeed
pip uninstall deepspeed -y
pip install deepspeed --no-cache-dir

# 测试
python -c "import deepspeed; print(deepspeed.__version__)"
```

### 问题 5: 版本冲突

```bash
# 检查 CUDA 版本
nvcc -V  # Conda 版本
nvidia-smi  # 驱动版本

# PyTorch 期望的 CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 如果不匹配，重新安装
conda remove cudatoolkit pytorch -y
conda install cudatoolkit=11.8 -c nvidia
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## 🎯 最佳实践

### 1. 环境隔离

```bash
# 为不同项目创建独立环境
conda create -n kava-cuda11.8 python=3.10 cudatoolkit=11.8
conda create -n kava-cuda12.1 python=3.10 cudatoolkit=12.1

# 快速切换
conda activate kava-cuda11.8
conda activate kava-cuda12.1
```

### 2. 自动化脚本

```bash
# 创建 setup.sh
cat > setup_kava_env.sh << 'EOF'
#!/bin/bash
set -e

ENV_NAME="kava"
CUDA_VERSION="11.8"
PYTHON_VERSION="3.10"

echo "Creating environment: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION cudatoolkit=$CUDA_VERSION -c nvidia -y

conda activate $ENV_NAME

echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia -y

echo "Installing dependencies..."
pip install -r requirements.txt
pip install peft wandb bitsandbytes

echo "Configuring environment variables..."
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'INNER_EOF'
#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
INNER_EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo "Creating lib64 link..."
cd $CONDA_PREFIX && ln -sf lib lib64

echo "Verifying installation..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo "✓ Environment setup complete!"
echo "Activate with: conda activate $ENV_NAME"
EOF

chmod +x setup_kava_env.sh
./setup_kava_env.sh
```

### 3. 备份和迁移

```bash
# 导出环境
conda activate kava
conda env export > kava_environment.yml

# 在新机器上重建
conda env create -f kava_environment.yml

# 或使用 requirements
pip freeze > requirements_full.txt
```

---

## 📊 性能对比

| CUDA 来源 | 安装时间 | 磁盘占用 | 灵活性 | 编译速度 |
|----------|---------|---------|--------|---------|
| 系统 CUDA | - | 共享 | ★★★☆☆ | ★★★★★ |
| Conda CUDA | ~5 分钟 | ~3GB/环境 | ★★★★★ | ★★★★☆ |
| Docker | ~10 分钟 | ~10GB | ★★★★☆ | ★★★★☆ |

---

## 🔗 资源链接

- **Conda CUDA Packages**: https://anaconda.org/nvidia/cudatoolkit
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **CUDA Toolkit Docs**: https://docs.nvidia.com/cuda/

---

## 💡 提示

1. ✅ **首选 Conda CUDA**: 适合个人开发，隔离性好
2. ✅ **系统 CUDA 用于 HPC**: 集群环境通常已配置
3. ✅ **永久配置环境变量**: 避免每次手动设置
4. ✅ **创建 lib64 链接**: 解决大部分链接问题
5. ✅ **定期更新**: `conda update cudatoolkit pytorch`
6. ⚠️ **驱动兼容性**: 确保 NVIDIA 驱动版本 >= CUDA 版本
7. ⚠️ **磁盘空间**: 每个环境约 3-5GB

---

**快速获取帮助**
```bash
# Conda 帮助
conda info
conda list

# CUDA 信息
nvcc --version
nvidia-smi

# PyTorch 信息
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```
