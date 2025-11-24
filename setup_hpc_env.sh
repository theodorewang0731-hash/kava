#!/bin/bash
################################################################################
# HPC 环境配置脚本（自动加载模块）
# 用法: source setup_hpc_env.sh  （注意：必须使用 source 或 .）
################################################################################

echo "════════════════════════════════════════════════════════════════"
echo "  KAVA HPC 环境配置"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 初始化 module 系统
if [ -f /usr/share/modules/init/bash ]; then
    echo "[1/5] 初始化 module 系统..."
    . /usr/share/modules/init/bash
    module use --append /home/share/modules/modulefiles
    echo "✓ Module 系统已初始化"
else
    echo "⚠ 未找到 module 系统，可能不在 HPC 环境中"
fi

# 加载 anaconda3
echo ""
echo "[2/5] 加载 Anaconda..."
if command -v module &> /dev/null; then
    module load anaconda3
    if [ $? -eq 0 ]; then
        echo "✓ Anaconda 已加载"
    else
        echo "⚠ Anaconda 加载失败，尝试直接查找 conda..."
    fi
fi

# 验证 conda
echo ""
echo "[3/5] 验证 conda..."
if command -v conda &> /dev/null; then
    echo "✓ Conda 可用: $(conda --version)"
else
    echo "✗ Conda 仍不可用"
    echo ""
    echo "可能的解决方案："
    echo "  1. 检查 module 是否可用: module avail"
    echo "  2. 手动加载: module load anaconda3"
    echo "  3. 或使用系统 Python: python3 -m venv venv"
    exit 1
fi

# 创建或激活 conda 环境
echo ""
echo "[4/5] 配置 KAVA 环境..."

if conda env list | grep -q "^kava "; then
    echo "⚠ 环境 'kava' 已存在，激活中..."
    conda activate kava
else
    echo "创建新环境 'kava'..."
    conda create -n kava python=3.10 -y
    conda activate kava
fi

# 安装依赖
echo ""
echo "[5/5] 安装依赖..."
pip install -r requirements.txt

# 完成
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  配置完成！"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "环境已激活。接下来你可以："
echo ""
echo "1. 提交训练任务："
echo "   sbatch submit_multi_seed.slurm"
echo ""
echo "2. 或交互式运行："
echo "   python train.py --config configs/llama1b_aug.yaml"
echo ""
echo "注意: 每次登录都需要重新激活环境："
echo "       conda activate kava"
echo ""
