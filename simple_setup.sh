#!/bin/bash
################################################################################
# 简化的 HPC 启动脚本（处理路径空格问题）
# 用法: bash simple_setup.sh
################################################################################

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════════"
echo "  KAVA 简化配置脚本"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""

# 获取脚本所在目录（处理空格）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "当前目录: $SCRIPT_DIR"
echo ""

# =============================================================================
# 步骤 1: 检查路径
# =============================================================================
echo -e "${BLUE}[1/5] 检查路径${NC}"
if [[ "$SCRIPT_DIR" == *" "* ]]; then
    echo -e "${YELLOW}⚠ 警告: 路径包含空格${NC}"
    echo "  这可能导致一些脚本失败"
    echo ""
    echo -e "${YELLOW}强烈建议重命名目录:${NC}"
    echo "  cd $(dirname "$SCRIPT_DIR")"
    echo "  mv \"$(basename "$SCRIPT_DIR")\" kava_review"
    echo "  cd kava_review"
    echo ""
    read -p "是否继续当前配置? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ 路径正常${NC}"
fi
echo ""

# =============================================================================
# 步骤 2: 检查 Python
# =============================================================================
echo -e "${BLUE}[2/5] 检查 Python${NC}"

# 尝试找到 Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}✓ 找到 Python3: $(python3 --version)${NC}"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo -e "${GREEN}✓ 找到 Python: $(python --version)${NC}"
else
    echo -e "${RED}✗ 未找到 Python${NC}"
    exit 1
fi
echo ""

# =============================================================================
# 步骤 3: 创建虚拟环境（不依赖 conda）
# =============================================================================
echo -e "${BLUE}[3/5] 创建 Python 虚拟环境${NC}"

VENV_DIR="venv_kava"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠ 虚拟环境已存在: $VENV_DIR${NC}"
    read -p "是否重新创建? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ 虚拟环境已重新创建${NC}"
    else
        echo "使用现有虚拟环境"
    fi
else
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ 虚拟环境已创建${NC}"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
echo ""

# =============================================================================
# 步骤 4: 安装依赖
# =============================================================================
echo -e "${BLUE}[4/5] 安装依赖${NC}"

# 升级 pip
pip install --upgrade pip -q

# 安装项目依赖
echo "正在安装 requirements.txt 中的包..."
pip install -r requirements.txt

echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# =============================================================================
# 步骤 5: 配置环境变量
# =============================================================================
echo -e "${BLUE}[5/5] 配置环境${NC}"

# HuggingFace 缓存
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface"

mkdir -p "$HF_HOME"
echo -e "${GREEN}✓ HuggingFace 缓存: $HF_HOME${NC}"

# 创建必要的目录
mkdir -p logs outputs/checkpoints outputs/results
echo -e "${GREEN}✓ 输出目录已创建${NC}"
echo ""

# =============================================================================
# 完成
# =============================================================================
echo -e "${GREEN}"
echo "════════════════════════════════════════════════════════════════"
echo "  配置完成！"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo "虚拟环境已激活。你现在可以："
echo ""
echo "1. 下载模型（如果还没有）："
echo "   python download_from_hf.py"
echo ""
echo "2. 运行训练："
echo "   python train.py --config configs/llama1b_aug.yaml"
echo ""
echo "3. 提交 SLURM 任务："
echo "   sbatch submit_multi_seed.slurm"
echo ""
echo -e "${YELLOW}注意: 每次登录都需要重新激活环境:${NC}"
echo "   source \"$VENV_DIR/bin/activate\""
echo ""
echo -e "${YELLOW}或者将激活命令添加到 ~/.bashrc:${NC}"
echo "   echo 'source \"/home/rpwang/kava review/$VENV_DIR/bin/activate\"' >> ~/.bashrc"
echo "   # 注意: 路径包含空格，必须使用引号"
echo ""
echo -e "${YELLOW}💡 推荐: 重命名目录以避免空格问题:${NC}"
echo "   cd /home/rpwang"
echo "   mv \"kava review\" kava_review"
echo ""
