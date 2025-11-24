#!/bin/bash

################################################################################
# KAVA HPC 快速设置脚本
################################################################################
# 这个脚本帮助你在 HPC 上快速设置 KAVA 项目环境
# 
# 用法:
#   bash setup_hpc.sh
#
# 功能:
#   1. 配置 HuggingFace 缓存目录
#   2. 创建必要的目录结构
#   3. 设置文件权限
#   4. 验证 SLURM 环境
#   5. 检查磁盘空间
################################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║        KAVA HPC 环境快速设置                                   ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}[1/6] 检查项目目录${NC}"
echo "当前目录: $SCRIPT_DIR"
if [ ! -f "train.py" ]; then
    echo -e "${RED}✗ 错误: 未找到 train.py，请确保在 KAVA 项目根目录运行此脚本${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 项目目录正确${NC}"
echo ""

echo -e "${BLUE}[2/6] 配置 HuggingFace 缓存${NC}"
HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$HF_CACHE"

# 检查是否已经配置
if grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
    echo -e "${YELLOW}⚠ ~/.bashrc 中已存在 HF_HOME 配置，跳过${NC}"
else
    echo "添加环境变量到 ~/.bashrc..."
    cat >> ~/.bashrc << 'EOF'

# KAVA HuggingFace 缓存配置
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface
EOF
    echo -e "${GREEN}✓ 已添加到 ~/.bashrc${NC}"
fi

# 立即生效
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE"
echo "HuggingFace 缓存目录: $HF_CACHE"
echo ""

echo -e "${BLUE}[3/6] 创建必要的目录${NC}"
mkdir -p logs
mkdir -p outputs/logs
mkdir -p outputs/results
mkdir -p outputs/checkpoints
echo -e "${GREEN}✓ 目录结构创建完成${NC}"
ls -ld logs outputs outputs/logs outputs/results outputs/checkpoints
echo ""

echo -e "${BLUE}[4/6] 设置脚本权限${NC}"
chmod +x run_reproduce.sh
chmod +x hpc_run_all.sh
chmod +x setup_hpc.sh
chmod +x *.slurm 2>/dev/null || true
echo -e "${GREEN}✓ 脚本权限设置完成${NC}"
ls -l *.sh *.slurm 2>/dev/null | grep -E '\.(sh|slurm)$'
echo ""

echo -e "${BLUE}[5/6] 检查磁盘空间${NC}"
echo "当前目录磁盘使用情况:"
df -h "$PWD" | tail -1 | awk '{print "  可用空间: " $4 " / " $2}'

# 检查是否有足够空间（至少 20GB）
available_kb=$(df -k "$HF_CACHE" | awk 'NR==2 {print $4}')
available_gb=$((available_kb / 1024 / 1024))
if [ $available_gb -lt 20 ]; then
    echo -e "${YELLOW}⚠ 警告: 可用空间仅 ${available_gb}GB，建议至少 20GB${NC}"
    echo "  KAVA 需要下载约 19GB 的模型文件"
else
    echo -e "${GREEN}✓ 磁盘空间充足 (${available_gb}GB available)${NC}"
fi
echo ""

echo -e "${BLUE}[6/6] 验证 SLURM 环境${NC}"
if command -v sbatch &> /dev/null; then
    echo -e "${GREEN}✓ sbatch 命令可用${NC}"
    
    # 检查分区
    if sinfo -p compute &> /dev/null 2>&1; then
        echo -e "${GREEN}✓ 'compute' 分区可访问${NC}"
        echo "  分区信息:"
        sinfo -p compute -o "    %P %a %l %D %t %N" | head -5
    else
        echo -e "${YELLOW}⚠ 无法访问 'compute' 分区${NC}"
    fi
    
    # 检查队列
    echo "  当前队列:"
    squeue -u $USER -o "    %.18i %.9P %.30j %.8T" | head -5
    if [ $(squeue -u $USER -h | wc -l) -eq 0 ]; then
        echo -e "${GREEN}    (无运行中的任务)${NC}"
    fi
else
    echo -e "${RED}✗ sbatch 命令不可用，请确保在 HPC 登录节点运行${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║  ✓ HPC 环境设置完成！                                          ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${CYAN}下一步操作:${NC}"
echo ""
echo -e "${YELLOW}重要: 需要先下载模型 (~19GB)${NC}"
echo "  HPC 公共库缺少 KAVA 所需模型，必须下载到个人目录"
echo ""
echo "选项 A: 使用自动化脚本（推荐）"
echo -e "  ${GREEN}bash run_reproduce.sh${NC}"
echo "  → 自动下载模型并提交所有训练任务"
echo ""
echo "选项 B: 手动下载模型"
echo "  参见: docs/KAVA_MODEL_DOWNLOAD.md"
echo "  然后运行: bash run_reproduce.sh --skip-download"
echo ""
echo "选项 C: 查看详细文档"
echo "  快速开始: REPRODUCTION_CHECKLIST.md"
echo "  完整指南: docs/GETTING_STARTED_HPC.md"
echo ""

echo -e "${CYAN}常用命令:${NC}"
echo "  检查队列:    squeue -u \$USER"
echo "  查看节点:    sinfo -p compute"
echo "  监控任务:    tail -f logs/kava_*.out"
echo "  取消任务:    scancel <job_id>"
echo ""

echo -e "${GREEN}设置完成！请重新登录或运行以下命令使环境变量生效:${NC}"
echo -e "  ${CYAN}source ~/.bashrc${NC}"
echo ""
