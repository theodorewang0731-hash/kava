#!/bin/bash

################################################################################
# KAVA 部署验证脚本
################################################################################
# 在 HPC 上传代码后运行此脚本，验证所有文件和配置是否正确
#
# 用法: bash verify_deployment.sh
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

log_check() {
    echo -e "${BLUE}[检查]${NC} $1"
}

log_ok() {
    echo -e "${GREEN}  ✓${NC} $1"
}

log_error() {
    echo -e "${RED}  ✗${NC} $1"
    ((ERRORS++))
}

log_warning() {
    echo -e "${YELLOW}  ⚠${NC} $1"
    ((WARNINGS++))
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  KAVA 部署验证"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. 检查必要文件
log_check "检查核心文件..."
REQUIRED_FILES=(
    "train.py"
    "evaluate.py"
    "inference.py"
    "run_multi_seed.py"
    "aggregate_multi_seed.py"
    "requirements.txt"
    "README.md"
    "run_everything.sh"
    "setup_hpc.sh"
    "hpc_run_all.sh"
    "submit_multi_seed.slurm"
    "REPRODUCTION_CHECKLIST.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_ok "$file"
    else
        log_error "缺少文件: $file"
    fi
done
echo ""

# 2. 检查配置文件
log_check "检查配置文件..."
CONFIGS=(
    "configs/llama1b_aug.yaml"
    "configs/llama1b_aug_nl.yaml"
    "configs/qwen05b_aug.yaml"
    "configs/llama3b_aug.yaml"
)

for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        log_ok "$config"
    else
        log_error "缺少配置: $config"
    fi
done
echo ""

# 3. 检查 src 目录
log_check "检查源代码模块..."
SRC_FILES=(
    "src/__init__.py"
    "src/rkv_compression.py"
    "src/losses.py"
    "src/latent_reasoning.py"
    "src/data_utils.py"
    "src/trainer.py"
    "src/utils.py"
    "src/evaluation_datasets.py"
)

for src in "${SRC_FILES[@]}"; do
    if [ -f "$src" ]; then
        log_ok "$src"
    else
        log_error "缺少源文件: $src"
    fi
done
echo ""

# 4. 检查脚本权限
log_check "检查脚本执行权限..."
SCRIPTS=(
    "run_reproduce.sh"
    "setup_hpc.sh"
    "hpc_run_all.sh"
    "verify_deployment.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            log_ok "$script 可执行"
        else
            log_warning "$script 不可执行，正在修复..."
            chmod +x "$script"
            log_ok "已设置 $script 为可执行"
        fi
    fi
done
echo ""

# 5. 检查换行符格式（CRLF vs LF）
log_check "检查文件换行符格式..."
if command -v file &> /dev/null; then
    for script in "${SCRIPTS[@]}"; do
        if [ -f "$script" ]; then
            file_type=$(file "$script")
            if echo "$file_type" | grep -q "CRLF"; then
                log_warning "$script 使用 Windows 换行符 (CRLF)"
                if command -v dos2unix &> /dev/null; then
                    dos2unix "$script" 2>/dev/null
                    log_ok "已转换 $script 为 Unix 格式 (LF)"
                else
                    log_warning "建议运行: dos2unix $script"
                fi
            else
                log_ok "$script (Unix 格式)"
            fi
        fi
    done
else
    log_warning "未安装 'file' 命令，跳过换行符检查"
fi
echo ""

# 6. 检查 SLURM 环境
log_check "检查 SLURM 环境..."
if command -v sbatch &> /dev/null; then
    log_ok "sbatch 命令可用"
    
    if command -v squeue &> /dev/null; then
        log_ok "squeue 命令可用"
    else
        log_error "squeue 命令不可用"
    fi
    
    if command -v sinfo &> /dev/null; then
        log_ok "sinfo 命令可用"
        
        # 检查 compute 分区
        if sinfo -p compute &> /dev/null 2>&1; then
            log_ok "compute 分区可访问"
        else
            log_warning "compute 分区不可访问"
        fi
    else
        log_error "sinfo 命令不可用"
    fi
else
    log_error "sbatch 命令不可用 - 请确保在 HPC 登录节点运行"
fi
echo ""

# 7. 检查 Python 环境
log_check "检查 Python 环境..."
if command -v python &> /dev/null || command -v python3 &> /dev/null; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_ok "Python $PYTHON_VERSION"
    
    # 检查 conda
    if command -v conda &> /dev/null; then
        log_ok "conda 可用"
    else
        log_warning "conda 不可用 - 可能需要加载模块: module load anaconda3"
    fi
else
    log_error "Python 不可用"
fi
echo ""

# 8. 检查磁盘空间
log_check "检查磁盘空间..."
CACHE_DIR="$HOME/.cache/huggingface"
mkdir -p "$CACHE_DIR" 2>/dev/null || true

AVAILABLE_KB=$(df -k "$CACHE_DIR" | awk 'NR==2 {print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))

if [ $AVAILABLE_GB -ge 20 ]; then
    log_ok "磁盘空间充足: ${AVAILABLE_GB}GB 可用"
elif [ $AVAILABLE_GB -ge 15 ]; then
    log_warning "磁盘空间紧张: ${AVAILABLE_GB}GB 可用 (建议 ≥20GB)"
else
    log_error "磁盘空间不足: ${AVAILABLE_GB}GB 可用 (需要 ≥20GB)"
fi
echo ""

# 9. 检查网络连接
log_check "检查网络连接..."
if ping -c 1 -W 2 huggingface.co &> /dev/null; then
    log_ok "可访问 huggingface.co"
elif ping -c 1 -W 2 hf-mirror.com &> /dev/null; then
    log_ok "可访问 hf-mirror.com (镜像站)"
    log_warning "建议使用: bash run_reproduce.sh --method mirror"
elif ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
    log_warning "网络连接受限，但可访问互联网"
else
    log_error "网络连接异常"
fi
echo ""

# 10. 检查环境变量
log_check "检查环境变量..."
if [ -n "$HF_HOME" ]; then
    log_ok "HF_HOME = $HF_HOME"
else
    log_warning "HF_HOME 未设置，建议运行: bash setup_hpc.sh"
fi

if [ -n "$TRANSFORMERS_CACHE" ]; then
    log_ok "TRANSFORMERS_CACHE = $TRANSFORMERS_CACHE"
else
    log_warning "TRANSFORMERS_CACHE 未设置"
fi
echo ""

# 总结
echo "════════════════════════════════════════════════════════════════"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ 验证通过！所有检查都成功。${NC}"
    echo ""
    echo "下一步："
    echo "  1. 运行快速设置: bash setup_hpc.sh"
    echo "  2. 一键启动: bash run_reproduce.sh"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ 验证完成，有 $WARNINGS 个警告${NC}"
    echo ""
    echo "建议先解决警告，然后："
    echo "  1. 运行: bash setup_hpc.sh"
    echo "  2. 启动: bash run_reproduce.sh"
else
    echo -e "${RED}❌ 验证失败：$ERRORS 个错误，$WARNINGS 个警告${NC}"
    echo ""
    echo "请修复上述错误后重新运行验证"
    exit 1
fi
echo "════════════════════════════════════════════════════════════════"
echo ""
