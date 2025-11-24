#!/bin/bash
################################################################################
# HPC 资源安全检查脚本
################################################################################
# 用途：在运行训练前检查磁盘配额和资源使用情况
# 使用：bash check_hpc_quota.sh
################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════════"
echo "  HPC 资源安全检查"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""

# =============================================================================
# 1. 磁盘空间检查
# =============================================================================
echo -e "${BLUE}[1/5] 磁盘空间检查${NC}"
echo ""

# HOME 目录空间
HOME_USAGE=$(df -h $HOME | awk 'NR==2 {print $5}' | sed 's/%//')
HOME_AVAIL=$(df -h $HOME | awk 'NR==2 {print $4}')

echo "HOME 目录 ($HOME):"
echo "  使用率: ${HOME_USAGE}%"
echo "  可用空间: ${HOME_AVAIL}"

if [ $HOME_USAGE -gt 90 ]; then
    echo -e "${RED}✗ 警告: 磁盘使用率过高 (${HOME_USAGE}%)${NC}"
    echo "  建议清理不需要的文件"
elif [ $HOME_USAGE -gt 80 ]; then
    echo -e "${YELLOW}⚠ 注意: 磁盘使用率较高 (${HOME_USAGE}%)${NC}"
else
    echo -e "${GREEN}✓ 磁盘空间充足${NC}"
fi
echo ""

# HuggingFace 缓存目录
HF_CACHE="${HOME}/.cache/huggingface"
if [ -d "$HF_CACHE" ]; then
    HF_SIZE=$(du -sh "$HF_CACHE" 2>/dev/null | cut -f1)
    echo "HuggingFace 缓存: $HF_SIZE"
fi

# 项目输出目录
if [ -d "outputs" ]; then
    OUTPUT_SIZE=$(du -sh outputs 2>/dev/null | cut -f1)
    echo "项目输出目录: $OUTPUT_SIZE"
fi
echo ""

# =============================================================================
# 2. 配额检查（如果 HPC 有配额系统）
# =============================================================================
echo -e "${BLUE}[2/5] 配额检查${NC}"
echo ""

if command -v quota &> /dev/null; then
    quota -s
    echo -e "${GREEN}✓ 配额信息见上${NC}"
else
    echo "此 HPC 系统未配置配额检查"
fi
echo ""

# =============================================================================
# 3. SLURM 任务检查
# =============================================================================
echo -e "${BLUE}[3/5] SLURM 任务状态${NC}"
echo ""

if command -v squeue &> /dev/null; then
    RUNNING=$(squeue -u $USER -t RUNNING | wc -l)
    PENDING=$(squeue -u $USER -t PENDING | wc -l)
    
    # 减1是因为表头行
    RUNNING=$((RUNNING - 1))
    PENDING=$((PENDING - 1))
    
    echo "当前任务:"
    echo "  运行中: $RUNNING"
    echo "  等待中: $PENDING"
    echo "  总计: $((RUNNING + PENDING))"
    
    if [ $RUNNING -gt 0 ] || [ $PENDING -gt 0 ]; then
        echo ""
        echo "任务详情:"
        squeue -u $USER --format="%.10i %.15j %.8T %.10M %.6D %.20R"
    fi
    
    # 检查任务数量是否过多
    TOTAL=$((RUNNING + PENDING))
    if [ $TOTAL -gt 20 ]; then
        echo -e "${YELLOW}⚠ 注意: 任务数量较多 ($TOTAL)${NC}"
        echo "  建议等待部分任务完成后再提交新任务"
    elif [ $TOTAL -gt 0 ]; then
        echo -e "${GREEN}✓ 任务数量正常${NC}"
    else
        echo -e "${GREEN}✓ 无运行中的任务${NC}"
    fi
else
    echo -e "${RED}✗ SLURM 命令不可用${NC}"
fi
echo ""

# =============================================================================
# 4. GPU 资源检查
# =============================================================================
echo -e "${BLUE}[4/5] GPU 可用性${NC}"
echo ""

if command -v sinfo &> /dev/null; then
    echo "GPU 分区状态:"
    sinfo -p compute --format="%.10P %.5a %.10l %.6D %.20N %.20C" 2>/dev/null || \
    sinfo --format="%.10P %.5a %.10l %.6D %.20N %.20C"
    echo ""
    echo -e "${GREEN}✓ GPU 分区信息见上${NC}"
else
    echo "无法查询 GPU 状态"
fi
echo ""

# =============================================================================
# 5. 项目文件检查
# =============================================================================
echo -e "${BLUE}[5/5] 项目文件完整性${NC}"
echo ""

REQUIRED_FILES=(
    "train.py"
    "requirements.txt"
    "submit_multi_seed.slurm"
    "configs/llama1b_aug.yaml"
)

ALL_OK=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (缺失)"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = true ]; then
    echo ""
    echo -e "${GREEN}✓ 所有必需文件存在${NC}"
fi
echo ""

# =============================================================================
# 摘要和建议
# =============================================================================
echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════════"
echo "  检查完成"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""

echo "安全提醒:"
echo "  1. 所有操作仅限于您的 HOME 目录 ($HOME)"
echo "  2. 不会影响其他用户的文件或进程"
echo "  3. 资源使用受 SLURM 配额限制"
echo "  4. 所有计算任务通过 SLURM 调度"
echo ""

if [ $HOME_USAGE -gt 80 ] || [ $TOTAL -gt 15 ]; then
    echo -e "${YELLOW}建议操作:${NC}"
    if [ $HOME_USAGE -gt 80 ]; then
        echo "  • 清理不需要的缓存: huggingface-cli delete-cache"
        echo "  • 删除旧的训练输出: rm -rf outputs/old_experiment"
    fi
    if [ $TOTAL -gt 15 ]; then
        echo "  • 等待部分任务完成后再提交新任务"
    fi
    echo ""
fi

echo "如果一切正常，可以运行:"
echo "  bash simple_setup.sh          # 配置环境"
echo "  bash submit_all_jobs.sh       # 提交训练任务"
echo "  bash monitor_jobs.sh          # 监控任务状态"
echo ""
