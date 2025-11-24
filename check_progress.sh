#!/bin/bash

################################################################################
# KAVA 进度检查脚本 - 实时查看下载和任务状态
################################################################################

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}          KAVA 项目进度实时监控${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# =============================================================================
# 1. 检查脚本运行状态
# =============================================================================
echo -e "${YELLOW}[1/6] 脚本运行状态${NC}"
echo "----------------------------------------"

if pgrep -f "run_reproduce_venv.sh" > /dev/null; then
    echo -e "${GREEN}✓ run_reproduce_venv.sh 正在运行${NC}"
    
    # 显示进程信息
    ps aux | grep "run_reproduce_venv.sh" | grep -v grep | awk '{printf "  进程ID: %s, 运行时间: %s, CPU: %s%%\n", $2, $10, $3}'
else
    echo -e "${RED}✗ run_reproduce_venv.sh 未运行（可能已完成或出错）${NC}"
fi
echo ""

# =============================================================================
# 2. 模型下载进度
# =============================================================================
echo -e "${YELLOW}[2/6] 模型下载进度${NC}"
echo "----------------------------------------"

# 检查 HuggingFace 缓存目录
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    cache_size=$(du -sh "$HOME/.cache/huggingface/hub" 2>/dev/null | cut -f1)
    echo -e "${GREEN}✓ HuggingFace 缓存目录存在${NC}"
    echo "  当前大小: ${cache_size}"
    echo "  目标大小: ~19GB"
    
    # 计算百分比（粗略估计）
    cache_mb=$(du -sm "$HOME/.cache/huggingface/hub" 2>/dev/null | cut -f1)
    target_mb=19000
    if [ "$cache_mb" -gt 0 ]; then
        percent=$((cache_mb * 100 / target_mb))
        if [ "$percent" -gt 100 ]; then
            percent=100
        fi
        echo -e "  进度: ${percent}% (${cache_mb}MB / ${target_mb}MB)"
        
        # 进度条
        bar_length=50
        filled=$((percent * bar_length / 100))
        empty=$((bar_length - filled))
        printf "  ["
        printf "%${filled}s" | tr ' ' '='
        printf "%${empty}s" | tr ' ' '-'
        printf "] ${percent}%%\n"
    fi
    
    echo ""
    echo "  已下载的模型："
    find "$HOME/.cache/huggingface/hub" -maxdepth 1 -type d -name "models--*" 2>/dev/null | while read dir; do
        model_name=$(basename "$dir" | sed 's/models--//' | tr '__' '/')
        model_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "    - $model_name ($model_size)"
    done
else
    echo -e "${YELLOW}⚠ HuggingFace 缓存目录不存在（下载尚未开始）${NC}"
fi
echo ""

# =============================================================================
# 3. 网络活动检查
# =============================================================================
echo -e "${YELLOW}[3/6] 网络下载活动${NC}"
echo "----------------------------------------"

if pgrep -f "python.*huggingface" > /dev/null; then
    echo -e "${GREEN}✓ 检测到 Python 下载进程${NC}"
    
    # 显示网络连接
    netstat -tn 2>/dev/null | grep ESTABLISHED | grep -E "huggingface|hf-mirror" | head -5 | while read line; do
        echo "  活动连接: $line"
    done
else
    echo -e "${YELLOW}⚠ 未检测到活动的下载进程${NC}"
fi
echo ""

# =============================================================================
# 4. 日志文件检查
# =============================================================================
echo -e "${YELLOW}[4/6] 最新日志信息${NC}"
echo "----------------------------------------"

# 检查 nohup.out
if [ -f "nohup.out" ]; then
    echo "nohup.out 最新 5 行:"
    tail -5 nohup.out 2>/dev/null | sed 's/^/  /'
    echo ""
fi

# 检查 outputs/logs 目录
if [ -d "outputs/logs" ] && [ "$(ls -A outputs/logs 2>/dev/null)" ]; then
    echo "训练日志文件:"
    ls -lht outputs/logs/*.{out,err} 2>/dev/null | head -3 | awk '{printf "  %s %s %s\n", $9, $5, $6" "$7" "$8}'
else
    echo -e "${YELLOW}  暂无训练日志（任务尚未提交）${NC}"
fi
echo ""

# =============================================================================
# 5. SLURM 任务状态
# =============================================================================
echo -e "${YELLOW}[5/6] SLURM 任务状态${NC}"
echo "----------------------------------------"

job_count=$(squeue -u $USER 2>/dev/null | grep -c "kava" || echo 0)

if [ "$job_count" -gt 0 ]; then
    echo -e "${GREEN}✓ 发现 ${job_count} 个 KAVA 任务${NC}"
    echo ""
    squeue -u $USER | grep -E "JOBID|kava" | head -13
else
    echo -e "${YELLOW}⚠ 当前无运行中的 SLURM 任务${NC}"
    echo "  （模型下载完成后才会提交任务）"
fi
echo ""

# =============================================================================
# 6. 整体进度判断
# =============================================================================
echo -e "${YELLOW}[6/6] 整体进度判断${NC}"
echo "----------------------------------------"

# 判断当前阶段
if [ ! -d "$HOME/.cache/huggingface/hub" ] || [ "$cache_mb" -lt 1000 ]; then
    stage="📥 阶段 1: 正在下载模型 (0-30%)"
    next_step="等待模型下载完成，预计还需 20-90 分钟"
elif [ "$cache_mb" -lt 15000 ]; then
    stage="📥 阶段 2: 模型下载进行中 (30-80%)"
    next_step="等待模型下载完成，预计还需 10-40 分钟"
elif [ "$cache_mb" -lt 19000 ]; then
    stage="📥 阶段 3: 模型下载接近完成 (80-100%)"
    next_step="等待下载完成并提交任务，预计还需 5-15 分钟"
elif [ "$job_count" -eq 0 ]; then
    stage="⚙️  阶段 4: 模型已下载，准备提交任务"
    next_step="等待脚本提交 SLURM 任务"
elif [ "$job_count" -gt 0 ] && [ "$job_count" -lt 12 ]; then
    stage="🚀 阶段 5: 任务提交中 ($job_count/12)"
    next_step="等待所有任务提交完成"
elif [ "$job_count" -eq 12 ]; then
    stage="✅ 阶段 6: 所有任务已提交，训练进行中"
    next_step="等待训练完成（预计 36-48 小时），可运行: bash monitor_jobs.sh"
else
    stage="❓ 状态未知"
    next_step="检查日志: tail -f nohup.out 或 outputs/logs/*.out"
fi

echo -e "${GREEN}当前阶段: ${stage}${NC}"
echo -e "${BLUE}下一步: ${next_step}${NC}"
echo ""

# =============================================================================
# 快速命令参考
# =============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}           快速命令参考${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "实时查看脚本输出:"
echo "  tail -f nohup.out"
echo ""
echo "查看模型下载详情:"
echo "  watch -n 10 'du -sh ~/.cache/huggingface/hub && ls -lh ~/.cache/huggingface/hub/models--*'"
echo ""
echo "检查任务状态:"
echo "  squeue -u \$USER"
echo "  bash monitor_jobs.sh    # (任务提交后可用)"
echo ""
echo "查看训练日志:"
echo "  tail -f outputs/logs/kava_*.out"
echo ""
echo "重新运行此检查:"
echo "  bash check_progress.sh"
echo ""
echo -e "${YELLOW}提示: 运行 'watch -n 30 bash check_progress.sh' 可以每 30 秒自动刷新${NC}"
echo ""
