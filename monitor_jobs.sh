#!/bin/bash

################################################################################
# KAVA 训练任务监控脚本（自动刷新版）
# 用法: bash monitor_jobs.sh [--auto]
#       --auto: 每 30 秒自动刷新
################################################################################

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# 检查是否自动模式
AUTO_MODE=false
if [[ "$1" == "--auto" ]]; then
    AUTO_MODE=true
    INTERVAL=30
fi

# 显示函数
show_status() {
    if [ "$AUTO_MODE" = true ]; then
        clear
    fi
    
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  KAVA 训练任务状态监控${NC}"
    if [ "$AUTO_MODE" = true ]; then
        echo -e "${BLUE}  更新时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    fi
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # 任务状态
    echo -e "${YELLOW}[任务状态]${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    job_count=$(squeue -u $USER 2>/dev/null | grep -c "kava" || echo 0)
    
    if [ "$job_count" -gt 0 ]; then
        echo -e "${GREEN}✓ 运行中的任务: ${job_count}${NC}"
        echo ""
        squeue -u $USER --format="%.10i %.12j %.8T %.10M %.10l %.6D %.15R" | head -15
    else
        echo -e "${RED}⚠ 当前无运行中的任务${NC}"
        echo ""
        echo "检查最近任务历史..."
        sacct -u $USER -S today --format=JobID,JobName,State,Elapsed,End -n | grep kava | tail -5
    fi
    echo ""
    
    # 任务统计
    echo -e "${YELLOW}[任务统计]${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -f "outputs/job_ids.txt" ]; then
        total_main=$(wc -l < outputs/job_ids.txt)
        total_jobs=$((total_main * 3))
        running=$(squeue -u $USER 2>/dev/null | grep -c " R " || echo 0)
        pending=$(squeue -u $USER 2>/dev/null | grep -c " PD " || echo 0)
        
        echo "提交的任务: ${total_main} 个主任务 (${total_jobs} 个子任务)"
        echo "运行中 (R):  ${running}"
        echo "等待中 (PD): ${pending}"
        
        if [ "$running" -gt 0 ]; then
            progress=$((running * 100 / total_jobs))
            echo -e "进度: ${GREEN}${progress}%${NC}"
        elif [ "$pending" -gt 0 ]; then
            echo -e "状态: ${YELLOW}等待资源分配${NC}"
        else
            echo -e "状态: ${CYAN}检查是否已完成${NC}"
        fi
    fi
    echo ""
    
    # 最新日志
    echo -e "${YELLOW}[最新训练日志]${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -d "outputs/logs" ]; then
        latest=$(ls -t outputs/logs/*.out 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "文件: $(basename "$latest")"
            echo ""
            tail -3 "$latest" | sed 's/^/  /'
        else
            echo "暂无日志文件"
        fi
    else
        echo "暂无训练日志"
    fi
    echo ""
    
    # 操作提示
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  快速操作${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "查看详细日志:  tail -f outputs/logs/kava_*.out"
    echo "查看所有任务:  squeue -u \$USER"
    echo "取消所有任务:  scancel -u \$USER"
    echo "收集结果:      bash collect_results.sh"
    
    if [ "$AUTO_MODE" = false ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "${CYAN}提示: 使用 'bash monitor_jobs.sh --auto' 启动自动刷新模式${NC}"
    fi
}

# 主逻辑
if [ "$AUTO_MODE" = true ]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  自动监控模式（每 ${INTERVAL} 秒刷新）${NC}"
    echo -e "${CYAN}  按 Ctrl+C 退出${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    sleep 2
    
    while true; do
        show_status
        echo ""
        echo -e "${CYAN}下次刷新: ${INTERVAL} 秒后...（按 Ctrl+C 退出）${NC}"
        sleep $INTERVAL
    done
else
    show_status
    echo ""
fi
