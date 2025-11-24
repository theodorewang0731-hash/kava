#!/bin/bash

################################################################################
# KAVA 自动监控脚本 - 每 30 秒刷新一次
################################################################################

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# 刷新间隔（秒）
INTERVAL=30

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  KAVA 训练自动监控（每 ${INTERVAL} 秒刷新）${NC}"
echo -e "${CYAN}  按 Ctrl+C 退出${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

while true; do
    # 清屏
    clear
    
    # 显示时间
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  更新时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # =============================================================================
    # 1. SLURM 任务状态
    # =============================================================================
    echo -e "${YELLOW}[1/5] SLURM 任务状态${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    job_count=$(squeue -u $USER 2>/dev/null | grep -c "kava" || echo 0)
    
    if [ "$job_count" -gt 0 ]; then
        echo -e "${GREEN}✓ 发现 ${job_count} 个 KAVA 任务${NC}"
        echo ""
        squeue -u $USER --format="%.10i %.12j %.8T %.10M %.10l %.6D %.15R" | head -15
    else
        echo -e "${RED}⚠ 当前无运行中的任务${NC}"
        echo "检查最近的任务历史..."
        echo ""
        sacct -u $USER -S today --format=JobID,JobName,State,Elapsed,End -n | grep kava | tail -5
    fi
    echo ""
    
    # =============================================================================
    # 2. 任务统计
    # =============================================================================
    echo -e "${YELLOW}[2/5] 任务统计${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -f "outputs/job_ids.txt" ]; then
        total_main_jobs=$(wc -l < outputs/job_ids.txt)
        total_jobs=$((total_main_jobs * 3))  # 每个主任务 3 个子任务
        
        running=$(squeue -u $USER 2>/dev/null | grep -c " R " || echo 0)
        pending=$(squeue -u $USER 2>/dev/null | grep -c " PD " || echo 0)
        
        echo "提交的任务: ${total_main_jobs} 个主任务 (${total_jobs} 个子任务)"
        echo "运行中 (R):  ${running}"
        echo "等待中 (PD): ${pending}"
        
        if [ "$running" -gt 0 ]; then
            progress=$((running * 100 / total_jobs))
            echo -e "进度: ${GREEN}${progress}%${NC} (${running}/${total_jobs} 任务运行中)"
        elif [ "$pending" -gt 0 ]; then
            echo -e "状态: ${YELLOW}等待资源分配${NC}"
        else
            echo -e "状态: ${CYAN}任务可能已完成或失败${NC}"
        fi
    else
        echo "未找到任务记录文件"
    fi
    echo ""
    
    # =============================================================================
    # 3. 最新日志信息
    # =============================================================================
    echo -e "${YELLOW}[3/5] 最新训练日志${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -d "outputs/logs" ] && [ "$(ls -A outputs/logs/*.out 2>/dev/null)" ]; then
        latest_log=$(ls -t outputs/logs/*.out 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo "最新日志: $(basename "$latest_log")"
            echo ""
            tail -5 "$latest_log" | sed 's/^/  /'
        else
            echo "暂无日志文件"
        fi
    else
        echo "暂无训练日志（任务尚未开始）"
    fi
    echo ""
    
    # =============================================================================
    # 4. GPU 节点信息
    # =============================================================================
    echo -e "${YELLOW}[4/5] GPU 使用情况${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ "$job_count" -gt 0 ]; then
        # 获取运行节点
        nodes=$(squeue -u $USER -o "%N" -h | grep -v "None assigned" | sort -u)
        if [ -n "$nodes" ]; then
            echo "运行节点: $nodes"
        else
            echo "等待节点分配..."
        fi
    else
        echo "无活动任务"
    fi
    echo ""
    
    # =============================================================================
    # 5. 预计完成时间
    # =============================================================================
    echo -e "${YELLOW}[5/5] 时间估算${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ "$running" -gt 0 ]; then
        # 获取最早启动的任务时间
        oldest_start=$(sacct -u $USER -S today --format=JobID,Start,State -n | grep RUNNING | head -1 | awk '{print $2}')
        if [ -n "$oldest_start" ]; then
            echo "训练开始: $oldest_start"
            echo "预计完成: 36-48 小时后"
        fi
    elif [ "$pending" -gt 0 ]; then
        echo "任务排队中，等待资源分配"
        echo "开始后预计: 36-48 小时完成"
    else
        echo "检查任务是否已完成或失败"
    fi
    echo ""
    
    # =============================================================================
    # 快速操作提示
    # =============================================================================
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  快速操作${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "查看详细日志:  tail -f outputs/logs/kava_*.out"
    echo "查看错误日志:  tail -f outputs/logs/kava_*.err"
    echo "手动检查状态:  squeue -u \$USER"
    echo "取消所有任务:  scancel -u \$USER"
    echo ""
    echo -e "${CYAN}下次刷新: ${INTERVAL} 秒后...（按 Ctrl+C 退出）${NC}"
    
    # 等待
    sleep $INTERVAL
done
