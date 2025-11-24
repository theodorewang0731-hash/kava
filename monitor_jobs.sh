#!/bin/bash

################################################################################
# KAVA 璁粌浠诲姟鐩戞帶鑴氭湰锛堣嚜鍔ㄥ埛鏂扮増锛?# 鐢ㄦ硶: bash monitor_jobs.sh [--auto]
#       --auto: 姣?30 绉掕嚜鍔ㄥ埛鏂?################################################################################

# 棰滆壊
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# 妫€鏌ユ槸鍚﹁嚜鍔ㄦā寮?AUTO_MODE=false
if [[ "$1" == "--auto" ]]; then
    AUTO_MODE=true
    INTERVAL=30
fi

# 鏄剧ず鍑芥暟
show_status() {
    if [ "$AUTO_MODE" = true ]; then
        clear
    fi
    
    echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
    echo -e "${BLUE}  KAVA 璁粌浠诲姟鐘舵€佺洃鎺?{NC}"
    if [ "$AUTO_MODE" = true ]; then
        echo -e "${BLUE}  鏇存柊鏃堕棿: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    fi
    echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
    echo ""
    
    # 浠诲姟鐘舵€?    echo -e "${YELLOW}[浠诲姟鐘舵€乚${NC}"
    echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
    
    job_count=$(squeue -u $USER 2>/dev/null | grep -c "kava" || echo 0)
    
    if [ "$job_count" -gt 0 ]; then
        echo -e "${GREEN}鉁?杩愯涓殑浠诲姟: ${job_count}${NC}"
        echo ""
        squeue -u $USER --format="%.10i %.12j %.8T %.10M %.10l %.6D %.15R" | head -15
    else
        echo -e "${RED}鈿?褰撳墠鏃犺繍琛屼腑鐨勪换鍔?{NC}"
        echo ""
        echo "妫€鏌ユ渶杩戜换鍔″巻鍙?.."
        sacct -u $USER -S today --format=JobID,JobName,State,Elapsed,End -n | grep kava | tail -5
    fi
    echo ""
    
    # 浠诲姟缁熻
    echo -e "${YELLOW}[浠诲姟缁熻]${NC}"
    echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
    
    if [ -f "outputs/job_ids.txt" ]; then
        total_main=$(wc -l < outputs/job_ids.txt)
        total_jobs=$((total_main * 3))
        running=$(squeue -u $USER 2>/dev/null | grep -c " R " || echo 0)
        pending=$(squeue -u $USER 2>/dev/null | grep -c " PD " || echo 0)
        
        echo "鎻愪氦鐨勪换鍔? ${total_main} 涓富浠诲姟 (${total_jobs} 涓瓙浠诲姟)"
        echo "杩愯涓?(R):  ${running}"
        echo "绛夊緟涓?(PD): ${pending}"
        
        if [ "$running" -gt 0 ]; then
            progress=$((running * 100 / total_jobs))
            echo -e "杩涘害: ${GREEN}${progress}%${NC}"
        elif [ "$pending" -gt 0 ]; then
            echo -e "鐘舵€? ${YELLOW}绛夊緟璧勬簮鍒嗛厤${NC}"
        else
            echo -e "鐘舵€? ${CYAN}妫€鏌ユ槸鍚﹀凡瀹屾垚${NC}"
        fi
    fi
    echo ""
    
    # 鏈€鏂版棩蹇?    echo -e "${YELLOW}[鏈€鏂拌缁冩棩蹇梋${NC}"
    echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
    
    if [ -d "outputs/logs" ]; then
        latest=$(ls -t outputs/logs/*.out 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "鏂囦欢: $(basename "$latest")"
            echo ""
            tail -3 "$latest" | sed 's/^/  /'
        else
            echo "鏆傛棤鏃ュ織鏂囦欢"
        fi
    else
        echo "鏆傛棤璁粌鏃ュ織"
    fi
    echo ""
    
    # 鎿嶄綔鎻愮ず
    echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
    echo -e "${BLUE}  蹇€熸搷浣?{NC}"
    echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
    echo ""
    echo "鏌ョ湅璇︾粏鏃ュ織:  tail -f outputs/logs/kava_*.out"
    echo "鏌ョ湅鎵€鏈変换鍔?  squeue -u \$USER"
    echo "鍙栨秷鎵€鏈変换鍔?  scancel -u \$USER"
    echo "鏀堕泦缁撴灉:      bash collect_results.sh"
    
    if [ "$AUTO_MODE" = false ]; then
        echo ""
        echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
        echo -e "${CYAN}鎻愮ず: 浣跨敤 'bash monitor_jobs.sh --auto' 鍚姩鑷姩鍒锋柊妯″紡${NC}"
    fi
}

# 涓婚€昏緫
if [ "$AUTO_MODE" = true ]; then
    echo -e "${CYAN}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
    echo -e "${CYAN}  鑷姩鐩戞帶妯″紡锛堟瘡 ${INTERVAL} 绉掑埛鏂帮級${NC}"
    echo -e "${CYAN}  鎸?Ctrl+C 閫€鍑?{NC}"
    echo -e "${CYAN}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
    echo ""
    sleep 2
    
    while true; do
        show_status
        echo ""
        echo -e "${CYAN}涓嬫鍒锋柊: ${INTERVAL} 绉掑悗...锛堟寜 Ctrl+C 閫€鍑猴級${NC}"
        sleep $INTERVAL
    done
else
    show_status
    echo ""
fi
