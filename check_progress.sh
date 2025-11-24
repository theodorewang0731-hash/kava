#!/bin/bash

################################################################################
# KAVA 杩涘害妫€鏌ヨ剼鏈?- 瀹炴椂鏌ョ湅涓嬭浇鍜屼换鍔＄姸鎬?################################################################################

# 棰滆壊
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo -e "${BLUE}          KAVA 椤圭洰杩涘害瀹炴椂鐩戞帶${NC}"
echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo ""

# =============================================================================
# 1. 妫€鏌ヨ剼鏈繍琛岀姸鎬?# =============================================================================
echo -e "${YELLOW}[1/6] 鑴氭湰杩愯鐘舵€?{NC}"
echo "----------------------------------------"

if pgrep -f "run_reproduce_venv.sh" > /dev/null; then
    echo -e "${GREEN}鉁?run_reproduce_venv.sh 姝ｅ湪杩愯${NC}"
    
    # 鏄剧ず杩涚▼淇℃伅
    ps aux | grep "run_reproduce_venv.sh" | grep -v grep | awk '{printf "  杩涚▼ID: %s, 杩愯鏃堕棿: %s, CPU: %s%%\n", $2, $10, $3}'
else
    echo -e "${RED}鉁?run_reproduce_venv.sh 鏈繍琛岋紙鍙兘宸插畬鎴愭垨鍑洪敊锛?{NC}"
fi
echo ""

# =============================================================================
# 2. 妯″瀷涓嬭浇杩涘害
# =============================================================================
echo -e "${YELLOW}[2/6] 妯″瀷涓嬭浇杩涘害${NC}"
echo "----------------------------------------"

# 妫€鏌?HuggingFace 缂撳瓨鐩綍
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    cache_size=$(du -sh "$HOME/.cache/huggingface/hub" 2>/dev/null | cut -f1)
    echo -e "${GREEN}鉁?HuggingFace 缂撳瓨鐩綍瀛樺湪${NC}"
    echo "  褰撳墠澶у皬: ${cache_size}"
    echo "  鐩爣澶у皬: ~19GB"
    
    # 璁＄畻鐧惧垎姣旓紙绮楃暐浼拌锛?    cache_mb=$(du -sm "$HOME/.cache/huggingface/hub" 2>/dev/null | cut -f1)
    target_mb=19000
    if [ "$cache_mb" -gt 0 ]; then
        percent=$((cache_mb * 100 / target_mb))
        if [ "$percent" -gt 100 ]; then
            percent=100
        fi
        echo -e "  杩涘害: ${percent}% (${cache_mb}MB / ${target_mb}MB)"
        
        # 杩涘害鏉?        bar_length=50
        filled=$((percent * bar_length / 100))
        empty=$((bar_length - filled))
        printf "  ["
        printf "%${filled}s" | tr ' ' '='
        printf "%${empty}s" | tr ' ' '-'
        printf "] ${percent}%%\n"
    fi
    
    echo ""
    echo "  宸蹭笅杞界殑妯″瀷锛?
    find "$HOME/.cache/huggingface/hub" -maxdepth 1 -type d -name "models--*" 2>/dev/null | while read dir; do
        model_name=$(basename "$dir" | sed 's/models--//' | tr '__' '/')
        model_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "    - $model_name ($model_size)"
    done
else
    echo -e "${YELLOW}鈿?HuggingFace 缂撳瓨鐩綍涓嶅瓨鍦紙涓嬭浇灏氭湭寮€濮嬶級${NC}"
fi
echo ""

# =============================================================================
# 3. 缃戠粶娲诲姩妫€鏌?# =============================================================================
echo -e "${YELLOW}[3/6] 缃戠粶涓嬭浇娲诲姩${NC}"
echo "----------------------------------------"

if pgrep -f "python.*huggingface" > /dev/null; then
    echo -e "${GREEN}鉁?妫€娴嬪埌 Python 涓嬭浇杩涚▼${NC}"
    
    # 鏄剧ず缃戠粶杩炴帴
    netstat -tn 2>/dev/null | grep ESTABLISHED | grep -E "huggingface|hf-mirror" | head -5 | while read line; do
        echo "  娲诲姩杩炴帴: $line"
    done
else
    echo -e "${YELLOW}鈿?鏈娴嬪埌娲诲姩鐨勪笅杞借繘绋?{NC}"
fi
echo ""

# =============================================================================
# 4. 鏃ュ織鏂囦欢妫€鏌?# =============================================================================
echo -e "${YELLOW}[4/6] 鏈€鏂版棩蹇椾俊鎭?{NC}"
echo "----------------------------------------"

# 妫€鏌?nohup.out
if [ -f "nohup.out" ]; then
    echo "nohup.out 鏈€鏂?5 琛?"
    tail -5 nohup.out 2>/dev/null | sed 's/^/  /'
    echo ""
fi

# 妫€鏌?outputs/logs 鐩綍
if [ -d "outputs/logs" ] && [ "$(ls -A outputs/logs 2>/dev/null)" ]; then
    echo "璁粌鏃ュ織鏂囦欢:"
    ls -lht outputs/logs/*.{out,err} 2>/dev/null | head -3 | awk '{printf "  %s %s %s\n", $9, $5, $6" "$7" "$8}'
else
    echo -e "${YELLOW}  鏆傛棤璁粌鏃ュ織锛堜换鍔″皻鏈彁浜わ級${NC}"
fi
echo ""

# =============================================================================
# 5. SLURM 浠诲姟鐘舵€?# =============================================================================
echo -e "${YELLOW}[5/6] SLURM 浠诲姟鐘舵€?{NC}"
echo "----------------------------------------"

job_count=$(squeue -u $USER 2>/dev/null | grep -c "kava" || echo 0)

if [ "$job_count" -gt 0 ]; then
    echo -e "${GREEN}鉁?鍙戠幇 ${job_count} 涓?KAVA 浠诲姟${NC}"
    echo ""
    squeue -u $USER | grep -E "JOBID|kava" | head -13
else
    echo -e "${YELLOW}鈿?褰撳墠鏃犺繍琛屼腑鐨?SLURM 浠诲姟${NC}"
    echo "  锛堟ā鍨嬩笅杞藉畬鎴愬悗鎵嶄細鎻愪氦浠诲姟锛?
fi
echo ""

# =============================================================================
# 6. 鏁翠綋杩涘害鍒ゆ柇
# =============================================================================
echo -e "${YELLOW}[6/6] 鏁翠綋杩涘害鍒ゆ柇${NC}"
echo "----------------------------------------"

# 鍒ゆ柇褰撳墠闃舵
if [ ! -d "$HOME/.cache/huggingface/hub" ] || [ "$cache_mb" -lt 1000 ]; then
    stage="馃摜 闃舵 1: 姝ｅ湪涓嬭浇妯″瀷 (0-30%)"
    next_step="绛夊緟妯″瀷涓嬭浇瀹屾垚锛岄璁¤繕闇€ 20-90 鍒嗛挓"
elif [ "$cache_mb" -lt 15000 ]; then
    stage="馃摜 闃舵 2: 妯″瀷涓嬭浇杩涜涓?(30-80%)"
    next_step="绛夊緟妯″瀷涓嬭浇瀹屾垚锛岄璁¤繕闇€ 10-40 鍒嗛挓"
elif [ "$cache_mb" -lt 19000 ]; then
    stage="馃摜 闃舵 3: 妯″瀷涓嬭浇鎺ヨ繎瀹屾垚 (80-100%)"
    next_step="绛夊緟涓嬭浇瀹屾垚骞舵彁浜や换鍔★紝棰勮杩橀渶 5-15 鍒嗛挓"
elif [ "$job_count" -eq 0 ]; then
    stage="鈿欙笍  闃舵 4: 妯″瀷宸蹭笅杞斤紝鍑嗗鎻愪氦浠诲姟"
    next_step="绛夊緟鑴氭湰鎻愪氦 SLURM 浠诲姟"
elif [ "$job_count" -gt 0 ] && [ "$job_count" -lt 12 ]; then
    stage="馃殌 闃舵 5: 浠诲姟鎻愪氦涓?($job_count/12)"
    next_step="绛夊緟鎵€鏈変换鍔℃彁浜ゅ畬鎴?
elif [ "$job_count" -eq 12 ]; then
    stage="鉁?闃舵 6: 鎵€鏈変换鍔″凡鎻愪氦锛岃缁冭繘琛屼腑"
    next_step="绛夊緟璁粌瀹屾垚锛堥璁?36-48 灏忔椂锛夛紝鍙繍琛? bash monitor_jobs.sh"
else
    stage="鉂?鐘舵€佹湭鐭?
    next_step="妫€鏌ユ棩蹇? tail -f nohup.out 鎴?outputs/logs/*.out"
fi

echo -e "${GREEN}褰撳墠闃舵: ${stage}${NC}"
echo -e "${BLUE}涓嬩竴姝? ${next_step}${NC}"
echo ""

# =============================================================================
# 蹇€熷懡浠ゅ弬鑰?# =============================================================================
echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo -e "${BLUE}           蹇€熷懡浠ゅ弬鑰?{NC}"
echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo ""
echo "瀹炴椂鏌ョ湅鑴氭湰杈撳嚭:"
echo "  tail -f nohup.out"
echo ""
echo "鏌ョ湅妯″瀷涓嬭浇璇︽儏:"
echo "  watch -n 10 'du -sh ~/.cache/huggingface/hub && ls -lh ~/.cache/huggingface/hub/models--*'"
echo ""
echo "妫€鏌ヤ换鍔＄姸鎬?"
echo "  squeue -u \$USER"
echo "  bash monitor_jobs.sh    # (浠诲姟鎻愪氦鍚庡彲鐢?"
echo ""
echo "鏌ョ湅璁粌鏃ュ織:"
echo "  tail -f outputs/logs/kava_*.out"
echo ""
echo "閲嶆柊杩愯姝ゆ鏌?"
echo "  bash check_progress.sh"
echo ""
echo -e "${YELLOW}鎻愮ず: 杩愯 'watch -n 30 bash check_progress.sh' 鍙互姣?30 绉掕嚜鍔ㄥ埛鏂?{NC}"
echo ""
