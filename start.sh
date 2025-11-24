#!/bin/bash

################################################################################
# KAVA 涓€閿惎鍔ㄨ剼鏈?- Linux HPC 鐜
################################################################################
# 姝よ剼鏈暣鍚堜簡楠岃瘉銆侀厤缃拰鍚姩鐨勫畬鏁存祦绋?#
# 鐢ㄦ硶: bash start.sh [options]
#
# 閫夐」:
#   --verify-only    浠呴獙璇侀儴缃诧紝涓嶅惎鍔ㄨ缁?#   --setup-only     浠呰缃幆澧冿紝涓嶅惎鍔ㄨ缁?#   --no-verify      璺宠繃楠岃瘉锛岀洿鎺ュ惎鍔?#   --method METHOD  妯″瀷涓嬭浇鏂规硶 (direct|proxy|mirror)
#   --skip-download  璺宠繃妯″瀷涓嬭浇
################################################################################

set -e

# 棰滆壊瀹氫箟
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# 榛樿閫夐」
VERIFY_ONLY=false
SETUP_ONLY=false
NO_VERIFY=false
METHOD=""
SKIP_DOWNLOAD=""

# 瑙ｆ瀽鍙傛暟
while [[ $# -gt 0 ]]; do
    case $1 in
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --no-verify)
            NO_VERIFY=true
            shift
            ;;
        --method)
            METHOD="--method $2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift
            ;;
        --help)
            head -n 20 "$0" | tail -n 18
            exit 0
            ;;
        *)
            echo -e "${RED}鏈煡閫夐」: $1${NC}"
            echo "浣跨敤 --help 鏌ョ湅甯姪"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
cat << "EOF"
鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
鈺?                                                               鈺?鈺?       KAVA Paper Reproduction - HPC 涓€閿惎鍔?                 鈺?鈺?                                                               鈺?鈺?       Knowledge-Augmented Verbal-Augmentation                 鈺?鈺?       Strict reproduction per paper specifications            鈺?鈺?                                                               鈺?鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆
EOF
echo -e "${NC}"
echo ""

# 姝ラ 1: 楠岃瘉閮ㄧ讲
if [ "$NO_VERIFY" = false ]; then
    echo -e "${MAGENTA}[姝ラ 1/3] 楠岃瘉閮ㄧ讲${NC}"
    echo "鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€"
    
    if [ -f "verify_deployment.sh" ]; then
        bash verify_deployment.sh
        
        if [ $? -ne 0 ]; then
            echo ""
            echo -e "${RED}楠岃瘉澶辫触锛佽淇閿欒鍚庨噸鏂拌繍琛屻€?{NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}鈿?verify_deployment.sh 鏈壘鍒帮紝璺宠繃楠岃瘉${NC}"
    fi
    
    echo ""
    
    if [ "$VERIFY_ONLY" = true ]; then
        echo -e "${GREEN}楠岃瘉瀹屾垚锛?{NC}"
        exit 0
    fi
fi

# 姝ラ 2: 蹇€熻缃?echo -e "${MAGENTA}[姝ラ 2/3] 鐜璁剧疆${NC}"
echo "鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€"

if [ -f "setup_hpc.sh" ]; then
    bash setup_hpc.sh
    
    # 浣跨幆澧冨彉閲忕敓鏁?    if [ -f ~/.bashrc ]; then
        source ~/.bashrc 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}鈿?setup_hpc.sh 鏈壘鍒帮紝璺宠繃璁剧疆${NC}"
fi

echo ""

if [ "$SETUP_ONLY" = true ]; then
    echo -e "${GREEN}璁剧疆瀹屾垚锛?{NC}"
    exit 0
fi

# 姝ラ 3: 鍚姩璁粌
echo -e "${MAGENTA}[姝ラ 3/3] 鍚姩璁粌浠诲姟${NC}"
echo "鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€"

if [ -f "run_reproduce.sh" ]; then
    echo -e "${CYAN}姝ｅ湪鍚姩鑷姩鍖栬缁冩祦绋?..${NC}"
    echo ""
    
    # 鏋勫缓鍛戒护
    CMD="bash run_reproduce.sh"
    [ -n "$METHOD" ] && CMD="$CMD $METHOD"
    [ -n "$SKIP_DOWNLOAD" ] && CMD="$CMD $SKIP_DOWNLOAD"
    
    echo -e "${BLUE}鎵ц鍛戒护: $CMD${NC}"
    echo ""
    
    # 鎵ц
    $CMD
    
    EXIT_CODE=$?
    
    echo ""
    echo "鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}"
        cat << "EOF"
鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
鈺?                                                               鈺?鈺? 鉁?鍚姩瀹屾垚锛佽缁冧换鍔″凡鎻愪氦鍒?SLURM 闃熷垪                     鈺?鈺?                                                               鈺?鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆
EOF
        echo -e "${NC}"
        
        echo ""
        echo -e "${CYAN}涓嬩竴姝ユ搷浣滐細${NC}"
        echo ""
        echo "1. 鐩戞帶浠诲姟杩涘害锛?
        echo -e "   ${GREEN}bash monitor_jobs.sh${NC}"
        echo ""
        echo "2. 鏌ョ湅闃熷垪鐘舵€侊細"
        echo -e "   ${GREEN}squeue -u \$USER${NC}"
        echo ""
        echo "3. 鏌ョ湅瀹炴椂鏃ュ織锛?
        echo -e "   ${GREEN}tail -f outputs/logs/llama1b_aug_seed42.log${NC}"
        echo ""
        echo "4. 璁粌瀹屾垚鍚庢敹闆嗙粨鏋滐細"
        echo -e "   ${GREEN}bash collect_results.sh${NC}"
        echo ""
        echo -e "${YELLOW}棰勮鏃堕棿锛?{NC}"
        echo "  - 妯″瀷涓嬭浇: 17-100 鍒嗛挓锛堝鏈烦杩囷級"
        echo "  - 璁粌浠诲姟: 36-48 灏忔椂锛堝苟琛屾墽琛岋級"
        echo ""
    else
        echo -e "${RED}"
        cat << "EOF"
鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
鈺?                                                               鈺?鈺? 鉂?鍚姩澶辫触                                                   鈺?鈺?                                                               鈺?鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆
EOF
        echo -e "${NC}"
        
        echo ""
        echo -e "${YELLOW}鏁呴殰鎺掗櫎寤鸿锛?{NC}"
        echo ""
        echo "1. 妫€鏌ユ棩蹇楄緭鍑轰腑鐨勯敊璇俊鎭?
        echo "2. 楠岃瘉纾佺洏绌洪棿: df -h \$HOME"
        echo "3. 妫€鏌ョ綉缁滆繛鎺? ping huggingface.co"
        echo "4. 鏌ョ湅璇︾粏鏂囨。: docs/GETTING_STARTED_HPC.md"
        echo ""
        exit $EXIT_CODE
    fi
else
    echo -e "${RED}鉁?run_reproduce.sh 鏈壘鍒?{NC}"
    exit 1
fi
