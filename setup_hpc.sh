#!/bin/bash

################################################################################
# KAVA HPC 蹇€熻缃剼鏈?################################################################################
# 杩欎釜鑴氭湰甯姪浣犲湪 HPC 涓婂揩閫熻缃?KAVA 椤圭洰鐜
# 
# 鐢ㄦ硶:
#   bash setup_hpc.sh
#
# 鍔熻兘:
#   1. 閰嶇疆 HuggingFace 缂撳瓨鐩綍
#   2. 鍒涘缓蹇呰鐨勭洰褰曠粨鏋?#   3. 璁剧疆鏂囦欢鏉冮檺
#   4. 楠岃瘉 SLURM 鐜
#   5. 妫€鏌ョ鐩樼┖闂?################################################################################

set -e

# 棰滆壊杈撳嚭
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽"
echo "鈺?                                                               鈺?
echo "鈺?       KAVA HPC 鐜蹇€熻缃?                                  鈺?
echo "鈺?                                                               鈺?
echo "鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆"
echo -e "${NC}"

# 鑾峰彇鑴氭湰鎵€鍦ㄧ洰褰?SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}[1/6] 妫€鏌ラ」鐩洰褰?{NC}"
echo "褰撳墠鐩綍: $SCRIPT_DIR"
if [ ! -f "train.py" ]; then
    echo -e "${RED}鉁?閿欒: 鏈壘鍒?train.py锛岃纭繚鍦?KAVA 椤圭洰鏍圭洰褰曡繍琛屾鑴氭湰${NC}"
    exit 1
fi
echo -e "${GREEN}鉁?椤圭洰鐩綍姝ｇ‘${NC}"
echo ""

echo -e "${BLUE}[2/6] 閰嶇疆 HuggingFace 缂撳瓨${NC}"
HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$HF_CACHE"

# 妫€鏌ユ槸鍚﹀凡缁忛厤缃?if grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
    echo -e "${YELLOW}鈿?~/.bashrc 涓凡瀛樺湪 HF_HOME 閰嶇疆锛岃烦杩?{NC}"
else
    echo "娣诲姞鐜鍙橀噺鍒?~/.bashrc..."
    cat >> ~/.bashrc << 'EOF'

# KAVA HuggingFace 缂撳瓨閰嶇疆
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HOME/.cache/huggingface
EOF
    echo -e "${GREEN}鉁?宸叉坊鍔犲埌 ~/.bashrc${NC}"
fi

# 绔嬪嵆鐢熸晥
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE"
echo "HuggingFace 缂撳瓨鐩綍: $HF_CACHE"
echo ""

echo -e "${BLUE}[3/6] 鍒涘缓蹇呰鐨勭洰褰?{NC}"
mkdir -p logs
mkdir -p outputs/logs
mkdir -p outputs/results
mkdir -p outputs/checkpoints
echo -e "${GREEN}鉁?鐩綍缁撴瀯鍒涘缓瀹屾垚${NC}"
ls -ld logs outputs outputs/logs outputs/results outputs/checkpoints
echo ""

echo -e "${BLUE}[4/6] 璁剧疆鑴氭湰鏉冮檺${NC}"
chmod +x run_reproduce.sh
chmod +x hpc_run_all.sh
chmod +x setup_hpc.sh
chmod +x *.slurm 2>/dev/null || true
echo -e "${GREEN}鉁?鑴氭湰鏉冮檺璁剧疆瀹屾垚${NC}"
ls -l *.sh *.slurm 2>/dev/null | grep -E '\.(sh|slurm)$'
echo ""

echo -e "${BLUE}[5/6] 妫€鏌ョ鐩樼┖闂?{NC}"
echo "褰撳墠鐩綍纾佺洏浣跨敤鎯呭喌:"
df -h "$PWD" | tail -1 | awk '{print "  鍙敤绌洪棿: " $4 " / " $2}'

# 妫€鏌ユ槸鍚︽湁瓒冲绌洪棿锛堣嚦灏?20GB锛?available_kb=$(df -k "$HF_CACHE" | awk 'NR==2 {print $4}')
available_gb=$((available_kb / 1024 / 1024))
if [ $available_gb -lt 20 ]; then
    echo -e "${YELLOW}鈿?璀﹀憡: 鍙敤绌洪棿浠?${available_gb}GB锛屽缓璁嚦灏?20GB${NC}"
    echo "  KAVA 闇€瑕佷笅杞界害 19GB 鐨勬ā鍨嬫枃浠?
else
    echo -e "${GREEN}鉁?纾佺洏绌洪棿鍏呰冻 (${available_gb}GB available)${NC}"
fi
echo ""

echo -e "${BLUE}[6/6] 楠岃瘉 SLURM 鐜${NC}"
if command -v sbatch &> /dev/null; then
    echo -e "${GREEN}鉁?sbatch 鍛戒护鍙敤${NC}"
    
    # 妫€鏌ュ垎鍖?    if sinfo -p compute &> /dev/null 2>&1; then
        echo -e "${GREEN}鉁?'compute' 鍒嗗尯鍙闂?{NC}"
        echo "  鍒嗗尯淇℃伅:"
        sinfo -p compute -o "    %P %a %l %D %t %N" | head -5
    else
        echo -e "${YELLOW}鈿?鏃犳硶璁块棶 'compute' 鍒嗗尯${NC}"
    fi
    
    # 妫€鏌ラ槦鍒?    echo "  褰撳墠闃熷垪:"
    squeue -u $USER -o "    %.18i %.9P %.30j %.8T" | head -5
    if [ $(squeue -u $USER -h | wc -l) -eq 0 ]; then
        echo -e "${GREEN}    (鏃犺繍琛屼腑鐨勪换鍔?${NC}"
    fi
else
    echo -e "${RED}鉁?sbatch 鍛戒护涓嶅彲鐢紝璇风‘淇濆湪 HPC 鐧诲綍鑺傜偣杩愯${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}"
echo "鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽"
echo "鈺?                                                               鈺?
echo "鈺? 鉁?HPC 鐜璁剧疆瀹屾垚锛?                                         鈺?
echo "鈺?                                                               鈺?
echo "鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆"
echo -e "${NC}"

echo -e "${CYAN}涓嬩竴姝ユ搷浣?${NC}"
echo ""
echo -e "${YELLOW}閲嶈: 闇€瑕佸厛涓嬭浇妯″瀷 (~19GB)${NC}"
echo "  HPC 鍏叡搴撶己灏?KAVA 鎵€闇€妯″瀷锛屽繀椤讳笅杞藉埌涓汉鐩綍"
echo ""
echo "閫夐」 A: 浣跨敤鑷姩鍖栬剼鏈紙鎺ㄨ崘锛?
echo -e "  ${GREEN}bash run_reproduce.sh${NC}"
echo "  鈫?鑷姩涓嬭浇妯″瀷骞舵彁浜ゆ墍鏈夎缁冧换鍔?
echo ""
echo "閫夐」 B: 鎵嬪姩涓嬭浇妯″瀷"
echo "  鍙傝: docs/KAVA_MODEL_DOWNLOAD.md"
echo "  鐒跺悗杩愯: bash run_reproduce.sh --skip-download"
echo ""
echo "閫夐」 C: 鏌ョ湅璇︾粏鏂囨。"
echo "  蹇€熷紑濮? REPRODUCTION_CHECKLIST.md"
echo "  瀹屾暣鎸囧崡: docs/GETTING_STARTED_HPC.md"
echo ""

echo -e "${CYAN}甯哥敤鍛戒护:${NC}"
echo "  妫€鏌ラ槦鍒?    squeue -u \$USER"
echo "  鏌ョ湅鑺傜偣:    sinfo -p compute"
echo "  鐩戞帶浠诲姟:    tail -f logs/kava_*.out"
echo "  鍙栨秷浠诲姟:    scancel <job_id>"
echo ""

echo -e "${GREEN}璁剧疆瀹屾垚锛佽閲嶆柊鐧诲綍鎴栬繍琛屼互涓嬪懡浠や娇鐜鍙橀噺鐢熸晥:${NC}"
echo -e "  ${CYAN}source ~/.bashrc${NC}"
echo ""
