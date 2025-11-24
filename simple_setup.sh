#!/bin/bash
################################################################################
# 绠€鍖栫殑 HPC 鍚姩鑴氭湰锛堝鐞嗚矾寰勭┖鏍奸棶棰橈級
# 鐢ㄦ硶: bash simple_setup.sh
################################################################################

set -e

# 棰滆壊
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  KAVA 绠€鍖栭厤缃剼鏈?
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo -e "${NC}"
echo ""

# 鑾峰彇鑴氭湰鎵€鍦ㄧ洰褰曪紙澶勭悊绌烘牸锛?SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "褰撳墠鐩綍: $SCRIPT_DIR"
echo ""

# =============================================================================
# 姝ラ 1: 妫€鏌ヨ矾寰?# =============================================================================
echo -e "${BLUE}[1/5] 妫€鏌ヨ矾寰?{NC}"
if [[ "$SCRIPT_DIR" == *" "* ]]; then
    echo -e "${YELLOW}鈿?璀﹀憡: 璺緞鍖呭惈绌烘牸${NC}"
    echo "  杩欏彲鑳藉鑷翠竴浜涜剼鏈け璐?
    echo ""
    echo -e "${YELLOW}寮虹儓寤鸿閲嶅懡鍚嶇洰褰?${NC}"
    echo "  cd $(dirname "$SCRIPT_DIR")"
    echo "  mv \"$(basename "$SCRIPT_DIR")\" kava_review"
    echo "  cd kava_review"
    echo ""
    read -p "鏄惁缁х画褰撳墠閰嶇疆? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}鉁?璺緞姝ｅ父${NC}"
fi
echo ""

# =============================================================================
# 姝ラ 2: 妫€鏌?Python
# =============================================================================
echo -e "${BLUE}[2/5] 妫€鏌?Python${NC}"

# 灏濊瘯鎵惧埌 Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}鉁?鎵惧埌 Python3: $(python3 --version)${NC}"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo -e "${GREEN}鉁?鎵惧埌 Python: $(python --version)${NC}"
else
    echo -e "${RED}鉁?鏈壘鍒?Python${NC}"
    exit 1
fi
echo ""

# =============================================================================
# 姝ラ 3: 鍒涘缓铏氭嫙鐜锛堜笉渚濊禆 conda锛?# =============================================================================
echo -e "${BLUE}[3/5] 鍒涘缓 Python 铏氭嫙鐜${NC}"

VENV_DIR="venv_kava"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}鈿?铏氭嫙鐜宸插瓨鍦? $VENV_DIR${NC}"
    read -p "鏄惁閲嶆柊鍒涘缓? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo -e "${GREEN}鉁?铏氭嫙鐜宸查噸鏂板垱寤?{NC}"
    else
        echo "浣跨敤鐜版湁铏氭嫙鐜"
    fi
else
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}鉁?铏氭嫙鐜宸插垱寤?{NC}"
fi

# 婵€娲昏櫄鎷熺幆澧?source "$VENV_DIR/bin/activate"
echo -e "${GREEN}鉁?铏氭嫙鐜宸叉縺娲?{NC}"
echo ""

# =============================================================================
# 姝ラ 4: 瀹夎渚濊禆
# =============================================================================
echo -e "${BLUE}[4/5] 瀹夎渚濊禆${NC}"

# 鍗囩骇 pip
pip install --upgrade pip -q

# 瀹夎椤圭洰渚濊禆
echo "姝ｅ湪瀹夎 requirements.txt 涓殑鍖?.."
pip install -r requirements.txt

echo -e "${GREEN}鉁?渚濊禆瀹夎瀹屾垚${NC}"
echo ""

# =============================================================================
# 姝ラ 5: 閰嶇疆鐜鍙橀噺
# =============================================================================
echo -e "${BLUE}[5/5] 閰嶇疆鐜${NC}"

# 妫€鏌?HPC 鍏变韩妯″瀷搴?HPC_MODELS="/home/share/models"
if [ -d "$HPC_MODELS" ]; then
    echo -e "${GREEN}鉁?妫€娴嬪埌 HPC 鍏变韩妯″瀷搴?{NC}"
    export HF_HOME="$HPC_MODELS"
    export TRANSFORMERS_CACHE="$HPC_MODELS"
    export HF_DATASETS_CACHE="$HOME/.cache/huggingface"
    export HUGGINGFACE_HUB_OFFLINE=1
    echo "  妯″瀷璺緞: $HPC_MODELS"
    echo "  鏁版嵁闆嗙紦瀛? $HF_DATASETS_CACHE"
    echo "  绂荤嚎妯″紡: 宸插惎鐢?
else
    echo -e "${YELLOW}鈿?HPC 鍏变韩妯″瀷搴撲笉鍙敤锛屼娇鐢ㄤ釜浜虹紦瀛?{NC}"
    export HF_HOME="$HOME/.cache/huggingface"
    export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
    export HF_DATASETS_CACHE="$HOME/.cache/huggingface"
    mkdir -p "$HF_HOME"
    echo "  缂撳瓨璺緞: $HF_HOME"
fi

# 鍒涘缓蹇呰鐨勭洰褰?mkdir -p logs outputs/checkpoints outputs/results
mkdir -p "$HF_DATASETS_CACHE"
echo -e "${GREEN}鉁?杈撳嚭鐩綍宸插垱寤?{NC}"
echo ""

# =============================================================================
# 瀹屾垚
# =============================================================================
echo -e "${GREEN}"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  閰嶇疆瀹屾垚锛?
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo -e "${NC}"
echo ""
echo "铏氭嫙鐜宸叉縺娲汇€備綘鐜板湪鍙互锛?
echo ""
echo "1. 鎻愪氦 SLURM 璁粌浠诲姟锛堟帹鑽愶級锛?
echo "   sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm"
echo "   sbatch submit_all_jobs.sh  # 鎻愪氦鎵€鏈夐厤缃?
echo ""
echo "2. 鎴栬€呯洿鎺ヨ繍琛岃缁冿紙鐢ㄤ簬娴嬭瘯锛夛細"
echo "   python train.py --config configs/llama1b_aug.yaml"
echo ""
echo "3. 鐩戞帶浠诲姟鐘舵€侊細"
echo "   squeue -u rpwang"
echo "   bash monitor_jobs.sh"
echo ""
if [ -d "$HPC_MODELS" ]; then
    echo -e "${GREEN}鉁?妯″瀷宸插氨缁? HPC 鍏变韩妯″瀷搴?$HPC_MODELS${NC}"
    echo "  鏃犻渶涓嬭浇妯″瀷锛岀洿鎺ユ彁浜よ缁冧换鍔″嵆鍙紒"
else
    echo -e "${YELLOW}鈿?娉ㄦ剰: 濡傞渶涓嬭浇妯″瀷锛岃繍琛?${NC}"
    echo "   python download_from_hf.py"
fi
echo ""
echo -e "${YELLOW}娉ㄦ剰: 姣忔鐧诲綍閮介渶瑕侀噸鏂版縺娲荤幆澧?${NC}"
echo "   source \"$VENV_DIR/bin/activate\""
echo ""
echo -e "${YELLOW}鎴栬€呭皢婵€娲诲懡浠ゆ坊鍔犲埌 ~/.bashrc:${NC}"
echo "   echo 'source \"/home/rpwang/kava review/$VENV_DIR/bin/activate\"' >> ~/.bashrc"
echo "   # 娉ㄦ剰: 璺緞鍖呭惈绌烘牸锛屽繀椤讳娇鐢ㄥ紩鍙?
echo ""
echo -e "${YELLOW}馃挕 鎺ㄨ崘: 閲嶅懡鍚嶇洰褰曚互閬垮厤绌烘牸闂:${NC}"
echo "   cd /home/rpwang"
echo "   mv \"kava review\" kava_review"
echo ""
