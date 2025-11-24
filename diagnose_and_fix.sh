#!/bin/bash
################################################################################
# HPC 鏁呴殰璇婃柇鍜屼慨澶嶈剼鏈?# 鐢ㄦ硶: bash diagnose_and_fix.sh
################################################################################

set +e  # 涓嶈鍦ㄩ敊璇椂閫€鍑猴紝鎴戜滑闇€瑕佹敹闆嗘墍鏈変俊鎭?
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  KAVA HPC 鏁呴殰璇婃柇"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo -e "${NC}"

# 璁板綍鎵€鏈夎緭鍑哄埌鏃ュ織鏂囦欢
LOG_FILE="diagnostic_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "璇婃柇鏃ュ織灏嗕繚瀛樺埌: $LOG_FILE"
echo ""

# =============================================================================
# 1. 妫€鏌ュ綋鍓嶇洰褰?# =============================================================================
echo -e "${BLUE}[1/10] 妫€鏌ュ綋鍓嶇洰褰?{NC}"
echo "褰撳墠宸ヤ綔鐩綍: $(pwd)"
echo "鐩綍鍐呭:"
ls -la | head -20

if [ ! -f "train.py" ]; then
    echo -e "${RED}鉁?鏈壘鍒?train.py${NC}"
    echo -e "${YELLOW}  鍙兘鍘熷洜: 鐩綍鍚嶅寘鍚┖鏍煎鑷磋矾寰勯敊璇?{NC}"
    echo -e "${YELLOW}  浣犵殑鐩綍: '/home/rpwang/kava review'${NC}"
    echo -e "${YELLOW}  鍖呭惈绌烘牸鐨勭洰褰曞湪 Linux 涓渶瑕佺壒娈婂鐞?{NC}"
else
    echo -e "${GREEN}鉁?train.py 瀛樺湪${NC}"
fi
echo ""

# =============================================================================
# 2. 妫€鏌ヨ剼鏈枃浠跺拰鏉冮檺
# =============================================================================
echo -e "${BLUE}[2/10] 妫€鏌ヨ剼鏈枃浠舵潈闄?{NC}"
for script in setup_hpc_env.sh run_reproduce.sh setup_hpc.sh; do
    if [ -f "$script" ]; then
        perms=$(ls -l "$script" | awk '{print $1}')
        echo "  $script: $perms"
        if [[ ! -x "$script" ]]; then
            echo -e "${YELLOW}    鈿?涓嶅彲鎵ц锛屾鍦ㄤ慨澶?..${NC}"
            chmod +x "$script"
        fi
    else
        echo -e "${RED}  鉁?$script 涓嶅瓨鍦?{NC}"
    fi
done
echo ""

# =============================================================================
# 3. 妫€鏌?Module 绯荤粺
# =============================================================================
echo -e "${BLUE}[3/10] 妫€鏌?Module 绯荤粺${NC}"
if [ -f /usr/share/modules/init/bash ]; then
    echo -e "${GREEN}鉁?Module 鍒濆鍖栨枃浠跺瓨鍦?{NC}"
    . /usr/share/modules/init/bash
    
    if command -v module &> /dev/null; then
        echo -e "${GREEN}鉁?module 鍛戒护鍙敤${NC}"
        
        # 妫€鏌?modulefiles 璺緞
        if [ -d /home/share/modules/modulefiles ]; then
            echo -e "${GREEN}鉁?modulefiles 鐩綍瀛樺湪${NC}"
            module use --append /home/share/modules/modulefiles
        else
            echo -e "${YELLOW}鈿?modulefiles 鐩綍涓嶅瓨鍦? /home/share/modules/modulefiles${NC}"
        fi
    else
        echo -e "${RED}鉁?module 鍛戒护涓嶅彲鐢?{NC}"
    fi
else
    echo -e "${RED}鉁?Module 绯荤粺鏈壘鍒?{NC}"
    echo "  浣犵殑 HPC 鍙兘涓嶄娇鐢?Environment Modules"
fi
echo ""

# =============================================================================
# 4. 妫€鏌?Anaconda/Conda
# =============================================================================
echo -e "${BLUE}[4/10] 妫€鏌?Conda 鍙敤鎬?{NC}"

# 灏濊瘯鍔犺浇 anaconda3
if command -v module &> /dev/null; then
    echo "灏濊瘯鍔犺浇 anaconda3 妯″潡..."
    module load anaconda3 2>&1 || echo "  鍔犺浇澶辫触"
fi

# 妫€鏌?conda
if command -v conda &> /dev/null; then
    echo -e "${GREEN}鉁?Conda 鍙敤: $(conda --version)${NC}"
    echo "  Conda 璺緞: $(which conda)"
else
    echo -e "${RED}鉁?Conda 涓嶅彲鐢?{NC}"
    echo ""
    echo "鍙兘鐨?conda 浣嶇疆:"
    find /opt /usr/local /home/share -name "conda" -type f 2>/dev/null | head -5
    echo ""
    echo -e "${YELLOW}鏇夸唬鏂规: 浣跨敤绯荤粺 Python${NC}"
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}鉁?Python3 鍙敤: $(python3 --version)${NC}"
    fi
fi
echo ""

# =============================================================================
# 5. 妫€鏌?Python 鐜
# =============================================================================
echo -e "${BLUE}[5/10] 妫€鏌?Python 鐜${NC}"
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}鉁?Python3: $(python3 --version)${NC}"
    echo "  璺緞: $(which python3)"
elif command -v python &> /dev/null; then
    echo -e "${GREEN}鉁?Python: $(python --version)${NC}"
    echo "  璺緞: $(which python)"
else
    echo -e "${RED}鉁?鏈壘鍒?Python${NC}"
fi

# 妫€鏌?pip
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}鉁?pip3 鍙敤${NC}"
elif command -v pip &> /dev/null; then
    echo -e "${GREEN}鉁?pip 鍙敤${NC}"
else
    echo -e "${RED}鉁?pip 涓嶅彲鐢?{NC}"
fi
echo ""

# =============================================================================
# 6. 妫€鏌ョ鐩樼┖闂?# =============================================================================
echo -e "${BLUE}[6/10] 妫€鏌ョ鐩樼┖闂?{NC}"
df -h "$HOME" | grep -v "Filesystem"
echo ""
AVAIL_GB=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -gt 20 ]; then
    echo -e "${GREEN}鉁?纾佺洏绌洪棿鍏呰冻 (${AVAIL_GB}GB 鍙敤锛岄渶瑕?~20GB)${NC}"
else
    echo -e "${YELLOW}鈿?纾佺洏绌洪棿鍙兘涓嶈冻 (${AVAIL_GB}GB 鍙敤锛屽缓璁?>20GB)${NC}"
fi
echo ""

# =============================================================================
# 7. 妫€鏌ョ綉缁滆繛鎺?# =============================================================================
echo -e "${BLUE}[7/10] 妫€鏌ョ綉缁滆繛鎺?{NC}"
if ping -c 1 huggingface.co &> /dev/null; then
    echo -e "${GREEN}鉁?鍙互璁块棶 huggingface.co${NC}"
else
    echo -e "${YELLOW}鈿?鏃犳硶璁块棶 huggingface.co (鍙兘闇€瑕佷唬鐞嗘垨浣跨敤闀滃儚)${NC}"
fi
echo ""

# =============================================================================
# 8. 妫€鏌?SLURM
# =============================================================================
echo -e "${BLUE}[8/10] 妫€鏌?SLURM${NC}"
if command -v sbatch &> /dev/null; then
    echo -e "${GREEN}鉁?SLURM 鍙敤${NC}"
    echo "  sbatch: $(which sbatch)"
    echo "  squeue: $(which squeue)"
else
    echo -e "${RED}鉁?SLURM 涓嶅彲鐢?{NC}"
    echo "  姝ゆ湇鍔″櫒鍙兘涓嶆槸 SLURM 闆嗙兢"
fi
echo ""

# =============================================================================
# 9. 妫€鏌ヤ緷璧栨枃浠?# =============================================================================
echo -e "${BLUE}[9/10] 妫€鏌ュ繀闇€鏂囦欢${NC}"
REQUIRED_FILES=(
    "requirements.txt"
    "train.py"
    "evaluate.py"
    "src/trainer.py"
    "src/losses.py"
    "configs/llama1b_aug.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}  鉁?$file${NC}"
    else
        echo -e "${RED}  鉁?$file 缂哄け${NC}"
    fi
done
echo ""

# =============================================================================
# 10. 璺緞闂妫€娴?# =============================================================================
echo -e "${BLUE}[10/10] 妫€鏌ヨ矾寰勯棶棰?{NC}"
CURRENT_DIR=$(pwd)
if [[ "$CURRENT_DIR" == *" "* ]]; then
    echo -e "${RED}鉁?涓ラ噸闂: 褰撳墠璺緞鍖呭惈绌烘牸锛?{NC}"
    echo "  褰撳墠璺緞: '$CURRENT_DIR'"
    echo ""
    echo -e "${YELLOW}瑙ｅ喅鏂规:${NC}"
    echo "  1. 閲嶅懡鍚嶇洰褰曪紝鍘绘帀绌烘牸:"
    echo "     cd /home/rpwang"
    echo "     mv 'kava review' kava_review"
    echo "     cd kava_review"
    echo ""
    echo "  2. 鎴栬€呮€绘槸浣跨敤寮曞彿:"
    echo "     cd \"/home/rpwang/kava review\""
    echo "     source setup_hpc_env.sh"
    echo ""
else
    echo -e "${GREEN}鉁?璺緞涓嶅寘鍚┖鏍?{NC}"
fi
echo ""

# =============================================================================
# 鎬荤粨鍜屽缓璁?# =============================================================================
echo -e "${CYAN}"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  璇婃柇瀹屾垚"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo -e "${NC}"

echo -e "${YELLOW}甯歌闂鍜岃В鍐虫柟妗?${NC}"
echo ""
echo "1. 濡傛灉璺緞鍖呭惈绌烘牸:"
echo "   mv '/home/rpwang/kava review' /home/rpwang/kava_review"
echo ""
echo "2. 濡傛灉 conda 涓嶅彲鐢紝浣跨敤 Python venv:"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "3. 濡傛灉 module 绯荤粺涓嶅彲鐢?"
echo "   # 鏌ユ壘 conda"
echo "   find /opt -name conda 2>/dev/null"
echo "   # 鎴栫洿鎺ヤ娇鐢ㄧ郴缁?Python"
echo ""
echo "4. 鏌ョ湅璇︾粏鏃ュ織:"
echo "   cat $LOG_FILE"
echo ""

echo -e "${GREEN}璇婃柇鎶ュ憡宸蹭繚瀛樺埌: $LOG_FILE${NC}"
