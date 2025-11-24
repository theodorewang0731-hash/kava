#!/bin/bash
################################################################################
# HPC 璧勬簮瀹夊叏妫€鏌ヨ剼鏈?################################################################################
# 鐢ㄩ€旓細鍦ㄨ繍琛岃缁冨墠妫€鏌ョ鐩橀厤棰濆拰璧勬簮浣跨敤鎯呭喌
# 浣跨敤锛歜ash check_hpc_quota.sh
################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  HPC 璧勬簮瀹夊叏妫€鏌?
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo -e "${NC}"
echo ""

# =============================================================================
# 1. 纾佺洏绌洪棿妫€鏌?# =============================================================================
echo -e "${BLUE}[1/5] 纾佺洏绌洪棿妫€鏌?{NC}"
echo ""

# HOME 鐩綍绌洪棿
HOME_USAGE=$(df -h $HOME | awk 'NR==2 {print $5}' | sed 's/%//')
HOME_AVAIL=$(df -h $HOME | awk 'NR==2 {print $4}')

echo "HOME 鐩綍 ($HOME):"
echo "  浣跨敤鐜? ${HOME_USAGE}%"
echo "  鍙敤绌洪棿: ${HOME_AVAIL}"

if [ $HOME_USAGE -gt 90 ]; then
    echo -e "${RED}鉁?璀﹀憡: 纾佺洏浣跨敤鐜囪繃楂?(${HOME_USAGE}%)${NC}"
    echo "  寤鸿娓呯悊涓嶉渶瑕佺殑鏂囦欢"
elif [ $HOME_USAGE -gt 80 ]; then
    echo -e "${YELLOW}鈿?娉ㄦ剰: 纾佺洏浣跨敤鐜囪緝楂?(${HOME_USAGE}%)${NC}"
else
    echo -e "${GREEN}鉁?纾佺洏绌洪棿鍏呰冻${NC}"
fi
echo ""

# HuggingFace 缂撳瓨鐩綍
HF_CACHE="${HOME}/.cache/huggingface"
if [ -d "$HF_CACHE" ]; then
    HF_SIZE=$(du -sh "$HF_CACHE" 2>/dev/null | cut -f1)
    echo "HuggingFace 缂撳瓨: $HF_SIZE"
fi

# 椤圭洰杈撳嚭鐩綍
if [ -d "outputs" ]; then
    OUTPUT_SIZE=$(du -sh outputs 2>/dev/null | cut -f1)
    echo "椤圭洰杈撳嚭鐩綍: $OUTPUT_SIZE"
fi
echo ""

# =============================================================================
# 2. 閰嶉妫€鏌ワ紙濡傛灉 HPC 鏈夐厤棰濈郴缁燂級
# =============================================================================
echo -e "${BLUE}[2/5] 閰嶉妫€鏌?{NC}"
echo ""

if command -v quota &> /dev/null; then
    quota -s
    echo -e "${GREEN}鉁?閰嶉淇℃伅瑙佷笂${NC}"
else
    echo "姝?HPC 绯荤粺鏈厤缃厤棰濇鏌?
fi
echo ""

# =============================================================================
# 3. SLURM 浠诲姟妫€鏌?# =============================================================================
echo -e "${BLUE}[3/5] SLURM 浠诲姟鐘舵€?{NC}"
echo ""

if command -v squeue &> /dev/null; then
    RUNNING=$(squeue -u $USER -t RUNNING | wc -l)
    PENDING=$(squeue -u $USER -t PENDING | wc -l)
    
    # 鍑?鏄洜涓鸿〃澶磋
    RUNNING=$((RUNNING - 1))
    PENDING=$((PENDING - 1))
    
    echo "褰撳墠浠诲姟:"
    echo "  杩愯涓? $RUNNING"
    echo "  绛夊緟涓? $PENDING"
    echo "  鎬昏: $((RUNNING + PENDING))"
    
    if [ $RUNNING -gt 0 ] || [ $PENDING -gt 0 ]; then
        echo ""
        echo "浠诲姟璇︽儏:"
        squeue -u $USER --format="%.10i %.15j %.8T %.10M %.6D %.20R"
    fi
    
    # 妫€鏌ヤ换鍔℃暟閲忔槸鍚﹁繃澶?    TOTAL=$((RUNNING + PENDING))
    if [ $TOTAL -gt 20 ]; then
        echo -e "${YELLOW}鈿?娉ㄦ剰: 浠诲姟鏁伴噺杈冨 ($TOTAL)${NC}"
        echo "  寤鸿绛夊緟閮ㄥ垎浠诲姟瀹屾垚鍚庡啀鎻愪氦鏂颁换鍔?
    elif [ $TOTAL -gt 0 ]; then
        echo -e "${GREEN}鉁?浠诲姟鏁伴噺姝ｅ父${NC}"
    else
        echo -e "${GREEN}鉁?鏃犺繍琛屼腑鐨勪换鍔?{NC}"
    fi
else
    echo -e "${RED}鉁?SLURM 鍛戒护涓嶅彲鐢?{NC}"
fi
echo ""

# =============================================================================
# 4. GPU 璧勬簮妫€鏌?# =============================================================================
echo -e "${BLUE}[4/5] GPU 鍙敤鎬?{NC}"
echo ""

if command -v sinfo &> /dev/null; then
    echo "GPU 鍒嗗尯鐘舵€?"
    sinfo -p compute --format="%.10P %.5a %.10l %.6D %.20N %.20C" 2>/dev/null || \
    sinfo --format="%.10P %.5a %.10l %.6D %.20N %.20C"
    echo ""
    echo -e "${GREEN}鉁?GPU 鍒嗗尯淇℃伅瑙佷笂${NC}"
else
    echo "鏃犳硶鏌ヨ GPU 鐘舵€?
fi
echo ""

# =============================================================================
# 5. 椤圭洰鏂囦欢妫€鏌?# =============================================================================
echo -e "${BLUE}[5/5] 椤圭洰鏂囦欢瀹屾暣鎬?{NC}"
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
        echo -e "${GREEN}鉁?{NC} $file"
    else
        echo -e "${RED}鉁?{NC} $file (缂哄け)"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = true ]; then
    echo ""
    echo -e "${GREEN}鉁?鎵€鏈夊繀闇€鏂囦欢瀛樺湪${NC}"
fi
echo ""

# =============================================================================
# 鎽樿鍜屽缓璁?# =============================================================================
echo -e "${BLUE}"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  妫€鏌ュ畬鎴?
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo -e "${NC}"
echo ""

echo "瀹夊叏鎻愰啋:"
echo "  1. 鎵€鏈夋搷浣滀粎闄愪簬鎮ㄧ殑 HOME 鐩綍: /home/rpwang"
echo "  2. 椤圭洰璺緞: /home/rpwang/kava review (鍖呭惈绌烘牸)"
echo "  3. 涓嶄細褰卞搷鍏朵粬鐢ㄦ埛鐨勬枃浠舵垨杩涚▼"
echo "  4. 璧勬簮浣跨敤鍙?SLURM 閰嶉闄愬埗"
echo "  5. 鎵€鏈夎绠椾换鍔￠€氳繃 SLURM 璋冨害"
echo ""
echo -e "${YELLOW}鈿狅笍  璺緞绌烘牸鎻愰啋:${NC}"
echo "  鈥?鍛戒护琛屾搷浣滄椂浣跨敤寮曞彿: cd \"/home/rpwang/kava review\""
echo "  鈥?寤鸿閲嶅懡鍚嶄负: mv \"kava review\" kava_review"
echo ""

if [ $HOME_USAGE -gt 80 ] || [ $TOTAL -gt 15 ]; then
    echo -e "${YELLOW}寤鸿鎿嶄綔:${NC}"
    if [ $HOME_USAGE -gt 80 ]; then
        echo "  鈥?娓呯悊涓嶉渶瑕佺殑缂撳瓨: huggingface-cli delete-cache"
        echo "  鈥?鍒犻櫎鏃х殑璁粌杈撳嚭: rm -rf outputs/old_experiment"
    fi
    if [ $TOTAL -gt 15 ]; then
        echo "  鈥?绛夊緟閮ㄥ垎浠诲姟瀹屾垚鍚庡啀鎻愪氦鏂颁换鍔?
    fi
    echo ""
fi

echo "濡傛灉涓€鍒囨甯革紝鍙互杩愯:"
echo "  bash simple_setup.sh          # 閰嶇疆鐜"
echo "  bash submit_all_jobs.sh       # 鎻愪氦璁粌浠诲姟"
echo "  bash monitor_jobs.sh          # 鐩戞帶浠诲姟鐘舵€?
echo ""
