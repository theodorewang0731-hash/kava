#!/bin/bash
################################################################################
# 妫€鏌?HPC 鍏变韩妯″瀷搴?################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  HPC 鍏变韩妯″瀷搴撴鏌?
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo -e "${NC}"
echo ""

HPC_MODELS="/home/share/models"

# 妫€鏌ョ洰褰曟槸鍚﹀瓨鍦?if [ ! -d "$HPC_MODELS" ]; then
    echo -e "${RED}鉁?HPC 鍏变韩妯″瀷搴撲笉瀛樺湪: $HPC_MODELS${NC}"
    echo ""
    echo "鍙兘鐨勫師鍥狅細"
    echo "  1. 璺緞涓嶆纭?
    echo "  2. 娌℃湁璁块棶鏉冮檺"
    echo "  3. 妯″瀷搴撴湭鎸傝浇"
    echo ""
    echo "璇疯仈绯?HPC 绠＄悊鍛樼‘璁ゅ叡浜ā鍨嬪簱璺緞"
    exit 1
fi

echo -e "${GREEN}鉁?HPC 鍏变韩妯″瀷搴撳瓨鍦? $HPC_MODELS${NC}"
echo ""

# 妫€鏌ユ潈闄?if [ -r "$HPC_MODELS" ]; then
    echo -e "${GREEN}鉁?鏈夎鍙栨潈闄?{NC}"
else
    echo -e "${RED}鉁?鏃犺鍙栨潈闄?{NC}"
    exit 1
fi
echo ""

# 妫€鏌ョ洰褰曞ぇ灏?MODELS_SIZE=$(du -sh "$HPC_MODELS" 2>/dev/null | cut -f1)
echo "鍏变韩妯″瀷搴撳ぇ灏? $MODELS_SIZE"
echo ""

# 妫€鏌ユ墍闇€妯″瀷
echo "妫€鏌ユ墍闇€妯″瀷:"
echo ""

REQUIRED_MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
)

ALL_FOUND=true

for model in "${REQUIRED_MODELS[@]}"; do
    # 杞崲璺緞鏍煎紡
    model_path="$HPC_MODELS/models--${model//\//'--'}"
    
    echo -n "  $model: "
    if [ -d "$model_path" ]; then
        echo -e "${GREEN}鉁?瀛樺湪${NC}"
        # 妫€鏌ユ槸鍚︽湁 snapshots
        if [ -d "$model_path/snapshots" ]; then
            snapshot_count=$(ls -1 "$model_path/snapshots" 2>/dev/null | wc -l)
            echo "    蹇収鏁? $snapshot_count"
        fi
    else
        echo -e "${RED}鉁?涓嶅瓨鍦?{NC}"
        echo "    鏈熸湜璺緞: $model_path"
        ALL_FOUND=false
    fi
done

echo ""

# 娴嬭瘯妯″瀷鍔犺浇
if [ "$ALL_FOUND" = true ]; then
    echo "娴嬭瘯妯″瀷鍔犺浇锛堜娇鐢ㄧ涓€涓ā鍨嬶級:"
    echo ""
    
    export HF_HOME="$HPC_MODELS"
    export TRANSFORMERS_CACHE="$HPC_MODELS"
    export HUGGINGFACE_HUB_OFFLINE=1
    
    python3 << 'EOF'
import os
import sys

try:
    from transformers import AutoTokenizer
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"灏濊瘯鍔犺浇: {model_name}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"绂荤嚎妯″紡: {os.environ.get('HUGGINGFACE_HUB_OFFLINE')}")
    print("")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True
    )
    
    print(f"鉁?鎴愬姛鍔犺浇 tokenizer")
    print(f"  璇嶆眹琛ㄥぇ灏? {len(tokenizer)}")
    print(f"  妯″瀷鏈€澶ч暱搴? {tokenizer.model_max_length}")
    
except Exception as e:
    print(f"鉁?鍔犺浇澶辫触: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}鉁?妯″瀷鍔犺浇娴嬭瘯鎴愬姛${NC}"
    else
        echo ""
        echo -e "${RED}鉁?妯″瀷鍔犺浇娴嬭瘯澶辫触${NC}"
    fi
else
    echo -e "${YELLOW}鈿?閮ㄥ垎妯″瀷缂哄け锛岃烦杩囧姞杞芥祴璇?{NC}"
fi

echo ""
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  妫€鏌ュ畬鎴?
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo ""

if [ "$ALL_FOUND" = true ]; then
    echo -e "${GREEN}鉁?鎵€鏈夋ā鍨嬪彲鐢紒${NC}"
    echo ""
    echo "鍙互鐩存帴杩愯璁粌浠诲姟:"
    echo "  sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm"
    echo ""
    echo "鐜鍙橀噺閰嶇疆:"
    echo "  export HF_HOME=/home/share/models"
    echo "  export TRANSFORMERS_CACHE=/home/share/models"
    echo "  export HUGGINGFACE_HUB_OFFLINE=1"
else
    echo -e "${YELLOW}鈿?閮ㄥ垎妯″瀷缂哄け${NC}"
    echo ""
    echo "璇疯仈绯?HPC 绠＄悊鍛樻垨鎵嬪姩涓嬭浇缂哄け鐨勬ā鍨?
fi
echo ""
