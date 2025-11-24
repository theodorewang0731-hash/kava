#!/bin/bash

################################################################################
# 妫€鏌?HPC 鍏变韩妯″瀷搴撲腑鐨勫彲鐢ㄦā鍨?################################################################################

echo "=== 妫€鏌?HPC 鍏变韩妯″瀷搴?==="
echo ""

SHARE_DIR="/home/share/models"

if [ ! -d "$SHARE_DIR" ]; then
    echo "閿欒: $SHARE_DIR 涓嶅瓨鍦?
    exit 1
fi

echo "鍏变韩妯″瀷鐩綍: $SHARE_DIR"
echo ""

echo "=== Llama 绯诲垪妯″瀷 ==="
ls -lh "$SHARE_DIR" | grep -i llama | awk '{print $9, $5}'
echo ""

echo "=== Qwen 绯诲垪妯″瀷 ==="
ls -lh "$SHARE_DIR" | grep -i qwen | awk '{print $9, $5}'
echo ""

echo "=== 妫€鏌?KAVA 鎵€闇€鐨勬ā鍨?==="
echo ""

# 闇€瑕佺殑妯″瀷
declare -A REQUIRED_MODELS=(
    ["Llama-3.2-1B-Instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["Llama-3.2-3B-Instruct"]="meta-llama/Llama-3.2-3B-Instruct"
    ["Qwen2.5-0.5B-Instruct"]="Qwen/Qwen2.5-0.5B-Instruct"
)

for model_name in "${!REQUIRED_MODELS[@]}"; do
    full_name="${REQUIRED_MODELS[$model_name]}"
    
    # 妫€鏌ュ绉嶅彲鑳界殑璺緞
    found=false
    
    # 妫€鏌?1: 瀹屾暣璺緞
    if [ -d "$SHARE_DIR/$full_name" ]; then
        echo "鉁?$model_name"
        echo "  璺緞: $SHARE_DIR/$full_name"
        found=true
    # 妫€鏌?2: 鍙湁妯″瀷鍚?    elif [ -d "$SHARE_DIR/$model_name" ]; then
        echo "鉁?$model_name"
        echo "  璺緞: $SHARE_DIR/$model_name"
        found=true
    # 妫€鏌?3: 杞崲涓哄皬鍐?涓嬪垝绾挎牸寮?    else
        # 灏濊瘯鍏朵粬鍙兘鐨勫懡鍚嶆牸寮?        for dir in "$SHARE_DIR"/*; do
            if [[ "$(basename "$dir" | tr '[:upper:]' '[:lower:]')" == *"$(echo "$model_name" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"* ]]; then
                echo "鉁?$model_name (鍙兘鍖归厤)"
                echo "  璺緞: $dir"
                found=true
                break
            fi
        done
    fi
    
    if [ "$found" = false ]; then
        echo "鉁?$model_name - 鏈壘鍒?
        echo "  闇€瑕? $full_name"
    fi
    echo ""
done

echo "=== 鍙兘鐨勬浛浠ｆā鍨?==="
echo ""

# Llama-3.2-1B 鐨勬浛浠?echo "鏇夸唬 Llama-3.2-1B-Instruct (1B 鍙傛暟):"
ls -lh "$SHARE_DIR" | grep -i llama | grep -E "1b|7b" | head -3 | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

# Llama-3.2-3B 鐨勬浛浠?echo "鏇夸唬 Llama-3.2-3B-Instruct (3B 鍙傛暟):"
ls -lh "$SHARE_DIR" | grep -i llama | grep -E "3b|7b|13b" | head -3 | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

# Qwen2.5-0.5B 鐨勬浛浠?echo "鏇夸唬 Qwen2.5-0.5B-Instruct (0.5B 鍙傛暟):"
ls -lh "$SHARE_DIR" | grep -i qwen | head -3 | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

echo "=== 鎵€鏈夊彲鐢ㄦā鍨嬪垪琛?==="
ls -lh "$SHARE_DIR" | grep "^d" | awk '{print $9, "(" $5 ")"}'
