#!/bin/bash

################################################################################
# 检查 HPC 共享模型库中的可用模型
################################################################################

echo "=== 检查 HPC 共享模型库 ==="
echo ""

SHARE_DIR="/home/share/models"

if [ ! -d "$SHARE_DIR" ]; then
    echo "错误: $SHARE_DIR 不存在"
    exit 1
fi

echo "共享模型目录: $SHARE_DIR"
echo ""

echo "=== Llama 系列模型 ==="
ls -lh "$SHARE_DIR" | grep -i llama | awk '{print $9, $5}'
echo ""

echo "=== Qwen 系列模型 ==="
ls -lh "$SHARE_DIR" | grep -i qwen | awk '{print $9, $5}'
echo ""

echo "=== 检查 KAVA 所需的模型 ==="
echo ""

# 需要的模型
declare -A REQUIRED_MODELS=(
    ["Llama-3.2-1B-Instruct"]="meta-llama/Llama-3.2-1B-Instruct"
    ["Llama-3.2-3B-Instruct"]="meta-llama/Llama-3.2-3B-Instruct"
    ["Qwen2.5-0.5B-Instruct"]="Qwen/Qwen2.5-0.5B-Instruct"
)

for model_name in "${!REQUIRED_MODELS[@]}"; do
    full_name="${REQUIRED_MODELS[$model_name]}"
    
    # 检查多种可能的路径
    found=false
    
    # 检查 1: 完整路径
    if [ -d "$SHARE_DIR/$full_name" ]; then
        echo "✓ $model_name"
        echo "  路径: $SHARE_DIR/$full_name"
        found=true
    # 检查 2: 只有模型名
    elif [ -d "$SHARE_DIR/$model_name" ]; then
        echo "✓ $model_name"
        echo "  路径: $SHARE_DIR/$model_name"
        found=true
    # 检查 3: 转换为小写/下划线格式
    else
        # 尝试其他可能的命名格式
        for dir in "$SHARE_DIR"/*; do
            if [[ "$(basename "$dir" | tr '[:upper:]' '[:lower:]')" == *"$(echo "$model_name" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"* ]]; then
                echo "✓ $model_name (可能匹配)"
                echo "  路径: $dir"
                found=true
                break
            fi
        done
    fi
    
    if [ "$found" = false ]; then
        echo "✗ $model_name - 未找到"
        echo "  需要: $full_name"
    fi
    echo ""
done

echo "=== 可能的替代模型 ==="
echo ""

# Llama-3.2-1B 的替代
echo "替代 Llama-3.2-1B-Instruct (1B 参数):"
ls -lh "$SHARE_DIR" | grep -i llama | grep -E "1b|7b" | head -3 | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

# Llama-3.2-3B 的替代
echo "替代 Llama-3.2-3B-Instruct (3B 参数):"
ls -lh "$SHARE_DIR" | grep -i llama | grep -E "3b|7b|13b" | head -3 | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

# Qwen2.5-0.5B 的替代
echo "替代 Qwen2.5-0.5B-Instruct (0.5B 参数):"
ls -lh "$SHARE_DIR" | grep -i qwen | head -3 | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

echo "=== 所有可用模型列表 ==="
ls -lh "$SHARE_DIR" | grep "^d" | awk '{print $9, "(" $5 ")"}'
