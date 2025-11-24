#!/bin/bash
################################################################################
# 检查 HPC 共享模型库
################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════════"
echo "  HPC 共享模型库检查"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""

HPC_MODELS="/home/share/models"

# 检查目录是否存在
if [ ! -d "$HPC_MODELS" ]; then
    echo -e "${RED}✗ HPC 共享模型库不存在: $HPC_MODELS${NC}"
    echo ""
    echo "可能的原因："
    echo "  1. 路径不正确"
    echo "  2. 没有访问权限"
    echo "  3. 模型库未挂载"
    echo ""
    echo "请联系 HPC 管理员确认共享模型库路径"
    exit 1
fi

echo -e "${GREEN}✓ HPC 共享模型库存在: $HPC_MODELS${NC}"
echo ""

# 检查权限
if [ -r "$HPC_MODELS" ]; then
    echo -e "${GREEN}✓ 有读取权限${NC}"
else
    echo -e "${RED}✗ 无读取权限${NC}"
    exit 1
fi
echo ""

# 检查目录大小
MODELS_SIZE=$(du -sh "$HPC_MODELS" 2>/dev/null | cut -f1)
echo "共享模型库大小: $MODELS_SIZE"
echo ""

# 检查所需模型
echo "检查所需模型:"
echo ""

REQUIRED_MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
)

ALL_FOUND=true

for model in "${REQUIRED_MODELS[@]}"; do
    # 转换路径格式
    model_path="$HPC_MODELS/models--${model//\//'--'}"
    
    echo -n "  $model: "
    if [ -d "$model_path" ]; then
        echo -e "${GREEN}✓ 存在${NC}"
        # 检查是否有 snapshots
        if [ -d "$model_path/snapshots" ]; then
            snapshot_count=$(ls -1 "$model_path/snapshots" 2>/dev/null | wc -l)
            echo "    快照数: $snapshot_count"
        fi
    else
        echo -e "${RED}✗ 不存在${NC}"
        echo "    期望路径: $model_path"
        ALL_FOUND=false
    fi
done

echo ""

# 测试模型加载
if [ "$ALL_FOUND" = true ]; then
    echo "测试模型加载（使用第一个模型）:"
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
    print(f"尝试加载: {model_name}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"离线模式: {os.environ.get('HUGGINGFACE_HUB_OFFLINE')}")
    print("")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True
    )
    
    print(f"✓ 成功加载 tokenizer")
    print(f"  词汇表大小: {len(tokenizer)}")
    print(f"  模型最大长度: {tokenizer.model_max_length}")
    
except Exception as e:
    print(f"✗ 加载失败: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ 模型加载测试成功${NC}"
    else
        echo ""
        echo -e "${RED}✗ 模型加载测试失败${NC}"
    fi
else
    echo -e "${YELLOW}⚠ 部分模型缺失，跳过加载测试${NC}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  检查完成"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ "$ALL_FOUND" = true ]; then
    echo -e "${GREEN}✓ 所有模型可用！${NC}"
    echo ""
    echo "可以直接运行训练任务:"
    echo "  sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm"
    echo ""
    echo "环境变量配置:"
    echo "  export HF_HOME=/home/share/models"
    echo "  export TRANSFORMERS_CACHE=/home/share/models"
    echo "  export HUGGINGFACE_HUB_OFFLINE=1"
else
    echo -e "${YELLOW}⚠ 部分模型缺失${NC}"
    echo ""
    echo "请联系 HPC 管理员或手动下载缺失的模型"
fi
echo ""
