#!/bin/bash
#==============================================================================
# HPC 公共模型库配置脚本
# 用途：一键配置 HuggingFace 使用 HPC 共享模型库
# 作者：KAVA Project
# 日期：2025-01-17
#==============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 公共模型库路径
MODELS_PATH="/home/share/models"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HPC 公共模型库配置脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

#==============================================================================
# 1. 检查公共模型库是否存在
#==============================================================================
echo -e "${YELLOW}[1/5] 检查公共模型库...${NC}"
if [ -d "$MODELS_PATH" ]; then
    echo -e "${GREEN}✓ 公共模型库存在: $MODELS_PATH${NC}"
    
    # 列出可用模型
    echo ""
    echo "可用模型："
    ls -1d $MODELS_PATH/models--* 2>/dev/null | head -10 | while read model; do
        model_name=$(basename $model | sed 's/models--//; s/--/\//g')
        echo "  - $model_name"
    done
    
    model_count=$(ls -1d $MODELS_PATH/models--* 2>/dev/null | wc -l)
    if [ $model_count -gt 10 ]; then
        echo "  ... 共 $model_count 个模型"
    fi
else
    echo -e "${RED}✗ 公共模型库不存在: $MODELS_PATH${NC}"
    echo "请联系管理员或使用个人缓存目录"
    exit 1
fi

echo ""

#==============================================================================
# 2. 配置环境变量到 ~/.bashrc
#==============================================================================
echo -e "${YELLOW}[2/5] 配置环境变量到 ~/.bashrc...${NC}"

# 检查是否已配置
if grep -q "HF_HOME=/home/share/models" ~/.bashrc 2>/dev/null; then
    echo -e "${GREEN}✓ 环境变量已配置在 ~/.bashrc${NC}"
else
    echo ""
    echo "添加以下内容到 ~/.bashrc："
    echo "  export HF_HOME=$MODELS_PATH"
    echo "  export TRANSFORMERS_CACHE=$MODELS_PATH"
    echo "  export HF_DATASETS_CACHE=$MODELS_PATH"
    echo ""
    
    # 备份 ~/.bashrc
    if [ -f ~/.bashrc ]; then
        cp ~/.bashrc ~/.bashrc.bak.$(date +%Y%m%d_%H%M%S)
        echo -e "${GREEN}✓ 已备份 ~/.bashrc${NC}"
    fi
    
    # 添加配置
    cat >> ~/.bashrc << 'EOF'

# HuggingFace 公共模型库 (添加于 KAVA 配置脚本)
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
EOF
    
    echo -e "${GREEN}✓ 已添加环境变量到 ~/.bashrc${NC}"
fi

echo ""

#==============================================================================
# 3. 配置到当前 Conda 环境（如果已激活）
#==============================================================================
echo -e "${YELLOW}[3/5] 配置到 Conda 环境...${NC}"

if [ -n "$CONDA_PREFIX" ]; then
    echo "检测到 Conda 环境: $CONDA_PREFIX"
    
    # 创建激活脚本目录
    ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
    mkdir -p $ACTIVATE_DIR
    
    # 创建激活脚本
    cat > $ACTIVATE_DIR/hf_models.sh << 'EOF'
#!/bin/bash
# HuggingFace 公共模型库配置
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
EOF
    
    chmod +x $ACTIVATE_DIR/hf_models.sh
    echo -e "${GREEN}✓ 已配置到 Conda 环境激活脚本${NC}"
    echo "  路径: $ACTIVATE_DIR/hf_models.sh"
else
    echo -e "${YELLOW}⚠ 未检测到激活的 Conda 环境${NC}"
    echo "  提示: 运行 'conda activate kava' 后重新执行此脚本"
fi

echo ""

#==============================================================================
# 4. 立即应用环境变量
#==============================================================================
echo -e "${YELLOW}[4/5] 应用环境变量到当前会话...${NC}"

export HF_HOME=$MODELS_PATH
export TRANSFORMERS_CACHE=$MODELS_PATH
export HF_DATASETS_CACHE=$MODELS_PATH

echo -e "${GREEN}✓ 环境变量已应用到当前会话${NC}"
echo "  HF_HOME=$HF_HOME"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

echo ""

#==============================================================================
# 5. 验证配置
#==============================================================================
echo -e "${YELLOW}[5/5] 验证配置...${NC}"

# 检查 KAVA 项目所需模型
REQUIRED_MODELS=(
    "models--meta-llama--Llama-3.2-1B-Instruct"
    "models--meta-llama--Llama-3.2-3B-Instruct"
    "models--Qwen--Qwen2.5-0.5B-Instruct"
)

echo ""
echo "检查 KAVA 项目所需模型："
for model in "${REQUIRED_MODELS[@]}"; do
    model_display=$(echo $model | sed 's/models--//; s/--/\//g')
    if [ -d "$MODELS_PATH/$model" ]; then
        echo -e "  ${GREEN}✓${NC} $model_display"
    else
        echo -e "  ${YELLOW}✗${NC} $model_display (未找到)"
    fi
done

echo ""

# Python 验证
if command -v python &> /dev/null; then
    echo "测试 Python 加载..."
    python << 'EOF'
import os
import sys

models_path = "/home/share/models"
hf_home = os.environ.get("HF_HOME")

print(f"  HF_HOME 环境变量: {hf_home}")
print(f"  路径存在: {os.path.exists(models_path)}")

# 测试 transformers
try:
    from transformers import AutoTokenizer
    print("  ✓ transformers 已安装")
    
    # 尝试加载 tokenizer（不下载）
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            cache_dir=models_path,
            local_files_only=True
        )
        print("  ✓ 成功从共享缓存加载模型")
    except Exception as e:
        print(f"  ⚠ 加载测试失败: {str(e)[:60]}...")
        print("    (这可能是正常的，如果模型尚未下载)")
        
except ImportError:
    print("  ⚠ transformers 未安装")
    print("    请运行: pip install transformers")
EOF

else
    echo -e "${YELLOW}⚠ Python 不可用，跳过 Python 验证${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配置完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "后续步骤："
echo "  1. 重新加载 shell 配置："
echo "     source ~/.bashrc"
echo ""
echo "  2. 或重新激活 Conda 环境："
echo "     conda deactivate && conda activate kava"
echo ""
echo "  3. 验证配置："
echo "     echo \$HF_HOME"
echo "     python -c 'import os; print(os.environ.get(\"HF_HOME\"))'"
echo ""
echo "  4. 开始训练："
echo "     python train.py --config configs/llama1b_aug.yaml"
echo ""
echo -e "${YELLOW}注意：下次登录会自动应用这些环境变量${NC}"
echo ""
