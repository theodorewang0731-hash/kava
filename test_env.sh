#!/bin/bash
################################################################################
# 测试环境配置
################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "测试 KAVA 环境配置..."
echo ""

# 测试 Python
echo -n "Python 版本: "
python --version || python3 --version

# 测试关键包
echo ""
echo "测试关键 Python 包:"

packages=("torch" "transformers" "datasets" "numpy" "pandas")

for pkg in "${packages[@]}"; do
    echo -n "  - $pkg: "
    if python -c "import $pkg; print($pkg.__version__)" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ 未安装${NC}"
    fi
done

# 测试 CUDA（如果有）
echo ""
echo -n "CUDA 可用性: "
if python -c "import torch; print('✓ CUDA', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✗ CPU only')" 2>/dev/null; then
    :
else
    echo -e "${YELLOW}无法检测${NC}"
fi

# 检查必要文件
echo ""
echo "检查必要文件:"
files=("train.py" "requirements.txt" "configs/llama1b_aug.yaml")
for f in "${files[@]}"; do
    echo -n "  - $f: "
    if [ -f "$f" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ 缺失${NC}"
    fi
done

echo ""
echo "环境测试完成！"
