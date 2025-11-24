#!/bin/bash
################################################################################
# 娴嬭瘯鐜閰嶇疆
################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "娴嬭瘯 KAVA 鐜閰嶇疆..."
echo ""

# 娴嬭瘯 Python
echo -n "Python 鐗堟湰: "
python --version || python3 --version

# 娴嬭瘯鍏抽敭鍖?echo ""
echo "娴嬭瘯鍏抽敭 Python 鍖?"

packages=("torch" "transformers" "datasets" "numpy" "pandas")

for pkg in "${packages[@]}"; do
    echo -n "  - $pkg: "
    if python -c "import $pkg; print($pkg.__version__)" 2>/dev/null; then
        echo -e "${GREEN}鉁?{NC}"
    else
        echo -e "${RED}鉁?鏈畨瑁?{NC}"
    fi
done

# 娴嬭瘯 CUDA锛堝鏋滄湁锛?echo ""
echo -n "CUDA 鍙敤鎬? "
if python -c "import torch; print('鉁?CUDA', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '鉁?CPU only')" 2>/dev/null; then
    :
else
    echo -e "${YELLOW}鏃犳硶妫€娴?{NC}"
fi

# 妫€鏌ュ繀瑕佹枃浠?echo ""
echo "妫€鏌ュ繀瑕佹枃浠?"
files=("train.py" "requirements.txt" "configs/llama1b_aug.yaml")
for f in "${files[@]}"; do
    echo -n "  - $f: "
    if [ -f "$f" ]; then
        echo -e "${GREEN}鉁?{NC}"
    else
        echo -e "${RED}鉁?缂哄け${NC}"
    fi
done

echo ""
echo "鐜娴嬭瘯瀹屾垚锛?
