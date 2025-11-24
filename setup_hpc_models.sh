#!/bin/bash
#==============================================================================
# HPC 鍏叡妯″瀷搴撻厤缃剼鏈?# 鐢ㄩ€旓細涓€閿厤缃?HuggingFace 浣跨敤 HPC 鍏变韩妯″瀷搴?# 浣滆€咃細KAVA Project
# 鏃ユ湡锛?025-01-17
#==============================================================================

set -e  # 閬囧埌閿欒绔嬪嵆閫€鍑?
# 棰滆壊杈撳嚭
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 鍏叡妯″瀷搴撹矾寰?MODELS_PATH="/home/share/models"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HPC 鍏叡妯″瀷搴撻厤缃剼鏈?{NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

#==============================================================================
# 1. 妫€鏌ュ叕鍏辨ā鍨嬪簱鏄惁瀛樺湪
#==============================================================================
echo -e "${YELLOW}[1/5] 妫€鏌ュ叕鍏辨ā鍨嬪簱...${NC}"
if [ -d "$MODELS_PATH" ]; then
    echo -e "${GREEN}鉁?鍏叡妯″瀷搴撳瓨鍦? $MODELS_PATH${NC}"
    
    # 鍒楀嚭鍙敤妯″瀷
    echo ""
    echo "鍙敤妯″瀷锛?
    ls -1d $MODELS_PATH/models--* 2>/dev/null | head -10 | while read model; do
        model_name=$(basename $model | sed 's/models--//; s/--/\//g')
        echo "  - $model_name"
    done
    
    model_count=$(ls -1d $MODELS_PATH/models--* 2>/dev/null | wc -l)
    if [ $model_count -gt 10 ]; then
        echo "  ... 鍏?$model_count 涓ā鍨?
    fi
else
    echo -e "${RED}鉁?鍏叡妯″瀷搴撲笉瀛樺湪: $MODELS_PATH${NC}"
    echo "璇疯仈绯荤鐞嗗憳鎴栦娇鐢ㄤ釜浜虹紦瀛樼洰褰?
    exit 1
fi

echo ""

#==============================================================================
# 2. 閰嶇疆鐜鍙橀噺鍒?~/.bashrc
#==============================================================================
echo -e "${YELLOW}[2/5] 閰嶇疆鐜鍙橀噺鍒?~/.bashrc...${NC}"

# 妫€鏌ユ槸鍚﹀凡閰嶇疆
if grep -q "HF_HOME=/home/share/models" ~/.bashrc 2>/dev/null; then
    echo -e "${GREEN}鉁?鐜鍙橀噺宸查厤缃湪 ~/.bashrc${NC}"
else
    echo ""
    echo "娣诲姞浠ヤ笅鍐呭鍒?~/.bashrc锛?
    echo "  export HF_HOME=$MODELS_PATH"
    echo "  export TRANSFORMERS_CACHE=$MODELS_PATH"
    echo "  export HF_DATASETS_CACHE=$MODELS_PATH"
    echo ""
    
    # 澶囦唤 ~/.bashrc
    if [ -f ~/.bashrc ]; then
        cp ~/.bashrc ~/.bashrc.bak.$(date +%Y%m%d_%H%M%S)
        echo -e "${GREEN}鉁?宸插浠?~/.bashrc${NC}"
    fi
    
    # 娣诲姞閰嶇疆
    cat >> ~/.bashrc << 'EOF'

# HuggingFace 鍏叡妯″瀷搴?(娣诲姞浜?KAVA 閰嶇疆鑴氭湰)
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
EOF
    
    echo -e "${GREEN}鉁?宸叉坊鍔犵幆澧冨彉閲忓埌 ~/.bashrc${NC}"
fi

echo ""

#==============================================================================
# 3. 閰嶇疆鍒板綋鍓?Conda 鐜锛堝鏋滃凡婵€娲伙級
#==============================================================================
echo -e "${YELLOW}[3/5] 閰嶇疆鍒?Conda 鐜...${NC}"

if [ -n "$CONDA_PREFIX" ]; then
    echo "妫€娴嬪埌 Conda 鐜: $CONDA_PREFIX"
    
    # 鍒涘缓婵€娲昏剼鏈洰褰?    ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
    mkdir -p $ACTIVATE_DIR
    
    # 鍒涘缓婵€娲昏剼鏈?    cat > $ACTIVATE_DIR/hf_models.sh << 'EOF'
#!/bin/bash
# HuggingFace 鍏叡妯″瀷搴撻厤缃?export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=/home/share/models
EOF
    
    chmod +x $ACTIVATE_DIR/hf_models.sh
    echo -e "${GREEN}鉁?宸查厤缃埌 Conda 鐜婵€娲昏剼鏈?{NC}"
    echo "  璺緞: $ACTIVATE_DIR/hf_models.sh"
else
    echo -e "${YELLOW}鈿?鏈娴嬪埌婵€娲荤殑 Conda 鐜${NC}"
    echo "  鎻愮ず: 杩愯 'conda activate kava' 鍚庨噸鏂版墽琛屾鑴氭湰"
fi

echo ""

#==============================================================================
# 4. 绔嬪嵆搴旂敤鐜鍙橀噺
#==============================================================================
echo -e "${YELLOW}[4/5] 搴旂敤鐜鍙橀噺鍒板綋鍓嶄細璇?..${NC}"

export HF_HOME=$MODELS_PATH
export TRANSFORMERS_CACHE=$MODELS_PATH
export HF_DATASETS_CACHE=$MODELS_PATH

echo -e "${GREEN}鉁?鐜鍙橀噺宸插簲鐢ㄥ埌褰撳墠浼氳瘽${NC}"
echo "  HF_HOME=$HF_HOME"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

echo ""

#==============================================================================
# 5. 楠岃瘉閰嶇疆
#==============================================================================
echo -e "${YELLOW}[5/5] 楠岃瘉閰嶇疆...${NC}"

# 妫€鏌?KAVA 椤圭洰鎵€闇€妯″瀷
REQUIRED_MODELS=(
    "models--meta-llama--Llama-3.2-1B-Instruct"
    "models--meta-llama--Llama-3.2-3B-Instruct"
    "models--Qwen--Qwen2.5-0.5B-Instruct"
)

echo ""
echo "妫€鏌?KAVA 椤圭洰鎵€闇€妯″瀷锛?
for model in "${REQUIRED_MODELS[@]}"; do
    model_display=$(echo $model | sed 's/models--//; s/--/\//g')
    if [ -d "$MODELS_PATH/$model" ]; then
        echo -e "  ${GREEN}鉁?{NC} $model_display"
    else
        echo -e "  ${YELLOW}鉁?{NC} $model_display (鏈壘鍒?"
    fi
done

echo ""

# Python 楠岃瘉
if command -v python &> /dev/null; then
    echo "娴嬭瘯 Python 鍔犺浇..."
    python << 'EOF'
import os
import sys

models_path = "/home/share/models"
hf_home = os.environ.get("HF_HOME")

print(f"  HF_HOME 鐜鍙橀噺: {hf_home}")
print(f"  璺緞瀛樺湪: {os.path.exists(models_path)}")

# 娴嬭瘯 transformers
try:
    from transformers import AutoTokenizer
    print("  鉁?transformers 宸插畨瑁?)
    
    # 灏濊瘯鍔犺浇 tokenizer锛堜笉涓嬭浇锛?    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            cache_dir=models_path,
            local_files_only=True
        )
        print("  鉁?鎴愬姛浠庡叡浜紦瀛樺姞杞芥ā鍨?)
    except Exception as e:
        print(f"  鈿?鍔犺浇娴嬭瘯澶辫触: {str(e)[:60]}...")
        print("    (杩欏彲鑳芥槸姝ｅ父鐨勶紝濡傛灉妯″瀷灏氭湭涓嬭浇)")
        
except ImportError:
    print("  鈿?transformers 鏈畨瑁?)
    print("    璇疯繍琛? pip install transformers")
EOF

else
    echo -e "${YELLOW}鈿?Python 涓嶅彲鐢紝璺宠繃 Python 楠岃瘉${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}閰嶇疆瀹屾垚锛?{NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "鍚庣画姝ラ锛?
echo "  1. 閲嶆柊鍔犺浇 shell 閰嶇疆锛?
echo "     source ~/.bashrc"
echo ""
echo "  2. 鎴栭噸鏂版縺娲?Conda 鐜锛?
echo "     conda deactivate && conda activate kava"
echo ""
echo "  3. 楠岃瘉閰嶇疆锛?
echo "     echo \$HF_HOME"
echo "     python -c 'import os; print(os.environ.get(\"HF_HOME\"))'"
echo ""
echo "  4. 寮€濮嬭缁冿細"
echo "     python train.py --config configs/llama1b_aug.yaml"
echo ""
echo -e "${YELLOW}娉ㄦ剰锛氫笅娆＄櫥褰曚細鑷姩搴旂敤杩欎簺鐜鍙橀噺${NC}"
echo ""
