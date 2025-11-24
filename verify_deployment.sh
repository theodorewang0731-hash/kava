#!/bin/bash

################################################################################
# KAVA 閮ㄧ讲楠岃瘉鑴氭湰
################################################################################
# 鍦?HPC 涓婁紶浠ｇ爜鍚庤繍琛屾鑴氭湰锛岄獙璇佹墍鏈夋枃浠跺拰閰嶇疆鏄惁姝ｇ‘
#
# 鐢ㄦ硶: bash verify_deployment.sh
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

log_check() {
    echo -e "${BLUE}[妫€鏌${NC} $1"
}

log_ok() {
    echo -e "${GREEN}  鉁?{NC} $1"
}

log_error() {
    echo -e "${RED}  鉁?{NC} $1"
    ((ERRORS++))
}

log_warning() {
    echo -e "${YELLOW}  鈿?{NC} $1"
    ((WARNINGS++))
}

echo ""
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  KAVA 閮ㄧ讲楠岃瘉"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo ""

# 1. 妫€鏌ュ繀瑕佹枃浠?log_check "妫€鏌ユ牳蹇冩枃浠?.."
REQUIRED_FILES=(
    "train.py"
    "evaluate.py"
    "inference.py"
    "run_multi_seed.py"
    "aggregate_multi_seed.py"
    "requirements.txt"
    "README.md"
    "run_everything.sh"
    "setup_hpc.sh"
    "hpc_run_all.sh"
    "submit_multi_seed.slurm"
    "REPRODUCTION_CHECKLIST.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_ok "$file"
    else
        log_error "缂哄皯鏂囦欢: $file"
    fi
done
echo ""

# 2. 妫€鏌ラ厤缃枃浠?log_check "妫€鏌ラ厤缃枃浠?.."
CONFIGS=(
    "configs/llama1b_aug.yaml"
    "configs/llama1b_aug_nl.yaml"
    "configs/qwen05b_aug.yaml"
    "configs/llama3b_aug.yaml"
)

for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        log_ok "$config"
    else
        log_error "缂哄皯閰嶇疆: $config"
    fi
done
echo ""

# 3. 妫€鏌?src 鐩綍
log_check "妫€鏌ユ簮浠ｇ爜妯″潡..."
SRC_FILES=(
    "src/__init__.py"
    "src/rkv_compression.py"
    "src/losses.py"
    "src/latent_reasoning.py"
    "src/data_utils.py"
    "src/trainer.py"
    "src/utils.py"
    "src/evaluation_datasets.py"
)

for src in "${SRC_FILES[@]}"; do
    if [ -f "$src" ]; then
        log_ok "$src"
    else
        log_error "缂哄皯婧愭枃浠? $src"
    fi
done
echo ""

# 4. 妫€鏌ヨ剼鏈潈闄?log_check "妫€鏌ヨ剼鏈墽琛屾潈闄?.."
SCRIPTS=(
    "run_reproduce.sh"
    "setup_hpc.sh"
    "hpc_run_all.sh"
    "verify_deployment.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            log_ok "$script 鍙墽琛?
        else
            log_warning "$script 涓嶅彲鎵ц锛屾鍦ㄤ慨澶?.."
            chmod +x "$script"
            log_ok "宸茶缃?$script 涓哄彲鎵ц"
        fi
    fi
done
echo ""

# 5. 妫€鏌ユ崲琛岀鏍煎紡锛圕RLF vs LF锛?log_check "妫€鏌ユ枃浠舵崲琛岀鏍煎紡..."
if command -v file &> /dev/null; then
    for script in "${SCRIPTS[@]}"; do
        if [ -f "$script" ]; then
            file_type=$(file "$script")
            if echo "$file_type" | grep -q "CRLF"; then
                log_warning "$script 浣跨敤 Windows 鎹㈣绗?(CRLF)"
                if command -v dos2unix &> /dev/null; then
                    dos2unix "$script" 2>/dev/null
                    log_ok "宸茶浆鎹?$script 涓?Unix 鏍煎紡 (LF)"
                else
                    log_warning "寤鸿杩愯: dos2unix $script"
                fi
            else
                log_ok "$script (Unix 鏍煎紡)"
            fi
        fi
    done
else
    log_warning "鏈畨瑁?'file' 鍛戒护锛岃烦杩囨崲琛岀妫€鏌?
fi
echo ""

# 6. 妫€鏌?SLURM 鐜
log_check "妫€鏌?SLURM 鐜..."
if command -v sbatch &> /dev/null; then
    log_ok "sbatch 鍛戒护鍙敤"
    
    if command -v squeue &> /dev/null; then
        log_ok "squeue 鍛戒护鍙敤"
    else
        log_error "squeue 鍛戒护涓嶅彲鐢?
    fi
    
    if command -v sinfo &> /dev/null; then
        log_ok "sinfo 鍛戒护鍙敤"
        
        # 妫€鏌?compute 鍒嗗尯
        if sinfo -p compute &> /dev/null 2>&1; then
            log_ok "compute 鍒嗗尯鍙闂?
        else
            log_warning "compute 鍒嗗尯涓嶅彲璁块棶"
        fi
    else
        log_error "sinfo 鍛戒护涓嶅彲鐢?
    fi
else
    log_error "sbatch 鍛戒护涓嶅彲鐢?- 璇风‘淇濆湪 HPC 鐧诲綍鑺傜偣杩愯"
fi
echo ""

# 7. 妫€鏌?Python 鐜
log_check "妫€鏌?Python 鐜..."
if command -v python &> /dev/null || command -v python3 &> /dev/null; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_ok "Python $PYTHON_VERSION"
    
    # 妫€鏌?conda
    if command -v conda &> /dev/null; then
        log_ok "conda 鍙敤"
    else
        log_warning "conda 涓嶅彲鐢?- 鍙兘闇€瑕佸姞杞芥ā鍧? module load anaconda3"
    fi
else
    log_error "Python 涓嶅彲鐢?
fi
echo ""

# 8. 妫€鏌ョ鐩樼┖闂?log_check "妫€鏌ョ鐩樼┖闂?.."
CACHE_DIR="$HOME/.cache/huggingface"
mkdir -p "$CACHE_DIR" 2>/dev/null || true

AVAILABLE_KB=$(df -k "$CACHE_DIR" | awk 'NR==2 {print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))

if [ $AVAILABLE_GB -ge 20 ]; then
    log_ok "纾佺洏绌洪棿鍏呰冻: ${AVAILABLE_GB}GB 鍙敤"
elif [ $AVAILABLE_GB -ge 15 ]; then
    log_warning "纾佺洏绌洪棿绱у紶: ${AVAILABLE_GB}GB 鍙敤 (寤鸿 鈮?0GB)"
else
    log_error "纾佺洏绌洪棿涓嶈冻: ${AVAILABLE_GB}GB 鍙敤 (闇€瑕?鈮?0GB)"
fi
echo ""

# 9. 妫€鏌ョ綉缁滆繛鎺?log_check "妫€鏌ョ綉缁滆繛鎺?.."
if ping -c 1 -W 2 huggingface.co &> /dev/null; then
    log_ok "鍙闂?huggingface.co"
elif ping -c 1 -W 2 hf-mirror.com &> /dev/null; then
    log_ok "鍙闂?hf-mirror.com (闀滃儚绔?"
    log_warning "寤鸿浣跨敤: bash run_reproduce.sh --method mirror"
elif ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
    log_warning "缃戠粶杩炴帴鍙楅檺锛屼絾鍙闂簰鑱旂綉"
else
    log_error "缃戠粶杩炴帴寮傚父"
fi
echo ""

# 10. 妫€鏌ョ幆澧冨彉閲?log_check "妫€鏌ョ幆澧冨彉閲?.."
if [ -n "$HF_HOME" ]; then
    log_ok "HF_HOME = $HF_HOME"
else
    log_warning "HF_HOME 鏈缃紝寤鸿杩愯: bash setup_hpc.sh"
fi

if [ -n "$TRANSFORMERS_CACHE" ]; then
    log_ok "TRANSFORMERS_CACHE = $TRANSFORMERS_CACHE"
else
    log_warning "TRANSFORMERS_CACHE 鏈缃?
fi
echo ""

# 鎬荤粨
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}鉁?楠岃瘉閫氳繃锛佹墍鏈夋鏌ラ兘鎴愬姛銆?{NC}"
    echo ""
    echo "涓嬩竴姝ワ細"
    echo "  1. 杩愯蹇€熻缃? bash setup_hpc.sh"
    echo "  2. 涓€閿惎鍔? bash run_reproduce.sh"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}鈿?楠岃瘉瀹屾垚锛屾湁 $WARNINGS 涓鍛?{NC}"
    echo ""
    echo "寤鸿鍏堣В鍐宠鍛婏紝鐒跺悗锛?
    echo "  1. 杩愯: bash setup_hpc.sh"
    echo "  2. 鍚姩: bash run_reproduce.sh"
else
    echo -e "${RED}鉂?楠岃瘉澶辫触锛?ERRORS 涓敊璇紝$WARNINGS 涓鍛?{NC}"
    echo ""
    echo "璇蜂慨澶嶄笂杩伴敊璇悗閲嶆柊杩愯楠岃瘉"
    exit 1
fi
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo ""
