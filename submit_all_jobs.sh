#!/bin/bash

################################################################################
# KAVA 涓€閿惎鍔ㄨ剼鏈?- 浣跨敤 HPC 鍏变韩妯″瀷搴擄紙鏃犻渶涓嬭浇锛?################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 棰滆壊
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo -e "${GREEN}  KAVA 璁粌浠诲姟鎻愪氦 - 浣跨敤 HPC 鍏变韩妯″瀷搴?{NC}"
echo -e "${GREEN}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo ""

# =============================================================================
# 1. 楠岃瘉鍏变韩妯″瀷搴?# =============================================================================

echo -e "${BLUE}[1/4] 楠岃瘉 HPC 鍏变韩妯″瀷搴?{NC}"
echo "----------------------------------------"

SHARE_MODELS="/home/share/models"
REQUIRED_MODELS=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Qwen2.5-0.5B-Instruct"
)

all_models_found=true
for model in "${REQUIRED_MODELS[@]}"; do
    if [ -d "$SHARE_MODELS/$model" ]; then
        echo -e "${GREEN}鉁?{NC} $model"
    else
        echo -e "${YELLOW}鉁?{NC} $model (鏈壘鍒?"
        all_models_found=false
    fi
done

if [ "$all_models_found" = false ]; then
    echo ""
    echo -e "${YELLOW}璀﹀憡: 閮ㄥ垎妯″瀷鏈壘鍒帮紝浣嗗皢缁х画鎵ц${NC}"
    echo "濡傛灉璁粌澶辫触锛岃妫€鏌ユā鍨嬭矾寰勬槸鍚︽纭?
fi

echo ""

# =============================================================================
# 2. 楠岃瘉鐜
# =============================================================================

echo -e "${BLUE}[2/4] 楠岃瘉 Python 鐜${NC}"
echo "----------------------------------------"

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}鉁?铏氭嫙鐜涓嶅瓨鍦?{NC}"
    echo "璇峰厛杩愯: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate
echo -e "${GREEN}鉁?{NC} 铏氭嫙鐜宸叉縺娲?
echo "  Python: $(python --version)"
echo ""

# =============================================================================
# 3. 鍑嗗鐩綍
# =============================================================================

echo -e "${BLUE}[3/4] 鍑嗗杈撳嚭鐩綍${NC}"
echo "----------------------------------------"

mkdir -p outputs/logs
mkdir -p logs
echo -e "${GREEN}鉁?{NC} 杈撳嚭鐩綍宸插垱寤?
echo ""

# =============================================================================
# 4. 鎻愪氦 SLURM 浠诲姟
# =============================================================================

echo -e "${BLUE}[4/4] 鎻愪氦 SLURM 璁粌浠诲姟${NC}"
echo "----------------------------------------"

CONFIGS=(
    "llama1b_aug"
    "llama1b_aug_nl"
    "llama3b_aug"
    "qwen05b_aug"
)

job_ids=()
total_jobs=$((${#CONFIGS[@]} * 3))  # 4 configs 脳 3 seeds
job_count=0

echo "鎻愪氦 $total_jobs 涓换鍔?(${#CONFIGS[@]} 閰嶇疆 脳 3 绉嶅瓙)..."
echo ""

for config in "${CONFIGS[@]}"; do
    echo "閰嶇疆: $config"
    
    # 鎻愪氦浠诲姟锛堜娇鐢?array job锛? 涓瀛愶級
    job_id=$(sbatch \
        --job-name="kava_${config}" \
        --output="outputs/logs/kava_${config}_%A_%a.out" \
        --error="outputs/logs/kava_${config}_%A_%a.err" \
        --export=ALL,CONFIG="$config" \
        submit_multi_seed.slurm | awk '{print $4}')
    
    if [ -n "$job_id" ]; then
        job_ids+=("$job_id")
        echo -e "  ${GREEN}鉁?{NC} 浠诲姟宸叉彁浜? $job_id (3 涓瓙浠诲姟)"
        job_count=$((job_count + 3))
    else
        echo -e "  ${YELLOW}鉁?{NC} 浠诲姟鎻愪氦澶辫触"
    fi
done

echo ""
echo -e "${GREEN}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo -e "${GREEN}  浠诲姟鎻愪氦瀹屾垚锛?{NC}"
echo -e "${GREEN}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo ""

# 淇濆瓨浠诲姟 ID
printf "%s\n" "${job_ids[@]}" > outputs/job_ids.txt
echo "宸叉彁浜や换鍔? ${#job_ids[@]} 涓富浠诲姟 (鍏?$job_count 涓瓙浠诲姟)"
echo "浠诲姟 ID 宸蹭繚瀛樺埌: outputs/job_ids.txt"
echo ""

# =============================================================================
# 鐢熸垚鐩戞帶鑴氭湰
# =============================================================================

echo "鐢熸垚鐩戞帶鑴氭湰..."

# monitor_jobs.sh
cat > monitor_jobs.sh << 'MONITOR_EOF'
#!/bin/bash
echo "=== KAVA 璁粌浠诲姟鐘舵€?==="
echo ""
squeue -u $USER --format="%.18i %.12j %.8T %.10M %.6D %.15R" | grep -E "JOBID|kava"
echo ""
echo "杩愯涓? $(squeue -u $USER | grep -c kava || echo 0)"
echo "鎬讳换鍔? $(cat outputs/job_ids.txt 2>/dev/null | wc -l || echo 0) 涓富浠诲姟"
MONITOR_EOF

chmod +x monitor_jobs.sh
echo -e "${GREEN}鉁?{NC} 宸插垱寤?monitor_jobs.sh"

# collect_results.sh
cat > collect_results.sh << 'COLLECT_EOF'
#!/bin/bash
echo "=== 鏀堕泦 KAVA 璁粌缁撴灉 ==="
echo ""

results_file="outputs/aggregated_results.csv"
echo "Config,Seed,EM,F1,Status" > "$results_file"

for log_file in outputs/logs/kava_*.out; do
    if [ -f "$log_file" ]; then
        # 鎻愬彇閰嶇疆鍚嶅拰绉嶅瓙
        base=$(basename "$log_file" .out)
        config=$(echo "$base" | sed 's/kava_\(.*\)_[0-9]*_[0-9]*/\1/')
        
        # 浠庢枃浠朵腑鎻愬彇绉嶅瓙锛堝鏋滄湁鐨勮瘽锛?        seed=$(grep "Seed:" "$log_file" | head -1 | awk '{print $2}')
        
        if grep -q "Final Test EM" "$log_file"; then
            em=$(grep "Final Test EM" "$log_file" | tail -1 | awk '{print $NF}')
            f1=$(grep "Final Test F1" "$log_file" | tail -1 | awk '{print $NF}')
            echo "$config,$seed,$em,$f1,COMPLETED" >> "$results_file"
        else
            echo "$config,$seed,N/A,N/A,IN_PROGRESS" >> "$results_file"
        fi
    fi
done

echo "缁撴灉宸蹭繚瀛樺埌: $results_file"
echo ""
cat "$results_file" | column -t -s,
COLLECT_EOF

chmod +x collect_results.sh
echo -e "${GREEN}鉁?{NC} 宸插垱寤?collect_results.sh"

echo ""
echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo -e "${BLUE}  涓嬩竴姝ユ搷浣?{NC}"
echo -e "${BLUE}鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?{NC}"
echo ""
echo "1. 妫€鏌ヤ换鍔＄姸鎬?"
echo "   bash monitor_jobs.sh"
echo "   鎴? squeue -u \$USER"
echo ""
echo "2. 鏌ョ湅瀹炴椂鏃ュ織:"
echo "   tail -f outputs/logs/kava_*.out"
echo ""
echo "3. 鍙栨秷浠诲姟 (濡傞渶瑕?:"
echo "   scancel <job_id>"
echo "   鎴栧彇娑堟墍鏈? scancel -u \$USER"
echo ""
echo "4. 璁粌瀹屾垚鍚庢敹闆嗙粨鏋?(36-48 灏忔椂鍚?:"
echo "   bash collect_results.sh"
echo ""
echo -e "${GREEN}鉁?鎵€鏈変换鍔″凡鎻愪氦鍒?SLURM 闃熷垪锛?{NC}"
echo -e "${YELLOW}鈴?棰勮瀹屾垚鏃堕棿: 36-48 灏忔椂${NC}"
echo ""
