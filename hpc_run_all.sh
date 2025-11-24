#!/bin/bash

#==============================================================================
# KAVA HPC 蹇€熷惎鍔ㄨ剼鏈?# 鑷姩鎻愪氦鎵€鏈夎缁冧换鍔″苟鑱氬悎缁撴灉
#
# 鐢ㄦ硶:
#   ./hpc_run_all.sh                          # 杩愯鎵€鏈夐厤缃?#   ./hpc_run_all.sh llama1b_aug              # 浠呰繍琛屾寚瀹氶厤缃?#   ./hpc_run_all.sh llama1b_aug qwen05b_aug  # 杩愯澶氫釜閰嶇疆
#==============================================================================

set -e  # 閬囧埌閿欒绔嬪嵆閫€鍑?
# 棰滆壊杈撳嚭
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 閰嶇疆鍒楄〃锛堝鏋滄病鏈夋寚瀹氾紝杩愯鎵€鏈夛級
if [ $# -eq 0 ]; then
    CONFIGS=("llama1b_aug" "llama1b_aug_nl" "qwen05b_aug" "llama3b_aug")
else
    CONFIGS=("$@")
fi

echo -e "${CYAN}=========================================="
echo "KAVA HPC 鎵归噺浠诲姟鎻愪氦"
echo "==========================================${NC}"
echo "灏嗘彁浜や互涓嬮厤缃細"
for config in "${CONFIGS[@]}"; do
    echo "  - $config"
done
echo ""

# 鍒涘缓蹇呰鐨勭洰褰?mkdir -p logs
mkdir -p outputs

# 鎻愪氦璁粌浠诲姟
echo -e "${CYAN}姝ラ 1: 鎻愪氦璁粌浠诲姟${NC}"
echo "-------------------------------------------"

TRAIN_JOB_IDS=()

for config in "${CONFIGS[@]}"; do
    echo -e "${YELLOW}鎻愪氦閰嶇疆: $config${NC}"
    
    # 妫€鏌ラ厤缃枃浠舵槸鍚﹀瓨鍦?    if [ ! -f "configs/${config}.yaml" ]; then
        echo -e "${RED}鉁?閰嶇疆鏂囦欢涓嶅瓨鍦? configs/${config}.yaml${NC}"
        continue
    fi
    
    # 鎻愪氦 SLURM 鏁扮粍浠诲姟锛? 涓瀛愶級
    job_output=$(sbatch --export=CONFIG=$config submit_multi_seed.slurm)
    job_id=$(echo $job_output | awk '{print $4}')
    
    if [ -n "$job_id" ]; then
        echo -e "${GREEN}鉁?宸叉彁浜や换鍔?ID: $job_id${NC}"
        TRAIN_JOB_IDS+=("$job_id")
    else
        echo -e "${RED}鉁?鎻愪氦澶辫触: $config${NC}"
    fi
done

echo ""
echo -e "${GREEN}鍏辨彁浜?${#TRAIN_JOB_IDS[@]} 涓缁冧换鍔?{NC}"
echo ""

# 绛夊緟鎵€鏈夎缁冧换鍔″畬鎴?echo -e "${CYAN}姝ラ 2: 绛夊緟璁粌瀹屾垚${NC}"
echo "-------------------------------------------"
echo "鐩戞帶浠诲姟鐘舵€?(Ctrl+C 閫€鍑虹洃鎺т絾涓嶄細鍙栨秷浠诲姟)..."
echo ""

# 鏄剧ず浠诲姟鐘舵€?squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"

echo ""
echo -e "${YELLOW}鎻愮ず: 浣跨敤浠ヤ笅鍛戒护鐩戞帶浠诲姟${NC}"
echo "  watch -n 10 'squeue -u \$USER'"
echo "  tail -f logs/kava_*.out"
echo ""

# 璇㈤棶鏄惁鑷姩鎻愪氦鑱氬悎浠诲姟
read -p "鏄惁鍦ㄨ缁冨畬鎴愬悗鑷姩鎻愪氦鑱氬悎浠诲姟? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}姝ラ 3: 鎻愪氦鑱氬悎浠诲姟 (渚濊禆璁粌瀹屾垚)${NC}"
    echo "-------------------------------------------"
    
    for config in "${CONFIGS[@]}"; do
        # 鎵惧埌瀵瑰簲鐨勮缁冧换鍔?ID
        for job_id in "${TRAIN_JOB_IDS[@]}"; do
            # 鎻愪氦鑱氬悎浠诲姟锛屼緷璧栬缁冧换鍔″畬鎴?            agg_output=$(sbatch --dependency=afterok:$job_id --export=CONFIG=$config submit_aggregate.slurm)
            agg_id=$(echo $agg_output | awk '{print $4}')
            
            if [ -n "$agg_id" ]; then
                echo -e "${GREEN}鉁?鑱氬悎浠诲姟 $agg_id (绛夊緟 $job_id 瀹屾垚): $config${NC}"
            fi
            break
        done
    done
    
    echo ""
    echo -e "${GREEN}鎵€鏈変换鍔″凡鎻愪氦锛?{NC}"
else
    echo -e "${YELLOW}璺宠繃鑷姩鑱氬悎銆傝缁冨畬鎴愬悗鎵嬪姩杩愯:${NC}"
    for config in "${CONFIGS[@]}"; do
        echo "  sbatch --export=CONFIG=$config submit_aggregate.slurm"
    done
fi

echo ""
echo -e "${CYAN}=========================================="
echo "浠诲姟鎽樿"
echo "==========================================${NC}"
echo "璁粌浠诲姟 ID: ${TRAIN_JOB_IDS[@]}"
echo "鏃ュ織鐩綍: logs/"
echo "杈撳嚭鐩綍: outputs/"
echo ""
echo -e "${YELLOW}甯哥敤鍛戒护:${NC}"
echo "  squeue -u \$USER                  # 鏌ョ湅浠诲姟鐘舵€?
echo "  scancel <JOB_ID>                # 鍙栨秷浠诲姟"
echo "  scontrol show job <JOB_ID>      # 鏌ョ湅浠诲姟璇︽儏"
echo "  sacct -j <JOB_ID>               # 鏌ョ湅宸插畬鎴愪换鍔′俊鎭?
echo ""
echo -e "${GREEN}瀹屾垚锛?{NC}"
