#!/bin/bash
#SBATCH --job-name=kava-aggregate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

#==============================================================================
# KAVA 结果聚合 SLURM 脚本
# 用法: sbatch --export=CONFIG=llama1b_aug submit_aggregate.slurm
#==============================================================================

# 配置参数
CONFIG=${CONFIG:-"llama1b_aug"}

echo "=========================================="
echo "Aggregating results for: $CONFIG"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# 加载 Python 环境
. /usr/share/modules/init/bash
module load anaconda3
source activate kava

# 输入和输出目录
INPUT_DIR="${SLURM_SUBMIT_DIR}/outputs/${CONFIG}_multi_seed"
OUTPUT_JSON="${INPUT_DIR}/aggregated_results.json"
OUTPUT_YAML="${INPUT_DIR}/aggregated_results.yaml"

echo "Input directory: $INPUT_DIR"
echo "Output files:"
echo "  - $OUTPUT_JSON"
echo "  - $OUTPUT_YAML"
echo ""

# 检查种子目录
SEED_DIRS=()
for seed in 42 123 456; do
    seed_dir="${INPUT_DIR}/seed_${seed}"
    if [ -d "$seed_dir" ]; then
        echo "✓ Found seed directory: $seed_dir"
        SEED_DIRS+=("$seed_dir")
    else
        echo "✗ Missing seed directory: $seed_dir"
    fi
done

if [ ${#SEED_DIRS[@]} -eq 0 ]; then
    echo "✗ No valid seed directories found!"
    exit 1
fi

echo ""
echo "Found ${#SEED_DIRS[@]} seed directories"
echo "=========================================="

# 运行聚合
echo "Running aggregation..."
python aggregate_multi_seed.py \
    --seed_dirs "${SEED_DIRS[@]}" \
    --datasets gsm8k gsm8k-hard svamp \
    --model_name "KAVA-${CONFIG}" \
    --output_json "$OUTPUT_JSON" \
    --output_yaml "$OUTPUT_YAML"

if [ $? -eq 0 ]; then
    echo "✓ Aggregation completed successfully"
    echo ""
    echo "Results preview:"
    head -n 30 "$OUTPUT_YAML"
else
    echo "✗ Aggregation failed"
    exit 1
fi

echo "=========================================="
echo "Job completed at $(date)"
