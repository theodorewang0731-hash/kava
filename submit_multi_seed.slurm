#!/bin/bash
#SBATCH --job-name=kava-multi-seed
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-sxm4-80gb:1   # A100 80GB锛坓pu06/08/09/19 鍙敤锛?
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/kava_%A_%a.out
#SBATCH --error=logs/kava_%A_%a.err
#SBATCH --array=0-2                   # 3 涓瀛愶細0, 1, 2
# #SBATCH -w gpu10                    # 鍙€夛細鎸囧畾鑺傜偣锛坓pu10-gpu14 鏀寔 SSH锛?

#==============================================================================
# KAVA 澶氱瀛愯缁?SLURM 鑴氭湰
# 鐢ㄦ硶: sbatch --export=CONFIG=llama1b_aug submit_multi_seed.slurm
#==============================================================================

# 閰嶇疆鍙傛暟
CONFIG=${CONFIG:-"llama1b_aug"}      # 榛樿閰嶇疆
SEEDS=(42 123 456)                   # 3 涓殢鏈虹瀛?
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# 1. 鍔犺浇 CUDA Module
echo "Loading CUDA module..."
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/11.8.0

# 2. 婵€娲?Python 铏氭嫙鐜锛坴env锛?
echo "Activating Python virtual environment..."
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# 3. 閰嶇疆 HuggingFace 缂撳瓨锛堜娇鐢?HPC 鍏变韩妯″瀷搴擄級
echo "Configuring HuggingFace cache..."
# 鉁?浣跨敤 HPC 鍏变韩妯″瀷搴擄紙宸茬‘璁ゅ寘鍚墍闇€妯″瀷锛?
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=$HOME/.cache/huggingface  # 鏁版嵁闆嗕粛鐢ㄤ釜浜虹洰褰?

# 鉁?寮哄埗绂荤嚎妯″紡 - 閬垮厤缃戠粶璁块棶锛堣В鍐?"Network is unreachable" 闂锛?
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p $HF_DATASETS_CACHE
echo "Using HPC shared models: $HF_HOME"
echo "Offline mode enabled: HUGGINGFACE_HUB_OFFLINE=1"

# 4. 楠岃瘉鐜
echo "Verifying environment..."
echo "CUDA Version:"
nvcc -V
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
echo "PyTorch Info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
echo ""
echo "HuggingFace Cache:"
python -c "import os; print(f'HF_HOME: {os.environ.get(\"HF_HOME\")}'); print(f'Cache exists: {os.path.exists(os.environ.get(\"HF_HOME\", \"\"))}')"
echo ""

# 5. 璁剧疆杈撳嚭鐩綍
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs/${CONFIG}_multi_seed/seed_${SEED}"
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# 6. 杩愯璁粌
echo "Starting training at $(date)"
python train.py \
    --config "configs/${CONFIG}.yaml" \
    --output_dir "$OUTPUT_DIR" \
    --seed $SEED

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "鉁?Training completed successfully"
else
    echo "鉁?Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

# 7. 杩愯璇勪及
echo "=========================================="
echo "Starting evaluation at $(date)"

CHECKPOINT_DIR="${OUTPUT_DIR}/best_checkpoint"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "鉁?Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# 璇勪及鏁版嵁闆?
EVAL_DATASETS=("gsm8k" "gsm8k-hard" "svamp")

for dataset in "${EVAL_DATASETS[@]}"; do
    echo "Evaluating on $dataset..."
    
    python evaluate.py \
        --checkpoint_dir $CHECKPOINT_DIR \
        --eval_dataset $dataset \
        --output ${OUTPUT_DIR}/results_${dataset}.yaml \
        --seed $SEED
    
    if [ $? -eq 0 ]; then
        echo "鉁?$dataset evaluation completed"
    else
        echo "鉁?$dataset evaluation failed"
    fi
done

echo "=========================================="
echo "Job completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# 8. 鎵撳嵃璧勬簮浣跨敤鎯呭喌
echo "Resource Usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
