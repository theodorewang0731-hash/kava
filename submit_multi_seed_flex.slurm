#!/bin/bash
#SBATCH --job-name=kava-multi-seed
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # 任意可用 GPU（更灵活）
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/kava_%A_%a.out
#SBATCH --error=logs/kava_%A_%a.err
#SBATCH --array=0-2                   # 3 个种子：0, 1, 2

#==============================================================================
# KAVA 多种子训练 SLURM 脚本（灵活 GPU 版本）
# 用法: sbatch --export=CONFIG=llama1b_aug submit_multi_seed_flex.slurm
#==============================================================================

# 配置参数
CONFIG=${CONFIG:-"llama1b_aug"}
SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# 1. 加载 CUDA Module
echo "Loading CUDA module..."
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/11.8.0

# 2. 激活 Python 虚拟环境（venv）
echo "Activating Python virtual environment..."
cd $SLURM_SUBMIT_DIR
source venv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# 3. 配置 HuggingFace 缓存（使用 HPC 共享模型库）
echo "Configuring HuggingFace cache..."
export HF_HOME=/home/share/models
export TRANSFORMERS_CACHE=/home/share/models
export HF_DATASETS_CACHE=$HOME/.cache/huggingface
mkdir -p $HF_DATASETS_CACHE
echo "Using HPC shared models: $HF_HOME"

# 4. 验证环境
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

# 5. 设置输出目录
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs/${CONFIG}_multi_seed/seed_${SEED}"
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# 6. 运行训练
echo "Starting training at $(date)"
python train.py \
    --config configs/${CONFIG}.yaml \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --use_wandb

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

# 7. 运行评估
echo "=========================================="
echo "Starting evaluation at $(date)"

CHECKPOINT_DIR="${OUTPUT_DIR}/best_checkpoint"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "✗ Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

EVAL_DATASETS=("gsm8k" "gsm8k-hard" "svamp")

for dataset in "${EVAL_DATASETS[@]}"; do
    echo "Evaluating on $dataset..."
    
    python evaluate.py \
        --checkpoint_dir $CHECKPOINT_DIR \
        --eval_dataset $dataset \
        --output ${OUTPUT_DIR}/results_${dataset}.yaml \
        --seed $SEED
    
    if [ $? -eq 0 ]; then
        echo "✓ $dataset evaluation completed"
    else
        echo "✗ $dataset evaluation failed"
    fi
done

echo "=========================================="
echo "Job completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# 8. 打印资源使用情况
echo "Resource Usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
