#!/usr/bin/bash
#==============================================================================
# KAVA Enroot 容器训练脚本
# 用途：使用 Enroot 容器在 HPC 上训练 KAVA 模型
# 用法：sbatch --export=CONFIG=llama1b_aug submit_enroot.slurm
#==============================================================================

#SBATCH --job-name=kava-enroot
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/kava_enroot_%A_%a.out
#SBATCH --error=logs/kava_enroot_%A_%a.err
#SBATCH --array=0-2  # 3 个种子

# ========== Enroot 容器配置 ==========
#SBATCH --container-writable                              # 容器内可写
#SBATCH --container-mount-home                            # 挂载家目录
#SBATCH --container-mounts /home/share/models:/models:ro  # 挂载公共模型库
#SBATCH --container-image pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh  # 镜像路径

#==============================================================================
# ⚠️ 注意：以下所有命令都在容器内执行
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
echo "Container: Running inside Enroot container"
echo "=========================================="

# 1. 验证环境
echo "Verifying environment..."
echo "CUDA Version:"
nvcc --version || echo "nvcc not found (runtime image)"
echo ""

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "PyTorch Info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

# 2. 配置 HuggingFace 使用挂载的公共模型库
echo "Configuring HuggingFace cache..."
export HF_HOME=/models
export TRANSFORMERS_CACHE=/models
export HF_DATASETS_CACHE=/models
echo "HF_HOME: $HF_HOME"
python -c "import os; print(f'Models directory exists: {os.path.exists(\"/models\")}')"
echo ""

# 3. 切换到项目目录（假设挂载了家目录）
cd $HOME/kava || { echo "Error: Project directory not found"; exit 1; }
echo "Working directory: $(pwd)"
echo ""

# 4. 检查依赖（如果容器内未预装）
echo "Checking dependencies..."
python -c "import transformers, peft, torch" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install --no-cache-dir -r requirements.txt
    pip install peft wandb bitsandbytes
}
echo ""

# 5. 设置输出目录
OUTPUT_DIR="outputs/${CONFIG}_multi_seed/seed_${SEED}"
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

# 评估数据集
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
