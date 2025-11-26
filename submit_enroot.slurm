#!/usr/bin/bash
#==============================================================================
# KAVA Enroot 瀹瑰櫒璁粌鑴氭湰
# 鐢ㄩ€旓細浣跨敤 Enroot 瀹瑰櫒鍦?HPC 涓婅缁?KAVA 妯″瀷
# 鐢ㄦ硶锛歴batch --export=CONFIG=llama1b_aug submit_enroot.slurm
#==============================================================================

#SBATCH --job-name=kava-enroot
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kava_enroot_%A_%a.out
#SBATCH --error=logs/kava_enroot_%A_%a.err
#SBATCH --array=0-2  # 3 涓瀛?

# ========== Enroot 瀹瑰櫒閰嶇疆 ==========
#SBATCH --container-writable                              # 瀹瑰櫒鍐呭彲鍐?
#SBATCH --container-mount-home                            # 鎸傝浇瀹剁洰褰?
#SBATCH --container-mounts /home/share/models:/models:ro  # 鎸傝浇鍏叡妯″瀷搴?
#SBATCH --container-image pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh  # 闀滃儚璺緞

#==============================================================================
# 鈿狅笍 娉ㄦ剰锛氫互涓嬫墍鏈夊懡浠ら兘鍦ㄥ鍣ㄥ唴鎵ц
#==============================================================================

# 閰嶇疆鍙傛暟
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

# 1. 楠岃瘉鐜
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

# 2. 閰嶇疆 HuggingFace 浣跨敤鎸傝浇鐨勫叕鍏辨ā鍨嬪簱
echo "Configuring HuggingFace cache..."
export HF_HOME=/models
export TRANSFORMERS_CACHE=/models
export HF_DATASETS_CACHE=/models
echo "HF_HOME: $HF_HOME"
python -c "import os; print(f'Models directory exists: {os.path.exists(\"/models\")}')"
echo ""

# 3. 鍒囨崲鍒伴」鐩洰褰曪紙鍋囪鎸傝浇浜嗗鐩綍锛?
cd $HOME/kava || { echo "Error: Project directory not found"; exit 1; }
echo "Working directory: $(pwd)"
echo ""

# 4. 妫€鏌ヤ緷璧栵紙濡傛灉瀹瑰櫒鍐呮湭棰勮锛?
echo "Checking dependencies..."
python -c "import transformers, peft, torch" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install --no-cache-dir -r requirements.txt
    pip install peft wandb bitsandbytes
}
echo ""

# 5. 璁剧疆杈撳嚭鐩綍
OUTPUT_DIR="outputs/${CONFIG}_multi_seed/seed_${SEED}"
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# 6. 杩愯璁粌
echo "Starting training at $(date)"
python train.py \
    --config configs/${CONFIG}.yaml \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --use_wandb

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
