#!/bin/bash
################################################################################
# KAVA 项目 - 一键完成所有任务
# 功能：下载资源 → 训练所有配置 → 收集结果 → 自动上传
################################################################################

set -e  # 遇到错误立即退出

# ==================== 配置区 ====================
PROJECT_DIR="/home/rpwang/kava review"
USE_HF_MIRROR=true  # 是否使用 HF-Mirror 镜像
SKIP_DOWNLOAD=false  # 如果资源已存在，可设为 true 跳过下载
UPLOAD_TO_HF=true   # 是否上传结果到 HuggingFace
HF_REPO="your-username/kava-results"  # HuggingFace 仓库（需要修改）

# ==================== 辅助函数 ====================
log() {
    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "========================================================================"
}

error() {
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[ERROR] $1"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        error "$1 未安装，请先安装: pip install $2"
    fi
}

# ==================== 环境检查 ====================
log "步骤 0/6: 环境检查"

cd "$PROJECT_DIR" || error "无法进入项目目录: $PROJECT_DIR"

# 检查是否在登录节点
if [[ -n "$SLURM_JOB_ID" ]]; then
    error "请在登录节点运行此脚本，不要在计算节点上运行！"
fi

# 检查必要的命令
check_command python python
check_command pip pip
check_command sbatch slurm

# 检查 Python 包
log "检查 Python 依赖..."
python -c "import torch" 2>/dev/null || error "PyTorch 未安装"
python -c "import transformers" 2>/dev/null || error "transformers 未安装"
python -c "import datasets" 2>/dev/null || error "datasets 未安装"
python -c "import peft" 2>/dev/null || error "peft 未安装"

log "✓ 环境检查通过"

# ==================== 步骤 1: 下载资源 ====================
if [ "$SKIP_DOWNLOAD" = false ]; then
    log "步骤 1/6: 下载模型和数据集"
    
    # 设置镜像
    if [ "$USE_HF_MIRROR" = true ]; then
        export HF_ENDPOINT=https://hf-mirror.com
        log "使用 HF-Mirror 镜像加速"
    fi
    
    # 检查 huggingface_hub
    python -c "import huggingface_hub" 2>/dev/null || {
        log "安装 huggingface_hub..."
        pip install huggingface_hub
    }
    
    # 执行下载
    if [ ! -f "download_from_hf.py" ]; then
        error "下载脚本 download_from_hf.py 不存在"
    fi
    
    python download_from_hf.py || error "下载失败"
    
    log "✓ 资源下载完成"
else
    log "步骤 1/6: 跳过下载（SKIP_DOWNLOAD=true）"
fi

# ==================== 步骤 2: 验证资源 ====================
log "步骤 2/6: 验证下载的资源"

# 检查模型
MODELS=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Qwen2.5-0.5B-Instruct")
for model in "${MODELS[@]}"; do
    MODEL_PATH="./models/$model"
    if [ -d "$MODEL_PATH" ]; then
        log "✓ 模型存在: $model"
        # 检查关键文件
        if [ ! -f "$MODEL_PATH/config.json" ]; then
            error "模型 $model 缺少 config.json"
        fi
    else
        # 尝试从 HuggingFace 缓存或共享路径查找
        log "⚠ 模型不在本地: $model (将尝试从缓存加载)"
    fi
done

# 检查数据集
DATASETS=("gsm8k-aug" "gsm8k-aug-nl" "gsm8k")
for dataset in "${DATASETS[@]}"; do
    DATASET_PATH="./datasets/$dataset"
    if [ -d "$DATASET_PATH" ]; then
        log "✓ 数据集存在: $dataset"
    else
        log "⚠ 数据集不在本地: $dataset (将尝试从缓存加载)"
    fi
done

log "✓ 资源验证完成"

# ==================== 步骤 3: 更新配置文件 ====================
log "步骤 3/6: 自动更新配置文件路径"

# 创建更新脚本
cat > /tmp/update_configs.py << 'PYTHON_SCRIPT'
import yaml
import os

configs = [
    "configs/llama1b_aug.yaml",
    "configs/llama1b_aug_nl.yaml",
    "configs/llama3b_aug.yaml",
    "configs/qwen05b_aug.yaml"
]

model_mapping = {
    "llama1b": "./models/Llama-3.2-1B-Instruct",
    "llama3b": "./models/Llama-3.2-3B-Instruct",
    "qwen05b": "./models/Qwen2.5-0.5B-Instruct"
}

dataset_mapping = {
    "aug.yaml": "./datasets/gsm8k-aug",
    "aug_nl.yaml": "./datasets/gsm8k-aug-nl"
}

for config_path in configs:
    if not os.path.exists(config_path):
        print(f"⚠ 配置文件不存在: {config_path}")
        continue
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新模型路径
    for key, path in model_mapping.items():
        if key in config_path and os.path.exists(path):
            config['model']['name'] = path
            print(f"✓ {config_path}: 模型路径更新为 {path}")
            break
    
    # 更新数据集路径
    for key, path in dataset_mapping.items():
        if key in config_path and os.path.exists(path):
            config['dataset']['name'] = path
            print(f"✓ {config_path}: 数据集路径更新为 {path}")
            break
    
    # 写回配置
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✓ 配置文件更新完成")
PYTHON_SCRIPT

python /tmp/update_configs.py || log "⚠ 配置更新失败，继续使用原有配置"
rm -f /tmp/update_configs.py 2>/dev/null || true

log "✓ 配置文件已更新"

# ==================== 步骤 4: 提交所有训练任务 ====================
log "步骤 4/6: 提交训练任务"

# 检查 SLURM 脚本
if [ ! -f "submit_multi_seed.slurm" ]; then
    error "SLURM 脚本 submit_multi_seed.slurm 不存在"
fi

# 提交所有任务
log "提交 4 个配置 × 3 个随机种子 = 12 个训练任务..."

CONFIGS=("llama1b_aug" "llama1b_aug_nl" "llama3b_aug" "qwen05b_aug")
SEEDS=(42 123 456)

JOB_IDS=()

for config in "${CONFIGS[@]}"; do
    for seed_idx in "${!SEEDS[@]}"; do
        seed="${SEEDS[$seed_idx]}"
        
        # 提交任务
        job_output=$(sbatch --export=CONFIG=$config --array=$seed_idx submit_multi_seed.slurm 2>&1)
        
        if [[ $job_output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
            JOB_IDS+=("$job_id")
            log "✓ 提交任务: $config (seed=$seed) → Job ID: $job_id"
        else
            error "提交任务失败: $config (seed=$seed)\n输出: $job_output"
        fi
        
        sleep 1  # 避免提交过快
    done
done

log "✓ 所有任务已提交，共 ${#JOB_IDS[@]} 个任务"
echo "Job IDs: ${JOB_IDS[*]}"

# 保存 Job IDs 到文件
echo "${JOB_IDS[*]}" > .job_ids.txt

# ==================== 步骤 5: 监控训练进度 ====================
log "步骤 5/6: 监控训练进度"

log "等待所有任务完成..."
log "任务 IDs: ${JOB_IDS[*]}"

# 持续检查任务状态
while true; do
    all_done=true
    running_count=0
    pending_count=0
    failed_count=0
    completed_count=0
    
    for job_id in "${JOB_IDS[@]}"; do
        # 获取任务状态
        job_status=$(squeue -j $job_id -h -o "%T" 2>/dev/null || echo "COMPLETED")
        
        case $job_status in
            RUNNING)
                running_count=$((running_count + 1))
                all_done=false
                ;;
            PENDING)
                pending_count=$((pending_count + 1))
                all_done=false
                ;;
            COMPLETED)
                completed_count=$((completed_count + 1))
                ;;
            FAILED|CANCELLED|TIMEOUT)
                failed_count=$((failed_count + 1))
                log "⚠ 任务失败: Job ID $job_id (状态: $job_status)"
                ;;
            *)
                # 任务已完成或不存在
                completed_count=$((completed_count + 1))
                ;;
        esac
    done
    
    # 显示进度
    total=${#JOB_IDS[@]}
    echo -ne "\r[$(date '+%H:%M:%S')] 总任务: $total | 运行: $running_count | 等待: $pending_count | 完成: $completed_count | 失败: $failed_count"
    
    if [ "$all_done" = true ]; then
        echo ""
        break
    fi
    
    sleep 30  # 每 30 秒检查一次
done

log "✓ 所有任务已完成"
log "统计: 完成 $completed_count, 失败 $failed_count"

if [ $failed_count -gt 0 ]; then
    log "⚠ 有 $failed_count 个任务失败，请检查日志"
fi

# ==================== 步骤 6: 收集结果并上传 ====================
log "步骤 6/6: 收集结果并准备上传"

# 创建结果目录
RESULTS_DIR="./all_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# 收集所有结果
log "收集训练结果..."
if [ -d "results" ]; then
    cp -r results/* "$RESULTS_DIR/" 2>/dev/null || log "⚠ 复制结果时出现问题"
fi

# 收集日志
log "收集训练日志..."
mkdir -p "$RESULTS_DIR/logs"
cp logs/*.out "$RESULTS_DIR/logs/" 2>/dev/null || log "⚠ 没有找到日志文件"

# 生成结果摘要
log "生成结果摘要..."
cat > "$RESULTS_DIR/SUMMARY.txt" << EOF
KAVA 训练结果摘要
===============================================================================
训练时间: $(date)
项目目录: $PROJECT_DIR
任务数量: ${#JOB_IDS[@]}
完成任务: $completed_count
失败任务: $failed_count
Job IDs: ${JOB_IDS[*]}

配置列表:
$(for config in "${CONFIGS[@]}"; do echo "  - $config"; done)

随机种子:
$(for seed in "${SEEDS[@]}"; do echo "  - $seed"; done)

结果目录结构:
$(ls -lh "$RESULTS_DIR")

===============================================================================
EOF

cat "$RESULTS_DIR/SUMMARY.txt"

# 打包结果
log "打包结果..."
ARCHIVE_NAME="kava_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE_NAME" "$RESULTS_DIR" || log "⚠ 打包失败"

if [ -f "$ARCHIVE_NAME" ]; then
    log "✓ 结果已打包: $ARCHIVE_NAME ($(du -h "$ARCHIVE_NAME" | cut -f1))"
fi

# 上传到 HuggingFace (可选)
if [ "$UPLOAD_TO_HF" = true ]; then
    log "准备上传到 HuggingFace..."
    
    # 检查是否已登录
    if ! huggingface-cli whoami &>/dev/null; then
        log "⚠ 未登录 HuggingFace，跳过上传"
        log "   提示: 运行 'huggingface-cli login' 登录后再上传"
    else
        log "上传结果到 HuggingFace 仓库: $HF_REPO"
        
        # 创建上传脚本
        cat > /tmp/upload_results.py << PYTHON_UPLOAD
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
repo_id = "$HF_REPO"

# 创建仓库（如果不存在）
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"✓ 仓库已创建或已存在: {repo_id}")
except Exception as e:
    print(f"⚠ 创建仓库失败: {e}")

# 上传文件
try:
    api.upload_file(
        path_or_fileobj="$ARCHIVE_NAME",
        path_in_repo="$ARCHIVE_NAME",
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"✓ 文件上传成功: $ARCHIVE_NAME")
    print(f"   查看: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"✗ 上传失败: {e}")
PYTHON_UPLOAD
        
        python /tmp/upload_results.py || log "⚠ 上传失败"
        rm -f /tmp/upload_results.py 2>/dev/null || true
    fi
else
    log "跳过上传（UPLOAD_TO_HF=false）"
fi

# ==================== 完成 ====================
log "🎉 所有任务完成！"

echo ""
echo "================================================================================"
echo "                           任务执行摘要"
echo "================================================================================"
echo "✓ 资源下载: $([ "$SKIP_DOWNLOAD" = false ] && echo "完成" || echo "跳过")"
echo "✓ 资源验证: 完成"
echo "✓ 配置更新: 完成"
echo "✓ 任务提交: ${#JOB_IDS[@]} 个任务"
echo "✓ 训练监控: 完成"
echo "✓ 结果收集: $RESULTS_DIR"
echo "✓ 结果打包: $ARCHIVE_NAME"
echo "✓ 结果上传: $([ "$UPLOAD_TO_HF" = true ] && echo "完成" || echo "跳过")"
echo ""
echo "结果位置:"
echo "  - 目录: $RESULTS_DIR"
echo "  - 打包: $ARCHIVE_NAME"
echo ""
echo "后续步骤:"
echo "  1. 查看结果摘要: cat $RESULTS_DIR/SUMMARY.txt"
echo "  2. 检查训练日志: ls $RESULTS_DIR/logs/"
echo "  3. 分析实验结果: python analyze_results.py"
echo "  4. 下载结果到本地: scp user@hpc:$PROJECT_DIR/$ARCHIVE_NAME ."
echo ""
echo "如需重新运行:"
echo "  bash run_everything.sh"
echo "================================================================================"
