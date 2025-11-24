#!/bin/bash
################################################################################
# KAVA 椤圭洰 - 涓€閿畬鎴愭墍鏈変换鍔?# 鍔熻兘锛氫笅杞借祫婧?鈫?璁粌鎵€鏈夐厤缃?鈫?鏀堕泦缁撴灉 鈫?鑷姩涓婁紶
################################################################################

set -e  # 閬囧埌閿欒绔嬪嵆閫€鍑?
# ==================== 閰嶇疆鍖?====================
PROJECT_DIR="/home/rpwang/kava review"
USE_HF_MIRROR=true  # 鏄惁浣跨敤 HF-Mirror 闀滃儚
SKIP_DOWNLOAD=false  # 濡傛灉璧勬簮宸插瓨鍦紝鍙涓?true 璺宠繃涓嬭浇
UPLOAD_TO_HF=true   # 鏄惁涓婁紶缁撴灉鍒?HuggingFace
HF_REPO="your-username/kava-results"  # HuggingFace 浠撳簱锛堥渶瑕佷慨鏀癸級

# ==================== 杈呭姪鍑芥暟 ====================
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
        error "$1 鏈畨瑁咃紝璇峰厛瀹夎: pip install $2"
    fi
}

# ==================== 鐜妫€鏌?====================
log "姝ラ 0/6: 鐜妫€鏌?

cd "$PROJECT_DIR" || error "鏃犳硶杩涘叆椤圭洰鐩綍: $PROJECT_DIR"

# 妫€鏌ユ槸鍚﹀湪鐧诲綍鑺傜偣
if [[ -n "$SLURM_JOB_ID" ]]; then
    error "璇峰湪鐧诲綍鑺傜偣杩愯姝よ剼鏈紝涓嶈鍦ㄨ绠楄妭鐐逛笂杩愯锛?
fi

# 妫€鏌ュ繀瑕佺殑鍛戒护
check_command python python
check_command pip pip
check_command sbatch slurm

# 妫€鏌?Python 鍖?log "妫€鏌?Python 渚濊禆..."
python -c "import torch" 2>/dev/null || error "PyTorch 鏈畨瑁?
python -c "import transformers" 2>/dev/null || error "transformers 鏈畨瑁?
python -c "import datasets" 2>/dev/null || error "datasets 鏈畨瑁?
python -c "import peft" 2>/dev/null || error "peft 鏈畨瑁?

log "鉁?鐜妫€鏌ラ€氳繃"

# ==================== 姝ラ 1: 涓嬭浇璧勬簮 ====================
if [ "$SKIP_DOWNLOAD" = false ]; then
    log "姝ラ 1/6: 涓嬭浇妯″瀷鍜屾暟鎹泦"
    
    # 璁剧疆闀滃儚
    if [ "$USE_HF_MIRROR" = true ]; then
        export HF_ENDPOINT=https://hf-mirror.com
        log "浣跨敤 HF-Mirror 闀滃儚鍔犻€?
    fi
    
    # 妫€鏌?huggingface_hub
    python -c "import huggingface_hub" 2>/dev/null || {
        log "瀹夎 huggingface_hub..."
        pip install huggingface_hub
    }
    
    # 鎵ц涓嬭浇
    if [ ! -f "download_from_hf.py" ]; then
        error "涓嬭浇鑴氭湰 download_from_hf.py 涓嶅瓨鍦?
    fi
    
    python download_from_hf.py || error "涓嬭浇澶辫触"
    
    log "鉁?璧勬簮涓嬭浇瀹屾垚"
else
    log "姝ラ 1/6: 璺宠繃涓嬭浇锛圫KIP_DOWNLOAD=true锛?
fi

# ==================== 姝ラ 2: 楠岃瘉璧勬簮 ====================
log "姝ラ 2/6: 楠岃瘉涓嬭浇鐨勮祫婧?

# 妫€鏌ユā鍨?MODELS=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Qwen2.5-0.5B-Instruct")
for model in "${MODELS[@]}"; do
    MODEL_PATH="./models/$model"
    if [ -d "$MODEL_PATH" ]; then
        log "鉁?妯″瀷瀛樺湪: $model"
        # 妫€鏌ュ叧閿枃浠?        if [ ! -f "$MODEL_PATH/config.json" ]; then
            error "妯″瀷 $model 缂哄皯 config.json"
        fi
    else
        # 灏濊瘯浠?HuggingFace 缂撳瓨鎴栧叡浜矾寰勬煡鎵?        log "鈿?妯″瀷涓嶅湪鏈湴: $model (灏嗗皾璇曚粠缂撳瓨鍔犺浇)"
    fi
done

# 妫€鏌ユ暟鎹泦
DATASETS=("gsm8k-aug" "gsm8k-aug-nl" "gsm8k")
for dataset in "${DATASETS[@]}"; do
    DATASET_PATH="./datasets/$dataset"
    if [ -d "$DATASET_PATH" ]; then
        log "鉁?鏁版嵁闆嗗瓨鍦? $dataset"
    else
        log "鈿?鏁版嵁闆嗕笉鍦ㄦ湰鍦? $dataset (灏嗗皾璇曚粠缂撳瓨鍔犺浇)"
    fi
done

log "鉁?璧勬簮楠岃瘉瀹屾垚"

# ==================== 姝ラ 3: 鏇存柊閰嶇疆鏂囦欢 ====================
log "姝ラ 3/6: 鑷姩鏇存柊閰嶇疆鏂囦欢璺緞"

# 鍒涘缓鏇存柊鑴氭湰
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
        print(f"鈿?閰嶇疆鏂囦欢涓嶅瓨鍦? {config_path}")
        continue
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 鏇存柊妯″瀷璺緞
    for key, path in model_mapping.items():
        if key in config_path and os.path.exists(path):
            config['model']['name'] = path
            print(f"鉁?{config_path}: 妯″瀷璺緞鏇存柊涓?{path}")
            break
    
    # 鏇存柊鏁版嵁闆嗚矾寰?    for key, path in dataset_mapping.items():
        if key in config_path and os.path.exists(path):
            config['dataset']['name'] = path
            print(f"鉁?{config_path}: 鏁版嵁闆嗚矾寰勬洿鏂颁负 {path}")
            break
    
    # 鍐欏洖閰嶇疆
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("鉁?閰嶇疆鏂囦欢鏇存柊瀹屾垚")
PYTHON_SCRIPT

python /tmp/update_configs.py || log "鈿?閰嶇疆鏇存柊澶辫触锛岀户缁娇鐢ㄥ師鏈夐厤缃?
rm -f /tmp/update_configs.py 2>/dev/null || true

log "鉁?閰嶇疆鏂囦欢宸叉洿鏂?

# ==================== 姝ラ 4: 鎻愪氦鎵€鏈夎缁冧换鍔?====================
log "姝ラ 4/6: 鎻愪氦璁粌浠诲姟"

# 妫€鏌?SLURM 鑴氭湰
if [ ! -f "submit_multi_seed.slurm" ]; then
    error "SLURM 鑴氭湰 submit_multi_seed.slurm 涓嶅瓨鍦?
fi

# 鎻愪氦鎵€鏈変换鍔?log "鎻愪氦 4 涓厤缃?脳 3 涓殢鏈虹瀛?= 12 涓缁冧换鍔?.."

CONFIGS=("llama1b_aug" "llama1b_aug_nl" "llama3b_aug" "qwen05b_aug")
SEEDS=(42 123 456)

JOB_IDS=()

for config in "${CONFIGS[@]}"; do
    for seed_idx in "${!SEEDS[@]}"; do
        seed="${SEEDS[$seed_idx]}"
        
        # 鎻愪氦浠诲姟
        job_output=$(sbatch --export=CONFIG=$config --array=$seed_idx submit_multi_seed.slurm 2>&1)
        
        if [[ $job_output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
            JOB_IDS+=("$job_id")
            log "鉁?鎻愪氦浠诲姟: $config (seed=$seed) 鈫?Job ID: $job_id"
        else
            error "鎻愪氦浠诲姟澶辫触: $config (seed=$seed)\n杈撳嚭: $job_output"
        fi
        
        sleep 1  # 閬垮厤鎻愪氦杩囧揩
    done
done

log "鉁?鎵€鏈変换鍔″凡鎻愪氦锛屽叡 ${#JOB_IDS[@]} 涓换鍔?
echo "Job IDs: ${JOB_IDS[*]}"

# 淇濆瓨 Job IDs 鍒版枃浠?echo "${JOB_IDS[*]}" > .job_ids.txt

# ==================== 姝ラ 5: 鐩戞帶璁粌杩涘害 ====================
log "姝ラ 5/6: 鐩戞帶璁粌杩涘害"

log "绛夊緟鎵€鏈変换鍔″畬鎴?.."
log "浠诲姟 IDs: ${JOB_IDS[*]}"

# 鎸佺画妫€鏌ヤ换鍔＄姸鎬?while true; do
    all_done=true
    running_count=0
    pending_count=0
    failed_count=0
    completed_count=0
    
    for job_id in "${JOB_IDS[@]}"; do
        # 鑾峰彇浠诲姟鐘舵€?        job_status=$(squeue -j $job_id -h -o "%T" 2>/dev/null || echo "COMPLETED")
        
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
                log "鈿?浠诲姟澶辫触: Job ID $job_id (鐘舵€? $job_status)"
                ;;
            *)
                # 浠诲姟宸插畬鎴愭垨涓嶅瓨鍦?                completed_count=$((completed_count + 1))
                ;;
        esac
    done
    
    # 鏄剧ず杩涘害
    total=${#JOB_IDS[@]}
    echo -ne "\r[$(date '+%H:%M:%S')] 鎬讳换鍔? $total | 杩愯: $running_count | 绛夊緟: $pending_count | 瀹屾垚: $completed_count | 澶辫触: $failed_count"
    
    if [ "$all_done" = true ]; then
        echo ""
        break
    fi
    
    sleep 30  # 姣?30 绉掓鏌ヤ竴娆?done

log "鉁?鎵€鏈変换鍔″凡瀹屾垚"
log "缁熻: 瀹屾垚 $completed_count, 澶辫触 $failed_count"

if [ $failed_count -gt 0 ]; then
    log "鈿?鏈?$failed_count 涓换鍔″け璐ワ紝璇锋鏌ユ棩蹇?
fi

# ==================== 姝ラ 6: 鏀堕泦缁撴灉骞朵笂浼?====================
log "姝ラ 6/6: 鏀堕泦缁撴灉骞跺噯澶囦笂浼?

# 鍒涘缓缁撴灉鐩綍
RESULTS_DIR="./all_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# 鏀堕泦鎵€鏈夌粨鏋?log "鏀堕泦璁粌缁撴灉..."
if [ -d "results" ]; then
    cp -r results/* "$RESULTS_DIR/" 2>/dev/null || log "鈿?澶嶅埗缁撴灉鏃跺嚭鐜伴棶棰?
fi

# 鏀堕泦鏃ュ織
log "鏀堕泦璁粌鏃ュ織..."
mkdir -p "$RESULTS_DIR/logs"
cp logs/*.out "$RESULTS_DIR/logs/" 2>/dev/null || log "鈿?娌℃湁鎵惧埌鏃ュ織鏂囦欢"

# 鐢熸垚缁撴灉鎽樿
log "鐢熸垚缁撴灉鎽樿..."
cat > "$RESULTS_DIR/SUMMARY.txt" << EOF
KAVA 璁粌缁撴灉鎽樿
===============================================================================
璁粌鏃堕棿: $(date)
椤圭洰鐩綍: $PROJECT_DIR
浠诲姟鏁伴噺: ${#JOB_IDS[@]}
瀹屾垚浠诲姟: $completed_count
澶辫触浠诲姟: $failed_count
Job IDs: ${JOB_IDS[*]}

閰嶇疆鍒楄〃:
$(for config in "${CONFIGS[@]}"; do echo "  - $config"; done)

闅忔満绉嶅瓙:
$(for seed in "${SEEDS[@]}"; do echo "  - $seed"; done)

缁撴灉鐩綍缁撴瀯:
$(ls -lh "$RESULTS_DIR")

===============================================================================
EOF

cat "$RESULTS_DIR/SUMMARY.txt"

# 鎵撳寘缁撴灉
log "鎵撳寘缁撴灉..."
ARCHIVE_NAME="kava_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE_NAME" "$RESULTS_DIR" || log "鈿?鎵撳寘澶辫触"

if [ -f "$ARCHIVE_NAME" ]; then
    log "鉁?缁撴灉宸叉墦鍖? $ARCHIVE_NAME ($(du -h "$ARCHIVE_NAME" | cut -f1))"
fi

# 涓婁紶鍒?HuggingFace (鍙€?
if [ "$UPLOAD_TO_HF" = true ]; then
    log "鍑嗗涓婁紶鍒?HuggingFace..."
    
    # 妫€鏌ユ槸鍚﹀凡鐧诲綍
    if ! huggingface-cli whoami &>/dev/null; then
        log "鈿?鏈櫥褰?HuggingFace锛岃烦杩囦笂浼?
        log "   鎻愮ず: 杩愯 'huggingface-cli login' 鐧诲綍鍚庡啀涓婁紶"
    else
        log "涓婁紶缁撴灉鍒?HuggingFace 浠撳簱: $HF_REPO"
        
        # 鍒涘缓涓婁紶鑴氭湰
        cat > /tmp/upload_results.py << PYTHON_UPLOAD
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
repo_id = "$HF_REPO"

# 鍒涘缓浠撳簱锛堝鏋滀笉瀛樺湪锛?try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"鉁?浠撳簱宸插垱寤烘垨宸插瓨鍦? {repo_id}")
except Exception as e:
    print(f"鈿?鍒涘缓浠撳簱澶辫触: {e}")

# 涓婁紶鏂囦欢
try:
    api.upload_file(
        path_or_fileobj="$ARCHIVE_NAME",
        path_in_repo="$ARCHIVE_NAME",
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"鉁?鏂囦欢涓婁紶鎴愬姛: $ARCHIVE_NAME")
    print(f"   鏌ョ湅: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"鉁?涓婁紶澶辫触: {e}")
PYTHON_UPLOAD
        
        python /tmp/upload_results.py || log "鈿?涓婁紶澶辫触"
        rm -f /tmp/upload_results.py 2>/dev/null || true
    fi
else
    log "璺宠繃涓婁紶锛圲PLOAD_TO_HF=false锛?
fi

# ==================== 瀹屾垚 ====================
log "馃帀 鎵€鏈変换鍔″畬鎴愶紒"

echo ""
echo "================================================================================"
echo "                           浠诲姟鎵ц鎽樿"
echo "================================================================================"
echo "鉁?璧勬簮涓嬭浇: $([ "$SKIP_DOWNLOAD" = false ] && echo "瀹屾垚" || echo "璺宠繃")"
echo "鉁?璧勬簮楠岃瘉: 瀹屾垚"
echo "鉁?閰嶇疆鏇存柊: 瀹屾垚"
echo "鉁?浠诲姟鎻愪氦: ${#JOB_IDS[@]} 涓换鍔?
echo "鉁?璁粌鐩戞帶: 瀹屾垚"
echo "鉁?缁撴灉鏀堕泦: $RESULTS_DIR"
echo "鉁?缁撴灉鎵撳寘: $ARCHIVE_NAME"
echo "鉁?缁撴灉涓婁紶: $([ "$UPLOAD_TO_HF" = true ] && echo "瀹屾垚" || echo "璺宠繃")"
echo ""
echo "缁撴灉浣嶇疆:"
echo "  - 鐩綍: $RESULTS_DIR"
echo "  - 鎵撳寘: $ARCHIVE_NAME"
echo ""
echo "鍚庣画姝ラ:"
echo "  1. 鏌ョ湅缁撴灉鎽樿: cat $RESULTS_DIR/SUMMARY.txt"
echo "  2. 妫€鏌ヨ缁冩棩蹇? ls $RESULTS_DIR/logs/"
echo "  3. 鍒嗘瀽瀹為獙缁撴灉: python analyze_results.py"
echo "  4. 涓嬭浇缁撴灉鍒版湰鍦? scp user@hpc:$PROJECT_DIR/$ARCHIVE_NAME ."
echo ""
echo "濡傞渶閲嶆柊杩愯:"
echo "  bash run_everything.sh"
echo "================================================================================"
