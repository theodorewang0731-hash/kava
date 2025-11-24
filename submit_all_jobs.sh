#!/bin/bash

################################################################################
# KAVA 一键启动脚本 - 使用 HPC 共享模型库（无需下载）
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  KAVA 训练任务提交 - 使用 HPC 共享模型库${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# =============================================================================
# 1. 验证共享模型库
# =============================================================================

echo -e "${BLUE}[1/4] 验证 HPC 共享模型库${NC}"
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
        echo -e "${GREEN}✓${NC} $model"
    else
        echo -e "${YELLOW}✗${NC} $model (未找到)"
        all_models_found=false
    fi
done

if [ "$all_models_found" = false ]; then
    echo ""
    echo -e "${YELLOW}警告: 部分模型未找到，但将继续执行${NC}"
    echo "如果训练失败，请检查模型路径是否正确"
fi

echo ""

# =============================================================================
# 2. 验证环境
# =============================================================================

echo -e "${BLUE}[2/4] 验证 Python 环境${NC}"
echo "----------------------------------------"

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}✗ 虚拟环境不存在${NC}"
    echo "请先运行: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate
echo -e "${GREEN}✓${NC} 虚拟环境已激活"
echo "  Python: $(python --version)"
echo ""

# =============================================================================
# 3. 准备目录
# =============================================================================

echo -e "${BLUE}[3/4] 准备输出目录${NC}"
echo "----------------------------------------"

mkdir -p outputs/logs
mkdir -p logs
echo -e "${GREEN}✓${NC} 输出目录已创建"
echo ""

# =============================================================================
# 4. 提交 SLURM 任务
# =============================================================================

echo -e "${BLUE}[4/4] 提交 SLURM 训练任务${NC}"
echo "----------------------------------------"

CONFIGS=(
    "llama1b_aug"
    "llama1b_aug_nl"
    "llama3b_aug"
    "qwen05b_aug"
)

job_ids=()
total_jobs=$((${#CONFIGS[@]} * 3))  # 4 configs × 3 seeds
job_count=0

echo "提交 $total_jobs 个任务 (${#CONFIGS[@]} 配置 × 3 种子)..."
echo ""

for config in "${CONFIGS[@]}"; do
    echo "配置: $config"
    
    # 提交任务（使用 array job，3 个种子）
    job_id=$(sbatch \
        --job-name="kava_${config}" \
        --output="outputs/logs/kava_${config}_%A_%a.out" \
        --error="outputs/logs/kava_${config}_%A_%a.err" \
        --export=ALL,CONFIG="$config" \
        submit_multi_seed.slurm | awk '{print $4}')
    
    if [ -n "$job_id" ]; then
        job_ids+=("$job_id")
        echo -e "  ${GREEN}✓${NC} 任务已提交: $job_id (3 个子任务)"
        job_count=$((job_count + 3))
    else
        echo -e "  ${YELLOW}✗${NC} 任务提交失败"
    fi
done

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  任务提交完成！${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# 保存任务 ID
printf "%s\n" "${job_ids[@]}" > outputs/job_ids.txt
echo "已提交任务: ${#job_ids[@]} 个主任务 (共 $job_count 个子任务)"
echo "任务 ID 已保存到: outputs/job_ids.txt"
echo ""

# =============================================================================
# 生成监控脚本
# =============================================================================

echo "生成监控脚本..."

# monitor_jobs.sh
cat > monitor_jobs.sh << 'MONITOR_EOF'
#!/bin/bash
echo "=== KAVA 训练任务状态 ==="
echo ""
squeue -u $USER --format="%.18i %.12j %.8T %.10M %.6D %.15R" | grep -E "JOBID|kava"
echo ""
echo "运行中: $(squeue -u $USER | grep -c kava || echo 0)"
echo "总任务: $(cat outputs/job_ids.txt 2>/dev/null | wc -l || echo 0) 个主任务"
MONITOR_EOF

chmod +x monitor_jobs.sh
echo -e "${GREEN}✓${NC} 已创建 monitor_jobs.sh"

# collect_results.sh
cat > collect_results.sh << 'COLLECT_EOF'
#!/bin/bash
echo "=== 收集 KAVA 训练结果 ==="
echo ""

results_file="outputs/aggregated_results.csv"
echo "Config,Seed,EM,F1,Status" > "$results_file"

for log_file in outputs/logs/kava_*.out; do
    if [ -f "$log_file" ]; then
        # 提取配置名和种子
        base=$(basename "$log_file" .out)
        config=$(echo "$base" | sed 's/kava_\(.*\)_[0-9]*_[0-9]*/\1/')
        
        # 从文件中提取种子（如果有的话）
        seed=$(grep "Seed:" "$log_file" | head -1 | awk '{print $2}')
        
        if grep -q "Final Test EM" "$log_file"; then
            em=$(grep "Final Test EM" "$log_file" | tail -1 | awk '{print $NF}')
            f1=$(grep "Final Test F1" "$log_file" | tail -1 | awk '{print $NF}')
            echo "$config,$seed,$em,$f1,COMPLETED" >> "$results_file"
        else
            echo "$config,$seed,N/A,N/A,IN_PROGRESS" >> "$results_file"
        fi
    fi
done

echo "结果已保存到: $results_file"
echo ""
cat "$results_file" | column -t -s,
COLLECT_EOF

chmod +x collect_results.sh
echo -e "${GREEN}✓${NC} 已创建 collect_results.sh"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  下一步操作${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "1. 检查任务状态:"
echo "   bash monitor_jobs.sh"
echo "   或: squeue -u \$USER"
echo ""
echo "2. 查看实时日志:"
echo "   tail -f outputs/logs/kava_*.out"
echo ""
echo "3. 取消任务 (如需要):"
echo "   scancel <job_id>"
echo "   或取消所有: scancel -u \$USER"
echo ""
echo "4. 训练完成后收集结果 (36-48 小时后):"
echo "   bash collect_results.sh"
echo ""
echo -e "${GREEN}✓ 所有任务已提交到 SLURM 队列！${NC}"
echo -e "${YELLOW}⏱ 预计完成时间: 36-48 小时${NC}"
echo ""
