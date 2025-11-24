#!/bin/bash

#==============================================================================
# KAVA HPC 快速启动脚本
# 自动提交所有训练任务并聚合结果
#
# 用法:
#   ./hpc_run_all.sh                          # 运行所有配置
#   ./hpc_run_all.sh llama1b_aug              # 仅运行指定配置
#   ./hpc_run_all.sh llama1b_aug qwen05b_aug  # 运行多个配置
#==============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置列表（如果没有指定，运行所有）
if [ $# -eq 0 ]; then
    CONFIGS=("llama1b_aug" "llama1b_aug_nl" "qwen05b_aug" "llama3b_aug")
else
    CONFIGS=("$@")
fi

echo -e "${CYAN}=========================================="
echo "KAVA HPC 批量任务提交"
echo "==========================================${NC}"
echo "将提交以下配置："
for config in "${CONFIGS[@]}"; do
    echo "  - $config"
done
echo ""

# 创建必要的目录
mkdir -p logs
mkdir -p outputs

# 提交训练任务
echo -e "${CYAN}步骤 1: 提交训练任务${NC}"
echo "-------------------------------------------"

TRAIN_JOB_IDS=()

for config in "${CONFIGS[@]}"; do
    echo -e "${YELLOW}提交配置: $config${NC}"
    
    # 检查配置文件是否存在
    if [ ! -f "configs/${config}.yaml" ]; then
        echo -e "${RED}✗ 配置文件不存在: configs/${config}.yaml${NC}"
        continue
    fi
    
    # 提交 SLURM 数组任务（3 个种子）
    job_output=$(sbatch --export=CONFIG=$config submit_multi_seed.slurm)
    job_id=$(echo $job_output | awk '{print $4}')
    
    if [ -n "$job_id" ]; then
        echo -e "${GREEN}✓ 已提交任务 ID: $job_id${NC}"
        TRAIN_JOB_IDS+=("$job_id")
    else
        echo -e "${RED}✗ 提交失败: $config${NC}"
    fi
done

echo ""
echo -e "${GREEN}共提交 ${#TRAIN_JOB_IDS[@]} 个训练任务${NC}"
echo ""

# 等待所有训练任务完成
echo -e "${CYAN}步骤 2: 等待训练完成${NC}"
echo "-------------------------------------------"
echo "监控任务状态 (Ctrl+C 退出监控但不会取消任务)..."
echo ""

# 显示任务状态
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"

echo ""
echo -e "${YELLOW}提示: 使用以下命令监控任务${NC}"
echo "  watch -n 10 'squeue -u \$USER'"
echo "  tail -f logs/kava_*.out"
echo ""

# 询问是否自动提交聚合任务
read -p "是否在训练完成后自动提交聚合任务? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}步骤 3: 提交聚合任务 (依赖训练完成)${NC}"
    echo "-------------------------------------------"
    
    for config in "${CONFIGS[@]}"; do
        # 找到对应的训练任务 ID
        for job_id in "${TRAIN_JOB_IDS[@]}"; do
            # 提交聚合任务，依赖训练任务完成
            agg_output=$(sbatch --dependency=afterok:$job_id --export=CONFIG=$config submit_aggregate.slurm)
            agg_id=$(echo $agg_output | awk '{print $4}')
            
            if [ -n "$agg_id" ]; then
                echo -e "${GREEN}✓ 聚合任务 $agg_id (等待 $job_id 完成): $config${NC}"
            fi
            break
        done
    done
    
    echo ""
    echo -e "${GREEN}所有任务已提交！${NC}"
else
    echo -e "${YELLOW}跳过自动聚合。训练完成后手动运行:${NC}"
    for config in "${CONFIGS[@]}"; do
        echo "  sbatch --export=CONFIG=$config submit_aggregate.slurm"
    done
fi

echo ""
echo -e "${CYAN}=========================================="
echo "任务摘要"
echo "==========================================${NC}"
echo "训练任务 ID: ${TRAIN_JOB_IDS[@]}"
echo "日志目录: logs/"
echo "输出目录: outputs/"
echo ""
echo -e "${YELLOW}常用命令:${NC}"
echo "  squeue -u \$USER                  # 查看任务状态"
echo "  scancel <JOB_ID>                # 取消任务"
echo "  scontrol show job <JOB_ID>      # 查看任务详情"
echo "  sacct -j <JOB_ID>               # 查看已完成任务信息"
echo ""
echo -e "${GREEN}完成！${NC}"
