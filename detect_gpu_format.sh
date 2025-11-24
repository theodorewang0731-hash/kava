#!/bin/bash

################################################################################
# 检测正确的 GPU GRES 格式
################################################################################

echo "=== 检测 HPC 的 GPU 配置格式 ==="
echo ""

echo "[1] 检查可用的 GPU 类型"
echo "----------------------------------------"
sinfo -o "%P %G" | grep -v "GRES" | grep "gpu"
echo ""

echo "[2] 检查节点的 GPU 配置"
echo "----------------------------------------"
scontrol show node | grep -E "NodeName|Gres=" | head -20
echo ""

echo "[3] 测试不同的 GPU 请求格式"
echo "----------------------------------------"

# 创建测试脚本模板
cat > /tmp/test_gpu.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:01:00
GRES_PLACEHOLDER

echo "Test"
EOF

# 测试不同的 GPU 格式
declare -a GPU_FORMATS=(
    "#SBATCH --gres=gpu:1"
    "#SBATCH --gres=gpu:a100:1"
    "#SBATCH --gres=gpu:A100:1"
    "#SBATCH --gres=gpu:h100:1"
    "#SBATCH --gres=gpu:H100:1"
    "#SBATCH --gres=gpu:v100:1"
    "#SBATCH --gpus=1"
    "#SBATCH --gpus-per-node=1"
    "#SBATCH --gpus-per-task=1"
)

working_formats=()

for format in "${GPU_FORMATS[@]}"; do
    # 替换占位符
    sed "s|GRES_PLACEHOLDER|$format|" /tmp/test_gpu.sh > /tmp/test_current.sh
    
    # 测试提交
    result=$(sbatch --test-only /tmp/test_current.sh 2>&1)
    
    if echo "$result" | grep -qi "error\|invalid"; then
        echo "✗ $format"
        echo "  错误: $(echo "$result" | grep -i error | head -1)"
    else
        echo "✓ $format - 可用"
        working_formats+=("$format")
    fi
    echo ""
done

rm -f /tmp/test_gpu.sh /tmp/test_current.sh

echo ""
echo "[4] 推荐配置"
echo "----------------------------------------"

if [ ${#working_formats[@]} -gt 0 ]; then
    echo "✓ 找到 ${#working_formats[@]} 个可用的 GPU 配置格式:"
    for format in "${working_formats[@]}"; do
        echo "  $format"
    done
    echo ""
    echo "推荐使用第一个: ${working_formats[0]}"
else
    echo "✗ 未找到可用的 GPU 配置格式"
    echo ""
    echo "建议尝试:"
    echo "  1. 不指定 GPU（让 SLURM 自动分配）"
    echo "  2. 联系 HPC 管理员确认正确的 GPU 请求格式"
fi

echo ""
echo "[5] 完整的推荐 SLURM 配置"
echo "----------------------------------------"

if [ ${#working_formats[@]} -gt 0 ]; then
    cat << SLURM_CONFIG
#!/bin/bash
#SBATCH --job-name=kava-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
${working_formats[0]}
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
SLURM_CONFIG
else
    cat << SLURM_CONFIG
#!/bin/bash
#SBATCH --job-name=kava-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 移除 --gres，让 SLURM 自动分配 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
SLURM_CONFIG
fi

echo ""
echo "=== 检查完成 ==="
echo ""
echo "下一步:"
echo "1. 使用上面推荐的配置更新 submit_multi_seed.slurm"
echo "2. 或者运行: bash submit_all_jobs.sh"
