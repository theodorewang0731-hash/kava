#!/bin/bash

################################################################################
# 检查 HPC SLURM 资源配置
################################################################################

echo "=== HPC SLURM 资源配置检查 ==="
echo ""

echo "[1] 检查可用分区和资源限制"
echo "----------------------------------------"
sinfo -o "%P %.5a %.10l %.6D %.6t %.8c %.8G %.10m %N"
echo ""

echo "[2] 检查 compute 分区详细信息"
echo "----------------------------------------"
scontrol show partition compute 2>/dev/null || scontrol show partition | grep -A 20 "PartitionName=compute"
echo ""

echo "[3] 检查可用节点配置"
echo "----------------------------------------"
sinfo -N -l | head -20
echo ""

echo "[4] 检查 GPU 节点的 TRES 配置"
echo "----------------------------------------"
scontrol show node | grep -E "NodeName|CPUTot|Gres" | head -30
echo ""

echo "[5] 推荐的资源配置"
echo "----------------------------------------"

# 分析可用资源
cpus_per_node=$(sinfo -p compute -o "%c" -h | head -1)
mem_per_node=$(sinfo -p compute -o "%m" -h | head -1)
gpu_per_node=$(sinfo -p compute -o "%G" -h | head -1 | grep -oP 'gpu:\d+' | cut -d: -f2)

echo "当前 compute 分区资源:"
echo "  每节点 CPU: $cpus_per_node"
echo "  每节点内存: ${mem_per_node}MB"
echo "  每节点 GPU: $gpu_per_node"
echo ""

# 计算推荐配置
if [ -n "$cpus_per_node" ] && [ "$cpus_per_node" -gt 0 ]; then
    recommended_cpus=$((cpus_per_node / 2))
    if [ "$recommended_cpus" -gt 8 ]; then
        recommended_cpus=8
    elif [ "$recommended_cpus" -lt 2 ]; then
        recommended_cpus=2
    fi
    
    echo "推荐 SLURM 配置:"
    echo "  #SBATCH --cpus-per-task=$recommended_cpus"
    echo "  #SBATCH --mem=32G"
    echo "  #SBATCH --gres=gpu:1"
else
    echo "推荐使用最小配置:"
    echo "  #SBATCH --cpus-per-task=1"
    echo "  #SBATCH --mem=16G"
    echo "  #SBATCH --gres=gpu:1"
fi

echo ""
echo "[6] 测试任务提交（不实际运行）"
echo "----------------------------------------"

cat > /tmp/test_slurm.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:01:00

echo "Test job"
EOF

echo "测试最小配置 (1 CPU, 16GB, 1 GPU):"
sbatch --test-only /tmp/test_slurm.sh 2>&1 | head -5

# 测试不同 CPU 配置
for cpus in 2 4 8; do
    sed -i "s/--cpus-per-task=.*/--cpus-per-task=$cpus/" /tmp/test_slurm.sh
    echo ""
    echo "测试 $cpus CPU 配置:"
    result=$(sbatch --test-only /tmp/test_slurm.sh 2>&1)
    if echo "$result" | grep -q "error"; then
        echo "  ✗ $cpus CPU 不可用"
    else
        echo "  ✓ $cpus CPU 可用"
    fi
done

rm -f /tmp/test_slurm.sh

echo ""
echo "=== 检查完成 ==="
