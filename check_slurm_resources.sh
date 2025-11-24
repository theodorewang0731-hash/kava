#!/bin/bash

################################################################################
# 妫€鏌?HPC SLURM 璧勬簮閰嶇疆
################################################################################

echo "=== HPC SLURM 璧勬簮閰嶇疆妫€鏌?==="
echo ""

echo "[1] 妫€鏌ュ彲鐢ㄥ垎鍖哄拰璧勬簮闄愬埗"
echo "----------------------------------------"
sinfo -o "%P %.5a %.10l %.6D %.6t %.8c %.8G %.10m %N"
echo ""

echo "[2] 妫€鏌?compute 鍒嗗尯璇︾粏淇℃伅"
echo "----------------------------------------"
scontrol show partition compute 2>/dev/null || scontrol show partition | grep -A 20 "PartitionName=compute"
echo ""

echo "[3] 妫€鏌ュ彲鐢ㄨ妭鐐归厤缃?
echo "----------------------------------------"
sinfo -N -l | head -20
echo ""

echo "[4] 妫€鏌?GPU 鑺傜偣鐨?TRES 閰嶇疆"
echo "----------------------------------------"
scontrol show node | grep -E "NodeName|CPUTot|Gres" | head -30
echo ""

echo "[5] 鎺ㄨ崘鐨勮祫婧愰厤缃?
echo "----------------------------------------"

# 鍒嗘瀽鍙敤璧勬簮
cpus_per_node=$(sinfo -p compute -o "%c" -h | head -1)
mem_per_node=$(sinfo -p compute -o "%m" -h | head -1)
gpu_per_node=$(sinfo -p compute -o "%G" -h | head -1 | grep -oP 'gpu:\d+' | cut -d: -f2)

echo "褰撳墠 compute 鍒嗗尯璧勬簮:"
echo "  姣忚妭鐐?CPU: $cpus_per_node"
echo "  姣忚妭鐐瑰唴瀛? ${mem_per_node}MB"
echo "  姣忚妭鐐?GPU: $gpu_per_node"
echo ""

# 璁＄畻鎺ㄨ崘閰嶇疆
if [ -n "$cpus_per_node" ] && [ "$cpus_per_node" -gt 0 ]; then
    recommended_cpus=$((cpus_per_node / 2))
    if [ "$recommended_cpus" -gt 8 ]; then
        recommended_cpus=8
    elif [ "$recommended_cpus" -lt 2 ]; then
        recommended_cpus=2
    fi
    
    echo "鎺ㄨ崘 SLURM 閰嶇疆:"
    echo "  #SBATCH --cpus-per-task=$recommended_cpus"
    echo "  #SBATCH --mem=32G"
    echo "  #SBATCH --gres=gpu:1"
else
    echo "鎺ㄨ崘浣跨敤鏈€灏忛厤缃?"
    echo "  #SBATCH --cpus-per-task=1"
    echo "  #SBATCH --mem=16G"
    echo "  #SBATCH --gres=gpu:1"
fi

echo ""
echo "[6] 娴嬭瘯浠诲姟鎻愪氦锛堜笉瀹為檯杩愯锛?
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

echo "娴嬭瘯鏈€灏忛厤缃?(1 CPU, 16GB, 1 GPU):"
sbatch --test-only /tmp/test_slurm.sh 2>&1 | head -5

# 娴嬭瘯涓嶅悓 CPU 閰嶇疆
for cpus in 2 4 8; do
    sed -i "s/--cpus-per-task=.*/--cpus-per-task=$cpus/" /tmp/test_slurm.sh
    echo ""
    echo "娴嬭瘯 $cpus CPU 閰嶇疆:"
    result=$(sbatch --test-only /tmp/test_slurm.sh 2>&1)
    if echo "$result" | grep -q "error"; then
        echo "  鉁?$cpus CPU 涓嶅彲鐢?
    else
        echo "  鉁?$cpus CPU 鍙敤"
    fi
done

rm -f /tmp/test_slurm.sh

echo ""
echo "=== 妫€鏌ュ畬鎴?==="
