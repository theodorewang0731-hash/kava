#!/bin/bash

################################################################################
# 妫€娴嬫纭殑 GPU GRES 鏍煎紡
################################################################################

echo "=== 妫€娴?HPC 鐨?GPU 閰嶇疆鏍煎紡 ==="
echo ""

echo "[1] 妫€鏌ュ彲鐢ㄧ殑 GPU 绫诲瀷"
echo "----------------------------------------"
sinfo -o "%P %G" | grep -v "GRES" | grep "gpu"
echo ""

echo "[2] 妫€鏌ヨ妭鐐圭殑 GPU 閰嶇疆"
echo "----------------------------------------"
scontrol show node | grep -E "NodeName|Gres=" | head -20
echo ""

echo "[3] 娴嬭瘯涓嶅悓鐨?GPU 璇锋眰鏍煎紡"
echo "----------------------------------------"

# 鍒涘缓娴嬭瘯鑴氭湰妯℃澘
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

# 娴嬭瘯涓嶅悓鐨?GPU 鏍煎紡
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
    # 鏇挎崲鍗犱綅绗?    sed "s|GRES_PLACEHOLDER|$format|" /tmp/test_gpu.sh > /tmp/test_current.sh
    
    # 娴嬭瘯鎻愪氦
    result=$(sbatch --test-only /tmp/test_current.sh 2>&1)
    
    if echo "$result" | grep -qi "error\|invalid"; then
        echo "鉁?$format"
        echo "  閿欒: $(echo "$result" | grep -i error | head -1)"
    else
        echo "鉁?$format - 鍙敤"
        working_formats+=("$format")
    fi
    echo ""
done

rm -f /tmp/test_gpu.sh /tmp/test_current.sh

echo ""
echo "[4] 鎺ㄨ崘閰嶇疆"
echo "----------------------------------------"

if [ ${#working_formats[@]} -gt 0 ]; then
    echo "鉁?鎵惧埌 ${#working_formats[@]} 涓彲鐢ㄧ殑 GPU 閰嶇疆鏍煎紡:"
    for format in "${working_formats[@]}"; do
        echo "  $format"
    done
    echo ""
    echo "鎺ㄨ崘浣跨敤绗竴涓? ${working_formats[0]}"
else
    echo "鉁?鏈壘鍒板彲鐢ㄧ殑 GPU 閰嶇疆鏍煎紡"
    echo ""
    echo "寤鸿灏濊瘯:"
    echo "  1. 涓嶆寚瀹?GPU锛堣 SLURM 鑷姩鍒嗛厤锛?
    echo "  2. 鑱旂郴 HPC 绠＄悊鍛樼‘璁ゆ纭殑 GPU 璇锋眰鏍煎紡"
fi

echo ""
echo "[5] 瀹屾暣鐨勬帹鑽?SLURM 閰嶇疆"
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
# 绉婚櫎 --gres锛岃 SLURM 鑷姩鍒嗛厤 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
SLURM_CONFIG
fi

echo ""
echo "=== 妫€鏌ュ畬鎴?==="
echo ""
echo "涓嬩竴姝?"
echo "1. 浣跨敤涓婇潰鎺ㄨ崘鐨勯厤缃洿鏂?submit_multi_seed.slurm"
echo "2. 鎴栬€呰繍琛? bash submit_all_jobs.sh"
