#!/bin/bash
################################################################################
# HPC 鐜閰嶇疆鑴氭湰锛堣嚜鍔ㄥ姞杞芥ā鍧楋級
# 鐢ㄦ硶: source setup_hpc_env.sh  锛堟敞鎰忥細蹇呴』浣跨敤 source 鎴?.锛?################################################################################

echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  KAVA HPC 鐜閰嶇疆"
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo ""

# 鍒濆鍖?module 绯荤粺
if [ -f /usr/share/modules/init/bash ]; then
    echo "[1/5] 鍒濆鍖?module 绯荤粺..."
    . /usr/share/modules/init/bash
    module use --append /home/share/modules/modulefiles
    echo "鉁?Module 绯荤粺宸插垵濮嬪寲"
else
    echo "鈿?鏈壘鍒?module 绯荤粺锛屽彲鑳戒笉鍦?HPC 鐜涓?
fi

# 鍔犺浇 anaconda3
echo ""
echo "[2/5] 鍔犺浇 Anaconda..."
if command -v module &> /dev/null; then
    module load anaconda3
    if [ $? -eq 0 ]; then
        echo "鉁?Anaconda 宸插姞杞?
    else
        echo "鈿?Anaconda 鍔犺浇澶辫触锛屽皾璇曠洿鎺ユ煡鎵?conda..."
    fi
fi

# 楠岃瘉 conda
echo ""
echo "[3/5] 楠岃瘉 conda..."
if command -v conda &> /dev/null; then
    echo "鉁?Conda 鍙敤: $(conda --version)"
else
    echo "鉁?Conda 浠嶄笉鍙敤"
    echo ""
    echo "鍙兘鐨勮В鍐虫柟妗堬細"
    echo "  1. 妫€鏌?module 鏄惁鍙敤: module avail"
    echo "  2. 鎵嬪姩鍔犺浇: module load anaconda3"
    echo "  3. 鎴栦娇鐢ㄧ郴缁?Python: python3 -m venv venv"
    exit 1
fi

# 鍒涘缓鎴栨縺娲?conda 鐜
echo ""
echo "[4/5] 閰嶇疆 KAVA 鐜..."

if conda env list | grep -q "^kava "; then
    echo "鈿?鐜 'kava' 宸插瓨鍦紝婵€娲讳腑..."
    conda activate kava
else
    echo "鍒涘缓鏂扮幆澧?'kava'..."
    conda create -n kava python=3.10 -y
    conda activate kava
fi

# 瀹夎渚濊禆
echo ""
echo "[5/5] 瀹夎渚濊禆..."
pip install -r requirements.txt

# 瀹屾垚
echo ""
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo "  閰嶇疆瀹屾垚锛?
echo "鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲"
echo ""
echo "鐜宸叉縺娲汇€傛帴涓嬫潵浣犲彲浠ワ細"
echo ""
echo "1. 鎻愪氦璁粌浠诲姟锛?
echo "   sbatch submit_multi_seed.slurm"
echo ""
echo "2. 鎴栦氦浜掑紡杩愯锛?
echo "   python train.py --config configs/llama1b_aug.yaml"
echo ""
echo "娉ㄦ剰: 姣忔鐧诲綍閮介渶瑕侀噸鏂版縺娲荤幆澧冿細"
echo "       conda activate kava"
echo ""
