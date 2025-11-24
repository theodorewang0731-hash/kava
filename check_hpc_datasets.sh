#!/bin/bash
#==============================================================================
# 妫€鏌?HPC 鍏变韩搴撴槸鍚︽湁鎵€闇€鏁版嵁闆?#==============================================================================

echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "馃摝 妫€鏌?HPC 鏁版嵁闆嗗叡浜簱"
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

# 闇€瑕佺殑鏁版嵁闆?REQUIRED_DATASETS=(
    "whynlp/gsm8k-aug"
    "whynlp/gsm8k-aug-nl"
    "gsm8k"
)

echo "闇€瑕佺殑鏁版嵁闆?"
for dataset in "${REQUIRED_DATASETS[@]}"; do
    echo "  - $dataset"
done
echo ""

echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "銆?銆戞鏌?HPC 鍏变韩鏁版嵁闆嗙洰褰?
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

# 鍙兘鐨勫叡浜暟鎹泦浣嶇疆
SHARED_PATHS=(
    "/home/share/datasets"
    "/home/share/data"
    "/home/share/huggingface/datasets"
    "/home/share/.cache/huggingface/datasets"
    "/datasets"
    "/data"
)

echo "鎼滅储鍙兘鐨勫叡浜暟鎹泦鐩綍..."
for path in "${SHARED_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "鉁?鎵惧埌鐩綍: $path"
        echo "  鍐呭锛堝墠 20 椤癸級:"
        ls -lh "$path" | head -20
        echo ""
    fi
done

echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "銆?銆戞悳绱?GSM8K 鐩稿叧鏁版嵁闆?
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

# 鍦?/home/share 涓嬫悳绱?gsm8k 鐩稿叧
if [ -d "/home/share" ]; then
    echo "鎼滅储 /home/share 涓嬬殑 gsm8k 鐩稿叧鍐呭..."
    find /home/share -iname "*gsm8k*" -type d 2>/dev/null | head -20
    
    echo ""
    echo "鎼滅储 /home/share 涓嬬殑 datasets 鐩綍..."
    find /home/share -iname "*dataset*" -type d -maxdepth 3 2>/dev/null | head -20
fi

echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "銆?銆戞鏌ュ父瑙?HuggingFace 缂撳瓨浣嶇疆"
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

# 妫€鏌ユ槸鍚︽湁鍏朵粬鐢ㄦ埛鐨勭紦瀛樺彲浠ュ叡浜?CACHE_PATHS=(
    "/home/share/.cache/huggingface/datasets"
    "$HOME/.cache/huggingface/datasets"
    "/scratch/.cache/huggingface/datasets"
)

for cache_path in "${CACHE_PATHS[@]}"; do
    if [ -d "$cache_path" ]; then
        echo "鉁?鎵惧埌缂撳瓨鐩綍: $cache_path"
        echo "  鍐呭:"
        ls -lh "$cache_path" | head -10
        echo ""
    else
        echo "鉁?鏈壘鍒? $cache_path"
    fi
done

echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "銆?銆戞鏌ョ鐞嗗憳鎻愪緵鐨勮鏄庢枃妗?
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

DOC_PATHS=(
    "/home/share/README"
    "/home/share/README.md"
    "/home/share/DATASETS.md"
    "/home/share/datasets/README"
)

for doc in "${DOC_PATHS[@]}"; do
    if [ -f "$doc" ]; then
        echo "鉁?鎵惧埌鏂囨。: $doc"
        echo "  鍐呭:"
        cat "$doc"
        echo ""
    fi
done

echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "銆?銆戝缓璁殑鏁版嵁闆嗚В鍐虫柟妗?
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""
echo "鏂规 1: 濡傛灉鎵惧埌鍏变韩鏁版嵁闆?鈫?閰嶇疆璺緞浣跨敤鍏变韩鏁版嵁闆?
echo "鏂规 2: 濡傛灉娌℃湁鍏变韩鏁版嵁闆?鈫?鑱旂郴绠＄悊鍛樻坊鍔犳垨鐢宠澶栫綉璁块棶"
echo "鏂规 3: 濡傛灉鏈変唬鐞嗚妭鐐?鈫?鍦ㄦ湁缃戠粶鐨勮妭鐐逛笅杞藉悗浼犺緭"
echo ""
echo "璇锋妸浠ヤ笂杈撳嚭鍙戠粰鎴戯紝鎴戜細鏍规嵁缁撴灉鎻愪緵鍏蜂綋瑙ｅ喅鏂规"
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
