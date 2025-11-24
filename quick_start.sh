#!/bin/bash
################################################################################
# KAVA 蹇€熷惎鍔ㄨ剼鏈?- 鏈€绠€鍗曠殑浣跨敤鏂瑰紡
# 鐢ㄦ硶: bash quick_start.sh
################################################################################

echo "
鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
鈺?                         KAVA 椤圭洰蹇€熷惎鍔?                                鈺?鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆
"

# 榛樿閰嶇疆
USE_MIRROR=true
SKIP_DOWNLOAD=false
AUTO_UPLOAD=false

# 瑙ｆ瀽鍛戒护琛屽弬鏁?while [[ $# -gt 0 ]]; do
    case $1 in
        --no-mirror)
            USE_MIRROR=false
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --upload)
            AUTO_UPLOAD=true
            shift
            ;;
        --help|-h)
            echo "鐢ㄦ硶: bash quick_start.sh [閫夐」]"
            echo ""
            echo "閫夐」:"
            echo "  --no-mirror       涓嶄娇鐢?HF-Mirror 闀滃儚锛堝浗澶栨湇鍔″櫒浣跨敤锛?
            echo "  --skip-download   璺宠繃涓嬭浇姝ラ锛堝鏋滆祫婧愬凡瀛樺湪锛?
            echo "  --upload          鑷姩涓婁紶缁撴灉鍒?HuggingFace"
            echo "  --help, -h        鏄剧ず姝ゅ府鍔╀俊鎭?
            echo ""
            echo "绀轰緥:"
            echo "  bash quick_start.sh                    # 鏍囧噯杩愯"
            echo "  bash quick_start.sh --skip-download    # 璺宠繃涓嬭浇"
            echo "  bash quick_start.sh --upload           # 璁粌鍚庝笂浼犵粨鏋?
            exit 0
            ;;
        *)
            echo "鏈煡閫夐」: $1"
            echo "浣跨敤 --help 鏌ョ湅甯姪"
            exit 1
            ;;
    esac
done

# 鏄剧ず閰嶇疆
echo "褰撳墠閰嶇疆:"
echo "  - 浣跨敤 HF-Mirror: $USE_MIRROR"
echo "  - 璺宠繃涓嬭浇: $SKIP_DOWNLOAD"
echo "  - 鑷姩涓婁紶: $AUTO_UPLOAD"
echo ""

# 妫€鏌ユ槸鍚﹀湪姝ｇ‘鐨勭洰褰?if [ ! -f "run_everything.sh" ]; then
    echo "閿欒: 璇峰湪椤圭洰鏍圭洰褰曡繍琛屾鑴氭湰"
    exit 1
fi

# 璁剧疆鐜鍙橀噺
export USE_HF_MIRROR=$USE_MIRROR
export SKIP_DOWNLOAD=$SKIP_DOWNLOAD
export UPLOAD_TO_HF=$AUTO_UPLOAD

# 杩愯涓昏剼鏈?echo "鍚姩涓昏剼鏈?.."
echo ""
bash run_everything.sh

echo ""
echo "鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽"
echo "鈺?                          浠诲姟鍏ㄩ儴瀹屾垚锛?                                  鈺?
echo "鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆"
