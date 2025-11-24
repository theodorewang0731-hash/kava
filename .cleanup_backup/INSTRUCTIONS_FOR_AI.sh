#!/bin/bash
#==============================================================================
# 缁?HPC AI 鍔╂墜鐨勬搷浣滄寚浠?# 闂锛氫换鍔″洜 "Network is unreachable" 蹇€熷け璐?# 瑙ｅ喅锛氬凡淇敼閰嶇疆鏂囦欢浣跨敤鏈湴璺緞 + 寮哄埗绂荤嚎妯″紡
#==============================================================================

echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "馃敡 KAVA 缃戠粶閿欒淇鏂规"
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""
echo "馃搵 闂璇婃柇锛?
echo "  - 鎵€鏈変换鍔″揩閫熷け璐ワ紙ExitCode 1:0 鎴?2:0锛?
echo "  - 鏃ュ織閿欒锛歂etwork is unreachable [Errno 101]"
echo "  - transformers 灏濊瘯璁块棶 huggingface.co 浣嗚妭鐐规棤澶栫綉"
echo ""
echo "鉁?宸插疄鏂界殑淇锛?
echo "  1. 閰嶇疆鏂囦欢鏀圭敤鏈湴璺緞锛?home/share/models/...锛?
echo "  2. SLURM 鑴氭湰娣诲姞绂荤嚎妯″紡锛圚UGGINGFACE_HUB_OFFLINE=1锛?
echo "  3. 鍒涘缓璇婃柇鑴氭湰楠岃瘉妯″瀷鍔犺浇"
echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

# 杩涘叆椤圭洰鐩綍
cd "/home/rpwang/kava review" || {
    echo "鉂?閿欒: 鏃犳硶杩涘叆椤圭洰鐩綍"
    exit 1
}

echo "馃搨 褰撳墠鐩綍: $(pwd)"
echo ""

#==============================================================================
# 閫夐」 A: 蹇€熼獙璇侊紙鎺ㄨ崘鍏堝仛锛屼笉鎻愪氦浣滀笟锛?#==============================================================================
echo "鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo "鈹?閫夐」 A: 蹇€熼獙璇侊紙鎺ㄨ崘锛?                                 鈹?
echo "鈹?娴嬭瘯妯″瀷鏄惁鑳戒粠鏈湴璺緞鍔犺浇锛堜笉鎻愪氦 SLURM 浣滀笟锛?        鈹?
echo "鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo ""
echo "鎵ц鍛戒护锛?
echo "  source venv/bin/activate"
echo "  export HF_HOME=/home/share/models"
echo "  export TRANSFORMERS_CACHE=/home/share/models"
echo "  export HUGGINGFACE_HUB_OFFLINE=1"
echo "  export TRANSFORMERS_OFFLINE=1"
echo "  python quick_model_test.py"
echo ""
echo "棰勬湡缁撴灉锛?
echo "  鉁?鎵€鏈?3 涓ā鍨嬮兘鑳戒粠鏈湴璺緞鍔犺浇"
echo "  鉁?鏃犵綉缁滆闂皾璇?
echo "  鉁?鏈€鍚庢樉绀?'鎺ㄨ崘鏂规: 鍦ㄩ厤缃枃浠朵腑浣跨敤鏈湴璺緞'"
echo ""

read -p "鏄惁绔嬪嵆杩愯楠岃瘉娴嬭瘯? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "馃攳 寮€濮嬮獙璇?.."
    source venv/bin/activate
    export HF_HOME=/home/share/models
    export TRANSFORMERS_CACHE=/home/share/models
    export HUGGINGFACE_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    python quick_model_test.py
    
    VERIFY_EXIT=$?
    echo ""
    if [ $VERIFY_EXIT -eq 0 ]; then
        echo "鉁?楠岃瘉閫氳繃锛佸彲浠ョ户缁彁浜や换鍔?
        echo ""
    else
        echo "鉂?楠岃瘉澶辫触锛佽妫€鏌ラ敊璇俊鎭?
        echo "   璇︾粏淇鎸囧崡: cat FIX_NETWORK_ERROR.md"
        exit 1
    fi
fi

#==============================================================================
# 閫夐」 B: 鍗曚换鍔℃祴璇曪紙鎺ㄨ崘鍦ㄥ叏閲忔彁浜ゅ墠锛?#==============================================================================
echo ""
echo "鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo "鈹?閫夐」 B: 鍗曚换鍔℃祴璇曪紙鎺ㄨ崘锛?                               鈹?
echo "鈹?鎻愪氦 1 涓渶灏忎换鍔￠獙璇?SLURM 鐜锛圦wen 0.5B锛屾渶蹇級       鈹?
echo "鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo ""
echo "鎵ц鍛戒护锛?
echo "  sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm"
echo ""
echo "楠岃瘉鏂规硶锛堢瓑寰?2-3 鍒嗛挓鍚庯級锛?
echo "  tail -n 50 outputs/logs/kava_qwen05b_aug_*.out"
echo "  tail -n 50 outputs/logs/kava_qwen05b_aug_*.err"
echo ""
echo "鎴愬姛鏍囧織锛?
echo "  鉁?鐪嬪埌 'Loading model from /home/share/models/Qwen2.5-0.5B-Instruct'"
echo "  鉁?鐪嬪埌 'Model loaded successfully'"
echo "  鉁?鐪嬪埌 'Epoch 0 | Step 0 | Loss: ...'"
echo "  鉁?涓嶅簲鐪嬪埌 'Network is unreachable'"
echo ""

read -p "鏄惁鎻愪氦鍗曚换鍔℃祴璇? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "馃殌 鎻愪氦鍗曚换鍔℃祴璇?.."
    JOB_ID=$(sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm | grep -oP '\d+')
    echo "鉁?浠诲姟宸叉彁浜? Job ID $JOB_ID"
    echo ""
    echo "鐩戞帶鍛戒护:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f outputs/logs/kava_qwen05b_aug_${JOB_ID}_0.out"
    echo ""
    echo "绛夊緟 2-3 鍒嗛挓鍚庢鏌ユ棩蹇楋紙鎸?Ctrl+C 鍋滄鏌ョ湅锛?
    sleep 5
    
    # 绛夊緟鏃ュ織鏂囦欢鍑虹幇
    LOG_FILE=""
    for i in {1..30}; do
        LOG_FILE=$(ls -t outputs/logs/kava_qwen05b_aug_${JOB_ID}_*.out 2>/dev/null | head -1)
        if [ -n "$LOG_FILE" ]; then
            break
        fi
        sleep 2
    done
    
    if [ -n "$LOG_FILE" ]; then
        echo "馃搫 鏃ュ織鏂囦欢: $LOG_FILE"
        tail -f "$LOG_FILE"
    else
        echo "鈴?鏃ュ織鏂囦欢灏氭湭鐢熸垚锛岃鎵嬪姩妫€鏌?
        echo "   鍛戒护: ls -lht outputs/logs/"
    fi
    
    exit 0
fi

#==============================================================================
# 閫夐」 C: 鐩存帴鎻愪氦鎵€鏈変换鍔★紙闇€瑕佺‘璁ら獙璇侀€氳繃锛?#==============================================================================
echo ""
echo "鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo "鈹?閫夐」 C: 鎻愪氦鎵€鏈変换鍔★紙12 涓换鍔★級                         鈹?
echo "鈹?鈿狅笍  寤鸿鍏堝畬鎴愰€夐」 A 鎴?B 鐨勯獙璇?                          鈹?
echo "鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo ""
echo "鎵ц鍛戒护:"
echo "  bash submit_all_jobs.sh"
echo ""
echo "灏嗘彁浜?"
echo "  - llama1b_aug (3 seeds)"
echo "  - llama1b_aug_nl (3 seeds)"
echo "  - llama3b_aug (3 seeds)"
echo "  - qwen05b_aug (3 seeds)"
echo "  鎬昏: 4 涓讳换鍔?脳 3 绉嶅瓙 = 12 涓瓙浠诲姟"
echo ""
echo "棰勮鏃堕棿: 36-48 灏忔椂"
echo ""

read -p "鈿狅笍  纭瑕佹彁浜ゆ墍鏈?12 涓换鍔? (yes/no) " CONFIRM
if [ "$CONFIRM" = "yes" ]; then
    echo "馃殌 鎻愪氦鎵€鏈変换鍔?.."
    bash submit_all_jobs.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "鉁?鎵€鏈変换鍔″凡鎻愪氦鎴愬姛锛?
        echo ""
        echo "鐩戞帶鍛戒护:"
        echo "  bash monitor_jobs.sh --auto    # 姣?30 绉掕嚜鍔ㄥ埛鏂?
        echo "  squeue --me                    # 鎵嬪姩鏌ョ湅闃熷垪"
        echo ""
        echo "鏌ョ湅鏃ュ織:"
        echo "  ls -lht outputs/logs/          # 鍒楀嚭鏈€鏂版棩蹇?
        echo "  tail -f outputs/logs/kava_*.out  # 瀹炴椂鏌ョ湅"
        echo ""
    else
        echo "鉂?浠诲姟鎻愪氦澶辫触锛岃妫€鏌ラ敊璇俊鎭?
        exit 1
    fi
else
    echo "鉂?宸插彇娑堟彁浜?
    echo "   寤鸿: 鍏堣繍琛岄€夐」 A 鎴?B 楠岃瘉淇鏁堟灉"
fi

echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "馃摎 瀹屾暣淇鏂囨。: cat FIX_NETWORK_ERROR.md"
echo "馃悰 闂鎺掓煡: python quick_model_test.py"
echo "馃搳 鐩戞帶浠诲姟: bash monitor_jobs.sh --auto"
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
