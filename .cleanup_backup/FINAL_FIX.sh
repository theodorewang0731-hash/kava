#!/bin/bash
#==============================================================================
# 鏈€缁堜慨澶嶆柟妗?- 浣跨敤 HPC 鍏变韩搴撴湰鍦拌矾寰?# 闂锛氬叡浜簱鏄洿鎺ョ洰褰曟牸寮忥紝涓嶆槸 transformers 鏍囧噯缂撳瓨鏍煎紡
# 瑙ｅ喅锛氶厤缃枃浠剁洿鎺ヤ娇鐢ㄧ粷瀵硅矾寰?+ 浠ｇ爜寮哄埗鏈湴鍔犺浇
#==============================================================================

echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "馃敡 KAVA 缃戠粶閿欒鏈€缁堜慨澶嶆柟妗?
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""
echo "鉁?宸插畬鎴愮殑淇锛?
echo "  1. 閰嶇疆鏂囦欢鏀圭敤鏈湴缁濆璺緞"
echo "  2. 浠ｇ爜娣诲姞鏈湴璺緞妫€娴嬪拰寮哄埗绂荤嚎鍔犺浇"
echo "  3. 鎵€鏈?3 涓ā鍨嬪凡纭瀹屾暣瀛樺湪"
echo ""
echo "馃搵 淇敼鐨勬枃浠讹細"
echo "  - configs/llama1b_aug.yaml 鈫?/home/share/models/Llama-3.2-1B-Instruct"
echo "  - configs/llama1b_aug_nl.yaml 鈫?/home/share/models/Llama-3.2-1B-Instruct"
echo "  - configs/llama3b_aug.yaml 鈫?/home/share/models/Llama-3.2-3B-Instruct"
echo "  - configs/qwen05b_aug.yaml 鈫?/home/share/models/Qwen2.5-0.5B-Instruct"
echo "  - src/trainer.py 鈫?娣诲姞鏈湴璺緞寮哄埗绂荤嚎"
echo "  - evaluate.py 鈫?娣诲姞鏈湴璺緞寮哄埗绂荤嚎"
echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

cd "/home/rpwang/kava review" || {
    echo "鉂?閿欒: 鏃犳硶杩涘叆椤圭洰鐩綍"
    exit 1
}

#==============================================================================
# 姝ラ 1: 蹇€熼獙璇侊紙鏈湴娴嬭瘯锛屼笉鎻愪氦 SLURM锛?#==============================================================================
echo "鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo "鈹?姝ラ 1: 蹇€熼獙璇侊紙鎺ㄨ崘锛?                                 鈹?
echo "鈹?鍦ㄧ櫥褰曡妭鐐规祴璇曟ā鍨嬪姞杞斤紙涓嶆彁浜や綔涓氾紝2 鍒嗛挓鍐呭畬鎴愶級        鈹?
echo "鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo ""

read -p "鏄惁杩愯蹇€熼獙璇? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "馃攳 寮€濮嬮獙璇?.."
    source venv/bin/activate
    
    # 娴嬭瘯鍔犺浇鎵€鏈?3 涓ā鍨嬬殑閰嶇疆锛堝揩閫燂紝涓嶅姞杞芥潈閲嶏級
    python -c "
import os
from transformers import AutoConfig

models = [
    ('/home/share/models/Llama-3.2-1B-Instruct', 'Llama 1B'),
    ('/home/share/models/Llama-3.2-3B-Instruct', 'Llama 3B'),
    ('/home/share/models/Qwen2.5-0.5B-Instruct', 'Qwen 0.5B'),
]

print('鈹? * 60)
print('娴嬭瘯浠庢湰鍦拌矾寰勫姞杞芥ā鍨嬮厤缃?)
print('鈹? * 60)
success = 0
for path, name in models:
    print(f'\n銆恵name}銆?)
    print(f'  璺緞: {path}')
    try:
        config = AutoConfig.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True
        )
        print(f'  鉁?鎴愬姛鍔犺浇')
        print(f'    妯″瀷绫诲瀷: {config.model_type}')
        print(f'    闅愯棌灞? {config.hidden_size}')
        success += 1
    except Exception as e:
        print(f'  鉁?澶辫触: {e}')

print('\n' + '鈹? * 60)
if success == 3:
    print('鉁?鎵€鏈夋ā鍨嬮獙璇侀€氳繃锛佸彲浠ユ彁浜よ缁冧换鍔?)
    exit(0)
else:
    print(f'鈿狅笍  閮ㄥ垎妯″瀷楠岃瘉澶辫触 ({success}/3)')
    exit(1)
"
    
    VERIFY_EXIT=$?
    echo ""
    if [ $VERIFY_EXIT -eq 0 ]; then
        echo "鉁?楠岃瘉閫氳繃锛?
    else
        echo "鉂?楠岃瘉澶辫触锛佽妫€鏌ラ敊璇俊鎭?
        echo ""
        echo "甯歌闂鎺掓煡锛?
        echo "  1. 妫€鏌ヨ矾寰勬槸鍚︽纭?"
        echo "     ls -lh /home/share/models/Llama-3.2-1B-Instruct/config.json"
        echo "  2. 妫€鏌ユ潈闄?"
        echo "     ls -ld /home/share/models/"
        echo "  3. 妫€鏌ユ枃浠跺畬鏁存€?"
        echo "     ls -lh /home/share/models/Llama-3.2-1B-Instruct/"
        exit 1
    fi
fi

#==============================================================================
# 姝ラ 2: 鍗曚换鍔℃祴璇曪紙鎻愪氦鍒?SLURM锛?#==============================================================================
echo ""
echo "鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo "鈹?姝ラ 2: 鍗曚换鍔℃祴璇曪紙鎺ㄨ崘锛?                               鈹?
echo "鈹?鎻愪氦 1 涓渶灏忎换鍔″埌 SLURM 楠岃瘉瀹屾暣娴佺▼                    鈹?
echo "鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo ""
echo "灏嗘彁浜? Qwen 0.5B 脳 1 涓瀛愶紙鏈€蹇紝绾?2-4 灏忔椂锛?
echo ""

read -p "鏄惁鎻愪氦鍗曚换鍔℃祴璇? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "馃殌 鎻愪氦鍗曚换鍔℃祴璇?.."
    
    # 娓呯悊鏃ф棩蹇?    rm -f outputs/logs/kava_qwen05b_aug_*.out outputs/logs/kava_qwen05b_aug_*.err 2>/dev/null
    
    JOB_ID=$(sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm 2>&1 | grep -oP '\d+')
    
    if [ -n "$JOB_ID" ]; then
        echo "鉁?浠诲姟宸叉彁浜? Job ID $JOB_ID"
        echo ""
        echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
        echo "馃搳 鐩戞帶浠诲姟"
        echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
        echo ""
        echo "鏌ョ湅闃熷垪:"
        echo "  squeue -j $JOB_ID"
        echo ""
        echo "鏌ョ湅鏃ュ織 (绛夊緟 2-3 鍒嗛挓鍚?:"
        echo "  tail -f outputs/logs/kava_qwen05b_aug_${JOB_ID}_0.out"
        echo "  tail -f outputs/logs/kava_qwen05b_aug_${JOB_ID}_0.err"
        echo ""
        echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
        echo "鉁?鎴愬姛鏍囧織锛堟棩蹇椾腑搴旀樉绀猴級锛?
        echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
        echo ""
        echo "Loading base model..."
        echo "Model: /home/share/models/Qwen2.5-0.5B-Instruct"
        echo "Loading mode: Local path                    鈫?鉁?鍏抽敭"
        echo "Model loaded successfully"
        echo "Training started"
        echo "Epoch 0 | Step 0 | Loss: ..."
        echo ""
        echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
        echo "鉂?涓嶅簲鍑虹幇锛堝鏋滅湅鍒拌鏄庝粛鏈夐棶棰橈級锛?
        echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
        echo ""
        echo "鉁?Network is unreachable"
        echo "鉁?Cannot connect to huggingface.co"
        echo "鉁?We couldn't connect to 'https://huggingface.co'"
        echo ""
        echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
        echo ""
        
        # 绛夊緟鏃ュ織鏂囦欢鍑虹幇骞舵樉绀?        echo "鈴?绛夊緟鏃ュ織鏂囦欢鐢熸垚..."
        for i in {1..60}; do
            LOG_FILE=$(ls -t outputs/logs/kava_qwen05b_aug_${JOB_ID}_*.out 2>/dev/null | head -1)
            if [ -n "$LOG_FILE" ]; then
                echo ""
                echo "馃搫 鎵惧埌鏃ュ織鏂囦欢: $LOG_FILE"
                echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
                echo "鏈€鏂版棩蹇楀唴瀹癸紙瀹炴椂鏇存柊锛屾寜 Ctrl+C 鍋滄锛?"
                echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
                tail -f "$LOG_FILE"
                break
            fi
            echo -n "."
            sleep 2
        done
        
        if [ -z "$LOG_FILE" ]; then
            echo ""
            echo "鈴?鏃ュ織鏂囦欢灏氭湭鐢熸垚锛堜换鍔″彲鑳藉湪鎺掗槦锛?
            echo ""
            echo "鎵嬪姩妫€鏌?"
            echo "  squeue -j $JOB_ID  # 鏌ョ湅浠诲姟鐘舵€?
            echo "  ls -lht outputs/logs/  # 鍒楀嚭鏃ュ織鏂囦欢"
        fi
    else
        echo "鉂?浠诲姟鎻愪氦澶辫触"
        echo ""
        echo "璋冭瘯淇℃伅:"
        sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm
        exit 1
    fi
    
    exit 0
fi

#==============================================================================
# 姝ラ 3: 鎻愪氦鎵€鏈変换鍔?#==============================================================================
echo ""
echo "鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo "鈹?姝ラ 3: 鎻愪氦鎵€鏈変换鍔★紙纭娴嬭瘯閫氳繃鍚庯級                    鈹?
echo "鈹?12 涓换鍔★紝棰勮 36-48 灏忔椂                                 鈹?
echo "鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
echo ""
echo "灏嗘彁浜?"
echo "  - llama1b_aug (3 seeds)"
echo "  - llama1b_aug_nl (3 seeds)"
echo "  - llama3b_aug (3 seeds)"
echo "  - qwen05b_aug (3 seeds)"
echo ""
echo "鈿狅笍  寤鸿: 鍏堝畬鎴愭楠?1 鍜屾楠?2 鐨勯獙璇?
echo ""

read -p "鈿狅笍  纭瑕佹彁浜ゆ墍鏈?12 涓换鍔? (yes/no) " CONFIRM
if [ "$CONFIRM" = "yes" ]; then
    echo "馃殌 鎻愪氦鎵€鏈変换鍔?.."
    bash submit_all_jobs.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "鉁?鎵€鏈変换鍔″凡鎻愪氦锛?
        echo ""
        echo "鐩戞帶鍛戒护:"
        echo "  bash monitor_jobs.sh --auto"
        echo "  squeue --me"
    else
        echo "鉂?浠诲姟鎻愪氦澶辫触"
        exit 1
    fi
else
    echo "鉂?宸插彇娑?
fi

echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "馃摎 淇鎬荤粨"
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""
echo "闂鏍规簮:"
echo "  - HPC 鍏变韩搴撴槸鐩存帴鐩綍鏍煎紡锛堜笉鏄?transformers 鏍囧噯缂撳瓨锛?
echo "  - 浣跨敤 repo ID 浼氬皾璇曡仈缃戯紝瀵艰嚧 Network is unreachable"
echo ""
echo "瑙ｅ喅鏂规:"
echo "  鉁?閰嶇疆鏂囦欢鏀圭敤缁濆璺緞: /home/share/models/Llama-3.2-1B-Instruct"
echo "  鉁?浠ｇ爜妫€娴嬫湰鍦拌矾寰勬椂寮哄埗 local_files_only=True"
echo "  鉁?閬垮厤浠讳綍缃戠粶璁块棶灏濊瘯"
echo ""
echo "楠岃瘉瑕佺偣:"
echo "  鉁?鏃ュ織鏄剧ず 'Loading mode: Local path'"
echo "  鉁?鏃?'Network is unreachable' 閿欒"
echo "  鉁?妯″瀷鍔犺浇鎴愬姛锛岃缁冩甯稿惎鍔?
echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
