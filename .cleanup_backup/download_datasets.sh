#!/bin/bash
#==============================================================================
# 鍦ㄧ櫥褰曡妭鐐归涓嬭浇鏁版嵁闆嗭紙鏈夌綉缁滆闂級
# 鏁版嵁闆嗕細缂撳瓨鍒?~/.cache/huggingface/datasets
#==============================================================================

echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "馃摝 KAVA 鏁版嵁闆嗛涓嬭浇"
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""
echo "璇存槑锛?
echo "  - 璁＄畻鑺傜偣鏃犲缃戣闂?
echo "  - 闇€鍦ㄧ櫥褰曡妭鐐癸紙鏈夌綉缁滐級棰勪笅杞芥暟鎹泦"
echo "  - 鏁版嵁闆嗗皢缂撳瓨鍒?~/.cache/huggingface/datasets"
echo "  - 璁＄畻鑺傜偣杩愯鏃朵細鑷姩浣跨敤缂撳瓨"
echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

cd "/home/rpwang/kava review" || {
    echo "鉂?閿欒: 鏃犳硶杩涘叆椤圭洰鐩綍"
    exit 1
}

# 婵€娲昏櫄鎷熺幆澧?echo "婵€娲昏櫄鎷熺幆澧?.."
source venv/bin/activate

# 璁剧疆缂撳瓨鐩綍
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"

echo "鏁版嵁闆嗙紦瀛樼洰褰? $HF_DATASETS_CACHE"
echo ""

# 涓嬭浇鎵€闇€鐨勬暟鎹泦
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo "寮€濮嬩笅杞芥暟鎹泦..."
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
echo ""

python -c "
import sys
from datasets import load_dataset

datasets_to_download = [
    ('whynlp/gsm8k-aug', 'GSM8K-AUG锛堟柟绋嬪紡 CoT锛?),
    ('whynlp/gsm8k-aug-nl', 'GSM8K-AUG-NL锛堣嚜鐒惰瑷€ CoT锛?),
    ('gsm8k', 'GSM8K锛堣瘎浼帮級'),
]

print('闇€瑕佷笅杞界殑鏁版嵁闆?')
for repo_id, desc in datasets_to_download:
    print(f'  - {repo_id}: {desc}')
print()

success_count = 0
for repo_id, desc in datasets_to_download:
    print('鈹? * 60)
    print(f'銆愪笅杞姐€憑desc}')
    print(f'鏁版嵁闆? {repo_id}')
    print('鈹? * 60)
    
    try:
        # 涓嬭浇鏁版嵁闆嗭紙浼氳嚜鍔ㄧ紦瀛橈級
        dataset = load_dataset(repo_id)
        
        # 鏄剧ず鏁版嵁闆嗕俊鎭?        print(f'鉁?涓嬭浇鎴愬姛')
        print(f'  鍖呭惈 splits: {list(dataset.keys())}')
        
        # 鏄剧ず鏍锋湰鏁伴噺
        for split_name, split_data in dataset.items():
            print(f'  - {split_name}: {len(split_data)} 鏉℃暟鎹?)
        
        success_count += 1
        print()
        
    except Exception as e:
        print(f'鉁?涓嬭浇澶辫触: {e}')
        print()

print('鈹? * 60)
print('涓嬭浇瀹屾垚鎽樿')
print('鈹? * 60)
print(f'鎴愬姛: {success_count}/{len(datasets_to_download)}')

if success_count == len(datasets_to_download):
    print()
    print('鉁?鎵€鏈夋暟鎹泦涓嬭浇鎴愬姛锛?)
    print()
    print('缂撳瓨浣嶇疆: $HOME/.cache/huggingface/datasets')
    print('璁＄畻鑺傜偣杩愯鏃朵細鑷姩浣跨敤杩欎簺缂撳瓨')
    sys.exit(0)
else:
    print()
    print('鈿狅笍  閮ㄥ垎鏁版嵁闆嗕笅杞藉け璐?)
    print('璇锋鏌ョ綉缁滆繛鎺ユ垨鏁版嵁闆嗗悕绉版槸鍚︽纭?)
    sys.exit(1)
"

DOWNLOAD_EXIT=$?

echo ""
echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"

if [ $DOWNLOAD_EXIT -eq 0 ]; then
    echo "鉁?鏁版嵁闆嗕笅杞藉畬鎴?
    echo ""
    echo "鏌ョ湅缂撳瓨:"
    echo "  ls -lh ~/.cache/huggingface/datasets/"
    echo ""
    echo "鐜板湪鍙互鎻愪氦璁粌浠诲姟:"
    echo "  bash submit_all_jobs.sh"
else
    echo "鉂?鏁版嵁闆嗕笅杞藉け璐?
    echo ""
    echo "鏁呴殰鎺掓煡:"
    echo "  1. 妫€鏌ョ櫥褰曡妭鐐规槸鍚︽湁缃戠粶:"
    echo "     ping -c 3 huggingface.co"
    echo ""
    echo "  2. 妫€鏌ユ暟鎹泦鏄惁瀛樺湪:"
    echo "     璁块棶 https://huggingface.co/datasets/whynlp/gsm8k-aug"
    echo ""
    echo "  3. 妫€鏌?datasets 搴撶増鏈?"
    echo "     pip show datasets"
fi

echo "鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣"
