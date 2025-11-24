#!/bin/bash
# 鍦?HPC 鐧诲綍鑺傜偣涓嬭浇鏁版嵁闆嗭紙浠呮暟鎹泦锛屼笉鍚ā鍨嬶級

echo "======================================================================"
echo "              KAVA 椤圭洰 - 鏁版嵁闆嗕笅杞借剼鏈?                            "
echo "======================================================================"
echo ""
echo "浣跨敤鏂规硶:"
echo "  1. 鐩磋繛涓嬭浇:"
echo "     bash download_datasets_only.sh"
echo ""
echo "  2. 浣跨敤闀滃儚涓嬭浇 (鎺ㄨ崘锛屽浗鍐呮洿蹇?:"
echo "     HF_ENDPOINT=https://hf-mirror.com bash download_datasets_only.sh"
echo ""
echo "======================================================================"
echo ""

# 妫€鏌ユ槸鍚﹀畨瑁呬簡 huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "閿欒: 鏈畨瑁?huggingface_hub"
    echo "璇疯繍琛? pip install huggingface_hub"
    exit 1
fi

# 鍒涘缓涓存椂 Python 鑴氭湰
cat > /tmp/download_datasets_temp.py << 'PYTHON_SCRIPT'
import os
from huggingface_hub import snapshot_download

datasets = [
    ("whynlp/gsm8k-aug", "./datasets/gsm8k-aug"),
    ("whynlp/gsm8k-aug-nl", "./datasets/gsm8k-aug-nl"),
    ("gsm8k", "./datasets/gsm8k")
]

for repo_id, local_dir in datasets:
    print(f"\n{'='*60}")
    print(f"涓嬭浇: {repo_id}")
    print(f"璺緞: {local_dir}")
    print(f"{'='*60}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"鉁?{repo_id} 涓嬭浇瀹屾垚")
    except Exception as e:
        print(f"鉁?{repo_id} 涓嬭浇澶辫触: {e}")

print("\n" + "="*60)
print("鏁版嵁闆嗕笅杞藉畬鎴?")
print("="*60)
PYTHON_SCRIPT

# 鎵ц涓嬭浇
python /tmp/download_datasets_temp.py

# 娓呯悊涓存椂鏂囦欢
rm /tmp/download_datasets_temp.py

echo ""
echo "涓嬩竴姝? 鏇存柊閰嶇疆鏂囦欢涓殑鏁版嵁闆嗚矾寰?
echo "  - configs/llama1b_aug.yaml"
echo "  - configs/llama1b_aug_nl.yaml"
echo "  - configs/llama3b_aug.yaml"
echo "  - configs/qwen05b_aug.yaml"
