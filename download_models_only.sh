#!/bin/bash
# 鍦?HPC 鐧诲綍鑺傜偣涓嬭浇妯″瀷锛堜粎妯″瀷锛屼笉鍚暟鎹泦锛?
echo "======================================================================"
echo "               KAVA 椤圭洰 - 妯″瀷涓嬭浇鑴氭湰                              "
echo "======================================================================"
echo ""
echo "浣跨敤鏂规硶:"
echo "  1. 鐩磋繛涓嬭浇:"
echo "     bash download_models_only.sh"
echo ""
echo "  2. 浣跨敤闀滃儚涓嬭浇 (鎺ㄨ崘锛屽浗鍐呮洿蹇?:"
echo "     HF_ENDPOINT=https://hf-mirror.com bash download_models_only.sh"
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
cat > /tmp/download_models_temp.py << 'PYTHON_SCRIPT'
import os
from huggingface_hub import snapshot_download

models = [
    ("meta-llama/Llama-3.2-1B-Instruct", "./models/Llama-3.2-1B-Instruct"),
    ("meta-llama/Llama-3.2-3B-Instruct", "./models/Llama-3.2-3B-Instruct"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "./models/Qwen2.5-0.5B-Instruct")
]

for repo_id, local_dir in models:
    print(f"\n{'='*60}")
    print(f"涓嬭浇: {repo_id}")
    print(f"璺緞: {local_dir}")
    print(f"{'='*60}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"鉁?{repo_id} 涓嬭浇瀹屾垚")
    except Exception as e:
        print(f"鉁?{repo_id} 涓嬭浇澶辫触: {e}")

print("\n" + "="*60)
print("妯″瀷涓嬭浇瀹屾垚!")
print("="*60)
PYTHON_SCRIPT

# 鎵ц涓嬭浇
python /tmp/download_models_temp.py

# 娓呯悊涓存椂鏂囦欢
rm /tmp/download_models_temp.py

echo ""
echo "鎻愮ず: 濡傛灉 LLaMA 妯″瀷涓嬭浇澶辫触锛岃纭繚:"
echo "  1. 宸插湪 HuggingFace 缃戠珯鐢宠璁块棶鏉冮檺"
echo "  2. 宸茶繍琛?huggingface-cli login 鐧诲綍"
