#!/bin/bash
# 在 HPC 登录节点下载数据集（仅数据集，不含模型）

echo "======================================================================"
echo "              KAVA 项目 - 数据集下载脚本                             "
echo "======================================================================"
echo ""
echo "使用方法:"
echo "  1. 直连下载:"
echo "     bash download_datasets_only.sh"
echo ""
echo "  2. 使用镜像下载 (推荐，国内更快):"
echo "     HF_ENDPOINT=https://hf-mirror.com bash download_datasets_only.sh"
echo ""
echo "======================================================================"
echo ""

# 检查是否安装了 huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "错误: 未安装 huggingface_hub"
    echo "请运行: pip install huggingface_hub"
    exit 1
fi

# 创建临时 Python 脚本
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
    print(f"下载: {repo_id}")
    print(f"路径: {local_dir}")
    print(f"{'='*60}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✓ {repo_id} 下载完成")
    except Exception as e:
        print(f"✗ {repo_id} 下载失败: {e}")

print("\n" + "="*60)
print("数据集下载完成!")
print("="*60)
PYTHON_SCRIPT

# 执行下载
python /tmp/download_datasets_temp.py

# 清理临时文件
rm /tmp/download_datasets_temp.py

echo ""
echo "下一步: 更新配置文件中的数据集路径"
echo "  - configs/llama1b_aug.yaml"
echo "  - configs/llama1b_aug_nl.yaml"
echo "  - configs/llama3b_aug.yaml"
echo "  - configs/qwen05b_aug.yaml"
