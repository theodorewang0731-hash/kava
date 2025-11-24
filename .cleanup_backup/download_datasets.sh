#!/bin/bash
#==============================================================================
# 在登录节点预下载数据集（有网络访问）
# 数据集会缓存到 ~/.cache/huggingface/datasets
#==============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 KAVA 数据集预下载"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "说明："
echo "  - 计算节点无外网访问"
echo "  - 需在登录节点（有网络）预下载数据集"
echo "  - 数据集将缓存到 ~/.cache/huggingface/datasets"
echo "  - 计算节点运行时会自动使用缓存"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "/home/rpwang/kava review" || {
    echo "❌ 错误: 无法进入项目目录"
    exit 1
}

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 设置缓存目录
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
mkdir -p "$HF_DATASETS_CACHE"

echo "数据集缓存目录: $HF_DATASETS_CACHE"
echo ""

# 下载所需的数据集
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "开始下载数据集..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -c "
import sys
from datasets import load_dataset

datasets_to_download = [
    ('whynlp/gsm8k-aug', 'GSM8K-AUG（方程式 CoT）'),
    ('whynlp/gsm8k-aug-nl', 'GSM8K-AUG-NL（自然语言 CoT）'),
    ('gsm8k', 'GSM8K（评估）'),
]

print('需要下载的数据集:')
for repo_id, desc in datasets_to_download:
    print(f'  - {repo_id}: {desc}')
print()

success_count = 0
for repo_id, desc in datasets_to_download:
    print('━' * 60)
    print(f'【下载】{desc}')
    print(f'数据集: {repo_id}')
    print('━' * 60)
    
    try:
        # 下载数据集（会自动缓存）
        dataset = load_dataset(repo_id)
        
        # 显示数据集信息
        print(f'✓ 下载成功')
        print(f'  包含 splits: {list(dataset.keys())}')
        
        # 显示样本数量
        for split_name, split_data in dataset.items():
            print(f'  - {split_name}: {len(split_data)} 条数据')
        
        success_count += 1
        print()
        
    except Exception as e:
        print(f'✗ 下载失败: {e}')
        print()

print('━' * 60)
print('下载完成摘要')
print('━' * 60)
print(f'成功: {success_count}/{len(datasets_to_download)}')

if success_count == len(datasets_to_download):
    print()
    print('✅ 所有数据集下载成功！')
    print()
    print('缓存位置: $HOME/.cache/huggingface/datasets')
    print('计算节点运行时会自动使用这些缓存')
    sys.exit(0)
else:
    print()
    print('⚠️  部分数据集下载失败')
    print('请检查网络连接或数据集名称是否正确')
    sys.exit(1)
"

DOWNLOAD_EXIT=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $DOWNLOAD_EXIT -eq 0 ]; then
    echo "✅ 数据集下载完成"
    echo ""
    echo "查看缓存:"
    echo "  ls -lh ~/.cache/huggingface/datasets/"
    echo ""
    echo "现在可以提交训练任务:"
    echo "  bash submit_all_jobs.sh"
else
    echo "❌ 数据集下载失败"
    echo ""
    echo "故障排查:"
    echo "  1. 检查登录节点是否有网络:"
    echo "     ping -c 3 huggingface.co"
    echo ""
    echo "  2. 检查数据集是否存在:"
    echo "     访问 https://huggingface.co/datasets/whynlp/gsm8k-aug"
    echo ""
    echo "  3. 检查 datasets 库版本:"
    echo "     pip show datasets"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
