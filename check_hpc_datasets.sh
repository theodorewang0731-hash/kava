#!/bin/bash
#==============================================================================
# 检查 HPC 共享库是否有所需数据集
#==============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 检查 HPC 数据集共享库"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 需要的数据集
REQUIRED_DATASETS=(
    "whynlp/gsm8k-aug"
    "whynlp/gsm8k-aug-nl"
    "gsm8k"
)

echo "需要的数据集:"
for dataset in "${REQUIRED_DATASETS[@]}"; do
    echo "  - $dataset"
done
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【1】检查 HPC 共享数据集目录"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 可能的共享数据集位置
SHARED_PATHS=(
    "/home/share/datasets"
    "/home/share/data"
    "/home/share/huggingface/datasets"
    "/home/share/.cache/huggingface/datasets"
    "/datasets"
    "/data"
)

echo "搜索可能的共享数据集目录..."
for path in "${SHARED_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✓ 找到目录: $path"
        echo "  内容（前 20 项）:"
        ls -lh "$path" | head -20
        echo ""
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【2】搜索 GSM8K 相关数据集"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 在 /home/share 下搜索 gsm8k 相关
if [ -d "/home/share" ]; then
    echo "搜索 /home/share 下的 gsm8k 相关内容..."
    find /home/share -iname "*gsm8k*" -type d 2>/dev/null | head -20
    
    echo ""
    echo "搜索 /home/share 下的 datasets 目录..."
    find /home/share -iname "*dataset*" -type d -maxdepth 3 2>/dev/null | head -20
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【3】检查常见 HuggingFace 缓存位置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查是否有其他用户的缓存可以共享
CACHE_PATHS=(
    "/home/share/.cache/huggingface/datasets"
    "$HOME/.cache/huggingface/datasets"
    "/scratch/.cache/huggingface/datasets"
)

for cache_path in "${CACHE_PATHS[@]}"; do
    if [ -d "$cache_path" ]; then
        echo "✓ 找到缓存目录: $cache_path"
        echo "  内容:"
        ls -lh "$cache_path" | head -10
        echo ""
    else
        echo "✗ 未找到: $cache_path"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【4】检查管理员提供的说明文档"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

DOC_PATHS=(
    "/home/share/README"
    "/home/share/README.md"
    "/home/share/DATASETS.md"
    "/home/share/datasets/README"
)

for doc in "${DOC_PATHS[@]}"; do
    if [ -f "$doc" ]; then
        echo "✓ 找到文档: $doc"
        echo "  内容:"
        cat "$doc"
        echo ""
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【5】建议的数据集解决方案"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "方案 1: 如果找到共享数据集 → 配置路径使用共享数据集"
echo "方案 2: 如果没有共享数据集 → 联系管理员添加或申请外网访问"
echo "方案 3: 如果有代理节点 → 在有网络的节点下载后传输"
echo ""
echo "请把以上输出发给我，我会根据结果提供具体解决方案"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
