#!/usr/bin/env python3
"""
下载所有数据集到本地 datasets/ 目录

在 HPC 登录节点运行:
    python datasets/download_datasets.py

使用镜像加速:
    python datasets/download_datasets.py --mirror
"""

import os
import sys
from huggingface_hub import snapshot_download

# =================配置区域=================
# 1. 如果下载速度慢，取消下面这行的注释使用国内镜像
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 定义数据集映射关系 (HF 仓库 ID -> 本地目录名)
DATASETS_TO_DOWNLOAD = {
    # 训练集 (KAVA 论文核心)
    "whynlp/gsm8k-aug": "gsm8k-aug",
    "whynlp/gsm8k-aug-nl": "gsm8k-aug-nl",
    
    # 评估集 (标准)
    "openai/gsm8k": "gsm8k",
    
    # 评估集 (OOD / Hard)
    "reasoning-machines/gsm-hard": "gsm8k-hard",
    "ChilleD/SVAMP": "svamp"
}
# ==========================================

def download_all():
    """下载所有数据集"""
    # 获取脚本所在目录（即 datasets/ 目录）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"📂 数据集将下载到: {base_dir}")
    print(f"🚀 开始下载 {len(DATASETS_TO_DOWNLOAD)} 个数据集...\n")

    for repo_id, dir_name in DATASETS_TO_DOWNLOAD.items():
        local_dir = os.path.join(base_dir, dir_name)
        
        print(f"⬇️  正在下载: {repo_id} -> {dir_name}/ ...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # 下载真实文件而非链接
                resume_download=True,          # 支持断点续传
                ignore_patterns=[".git*", "*.msgpack"] # 忽略非必要文件
            )
            print(f"✅ 成功: {dir_name}")
        except Exception as e:
            print(f"❌ 失败: {repo_id}")
            print(f"   错误信息: {str(e)}")
            print("   提示: 如果是网络问题，请尝试开启 HF_ENDPOINT 镜像设置")

    print("\n🎉 所有任务处理完成！")

if __name__ == "__main__":
    # 检查是否安装了 huggingface_hub
    try:
        import huggingface_hub
    except ImportError:
        print("❌ 错误: 未找到 huggingface_hub 库")
        print("请先运行: pip install huggingface_hub")
        sys.exit(1)
    
    # 支持 --mirror 参数启用镜像
    if "--mirror" in sys.argv:
        print("✓ 使用 HF-Mirror 镜像加速\n")
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    download_all()
