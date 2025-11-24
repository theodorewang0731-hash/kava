"""
在 HPC 登录节点下载模型和数据集
登录节点有网络访问，下载到用户目录后计算节点可以使用缓存
"""

import os
from huggingface_hub import snapshot_download

# 使用 HF-Mirror 镜像加速下载（可选）
# 如果直连 HuggingFace 速度慢，取消下面这行的注释
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_models():
    """下载所需的模型"""
    print("=" * 60)
    print("开始下载模型...")
    print("=" * 60)
    
    models = [
        {
            "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
            "local_dir": "./models/Llama-3.2-1B-Instruct"
        },
        {
            "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
            "local_dir": "./models/Llama-3.2-3B-Instruct"
        },
        {
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_dir": "./models/Qwen2.5-0.5B-Instruct"
        }
    ]
    
    for model_info in models:
        print(f"\n下载模型: {model_info['repo_id']}")
        print(f"保存路径: {model_info['local_dir']}")
        
        try:
            snapshot_download(
                repo_id=model_info['repo_id'],
                local_dir=model_info['local_dir'],
                local_dir_use_symlinks=False,  # 不使用符号链接
                resume_download=True  # 支持断点续传
            )
            print(f"✓ {model_info['repo_id']} 下载完成")
        except Exception as e:
            print(f"✗ {model_info['repo_id']} 下载失败: {e}")
            print("  提示: 如果是 LLaMA 模型，请确保已通过 HuggingFace 授权")


def download_datasets():
    """下载所需的数据集"""
    print("\n" + "=" * 60)
    print("开始下载数据集...")
    print("=" * 60)
    
    datasets = [
        {
            "repo_id": "whynlp/gsm8k-aug",
            "local_dir": "./datasets/gsm8k-aug"
        },
        {
            "repo_id": "whynlp/gsm8k-aug-nl",
            "local_dir": "./datasets/gsm8k-aug-nl"
        },
        {
            "repo_id": "gsm8k",
            "local_dir": "./datasets/gsm8k"
        }
    ]
    
    for dataset_info in datasets:
        print(f"\n下载数据集: {dataset_info['repo_id']}")
        print(f"保存路径: {dataset_info['local_dir']}")
        
        try:
            snapshot_download(
                repo_id=dataset_info['repo_id'],
                repo_type="dataset",
                local_dir=dataset_info['local_dir'],
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✓ {dataset_info['repo_id']} 下载完成")
        except Exception as e:
            print(f"✗ {dataset_info['repo_id']} 下载失败: {e}")


def check_hf_cache():
    """检查 HuggingFace 缓存目录"""
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    print("\n" + "=" * 60)
    print(f"HuggingFace 缓存目录: {cache_dir}")
    
    if os.path.exists(cache_dir):
        try:
            import subprocess
            result = subprocess.run(
                ["du", "-sh", cache_dir],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"缓存大小: {result.stdout.strip()}")
        except:
            print("无法获取缓存大小")
    else:
        print("缓存目录不存在")
    print("=" * 60)


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║           KAVA 项目 HuggingFace 资源下载工具               ║
╚════════════════════════════════════════════════════════════╝

使用方法:
  1. 直连下载:
     python download_from_hf.py
  
  2. 使用镜像下载 (推荐，国内更快):
     HF_ENDPOINT=https://hf-mirror.com python download_from_hf.py
  
  3. 只下载模型:
     python download_from_hf.py --models-only
  
  4. 只下载数据集:
     python download_from_hf.py --datasets-only

注意事项:
  - 请在 HPC 登录节点运行（有网络访问）
  - LLaMA 模型需要先在 HuggingFace 申请授权
  - 需要 HuggingFace token: huggingface-cli login
  - 下载完成后，计算节点可通过缓存访问
""")
    
    import sys
    
    download_models_flag = True
    download_datasets_flag = True
    
    if "--models-only" in sys.argv:
        download_datasets_flag = False
    elif "--datasets-only" in sys.argv:
        download_models_flag = False
    
    # 检查是否设置了镜像
    if os.environ.get('HF_ENDPOINT'):
        print(f"✓ 使用镜像: {os.environ['HF_ENDPOINT']}\n")
    else:
        print("✓ 使用官方源: https://huggingface.co\n")
    
    # 下载资源
    if download_models_flag:
        download_models()
    
    if download_datasets_flag:
        download_datasets()
    
    # 显示缓存信息
    check_hf_cache()
    
    print("\n" + "=" * 60)
    print("下载任务完成!")
    print("=" * 60)
    print("""
后续步骤:
  1. 检查下载的文件是否完整
  2. 更新配置文件中的模型和数据集路径:
     - configs/llama1b_aug.yaml
     - configs/llama1b_aug_nl.yaml
     - configs/llama3b_aug.yaml
     - configs/qwen05b_aug.yaml
  
  3. 提交训练任务:
     bash submit_all_jobs.sh
""")
