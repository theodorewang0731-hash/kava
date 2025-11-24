"""
在 HPC 登录节点下载模型和数据集
⚠️ 注意: 如果 HPC 已有共享模型库，无需运行此脚本！
先运行: bash check_hpc_models_availability.sh 检查共享模型

登录节点有网络访问，下载到用户目录后计算节点可以使用缓存
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# 使用 HF-Mirror 镜像加速下载（可选）
# 如果直连 HuggingFace 速度慢，取消下面这行的注释
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def check_hpc_shared_models():
    """检查 HPC 共享模型库"""
    hpc_models = Path("/home/share/models")
    
    if not hpc_models.exists():
        return False
    
    print("=" * 80)
    print("🔍 检测到 HPC 共享模型库: /home/share/models")
    print("=" * 80)
    
    # 检查所需模型
    required_models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct"
    ]
    
    all_found = True
    for model in required_models:
        model_path = hpc_models / f"models--{model.replace('/', '--')}"
        if model_path.exists():
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model} (未找到)")
            all_found = False
    
    print()
    
    if all_found:
        print("✅ 所有模型都在共享库中，无需下载！")
        print()
        print("请使用以下环境变量:")
        print("  export HF_HOME=/home/share/models")
        print("  export TRANSFORMERS_CACHE=/home/share/models")
        print("  export HUGGINGFACE_HUB_OFFLINE=1")
        print()
        print("或运行配置脚本:")
        print("  bash simple_setup.sh")
        print()
        return True
    else:
        print("⚠️  部分模型缺失，将下载到个人缓存")
        print()
        return False

def download_models():
    """下载所需的模型"""
    print("=" * 80)
    print("开始下载模型...")
    print("=" * 80)
    print()
    print("⚠️  注意: Llama 模型需要 HuggingFace 授权访问")
    print("请访问以下链接申请访问权限:")
    print("  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    print("  https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
    print()
    print("授权后，需要设置 HuggingFace token:")
    print("  export HF_TOKEN=your_token_here")
    print("=" * 80)
    print()
    
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

⚠️  重要提示：
  1. 优先检查 HPC 共享模型库（运行下面的检查）
  2. 如果共享库有模型，则无需下载，直接使用！
  3. Llama 模型需要 HuggingFace 授权访问

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
    
    # =========================================================================
    # 首先检查 HPC 共享模型库
    # =========================================================================
    if check_hpc_shared_models():
        print("🎉 建议：直接使用共享模型，无需下载！")
        print()
        response = input("是否仍要下载到个人缓存? (y/N): ").strip().lower()
        if response != 'y':
            print("\n✓ 已取消下载。请使用共享模型库运行训练。")
            sys.exit(0)
        print("\n⚠️  将下载到个人缓存...")
    
    # =========================================================================
    # 继续下载流程
    # =========================================================================
    
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
