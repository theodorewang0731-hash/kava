#!/usr/bin/env python3
"""
训练前验证脚本 - 确保所有组件正确无误
"""

import sys
import os
from pathlib import Path
import importlib.util


def check_file_exists(path: str, description: str) -> bool:
    """检查文件是否存在"""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}缺失: {path}")
        return False


def check_import(module_name: str) -> bool:
    """检查模块是否可导入"""
    try:
        importlib.import_module(module_name)
        print(f"✓ 模块可导入: {module_name}")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {module_name}")
        print(f"  错误: {e}")
        return False


def check_config_files() -> bool:
    """检查配置文件"""
    print("\n[1] 检查配置文件")
    print("-" * 50)
    
    configs = [
        "configs/llama1b_aug.yaml",
        "configs/llama1b_aug_nl.yaml",
        "configs/llama3b_aug.yaml",
        "configs/qwen05b_aug.yaml"
    ]
    
    all_ok = True
    for config in configs:
        if not check_file_exists(config, "配置"):
            all_ok = False
    
    return all_ok


def check_core_files() -> bool:
    """检查核心文件"""
    print("\n[2] 检查核心文件")
    print("-" * 50)
    
    files = {
        "train.py": "训练脚本",
        "evaluate.py": "评估脚本",
        "requirements.txt": "依赖文件",
        "submit_multi_seed.slurm": "SLURM 脚本"
    }
    
    all_ok = True
    for file, desc in files.items():
        if not check_file_exists(file, desc):
            all_ok = False
    
    return all_ok


def check_src_modules() -> bool:
    """检查源代码模块"""
    print("\n[3] 检查源代码模块")
    print("-" * 50)
    
    # 核心必需模块
    required_modules = [
        "src.trainer",
        "src.latent_reasoning",
        "src.data_utils",
        "src.utils",
        "src.losses"
    ]
    
    # 可选模块
    optional_modules = [
        "src.evaluation_datasets",
        "src.rkv_compression"
    ]
    
    all_ok = True
    
    print("核心模块:")
    for module in required_modules:
        if not check_import(module):
            all_ok = False
    
    print("\n可选模块:")
    for module in optional_modules:
        check_import(module)  # 不影响 all_ok
    
    return all_ok


def check_dependencies() -> bool:
    """检查依赖包"""
    print("\n[4] 检查关键依赖")
    print("-" * 50)
    
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT",
        "datasets": "Datasets",
        "yaml": "PyYAML"
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name} 未安装")
            all_ok = False
    
    return all_ok


def check_gpu() -> bool:
    """检查 GPU 可用性"""
    print("\n[5] 检查 GPU")
    print("-" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用: {torch.version.cuda}")
            print(f"✓ GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️ CUDA 不可用（登录节点正常，训练会在 GPU 节点运行）")
            return True  # 这是正常的
    except Exception as e:
        print(f"✗ GPU 检查失败: {e}")
        return False


def check_hpc_models() -> bool:
    """检查 HPC 共享模型"""
    print("\n[6] 检查 HPC 共享模型")
    print("-" * 50)
    
    models_dir = Path("/home/share/models")
    required_models = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Qwen2.5-0.5B-Instruct"
    ]
    
    if not models_dir.exists():
        print(f"⚠️ 共享模型目录不存在: {models_dir}")
        print("  将使用个人缓存目录下载模型")
        return True  # 不是致命错误
    
    all_ok = True
    for model in required_models:
        model_path = models_dir / model
        if model_path.exists():
            print(f"✓ {model}")
        else:
            print(f"⚠️ {model} (未找到，需要下载)")
            all_ok = False
    
    return all_ok


def check_slurm_config() -> bool:
    """检查 SLURM 配置"""
    print("\n[7] 检查 SLURM 配置")
    print("-" * 50)
    
    with open("submit_multi_seed.slurm", 'r') as f:
        content = f.read()
    
    checks = {
        '--gres=gpu': 'GPU 请求',
        '--cpus-per-task': 'CPU 配置',
        '--mem': '内存配置',
        'source venv/bin/activate': 'Python 环境激活',
        'python train.py': '训练命令'
    }
    
    all_ok = True
    for pattern, desc in checks.items():
        if pattern in content:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc} 缺失")
            all_ok = False
    
    return all_ok


def main():
    """主验证流程"""
    print("=" * 60)
    print("KAVA 训练前验证")
    print("=" * 60)
    
    checks = [
        ("配置文件", check_config_files),
        ("核心文件", check_core_files),
        ("源代码模块", check_src_modules),
        ("依赖包", check_dependencies),
        ("GPU", check_gpu),
        ("HPC 模型", check_hpc_models),
        ("SLURM 配置", check_slurm_config)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name}检查失败: {e}")
            results.append((name, False))
    
    # 汇总
    print("\n" + "=" * 60)
    print("验证汇总")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ 所有检查通过！可以开始训练。")
        return 0
    else:
        print("\n⚠️ 部分检查未通过，请修复后再训练。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
