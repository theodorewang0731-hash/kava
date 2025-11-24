#!/usr/bin/env python3
"""
快速验证脚本：测试共享模型库能否被 transformers 识别
用于诊断 HPC 上的模型加载问题
"""

import os
import sys
from pathlib import Path

def test_model_loading():
    """测试三种模型加载方式"""
    
    print("=" * 70)
    print("KAVA 模型加载诊断工具")
    print("=" * 70)
    print()
    
    # 显示当前环境变量
    print("【环境变量】")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '未设置')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '未设置')}")
    print(f"HUGGINGFACE_HUB_OFFLINE: {os.environ.get('HUGGINGFACE_HUB_OFFLINE', '未设置')}")
    print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE', '未设置')}")
    print()
    
    # 检查共享模型库
    shared_models = Path("/home/share/models")
    print("【共享模型库检查】")
    if shared_models.exists():
        print(f"✓ 共享库存在: {shared_models}")
        models_to_check = [
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-3B-Instruct", 
            "Qwen2.5-0.5B-Instruct"
        ]
        for model_name in models_to_check:
            model_path = shared_models / model_name
            if model_path.exists():
                print(f"  ✓ {model_name}")
                # 检查关键文件
                config_file = model_path / "config.json"
                model_file = list(model_path.glob("*.safetensors")) or list(model_path.glob("*.bin"))
                print(f"    - config.json: {'✓' if config_file.exists() else '✗'}")
                print(f"    - 模型文件: {'✓' if model_file else '✗'}")
            else:
                print(f"  ✗ {model_name} 不存在")
    else:
        print(f"✗ 共享库不存在: {shared_models}")
    print()
    
    # 测试 transformers 导入
    print("【测试 transformers 库】")
    try:
        import transformers
        print(f"✓ transformers 版本: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ 无法导入 transformers: {e}")
        return False
    print()
    
    # 测试方式 1: 使用 HF repo ID (会尝试联网)
    print("【测试 1: 使用 HF repo ID (可能尝试联网)】")
    test_configs = [
        ("meta-llama/Llama-3.2-1B-Instruct", "Llama 1B"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen 0.5B"),
    ]
    
    for repo_id, name in test_configs:
        print(f"\n尝试加载: {name} ({repo_id})")
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                repo_id,
                trust_remote_code=True,
                local_files_only=False  # 允许尝试联网
            )
            print(f"  ✓ 成功加载配置（可能来自缓存或网络）")
        except Exception as e:
            print(f"  ✗ 失败: {type(e).__name__}: {str(e)[:100]}")
    
    # 测试方式 2: 强制离线模式
    print("\n" + "=" * 70)
    print("【测试 2: 强制离线模式 (HUGGINGFACE_HUB_OFFLINE=1)】")
    os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    for repo_id, name in test_configs:
        print(f"\n尝试加载: {name} ({repo_id}) - 离线模式")
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                repo_id,
                trust_remote_code=True,
                local_files_only=True
            )
            print(f"  ✓ 成功加载配置（来自本地缓存）")
        except Exception as e:
            error_msg = str(e)
            if "Cannot find" in error_msg or "does not appear to have" in error_msg:
                print(f"  ✗ 本地缓存中没有找到该模型")
                print(f"     建议：使用本地路径代替 repo ID")
            else:
                print(f"  ✗ 失败: {type(e).__name__}: {error_msg[:150]}")
    
    # 测试方式 3: 直接使用本地路径
    print("\n" + "=" * 70)
    print("【测试 3: 直接使用本地路径 (推荐方式)】")
    
    local_paths = [
        ("/home/share/models/Llama-3.2-1B-Instruct", "Llama 1B"),
        ("/home/share/models/Qwen2.5-0.5B-Instruct", "Qwen 0.5B"),
    ]
    
    success_count = 0
    for local_path, name in local_paths:
        print(f"\n尝试加载: {name} ({local_path})")
        if not Path(local_path).exists():
            print(f"  ✗ 路径不存在")
            continue
            
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                local_path,
                trust_remote_code=True,
                local_files_only=True
            )
            print(f"  ✓ 成功加载配置")
            print(f"     模型类型: {config.model_type}")
            print(f"     隐藏层大小: {config.hidden_size}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ 失败: {type(e).__name__}: {str(e)[:150]}")
    
    # 最终建议
    print("\n" + "=" * 70)
    print("【诊断结果和建议】")
    print("=" * 70)
    
    if success_count == len(local_paths):
        print("\n✅ 推荐方案: 在配置文件中使用本地路径")
        print("\n修改 configs/*.yaml 中的 model.name 为:")
        print("  - llama1b: /home/share/models/Llama-3.2-1B-Instruct")
        print("  - llama3b: /home/share/models/Llama-3.2-3B-Instruct")
        print("  - qwen05b: /home/share/models/Qwen2.5-0.5B-Instruct")
        print("\n这样可以:")
        print("  ✓ 避免网络访问")
        print("  ✓ 加载速度更快")
        print("  ✓ 不依赖缓存布局")
        return True
    else:
        print("\n⚠️  警告: 部分本地路径测试失败")
        print("\n请检查:")
        print("  1. /home/share/models 目录是否可访问")
        print("  2. 模型文件是否完整（config.json 和 .safetensors/.bin）")
        print("  3. 文件权限是否正确")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
