"""
KAVA Multi-Model Configuration Management
Based on paper Table 6 and Appendix C hyperparameters
Supports LLaMA-3.2-1B/3B and Qwen2.5-0.5B with both GSM8k-AUG and GSM8k-AUG-NL
"""

from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional
import yaml
import json


@dataclass
class ModelConfig:
    """Configuration for a single model + dataset combination"""
    # Model identification
    model_name: str
    model_path: str
    dataset: str
    
    # LoRA configuration (consistent across all models)
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Training hyperparameters
    learning_rate: float = 8e-4
    weight_decay: float = 0.1
    num_epochs: int = 10
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 2.0
    warmup_ratio: float = 0.1
    
    # Loss weights
    alpha1_codi: float = 10.0
    alpha2_kv: float = 1.0
    
    # KV loss configuration
    kv_loss_type: str = 'smooth_l1'  # 'smooth_l1' or 'mse'
    layer_wise_std: bool = True
    use_projection: bool = True
    
    # R-KV compression
    rkv_lambda: float = 0.1
    
    # Latent reasoning
    num_latent_tokens: int = 24
    num_iterations: int = 3
    
    def __post_init__(self):
        """Set default target modules if not provided"""
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"✓ Config saved to {output_path}")
    
    def to_json(self, output_path: str):
        """Save configuration to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Config saved to {output_path}")


# ==================================================
# Predefined Configurations (Table 6)
# ==================================================

# LLaMA-3.2-1B + GSM8k-AUG
LLAMA1B_AUG = ModelConfig(
    model_name="llama1b",
    model_path="meta-llama/Llama-3.2-1B-Instruct",
    dataset="gsm8k-aug",
    learning_rate=8e-4,
    weight_decay=0.1,
    num_epochs=10,
    alpha1_codi=10.0,
    alpha2_kv=1.0,
    kv_loss_type='smooth_l1',
    layer_wise_std=True,
    use_projection=True,
    rkv_lambda=0.1
)

# LLaMA-3.2-1B + GSM8k-AUG-NL
LLAMA1B_AUG_NL = ModelConfig(
    model_name="llama1b",
    model_path="meta-llama/Llama-3.2-1B-Instruct",
    dataset="gsm8k-aug-nl",
    learning_rate=8e-4,
    weight_decay=0.1,
    num_epochs=10,
    alpha1_codi=10.0,
    alpha2_kv=1.0,
    kv_loss_type='mse',  # Different from AUG
    layer_wise_std=True,
    use_projection=True,
    rkv_lambda=0.1
)

# Qwen2.5-0.5B + GSM8k-AUG
QWEN05B_AUG = ModelConfig(
    model_name="qwen05b",
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    dataset="gsm8k-aug",
    learning_rate=5e-4,  # Lower LR
    weight_decay=0.01,   # Lower weight decay
    num_epochs=10,
    alpha1_codi=10.0,
    alpha2_kv=1.0,
    kv_loss_type='mse',
    layer_wise_std=False,  # No layer-wise normalization
    use_projection=True,
    rkv_lambda=0.1
)

# Qwen2.5-0.5B + GSM8k-AUG-NL
QWEN05B_AUG_NL = ModelConfig(
    model_name="qwen05b",
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    dataset="gsm8k-aug-nl",
    learning_rate=8e-4,  # Higher LR for NL
    weight_decay=0.1,
    num_epochs=10,
    alpha1_codi=10.0,
    alpha2_kv=1.0,
    kv_loss_type='mse',
    layer_wise_std=True,  # Use layer-wise for NL
    use_projection=True,
    rkv_lambda=0.1
)

# LLaMA-3.2-3B + GSM8k-AUG
LLAMA3B_AUG = ModelConfig(
    model_name="llama3b",
    model_path="meta-llama/Llama-3.2-3B-Instruct",
    dataset="gsm8k-aug",
    learning_rate=2e-4,  # Lower LR for larger model
    weight_decay=0.1,
    num_epochs=5,        # Fewer epochs
    alpha1_codi=20.0,    # Higher CODI weight
    alpha2_kv=2.0,       # Higher KV weight
    kv_loss_type='smooth_l1',
    layer_wise_std=False,
    use_projection=True,
    rkv_lambda=0.1
)

# LLaMA-3.2-3B + GSM8k-AUG-NL
LLAMA3B_AUG_NL = ModelConfig(
    model_name="llama3b",
    model_path="meta-llama/Llama-3.2-3B-Instruct",
    dataset="gsm8k-aug-nl",
    learning_rate=2e-4,
    weight_decay=0.1,
    num_epochs=5,
    alpha1_codi=20.0,
    alpha2_kv=2.0,
    kv_loss_type='smooth_l1',
    layer_wise_std=False,
    use_projection=False,  # No projection for NL
    rkv_lambda=0.0         # Only cosine similarity
)


# ==================================================
# Configuration Registry
# ==================================================

CONFIG_REGISTRY: Dict[str, ModelConfig] = {
    'llama1b_aug': LLAMA1B_AUG,
    'llama1b_aug_nl': LLAMA1B_AUG_NL,
    'qwen05b_aug': QWEN05B_AUG,
    'qwen05b_aug_nl': QWEN05B_AUG_NL,
    'llama3b_aug': LLAMA3B_AUG,
    'llama3b_aug_nl': LLAMA3B_AUG_NL,
}


def get_config(config_name: str) -> ModelConfig:
    """
    Get configuration by name.
    
    Args:
        config_name: Configuration name (e.g., 'llama1b_aug')
    
    Returns:
        ModelConfig object
    
    Raises:
        ValueError: If config_name not found
    
    Example:
        >>> config = get_config('llama1b_aug')
        >>> print(config.learning_rate)
        8e-4
    """
    if config_name not in CONFIG_REGISTRY:
        available = ', '.join(CONFIG_REGISTRY.keys())
        raise ValueError(
            f"Unknown config: '{config_name}'\n"
            f"Available configs: {available}"
        )
    
    return CONFIG_REGISTRY[config_name]


def list_configs(verbose: bool = False):
    """
    List all available configurations.
    
    Args:
        verbose: If True, print detailed parameters
    """
    print("="*70)
    print("Available KAVA Configurations")
    print("="*70)
    
    for name, config in CONFIG_REGISTRY.items():
        print(f"\n{name}:")
        print(f"  Model: {config.model_path}")
        print(f"  Dataset: {config.dataset}")
        
        if verbose:
            print(f"  Learning rate: {config.learning_rate:.0e}")
            print(f"  Weight decay: {config.weight_decay}")
            print(f"  Epochs: {config.num_epochs}")
            print(f"  Loss weights: α₁={config.alpha1_codi}, α₂={config.alpha2_kv}")
            print(f"  KV loss: {config.kv_loss_type}, Layer-wise std: {config.layer_wise_std}")
            print(f"  Projection: {config.use_projection}, R-KV λ: {config.rkv_lambda}")
    
    print("\n" + "="*70)


def compare_configs(config1_name: str, config2_name: str):
    """
    Compare two configurations and highlight differences.
    
    Args:
        config1_name: First config name
        config2_name: Second config name
    
    Example:
        >>> compare_configs('llama1b_aug', 'llama1b_aug_nl')
    """
    if config1_name not in CONFIG_REGISTRY or config2_name not in CONFIG_REGISTRY:
        print(f"❌ Invalid config name(s)")
        return
    
    config1 = CONFIG_REGISTRY[config1_name]
    config2 = CONFIG_REGISTRY[config2_name]
    
    print(f"\n{'='*70}")
    print(f"Comparing: {config1_name} vs {config2_name}")
    print('='*70)
    
    # Find differences
    differences = []
    for field in fields(ModelConfig):
        val1 = getattr(config1, field.name)
        val2 = getattr(config2, field.name)
        
        if val1 != val2:
            differences.append((field.name, val1, val2))
    
    if not differences:
        print("✓ Configurations are identical")
    else:
        print(f"\nFound {len(differences)} difference(s):\n")
        print(f"{'Field':<25} {config1_name:<25} {config2_name:<25}")
        print('-'*70)
        for field, val1, val2 in differences:
            val1_str = str(val1)[:20]
            val2_str = str(val2)[:20]
            print(f"{field:<25} {val1_str:<25} {val2_str:<25}")
    
    print('='*70 + '\n')


def export_all_configs(output_dir: str = "configs", format: str = "yaml"):
    """
    Export all configurations to files.
    
    Args:
        output_dir: Output directory
        format: 'yaml' or 'json'
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExporting all configurations to {output_dir}/")
    print("="*70)
    
    for name, config in CONFIG_REGISTRY.items():
        if format == "yaml":
            output_path = os.path.join(output_dir, f"{name}.yaml")
            config.to_yaml(output_path)
        elif format == "json":
            output_path = os.path.join(output_dir, f"{name}.json")
            config.to_json(output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    print(f"\n✓ Exported {len(CONFIG_REGISTRY)} configurations")


def get_model_by_size(size: str) -> List[str]:
    """
    Get all configurations for a specific model size.
    
    Args:
        size: '1b', '0.5b', or '3b'
    
    Returns:
        List of config names
    
    Example:
        >>> get_model_by_size('1b')
        ['llama1b_aug', 'llama1b_aug_nl']
    """
    size_map = {
        '1b': 'llama1b',
        '0.5b': 'qwen05b',
        '3b': 'llama3b'
    }
    
    if size not in size_map:
        raise ValueError(f"Unknown size: {size}. Available: {list(size_map.keys())}")
    
    model_prefix = size_map[size]
    return [name for name in CONFIG_REGISTRY.keys() if name.startswith(model_prefix)]


def get_configs_by_dataset(dataset: str) -> List[str]:
    """
    Get all configurations for a specific dataset.
    
    Args:
        dataset: 'aug' or 'aug_nl'
    
    Returns:
        List of config names
    
    Example:
        >>> get_configs_by_dataset('aug')
        ['llama1b_aug', 'qwen05b_aug', 'llama3b_aug']
    """
    return [name for name in CONFIG_REGISTRY.keys() if name.endswith(dataset)]


# ==================================================
# Command-line Interface
# ==================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="KAVA Model Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all configurations
  python model_configs.py --list
  
  # List with detailed parameters
  python model_configs.py --list --verbose
  
  # Compare two configurations
  python model_configs.py --compare llama1b_aug llama1b_aug_nl
  
  # Get specific configuration details
  python model_configs.py --get llama3b_aug
  
  # Export all configs to YAML
  python model_configs.py --export configs/ --format yaml
  
  # Get configs for 1B models
  python model_configs.py --filter-size 1b
        """
    )
    
    parser.add_argument('--list', action='store_true',
                        help='List all available configurations')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed parameters (use with --list)')
    parser.add_argument('--compare', nargs=2, metavar=('CONFIG1', 'CONFIG2'),
                        help='Compare two configurations')
    parser.add_argument('--get', type=str, metavar='CONFIG_NAME',
                        help='Get detailed info for a specific config')
    parser.add_argument('--export', type=str, metavar='OUTPUT_DIR',
                        help='Export all configs to directory')
    parser.add_argument('--format', type=str, choices=['yaml', 'json'], default='yaml',
                        help='Export format (default: yaml)')
    parser.add_argument('--filter-size', type=str, choices=['1b', '0.5b', '3b'],
                        help='Filter configs by model size')
    parser.add_argument('--filter-dataset', type=str, choices=['aug', 'aug_nl'],
                        help='Filter configs by dataset')
    
    args = parser.parse_args()
    
    if args.list:
        list_configs(verbose=args.verbose)
    
    elif args.compare:
        compare_configs(args.compare[0], args.compare[1])
    
    elif args.get:
        try:
            config = get_config(args.get)
            print(f"\nConfiguration: {args.get}")
            print("="*70)
            for field in fields(ModelConfig):
                value = getattr(config, field.name)
                print(f"  {field.name}: {value}")
            print("="*70)
        except ValueError as e:
            print(f"❌ {e}")
    
    elif args.export:
        export_all_configs(args.export, format=args.format)
    
    elif args.filter_size:
        configs = get_model_by_size(args.filter_size)
        print(f"\nConfigurations for {args.filter_size} models:")
        for name in configs:
            print(f"  - {name}")
    
    elif args.filter_dataset:
        configs = get_configs_by_dataset(args.filter_dataset)
        print(f"\nConfigurations for dataset '{args.filter_dataset}':")
        for name in configs:
            print(f"  - {name}")
    
    else:
        parser.print_help()
