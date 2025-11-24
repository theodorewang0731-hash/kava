"""
Main training script for KAVA.
Run with: python train.py --config configs/llama1b_aug.yaml
"""

import argparse
import os
import sys

from src.trainer import KAVATrainer, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train KAVA model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file (e.g., configs/llama1b_aug.yaml)"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override seed if specified
    if args.seed is not None:
        config['system']['seed'] = args.seed
        print(f"Using seed: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = KAVATrainer(config)
    
    # Start training
    trainer.train(use_wandb=args.wandb)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
