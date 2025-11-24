#!/usr/bin/env python3
"""
聚合多种子实验结果，生成 mean ± std 统计
支持多数据集、LaTeX 表格输出
适配 KAVA 项目结构
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml


def load_metrics(seed_dir, dataset_name):
    """从种子目录加载指定数据集的 metrics"""
    # 尝试多种文件格式
    possible_files = [
        Path(seed_dir) / f'metrics_{dataset_name}.json',
        Path(seed_dir) / f'results_{dataset_name}.json',
        Path(seed_dir) / f'results_{dataset_name}.yaml',
    ]
    
    for metrics_file in possible_files:
        if metrics_file.exists():
            if metrics_file.suffix == '.json':
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
            else:  # yaml
                with open(metrics_file, 'r') as f:
                    data = yaml.safe_load(f)
            
            # 处理嵌套的数据结构
            if dataset_name in data:
                return data[dataset_name]
            return data
    
    return None


def aggregate_multi_seed(seed_dirs, datasets):
    """
    聚合多个种子的结果
    返回: {dataset: {'accuracy': [values], 'forward_passes': [values]}}
    """
    results = defaultdict(lambda: {'accuracy': [], 'forward_passes': []})
    
    for seed_dir in seed_dirs:
        for dataset in datasets:
            metrics = load_metrics(seed_dir, dataset)
            if metrics:
                # 处理不同的键名
                accuracy = metrics.get('accuracy', metrics.get('exact_match', 0))
                forward_passes = metrics.get('avg_forward_passes', metrics.get('forward_passes_mean', 0))
                
                results[dataset]['accuracy'].append(accuracy)
                results[dataset]['forward_passes'].append(forward_passes)
    
    return results


def compute_statistics(values):
    """计算 mean ± std"""
    if not values:
        return None, None
    arr = np.array(values)
    return arr.mean(), arr.std()


def generate_latex_table(stats, model_name):
    """生成 LaTeX 表格（类似论文 Table 1）"""
    print(f"\n{'='*60}")
    print(f"LaTeX Table for {model_name}")
    print('='*60)
    print(r"\begin{tabular}{l|cc|cc|cc}")
    print(r"\hline")
    print(r"Method & \multicolumn{2}{c|}{GSM8k} & \multicolumn{2}{c|}{GSM8k-Hard} & \multicolumn{2}{c}{SVAMP} \\")
    print(r" & Acc & FP & Acc & FP & Acc & FP \\")
    print(r"\hline")
    
    # KAVA 行
    row = model_name
    for dataset in ['gsm8k', 'gsm8k-hard', 'svamp']:
        if dataset in stats:
            acc_mean, acc_std = stats[dataset]['accuracy']
            fp_mean, fp_std = stats[dataset]['forward_passes']
            if acc_mean is not None:
                # 转换为百分比
                acc_mean_pct = acc_mean * 100 if acc_mean <= 1.0 else acc_mean
                acc_std_pct = acc_std * 100 if acc_std <= 1.0 else acc_std
                row += f" & ${acc_mean_pct:.1f} \\pm {acc_std_pct:.1f}$"
                row += f" & ${fp_mean:.1f} \\pm {fp_std:.1f}$"
            else:
                row += " & - & -"
        else:
            row += " & - & -"
    
    row += r" \\"
    print(row)
    print(r"\hline")
    print(r"\end{tabular}")
    print()


def print_summary_table(stats):
    """打印可读的汇总表格"""
    print(f"\n{'='*80}")
    print(f"{'Dataset':<20} {'Accuracy (mean±std)':<30} {'Forward Passes (mean±std)':<30}")
    print('='*80)
    
    for dataset, metrics in sorted(stats.items()):
        acc_mean, acc_std = metrics['accuracy']
        fp_mean, fp_std = metrics['forward_passes']
        
        if acc_mean is not None:
            # 转换为百分比
            acc_mean_pct = acc_mean * 100 if acc_mean <= 1.0 else acc_mean
            acc_std_pct = acc_std * 100 if acc_std <= 1.0 else acc_std
            acc_str = f"{acc_mean_pct:.2f} ± {acc_std_pct:.2f}"
            fp_str = f"{fp_mean:.2f} ± {fp_std:.2f}"
        else:
            acc_str = "N/A"
            fp_str = "N/A"
        
        print(f"{dataset:<20} {acc_str:<30} {fp_str:<30}")
    
    print('='*80)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed KAVA experiment results")
    parser.add_argument('--seed_dirs', nargs='+', required=True,
                        help='List of seed output directories')
    parser.add_argument('--datasets', nargs='+', 
                        default=['gsm8k', 'gsm8k-hard', 'svamp'],
                        help='Datasets to aggregate')
    parser.add_argument('--model_name', type=str, default='KAVA',
                        help='Model name for LaTeX table')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save aggregated results to JSON')
    parser.add_argument('--output_yaml', type=str, default=None,
                        help='Save aggregated results to YAML')
    args = parser.parse_args()
    
    # 验证种子目录存在
    valid_seed_dirs = []
    for seed_dir in args.seed_dirs:
        if Path(seed_dir).exists():
            valid_seed_dirs.append(seed_dir)
        else:
            print(f"Warning: Seed directory not found: {seed_dir}")
    
    if not valid_seed_dirs:
        print("Error: No valid seed directories found!")
        return
    
    print(f"\nAggregating results from {len(valid_seed_dirs)} seeds:")
    for seed_dir in valid_seed_dirs:
        print(f"  - {seed_dir}")
    
    # 聚合结果
    results = aggregate_multi_seed(valid_seed_dirs, args.datasets)
    
    # 计算统计量
    stats = {}
    for dataset, metrics in results.items():
        stats[dataset] = {
            'accuracy': compute_statistics(metrics['accuracy']),
            'forward_passes': compute_statistics(metrics['forward_passes'])
        }
    
    # 打印结果
    print_summary_table(stats)
    generate_latex_table(stats, args.model_name)
    
    # 保存 JSON（可选）
    if args.output_json:
        output = {
            'model': args.model_name,
            'num_seeds': len(valid_seed_dirs),
            'datasets': {}
        }
        for dataset, stat in stats.items():
            acc_mean, acc_std = stat['accuracy']
            fp_mean, fp_std = stat['forward_passes']
            output['datasets'][dataset] = {
                'accuracy_mean': float(acc_mean) if acc_mean is not None else None,
                'accuracy_std': float(acc_std) if acc_std is not None else None,
                'forward_passes_mean': float(fp_mean) if fp_mean is not None else None,
                'forward_passes_std': float(fp_std) if fp_std is not None else None
            }
        
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to {args.output_json}")
    
    # 保存 YAML（可选）
    if args.output_yaml:
        output = {
            'model': args.model_name,
            'num_seeds': len(valid_seed_dirs),
            'datasets': {}
        }
        for dataset, stat in stats.items():
            acc_mean, acc_std = stat['accuracy']
            fp_mean, fp_std = stat['forward_passes']
            output['datasets'][dataset] = {
                'accuracy_mean': float(acc_mean) if acc_mean is not None else None,
                'accuracy_std': float(acc_std) if acc_std is not None else None,
                'forward_passes_mean': float(fp_mean) if fp_mean is not None else None,
                'forward_passes_std': float(fp_std) if fp_std is not None else None
            }
        
        with open(args.output_yaml, 'w') as f:
            yaml.dump(output, f)
        print(f"✓ Results saved to {args.output_yaml}")


if __name__ == '__main__':
    main()
