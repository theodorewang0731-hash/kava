"""
Aggregate results from multiple runs and compute mean ± std.
Generates final table matching paper format.
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_results(results_dir: str) -> Dict:
    """Load all result files from directory."""
    results_path = Path(results_dir)
    all_results = {}
    
    for result_file in results_path.glob("*.yaml"):
        # Parse filename: model-dataset-seed.yaml
        name = result_file.stem
        parts = name.split('-')
        
        if len(parts) >= 3:
            model = parts[0]
            dataset = parts[1]
            seed = parts[2] if 'seed' in parts[2] else None
            
            key = f"{model}-{dataset}"
            
            with open(result_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if key not in all_results:
                all_results[key] = []
            
            all_results[key].append(data)
    
    return all_results


def compute_statistics(results_list: List[Dict]) -> Dict:
    """Compute mean and std across multiple runs."""
    stats = {}
    
    # Get all dataset names from first result
    if not results_list:
        return stats
    
    dataset_names = list(results_list[0].keys())
    
    for dataset_name in dataset_names:
        accuracies = []
        forward_passes = []
        
        for result in results_list:
            if dataset_name in result:
                accuracies.append(result[dataset_name]['accuracy'] * 100)
                forward_passes.append(result[dataset_name]['avg_forward_passes'])
        
        if accuracies:
            stats[dataset_name] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'forward_passes_mean': np.mean(forward_passes),
                'forward_passes_std': np.std(forward_passes),
                'n_runs': len(accuracies)
            }
    
    return stats


def format_table(all_stats: Dict) -> str:
    """Format results as a table similar to paper."""
    table = []
    table.append("=" * 80)
    table.append("KAVA Results - Replication of Table 1 & 2")
    table.append("=" * 80)
    table.append("")
    
    # Group by model
    models = {}
    for key in all_stats.keys():
        model = key.split('-')[0]
        if model not in models:
            models[model] = []
        models[model].append(key)
    
    for model, configs in models.items():
        table.append(f"\n{model.upper()}")
        table.append("-" * 80)
        
        for config in configs:
            stats = all_stats[config]
            dataset_type = config.split('-')[1]
            
            table.append(f"\nDataset: {dataset_type}")
            table.append(f"  GSM8k:      {stats.get('gsm8k', {}).get('accuracy_mean', 0):.1f} ± {stats.get('gsm8k', {}).get('accuracy_std', 0):.1f}%  |  FP: {stats.get('gsm8k', {}).get('forward_passes_mean', 0):.1f} ± {stats.get('gsm8k', {}).get('forward_passes_std', 0):.1f}")
            table.append(f"  GSM8k-Hard: {stats.get('gsm8k-hard', {}).get('accuracy_mean', 0):.1f} ± {stats.get('gsm8k-hard', {}).get('accuracy_std', 0):.1f}%  |  FP: {stats.get('gsm8k-hard', {}).get('forward_passes_mean', 0):.1f} ± {stats.get('gsm8k-hard', {}).get('forward_passes_std', 0):.1f}")
            table.append(f"  SVAMP:      {stats.get('svamp', {}).get('accuracy_mean', 0):.1f} ± {stats.get('svamp', {}).get('accuracy_std', 0):.1f}%  |  FP: {stats.get('svamp', {}).get('forward_passes_mean', 0):.1f} ± {stats.get('svamp', {}).get('forward_passes_std', 0):.1f}")
    
    table.append("")
    table.append("=" * 80)
    table.append("Legend: FP = Forward Passes (avg over test set)")
    table.append("All results are mean ± std over 3 random seeds")
    table.append("=" * 80)
    
    return "\n".join(table)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate KAVA results")
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result YAML files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/summary.yaml",
        help="Output file for aggregated results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    
    # Load all results
    all_results = load_results(args.results_dir)
    
    # Compute statistics for each configuration
    all_stats = {}
    for config_name, results_list in all_results.items():
        print(f"Processing {config_name}: {len(results_list)} runs")
        all_stats[config_name] = compute_statistics(results_list)
    
    # Save aggregated results
    with open(args.output, 'w') as f:
        yaml.dump(all_stats, f, default_flow_style=False)
    
    print(f"\nAggregated results saved to: {args.output}")
    
    # Print formatted table
    table = format_table(all_stats)
    print("\n" + table)
    
    # Also save table as text
    table_file = args.output.replace('.yaml', '_table.txt')
    with open(table_file, 'w') as f:
        f.write(table)
    
    print(f"\nTable saved to: {table_file}")


if __name__ == "__main__":
    main()
