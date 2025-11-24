"""
Aggregate results from multiple experiments for paper-ready tables.
Supports multi-dataset and multi-seed aggregation.
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


def load_experiment_summary(experiment_dir: Path) -> Dict:
    """Load summary.yaml from an experiment directory."""
    summary_file = experiment_dir / "summary.yaml"
    
    if not summary_file.exists():
        print(f"Warning: {summary_file} not found")
        return None
    
    with open(summary_file, 'r') as f:
        return yaml.safe_load(f)


def load_seed_results(experiment_dir: Path, dataset_name: str = 'gsm8k') -> List[Dict]:
    """
    Load results from all seeds for a specific dataset.
    
    Args:
        experiment_dir: Experiment directory containing seed_XX subdirs
        dataset_name: Dataset to load results for
    
    Returns:
        List of result dictionaries from each seed
    """
    seed_dirs = sorted(experiment_dir.glob("seed_*"))
    results = []
    
    for seed_dir in seed_dirs:
        # Try JSON first (more reliable), then YAML
        json_file = seed_dir / f"results_{dataset_name}.json"
        yaml_file = seed_dir / f"results_{dataset_name}.yaml"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                if dataset_name in data:
                    results.append(data[dataset_name])
        elif yaml_file.exists():
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                if dataset_name in data:
                    results.append(data[dataset_name])
    
    return results


def aggregate_multi_seed(results: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple seeds.
    
    Args:
        results: List of result dicts from different seeds
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if not results:
        return {}
    
    metrics = {}
    
    # Aggregate accuracy
    accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
    if accuracies:
        metrics['accuracy_mean'] = np.mean(accuracies)
        metrics['accuracy_std'] = np.std(accuracies)
    
    # Aggregate forward passes
    forward_passes = [r['avg_forward_passes'] for r in results if 'avg_forward_passes' in r]
    if forward_passes:
        metrics['forward_passes_mean'] = np.mean(forward_passes)
        metrics['forward_passes_std'] = np.std(forward_passes)
    
    # Track number of seeds
    metrics['n_seeds'] = len(results)
    
    return metrics


def create_paper_table(experiments: Dict[str, Dict], datasets: List[str] = None) -> pd.DataFrame:
    """
    Create paper-ready table similar to Table 1.
    
    Args:
        experiments: Dict mapping experiment names to their summaries
        datasets: List of dataset names to include (default: ['gsm8k', 'gsm8k-hard', 'svamp'])
    
    Returns:
        DataFrame with formatted results
    """
    if datasets is None:
        datasets = ['gsm8k', 'gsm8k-hard', 'svamp']
    
    rows = []
    
    for exp_name, summary in experiments.items():
        # Parse model and dataset from experiment name
        if 'llama1b_gsm8k-aug-nl' in exp_name:
            model = 'LLaMA 3.2-1B'
            dataset = 'GSM8k-AUG-NL'
        elif 'llama1b_gsm8k-aug' in exp_name:
            model = 'LLaMA 3.2-1B'
            dataset = 'GSM8k-AUG'
        elif 'llama3b' in exp_name:
            model = 'LLaMA 3.2-3B'
            dataset = 'GSM8k-AUG'
        elif 'qwen05b' in exp_name:
            model = 'Qwen2.5-0.5B'
            dataset = 'GSM8k-AUG'
        else:
            model = exp_name
            dataset = 'Unknown'
        
        row = {
            'Model': model,
            'Training Data': dataset
        }
        
        # Add metrics for each evaluation dataset
        for eval_dataset in datasets:
            if eval_dataset in summary:
                stats = summary[eval_dataset]
                acc_mean = stats.get('accuracy_mean', 0) * 100  # Convert to percentage
                acc_std = stats.get('accuracy_std', 0) * 100
                fp_mean = stats.get('forward_passes_mean', 0)
                fp_std = stats.get('forward_passes_std', 0)
                
                # Format with mean ± std
                row[f'{eval_dataset.upper()} Acc (%)'] = f"{acc_mean:.2f} ± {acc_std:.2f}"
                row[f'{eval_dataset.upper()} FP'] = f"{fp_mean:.1f} ± {fp_std:.1f}"
            else:
                row[f'{eval_dataset.upper()} Acc (%)'] = "N/A"
                row[f'{eval_dataset.upper()} FP'] = "N/A"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment results")
    
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Directory containing all experiments"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="paper_results.csv",
        help="Output CSV file for aggregated results"
    )
    
    args = parser.parse_args()
    
    # Find all experiment directories
    experiments_dir = Path(args.experiments_dir)
    
    if not experiments_dir.exists():
        print(f"Error: {experiments_dir} does not exist")
        return
    
    # Load all experiment summaries
    experiments = {}
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            summary = load_experiment_summary(exp_dir)
            if summary:
                experiments[exp_dir.name] = summary
    
    if not experiments:
        print("No experiment summaries found!")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp_name in experiments.keys():
        print(f"  - {exp_name}")
    
    # Create paper table
    table = create_paper_table(experiments)
    
    # Print to console
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (Paper Format)")
    print("="*80 + "\n")
    print(table.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    # Save to CSV
    table.to_csv(args.output, index=False)
    print(f"✓ Results saved to {args.output}")
    
    # Also save as LaTeX for paper
    latex_file = args.output.replace('.csv', '.tex')
    with open(latex_file, 'w') as f:
        f.write(table.to_latex(index=False))
    print(f"✓ LaTeX table saved to {latex_file}")


if __name__ == "__main__":
    main()
