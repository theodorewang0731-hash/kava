"""
Multi-seed experiment runner for KAVA.
Runs training and evaluation with multiple random seeds and aggregates results.
"""

import argparse
import yaml
import subprocess
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np


class MultiSeedRunner:
    """
    Orchestrates multi-seed experiments for statistical significance.
    
    Paper requirement: 3 random seeds with mean ± std reporting
    """
    
    def __init__(
        self,
        config_path: str,
        seeds: List[int] = [42, 43, 44],
        base_output_dir: str = "experiments"
    ):
        """
        Args:
            config_path: Path to base config YAML
            seeds: List of random seeds (default: 42, 43, 44 as per paper)
            base_output_dir: Root directory for outputs
        """
        self.config_path = config_path
        self.seeds = seeds
        self.base_output_dir = base_output_dir
        
        # Load base config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create experiment name
        model_type = self.config['model']['type']
        dataset_name = self.config['dataset']['name'].split('/')[-1]
        self.experiment_name = f"{model_type}_{dataset_name}"
        
        print(f"Initialized MultiSeedRunner for: {self.experiment_name}")
        print(f"Seeds: {self.seeds}")
    
    def run_single_seed(self, seed: int) -> Dict:
        """
        Run training and evaluation for a single seed.
        
        Args:
            seed: Random seed
        
        Returns:
            Dict with results
        """
        print(f"\n{'='*80}")
        print(f"Running seed {seed}")
        print(f"{'='*80}")
        
        # Create output directory for this seed
        output_dir = Path(self.base_output_dir) / self.experiment_name / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # === TRAINING ===
        print(f"\n[1/2] Training with seed {seed}...")
        
        train_cmd = [
            "python", "train.py",
            "--config", self.config_path,
            "--seed", str(seed),
            "--output_dir", str(output_dir),
            "--wandb"  # Enable W&B logging
        ]
        
        try:
            subprocess.run(train_cmd, check=True)
            print(f"✓ Training completed for seed {seed}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Training failed for seed {seed}: {e}")
            return {"seed": seed, "status": "training_failed"}
        
        # === EVALUATION ===
        print(f"\n[2/2] Evaluating with seed {seed}...")
        
        # Find latest checkpoint
        checkpoint_dir = self._find_latest_checkpoint(output_dir)
        if checkpoint_dir is None:
            print(f"✗ No checkpoint found for seed {seed}")
            return {"seed": seed, "status": "no_checkpoint"}
        
        # Evaluate on all datasets
        eval_results = {}
        for dataset_name in ['gsm8k', 'gsm8k-hard', 'svamp']:
            print(f"\n  Evaluating on {dataset_name}...")
            
            results_file = output_dir / f"results_{dataset_name}.yaml"
            
            eval_cmd = [
                "python", "evaluate.py",
                "--checkpoint", str(checkpoint_dir),
                "--config", self.config_path,
                "--datasets", dataset_name,
                "--output", str(results_file)
            ]
            
            try:
                subprocess.run(eval_cmd, check=True)
                
                # Load results
                with open(results_file, 'r') as f:
                    results = yaml.safe_load(f)
                
                eval_results[dataset_name] = results[dataset_name]
                
                print(f"    ✓ {dataset_name}: Accuracy = {results[dataset_name]['accuracy']*100:.2f}%")
            
            except subprocess.CalledProcessError as e:
                print(f"    ✗ Evaluation failed on {dataset_name}: {e}")
                eval_results[dataset_name] = {"accuracy": 0.0, "avg_forward_passes": 0.0}
        
        return {
            "seed": seed,
            "status": "completed",
            "results": eval_results,
            "checkpoint": str(checkpoint_dir)
        }
    
    def _find_latest_checkpoint(self, output_dir: Path) -> Path:
        """Find the latest checkpoint in output directory."""
        checkpoint_dirs = list(output_dir.glob("*-epoch*"))
        
        if not checkpoint_dirs:
            return None
        
        # Sort by modification time and return latest
        checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_dirs[0]
    
    def run_all_seeds(self) -> List[Dict]:
        """
        Run training and evaluation for all seeds.
        
        Returns:
            List of results for each seed
        """
        all_results = []
        
        for seed in self.seeds:
            result = self.run_single_seed(seed)
            all_results.append(result)
            
            # Save intermediate results
            self._save_intermediate_results(all_results)
        
        return all_results
    
    def _save_intermediate_results(self, results: List[Dict]):
        """Save intermediate results to disk."""
        output_dir = Path(self.base_output_dir) / self.experiment_name
        results_file = output_dir / "all_seeds_results.yaml"
        
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
        
        print(f"\n✓ Intermediate results saved to {results_file}")
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict:
        """
        Aggregate results across seeds and compute statistics.
        
        Computes:
        - Mean and std for accuracy
        - Mean and std for forward passes
        - Per-dataset statistics
        
        Args:
            all_results: List of results from each seed
        
        Returns:
            Aggregated statistics
        """
        # Filter successful runs
        successful_results = [r for r in all_results if r['status'] == 'completed']
        
        if not successful_results:
            print("✗ No successful runs to aggregate!")
            return {}
        
        print(f"\n{'='*80}")
        print(f"Aggregating results from {len(successful_results)}/{len(all_results)} seeds")
        print(f"{'='*80}")
        
        # Collect metrics by dataset
        datasets = ['gsm8k', 'gsm8k-hard', 'svamp']
        aggregated = {}
        
        for dataset_name in datasets:
            accuracies = []
            forward_passes = []
            
            for result in successful_results:
                if dataset_name in result['results']:
                    accuracies.append(result['results'][dataset_name]['accuracy'] * 100)
                    forward_passes.append(result['results'][dataset_name]['avg_forward_passes'])
            
            if accuracies:
                aggregated[dataset_name] = {
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'forward_passes_mean': np.mean(forward_passes),
                    'forward_passes_std': np.std(forward_passes),
                    'n_seeds': len(accuracies)
                }
        
        return aggregated
    
    def print_summary_table(self, aggregated: Dict):
        """
        Print summary table in paper format.
        
        Similar to Table 1 in the paper.
        """
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS: {self.experiment_name}")
        print(f"{'='*80}\n")
        
        print(f"{'Dataset':<15} {'Accuracy (%)':<20} {'Forward Passes':<20}")
        print(f"{'-'*15} {'-'*20} {'-'*20}")
        
        for dataset_name in ['gsm8k', 'gsm8k-hard', 'svamp']:
            if dataset_name in aggregated:
                stats = aggregated[dataset_name]
                acc_str = f"{stats['accuracy_mean']:.2f} ± {stats['accuracy_std']:.2f}"
                fp_str = f"{stats['forward_passes_mean']:.1f} ± {stats['forward_passes_std']:.1f}"
                
                print(f"{dataset_name:<15} {acc_str:<20} {fp_str:<20}")
        
        print(f"\n{'='*80}")
        print(f"Results based on {aggregated['gsm8k']['n_seeds']} random seeds")
        print(f"{'='*80}\n")
    
    def save_summary(self, aggregated: Dict):
        """Save aggregated summary to file."""
        output_dir = Path(self.base_output_dir) / self.experiment_name
        
        # Save as YAML
        summary_file = output_dir / "summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(aggregated, f, default_flow_style=False)
        
        # Save as CSV for easy plotting
        csv_file = output_dir / "summary.csv"
        rows = []
        for dataset_name, stats in aggregated.items():
            rows.append({
                'dataset': dataset_name,
                'accuracy_mean': stats['accuracy_mean'],
                'accuracy_std': stats['accuracy_std'],
                'forward_passes_mean': stats['forward_passes_mean'],
                'forward_passes_std': stats['forward_passes_std'],
                'n_seeds': stats['n_seeds']
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        
        print(f"✓ Summary saved to:")
        print(f"  - {summary_file}")
        print(f"  - {csv_file}")
    
    def run_full_experiment(self):
        """
        Run complete multi-seed experiment.
        
        This is the main entry point for running full replication.
        """
        print(f"\n{'#'*80}")
        print(f"# KAVA Multi-Seed Experiment")
        print(f"# Experiment: {self.experiment_name}")
        print(f"# Seeds: {self.seeds}")
        print(f"# Config: {self.config_path}")
        print(f"{'#'*80}\n")
        
        # Run all seeds
        all_results = self.run_all_seeds()
        
        # Aggregate results
        aggregated = self.aggregate_results(all_results)
        
        # Print summary
        self.print_summary_table(aggregated)
        
        # Save summary
        self.save_summary(aggregated)
        
        print(f"\n{'#'*80}")
        print(f"# Experiment completed successfully!")
        print(f"{'#'*80}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed KAVA experiments")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds to use (default: 42 43 44)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Base output directory"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create runner
    runner = MultiSeedRunner(
        config_path=args.config,
        seeds=args.seeds,
        base_output_dir=args.output_dir
    )
    
    # Run full experiment
    runner.run_full_experiment()


if __name__ == "__main__":
    main()
