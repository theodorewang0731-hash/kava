#!/usr/bin/env python3
"""
Format KAVA experimental results into paper-ready tables
Generates LaTeX tables (Table 1 & 2) and CSV for Excel/plotting
"""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_aggregated_results(json_path: str) -> dict:
    """Load aggregated results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def format_value(mean: float, std: float, decimals: int = 1) -> str:
    """
    Format mean±std for display.
    
    Args:
        mean: Mean value
        std: Standard deviation
        decimals: Decimal places
    
    Returns:
        Formatted string like "56.5 (0.4)"
    """
    if mean is None or std is None:
        return "-"
    return f"{mean:.{decimals}f} ({std:.{decimals}f})"


def generate_latex_table_accuracy(results: Dict[str, dict]) -> str:
    """
    Generate LaTeX Table 1: Test Accuracy
    
    Format:
    Method | GSM8k | GSM8k-Hard | SVAMP
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Test accuracy (\%) on GSM8k, GSM8k-Hard, and SVAMP with mean and standard deviation over 3 seeds.}")
    lines.append(r"\label{tab:accuracy}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Dataset & GSM8k & GSM8k-Hard & SVAMP \\")
    lines.append(r"\midrule")
    
    # Model order
    models = [
        ('llama1b', 'LLaMA-3.2-1B'),
        ('qwen05b', 'Qwen2.5-0.5B'),
        ('llama3b', 'LLaMA-3.2-3B')
    ]
    
    for model_key, model_name in models:
        # AUG row
        aug_key = f"{model_key}_aug"
        if aug_key in results:
            row = f"{model_name} & GSM8k-AUG"
            for dataset in ['gsm8k', 'gsm8k-hard', 'svamp']:
                if dataset in results[aug_key]['datasets']:
                    ds = results[aug_key]['datasets'][dataset]
                    mean = ds.get('accuracy_mean', 0) * 100  # Convert to percentage
                    std = ds.get('accuracy_std', 0) * 100
                    row += f" & {format_value(mean, std)}"
                else:
                    row += " & -"
            row += r" \\"
            lines.append(row)
        
        # AUG-NL row
        nl_key = f"{model_key}_aug_nl"
        if nl_key in results:
            row = f"{model_name} & GSM8k-AUG-NL"
            for dataset in ['gsm8k', 'gsm8k-hard', 'svamp']:
                if dataset in results[nl_key]['datasets']:
                    ds = results[nl_key]['datasets'][dataset]
                    mean = ds.get('accuracy_mean', 0) * 100
                    std = ds.get('accuracy_std', 0) * 100
                    row += f" & {format_value(mean, std)}"
                else:
                    row += " & -"
            row += r" \\"
            lines.append(row)
        
        if model_key != models[-1][0]:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_latex_table_efficiency(results: Dict[str, dict]) -> str:
    """
    Generate LaTeX Table 2: Efficiency (Forward Passes)
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Average number of forward passes per question with mean and standard deviation over 3 seeds.}")
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Dataset & GSM8k & GSM8k-Hard & SVAMP \\")
    lines.append(r"\midrule")
    
    models = [
        ('llama1b', 'LLaMA-3.2-1B'),
        ('qwen05b', 'Qwen2.5-0.5B'),
        ('llama3b', 'LLaMA-3.2-3B')
    ]
    
    for model_key, model_name in models:
        # AUG row
        aug_key = f"{model_key}_aug"
        if aug_key in results:
            row = f"{model_name} & GSM8k-AUG"
            for dataset in ['gsm8k', 'gsm8k-hard', 'svamp']:
                if dataset in results[aug_key]['datasets']:
                    ds = results[aug_key]['datasets'][dataset]
                    mean = ds.get('forward_passes_mean', 0)
                    std = ds.get('forward_passes_std', 0)
                    row += f" & {format_value(mean, std, decimals=1)}"
                else:
                    row += " & -"
            row += r" \\"
            lines.append(row)
        
        # AUG-NL row
        nl_key = f"{model_key}_aug_nl"
        if nl_key in results:
            row = f"{model_name} & GSM8k-AUG-NL"
            for dataset in ['gsm8k', 'gsm8k-hard', 'svamp']:
                if dataset in results[nl_key]['datasets']:
                    ds = results[nl_key]['datasets'][dataset]
                    mean = ds.get('forward_passes_mean', 0)
                    std = ds.get('forward_passes_std', 0)
                    row += f" & {format_value(mean, std, decimals=1)}"
                else:
                    row += " & -"
            row += r" \\"
            lines.append(row)
        
        if model_key != models[-1][0]:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_csv(results: Dict[str, dict], output_path: str):
    """Generate CSV file with all results"""
    rows = []
    headers = [
        'Model', 'Dataset', 'Eval_Dataset',
        'Accuracy_Mean', 'Accuracy_Std',
        'Forward_Passes_Mean', 'Forward_Passes_Std',
        'Num_Seeds'
    ]
    rows.append(headers)
    
    for config_key, result in sorted(results.items()):
        # Parse config key (e.g., "llama1b_aug")
        parts = config_key.split('_')
        if len(parts) >= 2:
            model = parts[0]
            dataset = '_'.join(parts[1:])
        else:
            model = config_key
            dataset = "unknown"
        
        num_seeds = result.get('num_seeds', 3)
        
        # Add row for each evaluation dataset
        for eval_dataset, ds_results in result.get('datasets', {}).items():
            row = [
                model,
                dataset,
                eval_dataset,
                f"{ds_results.get('accuracy_mean', 0):.4f}",
                f"{ds_results.get('accuracy_std', 0):.4f}",
                f"{ds_results.get('forward_passes_mean', 0):.2f}",
                f"{ds_results.get('forward_passes_std', 0):.2f}",
                str(num_seeds)
            ]
            rows.append(row)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"✓ CSV saved to {output_path}")


def collect_results(input_dir: Path) -> Dict[str, dict]:
    """
    Collect all aggregated_results.json files from input directory.
    
    Args:
        input_dir: Directory containing experiment outputs
    
    Returns:
        Dictionary mapping config names to results
    """
    results = {}
    
    # Search for aggregated_results.json files
    for json_file in input_dir.rglob('aggregated_results.json'):
        # Infer config name from parent directory
        # e.g., outputs/llama1b_aug_multi_seed/aggregated_results.json
        parent_name = json_file.parent.name
        
        # Remove common suffixes
        config_name = parent_name.replace('_multi_seed', '').replace('_experiments', '')
        
        try:
            result = load_aggregated_results(str(json_file))
            results[config_name] = result
            print(f"✓ Loaded {config_name}")
        except Exception as e:
            print(f"✗ Failed to load {json_file}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Format KAVA results into LaTeX tables and CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format all results in outputs/ directory
  python format_results.py --input_dir outputs/
  
  # Specify custom output files
  python format_results.py --input_dir outputs/ \\
      --output_latex paper_tables.tex \\
      --output_csv results.csv
  
  # Only generate LaTeX
  python format_results.py --input_dir outputs/ --latex_only
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing experimental results')
    parser.add_argument('--output_latex', type=str, default='kava_tables.tex',
                        help='Output LaTeX file (default: kava_tables.tex)')
    parser.add_argument('--output_csv', type=str, default='kava_results.csv',
                        help='Output CSV file (default: kava_results.csv)')
    parser.add_argument('--latex_only', action='store_true',
                        help='Only generate LaTeX tables')
    parser.add_argument('--csv_only', action='store_true',
                        help='Only generate CSV file')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Collect all results
    print("\n" + "="*70)
    print("Collecting experimental results...")
    print("="*70 + "\n")
    
    results = collect_results(input_dir)
    
    if not results:
        print(f"❌ No aggregated_results.json files found in {input_dir}")
        print("\nExpected structure:")
        print("  outputs/")
        print("    llama1b_aug_multi_seed/")
        print("      aggregated_results.json")
        print("    qwen05b_aug_multi_seed/")
        print("      aggregated_results.json")
        sys.exit(1)
    
    print(f"\n✓ Found {len(results)} result files\n")
    
    # Generate LaTeX tables
    if not args.csv_only:
        print("="*70)
        print("Generating LaTeX tables...")
        print("="*70 + "\n")
        
        table_accuracy = generate_latex_table_accuracy(results)
        table_efficiency = generate_latex_table_efficiency(results)
        
        with open(args.output_latex, 'w') as f:
            f.write("% KAVA Results - Generated by format_results.py\n")
            f.write("% Table 1: Test Accuracy\n\n")
            f.write(table_accuracy)
            f.write("\n\n")
            f.write("% Table 2: Efficiency (Forward Passes)\n\n")
            f.write(table_efficiency)
            f.write("\n")
        
        print(f"✓ LaTeX tables saved to {args.output_latex}")
        
        # Print preview
        print("\n" + "="*70)
        print("LaTeX Preview (Table 1 - first 10 lines):")
        print("="*70)
        for line in table_accuracy.split('\n')[:10]:
            print(line)
        print("...")
    
    # Generate CSV
    if not args.latex_only:
        print("\n" + "="*70)
        print("Generating CSV...")
        print("="*70 + "\n")
        
        generate_csv(results, args.output_csv)
    
    # Summary
    print("\n" + "="*70)
    print("✅ All outputs generated successfully!")
    print("="*70)
    
    if not args.csv_only:
        print(f"\nLaTeX tables: {args.output_latex}")
        print("  → Copy to your paper with: \\input{" + args.output_latex + "}")
    
    if not args.latex_only:
        print(f"\nCSV file: {args.output_csv}")
        print("  → Open in Excel or use for plotting")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
