"""
分析训练结果并生成报告
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List
import pandas as pd


def find_results_dirs() -> List[Path]:
    """查找所有结果目录"""
    results_dirs = []
    
    # 查找 results/ 下的子目录
    if os.path.exists('results'):
        for item in os.listdir('results'):
            item_path = Path('results') / item
            if item_path.is_dir():
                results_dirs.append(item_path)
    
    return results_dirs


def load_metrics(result_dir: Path) -> Dict:
    """从结果目录加载训练指标"""
    metrics = {
        'config': result_dir.name,
        'train_loss': None,
        'val_loss': None,
        'accuracy': None,
        'epochs': None,
        'best_checkpoint': None
    }
    
    # 查找 trainer_state.json
    trainer_state_path = result_dir / 'trainer_state.json'
    if trainer_state_path.exists():
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)
            if 'log_history' in state:
                history = state['log_history']
                # 提取最后一个 epoch 的指标
                for entry in reversed(history):
                    if 'train_loss' in entry:
                        metrics['train_loss'] = entry['train_loss']
                    if 'eval_loss' in entry:
                        metrics['val_loss'] = entry['eval_loss']
                    if 'eval_accuracy' in entry:
                        metrics['accuracy'] = entry['eval_accuracy']
                    if 'epoch' in entry:
                        metrics['epochs'] = entry['epoch']
                        break
    
    # 查找评估结果
    eval_results_path = result_dir / 'eval_results.json'
    if eval_results_path.exists():
        with open(eval_results_path, 'r') as f:
            eval_data = json.load(f)
            if 'gsm8k_accuracy' in eval_data:
                metrics['accuracy'] = eval_data['gsm8k_accuracy']
    
    # 查找最佳 checkpoint
    checkpoints = list(result_dir.glob('checkpoint-*'))
    if checkpoints:
        # 按数字排序，取最后一个
        checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
        metrics['best_checkpoint'] = checkpoints[-1].name
    
    return metrics


def generate_summary_table(all_metrics: List[Dict]) -> pd.DataFrame:
    """生成汇总表格"""
    df = pd.DataFrame(all_metrics)
    
    # 按配置名称排序
    df = df.sort_values('config')
    
    return df


def generate_markdown_report(df: pd.DataFrame, output_path: str = 'RESULTS_REPORT.md'):
    """生成 Markdown 格式的报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# KAVA 训练结果报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
        
        f.write("## 实验概览\n\n")
        f.write(f"- 总配置数: {len(df)}\n")
        f.write(f"- 完成训练: {df['train_loss'].notna().sum()}\n")
        f.write(f"- 有评估结果: {df['accuracy'].notna().sum()}\n\n")
        
        f.write("## 详细结果\n\n")
        f.write("| 配置 | 训练损失 | 验证损失 | 准确率 | Epochs | 最佳 Checkpoint |\n")
        f.write("|------|----------|----------|--------|--------|----------------|\n")
        
        for _, row in df.iterrows():
            config = row['config']
            train_loss = f"{row['train_loss']:.4f}" if pd.notna(row['train_loss']) else "N/A"
            val_loss = f"{row['val_loss']:.4f}" if pd.notna(row['val_loss']) else "N/A"
            accuracy = f"{row['accuracy']:.2%}" if pd.notna(row['accuracy']) else "N/A"
            epochs = f"{row['epochs']:.0f}" if pd.notna(row['epochs']) else "N/A"
            checkpoint = row['best_checkpoint'] if pd.notna(row['best_checkpoint']) else "N/A"
            
            f.write(f"| {config} | {train_loss} | {val_loss} | {accuracy} | {epochs} | {checkpoint} |\n")
        
        f.write("\n## 性能排名\n\n")
        
        # 按准确率排序
        df_sorted = df.dropna(subset=['accuracy']).sort_values('accuracy', ascending=False)
        if not df_sorted.empty:
            f.write("### 按准确率排序\n\n")
            for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
                f.write(f"{i}. **{row['config']}**: {row['accuracy']:.2%}\n")
        
        f.write("\n## 配置说明\n\n")
        f.write("- `llama1b_aug`: LLaMA-1B + equation-only CoT\n")
        f.write("- `llama1b_aug_nl`: LLaMA-1B + natural language CoT\n")
        f.write("- `llama3b_aug`: LLaMA-3B + equation-only CoT\n")
        f.write("- `qwen05b_aug`: Qwen-0.5B + equation-only CoT\n")
        
        f.write("\n## 训练超参数\n\n")
        f.write("- LoRA: r=128, α=32, dropout=0.1\n")
        f.write("- Latent: M=24 tokens, T=3 iterations\n")
        f.write("- R-KV: λ=0.1\n")
        f.write("- Loss: α₁=10.0 (CODI), α₂=1.0 (KV distillation)\n")
        f.write("- Training: LR=8e-4, batch=128, epochs=10, bf16\n")
    
    print(f"✓ 报告已生成: {output_path}")


def main():
    print("="*80)
    print("KAVA 结果分析工具")
    print("="*80)
    
    # 查找结果目录
    print("\n查找结果目录...")
    results_dirs = find_results_dirs()
    
    if not results_dirs:
        print("⚠ 未找到结果目录")
        print("  请确保训练已完成，结果保存在 results/ 目录下")
        return
    
    print(f"✓ 找到 {len(results_dirs)} 个结果目录\n")
    
    # 加载所有指标
    all_metrics = []
    for result_dir in results_dirs:
        print(f"处理: {result_dir.name}")
        metrics = load_metrics(result_dir)
        all_metrics.append(metrics)
    
    # 生成汇总表
    print("\n生成汇总表...")
    df = generate_summary_table(all_metrics)
    
    # 显示表格
    print("\n" + "="*80)
    print("训练结果汇总")
    print("="*80)
    print(df.to_string(index=False))
    print()
    
    # 保存 CSV
    csv_path = 'results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV 已保存: {csv_path}")
    
    # 生成 Markdown 报告
    print("\n生成 Markdown 报告...")
    generate_markdown_report(df)
    
    # 统计信息
    print("\n" + "="*80)
    print("统计信息")
    print("="*80)
    
    if df['accuracy'].notna().any():
        best_config = df.loc[df['accuracy'].idxmax(), 'config']
        best_acc = df['accuracy'].max()
        print(f"✓ 最佳配置: {best_config} ({best_acc:.2%})")
    
    if df['train_loss'].notna().any():
        avg_train_loss = df['train_loss'].mean()
        print(f"✓ 平均训练损失: {avg_train_loss:.4f}")
    
    if df['val_loss'].notna().any():
        avg_val_loss = df['val_loss'].mean()
        print(f"✓ 平均验证损失: {avg_val_loss:.4f}")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print("\n查看详细报告: RESULTS_REPORT.md")


if __name__ == '__main__':
    main()
