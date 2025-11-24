#!/usr/bin/env python3
"""
KAVA 训练结果验证和可视化脚本
用于验证训练正确性、生成图表和对比论文结果
"""

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class KAVAResultValidator:
    """验证和可视化 KAVA 训练结果"""
    
    # 论文中的基准结果 (Table 2)
    PAPER_RESULTS = {
        'Llama-3.2-1B': {
            'GSM8K_EM': 41.6,
            'GSM8K_F1': 48.9,
            'GSM8K-Hard_EM': 28.4,
            'SVAMP_EM': 52.3
        },
        'Llama-3.2-3B': {
            'GSM8K_EM': 51.6,
            'GSM8K_F1': 58.2,
            'GSM8K-Hard_EM': 35.1,
            'SVAMP_EM': 61.2
        },
        'Qwen2.5-0.5B': {
            'GSM8K_EM': 32.9,
            'GSM8K_F1': 41.3,
            'GSM8K-Hard_EM': 21.7,
            'SVAMP_EM': 43.6
        }
    }
    
    def __init__(self, results_dir: str = "outputs"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_results(self) -> pd.DataFrame:
        """加载训练结果"""
        csv_path = self.results_dir / "aggregated_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"结果文件不存在: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"✓ 加载了 {len(df)} 个训练结果")
        return df
    
    def validate_results(self, df: pd.DataFrame) -> Dict:
        """验证结果完整性"""
        print("\n=== 验证训练结果 ===")
        
        validation = {
            'total_runs': len(df),
            'completed': len(df[df['Status'] == 'COMPLETED']),
            'failed': len(df[df['Status'] != 'COMPLETED']),
            'configs': df['Config'].unique().tolist(),
            'seeds': df['Seed'].unique().tolist(),
            'issues': []
        }
        
        # 检查完整性
        expected_configs = ['llama1b_aug', 'llama1b_aug_nl', 'llama3b_aug', 'qwen05b_aug']
        expected_seeds = [42, 123, 456]
        
        for config in expected_configs:
            for seed in expected_seeds:
                match = df[(df['Config'] == config) & (df['Seed'] == seed)]
                if len(match) == 0:
                    validation['issues'].append(f"缺失: {config} seed={seed}")
                elif match.iloc[0]['Status'] != 'COMPLETED':
                    validation['issues'].append(f"未完成: {config} seed={seed}")
        
        # 打印验证结果
        print(f"总运行数: {validation['total_runs']}")
        print(f"已完成: {validation['completed']}/{len(expected_configs) * len(expected_seeds)}")
        print(f"失败: {validation['failed']}")
        
        if validation['issues']:
            print("\n⚠️ 发现问题:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        else:
            print("\n✅ 所有训练任务成功完成！")
        
        return validation
    
    def compare_with_paper(self, df: pd.DataFrame):
        """对比论文结果"""
        print("\n=== 对比论文结果 ===\n")
        
        # 映射配置名到模型名
        config_to_model = {
            'llama1b_aug': 'Llama-3.2-1B',
            'llama1b_aug_nl': 'Llama-3.2-1B',
            'llama3b_aug': 'Llama-3.2-3B',
            'qwen05b_aug': 'Qwen2.5-0.5B'
        }
        
        comparison = []
        
        for config, model_name in config_to_model.items():
            if model_name not in self.PAPER_RESULTS:
                continue
            
            config_df = df[df['Config'] == config]
            if len(config_df) == 0:
                continue
            
            # 计算平均值
            mean_em = config_df['EM'].mean()
            mean_f1 = config_df['F1'].mean()
            std_em = config_df['EM'].std()
            std_f1 = config_df['F1'].std()
            
            # 对比论文
            paper_em = self.PAPER_RESULTS[model_name]['GSM8K_EM']
            paper_f1 = self.PAPER_RESULTS[model_name]['GSM8K_F1']
            
            diff_em = mean_em - paper_em
            diff_f1 = mean_f1 - paper_f1
            
            comparison.append({
                'Config': config,
                'Model': model_name,
                'Our_EM': f"{mean_em:.1f}±{std_em:.1f}",
                'Paper_EM': f"{paper_em:.1f}",
                'Diff_EM': f"{diff_em:+.1f}",
                'Our_F1': f"{mean_f1:.1f}±{std_f1:.1f}",
                'Paper_F1': f"{paper_f1:.1f}",
                'Diff_F1': f"{diff_f1:+.1f}",
                'Status': '✓' if abs(diff_em) < 2.0 else '⚠️'
            })
            
            print(f"{model_name} ({config}):")
            print(f"  EM: {mean_em:.1f}±{std_em:.1f} vs {paper_em:.1f} (论文) [{diff_em:+.1f}]")
            print(f"  F1: {mean_f1:.1f}±{std_f1:.1f} vs {paper_f1:.1f} (论文) [{diff_f1:+.1f}]")
            print(f"  状态: {'✅ 复现成功' if abs(diff_em) < 2.0 else '⚠️ 有偏差'}\n")
        
        return pd.DataFrame(comparison)
    
    def plot_results(self, df: pd.DataFrame):
        """生成可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        # 1. 所有配置的 EM/F1 对比
        self._plot_all_metrics(df)
        
        # 2. 多种子稳定性分析
        self._plot_seed_variance(df)
        
        # 3. 与论文对比
        self._plot_paper_comparison(df)
        
        # 4. 训练收敛曲线（如果有日志）
        self._plot_training_curves()
        
        print(f"\n✓ 图表已保存到: {self.figures_dir}/")
    
    def _plot_all_metrics(self, df: pd.DataFrame):
        """所有配置的指标对比"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # EM 对比
        df_completed = df[df['Status'] == 'COMPLETED']
        configs = df_completed.groupby('Config')['EM'].mean().sort_values(ascending=False)
        
        ax1.bar(range(len(configs)), configs.values, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs.index, rotation=45, ha='right')
        ax1.set_ylabel('Exact Match (%)')
        ax1.set_title('GSM8K Exact Match by Configuration')
        ax1.grid(axis='y', alpha=0.3)
        
        # F1 对比
        f1_scores = df_completed.groupby('Config')['F1'].mean().sort_values(ascending=False)
        ax2.bar(range(len(f1_scores)), f1_scores.values, color='coral', alpha=0.7)
        ax2.set_xticks(range(len(f1_scores)))
        ax2.set_xticklabels(f1_scores.index, rotation=45, ha='right')
        ax2.set_ylabel('F1 Score (%)')
        ax2.set_title('GSM8K F1 Score by Configuration')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'all_metrics.png', dpi=300, bbox_inches='tight')
        print("  ✓ all_metrics.png")
        plt.close()
    
    def _plot_seed_variance(self, df: pd.DataFrame):
        """多种子方差分析"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_completed = df[df['Status'] == 'COMPLETED']
        configs = df_completed['Config'].unique()
        
        x = np.arange(len(configs))
        width = 0.35
        
        em_means = [df_completed[df_completed['Config'] == c]['EM'].mean() for c in configs]
        em_stds = [df_completed[df_completed['Config'] == c]['EM'].std() for c in configs]
        f1_means = [df_completed[df_completed['Config'] == c]['F1'].mean() for c in configs]
        f1_stds = [df_completed[df_completed['Config'] == c]['F1'].std() for c in configs]
        
        ax.bar(x - width/2, em_means, width, yerr=em_stds, label='EM', 
               color='steelblue', alpha=0.7, capsize=5)
        ax.bar(x + width/2, f1_means, width, yerr=f1_stds, label='F1',
               color='coral', alpha=0.7, capsize=5)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Score (%)')
        ax.set_title('Multi-Seed Results (Mean ± Std)')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'seed_variance.png', dpi=300, bbox_inches='tight')
        print("  ✓ seed_variance.png")
        plt.close()
    
    def _plot_paper_comparison(self, df: pd.DataFrame):
        """与论文结果对比"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        config_to_model = {
            'llama1b_aug': 'Llama-3.2-1B',
            'llama3b_aug': 'Llama-3.2-3B',
            'qwen05b_aug': 'Qwen2.5-0.5B'
        }
        
        models = []
        our_em = []
        paper_em = []
        
        for config, model in config_to_model.items():
            if model in self.PAPER_RESULTS:
                config_df = df[(df['Config'] == config) & (df['Status'] == 'COMPLETED')]
                if len(config_df) > 0:
                    models.append(model)
                    our_em.append(config_df['EM'].mean())
                    paper_em.append(self.PAPER_RESULTS[model]['GSM8K_EM'])
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, paper_em, width, label='Paper', color='lightcoral', alpha=0.7)
        ax.bar(x + width/2, our_em, width, label='Ours', color='steelblue', alpha=0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('GSM8K EM (%)')
        ax.set_title('Reproduction vs Paper Results')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 添加差异标注
        for i, (p, o) in enumerate(zip(paper_em, our_em)):
            diff = o - p
            ax.text(i, max(p, o) + 1, f'{diff:+.1f}', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'paper_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ paper_comparison.png")
        plt.close()
    
    def _plot_training_curves(self):
        """绘制训练曲线（如果有日志）"""
        # 这里可以解析训练日志绘制 loss 曲线
        # 暂时跳过，因为需要解析日志文件
        pass
    
    def generate_report(self, df: pd.DataFrame, validation: Dict, comparison: pd.DataFrame):
        """生成 Markdown 报告"""
        report_path = self.results_dir / "REPRODUCTION_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# KAVA 论文复现报告\n\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
            
            f.write("## 1. 训练完成情况\n\n")
            f.write(f"- 总任务数: {validation['total_runs']}\n")
            f.write(f"- 已完成: {validation['completed']}\n")
            f.write(f"- 失败: {validation['failed']}\n")
            f.write(f"- 配置: {', '.join(validation['configs'])}\n")
            f.write(f"- 种子: {', '.join(map(str, validation['seeds']))}\n\n")
            
            if validation['issues']:
                f.write("### ⚠️ 问题\n\n")
                for issue in validation['issues']:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            f.write("## 2. 与论文结果对比\n\n")
            f.write(comparison.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## 3. 详细结果\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## 4. 可视化图表\n\n")
            f.write("![All Metrics](figures/all_metrics.png)\n\n")
            f.write("![Seed Variance](figures/seed_variance.png)\n\n")
            f.write("![Paper Comparison](figures/paper_comparison.png)\n\n")
            
            f.write("## 5. 结论\n\n")
            
            max_diff = comparison['Diff_EM'].apply(lambda x: abs(float(x))).max()
            if max_diff < 2.0:
                f.write("✅ **复现成功！** 所有模型的结果与论文偏差在 ±2% 以内。\n")
            elif max_diff < 5.0:
                f.write("⚠️ **基本复现成功**，部分模型有 2-5% 的偏差，可能是由于:\n")
                f.write("- 随机种子差异\n")
                f.write("- 训练超参数微调\n")
                f.write("- 硬件/软件环境差异\n")
            else:
                f.write("❌ **复现存在较大偏差**，需要进一步检查:\n")
                f.write("- 数据预处理\n")
                f.write("- 模型配置\n")
                f.write("- 训练参数\n")
        
        print(f"\n✓ 报告已保存到: {report_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("KAVA 训练结果验证和可视化")
    print("=" * 60)
    
    validator = KAVAResultValidator()
    
    try:
        # 1. 加载结果
        df = validator.load_results()
        
        # 2. 验证完整性
        validation = validator.validate_results(df)
        
        # 3. 对比论文
        comparison = validator.compare_with_paper(df)
        
        # 4. 生成图表
        validator.plot_results(df)
        
        # 5. 生成报告
        validator.generate_report(df, validation, comparison)
        
        print("\n" + "=" * 60)
        print("✅ 验证和可视化完成！")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请确保训练已完成并运行了 collect_results.sh")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
