# Run all experiments to replicate paper Table 1 and Table 2
# This will train all model/dataset combinations with 3 seeds each

Write-Host "============================================"
Write-Host "KAVA Full Paper Replication"
Write-Host "Reproducing Table 1 (Accuracy) and Table 2 (Forward Passes)"
Write-Host "============================================"

# LLaMA 3.2-1B experiments
Write-Host "`n[1/4] Running LLaMA 3.2-1B on GSM8k-AUG..."
.\scripts\run_llama1b_aug.ps1

Write-Host "`n[2/4] Running LLaMA 3.2-1B on GSM8k-AUG-NL..."
.\scripts\run_llama1b_aug_nl.ps1

# Qwen2.5-0.5B experiments
Write-Host "`n[3/4] Running Qwen2.5-0.5B on GSM8k-AUG..."
.\scripts\run_qwen05b_aug.ps1

# LLaMA 3.2-3B experiments
Write-Host "`n[4/4] Running LLaMA 3.2-3B on GSM8k-AUG..."
.\scripts\run_llama3b_aug.ps1

Write-Host "`n============================================"
Write-Host "All experiments completed!"
Write-Host "Results saved in results/ directory"
Write-Host "============================================"

# Aggregate results
Write-Host "`nAggregating results..."
python scripts/aggregate_results.py --results_dir results/ --output results/summary.yaml
