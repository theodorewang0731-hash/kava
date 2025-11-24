# Multi-Seed Experiment Runner
# Runs training and evaluation for all Table 6 configurations with statistical significance

# LLaMA 3.2-1B on GSM8k-AUG
Write-Host "=== Experiment 1/4: LLaMA 3.2-1B + GSM8k-AUG ===" -ForegroundColor Cyan
python run_multi_seed.py `
    --config configs/llama1b_aug.yaml `
    --seeds 42 43 44 `
    --output_dir experiments

# LLaMA 3.2-1B on GSM8k-AUG-NL
Write-Host "`n=== Experiment 2/4: LLaMA 3.2-1B + GSM8k-AUG-NL ===" -ForegroundColor Cyan
python run_multi_seed.py `
    --config configs/llama1b_aug_nl.yaml `
    --seeds 42 43 44 `
    --output_dir experiments

# Qwen2.5-0.5B on GSM8k-AUG
Write-Host "`n=== Experiment 3/4: Qwen2.5-0.5B + GSM8k-AUG ===" -ForegroundColor Cyan
python run_multi_seed.py `
    --config configs/qwen05b_aug.yaml `
    --seeds 42 43 44 `
    --output_dir experiments

# LLaMA 3.2-3B on GSM8k-AUG
Write-Host "`n=== Experiment 4/4: LLaMA 3.2-3B + GSM8k-AUG ===" -ForegroundColor Cyan
python run_multi_seed.py `
    --config configs/llama3b_aug.yaml `
    --seeds 42 43 44 `
    --output_dir experiments

Write-Host "`n=== All experiments completed! ===" -ForegroundColor Green
Write-Host "Results saved to: experiments/" -ForegroundColor Green
