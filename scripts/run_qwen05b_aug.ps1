# Run Qwen2.5-0.5B on GSM8k-AUG
# Paper configuration: Table 6

$seeds = @(42, 43, 44)

foreach ($seed in $seeds) {
    Write-Host "Training Qwen2.5-0.5B on GSM8k-AUG with seed $seed"
    
    python train.py `
        --config configs/qwen05b_aug.yaml `
        --seed $seed `
        --wandb `
        --output_dir "checkpoints/qwen05b-aug-seed$seed"
    
    Write-Host "Evaluating checkpoint..."
    
    python evaluate.py `
        --checkpoint "checkpoints/qwen05b-aug-seed$seed/qwen-gsm8k_aug-epoch10" `
        --config configs/qwen05b_aug.yaml `
        --datasets gsm8k gsm8k-hard svamp `
        --output "results/qwen05b-aug-seed$seed.yaml"
}

Write-Host "All runs completed for Qwen2.5-0.5B on GSM8k-AUG"
