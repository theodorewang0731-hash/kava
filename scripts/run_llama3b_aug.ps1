# Run LLaMA 3.2-3B on GSM8k-AUG
# Paper configuration: Table 6
# Note: Uses different hyperparameters (α₁=20, α₂=2, 5 epochs)

$seeds = @(42, 43, 44)

foreach ($seed in $seeds) {
    Write-Host "Training LLaMA 3.2-3B on GSM8k-AUG with seed $seed"
    
    python train.py `
        --config configs/llama3b_aug.yaml `
        --seed $seed `
        --wandb `
        --output_dir "checkpoints/llama3b-aug-seed$seed"
    
    Write-Host "Evaluating checkpoint..."
    
    python evaluate.py `
        --checkpoint "checkpoints/llama3b-aug-seed$seed/llama-gsm8k_aug-epoch5" `
        --config configs/llama3b_aug.yaml `
        --datasets gsm8k gsm8k-hard svamp `
        --output "results/llama3b-aug-seed$seed.yaml"
}

Write-Host "All runs completed for LLaMA 3.2-3B on GSM8k-AUG"
