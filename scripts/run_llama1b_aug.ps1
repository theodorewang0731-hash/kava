# Run LLaMA 3.2-1B on GSM8k-AUG (Equation-only CoT)
# Paper configuration: Table 6

# Run with 3 different seeds as per paper
$seeds = @(42, 43, 44)

foreach ($seed in $seeds) {
    Write-Host "Training LLaMA 3.2-1B on GSM8k-AUG with seed $seed"
    
    python train.py `
        --config configs/llama1b_aug.yaml `
        --seed $seed `
        --wandb `
        --output_dir "checkpoints/llama1b-aug-seed$seed"
    
    Write-Host "Evaluating checkpoint..."
    
    python evaluate.py `
        --checkpoint "checkpoints/llama1b-aug-seed$seed/llama-gsm8k_aug-epoch10" `
        --config configs/llama1b_aug.yaml `
        --datasets gsm8k gsm8k-hard svamp `
        --output "results/llama1b-aug-seed$seed.yaml"
}

Write-Host "All runs completed for LLaMA 3.2-1B on GSM8k-AUG"
