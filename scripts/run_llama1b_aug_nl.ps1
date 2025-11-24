# Run LLaMA 3.2-1B on GSM8k-AUG-NL (Natural Language CoT)
# Paper configuration: Table 6

$seeds = @(42, 43, 44)

foreach ($seed in $seeds) {
    Write-Host "Training LLaMA 3.2-1B on GSM8k-AUG-NL with seed $seed"
    
    python train.py `
        --config configs/llama1b_aug_nl.yaml `
        --seed $seed `
        --wandb `
        --output_dir "checkpoints/llama1b-aug-nl-seed$seed"
    
    Write-Host "Evaluating checkpoint..."
    
    python evaluate.py `
        --checkpoint "checkpoints/llama1b-aug-nl-seed$seed/llama-gsm8k_aug_nl-epoch10" `
        --config configs/llama1b_aug_nl.yaml `
        --datasets gsm8k gsm8k-hard svamp `
        --output "results/llama1b-aug-nl-seed$seed.yaml"
}

Write-Host "All runs completed for LLaMA 3.2-1B on GSM8k-AUG-NL"
