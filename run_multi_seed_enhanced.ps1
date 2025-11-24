# Enhanced Multi-Seed Experiment Runner
# Runs training and evaluation with multiple seeds and aggregates results
# Usage: .\run_multi_seed_enhanced.ps1 -Config "llama1b_aug" -Seeds 42,43,44

param(
    [Parameter(Mandatory=$true)]
    [string]$Config,  # Config name without .yaml (e.g., "llama1b_aug")
    
    [Parameter(Mandatory=$false)]
    [int[]]$Seeds = @(42, 43, 44),  # Random seeds
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "experiments",  # Base output directory
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTraining,  # Skip training if checkpoints exist
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipEvaluation,  # Skip evaluation
    
    [Parameter(Mandatory=$false)]
    [switch]$UseWandB  # Enable W&B logging
)

# Color output functions
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$msg) Write-ColorOutput $msg "Green" }
function Write-Error { param([string]$msg) Write-ColorOutput $msg "Red" }
function Write-Warning { param([string]$msg) Write-ColorOutput $msg "Yellow" }
function Write-Info { param([string]$msg) Write-ColorOutput $msg "Cyan" }

# Main script
$ConfigPath = "configs/$Config.yaml"
$ExperimentName = $Config.Replace("_", "-")
$ExperimentDir = "$OutputDir/$ExperimentName"

Write-ColorOutput "`n$('='*80)" "Cyan"
Write-ColorOutput "KAVA Multi-Seed Experiment Runner" "Cyan"
Write-ColorOutput "$('='*80)`n" "Cyan"

Write-Info "Configuration:"
Write-Host "  Config file: $ConfigPath"
Write-Host "  Seeds: $($Seeds -join ', ')"
Write-Host "  Output dir: $ExperimentDir"
Write-Host "  Skip training: $SkipTraining"
Write-Host "  Skip evaluation: $SkipEvaluation"
Write-Host "  Use W&B: $UseWandB"
Write-Host ""

# Check if config exists
if (-not (Test-Path $ConfigPath)) {
    Write-Error "Config file not found: $ConfigPath"
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Force -Path $ExperimentDir | Out-Null

# Track results
$AllResults = @()

# Run for each seed
foreach ($Seed in $Seeds) {
    Write-ColorOutput "`n$('='*80)" "Yellow"
    Write-ColorOutput "Running Seed: $Seed" "Yellow"
    Write-ColorOutput "$('='*80)`n" "Yellow"
    
    $SeedDir = "$ExperimentDir/seed_$Seed"
    
    # === TRAINING ===
    if (-not $SkipTraining) {
        Write-Info "[1/2] Training with seed $Seed..."
        
        $TrainArgs = @(
            "train.py",
            "--config", $ConfigPath,
            "--seed", $Seed,
            "--output_dir", $SeedDir
        )
        
        if ($UseWandB) {
            $TrainArgs += "--wandb"
        }
        
        Write-Host "Command: python $($TrainArgs -join ' ')" -ForegroundColor Gray
        
        $TrainStart = Get-Date
        & python $TrainArgs
        $TrainEnd = Get-Date
        $TrainDuration = ($TrainEnd - $TrainStart).TotalMinutes
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "✓ Training completed in $([math]::Round($TrainDuration, 1)) minutes"
        } else {
            Write-Error "✗ Training failed for seed $Seed"
            continue
        }
    } else {
        Write-Warning "Skipping training (--SkipTraining flag set)"
    }
    
    # Find checkpoint
    $Checkpoints = Get-ChildItem -Path $SeedDir -Directory -Filter "checkpoint-*" | Sort-Object Name -Descending
    
    if ($Checkpoints.Count -eq 0) {
        Write-Error "✗ No checkpoint found in $SeedDir"
        continue
    }
    
    $LatestCheckpoint = $Checkpoints[0].FullName
    Write-Info "Using checkpoint: $($Checkpoints[0].Name)"
    
    # === EVALUATION ===
    if (-not $SkipEvaluation) {
        Write-Info "`n[2/2] Evaluating with seed $Seed..."
        
        $Datasets = @("gsm8k", "gsm8k-hard", "svamp")
        $SeedResults = @{
            seed = $Seed
            checkpoint = $LatestCheckpoint
            results = @{}
        }
        
        foreach ($Dataset in $Datasets) {
            Write-Host "`n  Evaluating on $Dataset..." -ForegroundColor Gray
            
            $OutputFile = "$SeedDir/results_$Dataset.yaml"
            
            $EvalArgs = @(
                "evaluate.py",
                "--checkpoint", $LatestCheckpoint,
                "--config", $ConfigPath,
                "--datasets", $Dataset,
                "--output", $OutputFile
            )
            
            $EvalStart = Get-Date
            & python $EvalArgs 2>&1 | Out-Null
            $EvalEnd = Get-Date
            $EvalDuration = ($EvalEnd - $EvalStart).TotalSeconds
            
            if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputFile)) {
                # Load results
                $JsonFile = $OutputFile.Replace(".yaml", ".json")
                if (Test-Path $JsonFile) {
                    $Result = Get-Content $JsonFile | ConvertFrom-Json
                    
                    if ($Result.PSObject.Properties[$Dataset]) {
                        $Metrics = $Result.$Dataset
                        $SeedResults.results[$Dataset] = $Metrics
                        
                        Write-Success "    ✓ $Dataset`: Accuracy = $([math]::Round($Metrics.accuracy * 100, 2))%, FP = $([math]::Round($Metrics.avg_forward_passes, 1)) ($([math]::Round($EvalDuration, 1))s)"
                    }
                }
            } else {
                Write-Error "    ✗ Evaluation failed on $Dataset"
            }
        }
        
        $AllResults += $SeedResults
    } else {
        Write-Warning "Skipping evaluation (--SkipEvaluation flag set)"
    }
}

# === AGGREGATION ===
if ($AllResults.Count -gt 0 -and -not $SkipEvaluation) {
    Write-ColorOutput "`n$('='*80)" "Cyan"
    Write-ColorOutput "Aggregating Results" "Cyan"
    Write-ColorOutput "$('='*80)`n" "Cyan"
    
    # Save individual seed results
    $AllResultsJson = $AllResults | ConvertTo-Json -Depth 10
    $AllResultsJson | Out-File "$ExperimentDir/all_seeds_results.json" -Encoding UTF8
    Write-Info "✓ Saved individual results to all_seeds_results.json"
    
    # Calculate statistics
    $Datasets = @("gsm8k", "gsm8k-hard", "svamp")
    $Summary = @{}
    
    foreach ($Dataset in $Datasets) {
        $Accuracies = @()
        $ForwardPasses = @()
        
        foreach ($Result in $AllResults) {
            if ($Result.results.ContainsKey($Dataset)) {
                $Accuracies += $Result.results[$Dataset].accuracy
                $ForwardPasses += $Result.results[$Dataset].avg_forward_passes
            }
        }
        
        if ($Accuracies.Count -gt 0) {
            $AccMean = ($Accuracies | Measure-Object -Average).Average
            $AccStd = [math]::Sqrt(($Accuracies | ForEach-Object { [math]::Pow($_ - $AccMean, 2) } | Measure-Object -Average).Average)
            
            $FpMean = ($ForwardPasses | Measure-Object -Average).Average
            $FpStd = [math]::Sqrt(($ForwardPasses | ForEach-Object { [math]::Pow($_ - $FpMean, 2) } | Measure-Object -Average).Average)
            
            $Summary[$Dataset] = @{
                accuracy_mean = $AccMean
                accuracy_std = $AccStd
                forward_passes_mean = $FpMean
                forward_passes_std = $FpStd
                n_seeds = $Accuracies.Count
            }
        }
    }
    
    # Save summary
    $SummaryJson = $Summary | ConvertTo-Json -Depth 10
    $SummaryJson | Out-File "$ExperimentDir/summary.json" -Encoding UTF8
    
    # Also save as YAML for compatibility
    $SummaryYaml = @"
# KAVA Multi-Seed Results Summary
# Experiment: $ExperimentName
# Seeds: $($Seeds -join ', ')
# Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

"@
    
    foreach ($Dataset in $Datasets) {
        if ($Summary.ContainsKey($Dataset)) {
            $Stats = $Summary[$Dataset]
            $SummaryYaml += @"
$Dataset`:
  accuracy_mean: $([math]::Round($Stats.accuracy_mean, 4))
  accuracy_std: $([math]::Round($Stats.accuracy_std, 4))
  forward_passes_mean: $([math]::Round($Stats.forward_passes_mean, 2))
  forward_passes_std: $([math]::Round($Stats.forward_passes_std, 2))
  n_seeds: $($Stats.n_seeds)

"@
        }
    }
    
    $SummaryYaml | Out-File "$ExperimentDir/summary.yaml" -Encoding UTF8
    
    Write-Info "✓ Saved summary to summary.json and summary.yaml"
    
    # Print summary table
    Write-ColorOutput "`n$('='*80)" "Green"
    Write-ColorOutput "FINAL RESULTS: $ExperimentName" "Green"
    Write-ColorOutput "$('='*80)`n" "Green"
    
    Write-Host $("{0,-15} {1,-25} {2,-25}" -f "Dataset", "Accuracy (%)", "Forward Passes")
    Write-Host $("{0,-15} {1,-25} {2,-25}" -f "-"*15, "-"*25, "-"*25)
    
    foreach ($Dataset in $Datasets) {
        if ($Summary.ContainsKey($Dataset)) {
            $Stats = $Summary[$Dataset]
            $AccStr = "$([math]::Round($Stats.accuracy_mean * 100, 2)) ± $([math]::Round($Stats.accuracy_std * 100, 2))"
            $FpStr = "$([math]::Round($Stats.forward_passes_mean, 1)) ± $([math]::Round($Stats.forward_passes_std, 1))"
            
            Write-Host $("{0,-15} {1,-25} {2,-25}" -f $Dataset, $AccStr, $FpStr)
        }
    }
    
    Write-ColorOutput "`n$('='*80)" "Green"
    Write-ColorOutput "Results based on $($AllResults.Count) seeds" "Green"
    Write-ColorOutput "$('='*80)`n" "Green"
}

Write-Success "`n✅ Multi-seed experiment completed successfully!`n"
