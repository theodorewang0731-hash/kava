#!/usr/bin/env pwsh
#
# 多种子训练 + 推理 + 聚合脚本（PowerShell 版本）
# 用法: .\run_multi_seed.ps1 -Config llama1b_aug -Seeds 42,123,456
#

param(
    [Parameter(Mandatory=$false)]
    [string]$Config,
    
    [Parameter(Mandatory=$false)]
    [int[]]$Seeds,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTraining,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipEvaluation,
    
    [Parameter(Mandatory=$false)]
    [switch]$UseWandB
)

# 设置默认值
if (-not $Config) { $Config = "llama1b_aug" }
if (-not $Seeds) { $Seeds = @(42, 123, 456) }

# 颜色输出函数
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput "✓ $Message" "Green" }
function Write-Error { param([string]$Message) Write-ColorOutput "✗ $Message" "Red" }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠ $Message" "Yellow" }
function Write-Info { param([string]$Message) Write-ColorOutput "→ $Message" "Cyan" }

# 配置映射
$ConfigMap = @{
    "llama1b_aug" = @{
        "file" = "configs/llama1b_aug.yaml"
        "model" = "LLaMA-3.2-1B"
        "dataset" = "gsm8k-aug"
    }
    "llama1b_aug_nl" = @{
        "file" = "configs/llama1b_aug_nl.yaml"
        "model" = "LLaMA-3.2-1B"
        "dataset" = "gsm8k-aug-nl"
    }
    "qwen05b_aug" = @{
        "file" = "configs/qwen05b_aug.yaml"
        "model" = "Qwen2.5-0.5B"
        "dataset" = "gsm8k-aug"
    }
    "llama3b_aug" = @{
        "file" = "configs/llama3b_aug.yaml"
        "model" = "LLaMA-3.2-3B"
        "dataset" = "gsm8k-aug"
    }
}

# 验证配置
if (-not $ConfigMap.ContainsKey($Config)) {
    Write-Error "Unknown config: $Config"
    Write-Host "Available configs: $($ConfigMap.Keys -join ', ')" -ForegroundColor Yellow
    exit 1
}

$SelectedConfig = $ConfigMap[$Config]
$ConfigFile = $SelectedConfig["file"]
$ModelName = $SelectedConfig["model"]
$DatasetName = $SelectedConfig["dataset"]

# 设置输出目录
if ($OutputDir -eq "") {
    $OutputDir = "outputs/${Config}_multi_seed"
}

Write-ColorOutput "`n$('='*60)" "Magenta"
Write-ColorOutput "KAVA Multi-Seed Experiment" "Magenta"
Write-ColorOutput "$('='*60)" "Magenta"
Write-Host "Config:   $Config ($ConfigFile)"
Write-Host "Model:    $ModelName"
Write-Host "Dataset:  $DatasetName"
Write-Host "Seeds:    $($Seeds -join ', ')"
Write-Host "Output:   $OutputDir"
Write-ColorOutput "$('='*60)`n" "Magenta"

# 创建输出目录
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# 评估数据集
$EvalDatasets = @("gsm8k", "gsm8k-hard", "svamp")

# 训练每个种子
if (-not $SkipTraining) {
    foreach ($Seed in $Seeds) {
        $SeedDir = "$OutputDir/seed_$Seed"
        
        Write-ColorOutput "`n$('='*60)" "Cyan"
        Write-Info "Training seed $Seed"
        Write-ColorOutput "$('='*60)" "Cyan"
        
        $TrainArgs = @(
            "train.py",
            "--config", $ConfigFile,
            "--output_dir", $SeedDir,
            "--seed", $Seed
        )
        
        if ($UseWandB) {
            $TrainArgs += "--use_wandb"
        }
        
        $StartTime = Get-Date
        
        try {
            python @TrainArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Training failed for seed $Seed"
                continue
            }
            
            $Duration = (Get-Date) - $StartTime
            Write-Success "Training completed for seed $Seed (Duration: $($Duration.ToString('hh\:mm\:ss')))"
        }
        catch {
            Write-Error "Exception during training: $_"
            continue
        }
    }
} else {
    Write-Warning "Skipping training (--SkipTraining flag set)"
}

# 推理每个种子
if (-not $SkipEvaluation) {
    foreach ($Seed in $Seeds) {
        $SeedDir = "$OutputDir/seed_$Seed"
        $CheckpointDir = "$SeedDir/best_checkpoint"
        
        if (-not (Test-Path $CheckpointDir)) {
            Write-Warning "Checkpoint not found for seed $Seed, skipping evaluation"
            continue
        }
        
        Write-ColorOutput "`n$('='*60)" "Cyan"
        Write-Info "Evaluating seed $Seed"
        Write-ColorOutput "$('='*60)" "Cyan"
        
        foreach ($EvalDataset in $EvalDatasets) {
            Write-Info "Evaluating on $EvalDataset..."
            
            $EvalArgs = @(
                "evaluate.py",
                "--checkpoint_dir", $CheckpointDir,
                "--eval_dataset", $EvalDataset,
                "--output", "$SeedDir/results_$EvalDataset.yaml",
                "--seed", $Seed
            )
            
            $StartTime = Get-Date
            
            try {
                python @EvalArgs
                if ($LASTEXITCODE -eq 0) {
                    $Duration = (Get-Date) - $StartTime
                    Write-Success "$EvalDataset completed ($($Duration.TotalSeconds.ToString('F1'))s)"
                } else {
                    Write-Error "Evaluation failed for $EvalDataset"
                }
            }
            catch {
                Write-Error "Exception during evaluation: $_"
            }
        }
        
        Write-Success "All evaluations completed for seed $Seed"
    }
} else {
    Write-Warning "Skipping evaluation (--SkipEvaluation flag set)"
}

# 聚合结果
Write-ColorOutput "`n$('='*60)" "Cyan"
Write-Info "Aggregating results across seeds"
Write-ColorOutput "$('='*60)" "Cyan"

$SeedDirs = @()
foreach ($Seed in $Seeds) {
    $SeedDir = "$OutputDir/seed_$Seed"
    if (Test-Path $SeedDir) {
        $SeedDirs += $SeedDir
    }
}

if ($SeedDirs.Count -eq 0) {
    Write-Error "No valid seed directories found!"
    exit 1
}

$AggregateArgs = @(
    "aggregate_multi_seed.py",
    "--seed_dirs"
) + $SeedDirs + @(
    "--datasets"
) + $EvalDatasets + @(
    "--model_name", "KAVA-$ModelName",
    "--output_json", "$OutputDir/aggregated_results.json",
    "--output_yaml", "$OutputDir/aggregated_results.yaml"
)

try {
    python @AggregateArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Results aggregated successfully"
    } else {
        Write-Error "Aggregation failed"
    }
}
catch {
    Write-Error "Exception during aggregation: $_"
}

# 完成
Write-ColorOutput "`n$('='*60)" "Green"
Write-Success "All experiments completed!"
Write-ColorOutput "$('='*60)" "Green"
Write-Host "Results saved to: $OutputDir"
Write-Host "  - Individual seeds: $OutputDir/seed_*/"
Write-Host "  - Aggregated: $OutputDir/aggregated_results.{json,yaml}"
Write-ColorOutput "$('='*60)`n" "Green"