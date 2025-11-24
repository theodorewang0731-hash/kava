# KAVA 一键复现脚本 - Windows 版本
# 用法: .\run_reproduce.ps1

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  KAVA 项目 - Windows 一键复现" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# 检查必需工具
Write-Host "[1/5] 检查环境..." -ForegroundColor Yellow

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "✗ 错误: 未找到 Python" -ForegroundColor Red
    Write-Host "请安装 Python 3.8+ 并添加到 PATH" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python 已安装: $(python --version)" -ForegroundColor Green

# 检查 Git Bash
$gitBashPath = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $gitBashPath)) {
    $gitBashPath = "C:\Program Files (x86)\Git\bin\bash.exe"
}

if (Test-Path $gitBashPath) {
    Write-Host "✓ 检测到 Git Bash" -ForegroundColor Green
    Write-Host ""
    Write-Host "[2/5] 使用 Git Bash 运行脚本..." -ForegroundColor Yellow
    
    # 转换路径为 Unix 格式
    $scriptPath = (Get-Location).Path.Replace('\', '/').Replace(':', '')
    $scriptPath = "/$scriptPath/run_reproduce.sh"
    
    # 运行 bash 脚本
    & $gitBashPath -c "cd '$((Get-Location).Path.Replace('\', '/'))' && bash run_reproduce.sh"
    
} else {
    Write-Host "⚠ 未找到 Git Bash，尝试直接使用 Python..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "[2/5] 安装依赖..." -ForegroundColor Yellow
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    Write-Host ""
    Write-Host "[3/5] 检查环境..." -ForegroundColor Yellow
    python pre_training_check.py
    
    Write-Host ""
    Write-Host "[4/5] 开始训练..." -ForegroundColor Yellow
    Write-Host "注意: 在 Windows 上建议单独运行训练任务" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "示例命令:" -ForegroundColor Cyan
    Write-Host "  python train.py --config configs/llama1b_aug.yaml --wandb" -ForegroundColor White
    Write-Host ""
    
    $continue = Read-Host "是否现在启动训练? (y/N)"
    if ($continue -eq 'y' -or $continue -eq 'Y') {
        python train.py --config configs/llama1b_aug.yaml
    } else {
        Write-Host "训练已取消。请手动运行上述命令启动训练。" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  完成！" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
