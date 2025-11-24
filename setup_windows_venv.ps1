# KAVA Windows 环境配置脚本（使用 venv）
# 用法: .\setup_windows_venv.ps1

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  KAVA 项目 - Windows 虚拟环境配置" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
Write-Host "[1/4] 检查 Python..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "✗ 错误: 未找到 Python" -ForegroundColor Red
    Write-Host "请从 https://www.python.org/downloads/ 安装 Python 3.8+" -ForegroundColor Red
    exit 1
}

$pythonVersion = python --version
Write-Host "✓ Python 已安装: $pythonVersion" -ForegroundColor Green

# 创建虚拟环境
Write-Host ""
Write-Host "[2/4] 创建虚拟环境..." -ForegroundColor Yellow

if (Test-Path "venv") {
    Write-Host "⚠ 虚拟环境已存在" -ForegroundColor Yellow
    $recreate = Read-Host "是否重新创建? (y/N)"
    if ($recreate -eq 'y' -or $recreate -eq 'Y') {
        Remove-Item -Recurse -Force venv
        python -m venv venv
        Write-Host "✓ 虚拟环境已重新创建" -ForegroundColor Green
    } else {
        Write-Host "→ 使用现有虚拟环境" -ForegroundColor Cyan
    }
} else {
    python -m venv venv
    Write-Host "✓ 虚拟环境已创建" -ForegroundColor Green
}

# 激活虚拟环境
Write-Host ""
Write-Host "[3/4] 激活虚拟环境并安装依赖..." -ForegroundColor Yellow

# 激活并安装
& .\venv\Scripts\Activate.ps1

# 升级 pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ 依赖安装完成" -ForegroundColor Green
} else {
    Write-Host "✗ 依赖安装失败" -ForegroundColor Red
    exit 1
}

# 验证安装
Write-Host ""
Write-Host "[4/4] 验证安装..." -ForegroundColor Yellow

$testResult = python -c "from src.trainer import KAVATrainer; from src.losses import KAVALoss; print('OK')" 2>&1
if ($testResult -match "OK") {
    Write-Host "✓ 核心模块导入成功" -ForegroundColor Green
} else {
    Write-Host "⚠ 模块导入警告（可能需要 GPU 才能正常运行）" -ForegroundColor Yellow
}

# 完成提示
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  配置完成！" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "虚拟环境已激活。接下来你可以：" -ForegroundColor White
Write-Host ""
Write-Host "1. 运行训练：" -ForegroundColor Cyan
Write-Host "   python train.py --config configs/llama1b_aug.yaml" -ForegroundColor White
Write-Host ""
Write-Host "2. 运行评估：" -ForegroundColor Cyan
Write-Host "   python evaluate.py --checkpoint <路径> --config configs/llama1b_aug.yaml" -ForegroundColor White
Write-Host ""
Write-Host "3. 快速测试：" -ForegroundColor Cyan
Write-Host "   python smoke_test_lite.py" -ForegroundColor White
Write-Host ""
Write-Host "注意: 如果关闭终端，下次需要重新激活虚拟环境：" -ForegroundColor Yellow
Write-Host "      .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
