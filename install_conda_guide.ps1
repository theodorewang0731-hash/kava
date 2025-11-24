# KAVA Windows Conda 安装指南
# 如果你想在 Windows 上使用 conda

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Conda 安装指南（Windows）" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "检测到系统中没有 conda。" -ForegroundColor Yellow
Write-Host ""
Write-Host "推荐安装 Miniconda（轻量级）：" -ForegroundColor Cyan
Write-Host "  1. 下载地址: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor White
Write-Host "  2. 选择 Windows 64-bit 版本" -ForegroundColor White
Write-Host "  3. 安装时勾选 'Add to PATH' 选项" -ForegroundColor White
Write-Host ""
Write-Host "或者使用 Anaconda（完整版）：" -ForegroundColor Cyan
Write-Host "  下载地址: https://www.anaconda.com/download" -ForegroundColor White
Write-Host ""
Write-Host "安装完成后，重新打开 PowerShell 并运行：" -ForegroundColor Yellow
Write-Host "  conda create -n kava python=3.10" -ForegroundColor White
Write-Host "  conda activate kava" -ForegroundColor White
Write-Host "  pip install -r requirements.txt" -ForegroundColor White
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "临时解决方案（无需 conda）：" -ForegroundColor Green
Write-Host "  如果你不想安装 conda，可以使用 Python venv：" -ForegroundColor White
Write-Host "  .\setup_windows_venv.ps1" -ForegroundColor Cyan
Write-Host ""
