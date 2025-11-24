#!/bin/bash

################################################################################
# 一次性权限设置脚本
################################################################################
# 这个脚本会自动设置所有需要执行权限的文件
# 运行方式：bash fix_permissions.sh
################################################################################

echo "正在设置文件执行权限..."

# 当前存在的脚本
chmod +x verify_deployment.sh 2>/dev/null && echo "✓ verify_deployment.sh" || echo "⚠ verify_deployment.sh 不存在"
chmod +x setup_hpc.sh 2>/dev/null && echo "✓ setup_hpc.sh" || echo "⚠ setup_hpc.sh 不存在"
chmod +x start.sh 2>/dev/null && echo "✓ start.sh" || echo "⚠ start.sh 不存在"
chmod +x run_reproduce.sh 2>/dev/null && echo "✓ run_reproduce.sh" || echo "⚠ run_reproduce.sh 不存在"
chmod +x hpc_run_all.sh 2>/dev/null && echo "✓ hpc_run_all.sh" || echo "⚠ hpc_run_all.sh 不存在"
chmod +x fix_permissions.sh 2>/dev/null && echo "✓ fix_permissions.sh" || echo "⚠ fix_permissions.sh 不存在"

# SLURM 脚本
chmod +x submit_multi_seed.slurm 2>/dev/null && echo "✓ submit_multi_seed.slurm" || echo "⚠ submit_multi_seed.slurm 不存在"

# 自动生成的脚本（如果存在）
chmod +x monitor_jobs.sh 2>/dev/null && echo "✓ monitor_jobs.sh" || echo "⚠ monitor_jobs.sh (运行 start.sh 后会自动生成)"
chmod +x collect_results.sh 2>/dev/null && echo "✓ collect_results.sh" || echo "⚠ collect_results.sh (运行 start.sh 后会自动生成)"

echo ""
echo "权限设置完成！"
echo ""
echo "注意："
echo "  - monitor_jobs.sh 和 collect_results.sh 会在运行 start.sh 后自动生成"
echo "  - 生成后它们会自动具有执行权限"
echo ""
echo "现在可以运行："
echo "  bash verify_deployment.sh"
