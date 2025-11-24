#!/bin/bash
#==============================================================================
# 给 HPC AI 助手的操作指令
# 问题：任务因 "Network is unreachable" 快速失败
# 解决：已修改配置文件使用本地路径 + 强制离线模式
#==============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 KAVA 网络错误修复方案"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 问题诊断："
echo "  - 所有任务快速失败（ExitCode 1:0 或 2:0）"
echo "  - 日志错误：Network is unreachable [Errno 101]"
echo "  - transformers 尝试访问 huggingface.co 但节点无外网"
echo ""
echo "✅ 已实施的修复："
echo "  1. 配置文件改用本地路径（/home/share/models/...）"
echo "  2. SLURM 脚本添加离线模式（HUGGINGFACE_HUB_OFFLINE=1）"
echo "  3. 创建诊断脚本验证模型加载"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 进入项目目录
cd "/home/rpwang/kava review" || {
    echo "❌ 错误: 无法进入项目目录"
    exit 1
}

echo "📂 当前目录: $(pwd)"
echo ""

#==============================================================================
# 选项 A: 快速验证（推荐先做，不提交作业）
#==============================================================================
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ 选项 A: 快速验证（推荐）                                  │"
echo "│ 测试模型是否能从本地路径加载（不提交 SLURM 作业）         │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""
echo "执行命令："
echo "  source venv/bin/activate"
echo "  export HF_HOME=/home/share/models"
echo "  export TRANSFORMERS_CACHE=/home/share/models"
echo "  export HUGGINGFACE_HUB_OFFLINE=1"
echo "  export TRANSFORMERS_OFFLINE=1"
echo "  python quick_model_test.py"
echo ""
echo "预期结果："
echo "  ✓ 所有 3 个模型都能从本地路径加载"
echo "  ✓ 无网络访问尝试"
echo "  ✓ 最后显示 '推荐方案: 在配置文件中使用本地路径'"
echo ""

read -p "是否立即运行验证测试? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔍 开始验证..."
    source venv/bin/activate
    export HF_HOME=/home/share/models
    export TRANSFORMERS_CACHE=/home/share/models
    export HUGGINGFACE_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    python quick_model_test.py
    
    VERIFY_EXIT=$?
    echo ""
    if [ $VERIFY_EXIT -eq 0 ]; then
        echo "✅ 验证通过！可以继续提交任务"
        echo ""
    else
        echo "❌ 验证失败！请检查错误信息"
        echo "   详细修复指南: cat FIX_NETWORK_ERROR.md"
        exit 1
    fi
fi

#==============================================================================
# 选项 B: 单任务测试（推荐在全量提交前）
#==============================================================================
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ 选项 B: 单任务测试（推荐）                                │"
echo "│ 提交 1 个最小任务验证 SLURM 环境（Qwen 0.5B，最快）       │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""
echo "执行命令："
echo "  sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm"
echo ""
echo "验证方法（等待 2-3 分钟后）："
echo "  tail -n 50 outputs/logs/kava_qwen05b_aug_*.out"
echo "  tail -n 50 outputs/logs/kava_qwen05b_aug_*.err"
echo ""
echo "成功标志："
echo "  ✓ 看到 'Loading model from /home/share/models/Qwen2.5-0.5B-Instruct'"
echo "  ✓ 看到 'Model loaded successfully'"
echo "  ✓ 看到 'Epoch 0 | Step 0 | Loss: ...'"
echo "  ✗ 不应看到 'Network is unreachable'"
echo ""

read -p "是否提交单任务测试? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 提交单任务测试..."
    JOB_ID=$(sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm | grep -oP '\d+')
    echo "✓ 任务已提交: Job ID $JOB_ID"
    echo ""
    echo "监控命令:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f outputs/logs/kava_qwen05b_aug_${JOB_ID}_0.out"
    echo ""
    echo "等待 2-3 分钟后检查日志（按 Ctrl+C 停止查看）"
    sleep 5
    
    # 等待日志文件出现
    LOG_FILE=""
    for i in {1..30}; do
        LOG_FILE=$(ls -t outputs/logs/kava_qwen05b_aug_${JOB_ID}_*.out 2>/dev/null | head -1)
        if [ -n "$LOG_FILE" ]; then
            break
        fi
        sleep 2
    done
    
    if [ -n "$LOG_FILE" ]; then
        echo "📄 日志文件: $LOG_FILE"
        tail -f "$LOG_FILE"
    else
        echo "⏳ 日志文件尚未生成，请手动检查"
        echo "   命令: ls -lht outputs/logs/"
    fi
    
    exit 0
fi

#==============================================================================
# 选项 C: 直接提交所有任务（需要确认验证通过）
#==============================================================================
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ 选项 C: 提交所有任务（12 个任务）                         │"
echo "│ ⚠️  建议先完成选项 A 或 B 的验证                           │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""
echo "执行命令:"
echo "  bash submit_all_jobs.sh"
echo ""
echo "将提交:"
echo "  - llama1b_aug (3 seeds)"
echo "  - llama1b_aug_nl (3 seeds)"
echo "  - llama3b_aug (3 seeds)"
echo "  - qwen05b_aug (3 seeds)"
echo "  总计: 4 主任务 × 3 种子 = 12 个子任务"
echo ""
echo "预计时间: 36-48 小时"
echo ""

read -p "⚠️  确认要提交所有 12 个任务? (yes/no) " CONFIRM
if [ "$CONFIRM" = "yes" ]; then
    echo "🚀 提交所有任务..."
    bash submit_all_jobs.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 所有任务已提交成功！"
        echo ""
        echo "监控命令:"
        echo "  bash monitor_jobs.sh --auto    # 每 30 秒自动刷新"
        echo "  squeue --me                    # 手动查看队列"
        echo ""
        echo "查看日志:"
        echo "  ls -lht outputs/logs/          # 列出最新日志"
        echo "  tail -f outputs/logs/kava_*.out  # 实时查看"
        echo ""
    else
        echo "❌ 任务提交失败，请检查错误信息"
        exit 1
    fi
else
    echo "❌ 已取消提交"
    echo "   建议: 先运行选项 A 或 B 验证修复效果"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 完整修复文档: cat FIX_NETWORK_ERROR.md"
echo "🐛 问题排查: python quick_model_test.py"
echo "📊 监控任务: bash monitor_jobs.sh --auto"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
