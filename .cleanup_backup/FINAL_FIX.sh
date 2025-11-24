#!/bin/bash
#==============================================================================
# 最终修复方案 - 使用 HPC 共享库本地路径
# 问题：共享库是直接目录格式，不是 transformers 标准缓存格式
# 解决：配置文件直接使用绝对路径 + 代码强制本地加载
#==============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 KAVA 网络错误最终修复方案"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ 已完成的修复："
echo "  1. 配置文件改用本地绝对路径"
echo "  2. 代码添加本地路径检测和强制离线加载"
echo "  3. 所有 3 个模型已确认完整存在"
echo ""
echo "📋 修改的文件："
echo "  - configs/llama1b_aug.yaml → /home/share/models/Llama-3.2-1B-Instruct"
echo "  - configs/llama1b_aug_nl.yaml → /home/share/models/Llama-3.2-1B-Instruct"
echo "  - configs/llama3b_aug.yaml → /home/share/models/Llama-3.2-3B-Instruct"
echo "  - configs/qwen05b_aug.yaml → /home/share/models/Qwen2.5-0.5B-Instruct"
echo "  - src/trainer.py → 添加本地路径强制离线"
echo "  - evaluate.py → 添加本地路径强制离线"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "/home/rpwang/kava review" || {
    echo "❌ 错误: 无法进入项目目录"
    exit 1
}

#==============================================================================
# 步骤 1: 快速验证（本地测试，不提交 SLURM）
#==============================================================================
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ 步骤 1: 快速验证（推荐）                                  │"
echo "│ 在登录节点测试模型加载（不提交作业，2 分钟内完成）        │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""

read -p "是否运行快速验证? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔍 开始验证..."
    source venv/bin/activate
    
    # 测试加载所有 3 个模型的配置（快速，不加载权重）
    python -c "
import os
from transformers import AutoConfig

models = [
    ('/home/share/models/Llama-3.2-1B-Instruct', 'Llama 1B'),
    ('/home/share/models/Llama-3.2-3B-Instruct', 'Llama 3B'),
    ('/home/share/models/Qwen2.5-0.5B-Instruct', 'Qwen 0.5B'),
]

print('━' * 60)
print('测试从本地路径加载模型配置')
print('━' * 60)
success = 0
for path, name in models:
    print(f'\n【{name}】')
    print(f'  路径: {path}')
    try:
        config = AutoConfig.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True
        )
        print(f'  ✓ 成功加载')
        print(f'    模型类型: {config.model_type}')
        print(f'    隐藏层: {config.hidden_size}')
        success += 1
    except Exception as e:
        print(f'  ✗ 失败: {e}')

print('\n' + '━' * 60)
if success == 3:
    print('✅ 所有模型验证通过！可以提交训练任务')
    exit(0)
else:
    print(f'⚠️  部分模型验证失败 ({success}/3)')
    exit(1)
"
    
    VERIFY_EXIT=$?
    echo ""
    if [ $VERIFY_EXIT -eq 0 ]; then
        echo "✅ 验证通过！"
    else
        echo "❌ 验证失败！请检查错误信息"
        echo ""
        echo "常见问题排查："
        echo "  1. 检查路径是否正确:"
        echo "     ls -lh /home/share/models/Llama-3.2-1B-Instruct/config.json"
        echo "  2. 检查权限:"
        echo "     ls -ld /home/share/models/"
        echo "  3. 检查文件完整性:"
        echo "     ls -lh /home/share/models/Llama-3.2-1B-Instruct/"
        exit 1
    fi
fi

#==============================================================================
# 步骤 2: 单任务测试（提交到 SLURM）
#==============================================================================
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ 步骤 2: 单任务测试（推荐）                                │"
echo "│ 提交 1 个最小任务到 SLURM 验证完整流程                    │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""
echo "将提交: Qwen 0.5B × 1 个种子（最快，约 2-4 小时）"
echo ""

read -p "是否提交单任务测试? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 提交单任务测试..."
    
    # 清理旧日志
    rm -f outputs/logs/kava_qwen05b_aug_*.out outputs/logs/kava_qwen05b_aug_*.err 2>/dev/null
    
    JOB_ID=$(sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm 2>&1 | grep -oP '\d+')
    
    if [ -n "$JOB_ID" ]; then
        echo "✓ 任务已提交: Job ID $JOB_ID"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 监控任务"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "查看队列:"
        echo "  squeue -j $JOB_ID"
        echo ""
        echo "查看日志 (等待 2-3 分钟后):"
        echo "  tail -f outputs/logs/kava_qwen05b_aug_${JOB_ID}_0.out"
        echo "  tail -f outputs/logs/kava_qwen05b_aug_${JOB_ID}_0.err"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ 成功标志（日志中应显示）："
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Loading base model..."
        echo "Model: /home/share/models/Qwen2.5-0.5B-Instruct"
        echo "Loading mode: Local path                    ← ✅ 关键"
        echo "Model loaded successfully"
        echo "Training started"
        echo "Epoch 0 | Step 0 | Loss: ..."
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "❌ 不应出现（如果看到说明仍有问题）："
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "✗ Network is unreachable"
        echo "✗ Cannot connect to huggingface.co"
        echo "✗ We couldn't connect to 'https://huggingface.co'"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # 等待日志文件出现并显示
        echo "⏳ 等待日志文件生成..."
        for i in {1..60}; do
            LOG_FILE=$(ls -t outputs/logs/kava_qwen05b_aug_${JOB_ID}_*.out 2>/dev/null | head -1)
            if [ -n "$LOG_FILE" ]; then
                echo ""
                echo "📄 找到日志文件: $LOG_FILE"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "最新日志内容（实时更新，按 Ctrl+C 停止）:"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                tail -f "$LOG_FILE"
                break
            fi
            echo -n "."
            sleep 2
        done
        
        if [ -z "$LOG_FILE" ]; then
            echo ""
            echo "⏳ 日志文件尚未生成（任务可能在排队）"
            echo ""
            echo "手动检查:"
            echo "  squeue -j $JOB_ID  # 查看任务状态"
            echo "  ls -lht outputs/logs/  # 列出日志文件"
        fi
    else
        echo "❌ 任务提交失败"
        echo ""
        echo "调试信息:"
        sbatch --export=CONFIG=qwen05b_aug --array=0 submit_multi_seed.slurm
        exit 1
    fi
    
    exit 0
fi

#==============================================================================
# 步骤 3: 提交所有任务
#==============================================================================
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ 步骤 3: 提交所有任务（确认测试通过后）                    │"
echo "│ 12 个任务，预计 36-48 小时                                 │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""
echo "将提交:"
echo "  - llama1b_aug (3 seeds)"
echo "  - llama1b_aug_nl (3 seeds)"
echo "  - llama3b_aug (3 seeds)"
echo "  - qwen05b_aug (3 seeds)"
echo ""
echo "⚠️  建议: 先完成步骤 1 和步骤 2 的验证"
echo ""

read -p "⚠️  确认要提交所有 12 个任务? (yes/no) " CONFIRM
if [ "$CONFIRM" = "yes" ]; then
    echo "🚀 提交所有任务..."
    bash submit_all_jobs.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 所有任务已提交！"
        echo ""
        echo "监控命令:"
        echo "  bash monitor_jobs.sh --auto"
        echo "  squeue --me"
    else
        echo "❌ 任务提交失败"
        exit 1
    fi
else
    echo "❌ 已取消"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 修复总结"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "问题根源:"
echo "  - HPC 共享库是直接目录格式（不是 transformers 标准缓存）"
echo "  - 使用 repo ID 会尝试联网，导致 Network is unreachable"
echo ""
echo "解决方案:"
echo "  ✓ 配置文件改用绝对路径: /home/share/models/Llama-3.2-1B-Instruct"
echo "  ✓ 代码检测本地路径时强制 local_files_only=True"
echo "  ✓ 避免任何网络访问尝试"
echo ""
echo "验证要点:"
echo "  ✓ 日志显示 'Loading mode: Local path'"
echo "  ✓ 无 'Network is unreachable' 错误"
echo "  ✓ 模型加载成功，训练正常启动"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
