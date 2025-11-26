#!/usr/bin/bash
#==============================================================================
# Enroot 容器镜像导入脚本
# 用途：在 HPC 登录节点导入 PyTorch 官方镜像
# 使用方法：bash setup_enroot_container.sh
#==============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "KAVA Enroot 容器环境设置"
echo "=========================================="
echo ""

# 1. 检查当前目录
echo "步骤 1/4: 检查环境"
echo "当前目录: $(pwd)"
echo "当前用户: $(whoami)"
echo ""

# 检查目录名是否包含空格
if [[ "$(pwd)" == *" "* ]]; then
    echo "❌ 错误：当前目录包含空格，这可能导致问题"
    echo "建议：请切换到不包含空格的目录，例如 ~/kava 或 ~/kava_reproduction"
    exit 1
fi

# 2. 检查 enroot 是否可用
if ! command -v enroot &> /dev/null; then
    echo "❌ 错误：enroot 命令未找到"
    echo "请确保您在 HPC 登录节点，并且已加载 enroot 模块"
    echo "尝试运行: module load enroot"
    exit 1
fi

echo "✓ enroot 已找到: $(which enroot)"
enroot version
echo ""

# 3. 检查是否已存在镜像文件
IMAGE_FILE="pytorch+pytorch+2.5.1-cuda12.1-cudnn9-runtime.sqsh"

if [ -f "$IMAGE_FILE" ]; then
    echo "⚠️  警告：镜像文件已存在"
    echo "文件: $IMAGE_FILE"
    echo "大小: $(du -h $IMAGE_FILE | cut -f1)"
    echo ""
    read -p "是否删除并重新下载？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧镜像..."
        rm -f "$IMAGE_FILE"
    else
        echo "保留现有镜像，跳过下载"
        echo ""
        echo "=========================================="
        echo "✓ 设置完成！镜像已就绪"
        echo "=========================================="
        echo ""
        echo "下一步："
        echo "1. 提交训练任务："
        echo "   sbatch --export=CONFIG=llama1b_aug submit_enroot.slurm"
        echo ""
        echo "2. 查看任务状态："
        echo "   squeue -u \$USER"
        echo ""
        echo "3. 查看日志："
        echo "   tail -f logs/kava_enroot_*.out"
        exit 0
    fi
fi

# 4. 导入镜像（使用国内镜像加速）
echo "步骤 2/4: 导入 PyTorch 镜像"
echo "镜像源: dockerpull.org/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
echo "这可能需要几分钟时间，请耐心等待..."
echo ""

# 使用 dockerpull.org 国内镜像加速
enroot import "docker://dockerpull.org/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"

# 5. 验证镜像文件
echo ""
echo "步骤 3/4: 验证镜像"
if [ -f "$IMAGE_FILE" ]; then
    echo "✓ 镜像文件创建成功"
    echo "文件: $IMAGE_FILE"
    echo "大小: $(du -h $IMAGE_FILE | cut -f1)"
else
    echo "❌ 错误：镜像文件未找到"
    echo "导入可能失败，请检查网络连接和 enroot 配置"
    exit 1
fi
echo ""

# 6. 检查共享模型目录
echo "步骤 4/4: 检查共享资源"
if [ -d "/home/share/models" ]; then
    echo "✓ 共享模型目录存在: /home/share/models"
    echo "模型数量: $(ls /home/share/models 2>/dev/null | wc -l) 个"
else
    echo "⚠️  警告：共享模型目录不存在或无权访问"
    echo "路径: /home/share/models"
    echo "这不会阻止容器启动，但可能需要手动下载模型"
fi
echo ""

# 7. 完成提示
echo "=========================================="
echo "✓ Enroot 容器环境设置完成！"
echo "=========================================="
echo ""
echo "镜像信息："
echo "  文件: $IMAGE_FILE"
echo "  大小: $(du -h $IMAGE_FILE | cut -f1)"
echo ""
echo "下一步操作："
echo ""
echo "1. 提交训练任务："
echo "   sbatch --export=CONFIG=llama1b_aug submit_enroot.slurm"
echo ""
echo "   可用配置："
echo "   - llama1b_aug      (Llama-3.2-1B, 数据增强)"
echo "   - llama3b_aug      (Llama-3.2-3B, 数据增强)"
echo "   - phi3_aug         (Phi-3.5-mini, 数据增强)"
echo "   - qwen2_aug        (Qwen2.5-1.5B, 数据增强)"
echo ""
echo "2. 查看任务队列："
echo "   squeue -u \$USER"
echo ""
echo "3. 监控日志（任务开始后）："
echo "   tail -f logs/kava_enroot_*.out"
echo ""
echo "4. 取消任务（如需要）："
echo "   scancel <JOB_ID>"
echo ""
echo "提示："
echo "  - 任务时间已设置为 4 小时以避免余额问题"
echo "  - 容器会自动挂载 /home/share/models 到 /models"
echo "  - 依赖会在容器启动时自动安装"
echo "  - 使用 array 任务并行运行 3 个随机种子"
echo ""
echo "=========================================="
