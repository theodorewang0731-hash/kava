#!/bin/bash
################################################################################
# KAVA 快速启动脚本 - 最简单的使用方式
# 用法: bash quick_start.sh
################################################################################

echo "
╔════════════════════════════════════════════════════════════════════════════╗
║                          KAVA 项目快速启动                                 ║
╚════════════════════════════════════════════════════════════════════════════╝
"

# 默认配置
USE_MIRROR=true
SKIP_DOWNLOAD=false
AUTO_UPLOAD=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-mirror)
            USE_MIRROR=false
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --upload)
            AUTO_UPLOAD=true
            shift
            ;;
        --help|-h)
            echo "用法: bash quick_start.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --no-mirror       不使用 HF-Mirror 镜像（国外服务器使用）"
            echo "  --skip-download   跳过下载步骤（如果资源已存在）"
            echo "  --upload          自动上传结果到 HuggingFace"
            echo "  --help, -h        显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  bash quick_start.sh                    # 标准运行"
            echo "  bash quick_start.sh --skip-download    # 跳过下载"
            echo "  bash quick_start.sh --upload           # 训练后上传结果"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 显示配置
echo "当前配置:"
echo "  - 使用 HF-Mirror: $USE_MIRROR"
echo "  - 跳过下载: $SKIP_DOWNLOAD"
echo "  - 自动上传: $AUTO_UPLOAD"
echo ""

# 检查是否在正确的目录
if [ ! -f "run_everything.sh" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 设置环境变量
export USE_HF_MIRROR=$USE_MIRROR
export SKIP_DOWNLOAD=$SKIP_DOWNLOAD
export UPLOAD_TO_HF=$AUTO_UPLOAD

# 运行主脚本
echo "启动主脚本..."
echo ""
bash run_everything.sh

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           任务全部完成！                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
