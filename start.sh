#!/bin/bash

################################################################################
# KAVA 一键启动脚本 - Linux HPC 环境
################################################################################
# 此脚本整合了验证、配置和启动的完整流程
#
# 用法: bash start.sh [options]
#
# 选项:
#   --verify-only    仅验证部署，不启动训练
#   --setup-only     仅设置环境，不启动训练
#   --no-verify      跳过验证，直接启动
#   --method METHOD  模型下载方法 (direct|proxy|mirror)
#   --skip-download  跳过模型下载
################################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# 默认选项
VERIFY_ONLY=false
SETUP_ONLY=false
NO_VERIFY=false
METHOD=""
SKIP_DOWNLOAD=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --no-verify)
            NO_VERIFY=true
            shift
            ;;
        --method)
            METHOD="--method $2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift
            ;;
        --help)
            head -n 20 "$0" | tail -n 18
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║        KAVA Paper Reproduction - HPC 一键启动                  ║
║                                                                ║
║        Knowledge-Augmented Verbal-Augmentation                 ║
║        Strict reproduction per paper specifications            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

# 步骤 1: 验证部署
if [ "$NO_VERIFY" = false ]; then
    echo -e "${MAGENTA}[步骤 1/3] 验证部署${NC}"
    echo "────────────────────────────────────────────────────────────────"
    
    if [ -f "verify_deployment.sh" ]; then
        bash verify_deployment.sh
        
        if [ $? -ne 0 ]; then
            echo ""
            echo -e "${RED}验证失败！请修复错误后重新运行。${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠ verify_deployment.sh 未找到，跳过验证${NC}"
    fi
    
    echo ""
    
    if [ "$VERIFY_ONLY" = true ]; then
        echo -e "${GREEN}验证完成！${NC}"
        exit 0
    fi
fi

# 步骤 2: 快速设置
echo -e "${MAGENTA}[步骤 2/3] 环境设置${NC}"
echo "────────────────────────────────────────────────────────────────"

if [ -f "setup_hpc.sh" ]; then
    bash setup_hpc.sh
    
    # 使环境变量生效
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}⚠ setup_hpc.sh 未找到，跳过设置${NC}"
fi

echo ""

if [ "$SETUP_ONLY" = true ]; then
    echo -e "${GREEN}设置完成！${NC}"
    exit 0
fi

# 步骤 3: 启动训练
echo -e "${MAGENTA}[步骤 3/3] 启动训练任务${NC}"
echo "────────────────────────────────────────────────────────────────"

if [ -f "run_reproduce.sh" ]; then
    echo -e "${CYAN}正在启动自动化训练流程...${NC}"
    echo ""
    
    # 构建命令
    CMD="bash run_reproduce.sh"
    [ -n "$METHOD" ] && CMD="$CMD $METHOD"
    [ -n "$SKIP_DOWNLOAD" ] && CMD="$CMD $SKIP_DOWNLOAD"
    
    echo -e "${BLUE}执行命令: $CMD${NC}"
    echo ""
    
    # 执行
    $CMD
    
    EXIT_CODE=$?
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}"
        cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  ✅ 启动完成！训练任务已提交到 SLURM 队列                     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
        echo -e "${NC}"
        
        echo ""
        echo -e "${CYAN}下一步操作：${NC}"
        echo ""
        echo "1. 监控任务进度："
        echo -e "   ${GREEN}bash monitor_jobs.sh${NC}"
        echo ""
        echo "2. 查看队列状态："
        echo -e "   ${GREEN}squeue -u \$USER${NC}"
        echo ""
        echo "3. 查看实时日志："
        echo -e "   ${GREEN}tail -f outputs/logs/llama1b_aug_seed42.log${NC}"
        echo ""
        echo "4. 训练完成后收集结果："
        echo -e "   ${GREEN}bash collect_results.sh${NC}"
        echo ""
        echo -e "${YELLOW}预计时间：${NC}"
        echo "  - 模型下载: 17-100 分钟（如未跳过）"
        echo "  - 训练任务: 36-48 小时（并行执行）"
        echo ""
    else
        echo -e "${RED}"
        cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  ❌ 启动失败                                                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
        echo -e "${NC}"
        
        echo ""
        echo -e "${YELLOW}故障排除建议：${NC}"
        echo ""
        echo "1. 检查日志输出中的错误信息"
        echo "2. 验证磁盘空间: df -h \$HOME"
        echo "3. 检查网络连接: ping huggingface.co"
        echo "4. 查看详细文档: docs/GETTING_STARTED_HPC.md"
        echo ""
        exit $EXIT_CODE
    fi
else
    echo -e "${RED}✗ run_reproduce.sh 未找到${NC}"
    exit 1
fi
