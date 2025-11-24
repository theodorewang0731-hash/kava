#!/bin/bash
################################################################################
# HPC 故障诊断和修复脚本
# 用法: bash diagnose_and_fix.sh
################################################################################

set +e  # 不要在错误时退出，我们需要收集所有信息

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "════════════════════════════════════════════════════════════════"
echo "  KAVA HPC 故障诊断"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# 记录所有输出到日志文件
LOG_FILE="diagnostic_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "诊断日志将保存到: $LOG_FILE"
echo ""

# =============================================================================
# 1. 检查当前目录
# =============================================================================
echo -e "${BLUE}[1/10] 检查当前目录${NC}"
echo "当前工作目录: $(pwd)"
echo "目录内容:"
ls -la | head -20

if [ ! -f "train.py" ]; then
    echo -e "${RED}✗ 未找到 train.py${NC}"
    echo -e "${YELLOW}  可能原因: 目录名包含空格导致路径错误${NC}"
    echo -e "${YELLOW}  你的目录: '/home/rpwang/kava review'${NC}"
    echo -e "${YELLOW}  包含空格的目录在 Linux 中需要特殊处理${NC}"
else
    echo -e "${GREEN}✓ train.py 存在${NC}"
fi
echo ""

# =============================================================================
# 2. 检查脚本文件和权限
# =============================================================================
echo -e "${BLUE}[2/10] 检查脚本文件权限${NC}"
for script in setup_hpc_env.sh run_reproduce.sh setup_hpc.sh; do
    if [ -f "$script" ]; then
        perms=$(ls -l "$script" | awk '{print $1}')
        echo "  $script: $perms"
        if [[ ! -x "$script" ]]; then
            echo -e "${YELLOW}    ⚠ 不可执行，正在修复...${NC}"
            chmod +x "$script"
        fi
    else
        echo -e "${RED}  ✗ $script 不存在${NC}"
    fi
done
echo ""

# =============================================================================
# 3. 检查 Module 系统
# =============================================================================
echo -e "${BLUE}[3/10] 检查 Module 系统${NC}"
if [ -f /usr/share/modules/init/bash ]; then
    echo -e "${GREEN}✓ Module 初始化文件存在${NC}"
    . /usr/share/modules/init/bash
    
    if command -v module &> /dev/null; then
        echo -e "${GREEN}✓ module 命令可用${NC}"
        
        # 检查 modulefiles 路径
        if [ -d /home/share/modules/modulefiles ]; then
            echo -e "${GREEN}✓ modulefiles 目录存在${NC}"
            module use --append /home/share/modules/modulefiles
        else
            echo -e "${YELLOW}⚠ modulefiles 目录不存在: /home/share/modules/modulefiles${NC}"
        fi
    else
        echo -e "${RED}✗ module 命令不可用${NC}"
    fi
else
    echo -e "${RED}✗ Module 系统未找到${NC}"
    echo "  你的 HPC 可能不使用 Environment Modules"
fi
echo ""

# =============================================================================
# 4. 检查 Anaconda/Conda
# =============================================================================
echo -e "${BLUE}[4/10] 检查 Conda 可用性${NC}"

# 尝试加载 anaconda3
if command -v module &> /dev/null; then
    echo "尝试加载 anaconda3 模块..."
    module load anaconda3 2>&1 || echo "  加载失败"
fi

# 检查 conda
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓ Conda 可用: $(conda --version)${NC}"
    echo "  Conda 路径: $(which conda)"
else
    echo -e "${RED}✗ Conda 不可用${NC}"
    echo ""
    echo "可能的 conda 位置:"
    find /opt /usr/local /home/share -name "conda" -type f 2>/dev/null | head -5
    echo ""
    echo -e "${YELLOW}替代方案: 使用系统 Python${NC}"
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}✓ Python3 可用: $(python3 --version)${NC}"
    fi
fi
echo ""

# =============================================================================
# 5. 检查 Python 环境
# =============================================================================
echo -e "${BLUE}[5/10] 检查 Python 环境${NC}"
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Python3: $(python3 --version)${NC}"
    echo "  路径: $(which python3)"
elif command -v python &> /dev/null; then
    echo -e "${GREEN}✓ Python: $(python --version)${NC}"
    echo "  路径: $(which python)"
else
    echo -e "${RED}✗ 未找到 Python${NC}"
fi

# 检查 pip
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓ pip3 可用${NC}"
elif command -v pip &> /dev/null; then
    echo -e "${GREEN}✓ pip 可用${NC}"
else
    echo -e "${RED}✗ pip 不可用${NC}"
fi
echo ""

# =============================================================================
# 6. 检查磁盘空间
# =============================================================================
echo -e "${BLUE}[6/10] 检查磁盘空间${NC}"
df -h "$HOME" | grep -v "Filesystem"
echo ""
AVAIL_GB=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -gt 20 ]; then
    echo -e "${GREEN}✓ 磁盘空间充足 (${AVAIL_GB}GB 可用，需要 ~20GB)${NC}"
else
    echo -e "${YELLOW}⚠ 磁盘空间可能不足 (${AVAIL_GB}GB 可用，建议 >20GB)${NC}"
fi
echo ""

# =============================================================================
# 7. 检查网络连接
# =============================================================================
echo -e "${BLUE}[7/10] 检查网络连接${NC}"
if ping -c 1 huggingface.co &> /dev/null; then
    echo -e "${GREEN}✓ 可以访问 huggingface.co${NC}"
else
    echo -e "${YELLOW}⚠ 无法访问 huggingface.co (可能需要代理或使用镜像)${NC}"
fi
echo ""

# =============================================================================
# 8. 检查 SLURM
# =============================================================================
echo -e "${BLUE}[8/10] 检查 SLURM${NC}"
if command -v sbatch &> /dev/null; then
    echo -e "${GREEN}✓ SLURM 可用${NC}"
    echo "  sbatch: $(which sbatch)"
    echo "  squeue: $(which squeue)"
else
    echo -e "${RED}✗ SLURM 不可用${NC}"
    echo "  此服务器可能不是 SLURM 集群"
fi
echo ""

# =============================================================================
# 9. 检查依赖文件
# =============================================================================
echo -e "${BLUE}[9/10] 检查必需文件${NC}"
REQUIRED_FILES=(
    "requirements.txt"
    "train.py"
    "evaluate.py"
    "src/trainer.py"
    "src/losses.py"
    "configs/llama1b_aug.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}  ✓ $file${NC}"
    else
        echo -e "${RED}  ✗ $file 缺失${NC}"
    fi
done
echo ""

# =============================================================================
# 10. 路径问题检测
# =============================================================================
echo -e "${BLUE}[10/10] 检查路径问题${NC}"
CURRENT_DIR=$(pwd)
if [[ "$CURRENT_DIR" == *" "* ]]; then
    echo -e "${RED}✗ 严重问题: 当前路径包含空格！${NC}"
    echo "  当前路径: '$CURRENT_DIR'"
    echo ""
    echo -e "${YELLOW}解决方案:${NC}"
    echo "  1. 重命名目录，去掉空格:"
    echo "     cd /home/rpwang"
    echo "     mv 'kava review' kava_review"
    echo "     cd kava_review"
    echo ""
    echo "  2. 或者总是使用引号:"
    echo "     cd \"/home/rpwang/kava review\""
    echo "     source setup_hpc_env.sh"
    echo ""
else
    echo -e "${GREEN}✓ 路径不包含空格${NC}"
fi
echo ""

# =============================================================================
# 总结和建议
# =============================================================================
echo -e "${CYAN}"
echo "════════════════════════════════════════════════════════════════"
echo "  诊断完成"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"

echo -e "${YELLOW}常见问题和解决方案:${NC}"
echo ""
echo "1. 如果路径包含空格:"
echo "   mv '/home/rpwang/kava review' /home/rpwang/kava_review"
echo ""
echo "2. 如果 conda 不可用，使用 Python venv:"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "3. 如果 module 系统不可用:"
echo "   # 查找 conda"
echo "   find /opt -name conda 2>/dev/null"
echo "   # 或直接使用系统 Python"
echo ""
echo "4. 查看详细日志:"
echo "   cat $LOG_FILE"
echo ""

echo -e "${GREEN}诊断报告已保存到: $LOG_FILE${NC}"
