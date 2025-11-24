# 与 HPC AI 助手的对话脚本

> 这是一个完整的对话流程，告诉你每一步该说什么，以及如何回应 AI 的反馈

---

## 🎬 第一轮对话：项目介绍

### 你说：
```
你好！我需要在这个 HPC 集群上运行一个深度学习项目。项目叫 KAVA，是一个论文复现项目，需要训练 12 个深度学习模型。

项目代码已经上传到 ~/kava 目录，包含完整的自动化脚本。

我需要你帮我：
1. 验证环境配置
2. 运行自动化部署脚本
3. 监控训练进度
4. 收集最终结果

我会一步步告诉你具体命令，你执行后告诉我结果，可以吗？
```

### AI 可能回复：
```
好的，我可以帮你完成这个任务。请告诉我第一步需要执行什么命令。
```

---

## 🎬 第二轮对话：验证环境

### 你说：
```
第一步：验证环境

请切换到项目目录并运行验证脚本：

cd ~/kava
bash verify_deployment.sh

这个脚本会检查：
- 所有必需的文件是否存在
- 脚本权限是否正确
- SLURM 环境是否可用
- 磁盘空间是否充足

请把完整的输出告诉我，特别注意是否有 ✗（红色叉号）或错误。
```

### AI 可能的回复情况：

#### 情况 A：全部通过
```
验证完成，所有检查都显示 ✓（绿色对勾）：
✓ 核心文件检查
✓ 配置文件检查
✓ 源代码模块
✓ 脚本权限
✓ SLURM 环境
✓ 磁盘空间：25GB 可用

验证通过！所有检查都成功。
```

**你的回复：**
```
太好了！环境验证通过。现在进入第二步：配置环境。

请运行：
bash setup_hpc.sh

这会自动配置 HuggingFace 缓存目录和必要的权限。
```

#### 情况 B：有警告
```
验证完成，有 2 个警告：
⚠ xxx.sh 使用 Windows 换行符 (CRLF)
⚠ 环境变量 HF_HOME 未设置
```

**你的回复：**
```
看到有换行符警告，这很正常（项目从 Windows 上传的）。

请继续运行配置脚本，它会自动修复这些问题：
bash setup_hpc.sh
```

#### 情况 C：有错误
```
验证失败：2 个错误
✗ 缺少文件: train.py
✗ sbatch 命令不可用
```

**你的回复：**
```
看起来有些问题。请先检查：

1. 确认当前目录：pwd
2. 列出目录内容：ls -la

然后告诉我输出，我会帮你解决。
```

---

## 🎬 第三轮对话：配置环境

### 你说（接上面情况 A）：
```
现在运行配置脚本：

bash setup_hpc.sh

这会：
- 配置 HuggingFace 缓存到 $HOME/.cache/huggingface
- 创建输出目录
- 设置脚本权限
- 添加环境变量到 ~/.bashrc

完成后请运行：
source ~/.bashrc

然后告诉我是否成功。
```

### AI 可能回复：
```
配置完成！输出显示：
✓ 项目目录正确
✓ 已添加到 ~/.bashrc
✓ 目录结构创建完成
✓ 脚本权限设置完成
✓ 磁盘空间充足 (25GB available)
✓ sbatch 命令可用
✓ compute 分区可访问

设置完成！
```

**你的回复：**
```
完美！现在可以启动训练了。
```

---

## 🎬 第四轮对话：启动训练

### 你说：
```
现在运行一键启动脚本：

bash start.sh --method mirror

参数说明：
- --method mirror：使用中国镜像站加速模型下载（推荐）

这个脚本会自动：
1. 再次验证环境
2. 创建 Python 环境（conda）
3. 安装依赖
4. 下载模型（~19GB，约 30-60 分钟）
5. 提交 12 个 SLURM 训练任务

过程可能需要 30-90 分钟，请耐心等待。

完成后请告诉我：
- 显示提交了多少个任务？
- 最后的输出是什么？
```

### AI 可能回复：

#### 情况 A：成功
```
脚本执行完成！

[步骤 1/5] 前置检查 - 完成
[步骤 2/5] 环境配置 - 完成（创建了 kava_env）
[步骤 3/5] 模型下载 - 完成（下载了 3 个模型，用时 45 分钟）
[步骤 4/5] 任务提交 - 完成（提交了 12 个任务）
[步骤 5/5] 监控配置 - 完成

✅ 启动完成！训练任务已提交到 SLURM 队列

任务 ID: 
123456, 123457, 123458, 123459,
123460, 123461, 123462, 123463,
123464, 123465, 123466, 123467

下一步操作：
1. 监控任务进度：bash monitor_jobs.sh
2. 查看队列状态：squeue -u $USER
```

**你的回复：**
```
太好了！12 个任务都成功提交了。

现在请运行：
squeue -u $USER

告诉我任务的状态（RUNNING 或 PENDING）。
```

#### 情况 B：部分失败
```
模型下载失败：ConnectionError: timeout
```

**你的回复：**
```
网络超时了。让我们尝试其他方法：

方法 1：重试（网络可能临时故障）
bash start.sh --method mirror --skip-env

方法 2：使用直连
bash start.sh --method direct --skip-env

方法 3：检查网络
ping -c 3 huggingface.co
ping -c 3 hf-mirror.com

请先试方法 1，告诉我结果。
```

---

## 🎬 第五轮对话：监控进度

### 你说：
```
现在检查任务状态：

squeue -u $USER

这会显示你的所有 SLURM 任务。

正常情况应该看到：
- 12 个任务
- 状态是 RUNNING 或 PENDING
- 任务名类似 kava_llama1b_aug_42

请把输出告诉我。
```

### AI 可能回复：

#### 情况 A：任务运行中
```
JOBID    PARTITION  NAME                   ST  TIME
123456   compute    kava_llama1b_aug_42    R   5:23
123457   compute    kava_llama1b_aug_123   R   5:23
123458   compute    kava_llama1b_aug_456   R   5:23
... (共 12 个任务)

9 个任务状态为 R (RUNNING)
3 个任务状态为 PD (PENDING)
```

**你的回复：**
```
完美！任务正在运行。9 个已经开始训练，3 个在等待资源。

让我们看看训练日志，确认一切正常：

tail -30 outputs/logs/kava_123456_0.out

（将 123456 替换为你的第一个任务 ID）

看看日志是否显示训练在进行？
```

#### 情况 B：全部排队
```
所有 12 个任务都显示 PD (PENDING)
原因：Resources
```

**你的回复：**
```
任务在等待 GPU 资源，这很正常。集群可能比较繁忙。

可以查看预计开始时间：
squeue -u $USER --start

任务会自动开始，不需要手动干预。

你可以：
1. 等待任务开始（推荐）
2. 定期检查：watch -n 300 'squeue -u $USER'  # 每 5 分钟刷新
```

---

## 🎬 第六轮对话：查看日志

### 你说：
```
查看训练日志，确认一切正常：

tail -50 outputs/logs/kava_*.out | head -100

或者查看具体某个任务：
tail -30 outputs/logs/kava_<第一个任务ID>_0.out

看看日志中是否有：
- Loading model... 
- Training epoch 1/3...
- Loss: xxx

如果有这些内容，说明训练正常进行。
```

### AI 可能回复：
```
日志显示：
========================================
Job ID: 123456
Config: llama1b_aug
Seed: 42
========================================
Loading CUDA module...
Loading Anaconda...
Activating conda environment...
CUDA Version: 11.8
GPU Info: Tesla A100-SXM4-80GB, 81920 MiB
PyTorch: 2.0.1, CUDA Available: True
========================================
Loading model: meta-llama/Llama-3.2-1B-Instruct
Loading tokenizer...
Applying LoRA...
Training epoch 1/3...
Step 10/500: loss=2.456, lr=4.8e-05
Step 20/500: loss=2.234, lr=4.6e-05
...
```

**你的回复：**
```
完美！训练正在正常进行。

现在任务已经启动，预计需要 36-48 小时完成所有训练。

你可以：
1. 使用监控脚本：bash monitor_jobs.sh
2. 等待所有任务完成（squeue -u $USER 显示为空）
3. 然后运行收集脚本：bash collect_results.sh

我建议设置一个定时检查：
*/30 * * * * cd ~/kava && bash monitor_jobs.sh > monitor.log 2>&1

这会每 30 分钟自动检查一次进度。

任务完成后（大约 1-2 天），请告诉我，我们将收集结果。
```

---

## 🎬 最终对话：收集结果

### 你说（1-2 天后）：
```
训练应该差不多完成了。请检查：

squeue -u $USER

如果显示为空（没有任务），说明全部完成。

然后运行结果收集脚本：
bash collect_results.sh

这会聚合所有 12 个任务的结果。
```

### AI 回复：
```
squeue 显示为空，所有任务已完成。

运行 collect_results.sh：
✓ 找到 12 个结果文件
✓ 聚合完成

结果保存在：outputs/aggregated_results.csv
```

**你说：**
```
太好了！最后一步，查看结果：

cat outputs/aggregated_results.csv | column -t -s,

这会显示所有模型在 GSM8K、GSM8K-Hard、SVAMP 上的准确率。

请把这个表格发给我。
```

### AI 回复：
```
config           seed  gsm8k_acc  gsm8k_hard_acc  svamp_acc  avg_passes
llama1b_aug      42    75.2       62.8            78.5       6.2
llama1b_aug      123   74.8       63.1            77.9       6.3
llama1b_aug      456   75.5       62.5            78.2       6.1
llama1b_aug_nl   42    76.1       63.5            79.2       5.8
... (共 12 行)
```

**你说：**
```
完美！KAVA 项目复现成功！🎉

结果符合预期，准确率在论文报告范围内（±2%）。

总结：
✅ 12 个模型训练完成
✅ 3 个数据集评估完成
✅ 结果已聚合并保存

如果需要进一步分析，结果文件在：
- 详细结果：outputs/results/*.json
- 汇总结果：outputs/aggregated_results.csv
- 训练日志：outputs/logs/*.log
- 模型检查点：outputs/checkpoints/

感谢你的帮助！
```

---

## 📋 快速命令总结

如果 AI 需要快速参考，提供这个列表：

```bash
# 1. 验证环境
cd ~/kava && bash verify_deployment.sh

# 2. 配置环境
bash setup_hpc.sh && source ~/.bashrc

# 3. 启动训练
bash start.sh --method mirror

# 4. 检查状态
squeue -u $USER

# 5. 查看日志
tail -30 outputs/logs/kava_*.out

# 6. 监控进度
bash monitor_jobs.sh

# 7. 收集结果（完成后）
bash collect_results.sh

# 8. 查看结果
cat outputs/aggregated_results.csv | column -t -s,
```

---

## 💡 沟通技巧

### 1. 清晰的指令
- ✅ "请运行：bash xxx.sh"
- ❌ "你可以试试运行那个脚本"

### 2. 要求完整输出
- ✅ "请把完整的输出告诉我"
- ❌ "告诉我结果"

### 3. 分步执行
- ✅ "先做 A，完成后再做 B"
- ❌ "做 A、B、C"（可能会跳步骤）

### 4. 明确期望
- ✅ "应该看到 12 个任务，状态为 RUNNING"
- ❌ "看看任务状态"

### 5. 提供上下文
- ✅ "如果看到错误 X，运行命令 Y"
- ❌ "如果有问题就处理一下"

---

使用这个对话脚本，你可以引导 AI 助手一步步完成整个项目！🚀
