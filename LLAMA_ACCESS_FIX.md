# Llama 模型访问权限问题 - 解决方案

## 问题
下载 `meta-llama/Llama-3.2-1B-Instruct` 时出现 403 错误：
```
Repository is gated. You must be authenticated and request access.
```

---

## ✅ 解决方案 1：使用 HuggingFace Token（推荐）

### 步骤 1：获取 HuggingFace Token

1. 访问 https://huggingface.co/settings/tokens
2. 登录你的 HuggingFace 账号
3. 点击 "New token"
4. 创建一个 **Read** 权限的 token
5. 复制 token（格式类似：`hf_xxxxxxxxxxxx`）

### 步骤 2：申请 Llama 模型访问权限

1. 访问 https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. 点击 "Agree and access repository"
3. 同意 Meta 的使用条款
4. 等待几分钟（通常立即批准）

同样申请这两个模型的访问：
- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### 步骤 3：在 HPC 上设置 Token

```bash
cd "/home/rpwang/kava review"

# 方法 A：设置环境变量（临时）
export HF_TOKEN="hf_你的token"

# 方法 B：使用 huggingface-cli 登录（永久）
source venv/bin/activate
pip install -U huggingface_hub
huggingface-cli login
# 输入你的 token

# 验证登录成功
huggingface-cli whoami
```

### 步骤 4：重新运行脚本

```bash
cd "/home/rpwang/kava review"
source venv/bin/activate
export HF_TOKEN="hf_你的token"  # 如果使用方法 A
bash run_reproduce_venv.sh --method mirror
```

---

## ✅ 解决方案 2：使用无需授权的替代模型

如果不想申请权限，可以用开放模型替代：

### 替代方案

| 原模型 | 替代模型 | 参数量 | 说明 |
|--------|---------|--------|------|
| Llama-3.2-1B-Instruct | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 开放模型 |
| Llama-3.2-3B-Instruct | stabilityai/stablelm-3b-4e1t | 3B | 开放模型 |
| Qwen2.5-0.5B-Instruct | ✅ 已开放 | 0.5B | 无需授权 |

### 修改配置文件

但这**不符合论文复现要求**，不推荐。

---

## 🎯 推荐方案：使用 Token

**最简单的方法：**

```bash
# 1. 获取 token（访问 https://huggingface.co/settings/tokens）

# 2. 申请 Llama 访问权限（访问模型页面点击 Agree）

# 3. 设置 token 并运行
cd "/home/rpwang/kava review"
source venv/bin/activate
huggingface-cli login  # 输入 token
bash run_reproduce_venv.sh --method mirror
```

---

## 📋 给 AI 的完整指令

```
需要先获取 HuggingFace token 和申请 Llama 模型访问权限。

请访问：
1. https://huggingface.co/settings/tokens
   - 创建一个 Read token
   - 复制 token（格式：hf_xxxxxxxxxxxx）

2. https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - 点击 "Agree and access repository"
   
3. https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
   - 同样点击 "Agree and access repository"

获取 token 后，在 HPC 上运行：

cd "/home/rpwang/kava review"
source venv/bin/activate
huggingface-cli login
# 粘贴你的 token

然后重新运行：
bash run_reproduce_venv.sh --method mirror

这次应该能成功下载模型。
```

---

## ⚠️ 注意事项

1. **Token 保密**：不要分享你的 HuggingFace token
2. **访问批准**：Llama 模型通常立即批准访问请求
3. **Token 权限**：只需要 Read 权限即可
4. **一次性设置**：登录后会保存在 `~/.cache/huggingface/token`

---

## 🔍 验证访问权限

```bash
# 检查是否已登录
huggingface-cli whoami

# 测试访问 Llama 模型
python -c "
from huggingface_hub import list_repo_files
try:
    files = list_repo_files('meta-llama/Llama-3.2-1B-Instruct')
    print('✓ 访问权限正常')
except Exception as e:
    print(f'✗ 访问被拒绝: {e}')
"
```
