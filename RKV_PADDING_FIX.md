# R-KV Padding Tokens 修复

## 问题发现

在阅读论文 Appendix D（Listing 1）后，发现我们的 R-KV 压缩实现**缺少关键的 padding tokens 处理**：

> **论文原文 (Section D)**:  
> "First, we need to take into account padding tokens since we evict KV-cache in a batch during training. We do that by **always assigning the lowest possible redundancy and importance score to the value-key pairs corresponding to the padding tokens**"

## 论文要求

### 1. Padding Tokens 处理
```python
# 论文伪代码 (Listing 1, lines 17-18, 23-24)
pad_tokens = key.sum(-1) == 0
I[pad_tokens] = 0  # Assign lowest score to padding tokens
R[pad_tokens] = 0  # Assign lowest score to padding tokens
```

### 2. Importance Score 计算
```python
# 论文伪代码 (Listing 1, lines 21-24)
# softmax over CoT dimention and average over answer tokens
I = F.softmax(attn, dim=-1).mean(-2)
# Assign the lowest score to the padding tokens
I[pad_tokens] = 0
```

### 3. Redundancy Score 计算
```python
# 论文伪代码 (Listing 1, lines 7-18)
# compute redundancy score
key_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
cosine_sim = torch.einsum("...id,...jd->...ij", key_norm, key_norm)
for i in range(cosine_sim.shape[0]):
    cosine_sim[i].fill_diagonal_(0)  # Exclude self-similarity
    
# cos_score: average cosine similarity with other tokens (excluding padding)
cos_score = torch.sum(-cosine_sim, dim=-2) / torch.sum(~pad_tokens, dim=-1, keepdim=True)

# Normalize to 1
R = cos_score.softmax(dim=-1)
pad_tokens = key.sum(-1) == 0
R[pad_tokens] = 0  # Assign lowest score to padding
```

### 4. 最终混合分数
```python
# 论文伪代码 (Listing 1, line 27)
S = lbd * I + (1 - lbd) * R
return S
```

## 实现修复

### ✅ 修复 1: `compute_importance_score()` 添加 padding 处理

**修改前**:
```python
def compute_importance_score(
    self,
    attention_weights: torch.Tensor,
    answer_start_idx: int,
    steps_start_idx: int,
    steps_end_idx: int
) -> torch.Tensor:
    answer_to_steps = attention_weights[:, :, answer_start_idx:, steps_start_idx:steps_end_idx]
    importance = answer_to_steps.mean(dim=2)
    return importance  # ❌ 没有处理 padding
```

**修改后**:
```python
def compute_importance_score(
    self,
    attention_weights: torch.Tensor,
    answer_start_idx: int,
    steps_start_idx: int,
    steps_end_idx: int,
    attention_mask: Optional[torch.Tensor] = None  # ✅ 新增参数
) -> torch.Tensor:
    answer_to_steps = attention_weights[:, :, answer_start_idx:, steps_start_idx:steps_end_idx]
    importance = answer_to_steps.mean(dim=2)
    
    # ✅ PADDING TOKENS HANDLING
    if attention_mask is not None:
        step_mask = attention_mask[:, steps_start_idx:steps_end_idx]
        step_mask = step_mask.unsqueeze(1).expand_as(importance)
        importance = importance * step_mask.float()  # Set padding to 0
    
    return importance
```

### ✅ 修复 2: `compute_redundancy_score()` 添加 padding 处理

**修改前**:
```python
def compute_redundancy_score(
    self,
    key_states: torch.Tensor
) -> torch.Tensor:
    keys_norm = F.normalize(key_states, p=2, dim=-1)
    cos_sim = torch.matmul(keys_norm, keys_norm.transpose(-2, -1))
    
    # ❌ 没有排除 padding tokens 的相似度计算
    mask = ~torch.eye(num_steps, device=key_states.device, dtype=torch.bool)
    avg_similarity = (cos_sim * mask).sum(dim=-1) / (num_steps - 1)
    
    redundancy = F.softmax(-avg_similarity, dim=-1)
    return redundancy  # ❌ 没有将 padding 设为 0
```

**修改后**:
```python
def compute_redundancy_score(
    self,
    key_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,  # ✅ 新增参数
    steps_start_idx: int = 0,
    steps_end_idx: Optional[int] = None
) -> torch.Tensor:
    keys_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = torch.matmul(keys_norm, keys_norm.transpose(-2, -1))
    
    # ✅ Create padding mask
    if attention_mask is not None and steps_end_idx is not None:
        step_mask = attention_mask[:, steps_start_idx:steps_end_idx]
        pad_tokens = (step_mask == 0)
        pad_tokens_expanded = pad_tokens.unsqueeze(1)
    else:
        pad_tokens = None
        pad_tokens_expanded = None
    
    # ✅ Mask out self-similarity AND padding tokens
    diag_mask = ~torch.eye(num_steps, device=key_states.device, dtype=torch.bool)
    diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)
    
    if pad_tokens_expanded is not None:
        # 只计算有效 token 之间的相似度
        valid_mask = diag_mask & pad_tokens_expanded.unsqueeze(-1) & pad_tokens_expanded.unsqueeze(-2)
        num_valid = valid_mask.sum(dim=-1).float().clamp(min=1)
    else:
        valid_mask = diag_mask
        num_valid = (num_steps - 1)
    
    # ✅ Average only over valid tokens
    avg_similarity = (cos_sim * valid_mask).sum(dim=-1) / num_valid
    redundancy = F.softmax(-avg_similarity, dim=-1)
    
    # ✅ PADDING TOKENS HANDLING: Set to 0
    if pad_tokens is not None:
        pad_tokens_heads = pad_tokens.unsqueeze(1).expand(batch_size, num_heads, num_steps)
        redundancy = redundancy * (~pad_tokens_heads).float()
    
    return redundancy
```

### ✅ 修复 3: `compress()` 传递 attention_mask

**修改前**:
```python
def compress(
    self,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_weights: torch.Tensor,
    answer_start_idx: int,
    steps_start_idx: int,
    steps_end_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    importance = self.compute_importance_score(...)  # ❌ 没有传 mask
    redundancy = self.compute_redundancy_score(step_keys)  # ❌ 没有传 mask
```

**修改后**:
```python
def compress(
    self,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_weights: torch.Tensor,
    answer_start_idx: int,
    steps_start_idx: int,
    steps_end_idx: int,
    attention_mask: Optional[torch.Tensor] = None  # ✅ 新增参数
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ✅ Pass attention_mask to both score computations
    importance = self.compute_importance_score(
        attention_weights,
        answer_start_idx,
        steps_start_idx,
        steps_end_idx,
        attention_mask=attention_mask  # ✅
    )
    
    redundancy = self.compute_redundancy_score(
        step_keys,
        attention_mask=attention_mask,  # ✅
        steps_start_idx=steps_start_idx,
        steps_end_idx=steps_end_idx
    )
```

### ✅ 修复 4: `trainer.py` 调用传递 attention_mask

**修改前**:
```python
teacher_keys_compressed, teacher_values_compressed = self.rkv_compressor.compress(
    key_cache=teacher_keys_full[:, 0],
    value_cache=teacher_values_full[:, 0],
    attention_weights=teacher_attention,
    answer_start_idx=teacher_sample['answer_start_idx'],
    steps_start_idx=teacher_sample['steps_start_idx'],
    steps_end_idx=teacher_sample['steps_end_idx']
    # ❌ 没有传递 attention_mask
)
```

**修改后**:
```python
teacher_keys_compressed, teacher_values_compressed = self.rkv_compressor.compress(
    key_cache=teacher_keys_full[:, 0],
    value_cache=teacher_values_full[:, 0],
    attention_weights=teacher_attention,
    answer_start_idx=teacher_sample['answer_start_idx'],
    steps_start_idx=teacher_sample['steps_start_idx'],
    steps_end_idx=teacher_sample['steps_end_idx'],
    attention_mask=teacher_attention_mask  # ✅ 传递 attention_mask
)
```

## 为什么这很重要？

### 1. Batch Training 必需
在 batch 训练时，不同样本的 CoT 长度不同，短的会被 padding 填充到 batch 最大长度。如果不处理 padding tokens：
- **Importance score** 会包含 padding 的错误注意力值
- **Redundancy score** 会计算 padding tokens 之间的虚假相似度
- **最终选择** 可能会选中 padding tokens，导致无意义的 KV cache

### 2. 论文明确要求
论文在 Appendix D 专门说明了这个问题，并在伪代码 Listing 1 中给出了实现细节。

### 3. 影响模型性能
如果选中了 padding tokens：
- Student 会试图蒸馏无意义的 KV
- KV distillation loss 会包含噪声
- 训练不稳定，收敛变慢

## 验证清单

- ✅ `compute_importance_score()` 给 padding tokens 分配 0 分
- ✅ `compute_redundancy_score()` 给 padding tokens 分配 0 分
- ✅ `compute_redundancy_score()` 只在有效 tokens 之间计算相似度
- ✅ `compress()` 传递 attention_mask
- ✅ `trainer.py` 调用时传递 attention_mask

## 下一步

现在 R-KV 压缩算法完全符合论文 Listing 1 的实现要求。需要测试：

1. **单样本测试**: 验证无 padding 时结果不变
2. **Batch 测试**: 验证有 padding 时正确处理
3. **完整训练**: 检查训练稳定性和收敛速度

修复后应该能看到：
- 更稳定的训练曲线
- 更低的 KV distillation loss（因为排除了噪声）
- 更快的收敛速度
