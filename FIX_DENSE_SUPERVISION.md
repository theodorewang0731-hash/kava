# KAVA 密集监督修复方案 - 代码示例

## 1. 修改 `src/latent_reasoning.py` - 保存所有迭代的 KV

### 原始 forward_student 方法（简化版）
```python
def forward_student(self, ...):
    # ...
    for t in range(self.T):
        latent_embeds, outputs = self.jacobi_iteration(...)
        if t == self.T - 1:
            final_outputs = outputs
    
    return {
        'logits': final_outputs['logits'],
        'latent_kv': extract_kv(...),  # 仅最后一步
        'hidden_states': final_outputs['hidden_states'],
    }
```

### 修复后的版本
```python
def forward_student(self, ...):
    # ...
    all_outputs = []  # ✅ 新增：保存所有迭代的输出
    
    for t in range(self.T):
        latent_embeds, outputs = self.jacobi_iteration(...)
        all_outputs.append(outputs)  # ✅ 新增：收集每一步的输出
        
        if t == self.T - 1:
            final_outputs = outputs
    
    # ✅ 新增：提取所有步骤的 KV 缓存
    all_kv_steps = []
    for t, outputs in enumerate(all_outputs):
        kv_step = extract_kv_from_output(outputs)  # [batch, ..., M, head_dim]
        all_kv_steps.append(kv_step)
    
    return {
        'logits': final_outputs['logits'],
        'all_kv_steps': all_kv_steps,  # ✅ 新增：[(KV_z1), (KV_z2), (KV_z3), (KV_a)]
        'final_kv': all_kv_steps[-1],   # 保留向后兼容
        'hidden_states': final_outputs['hidden_states'],
    }
```

---

## 2. 修改 `src/trainer.py` - 对所有步骤计算 KV 蒸馏损失

### 原始 train_step 中关于损失计算的部分
```python
def train_step(self, batch_data):
    # ... teacher forward ...
    # ... r-kv compression ...
    
    # ⚠️ 问题：仅取最后一步
    student_keys, student_values = student_outputs['latent_kv']
    
    loss, loss_dict = self.criterion(
        ...
        student_keys=student_keys.unsqueeze(0),
        student_values=student_values.unsqueeze(0),
        teacher_keys=teacher_keys_compressed.unsqueeze(0),
        teacher_values=teacher_values_compressed.unsqueeze(0),
        ...
    )
```

### 修复后的版本
```python
def train_step(self, batch_data):
    # ... existing code for teacher forward ...
    
    # ========== R-KV COMPRESSION FOR ALL STEPS ==========
    # ✅ 新增：为每一个 Jacobi 迭代步骤压缩 KV
    teacher_all_kv_steps = teacher_outputs['all_kv_steps']  # 需要 teacher 也返回
    teacher_kv_compressed_all = []
    
    for step_idx, (teacher_kv_step) in enumerate(teacher_all_kv_steps):
        # 对每一步进行 R-KV 压缩
        kv_compressed = self.rkv_compressor.compress(
            key_cache=teacher_kv_step[0],  # Keys
            value_cache=teacher_kv_step[1],  # Values
            attention_weights=teacher_attention,
            # ... other parameters ...
        )
        teacher_kv_compressed_all.append(kv_compressed)
    
    # ========== STUDENT FORWARD ==========
    student_outputs = self.latent_module.forward_student(...)
    
    # ✅ 修改：获取所有步骤的 KV
    student_kv_all_steps = student_outputs['all_kv_steps']  # [(KV_z1), (KV_z2), (KV_z3), (KV_a)]
    
    # ========== COMPUTE LOSSES ==========
    # ... prepare labels ...
    
    # ✅ 修改：传入所有步骤的 KV
    loss, loss_dict = self.criterion(
        student_logits=student_outputs['logits'],
        student_labels=student_labels,
        teacher_logits=teacher_outputs['logits'],
        teacher_labels=teacher_labels,
        student_kv_all_steps=student_kv_all_steps,  # ✅ 新增
        teacher_kv_all_steps=teacher_kv_compressed_all,  # ✅ 新增
        student_hidden_states=student_outputs['hidden_states'],
        teacher_hidden_states=teacher_outputs['hidden_states'],
        distill_token_idx=-self.config['latent']['num_tokens']-1
    )
```

---

## 3. 修改 `src/losses.py` - 支持多步骤 KV 蒸馏

### 原始 KAVALoss.forward 方法
```python
def forward(self, ..., student_keys, student_values, teacher_keys, teacher_values, ...):
    # ... compute CE losses ...
    
    # ⚠️ 问题：仅计算最后一步的 KV 蒸馏
    loss_kv = self.kv_loss(
        teacher_keys,
        teacher_values,
        student_keys,
        student_values
    )
    
    total_loss = (
        loss_student_ce +
        loss_teacher_ce +
        self.alpha1 * loss_codi +
        self.alpha2 * loss_kv  # 仅最后一步
    )
```

### 修复后的版本
```python
def forward(
    self,
    student_logits, student_labels,
    teacher_logits, teacher_labels,
    student_kv_all_steps,  # ✅ 新增：[(KV_z1_s, KV_z1_s), ..., (KV_a_s, KV_a_s)]
    teacher_kv_all_steps,  # ✅ 新增：[(KV_z1_t, KV_z1_t), ..., (KV_a_t, KV_a_t)]
    student_hidden_states, teacher_hidden_states,
    distill_token_idx,
    ...
):
    """
    Compute full KAVA loss with dense supervision.
    
    New: Supervise all Jacobi iteration steps, not just the final answer.
    """
    
    # 1. Student CE loss
    loss_student_ce = self.compute_ce_loss(student_logits, student_labels)
    
    # 2. Teacher CE loss
    loss_teacher_ce = self.compute_ce_loss(teacher_logits, teacher_labels)
    
    # 3. CODI loss
    loss_codi = self.codi_loss(
        teacher_hidden_states,
        student_hidden_states,
        distill_token_idx
    )
    
    # 4. ✅ 修改：多步骤 KV 蒸馏损失（密集监督）
    kv_losses_per_step = []
    
    for step_idx, (student_kv_step, teacher_kv_step) in enumerate(
        zip(student_kv_all_steps, teacher_kv_all_steps)
    ):
        # 对每个 Jacobi 迭代步骤计算 KV 蒸馏损失
        loss_kv_step = self.kv_loss(
            teacher_keys=teacher_kv_step[0],    # Teacher keys at step t
            teacher_values=teacher_kv_step[1],  # Teacher values at step t
            student_keys=student_kv_step[0],    # Student keys at step t
            student_values=student_kv_step[1]   # Student values at step t
        )
        kv_losses_per_step.append(loss_kv_step)
        
        # 可选：记录每一步的损失用于调试
        self.last_kv_losses = kv_losses_per_step
    
    # 对所有步骤的 KV 损失取平均（密集监督）
    loss_kv_total = torch.stack(kv_losses_per_step).mean()
    
    # Total loss
    total_loss = (
        loss_student_ce +
        loss_teacher_ce +
        self.alpha1 * loss_codi +
        self.alpha2 * loss_kv_total  # ✅ 修改：使用所有步骤的平均损失
    )
    
    # Return with detailed loss breakdown
    return total_loss, {
        'loss_student_ce': loss_student_ce.item() if hasattr(loss_student_ce, 'item') else float(loss_student_ce),
        'loss_teacher_ce': loss_teacher_ce.item() if hasattr(loss_teacher_ce, 'item') else float(loss_teacher_ce),
        'loss_codi': loss_codi.item() if hasattr(loss_codi, 'item') else float(loss_codi),
        'loss_kv_total': loss_kv_total.item() if hasattr(loss_kv_total, 'item') else float(loss_kv_total),
        'kv_losses_per_step': [
            l.item() if hasattr(l, 'item') else float(l) 
            for l in kv_losses_per_step
        ],  # ✅ 新增：用于监控每一步的监督效果
        'total_loss': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss),
    }
```

---

## 4. 修改 `forward_teacher()` - 也要保存中间步骤

### 需要确认的修改
```python
def forward_teacher(self, input_ids, attention_mask, ...):
    """
    Teacher forward pass.
    
    ✅ 新增要求：也要返回所有中间步骤的 KV 缓存（如果是多步推理）
    """
    
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True
    )
    
    # ✅ 如果教师也使用 Jacobi 迭代，需要保存所有步骤
    # ✅ 如果教师只是一步前向，则仅保存该步骤（但包装成列表便于与学生对齐）
    
    return {
        'logits': outputs.logits,
        'hidden_states': outputs.hidden_states,
        'past_key_values': outputs.past_key_values,
        'all_kv_steps': [extract_kv(outputs)],  # 包装成列表，便于与学生对齐
        'attentions': outputs.attentions,
    }
```

---

## 5. 配置文件 - 确认超参数

### `configs/llama1b_aug.yaml`
```yaml
# ✅ 确保这些值正确
loss:
  alpha1_codi: 10.0       # ✓ 论文 Table 6
  alpha2_kv: 1.0          # ✓ 论文 Table 6
  kv_loss_type: "smooth_l1"
  layerwise_std: true     # ✓ LLaMA-1B 需要
```

### `configs/llama3b_aug.yaml`
```yaml
# ✅ 注意 3B 模型的不同配置
loss:
  alpha1_codi: 20.0       # ✓ 论文 Table 6：更大的模型权重加倍
  alpha2_kv: 2.0          # ✓ 论文 Table 6：KV 权重也加倍
  kv_loss_type: "smooth_l1"
  layerwise_std: false    # ✓ LLaMA-3B 不需要层级归一化
```

---

## 测试验证

修复后，验证输出日志应该包含：

```
[Step 100] Loss breakdown:
  - student_ce: 2.31
  - teacher_ce: 1.89
  - codi: 0.45
  - kv_step_1: 0.82        # ✅ 新增：每一步的 KV 损失
  - kv_step_2: 0.78        # ✅ 新增
  - kv_step_3: 0.75        # ✅ 新增
  - kv_step_4: 0.71        # ✅ 新增：最后答案步骤
  - kv_total: 0.77 (mean)  # ✅ 新增：平均值
  - total: 5.83
```

---

## 性能预期

修复前后的准确率对比：

```
未修复：
  GSM8K: ~81-82% (缺少中间监督)
  GSM8K-Hard: ~68-69%
  SVAMP: ~75-76%

修复后（预期）：
  GSM8K: ~83.7% (+1.7-2.7%) ✓
  GSM8K-Hard: ~70.5%
  SVAMP: ~77.3%
```

---

## 优先级总结

```
🔴 高优先级（关键）
  1. 保存所有 Jacobi 迭代的 KV ← 开始这里
  2. 计算所有步骤的 KV 蒸馏损失

🟡 中优先级（重要）
  3. 教师端也返回中间步骤
  4. 验证超参数配置

🟢 低优先级（可选）
  5. 性能监控日志
  6. 可视化调试工具
```

---

**估计修复时间**：2-3 小时
**影响范围**：核心训练逻辑，需要仔细测试
**回归风险**：中等（修改了核心损失计算）
