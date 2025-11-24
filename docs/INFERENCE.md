# Inference Guide

Complete guide for using trained KAVA models for inference.

## Overview

The `inference.py` script provides:
- **Interactive mode**: Chat-like interface for testing
- **Batch mode**: Process multiple questions at once
- **Latent reasoning**: Use compressed KV-cache for efficient generation
- **Standard generation**: Baseline comparison without latent tokens
- **Forward pass counting**: Track computational cost

---

## Quick Start

### Interactive Mode

Chat with a trained KAVA model:

```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --mode interactive
```

**Example session**:
```
Question: What is 25% of 80?

Generating answer...

Answer: 25% of 80 is 20.
Forward passes: 27
```

**Commands**:
- `/latent on|off` - Toggle latent reasoning
- `/temp <float>` - Set temperature (0 = greedy)
- `/quit` - Exit

---

## Modes

### 1. Interactive Mode

Real-time question answering with configurable settings.

**Usage**:
```bash
python inference.py \
    --checkpoint <checkpoint_path> \
    --config <config_path> \
    --mode interactive
```

**Example session**:
```
Question: A store has 120 apples. They sell 30% in the morning. How many are left?

Answer: They sold 0.3 × 120 = 36 apples.
So 120 - 36 = 84 apples are left.
Forward passes: 32

Question: /latent off

✓ Latent reasoning: OFF

Question: What is 15 × 24?

Answer: 15 × 24 = 360
Forward passes: 18
```

### 2. Batch Mode (Command Line)

Process multiple questions from command line:

```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --questions "What is 5+3?" "Calculate 12×7" \
    --output_file results.txt
```

**Output** (`results.txt`):
```
Q: What is 5+3?
A: 5 + 3 = 8
--------------------------------------------------------------------------------
Q: Calculate 12×7
A: 12 × 7 = 84
--------------------------------------------------------------------------------
```

### 3. Batch Mode (File Input)

Process questions from a file:

**Input file** (`questions.txt`):
```
What is 25% of 80?
Calculate the area of a rectangle with length 12 and width 8.
If John has $50 and spends $18, how much does he have left?
```

**Command**:
```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --input_file questions.txt \
    --output_file answers.txt
```

---

## Options

### Latent Reasoning

**Enable** (default):
```bash
python inference.py --checkpoint <path> --config <path> --use_latent
```

**Disable** (baseline):
```bash
python inference.py --checkpoint <path> --config <path> --no_latent
```

**Comparison**:
```bash
# With latent reasoning (M=24, T=3)
python inference.py --checkpoint <path> --config <path> --use_latent --output_file latent_answers.txt

# Without latent reasoning
python inference.py --checkpoint <path> --config <path> --no_latent --output_file standard_answers.txt
```

### Generation Length

Control maximum answer length:

```bash
python inference.py \
    --checkpoint <path> \
    --config <path> \
    --max_new_tokens 512  # Default: 256
```

### Sampling vs. Greedy

**Greedy decoding** (deterministic, default):
```bash
python inference.py --checkpoint <path> --config <path>  # temperature=0.0 by default
```

**Sampling** (creative):
```bash
# In interactive mode
Question: /temp 0.7

# Or set in code (modify inference.py generate() call)
```

### Device Selection

**GPU**:
```bash
python inference.py --checkpoint <path> --config <path> --device cuda
```

**CPU** (slower):
```bash
python inference.py --checkpoint <path> --config <path> --device cpu
```

---

## Examples

### Example 1: Quick Test

Test a checkpoint with one question:

```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --questions "What is 15% of 200?" \
    --use_latent
```

### Example 2: Compare Latent vs. Standard

**With latent**:
```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --input_file test_questions.txt \
    --output_file latent_results.txt \
    --use_latent
```

**Without latent**:
```bash
python inference.py \
    --checkpoint experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch1 \
    --config configs/llama1b_aug.yaml \
    --mode batch \
    --input_file test_questions.txt \
    --output_file standard_results.txt \
    --no_latent
```

**Compare**:
```bash
diff latent_results.txt standard_results.txt
```

### Example 3: Batch Processing with Forward Counts

Modify `inference.py` to save forward counts:

```python
# In batch_generate(), add:
answers_with_counts = []
for question in questions:
    answer, forward_count = self.generate(
        question,
        use_latent=use_latent,
        return_forward_count=True
    )
    answers_with_counts.append((answer, forward_count))

# Save to output file:
for q, (a, fc) in zip(questions, answers_with_counts):
    f.write(f"Q: {q}\n")
    f.write(f"A: {a}\n")
    f.write(f"Forward passes: {fc}\n")
    f.write("-" * 80 + "\n")
```

---

## Understanding Forward Passes

### With Latent Reasoning (M=24, T=3)

**Formula**:
```
Total FP = T (Jacobi iterations) + N (answer tokens)
         = 3 + N
```

**Example**:
- Question: "What is 5+3?"
- Answer: "5 + 3 = 8" (4 tokens)
- Forward passes: 3 + 4 = **7**

### Without Latent Reasoning

**Formula**:
```
Total FP = N (answer tokens)
```

**Example**:
- Question: "What is 5+3?"
- Answer: "5 + 3 = 8" (4 tokens)
- Forward passes: **4**

### Efficiency Gains

Latent reasoning adds **T=3** overhead but enables:
1. **Better quality** (higher accuracy)
2. **More concise answers** (fewer tokens N)
3. **Net reduction** when N decreases significantly

**Paper result** (Table 2):
- Standard generation: ~60 forward passes
- KAVA latent generation: ~48 forward passes
- **Speedup**: 20% reduction

---

## Model-Specific Notes

### LLaMA 3.2 Models

**Prompt format**:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}
```

**Special tokens**:
- `<bot>`: Begin of thinking (latent region)
- `<eot>`: End of thinking

**Config**: `configs/llama1b_aug.yaml` or `configs/llama3b_aug.yaml`

### Qwen2.5 Models

**Prompt format**:
```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}
```

**Config**: `configs/qwen05b_aug.yaml`

---

## Troubleshooting

### Issue: Model generates gibberish

**Cause**: Wrong prompt format or model not fully trained.

**Solution**:
1. Check config file matches checkpoint
2. Verify training completed (check loss curve)
3. Try different checkpoint epoch

### Issue: OOM during inference

**Cause**: Model too large for GPU.

**Solution**:
```bash
# Use CPU
python inference.py --checkpoint <path> --config <path> --device cpu

# Or reduce batch size (for batch mode)
# Modify inference.py to process in smaller batches
```

### Issue: Slow generation

**Cause**: CPU inference or large model.

**Solution**:
1. Use GPU: `--device cuda`
2. Use smaller model (LLaMA 1B or Qwen 0.5B)
3. Reduce `max_new_tokens`

### Issue: Answers are too short

**Cause**: EOS token generated early.

**Solution**:
```bash
# Increase max_new_tokens
python inference.py --checkpoint <path> --config <path> --max_new_tokens 512
```

---

## API Usage (Python)

Use `KAVAInference` class in your own scripts:

```python
from inference import KAVAInference

# Initialize
model = KAVAInference(
    checkpoint_path="experiments/llama1b_gsm8k-aug/seed_42/checkpoint-epoch1",
    config_path="configs/llama1b_aug.yaml",
    device="cuda"
)

# Single question
answer = model.generate(
    question="What is 25% of 80?",
    use_latent=True,
    max_new_tokens=256
)
print(answer)

# With forward count
answer, forward_count = model.generate(
    question="What is 25% of 80?",
    use_latent=True,
    return_forward_count=True
)
print(f"Answer: {answer}")
print(f"Forward passes: {forward_count}")

# Batch processing
questions = [
    "What is 5+3?",
    "Calculate 12×7",
    "What is 25% of 80?"
]
answers = model.batch_generate(questions, use_latent=True)
for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")
```

---

## Best Practices

1. **Always use latent reasoning** for math problems (better accuracy)
2. **Use greedy decoding** (temperature=0) for deterministic results
3. **Set max_new_tokens=512** for complex problems requiring long solutions
4. **Compare checkpoints** from different epochs to find best one
5. **Log forward passes** to track efficiency gains

---

## Next Steps

- ✅ Test your checkpoint: `python inference.py --mode interactive`
- 🔄 Compare latent vs. standard generation
- 🔄 Run batch evaluation on custom questions
- 🔄 Integrate into your application using the API
