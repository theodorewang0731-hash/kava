"""
Evaluation script for KAVA models.
Evaluates on GSM8k, GSM8k-Hard, and SVAMP with latent reasoning.
"""

import argparse
import os
import torch
import yaml
from tqdm import tqdm
from typing import Dict, List
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.latent_reasoning import LatentReasoningModule
from src.evaluation_datasets import EvaluationDatasets
from src.utils import exact_match_numeric, calculate_accuracy, format_metrics


class KAVAEvaluator:
    """
    Evaluator for KAVA model on math reasoning tasks.
    
    Implements:
    - Latent-only inference (no explicit CoT)
    - Forward pass counting
    - Exact match accuracy
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Args:
            checkpoint_path: Path to saved KAVA checkpoint
            config: Configuration dict
            device: Device to run on
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.load_model(checkpoint_path)
        
        # Initialize latent module
        self.latent_module = LatentReasoningModule(
            model=self.model,
            num_latent_tokens=config['latent']['num_tokens'],
            num_iterations=config['latent']['num_iterations'],
            init_strategy="embedding"
        )
        
        # Forward pass counter
        self.forward_count = 0
    
    def load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        print(f"Loading model from: {checkpoint_path}")
        
        # 检查是否为本地路径
        model_name = self.config['model']['name']
        is_local_path = os.path.exists(model_name) or model_name.startswith('/')
        use_local = is_local_path or os.environ.get('HUGGINGFACE_HUB_OFFLINE') == '1'
        
        print(f"Base model: {model_name}")
        print(f"Loading mode: {'Local path' if is_local_path else 'Offline' if use_local else 'Online'}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            local_files_only=use_local  # ✅ 强制使用本地文件
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            local_files_only=True  # ✅ checkpoint 总是本地的
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully")
    
    def generate_answer(
        self,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        count_forwards: bool = True
    ) -> tuple[str, int]:
        """
        Generate answer using latent reasoning (KAVA inference).
        
        Process (strictly following paper Section 4):
        1. Tokenize question Q
        2. Initialize M=24 latent tokens Z
        3. Run T=3 Jacobi iterations (3 forward passes)
        4. Generate answer A autoregressively with Q+Z as prefix
        
        Args:
            question: Input math question
            max_new_tokens: Maximum answer tokens to generate
            temperature: Sampling temperature (0 = greedy decoding, paper default)
            count_forwards: Whether to count forward passes for Table 2
        
        Returns:
            (answer_text, forward_count): Generated answer and number of forward passes
        """
        # Reset forward counter
        if count_forwards:
            self.forward_count = 0
        
        # Format question (same as training)
        question_prompt = f"Question: {question.strip()}\n\nAnswer:"
        
        # Tokenize question
        inputs = self.tokenizer(
            question_prompt,
            return_tensors="pt",
            padding=False,
            truncation=False
        ).to(self.device)
        
        # Get special token IDs
        bot_token_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        
        with torch.no_grad():
            # === STEP 1: Run T=3 Jacobi iterations ===
            # This generates M=24 latent tokens through parallel reasoning
            if count_forwards:
                self.forward_count += self.config['latent']['num_iterations']  # +3
            
            # Get question embeddings
            question_embeds = self.model.get_input_embeddings()(inputs['input_ids'])
            
            # Initialize latent tokens
            latent_embeds = self.latent_module.initialize_latent_tokens(
                batch_size=1,
                device=self.device,
                bot_token_id=bot_token_id
            )
            
            # Run Jacobi iterations
            for t in range(self.config['latent']['num_iterations']):
                latent_embeds, _ = self.latent_module.jacobi_iteration(
                    question_embeds=question_embeds,
                    question_attention_mask=inputs['attention_mask'],
                    latent_embeds=latent_embeds,
                    iteration=t
                )
            
            # === STEP 2: Generate answer autoregressively ===
            # Prefix: [Question tokens] [<bot>] [Latent tokens] [<eot>]
            
            # Add <bot> token
            bot_embed = self.model.get_input_embeddings()(
                torch.tensor([[bot_token_id]], device=self.device)
            )
            
            # Add <eot> token after latents
            eot_embed = self.model.get_input_embeddings()(
                torch.tensor([[eot_token_id]], device=self.device)
            )
            
            # Build prefix: Q + <bot> + Z + <eot>
            prefix_embeds = torch.cat([
                question_embeds,
                bot_embed,
                latent_embeds,
                eot_embed
            ], dim=1)
            
            # Create attention mask
            prefix_len = prefix_embeds.size(1)
            prefix_mask = torch.ones(
                1, prefix_len,
                dtype=inputs['attention_mask'].dtype,
                device=self.device
            )
            
            # Greedy decoding
            generated_ids = []
            current_embeds = prefix_embeds
            current_mask = prefix_mask
            
            for step in range(max_new_tokens):
                # Forward pass to get next token logits
                if count_forwards:
                    self.forward_count += 1  # +1 per answer token
                
                outputs = self.model(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    use_cache=False
                )
                
                # Get logits for next token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Greedy decoding (temperature=0) or sampling
                if temperature == 0.0:
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                else:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # Stop conditions
                token_id = next_token_id.item()
                
                # Stop if EOS or special tokens
                if token_id in [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.pad_token_id
                ]:
                    break
                
                # Also stop on newline patterns (answer complete)
                if len(generated_ids) > 0:
                    partial_text = self.tokenizer.decode(generated_ids + [token_id])
                    if '\n\n' in partial_text or len(generated_ids) > 50:
                        # Answer typically short for GSM8k
                        break
                
                generated_ids.append(token_id)
                
                # Append to sequence for next iteration
                next_embed = self.model.get_input_embeddings()(
                    next_token_id.unsqueeze(0)
                )
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones(1, 1, dtype=current_mask.dtype, device=self.device)
                ], dim=1)
        
        # Decode generated answer
        answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer_text, self.forward_count
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> Dict:
        """
        Evaluate on a dataset with exact match accuracy and forward pass counting.
        
        Implements paper Section 4 evaluation protocol:
        - Greedy decoding (temperature=0, top_p=1)
        - Extract numerical answer
        - Exact match accuracy
        - Count forward passes per sample
        
        Args:
            dataset_name: "gsm8k", "gsm8k-hard", or "svamp"
            split: Dataset split to evaluate on
            max_samples: Maximum number of samples (None = all)
        
        Returns:
            Dict with accuracy, forward passes, predictions
        """
        print(f"\nEvaluating on {dataset_name} ({split})...")
        
        # Load dataset using improved loader
        if dataset_name == "gsm8k":
            dataset = EvaluationDatasets.load_gsm8k(split=split)
        elif dataset_name == "gsm8k-hard":
            dataset = EvaluationDatasets.load_gsm8k_hard()
        elif dataset_name == "svamp":
            dataset = EvaluationDatasets.load_svamp()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        correct = 0
        total = 0
        total_forwards = 0
        predictions = []
        
        # Track detailed statistics
        forward_counts = []
        
        # Evaluate
        for idx, sample in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            # Handle different dataset formats
            if 'question' in sample:
                question = sample['question']
                answer = sample['answer']
            elif 'Question' in sample:
                question = sample['Question']
                answer = sample['Answer']
            else:
                print(f"Warning: Unknown format in sample {idx}, skipping")
                continue
            
            # Extract ground truth number
            gt_number = EvaluationDatasets.extract_numerical_answer(answer)
            if gt_number is None:
                print(f"Warning: Cannot extract GT answer from: {answer}")
                continue
            
            # Generate prediction
            try:
                pred_text, forward_count = self.generate_answer(
                    question,
                    max_new_tokens=self.config['evaluation']['max_new_tokens'],
                    temperature=self.config['evaluation']['temperature'],
                    count_forwards=True
                )
                
                # Extract predicted number
                pred_number = EvaluationDatasets.extract_numerical_answer(pred_text)
                
                # Check correctness (exact match with small tolerance for float)
                is_correct = (
                    pred_number is not None and 
                    abs(pred_number - gt_number) < 1e-3
                )
                
                if is_correct:
                    correct += 1
                
                total += 1
                total_forwards += forward_count
                forward_counts.append(forward_count)
                
                predictions.append({
                    'question': question,
                    'ground_truth': gt_number,
                    'ground_truth_text': answer,
                    'prediction': pred_number,
                    'prediction_text': pred_text,
                    'correct': is_correct,
                    'forward_passes': forward_count
                })
                
                # Print progress every 50 samples
                if (idx + 1) % 50 == 0:
                    current_acc = correct / total if total > 0 else 0
                    current_fp = sum(forward_counts) / len(forward_counts)
                    print(f"  [{idx+1}/{len(dataset)}] Accuracy: {current_acc*100:.1f}%, Avg FP: {current_fp:.1f}")
            
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        avg_forwards = total_forwards / total if total > 0 else 0
        std_forwards = torch.tensor(forward_counts).float().std().item() if len(forward_counts) > 1 else 0
        
        results = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_forward_passes': avg_forwards,
            'std_forward_passes': std_forwards,
            'predictions': predictions
        }
        
        print(f"\n{'='*60}")
        print(f"Results on {dataset_name}:")
        print(f"  Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
        print(f"  Avg Forward Passes: {avg_forwards:.2f} ± {std_forwards:.2f}")
        print(f"{'='*60}")
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KAVA model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k"],
        help="Datasets to evaluate on (gsm8k, gsm8k-hard, svamp)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples per dataset (for quick testing)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results.yaml",
        help="Output file for results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator
    evaluator = KAVAEvaluator(
        checkpoint_path=args.checkpoint,
        config=config,
        device="cuda"
    )
    
    # Evaluate on each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        results = evaluator.evaluate_dataset(
            dataset_name=dataset_name,
            split="test",
            max_samples=args.max_samples
        )
        
        all_results[dataset_name] = {
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total'],
            'avg_forward_passes': results['avg_forward_passes']
        }
    
    # Save results as YAML
    with open(args.output, 'w') as f:
        yaml.dump(all_results, f)
    
    # Also save as JSON for easier parsing
    json_output = args.output.replace('.yaml', '.json') if args.output.endswith('.yaml') else args.output + '.json'
    with open(json_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {args.output}")
    print(f"  - {json_output}")
    
    # Print summary
    print("\n=== Summary ===")
    for dataset_name, metrics in all_results.items():
        print(f"{dataset_name}: {metrics['accuracy'] * 100:.2f}% ({metrics['correct']}/{metrics['total']}), "
              f"Avg FP: {metrics['avg_forward_passes']:.1f}")
    
    return all_results


if __name__ == "__main__":
    from typing import Optional
    import json
    main()
