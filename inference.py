"""
Standalone inference script for KAVA model.
Supports interactive mode and batch processing.
"""

import argparse
import torch
from pathlib import Path
from typing import Optional, List
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.latent_reasoning import LatentReasoningModule


class KAVAInference:
    """
    Inference wrapper for trained KAVA models.
    
    Supports:
    - Latent-based generation (PCCoT)
    - Standard generation (baseline)
    - Forward pass counting
    - Batch processing
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            checkpoint_path: Path to trained LoRA checkpoint
            config_path: Path to config YAML
            device: Device to run on
        """
        self.device = device
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model and tokenizer
        print(f"Loading base model: {self.config['model']['name_or_path']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name_or_path'],
            trust_remote_code=True
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name_or_path'],
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA checkpoint from: {checkpoint_path}")
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model.eval()
        
        # Initialize latent reasoning module
        self.latent_module = LatentReasoningModule(
            latent_tokens=self.config['latent_reasoning']['latent_tokens'],
            num_jacobi_iterations=self.config['latent_reasoning']['num_jacobi_iterations'],
            device=device
        )
        
        print(f"✓ Model loaded on {device}")
        print(f"✓ Latent reasoning: M={self.latent_module.latent_tokens}, T={self.latent_module.num_jacobi_iterations}")
    
    def generate(
        self,
        question: str,
        use_latent: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        return_forward_count: bool = False
    ) -> str:
        """
        Generate answer for a question.
        
        Args:
            question: Input question
            use_latent: Whether to use latent reasoning
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (0 = greedy)
            return_forward_count: Whether to return forward pass count
        
        Returns:
            Generated answer text (and optionally forward count)
        """
        # Format input
        if self.config['model']['type'] == 'llama':
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif self.config['model']['type'] == 'qwen':
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = question
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        forward_count = 0
        
        if use_latent:
            # === Latent-based generation with Incremental Decoding ===
            with torch.no_grad():
                # Step 1: Initialize latent tokens Z
                Z_ids = torch.full(
                    (input_ids.size(0), self.latent_module.latent_tokens),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self.device
                )
                
                # Step 2: Run Jacobi iterations with KV-cache reuse
                # First iteration: compute KV for question Q
                question_kv_cache = None
                
                for t in range(self.latent_module.num_jacobi_iterations):
                    if t == 0:
                        # First iteration: compute full Q || Z
                        context_ids = torch.cat([input_ids, Z_ids], dim=1)
                        
                        outputs = self.model(
                            input_ids=context_ids,
                            use_cache=True,  # Enable KV-cache
                            output_hidden_states=True
                        )
                        
                        forward_count += 1
                        
                        # Cache KV for question part (Q)
                        # Extract KV only for question tokens (avoid caching Z tokens)
                        question_length = input_ids.size(1)
                        question_kv_cache = outputs.past_key_values
                        
                        # For proper KV extraction, we need to slice the cached states
                        # Note: This is a simplified version - full implementation depends on model architecture
                        question_kv_cache = tuple(
                            (k[:, :, :question_length, :], v[:, :, :question_length, :])
                            for k, v in question_kv_cache
                        )
                    
                    else:
                        # Subsequent iterations: reuse Q's KV-cache, only compute Z
                        # This reduces computation from O(|Q| + |Z|) to O(|Z|)
                        outputs = self.model(
                            input_ids=Z_ids,  # Only pass Z tokens
                            past_key_values=question_kv_cache,  # Reuse Q's KV-cache
                            use_cache=True,
                            output_hidden_states=True
                        )
                        
                        forward_count += 1
                    
                    # Update Z using next-token prediction
                    logits = outputs.logits[:, -self.latent_module.latent_tokens:, :]
                    Z_ids = torch.argmax(logits, dim=-1)
                
                # Step 3: Generate answer tokens with full KV-cache reuse
                # Input: Q || Z || <bot>
                bot_token_id = self.tokenizer.convert_tokens_to_ids("<bot>") if "<bot>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id
                
                # Compute final Q || Z with KV-cache
                final_context_ids = torch.cat([input_ids, Z_ids], dim=1)
                
                # Get final KV-cache for Q || Z
                outputs = self.model(
                    input_ids=final_context_ids,
                    use_cache=True,
                    output_hidden_states=True
                )
                final_kv_cache = outputs.past_key_values
                forward_count += 1
                
                # Now generate answer autoregressively with KV-cache
                # Start with <bot> token
                generation_input = torch.tensor([[bot_token_id]], device=self.device)
                
                # Use model.generate() with past_key_values for efficient decoding
                output_ids = self.model.generate(
                    generation_input,
                    past_key_values=final_kv_cache,  # Reuse KV-cache from Q || Z
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True  # Continue using cache during generation
                )
                
                # Count forward passes (answer generation)
                answer_length = output_ids.size(1) - generation_input.size(1)
                forward_count += answer_length
                
                # Reconstruct full output for decoding
                # Need to prepend Q || Z || <bot> for proper decoding
                output_ids = torch.cat([final_context_ids, output_ids], dim=1)
        
        else:
            # === Standard generation with incremental decoding ===
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True  # Enable KV-cache for standard generation too
                )
                
                answer_length = output_ids.size(1) - input_ids.size(1)
                forward_count = answer_length
        
        # Decode answer
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the generated part
        answer = answer[len(prompt):]
        
        if return_forward_count:
            return answer, forward_count
        return answer
    
    def batch_generate(
        self,
        questions: List[str],
        use_latent: bool = True,
        max_new_tokens: int = 256
    ) -> List[str]:
        """
        Generate answers for multiple questions.
        
        Args:
            questions: List of questions
            use_latent: Whether to use latent reasoning
            max_new_tokens: Max tokens per answer
        
        Returns:
            List of generated answers
        """
        answers = []
        
        for i, question in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] Generating...")
            answer = self.generate(
                question,
                use_latent=use_latent,
                max_new_tokens=max_new_tokens
            )
            answers.append(answer)
        
        return answers
    
    def interactive_mode(self):
        """Run interactive question-answering loop."""
        print("\n" + "="*80)
        print("KAVA Interactive Inference")
        print("="*80)
        print("Type your question and press Enter. Type 'quit' to exit.")
        print("Commands:")
        print("  /latent on|off  - Toggle latent reasoning")
        print("  /temp <float>   - Set temperature")
        print("  /quit           - Exit")
        print("="*80 + "\n")
        
        use_latent = True
        temperature = 0.0
        
        while True:
            try:
                user_input = input("\nQuestion: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if user_input == "/quit":
                        break
                    elif user_input.startswith("/latent"):
                        mode = user_input.split()[1].lower()
                        use_latent = mode == "on"
                        print(f"✓ Latent reasoning: {'ON' if use_latent else 'OFF'}")
                    elif user_input.startswith("/temp"):
                        temperature = float(user_input.split()[1])
                        print(f"✓ Temperature: {temperature}")
                    else:
                        print("Unknown command")
                    continue
                
                # Generate answer
                print("\nGenerating answer...")
                answer, forward_count = self.generate(
                    user_input,
                    use_latent=use_latent,
                    temperature=temperature,
                    return_forward_count=True
                )
                
                print(f"\nAnswer: {answer}")
                print(f"Forward passes: {forward_count}")
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="KAVA model inference")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Inference mode"
    )
    
    parser.add_argument(
        "--questions",
        nargs="+",
        type=str,
        help="Questions for batch mode"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file with questions (one per line) for batch mode"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for batch mode results"
    )
    
    parser.add_argument(
        "--use_latent",
        action="store_true",
        default=True,
        help="Use latent reasoning (default: True)"
    )
    
    parser.add_argument(
        "--no_latent",
        action="store_true",
        help="Disable latent reasoning"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize inference
    inference = KAVAInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    use_latent = args.use_latent and not args.no_latent
    
    if args.mode == "interactive":
        # Interactive mode
        inference.interactive_mode()
    
    else:
        # Batch mode
        if args.input_file:
            # Load questions from file
            with open(args.input_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        elif args.questions:
            questions = args.questions
        else:
            print("Error: Provide --questions or --input_file for batch mode")
            return
        
        print(f"Processing {len(questions)} questions...")
        
        # Generate answers
        answers = inference.batch_generate(
            questions,
            use_latent=use_latent,
            max_new_tokens=args.max_new_tokens
        )
        
        # Output results
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for q, a in zip(questions, answers):
                    f.write(f"Q: {q}\n")
                    f.write(f"A: {a}\n")
                    f.write("-" * 80 + "\n")
            print(f"✓ Results saved to {args.output_file}")
        else:
            for i, (q, a) in enumerate(zip(questions, answers)):
                print(f"\n[{i+1}] Q: {q}")
                print(f"A: {a}")


if __name__ == "__main__":
    main()
