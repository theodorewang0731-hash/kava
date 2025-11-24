#!/usr/bin/env python3
"""
Benchmark Incremental Decoding Speedup
Compares naive vs. KV-cache-optimized inference
"""

import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import numpy as np


class IncrementalDecodingBenchmark:
    """
    Benchmark to measure speedup from past_key_values caching.
    
    Tests:
    1. Naive decoding (recompute everything each step)
    2. Cached decoding (reuse past_key_values)
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on
        """
        self.device = device
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded on {device}")
    
    def generate_naive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Naive generation: recompute full sequence each step.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
        
        Returns:
            Dict with generated tokens, time, and forward pass count
        """
        start_time = time.time()
        forward_count = 0
        
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Full forward pass on entire sequence (INEFFICIENT)
                outputs = self.model(
                    input_ids=current_ids,
                    use_cache=False  # Disable caching
                )
                forward_count += 1
                
                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        elapsed = time.time() - start_time
        
        return {
            'output_ids': current_ids,
            'time': elapsed,
            'forward_passes': forward_count,
            'tokens_per_second': forward_count / elapsed if elapsed > 0 else 0
        }
    
    def generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Cached generation: reuse past_key_values (EFFICIENT).
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
        
        Returns:
            Dict with generated tokens, time, and forward pass count
        """
        start_time = time.time()
        forward_count = 0
        
        current_ids = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                if step == 0:
                    # First step: compute full input
                    outputs = self.model(
                        input_ids=current_ids,
                        use_cache=True,
                        past_key_values=None
                    )
                else:
                    # Subsequent steps: only process new token
                    outputs = self.model(
                        input_ids=next_token,  # Only new token!
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                
                forward_count += 1
                
                # Update cache
                past_key_values = outputs.past_key_values
                
                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        elapsed = time.time() - start_time
        
        return {
            'output_ids': current_ids,
            'time': elapsed,
            'forward_passes': forward_count,
            'tokens_per_second': forward_count / elapsed if elapsed > 0 else 0
        }
    
    def generate_transformers_optimized(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Use transformers' built-in generate() with caching.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
        
        Returns:
            Dict with generated tokens and time
        """
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                use_cache=True,   # Enable KV-cache
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        elapsed = time.time() - start_time
        
        return {
            'output_ids': output_ids,
            'time': elapsed,
            'forward_passes': output_ids.size(1) - input_ids.size(1),
            'tokens_per_second': (output_ids.size(1) - input_ids.size(1)) / elapsed if elapsed > 0 else 0
        }
    
    def run_comparison(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        num_warmup: int = 2
    ) -> Dict:
        """
        Run comparison between naive, cached, and transformers methods.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Tokens to generate per prompt
            num_warmup: Warmup iterations
        
        Returns:
            Dictionary with timing results
        """
        print(f"\n{'='*80}")
        print(f"Benchmark: Incremental Decoding Speedup")
        print(f"{'='*80}")
        print(f"Model: {self.model.config._name_or_path}")
        print(f"Device: {self.device}")
        print(f"Prompts: {len(prompts)}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"{'='*80}\n")
        
        results = {
            'naive': [],
            'cached': [],
            'transformers': []
        }
        
        # Warmup
        print(f"Warming up GPU with {num_warmup} iterations...")
        warmup_ids = self.tokenizer("Test warmup prompt", return_tensors="pt").input_ids.to(self.device)
        for _ in range(num_warmup):
            _ = self.generate_transformers_optimized(warmup_ids, max_new_tokens=10)
        print("✓ Warmup complete\n")
        
        # Run benchmarks
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Prompt: {prompt[:60]}...")
            
            # Tokenize
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            input_length = input_ids.size(1)
            
            # Method 1: Naive (slowest)
            print("  → Running naive decoding...")
            naive_result = self.generate_naive(input_ids, max_new_tokens)
            results['naive'].append(naive_result)
            print(f"    Time: {naive_result['time']:.3f}s, FP: {naive_result['forward_passes']}, Tok/s: {naive_result['tokens_per_second']:.1f}")
            
            # Method 2: Manual cached (medium)
            print("  → Running cached decoding...")
            cached_result = self.generate_cached(input_ids, max_new_tokens)
            results['cached'].append(cached_result)
            print(f"    Time: {cached_result['time']:.3f}s, FP: {cached_result['forward_passes']}, Tok/s: {cached_result['tokens_per_second']:.1f}")
            
            # Method 3: Transformers optimized (fastest)
            print("  → Running transformers generate()...")
            trans_result = self.generate_transformers_optimized(input_ids, max_new_tokens)
            results['transformers'].append(trans_result)
            print(f"    Time: {trans_result['time']:.3f}s, FP: {trans_result['forward_passes']}, Tok/s: {trans_result['tokens_per_second']:.1f}")
            
            # Speedup
            speedup_cached = naive_result['time'] / cached_result['time'] if cached_result['time'] > 0 else 0
            speedup_trans = naive_result['time'] / trans_result['time'] if trans_result['time'] > 0 else 0
            print(f"    Speedup (cached): {speedup_cached:.2f}x")
            print(f"    Speedup (transformers): {speedup_trans:.2f}x")
            print()
        
        # Summary
        print(f"\n{'='*80}")
        print("Summary Statistics")
        print(f"{'='*80}\n")
        
        for method in ['naive', 'cached', 'transformers']:
            times = [r['time'] for r in results[method]]
            tok_per_sec = [r['tokens_per_second'] for r in results[method]]
            
            print(f"{method.capitalize()}:")
            print(f"  Avg time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
            print(f"  Avg tok/s: {np.mean(tok_per_sec):.1f} ± {np.std(tok_per_sec):.1f}")
            print()
        
        # Overall speedup
        naive_times = [r['time'] for r in results['naive']]
        cached_times = [r['time'] for r in results['cached']]
        trans_times = [r['time'] for r in results['transformers']]
        
        avg_speedup_cached = np.mean(naive_times) / np.mean(cached_times)
        avg_speedup_trans = np.mean(naive_times) / np.mean(trans_times)
        
        print(f"{'='*80}")
        print(f"Overall Speedup:")
        print(f"  Cached vs Naive: {avg_speedup_cached:.2f}x")
        print(f"  Transformers vs Naive: {avg_speedup_trans:.2f}x")
        print(f"{'='*80}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark incremental decoding speedup")
    
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name or path"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Max tokens to generate per prompt"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    parser.add_argument(
        "--prompts",
        nargs="+",
        type=str,
        default=[
            "What is 15 + 37?",
            "Calculate the area of a circle with radius 5.",
            "If John has 5 apples and gives 2 to Mary, how many does he have left?"
        ],
        help="Test prompts"
    )
    
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=2,
        help="Number of warmup iterations"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = IncrementalDecodingBenchmark(
        model_name=args.model,
        device=args.device
    )
    
    # Run comparison
    results = benchmark.run_comparison(
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        num_warmup=args.num_warmup
    )
    
    print("✅ Benchmark complete!")


if __name__ == "__main__":
    main()
