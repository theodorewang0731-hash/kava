"""
PCCoT Latent Reasoning Module
Implements parallel-decoding continuous Chain-of-Thought with Jacobi iterations.
Following the paper: 24 latent tokens, 3 iterations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class LatentReasoningModule(nn.Module):
    """
    Latent reasoning module for KAVA/PCCoT.
    
    Key features:
    - M=24 continuous latent tokens
    - T=3 Jacobi parallel iterations
    - Extracts KV-cache from final iteration
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        num_latent_tokens: int = 24,
        num_iterations: int = 3,
        init_strategy: str = "embedding"
    ):
        """
        Args:
            model: Base language model (LLaMA/Qwen)
            num_latent_tokens: M, number of latent tokens (24 in paper)
            num_iterations: T, number of Jacobi iterations (3 in paper)
            init_strategy: How to initialize latent tokens
                - "embedding": Use special token embedding
                - "learned": Learn initialization parameters
        """
        super().__init__()
        
        self.model = model
        self.M = num_latent_tokens
        self.T = num_iterations
        self.init_strategy = init_strategy
        
        # Get model dimensions
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        # Initialize latent token representations
        if init_strategy == "learned":
            # Learnable initialization
            self.latent_init = nn.Parameter(
                torch.randn(num_latent_tokens, self.hidden_dim) * 0.02
            )
        else:
            # Use embedding from special tokens (e.g., <bot> token)
            self.latent_init = None  # Will be initialized from tokenizer
        
        # Projection layer for updating latents from hidden states
        self.latent_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    def initialize_latent_tokens(
        self,
        batch_size: int,
        device: torch.device,
        bot_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Initialize latent token embeddings.
        
        Args:
            batch_size: Number of samples in batch
            device: Device to create tensors on
            bot_token_id: Token ID for <bot> token (if using embedding strategy)
        
        Returns:
            latent_embeds: [batch_size, M, hidden_dim]
        """
        if self.init_strategy == "learned":
            # Use learned initialization
            latent_embeds = self.latent_init.unsqueeze(0).expand(
                batch_size, -1, -1
            ).to(device)
        else:
            # Use <bot> token embedding
            if bot_token_id is None:
                raise ValueError("bot_token_id required for embedding initialization")
            
            # Get embedding layer
            embed_layer = self.model.get_input_embeddings()
            bot_embed = embed_layer(torch.tensor([bot_token_id], device=device))
            
            # Expand to M tokens
            latent_embeds = bot_embed.unsqueeze(0).expand(
                batch_size, self.M, -1
            )
        
        return latent_embeds
    
    def jacobi_iteration(
        self,
        question_embeds: torch.Tensor,
        question_attention_mask: torch.Tensor,
        latent_embeds: torch.Tensor,
        iteration: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single Jacobi iteration: forward pass through model and update latents.
        
        Args:
            question_embeds: [batch, q_len, hidden_dim]
            question_attention_mask: [batch, q_len]
            latent_embeds: [batch, M, hidden_dim]
            iteration: Current iteration number (0, 1, 2)
        
        Returns:
            updated_latent_embeds: [batch, M, hidden_dim]
            outputs: Dict with logits, hidden_states, kv_cache
        """
        batch_size = question_embeds.size(0)
        
        # Concatenate question + latent tokens
        # [batch, q_len + M, hidden_dim]
        inputs_embeds = torch.cat([question_embeds, latent_embeds], dim=1)
        
        # Attention mask: 1 for question and latent tokens
        latent_mask = torch.ones(
            batch_size, self.M,
            dtype=question_attention_mask.dtype,
            device=question_attention_mask.device
        )
        attention_mask = torch.cat([question_attention_mask, latent_mask], dim=1)
        
        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=(iteration == self.T - 1),  # Only cache on last iteration
            return_dict=True
        )
        
        # Extract hidden states for latent tokens
        # Last layer hidden states: [batch, q_len + M, hidden_dim]
        last_hidden = outputs.hidden_states[-1]
        latent_hidden = last_hidden[:, -self.M:, :]  # [batch, M, hidden_dim]
        
        # Update latent embeddings through projection
        updated_latent_embeds = self.latent_proj(latent_hidden)
        
        # Residual connection
        updated_latent_embeds = latent_embeds + updated_latent_embeds
        
        return updated_latent_embeds, {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'past_key_values': outputs.past_key_values if iteration == self.T - 1 else None
        }
    
    def forward_student(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_input_ids: Optional[torch.Tensor] = None,
        bot_token_id: Optional[int] = None,
        eot_token_id: Optional[int] = None,
        return_kv: bool = True,
        return_all_hidden: bool = False
    ) -> Dict:
        """
        Student forward pass with latent reasoning.
        
        Process:
        1. Embed question
        2. Initialize M latent tokens
        3. Run T Jacobi iterations
        4. Optionally append answer and continue generation
        
        Args:
            input_ids: [batch, q_len] Question tokens
            attention_mask: [batch, q_len]
            answer_input_ids: [batch, a_len] Answer tokens (for training)
            bot_token_id: Beginning of thought token
            eot_token_id: End of thought token
            return_kv: Whether to return KV cache from final iteration
            return_all_hidden: Whether to return hidden states from all layers
        
        Returns:
            Dict with:
                - logits: [batch, seq_len, vocab_size]
                - latent_embeds: [batch, M, hidden_dim]
                - kv_cache: Past key values (if return_kv=True)
                - hidden_states: All layer hidden states (if return_all_hidden=True)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 1. Embed question
        embed_layer = self.model.get_input_embeddings()
        question_embeds = embed_layer(input_ids)  # [batch, q_len, hidden_dim]
        
        # 2. Initialize latent tokens
        latent_embeds = self.initialize_latent_tokens(
            batch_size, device, bot_token_id
        )
        
        # 3. Run T Jacobi iterations
        final_outputs = None
        for t in range(self.T):
            latent_embeds, outputs = self.jacobi_iteration(
                question_embeds,
                attention_mask,
                latent_embeds,
                iteration=t
            )
            
            # Save outputs from last iteration
            if t == self.T - 1:
                final_outputs = outputs
        
        # 4. If answer is provided (training), append and generate
        if answer_input_ids is not None:
            # Insert <eot> token between latents and answer
            if eot_token_id is not None:
                eot_embeds = embed_layer(
                    torch.tensor([[eot_token_id]], device=device)
                ).expand(batch_size, 1, -1)
                latent_embeds = torch.cat([latent_embeds, eot_embeds], dim=1)
            
            # Embed answer tokens
            answer_embeds = embed_layer(answer_input_ids)
            
            # Full sequence: question + latents + <eot> + answer
            full_embeds = torch.cat([
                question_embeds,
                latent_embeds,
                answer_embeds
            ], dim=1)
            
            # Create attention mask
            q_len = question_embeds.size(1)
            l_len = latent_embeds.size(1)
            a_len = answer_embeds.size(1)
            
            full_mask = torch.ones(
                batch_size, q_len + l_len + a_len,
                dtype=attention_mask.dtype,
                device=device
            )
            
            # Forward pass with answer
            outputs = self.model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                output_hidden_states=return_all_hidden or return_kv,
                use_cache=return_kv,
                return_dict=True
            )
            
            final_outputs = {
                'logits': outputs.logits,
                'hidden_states': outputs.hidden_states if return_all_hidden else None,
                'past_key_values': outputs.past_key_values if return_kv else None
            }
        
        # Extract KV cache for latent tokens (for KV distillation)
        latent_kv = None
        if return_kv and final_outputs['past_key_values'] is not None:
            latent_kv = self.extract_latent_kv(
                final_outputs['past_key_values'],
                question_len=input_ids.size(1),
                latent_len=self.M
            )
        
        return {
            'logits': final_outputs['logits'],
            'latent_embeds': latent_embeds,
            'latent_kv': latent_kv,
            'hidden_states': final_outputs['hidden_states'],
            'past_key_values': final_outputs['past_key_values']
        }
    
    def extract_latent_kv(
        self,
        past_key_values: Tuple,
        question_len: int,
        latent_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract KV cache corresponding to latent tokens.
        
        Args:
            past_key_values: Tuple of (key, value) for each layer
            question_len: Length of question sequence
            latent_len: Number of latent tokens (M)
        
        Returns:
            latent_keys: [batch, num_layers, num_heads, M, head_dim]
            latent_values: [batch, num_layers, num_heads, M, head_dim]
        """
        num_layers = len(past_key_values)
        
        latent_keys = []
        latent_values = []
        
        for layer_idx in range(num_layers):
            key, value = past_key_values[layer_idx]
            # key/value: [batch, num_heads, seq_len, head_dim]
            
            # Extract latent portion: [question_len : question_len + latent_len]
            latent_k = key[:, :, question_len:question_len + latent_len, :]
            latent_v = value[:, :, question_len:question_len + latent_len, :]
            
            latent_keys.append(latent_k)
            latent_values.append(latent_v)
        
        # Stack layers: [batch, num_layers, num_heads, M, head_dim]
        latent_keys = torch.stack(latent_keys, dim=1)
        latent_values = torch.stack(latent_values, dim=1)
        
        return latent_keys, latent_values
    
    def forward_teacher(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden: bool = False,
        return_kv: bool = False,
        return_attention: bool = False
    ) -> Dict:
        """
        Teacher forward pass with full CoT.
        
        Standard autoregressive forward with question + steps + answer.
        
        Args:
            input_ids: [batch, seq_len] Full sequence (Q + C + A)
            attention_mask: [batch, seq_len]
            return_hidden: Return hidden states for CODI loss
            return_kv: Return KV cache for R-KV compression
            return_attention: Return attention weights for importance score
        
        Returns:
            Dict with logits, hidden_states, kv_cache, attention_weights
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden,
            output_attentions=return_attention,
            use_cache=return_kv,
            return_dict=True
        )
        
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if return_hidden else None,
            'past_key_values': outputs.past_key_values if return_kv else None,
            'attentions': outputs.attentions if return_attention else None
        }


def prepare_labels_for_student(
    question_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    num_latent_tokens: int,
    pad_token_id: int = -100
) -> torch.Tensor:
    """
    Prepare labels for student model training.
    
    Student sequence: [question] [latent_1...latent_M] [<eot>] [answer]
    Labels: [-100...] [-100...] [-100] [answer]
    
    Only compute loss on answer tokens.
    
    Args:
        question_ids: [batch, q_len]
        answer_ids: [batch, a_len]
        num_latent_tokens: M
        pad_token_id: Token ID for padding/ignoring (-100)
    
    Returns:
        labels: [batch, q_len + M + 1 + a_len]
    """
    batch_size = question_ids.size(0)
    device = question_ids.device
    
    # Create padding for question + latents + <eot>
    prefix_len = question_ids.size(1) + num_latent_tokens + 1
    prefix_labels = torch.full(
        (batch_size, prefix_len),
        pad_token_id,
        dtype=answer_ids.dtype,
        device=device
    )
    
    # Concatenate with answer labels
    labels = torch.cat([prefix_labels, answer_ids], dim=1)
    
    return labels


def prepare_labels_for_teacher(
    question_ids: torch.Tensor,
    steps_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    pad_token_id: int = -100
) -> torch.Tensor:
    """
    Prepare labels for teacher model training.
    
    Teacher sequence: [question] [steps] [answer]
    Labels: [-100...] [steps] [answer]
    
    Compute loss on both steps and answer.
    
    Args:
        question_ids: [batch, q_len]
        steps_ids: [batch, s_len]
        answer_ids: [batch, a_len]
        pad_token_id: Token ID for padding/ignoring (-100)
    
    Returns:
        labels: [batch, q_len + s_len + a_len]
    """
    batch_size = question_ids.size(0)
    device = question_ids.device
    
    # Create padding for question
    question_labels = torch.full(
        question_ids.shape,
        pad_token_id,
        dtype=steps_ids.dtype,
        device=device
    )
    
    # Concatenate: [-100...question] [steps] [answer]
    labels = torch.cat([question_labels, steps_ids, answer_ids], dim=1)
    
    return labels
