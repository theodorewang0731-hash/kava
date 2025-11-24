"""
R-KV: Redundancy-aware KV-Cache Compression
Strictly following Section 3.2 in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RKVCompressor:
    """
    R-KV compression algorithm for teacher KV-cache.
    
    Algorithm from paper:
    1. Compute importance score I from answer→steps attention
    2. Compute redundancy score R from key cosine similarity
    3. Mix scores: S = λ*I + (1-λ)*R
    4. Select top-M tokens per layer/head
    """
    
    def __init__(
        self,
        num_latent_tokens: int = 24,
        lambda_mix: float = 0.1,
        layerwise_std: bool = True,
        use_projection: bool = True,
        hidden_dim: Optional[int] = None
    ):
        """
        Args:
            num_latent_tokens: M, number of tokens to keep (24 in paper)
            lambda_mix: λ, weight for importance vs redundancy (0.1 in most configs)
            layerwise_std: Whether to normalize by layer-wise std
            use_projection: Whether to use projection for KV distillation
            hidden_dim: Hidden dimension for projection layers
        """
        self.M = num_latent_tokens
        self.lambda_mix = lambda_mix
        self.layerwise_std = layerwise_std
        self.use_projection = use_projection
        
        # Projection layers (if needed)
        self.k_proj = None
        self.v_proj = None
        if use_projection and hidden_dim is not None:
            self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def compute_importance_score(
        self,
        attention_weights: torch.Tensor,
        answer_start_idx: int,
        steps_start_idx: int,
        steps_end_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute importance score I from answer→steps attention.
        
        Formula (from paper Listing 1):
        I_{i,h,l} = F.softmax(attn, dim=-1).mean(-2)
        where attn is attention from answer tokens to CoT tokens
        
        IMPORTANT: Padding tokens are assigned the lowest score (0).
        
        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
                Full attention matrix from teacher forward
            answer_start_idx: Index where answer tokens start
            steps_start_idx: Index where CoT steps start
            steps_end_idx: Index where CoT steps end
            attention_mask: [batch, seq_len] attention mask (1 = real, 0 = padding)
        
        Returns:
            importance: [batch, num_heads, num_steps]
                Average attention from answer tokens to each step token
        """
        # Extract answer→steps attention submatrix
        # attention_weights[b, h, answer_tokens, step_tokens]
        answer_to_steps = attention_weights[
            :, :, answer_start_idx:, steps_start_idx:steps_end_idx
        ]
        
        # Average over answer tokens (dim=2)
        # Formula: I = (1/N_A) * Σ_j A_{j,i}
        importance = answer_to_steps.mean(dim=2)  # [batch, num_heads, num_steps]
        
        # ========== PADDING TOKENS HANDLING (Paper Section D) ==========
        # "we evict KV-cache in a batch during training. We do that by always 
        #  assigning the lowest possible redundancy and importance score to the 
        #  value-key pairs corresponding to the padding tokens"
        if attention_mask is not None:
            # Get mask for step tokens
            step_mask = attention_mask[:, steps_start_idx:steps_end_idx]  # [batch, num_steps]
            # Expand for heads: [batch, num_heads, num_steps]
            step_mask = step_mask.unsqueeze(1).expand_as(importance)
            # Set padding tokens to 0 (lowest importance)
            importance = importance * step_mask.float()
        
        return importance
    
    def compute_redundancy_score(
        self,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        steps_start_idx: int = 0,
        steps_end_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute redundancy score R from key cosine similarity.
        
        Formula (from paper Listing 1):
        key_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
        cosine_sim = torch.einsum("...id,...jd->...ij", key_norm, key_norm)
        cos_score = torch.sum(-cosine_sim, dim=-2) / torch.sum(~pad_tokens, dim=-1, keepdim=True)
        R = cos_score.softmax(dim=-1)
        R[pad_tokens] = 0  # Assign lowest score to padding tokens
        
        Higher score = less similar to others = more unique = keep
        
        Args:
            key_states: [batch, num_heads, num_steps, head_dim]
                Key vectors for step tokens
            attention_mask: [batch, seq_len] attention mask (1 = real, 0 = padding)
            steps_start_idx: Start index of step tokens in full sequence
            steps_end_idx: End index of step tokens in full sequence
        
        Returns:
            redundancy: [batch, num_heads, num_steps]
                Redundancy score (higher = more unique)
        """
        batch_size, num_heads, num_steps, head_dim = key_states.shape
        
        # Normalize keys for cosine similarity (with epsilon for numerical stability)
        keys_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute pairwise cosine similarity: k_i · k_j
        # [b, h, n, d] @ [b, h, d, n] = [b, h, n, n]
        cos_sim = torch.matmul(keys_norm, keys_norm.transpose(-2, -1))
        
        # Create mask for padding tokens
        if attention_mask is not None and steps_end_idx is not None:
            # Extract mask for step tokens: [batch, num_steps]
            step_mask = attention_mask[:, steps_start_idx:steps_end_idx]
            pad_tokens = (step_mask == 0)  # [batch, num_steps]
            # Expand for heads and matrix: [batch, 1, num_steps]
            pad_tokens_expanded = pad_tokens.unsqueeze(1)
        else:
            pad_tokens = None
            pad_tokens_expanded = None
        
        # Mask out self-similarity (diagonal) and padding tokens
        diag_mask = ~torch.eye(num_steps, device=key_states.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]
        
        if pad_tokens_expanded is not None:
            # Also mask out similarities involving padding tokens
            valid_mask = diag_mask & pad_tokens_expanded.unsqueeze(-1) & pad_tokens_expanded.unsqueeze(-2)
            num_valid = valid_mask.sum(dim=-1).float().clamp(min=1)  # [batch, num_heads, num_steps]
        else:
            valid_mask = diag_mask
            num_valid = (num_steps - 1)
        
        # Average similarity with other valid tokens
        avg_similarity = (cos_sim * valid_mask).sum(dim=-1) / num_valid
        # avg_similarity: [batch, num_heads, num_steps]
        
        # Redundancy = softmax(-avg_similarity)
        # Less similar = higher redundancy score = more likely to keep
        redundancy = F.softmax(-avg_similarity, dim=-1)
        
        # ========== PADDING TOKENS HANDLING ==========
        # Assign lowest score (0) to padding tokens
        if pad_tokens is not None:
            pad_tokens_heads = pad_tokens.unsqueeze(1).expand(batch_size, num_heads, num_steps)
            redundancy = redundancy * (~pad_tokens_heads).float()
        
        return redundancy
    
    def select_top_tokens(
        self,
        importance: torch.Tensor,
        redundancy: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mix importance and redundancy scores, select top-M tokens.
        
        Formula: S_i = λ*I_i + (1-λ)*R_i
        
        Args:
            importance: [batch, num_heads, num_steps]
            redundancy: [batch, num_heads, num_steps]
            key_states: [batch, num_heads, num_steps, head_dim]
            value_states: [batch, num_heads, num_steps, head_dim]
        
        Returns:
            compressed_keys: [batch, num_heads, M, head_dim]
            compressed_values: [batch, num_heads, M, head_dim]
            selected_indices: [batch, num_heads, M]
        """
        # Mix scores: S = λ*I + (1-λ)*R
        mixed_score = self.lambda_mix * importance + (1 - self.lambda_mix) * redundancy
        # mixed_score: [batch, num_heads, num_steps]
        
        # Select top-M indices per head
        top_indices = torch.topk(mixed_score, k=self.M, dim=-1).indices
        # top_indices: [batch, num_heads, M]
        
        # Gather selected keys and values
        batch_size, num_heads, _, head_dim = key_states.shape
        
        # Expand indices for gather operation
        indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        # indices_expanded: [batch, num_heads, M, head_dim]
        
        compressed_keys = torch.gather(key_states, dim=2, index=indices_expanded)
        compressed_values = torch.gather(value_states, dim=2, index=indices_expanded)
        
        return compressed_keys, compressed_values, top_indices
    
    def compress(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_weights: torch.Tensor,
        answer_start_idx: int,
        steps_start_idx: int,
        steps_end_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main compression function following paper Listing 1.
        
        Implements R-KV score:
        S = lbd * I + (1 - lbd) * R
        where I = importance score, R = redundancy score
        
        IMPORTANT: Padding tokens are assigned lowest scores in both I and R.
        
        Args:
            key_cache: [batch, num_heads, seq_len, head_dim]
                Full key cache from teacher
            value_cache: [batch, num_heads, seq_len, head_dim]
                Full value cache from teacher
            attention_weights: [batch, num_heads, seq_len, seq_len]
                Attention weights from teacher's answer generation
            answer_start_idx: Where answer tokens start in sequence
            steps_start_idx: Where CoT step tokens start
            steps_end_idx: Where CoT step tokens end
            attention_mask: [batch, seq_len] attention mask (1 = real, 0 = padding)
        
        Returns:
            compressed_keys: [batch, num_heads, M, head_dim]
            compressed_values: [batch, num_heads, M, head_dim]
        """
        # Extract keys and values for step tokens only
        step_keys = key_cache[:, :, steps_start_idx:steps_end_idx, :]
        step_values = value_cache[:, :, steps_start_idx:steps_end_idx, :]
        
        # 1. Compute importance from attention (with padding handling)
        importance = self.compute_importance_score(
            attention_weights,
            answer_start_idx,
            steps_start_idx,
            steps_end_idx,
            attention_mask=attention_mask
        )
        
        # 2. Compute redundancy from key similarity (with padding handling)
        redundancy = self.compute_redundancy_score(
            step_keys,
            attention_mask=attention_mask,
            steps_start_idx=steps_start_idx,
            steps_end_idx=steps_end_idx
        )
        
        # 3. Select top-M tokens
        compressed_keys, compressed_values, _ = self.select_top_tokens(
            importance,
            redundancy,
            step_keys,
            step_values
        )
        
        # 4. Apply projection if needed
        if self.use_projection and self.k_proj is not None:
            compressed_keys = self.apply_projection(compressed_keys, self.k_proj)
            compressed_values = self.apply_projection(compressed_values, self.v_proj)
        
        return compressed_keys, compressed_values
    
    def apply_projection(
        self,
        states: torch.Tensor,
        proj_layer: nn.Linear
    ) -> torch.Tensor:
        """
        Apply projection layer to KV states.
        
        Args:
            states: [batch, num_heads, M, head_dim]
            proj_layer: Linear projection layer
        
        Returns:
            projected_states: [batch, num_heads, M, head_dim]
        """
        b, h, m, d = states.shape
        # Reshape for projection
        states_flat = states.reshape(b * h * m, d)
        projected = proj_layer(states_flat)
        projected = projected.reshape(b, h, m, d)
        return projected
    
    def normalize_layerwise(
        self,
        kv_states: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Apply layer-wise standard deviation normalization.
        
        Only used when layerwise_std=True in config.
        
        Args:
            kv_states: [batch, num_heads, M, head_dim]
            layer_idx: Current layer index
        
        Returns:
            normalized_states: [batch, num_heads, M, head_dim]
        """
        if not self.layerwise_std:
            return kv_states
        
        # Compute std across the head_dim
        std = kv_states.std(dim=-1, keepdim=True)
        std = std.clamp(min=1e-6)  # Avoid division by zero
        
        normalized = kv_states / std
        return normalized


def extract_kv_from_outputs(
    model_outputs,
    layer_indices: Optional[list] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract key and value states from model outputs.
    
    Args:
        model_outputs: Model outputs with past_key_values
        layer_indices: Which layers to extract (None = all layers)
    
    Returns:
        keys: [batch, num_layers, num_heads, seq_len, head_dim]
        values: [batch, num_layers, num_heads, seq_len, head_dim]
    """
    past_key_values = model_outputs.past_key_values
    
    if layer_indices is None:
        layer_indices = range(len(past_key_values))
    
    keys = []
    values = []
    
    for layer_idx in layer_indices:
        key, value = past_key_values[layer_idx]
        keys.append(key)
        values.append(value)
    
    keys = torch.stack(keys, dim=1)  # [batch, num_layers, num_heads, seq_len, head_dim]
    values = torch.stack(values, dim=1)
    
    return keys, values
