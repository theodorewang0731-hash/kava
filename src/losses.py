"""
Loss functions for KAVA training.
Strictly following Section 3 in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal


class KVDistillationLoss(nn.Module):
    """
    KV-cache distillation loss (L_KV).
    
    Formula: L_KV = (1/2M) * (||K_teacher - K_student||_p^p + ||V_teacher - V_student||_p^p)
    
    Where:
    - K_teacher, V_teacher: Compressed teacher KV (M tokens) from R-KV
    - K_student, V_student: Student latent KV (M tokens)
    - p: 1 (L1/Smooth L1) or 2 (MSE), depends on config
    """
    
    def __init__(
        self,
        loss_type: Literal["l1", "mse", "smooth_l1"] = "smooth_l1",
        layerwise_std: bool = True,
        smooth_l1_beta: float = 1.0
    ):
        """
        Args:
            loss_type: Type of loss function
                - "l1": L1 loss
                - "mse": Mean squared error (L2)
                - "smooth_l1": Smooth L1 loss (default in paper)
            layerwise_std: Whether to normalize by layer-wise std
            smooth_l1_beta: Beta parameter for smooth L1 loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.layerwise_std = layerwise_std
        self.smooth_l1_beta = smooth_l1_beta
    
    def normalize_layerwise(
        self,
        kv_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize by layer-wise standard deviation.
        
        Args:
            kv_states: [batch, num_layers, num_heads, M, head_dim]
        
        Returns:
            normalized: Same shape as input
        """
        if not self.layerwise_std:
            return kv_states
        
        # Compute std per layer (across all other dimensions except layer)
        # Shape: [num_layers, 1, 1, 1]
        std = kv_states.std(dim=(0, 2, 3, 4), keepdim=True)
        std = std.clamp(min=1e-6)  # Avoid division by zero
        
        normalized = kv_states / std
        return normalized
    
    def compute_loss(
        self,
        teacher_kv: torch.Tensor,
        student_kv: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between teacher and student KV.
        
        Args:
            teacher_kv: [batch, num_layers, num_heads, M, head_dim]
            student_kv: [batch, num_layers, num_heads, M, head_dim]
        
        Returns:
            loss: Scalar tensor
        """
        # Apply layer-wise normalization if needed
        if self.layerwise_std:
            teacher_kv = self.normalize_layerwise(teacher_kv)
            student_kv = self.normalize_layerwise(student_kv)
        
        # Compute loss based on type
        if self.loss_type == "l1":
            loss = F.l1_loss(student_kv, teacher_kv, reduction='mean')
        elif self.loss_type == "mse":
            loss = F.mse_loss(student_kv, teacher_kv, reduction='mean')
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(
                student_kv,
                teacher_kv,
                reduction='mean',
                beta=self.smooth_l1_beta
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def forward(
        self,
        teacher_keys: torch.Tensor,
        teacher_values: torch.Tensor,
        student_keys: torch.Tensor,
        student_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute full KV distillation loss.
        
        Formula: L_KV = (1/2M) * (L(K) + L(V))
        
        Args:
            teacher_keys: [batch, num_layers, num_heads, M, head_dim]
            teacher_values: [batch, num_layers, num_heads, M, head_dim]
            student_keys: [batch, num_layers, num_heads, M, head_dim]
            student_values: [batch, num_layers, num_heads, M, head_dim]
        
        Returns:
            loss: Scalar tensor
        """
        # Detach teacher (stop gradient)
        teacher_keys = teacher_keys.detach()
        teacher_values = teacher_values.detach()
        
        # Compute losses for keys and values
        loss_k = self.compute_loss(teacher_keys, student_keys)
        loss_v = self.compute_loss(teacher_values, student_values)
        
        # Average: (1/2) * (L_K + L_V)
        # The (1/M) factor is already in the mean reduction
        loss_kv = 0.5 * (loss_k + loss_v)
        
        return loss_kv


class CODILoss(nn.Module):
    """
    CODI hidden state distillation loss (L_CODI).
    
    Formula: L_CODI = (1/L) * Σ_l ||h_teacher^l - h_student^l||_1
    
    Where:
    - h_teacher^l, h_student^l: Hidden states at layer l for distillation token
    - L: Number of layers
    """
    
    def __init__(
        self,
        loss_type: Literal["l1", "smooth_l1"] = "l1",
        smooth_l1_beta: float = 1.0
    ):
        """
        Args:
            loss_type: Type of loss ("l1" or "smooth_l1")
            smooth_l1_beta: Beta parameter for smooth L1
        """
        super().__init__()
        self.loss_type = loss_type
        self.smooth_l1_beta = smooth_l1_beta
    
    def forward(
        self,
        teacher_hidden_states: torch.Tensor,
        student_hidden_states: torch.Tensor,
        distill_token_idx: int
    ) -> torch.Tensor:
        """
        Compute CODI loss for hidden state distillation.
        
        Args:
            teacher_hidden_states: Tuple of [batch, seq_len, hidden_dim] for each layer
                or stacked tensor [num_layers, batch, seq_len, hidden_dim]
            student_hidden_states: Same format as teacher
            distill_token_idx: Token position to extract (e.g., last token before answer)
        
        Returns:
            loss: Scalar tensor
        """
        # Convert tuple to tensor if needed
        if isinstance(teacher_hidden_states, tuple):
            teacher_hidden_states = torch.stack(teacher_hidden_states, dim=0)
        if isinstance(student_hidden_states, tuple):
            student_hidden_states = torch.stack(student_hidden_states, dim=0)
        
        # Extract hidden states at distillation token position
        # teacher_hidden_states: [num_layers, batch, seq_len, hidden_dim]
        teacher_h = teacher_hidden_states[:, :, distill_token_idx, :]
        student_h = student_hidden_states[:, :, distill_token_idx, :]
        # Shape: [num_layers, batch, hidden_dim]
        
        # Detach teacher (stop gradient)
        teacher_h = teacher_h.detach()
        
        # Compute loss per layer and average
        if self.loss_type == "l1":
            loss = F.l1_loss(student_h, teacher_h, reduction='mean')
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(
                student_h,
                teacher_h,
                reduction='mean',
                beta=self.smooth_l1_beta
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class KAVALoss(nn.Module):
    """
    Full KAVA training loss.
    
    Formula:
    L_KAVA = -log p(A | Z, Q) - log p(A, C | Q) + α₁ * L_CODI + α₂ * L_KV
    
    Where:
    - First term: Student CE loss (latent → answer)
    - Second term: Teacher CE loss (full CoT → answer)
    - Third term: CODI hidden state distillation
    - Fourth term: KV-cache distillation
    """
    
    def __init__(
        self,
        alpha1_codi: float = 10.0,
        alpha2_kv: float = 1.0,
        kv_loss_type: str = "smooth_l1",
        layerwise_std: bool = True,
        codi_loss_type: str = "l1"
    ):
        """
        Args:
            alpha1_codi: Weight for CODI loss (α₁)
            alpha2_kv: Weight for KV loss (α₂)
            kv_loss_type: Type of KV loss ("l1", "mse", "smooth_l1")
            layerwise_std: Whether to use layer-wise std normalization
            codi_loss_type: Type of CODI loss ("l1", "smooth_l1")
        """
        super().__init__()
        
        self.alpha1 = alpha1_codi
        self.alpha2 = alpha2_kv
        
        # Initialize loss components
        self.kv_loss = KVDistillationLoss(
            loss_type=kv_loss_type,
            layerwise_std=layerwise_std
        )
        
        self.codi_loss = CODILoss(loss_type=codi_loss_type)
        
        # Cross-entropy for language modeling
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len], with -100 for ignored tokens
        
        Returns:
            loss: Scalar tensor
        """
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Flatten for cross-entropy
        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def forward(
        self,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
        student_keys: torch.Tensor,
        student_values: torch.Tensor,
        teacher_keys: torch.Tensor,
        teacher_values: torch.Tensor,
        student_hidden_states: torch.Tensor,
        teacher_hidden_states: torch.Tensor,
        distill_token_idx: int
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute full KAVA loss.
        
        Args:
            student_logits: [batch, seq_len, vocab_size] from latent forward
            student_labels: [batch, seq_len] for student CE loss
            teacher_logits: [batch, seq_len, vocab_size] from full CoT forward
            teacher_labels: [batch, seq_len] for teacher CE loss
            student_keys: [batch, num_layers, num_heads, M, head_dim]
            student_values: [batch, num_layers, num_heads, M, head_dim]
            teacher_keys: [batch, num_layers, num_heads, M, head_dim] (compressed)
            teacher_values: [batch, num_layers, num_heads, M, head_dim] (compressed)
            student_hidden_states: Hidden states for CODI
            teacher_hidden_states: Hidden states for CODI
            distill_token_idx: Token position for CODI distillation
        
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with individual loss components
        """
        # 1. Student CE loss: -log p(A | Z, Q)
        loss_student_ce = self.compute_ce_loss(student_logits, student_labels)
        
        # 2. Teacher CE loss: -log p(A, C | Q)
        loss_teacher_ce = self.compute_ce_loss(teacher_logits, teacher_labels)
        
        # 3. CODI loss: hidden state distillation
        loss_codi = self.codi_loss(
            teacher_hidden_states,
            student_hidden_states,
            distill_token_idx
        )
        
        # 4. KV loss: KV-cache distillation
        loss_kv = self.kv_loss(
            teacher_keys,
            teacher_values,
            student_keys,
            student_values
        )
        
        # Total loss
        total_loss = (
            loss_student_ce +
            loss_teacher_ce +
            self.alpha1 * loss_codi +
            self.alpha2 * loss_kv
        )
        
        # Return loss breakdown for logging
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_student_ce': loss_student_ce.item(),
            'loss_teacher_ce': loss_teacher_ce.item(),
            'loss_codi': loss_codi.item(),
            'loss_kv': loss_kv.item(),
            'loss_codi_weighted': (self.alpha1 * loss_codi).item(),
            'loss_kv_weighted': (self.alpha2 * loss_kv).item(),
        }
        
        return total_loss, loss_dict
