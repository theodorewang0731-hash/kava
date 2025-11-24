"""
KAVA: Latent Reasoning via Compressed KV-Cache Distillation
Strict paper implementation for reproducibility
"""

__version__ = "1.0.0"
__author__ = "Paper Replication Project"

from src.latent_reasoning import LatentReasoningModule
from src.losses import KAVALoss, KVDistillationLoss, CODILoss
from src.rkv_compression import RKVCompressor
from src.data_utils import GSM8KDataset
from src.trainer import KAVATrainer

__all__ = [
    "LatentReasoningModule",
    "KAVALoss",
    "KVDistillationLoss",
    "CODILoss",
    "RKVCompressor",
    "GSM8KDataset",
    "KAVATrainer",
]
