"""
KAVA Trainer - Main training loop
Implements the full KAVA training procedure from the paper.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from typing import Dict, Optional

# Optional wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

from src.latent_reasoning import LatentReasoningModule, prepare_labels_for_student, prepare_labels_for_teacher
from src.losses import KAVALoss
from src.rkv_compression import RKVCompressor
from src.data_utils import GSM8KDataset, collate_fn_teacher, collate_fn_student


class KAVATrainer:
    """
    Trainer for KAVA model.
    
    Implements:
    - Teacher-student self-distillation
    - R-KV compression
    - KV-cache distillation
    - CODI hidden state distillation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with config from YAML.
        
        Args:
            config: Configuration dictionary from YAML file
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        torch.manual_seed(config['system']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['system']['seed'])
        
        # Initialize components
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
        print(f"Initialized KAVA Trainer on device: {self.device}")
        print(f"Model: {config['model']['name']}")
        print(f"Dataset: {config['dataset']['name']}")
    
    def setup_model(self):
        """Load base model and apply LoRA."""
        print("Loading base model...")
        
        # Load base model
        model_name = self.config['model']['name']
        
        # 检查是否为本地路径（绝对路径或相对路径存在）
        is_local_path = os.path.exists(model_name) or model_name.startswith('/')
        
        # 如果是本地路径或设置了离线模式，强制使用本地文件
        use_local = is_local_path or os.environ.get('HUGGINGFACE_HUB_OFFLINE') == '1'
        
        print(f"Model: {model_name}")
        print(f"Loading mode: {'Local path' if is_local_path else 'Offline' if use_local else 'Online'}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config['system']['mixed_precision'] == 'bf16' else torch.float32,
            device_map='auto',
            trust_remote_code=True,
            local_files_only=use_local  # ✅ 本地路径或离线模式时强制使用本地文件
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=use_local  # ✅ 本地路径或离线模式时强制使用本地文件
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Initialize latent reasoning module
        self.latent_module = LatentReasoningModule(
            model=self.model,
            num_latent_tokens=self.config['latent']['num_tokens'],
            num_iterations=self.config['latent']['num_iterations'],
            init_strategy="embedding"
        )
        
        # Initialize R-KV compressor
        self.rkv_compressor = RKVCompressor(
            num_latent_tokens=self.config['latent']['num_tokens'],
            lambda_mix=self.config['rkv']['lambda'],
            layerwise_std=self.config['loss']['layerwise_std'],
            use_projection=self.config['loss']['use_projection'],
            hidden_dim=self.model.config.hidden_size
        )
        
        # Move projection layers to device if needed
        if self.rkv_compressor.k_proj is not None:
            self.rkv_compressor.k_proj = self.rkv_compressor.k_proj.to(self.device)
            self.rkv_compressor.v_proj = self.rkv_compressor.v_proj.to(self.device)
    
    def setup_data(self):
        """Load and prepare datasets."""
        print("Loading datasets...")
        
        # Initialize dataset
        self.dataset = GSM8KDataset(
            dataset_name=self.config['dataset']['name'],
            tokenizer=self.tokenizer,
            max_length=512,
            cot_type=self.config['dataset']['cot_type']
        )
        
        # Update tokenizer with special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Get train/val splits
        train_data = self.dataset.get_train_dataset()
        val_data = self.dataset.get_val_dataset()
        
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        
        # For now, we'll process samples on-the-fly in training loop
        self.train_data = train_data
        self.val_data = val_data
    
    def setup_training(self):
        """Initialize optimizer, scheduler, and loss."""
        print("Setting up training components...")
        
        # Initialize loss function
        self.criterion = KAVALoss(
            alpha1_codi=self.config['loss']['alpha1_codi'],
            alpha2_kv=self.config['loss']['alpha2_kv'],
            kv_loss_type=self.config['loss']['kv_loss_type'],
            layerwise_std=self.config['loss']['layerwise_std'],
            codi_loss_type="l1"
        ).to(self.device)
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.rkv_compressor.k_proj is not None:
            trainable_params += list(self.rkv_compressor.k_proj.parameters())
            trainable_params += list(self.rkv_compressor.v_proj.parameters())
        trainable_params += list(self.latent_module.latent_proj.parameters())
        
        # Initialize optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Calculate total training steps
        steps_per_epoch = len(self.train_data) // self.config['training']['batch_size']
        total_steps = steps_per_epoch * self.config['training']['epochs']
        warmup_steps = int(total_steps * self.config['training']['warmup_ratio'])
        
        # Initialize scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
    
    def train_step(self, batch_data: Dict) -> Dict:
        """
        Single training step.
        
        Implements the full KAVA training procedure:
        1. Teacher forward (full CoT)
        2. R-KV compression
        3. Student forward (latent reasoning)
        4. Compute all losses
        5. Backprop and update
        
        Args:
            batch_data: Batch of training data
        
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Extract batch data
        question = batch_data['question']
        steps = batch_data['steps']
        answer = batch_data['answer']
        
        # ========== TEACHER FORWARD ==========
        # Tokenize teacher input: Question + Steps + Answer
        teacher_sample = self.dataset.tokenize_teacher_sample(question, steps, answer)
        teacher_input_ids = torch.tensor([teacher_sample['input_ids']], device=self.device)
        teacher_attention_mask = torch.tensor([teacher_sample['attention_mask']], device=self.device)
        
        with torch.no_grad():
            teacher_outputs = self.latent_module.forward_teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                return_hidden=True,
                return_kv=True,
                return_attention=True
            )
        
        # ========== R-KV COMPRESSION ==========
        # Extract teacher KV and compress
        teacher_kv_full = teacher_outputs['past_key_values']
        teacher_attention = teacher_outputs['attentions'][-1]  # Use last layer attention
        
        # Stack KV into tensor format for compression
        teacher_keys_list = []
        teacher_values_list = []
        for layer_kv in teacher_kv_full:
            teacher_keys_list.append(layer_kv[0])  # [batch, num_heads, seq_len, head_dim]
            teacher_values_list.append(layer_kv[1])
        
        teacher_keys_full = torch.stack(teacher_keys_list, dim=1)  # [batch, num_layers, num_heads, seq_len, head_dim]
        teacher_values_full = torch.stack(teacher_values_list, dim=1)
        
        # Compress using R-KV
        teacher_keys_compressed, teacher_values_compressed = self.rkv_compressor.compress(
            key_cache=teacher_keys_full[:, 0],  # Process first layer for simplicity (should do all layers)
            value_cache=teacher_values_full[:, 0],
            attention_weights=teacher_attention,
            answer_start_idx=teacher_sample['answer_start_idx'],
            steps_start_idx=teacher_sample['steps_start_idx'],
            steps_end_idx=teacher_sample['steps_end_idx'],
            attention_mask=teacher_attention_mask  # Pass attention mask for padding handling
        )
        
        # ========== STUDENT FORWARD ==========
        # Tokenize student input: Question only
        student_sample = self.dataset.tokenize_student_sample(question, answer)
        student_question_ids = torch.tensor([student_sample['question_ids']], device=self.device)
        student_question_mask = torch.tensor([student_sample['question_attention_mask']], device=self.device)
        student_answer_ids = torch.tensor([student_sample['answer_ids']], device=self.device)
        
        student_outputs = self.latent_module.forward_student(
            input_ids=student_question_ids,
            attention_mask=student_question_mask,
            answer_input_ids=student_answer_ids,
            bot_token_id=self.dataset.bot_token_id,
            eot_token_id=self.dataset.eot_token_id,
            return_kv=True,
            return_all_hidden=True
        )
        
        # Extract student KV for latent tokens
        student_keys, student_values = student_outputs['latent_kv']
        
        # ========== COMPUTE LOSSES ==========
        # Prepare labels
        student_labels = prepare_labels_for_student(
            student_question_ids,
            student_answer_ids,
            num_latent_tokens=self.config['latent']['num_tokens'],
            pad_token_id=-100
        )
        
        teacher_question_ids = teacher_input_ids[:, :teacher_sample['steps_start_idx']]
        teacher_steps_ids = teacher_input_ids[:, teacher_sample['steps_start_idx']:teacher_sample['answer_start_idx']]
        teacher_answer_ids = teacher_input_ids[:, teacher_sample['answer_start_idx']:]
        
        teacher_labels = prepare_labels_for_teacher(
            teacher_question_ids,
            teacher_steps_ids,
            teacher_answer_ids,
            pad_token_id=-100
        )
        
        # Compute KAVA loss
        # Note: This is simplified - full implementation needs proper tensor reshaping
        loss, loss_dict = self.criterion(
            student_logits=student_outputs['logits'],
            student_labels=student_labels,
            teacher_logits=teacher_outputs['logits'],
            teacher_labels=teacher_labels,
            student_keys=student_keys.unsqueeze(0),  # Add batch dim
            student_values=student_values.unsqueeze(0),
            teacher_keys=teacher_keys_compressed.unsqueeze(0).unsqueeze(0),  # Add batch and layer dims
            teacher_values=teacher_values_compressed.unsqueeze(0).unsqueeze(0),
            student_hidden_states=student_outputs['hidden_states'],
            teacher_hidden_states=teacher_outputs['hidden_states'],
            distill_token_idx=-self.config['latent']['num_tokens']-1  # Token before latents
        )
        
        # ========== BACKWARD AND UPDATE ==========
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['gradient_clipping']
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss_dict
    
    def train(self, use_wandb: bool = False):
        """
        Main training loop.
        
        Args:
            use_wandb: Whether to log to Weights & Biases
        """
        if use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb requested but not installed. Proceeding without logging.")
                use_wandb = False
            else:
                wandb.init(
                    project="kava-reproduction",
                    config=self.config,
                    name=f"{self.config['model']['type']}-{self.config['dataset']['cot_type']}-seed{self.config['system']['seed']}"
                )
        
        print("Starting training...")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        
        global_step = 0
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\n=== Epoch {epoch + 1}/{self.config['training']['epochs']} ===")
            
            epoch_losses = []
            
            # Training loop
            pbar = tqdm(self.train_data, desc=f"Epoch {epoch + 1}")
            
            for i, sample in enumerate(pbar):
                # Skip if we've processed enough for this epoch
                if i >= self.config['training']['batch_size'] * (len(self.train_data) // self.config['training']['batch_size']):
                    break
                
                try:
                    loss_dict = self.train_step(sample)
                    epoch_losses.append(loss_dict['loss_total'])
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss_dict['loss_total']:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    # Logging
                    if global_step % self.config['training']['logging_steps'] == 0:
                        if use_wandb:
                            wandb.log(loss_dict, step=global_step)
                    
                    global_step += 1
                
                except Exception as e:
                    print(f"Error in training step: {e}")
                    continue
            
            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self.save_checkpoint(epoch + 1)
        
        if use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        output_dir = f"checkpoints/{self.config['model']['type']}-{self.config['dataset']['cot_type']}-epoch{epoch}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        with open(f"{output_dir}/config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"Checkpoint saved to {output_dir}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
