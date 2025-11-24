"""
Data loading and preprocessing for KAVA training.
Loads GSM8k-AUG and GSM8k-AUG-NL datasets from HuggingFace.
"""

import re
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


class GSM8KDataset:
    """
    Loads and preprocesses GSM8k-AUG and GSM8k-AUG-NL datasets.
    
    Dataset fields (from paper Appendix B):
    - question: The math problem
    - steps: Chain-of-thought reasoning (equation-only or natural language)
    - answer: Final numerical answer
    
    Sizes:
    - Train: 385,620
    - Val: 500
    - Test: 1,319
    """
    
    def __init__(
        self,
        dataset_name: str = "whynlp/gsm8k-aug",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        cot_type: str = "equation"
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name
                - "whynlp/gsm8k-aug" for equation-only CoT
                - "whynlp/gsm8k-aug-nl" for natural language CoT
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            cot_type: "equation" or "natural_language"
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cot_type = cot_type
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name)
        
        # Verify dataset sizes (should match paper)
        self.verify_dataset_sizes()
        
        # Define special tokens
        self.BOT_TOKEN = "<bot>"  # Beginning of thought
        self.EOT_TOKEN = "<eot>"  # End of thought
        
        # Add special tokens to tokenizer if needed
        if tokenizer is not None:
            self.add_special_tokens()
    
    def verify_dataset_sizes(self):
        """Verify dataset sizes match paper (385620 train, 500 val, 1319 test)."""
        train_size = len(self.dataset['train'])
        val_size = len(self.dataset['validation']) if 'validation' in self.dataset else 0
        test_size = len(self.dataset['test'])
        
        print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Note: Some datasets might have slightly different splits
        # The paper mentions 385,620 train samples
        if train_size != 385620:
            print(f"Warning: Train size {train_size} doesn't match paper (385620)")
    
    def add_special_tokens(self):
        """Add special tokens (<bot>, <eot>) to tokenizer."""
        special_tokens = {
            'additional_special_tokens': [self.BOT_TOKEN, self.EOT_TOKEN]
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens to tokenizer")
        
        # Get token IDs
        self.bot_token_id = self.tokenizer.convert_tokens_to_ids(self.BOT_TOKEN)
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids(self.EOT_TOKEN)
    
    def format_teacher_prompt(
        self,
        question: str,
        steps: str,
        answer: str
    ) -> str:
        """
        Format prompt for teacher (full CoT).
        
        Format: [Question] [Steps] [Answer]
        
        Args:
            question: The math problem
            steps: Chain-of-thought reasoning
            answer: Final answer
        
        Returns:
            Formatted prompt string
        """
        # Clean and format
        question = question.strip()
        steps = steps.strip()
        answer = answer.strip()
        
        # Teacher sees full CoT
        prompt = f"Question: {question}\n\nSolution:\n{steps}\n\nAnswer: {answer}"
        
        return prompt
    
    def format_student_prompt(
        self,
        question: str,
        answer: str
    ) -> Tuple[str, str]:
        """
        Format prompt for student (question only + answer).
        
        Format: [Question] <bot> [latent tokens] <eot> [Answer]
        
        Args:
            question: The math problem
            answer: Final answer (for training labels)
        
        Returns:
            question_prompt: Just the question part
            answer_text: Just the answer part
        """
        question = question.strip()
        answer = answer.strip()
        
        # Student only sees question
        question_prompt = f"Question: {question}\n\nAnswer:"
        answer_text = f" {answer}"
        
        return question_prompt, answer_text
    
    def tokenize_teacher_sample(
        self,
        question: str,
        steps: str,
        answer: str
    ) -> Dict:
        """
        Tokenize a sample for teacher training.
        
        Returns:
            Dict with:
                - input_ids: Full sequence tokens
                - attention_mask: Attention mask
                - labels: Labels for loss (ignore question part)
                - steps_start_idx: Where CoT steps start
                - steps_end_idx: Where CoT steps end
                - answer_start_idx: Where answer starts
        """
        # Format full prompt
        full_prompt = self.format_teacher_prompt(question, steps, answer)
        
        # Tokenize
        encoding = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Also tokenize parts separately to find boundaries
        question_prompt = f"Question: {question}\n\nSolution:\n"
        steps_text = f"{steps}\n\nAnswer:"
        
        q_tokens = self.tokenizer(question_prompt, add_special_tokens=False)['input_ids']
        s_tokens = self.tokenizer(steps_text, add_special_tokens=False)['input_ids']
        
        # Calculate indices (approximate - may need adjustment)
        steps_start_idx = len(q_tokens)
        steps_end_idx = steps_start_idx + len(s_tokens)
        answer_start_idx = steps_end_idx
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'steps_start_idx': steps_start_idx,
            'steps_end_idx': steps_end_idx,
            'answer_start_idx': answer_start_idx,
            'question': question,
            'steps': steps,
            'answer': answer
        }
    
    def tokenize_student_sample(
        self,
        question: str,
        answer: str
    ) -> Dict:
        """
        Tokenize a sample for student training.
        
        Returns:
            Dict with:
                - question_ids: Question tokens only
                - question_attention_mask: Attention mask for question
                - answer_ids: Answer tokens only
                - question: Raw question text
                - answer: Raw answer text
        """
        question_prompt, answer_text = self.format_student_prompt(question, answer)
        
        # Tokenize question
        q_encoding = self.tokenizer(
            question_prompt,
            max_length=self.max_length // 2,  # Leave room for latents + answer
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Tokenize answer
        a_encoding = self.tokenizer(
            answer_text,
            max_length=self.max_length // 4,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # Don't add BOS/EOS
            return_tensors=None
        )
        
        return {
            'question_ids': q_encoding['input_ids'],
            'question_attention_mask': q_encoding['attention_mask'],
            'answer_ids': a_encoding['input_ids'],
            'question': question,
            'answer': answer
        }
    
    def get_train_dataset(self) -> Dataset:
        """Get training split."""
        return self.dataset['train']
    
    def get_val_dataset(self) -> Dataset:
        """Get validation split."""
        if 'validation' in self.dataset:
            return self.dataset['validation']
        elif 'val' in self.dataset:
            return self.dataset['val']
        else:
            # Use first 500 from test as validation
            return self.dataset['test'].select(range(500))
    
    def get_test_dataset(self) -> Dataset:
        """Get test split."""
        return self.dataset['test']


def collate_fn_teacher(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict:
    """
    Collate function for teacher training batches.
    
    Args:
        batch: List of tokenized samples
        tokenizer: Tokenizer for padding
    
    Returns:
        Batched tensors
    """
    import torch
    
    # Pad sequences
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    
    # Pad to max length in batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0
    )
    
    # Collect indices
    steps_start_idx = [item['steps_start_idx'] for item in batch]
    steps_end_idx = [item['steps_end_idx'] for item in batch]
    answer_start_idx = [item['answer_start_idx'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'steps_start_idx': torch.tensor(steps_start_idx),
        'steps_end_idx': torch.tensor(steps_end_idx),
        'answer_start_idx': torch.tensor(answer_start_idx),
    }


def collate_fn_student(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict:
    """
    Collate function for student training batches.
    
    Args:
        batch: List of tokenized samples
        tokenizer: Tokenizer for padding
    
    Returns:
        Batched tensors
    """
    import torch
    
    # Pad question sequences
    question_ids = [torch.tensor(item['question_ids']) for item in batch]
    question_attention_mask = [torch.tensor(item['question_attention_mask']) for item in batch]
    
    question_ids = torch.nn.utils.rnn.pad_sequence(
        question_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    question_attention_mask = torch.nn.utils.rnn.pad_sequence(
        question_attention_mask,
        batch_first=True,
        padding_value=0
    )
    
    # Pad answer sequences
    answer_ids = [torch.tensor(item['answer_ids']) for item in batch]
    answer_ids = torch.nn.utils.rnn.pad_sequence(
        answer_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    return {
        'question_ids': question_ids,
        'question_attention_mask': question_attention_mask,
        'answer_ids': answer_ids,
    }


def extract_answer_number(answer_text: str) -> Optional[float]:
    """
    Extract numerical answer from text.
    
    Follows GSM8k convention: answer is typically in format "####123" or just a number.
    
    Args:
        answer_text: Answer text that may contain formatting
    
    Returns:
        Extracted number or None if not found
    """
    # Remove common formatting
    answer_text = answer_text.strip()
    
    # Look for #### pattern (GSM8k format)
    if '####' in answer_text:
        answer_text = answer_text.split('####')[-1].strip()
    
    # Remove common separators and units
    answer_text = answer_text.replace(',', '')
    answer_text = answer_text.replace('$', '')
    answer_text = answer_text.replace('%', '')
    
    # Extract first number found
    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    
    return None
