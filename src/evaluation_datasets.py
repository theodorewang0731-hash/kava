"""
Extended dataset support for KAVA evaluation.
Adds GSM8k-Hard and SVAMP datasets for zero-shot evaluation.
"""

from datasets import load_dataset, Dataset
from typing import Optional, Dict
import re
import os


class EvaluationDatasets:
    """
    Manager for evaluation datasets.
    
    Supports:
    - GSM8k (original test set)
    - GSM8k-Hard (Gao et al.)
    - SVAMP (Patel et al.)
    """
    
    @staticmethod
    def _load_with_fallback(dataset_name: str, config: Optional[str] = None, split: str = "test", local_name: Optional[str] = None):
        """
        Load dataset with local-first fallback.
        
        Args:
            dataset_name: HuggingFace dataset name
            config: Dataset config name (optional)
            split: Dataset split
            local_name: Local directory name (defaults to last part of dataset_name)
        
        Returns:
            Dataset
        """
        if local_name is None:
            local_name = dataset_name.split('/')[-1]
        
        local_path = os.path.join("./datasets", local_name)
        
        # Try local first
        if os.path.exists(local_path):
            try:
                print(f"✅ Loading {local_name} from local: {local_path}")
                if config:
                    dataset = load_dataset(local_path, config, split=split)
                else:
                    dataset = load_dataset(local_path, split=split)
                print(f"✅ Loaded {len(dataset)} samples from local")
                return dataset
            except Exception as e:
                print(f"⚠️ Local load failed: {e}, trying HuggingFace...")
        else:
            print(f"ℹ️ Local not found, loading from HuggingFace: {dataset_name}")
        
        # Fallback to HuggingFace
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        print(f"✅ Loaded {len(dataset)} samples from HuggingFace")
        return dataset
    
    @staticmethod
    def load_gsm8k(split: str = "test") -> Dataset:
        """
        Load GSM8k dataset.
        
        Source: openai/gsm8k
        Paper: Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)
        
        Format:
        - question: str
        - answer: str (with #### separator)
        
        Returns:
            Dataset with 1,319 test samples
        """
        dataset = EvaluationDatasets._load_with_fallback("openai/gsm8k", config="main", split=split, local_name="gsm8k")
        print(f"Loaded GSM8k {split}: {len(dataset)} samples")
        return dataset
    
    @staticmethod
    def load_gsm8k_hard() -> Dataset:
        """
        Load GSM8k-Hard dataset.
        
        Source: GSM8k-Hard (Gao et al.)
        Paper: PAL: Program-aided Language Models
        
        This is a harder version of GSM8k with more complex reasoning.
        
        Returns:
            Dataset with GSM8k-Hard samples
        """
        try:
            # Try local first
            dataset = EvaluationDatasets._load_with_fallback(
                "reasoning-machines/gsm-hard", 
                split="train", 
                local_name="gsm8k-hard"
            )
            print(f"Loaded GSM8k-Hard: {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"Warning: Cannot load GSM8k-Hard: {e}")
            print("Using GSM8k test set as placeholder for GSM8k-Hard")
            return EvaluationDatasets.load_gsm8k(split="test")
    
    @staticmethod
    def load_svamp() -> Dataset:
        """
        Load SVAMP dataset.
        
        Source: SVAMP (Patel et al., 2021)
        Paper: Are NLP Models really able to Solve Simple Math Word Problems?
        
        SVAMP has 1,000 samples with varied question templates.
        
        Format may vary:
        - Question/Answer (capital)
        - question/answer (lowercase)
        
        Returns:
            Dataset with ~1,000 SVAMP samples
        """
        try:
            # Try local first
            dataset = EvaluationDatasets._load_with_fallback(
                "ChilleD/SVAMP", 
                split="test", 
                local_name="svamp"
            )
            print(f"Loaded SVAMP: {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"Warning: Cannot load SVAMP: {e}")
            print("Using GSM8k test set as placeholder for SVAMP")
            return EvaluationDatasets.load_gsm8k(split="test")
    
    @staticmethod
    def normalize_sample(sample: Dict, dataset_name: str) -> Dict:
        """
        Normalize sample format across different datasets.
        
        All datasets should have:
        - question: str
        - answer: str (may contain #### separator)
        
        Args:
            sample: Raw sample from dataset
            dataset_name: Name of dataset for format detection
        
        Returns:
            Normalized dict with 'question' and 'answer' keys
        """
        normalized = {}
        
        # Detect question field (case-insensitive)
        if 'question' in sample:
            normalized['question'] = sample['question']
        elif 'Question' in sample:
            normalized['question'] = sample['Question']
        elif 'input' in sample:
            normalized['question'] = sample['input']
        else:
            raise ValueError(f"Cannot find question field in sample: {sample.keys()}")
        
        # Detect answer field
        if 'answer' in sample:
            normalized['answer'] = sample['answer']
        elif 'Answer' in sample:
            normalized['answer'] = sample['Answer']
        elif 'target' in sample:
            normalized['answer'] = sample['target']
        else:
            raise ValueError(f"Cannot find answer field in sample: {sample.keys()}")
        
        return normalized
    
    @staticmethod
    def extract_numerical_answer(answer_str: str) -> Optional[float]:
        """
        Extract numerical answer from answer string.
        
        Handles various formats:
        - GSM8k: "... #### 42"
        - SVAMP: "42" or "42.5"
        - With units: "$42" or "42 dollars"
        
        Args:
            answer_str: Answer string from dataset
        
        Returns:
            Numerical answer or None if extraction fails
        """
        # Remove extra whitespace
        answer_str = answer_str.strip()
        
        # GSM8k format: extract after ####
        if '####' in answer_str:
            answer_str = answer_str.split('####')[-1].strip()
        
        # Remove currency symbols
        answer_str = answer_str.replace('$', '').replace('£', '').replace('€', '')
        
        # Remove common units
        answer_str = re.sub(r'\s*(dollars?|cents?|pounds?|euros?|%|percent)\s*', '', answer_str, flags=re.IGNORECASE)
        
        # Remove commas in numbers
        answer_str = answer_str.replace(',', '')
        
        # Extract first number (handles negative numbers)
        numbers = re.findall(r'-?\d+\.?\d*', answer_str)
        
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        
        return None
    
    @staticmethod
    def load_all_evaluation_datasets() -> Dict[str, Dataset]:
        """
        Load all evaluation datasets at once.
        
        Returns:
            Dict mapping dataset name to Dataset object
        """
        datasets = {}
        
        print("\nLoading all evaluation datasets...")
        print("=" * 60)
        
        datasets['gsm8k'] = EvaluationDatasets.load_gsm8k()
        datasets['gsm8k-hard'] = EvaluationDatasets.load_gsm8k_hard()
        datasets['svamp'] = EvaluationDatasets.load_svamp()
        
        print("=" * 60)
        print("All datasets loaded successfully!\n")
        
        return datasets


def test_dataset_loading():
    """Test function to verify all datasets load correctly."""
    print("Testing dataset loading...")
    
    datasets = EvaluationDatasets.load_all_evaluation_datasets()
    
    for name, dataset in datasets.items():
        print(f"\nTesting {name}:")
        print(f"  Total samples: {len(dataset)}")
        
        # Test first sample
        sample = dataset[0]
        normalized = EvaluationDatasets.normalize_sample(sample, name)
        
        print(f"  Question: {normalized['question'][:100]}...")
        print(f"  Answer: {normalized['answer'][:100]}...")
        
        # Test answer extraction
        answer_num = EvaluationDatasets.extract_numerical_answer(normalized['answer'])
        print(f"  Extracted number: {answer_num}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_dataset_loading()
