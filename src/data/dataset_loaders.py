#!/usr/bin/env python3
"""
Dataset loaders for different medical evaluation benchmarks
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


class DatasetLoader:
    """Base class for dataset loaders"""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def load(self, n_samples: Optional[int] = None) -> List[Dict]:
        """Load dataset instances"""
        raise NotImplementedError

    def get_question(self, instance: Dict) -> str:
        """Extract question from instance"""
        raise NotImplementedError

    def get_context(self, instance: Dict) -> str:
        """Extract context from instance (if available)"""
        return ""

    def format_for_evaluation(self, instance: Dict) -> Dict:
        """Format instance for evaluation pipeline"""
        return {
            'id': instance.get('id', 'unknown'),
            'question': self.get_question(instance),
            'context': self.get_context(instance),
            'original': instance
        }


class PubMedQALoader(DatasetLoader):
    """Loader for PubMedQA dataset (from HuggingFace)"""

    def __init__(self):
        # Use a dummy base_path since we're loading from HuggingFace
        base_path = Path(__file__).parent.parent.parent / "data"
        super().__init__(base_path)

    def load(self, n_samples: Optional[int] = None, split: str = "test") -> List[Dict]:
        """Load PubMedQA instances from HuggingFace datasets

        Args:
            n_samples: Optional number of samples to load
            split: Dataset split (only 'train' available, which is the test set)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets library: pip install datasets")

        # Load from HuggingFace (pqa_labeled only has 'train' split, which is the labeled test set)
        dataset = load_dataset('pubmed_qa', 'pqa_labeled', split='train')

        instances = []
        for idx, item in enumerate(dataset):
            if n_samples and idx >= n_samples:
                break
            # Convert to dict format expected by format_for_evaluation
            instance = {
                'id': f"pubmedqa_{idx:04d}",
                'question': item['question'],
                'context': ' '.join(item['context']['contexts']),
                'long_answer': item['long_answer'],
                'final_decision': item['final_decision']
            }
            instances.append(instance)

        return [self.format_for_evaluation(inst) for inst in instances]

    def get_question(self, instance: Dict) -> str:
        """Extract question from PubMedQA instance"""
        return instance['question']

    def get_context(self, instance: Dict) -> str:
        """Extract context from PubMedQA instance"""
        return instance.get('context', '')


class MedQALoader(DatasetLoader):
    """Loader for MedQA dataset (from HuggingFace)"""

    def __init__(self, variant: str = "US"):
        # Use a dummy base_path since we're loading from HuggingFace
        base_path = Path(__file__).parent.parent.parent / "data"
        super().__init__(base_path)
        self.variant = variant

    def load(self, n_samples: Optional[int] = None, split: str = "test") -> List[Dict]:
        """Load MedQA instances from HuggingFace datasets"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets library: pip install datasets")

        # Load from HuggingFace GBaker/MedQA-USMLE-4-options
        dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split=split)

        instances = []
        for idx, item in enumerate(dataset):
            if n_samples and idx >= n_samples:
                break
            # Convert to dict format expected by format_for_evaluation
            instance = {
                'id': f"medqa_{idx:04d}",
                'question': item['question'],
                'options': item['options'],  # Already a dict with A, B, C, D keys
                'answer': item['answer'],
                'answer_idx': item.get('answer_idx', '')
            }
            instances.append(instance)

        return [self.format_for_evaluation(inst) for inst in instances]

    def get_question(self, instance: Dict) -> str:
        """Extract question from MedQA instance"""
        return instance['question']

    def get_context(self, instance: Dict) -> str:
        """MedQA questions are self-contained clinical scenarios"""
        # The question itself contains the full clinical scenario
        return instance['question']

    def format_for_evaluation(self, instance: Dict) -> Dict:
        """Format MedQA instance with options"""
        formatted = super().format_for_evaluation(instance)
        # Include answer options for reference
        formatted['options'] = instance.get('options', {})
        formatted['answer'] = instance.get('answer', '')
        return formatted


class MedMCQALoader(DatasetLoader):
    """Loader for MedMCQA dataset (from HuggingFace)"""

    def __init__(self):
        # Use a dummy base_path since we're loading from HuggingFace
        base_path = Path(__file__).parent.parent.parent / "data"
        super().__init__(base_path)

    def load(self, n_samples: Optional[int] = None, split: str = "validation") -> List[Dict]:
        """Load MedMCQA instances from HuggingFace datasets"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets library: pip install datasets")

        # Load from HuggingFace (note: 'dev' doesn't exist, use 'validation' instead)
        if split == "dev":
            split = "validation"  # Map old split name to correct one

        dataset = load_dataset('medmcqa', split=split)

        instances = []
        for idx, item in enumerate(dataset):
            if n_samples and idx >= n_samples:
                break
            # Convert to dict format expected by format_for_evaluation
            instance = {
                'id': f"medmcqa_{idx:04d}",
                'question': item['question'],
                'opa': item['opa'],
                'opb': item['opb'],
                'opc': item['opc'],
                'opd': item['opd'],
                'cop': item['cop'],  # Correct option (1=A, 2=B, 3=C, 4=D)
                'subject_name': item.get('subject_name', ''),
                'topic_name': item.get('topic_name', ''),
                'exp': item.get('exp', '')
            }
            instances.append(instance)

        return [self.format_for_evaluation(inst) for inst in instances]

    def get_question(self, instance: Dict) -> str:
        """Extract question from MedMCQA instance"""
        return instance['question']

    def get_context(self, instance: Dict) -> str:
        """MedMCQA questions are knowledge-based with options"""
        # Include explanation if available
        exp = instance.get('exp', '')
        if exp:
            return f"Question: {instance['question']}\nExplanation available: {exp[:200]}"
        return instance['question']

    def format_for_evaluation(self, instance: Dict) -> Dict:
        """Format MedMCQA instance with options"""
        formatted = super().format_for_evaluation(instance)
        # Include options
        formatted['options'] = {
            'A': instance.get('opa', ''),
            'B': instance.get('opb', ''),
            'C': instance.get('opc', ''),
            'D': instance.get('opd', '')
        }
        formatted['subject'] = instance.get('subject_name', '')
        formatted['topic'] = instance.get('topic_name', '')
        return formatted


def get_dataset_loader(dataset_name: str, **kwargs) -> DatasetLoader:
    """Factory function to get appropriate dataset loader"""
    loaders = {
        'pubmedqa': PubMedQALoader,
        'medqa': MedQALoader,
        'medmcqa': MedMCQALoader
    }

    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    return loaders[dataset_name_lower](**kwargs)


# Available datasets with descriptions
AVAILABLE_DATASETS = {
    'pubmedqa': {
        'description': 'PubMedQA - Biomedical research questions with evidence-based context',
        'size': 1000,
        'loader_kwargs': {}
    },
    'medqa': {
        'description': 'MedQA - Clinical case scenarios from medical licensing exams',
        'size': 1273,
        'loader_kwargs': {'variant': 'US'}  # Can be US, Mainland, or Taiwan
    },
    'medmcqa': {
        'description': 'MedMCQA - Medical knowledge questions from Indian entrance exams',
        'size': 4183,
        'loader_kwargs': {}
    }
}
