"""
Data module - Dataset loading utilities (v2.3-style loaders).
"""

from .dataset_loaders import (
    DatasetLoader,
    PubMedQALoader,
    MedQALoader,
    MedMCQALoader,
    get_dataset_loader,
    AVAILABLE_DATASETS,
)

__all__ = [
    "DatasetLoader",
    "PubMedQALoader",
    "MedQALoader",
    "MedMCQALoader",
    "get_dataset_loader",
    "AVAILABLE_DATASETS",
]
