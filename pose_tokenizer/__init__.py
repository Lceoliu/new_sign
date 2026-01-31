"""
Pose Tokenizer: Vector Quantization for Skeleton Sequences

A comprehensive toolkit for training and using VQ/RVQ tokenizers on pose sequences.
Supports unsupervised learning of discrete pose representations with reconstruction.
"""

__version__ = "1.0.0"
__author__ = "Uni-Sign Team"

from .models import PoseTokenizer
from .data import PoseDataset
from .train import PoseTokenizerTrainer
from .eval import PoseTokenizerEvaluator

__all__ = [
    "PoseTokenizer",
    "PoseDataset",
    "PoseTokenizerTrainer",
    "PoseTokenizerEvaluator"
]