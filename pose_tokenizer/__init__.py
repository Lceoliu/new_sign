"""
Pose Tokenizer: Vector Quantization for Skeleton Sequences

A comprehensive toolkit for training and using VQ/RVQ tokenizers on pose sequences.
Supports unsupervised learning of discrete pose representations with reconstruction.
"""

__version__ = "1.0.0"
__author__ = "Uni-Sign Team"

from .models import PoseTokenizer, PoseTokenizerModel
from .datasets.pose_dataset import PoseDataset
from .training.trainer import PoseTokenizerTrainer
from .utils.evaluation import EvaluationSuite

__all__ = [
    "PoseTokenizer",
    "PoseTokenizerModel",
    "PoseDataset",
    "PoseTokenizerTrainer",
    "EvaluationSuite",
]
