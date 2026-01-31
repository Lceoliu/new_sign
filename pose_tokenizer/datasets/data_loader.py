"""Data loading utilities for pose tokenizer."""

import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Optional, Callable, Any
import numpy as np

from .pose_dataset import PoseDataset
from .preprocessing import PosePreprocessor


def collate_pose_sequences(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate function for pose sequences.

    Args:
        batch: List of samples from PoseDataset

    Returns:
        Batched data dictionary
    """
    # Extract components
    pose_sequences = []
    masks = []
    metadata = []

    for sample in batch:
        pose_sequences.append(sample['pose_sequence'])
        masks.append(sample['mask'])
        metadata.append(sample['metadata'])

    # Stack tensors
    batched_poses = torch.stack(pose_sequences, dim=0)  # (B, T, N, D)
    batched_masks = torch.stack(masks, dim=0)  # (B, T)

    return {
        'pose_sequence': batched_poses,
        'mask': batched_masks,
        'metadata': metadata
    }


def create_data_loaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    sequence_length: int = 64,
    num_keypoints: int = 133,
    keypoint_dim: int = 3,
    normalize: bool = True,
    augment: bool = True,
    distributed: bool = False,
    pin_memory: bool = True,
    drop_last: bool = True
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing.

    Args:
        data_path: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        sequence_length: Target sequence length
        num_keypoints: Number of keypoints per frame
        keypoint_dim: Dimension of each keypoint
        normalize: Whether to normalize pose coordinates
        augment: Whether to apply data augmentation
        distributed: Whether to use distributed training
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch

    Returns:
        Dictionary of data loaders for each split
    """
    data_loaders = {}

    for split in ['train', 'val', 'test']:
        # Create dataset
        dataset = PoseDataset(
            data_path=data_path,
            sequence_length=sequence_length,
            num_keypoints=num_keypoints,
            keypoint_dim=keypoint_dim,
            normalize=normalize,
            augment=augment and split == 'train',  # Only augment training data
            split=split
        )

        # Create sampler for distributed training
        sampler = None
        shuffle = split == 'train'

        if distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=drop_last
            )
            shuffle = False  # Sampler handles shuffling

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_pose_sequences,
            pin_memory=pin_memory,
            drop_last=drop_last and split == 'train',  # Only drop last for training
            persistent_workers=num_workers > 0
        )

        data_loaders[split] = data_loader

    return data_loaders


class PoseDataModule:
    """Data module for pose tokenizer training."""

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        sequence_length: int = 64,
        num_keypoints: int = 133,
        keypoint_dim: int = 3,
        normalize: bool = True,
        augment: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True
    ):
        """Initialize data module.

        Args:
            data_path: Path to dataset directory
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            sequence_length: Target sequence length
            num_keypoints: Number of keypoints per frame
            keypoint_dim: Dimension of each keypoint
            normalize: Whether to normalize pose coordinates
            augment: Whether to apply data augmentation
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop last incomplete batch
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.normalize = normalize
        self.augment = augment
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.data_loaders = None
        self.datasets = None

    def setup(self, distributed: bool = False) -> None:
        """Setup data loaders.

        Args:
            distributed: Whether to use distributed training
        """
        self.data_loaders = create_data_loaders(
            data_path=self.data_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sequence_length=self.sequence_length,
            num_keypoints=self.num_keypoints,
            keypoint_dim=self.keypoint_dim,
            normalize=self.normalize,
            augment=self.augment,
            distributed=distributed,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

        # Store datasets for easy access
        self.datasets = {
            split: loader.dataset for split, loader in self.data_loaders.items()
        }

    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        if self.data_loaders is None:
            self.setup()
        return self.data_loaders['train']

    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        if self.data_loaders is None:
            self.setup()
        return self.data_loaders['val']

    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        if self.data_loaders is None:
            self.setup()
        return self.data_loaders['test']

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        if self.datasets is None:
            self.setup()

        stats = {}
        for split, dataset in self.datasets.items():
            stats[split] = {
                'num_samples': len(dataset),
                'sequence_length': dataset.sequence_length,
                'num_keypoints': dataset.num_keypoints,
                'keypoint_dim': dataset.keypoint_dim
            }

            # Add normalization stats if available
            if hasattr(dataset, 'norm_stats') and dataset.norm_stats is not None:
                stats[split]['normalization'] = {
                    'mean': dataset.norm_stats['mean'].tolist(),
                    'std': dataset.norm_stats['std'].tolist()
                }

        return stats

    def get_sample_batch(self, split: str = 'train') -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing.

        Args:
            split: Dataset split to sample from

        Returns:
            Sample batch
        """
        if self.data_loaders is None:
            self.setup()

        data_loader = self.data_loaders[split]
        return next(iter(data_loader))


def compute_dataset_statistics(data_path: str) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics.

    Args:
        data_path: Path to dataset directory

    Returns:
        Dictionary with dataset statistics
    """
    # Create temporary dataset to analyze
    dataset = PoseDataset(
        data_path=data_path,
        normalize=False,  # Don't normalize for statistics
        augment=False,
        split='train'
    )

    if len(dataset) == 0:
        return {'error': 'No data found'}

    # Sample subset for efficiency
    sample_size = min(100, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    # Collect statistics
    sequence_lengths = []
    pose_coordinates = []
    confidence_scores = []

    for idx in indices:
        try:
            sample = dataset[idx]
            pose_seq = sample['pose_sequence'].numpy()  # (T, N, D)
            mask = sample['mask'].numpy()  # (T,)

            # Valid sequence length
            valid_length = mask.sum()
            sequence_lengths.append(valid_length)

            # Valid poses only
            valid_poses = pose_seq[mask]  # (valid_T, N, D)

            if valid_poses.shape[0] > 0:
                # Coordinates (x, y)
                coords = valid_poses[:, :, :2].reshape(-1, 2)
                pose_coordinates.append(coords)

                # Confidence scores if available
                if pose_seq.shape[-1] > 2:
                    conf = valid_poses[:, :, 2].flatten()
                    confidence_scores.append(conf)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # Compute statistics
    stats = {
        'dataset_size': len(dataset),
        'sample_size': len(sequence_lengths),
        'sequence_lengths': {
            'mean': float(np.mean(sequence_lengths)),
            'std': float(np.std(sequence_lengths)),
            'min': int(np.min(sequence_lengths)),
            'max': int(np.max(sequence_lengths)),
            'median': float(np.median(sequence_lengths))
        }
    }

    if pose_coordinates:
        all_coords = np.concatenate(pose_coordinates, axis=0)
        stats['pose_coordinates'] = {
            'mean': all_coords.mean(axis=0).tolist(),
            'std': all_coords.std(axis=0).tolist(),
            'min': all_coords.min(axis=0).tolist(),
            'max': all_coords.max(axis=0).tolist(),
            'shape': f"({all_coords.shape[0]}, {all_coords.shape[1]})"
        }

    if confidence_scores:
        all_conf = np.concatenate(confidence_scores)
        stats['confidence_scores'] = {
            'mean': float(all_conf.mean()),
            'std': float(all_conf.std()),
            'min': float(all_conf.min()),
            'max': float(all_conf.max()),
            'median': float(np.median(all_conf))
        }

    return stats