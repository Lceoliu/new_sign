"""Base dataset class for pose sequences."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json


class PoseDataset(Dataset):
    """Base dataset class for pose sequences.

    This class handles loading pose sequences from npz files and provides
    basic preprocessing functionality.
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 64,
        num_keypoints: int = 133,
        keypoint_dim: int = 3,
        normalize: bool = True,
        augment: bool = False,
        split: str = "train"
    ):
        """Initialize pose dataset.

        Args:
            data_path: Path to dataset directory
            sequence_length: Target sequence length
            num_keypoints: Number of keypoints per frame
            keypoint_dim: Dimension of each keypoint (x, y, confidence)
            normalize: Whether to normalize pose coordinates
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.normalize = normalize
        self.augment = augment
        self.split = split

        # Load dataset files
        self.data_files = self._load_data_files()

        # Compute normalization statistics if needed
        if self.normalize:
            self.norm_stats = self._compute_normalization_stats()
        else:
            self.norm_stats = None

    def _load_data_files(self) -> List[Path]:
        """Load list of data files for the current split."""
        split_file = self.data_path / f"{self.split}_files.txt"

        if split_file.exists():
            # Load from split file
            with open(split_file, 'r') as f:
                files = [self.data_path / line.strip() for line in f.readlines()]
        else:
            # Auto-discover files
            files = list(self.data_path.glob("**/*.npz"))

            # Simple split based on filename hash
            if self.split == "train":
                files = [f for f in files if hash(f.name) % 10 < 8]
            elif self.split == "val":
                files = [f for f in files if hash(f.name) % 10 == 8]
            else:  # test
                files = [f for f in files if hash(f.name) % 10 == 9]

        return files

    def _compute_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Compute normalization statistics from training data."""
        stats_file = self.data_path / "norm_stats.json"

        if stats_file.exists():
            # Load precomputed stats
            with open(stats_file, 'r') as f:
                stats_dict = json.load(f)
                return {
                    'mean': np.array(stats_dict['mean']),
                    'std': np.array(stats_dict['std'])
                }

        # Compute stats from data
        all_poses = []
        for file_path in self.data_files[:100]:  # Sample subset for efficiency
            try:
                data = np.load(file_path)
                pose_seq = data['pose_sequence']  # Shape: (T, N, D)
                all_poses.append(pose_seq.reshape(-1, self.keypoint_dim))
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue

        if not all_poses:
            # Default normalization
            return {
                'mean': np.zeros(self.keypoint_dim),
                'std': np.ones(self.keypoint_dim)
            }

        all_poses = np.concatenate(all_poses, axis=0)

        # Compute mean and std, handling confidence separately
        if self.keypoint_dim == 3:  # x, y, confidence
            # Don't normalize confidence channel
            mean = np.zeros(3)
            std = np.ones(3)

            # Only normalize x, y coordinates
            mean[:2] = np.mean(all_poses[:, :2], axis=0)
            std[:2] = np.std(all_poses[:, :2], axis=0)
            std[:2] = np.maximum(std[:2], 1e-6)  # Avoid division by zero
        else:
            mean = np.mean(all_poses, axis=0)
            std = np.std(all_poses, axis=0)
            std = np.maximum(std, 1e-6)

        stats = {'mean': mean, 'std': std}

        # Save stats for future use
        if self.split == "train":
            stats_dict = {
                'mean': mean.tolist(),
                'std': std.tolist()
            }
            with open(stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)

        return stats

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - pose_sequence: Pose sequence tensor (T, N, D)
                - mask: Valid frame mask (T,)
                - metadata: Additional metadata
        """
        file_path = self.data_files[idx]

        try:
            # Load pose sequence
            data = np.load(file_path)
            pose_seq = data['pose_sequence']  # Shape: (T, N, D)

            # Get metadata if available
            metadata = {}
            if 'metadata' in data:
                metadata = data['metadata'].item()
            metadata['file_path'] = str(file_path)

            # Process sequence
            pose_seq, mask = self._process_sequence(pose_seq)

            # Apply normalization
            if self.normalize and self.norm_stats is not None:
                pose_seq = self._normalize_pose(pose_seq)

            # Apply augmentation
            if self.augment and self.split == "train":
                pose_seq = self._augment_pose(pose_seq, mask)

            return {
                'pose_sequence': torch.from_numpy(pose_seq).float(),
                'mask': torch.from_numpy(mask).bool(),
                'metadata': metadata
            }

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy data
            return self._get_dummy_sample()

    def _process_sequence(self, pose_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process pose sequence to target length.

        Args:
            pose_seq: Input pose sequence (T, N, D)

        Returns:
            Processed sequence and validity mask
        """
        T, N, D = pose_seq.shape

        # Create output arrays
        output_seq = np.zeros((self.sequence_length, N, D), dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=bool)

        if T >= self.sequence_length:
            # Truncate or sample
            if self.split == "train":
                # Random crop during training
                start_idx = np.random.randint(0, T - self.sequence_length + 1)
            else:
                # Center crop during evaluation
                start_idx = (T - self.sequence_length) // 2

            output_seq = pose_seq[start_idx:start_idx + self.sequence_length]
            mask[:] = True
        else:
            # Pad sequence
            output_seq[:T] = pose_seq
            mask[:T] = True

        return output_seq, mask

    def _normalize_pose(self, pose_seq: np.ndarray) -> np.ndarray:
        """Normalize pose coordinates.

        Args:
            pose_seq: Pose sequence (T, N, D)

        Returns:
            Normalized pose sequence
        """
        mean = self.norm_stats['mean']
        std = self.norm_stats['std']

        # Apply normalization
        normalized = (pose_seq - mean) / std

        return normalized.astype(np.float32)

    def _augment_pose(self, pose_seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply data augmentation to pose sequence.

        Args:
            pose_seq: Pose sequence (T, N, D)
            mask: Valid frame mask (T,)

        Returns:
            Augmented pose sequence
        """
        # Only augment valid frames
        valid_frames = mask.sum()
        if valid_frames == 0:
            return pose_seq

        augmented = pose_seq.copy()

        # Random horizontal flip
        if np.random.random() < 0.5:
            augmented = self._flip_pose_horizontal(augmented, mask)

        # Random rotation
        if np.random.random() < 0.3:
            angle = np.random.uniform(-15, 15)  # degrees
            augmented = self._rotate_pose(augmented, mask, angle)

        # Random scale
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            augmented = self._scale_pose(augmented, mask, scale)

        # Random noise
        if np.random.random() < 0.2:
            noise_std = 0.01
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented[mask] += noise[mask]

        return augmented

    def _flip_pose_horizontal(self, pose_seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Flip pose horizontally."""
        flipped = pose_seq.copy()

        # Flip x coordinates
        if self.keypoint_dim >= 2:
            flipped[mask, :, 0] = -flipped[mask, :, 0]

        # TODO: Add keypoint-specific flipping logic for MediaPipe landmarks
        # This would involve swapping left/right keypoints

        return flipped

    def _rotate_pose(self, pose_seq: np.ndarray, mask: np.ndarray, angle: float) -> np.ndarray:
        """Rotate pose by given angle."""
        if self.keypoint_dim < 2:
            return pose_seq

        rotated = pose_seq.copy()
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Rotation matrix
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Apply rotation to x, y coordinates
        for t in range(self.sequence_length):
            if mask[t]:
                xy = rotated[t, :, :2]  # (N, 2)
                rotated[t, :, :2] = xy @ R.T

        return rotated

    def _scale_pose(self, pose_seq: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
        """Scale pose by given factor."""
        if self.keypoint_dim < 2:
            return pose_seq

        scaled = pose_seq.copy()
        scaled[mask, :, :2] *= scale

        return scaled

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return dummy sample for error cases."""
        return {
            'pose_sequence': torch.zeros(self.sequence_length, self.num_keypoints, self.keypoint_dim),
            'mask': torch.zeros(self.sequence_length, dtype=torch.bool),
            'metadata': {'file_path': 'dummy', 'error': True}
        }