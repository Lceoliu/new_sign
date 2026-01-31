"""Data preprocessing utilities for pose sequences."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d


class PosePreprocessor:
    """Preprocessing utilities for pose sequences."""

    def __init__(
        self,
        num_keypoints: int = 133,
        keypoint_dim: int = 3,
        confidence_threshold: float = 0.5
    ):
        """Initialize preprocessor.

        Args:
            num_keypoints: Number of keypoints per frame
            keypoint_dim: Dimension of each keypoint
            confidence_threshold: Minimum confidence for valid keypoints
        """
        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.confidence_threshold = confidence_threshold

    def process_raw_poses(
        self,
        raw_poses: np.ndarray,
        target_fps: Optional[int] = None,
        smooth: bool = True,
        fill_missing: bool = True
    ) -> np.ndarray:
        """Process raw pose sequences.

        Args:
            raw_poses: Raw pose array (T, N, D)
            target_fps: Target frame rate for resampling
            smooth: Whether to apply smoothing
            fill_missing: Whether to fill missing keypoints

        Returns:
            Processed pose sequence
        """
        poses = raw_poses.copy()

        # Fill missing keypoints
        if fill_missing:
            poses = self._fill_missing_keypoints(poses)

        # Smooth trajectories
        if smooth:
            poses = self._smooth_poses(poses)

        # Resample to target FPS
        if target_fps is not None:
            poses = self._resample_poses(poses, target_fps)

        # Normalize pose coordinates
        poses = self._normalize_coordinates(poses)

        return poses

    def _fill_missing_keypoints(self, poses: np.ndarray) -> np.ndarray:
        """Fill missing keypoints using interpolation.

        Args:
            poses: Pose sequence (T, N, D)

        Returns:
            Pose sequence with filled keypoints
        """
        T, N, D = poses.shape
        filled_poses = poses.copy()

        for n in range(N):  # For each keypoint
            for d in range(min(2, D)):  # Only interpolate x, y coordinates
                trajectory = poses[:, n, d]

                # Find valid points (confidence > threshold if available)
                if D > 2:  # Has confidence channel
                    confidence = poses[:, n, 2]
                    valid_mask = confidence > self.confidence_threshold
                else:
                    # Assume non-zero coordinates are valid
                    valid_mask = trajectory != 0

                if valid_mask.sum() < 2:
                    continue  # Not enough valid points for interpolation

                # Get valid indices and values
                valid_indices = np.where(valid_mask)[0]
                valid_values = trajectory[valid_mask]

                # Interpolate missing values
                if len(valid_indices) >= 2:
                    interp_func = interp1d(
                        valid_indices, valid_values,
                        kind='linear', bounds_error=False,
                        fill_value='extrapolate'
                    )

                    # Fill all frames
                    filled_trajectory = interp_func(np.arange(T))
                    filled_poses[:, n, d] = filled_trajectory

        return filled_poses

    def _smooth_poses(self, poses: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply smoothing to pose trajectories.

        Args:
            poses: Pose sequence (T, N, D)
            window_size: Smoothing window size

        Returns:
            Smoothed pose sequence
        """
        if window_size <= 1:
            return poses

        T, N, D = poses.shape
        smoothed_poses = poses.copy()

        # Apply moving average smoothing
        for n in range(N):
            for d in range(min(2, D)):  # Only smooth x, y coordinates
                trajectory = poses[:, n, d]

                # Pad trajectory for edge handling
                padded = np.pad(trajectory, window_size//2, mode='edge')

                # Apply moving average
                smoothed = np.convolve(
                    padded, np.ones(window_size)/window_size, mode='valid'
                )

                smoothed_poses[:, n, d] = smoothed

        return smoothed_poses

    def _resample_poses(self, poses: np.ndarray, target_fps: int) -> np.ndarray:
        """Resample pose sequence to target frame rate.

        Args:
            poses: Pose sequence (T, N, D)
            target_fps: Target frame rate

        Returns:
            Resampled pose sequence
        """
        T, N, D = poses.shape

        # Assume original FPS is 30 (common for video)
        original_fps = 30

        # Calculate new sequence length
        new_T = int(T * target_fps / original_fps)

        if new_T == T:
            return poses

        # Create interpolation indices
        original_indices = np.linspace(0, T-1, T)
        new_indices = np.linspace(0, T-1, new_T)

        # Resample each trajectory
        resampled_poses = np.zeros((new_T, N, D), dtype=poses.dtype)

        for n in range(N):
            for d in range(D):
                trajectory = poses[:, n, d]

                # Interpolate
                interp_func = interp1d(
                    original_indices, trajectory,
                    kind='linear', bounds_error=False,
                    fill_value='extrapolate'
                )

                resampled_poses[:, n, d] = interp_func(new_indices)

        return resampled_poses

    def _normalize_coordinates(self, poses: np.ndarray) -> np.ndarray:
        """Normalize pose coordinates to [-1, 1] range.

        Args:
            poses: Pose sequence (T, N, D)

        Returns:
            Normalized pose sequence
        """
        if self.keypoint_dim < 2:
            return poses

        normalized_poses = poses.copy()
        T, N, D = poses.shape

        for t in range(T):
            frame_poses = poses[t]  # (N, D)

            # Get valid keypoints (confidence > threshold if available)
            if D > 2:
                valid_mask = frame_poses[:, 2] > self.confidence_threshold
            else:
                valid_mask = np.any(frame_poses[:, :2] != 0, axis=1)

            if valid_mask.sum() == 0:
                continue

            valid_points = frame_poses[valid_mask, :2]  # (M, 2)

            # Compute bounding box
            min_coords = np.min(valid_points, axis=0)
            max_coords = np.max(valid_points, axis=0)

            # Avoid division by zero
            range_coords = max_coords - min_coords
            range_coords = np.maximum(range_coords, 1e-6)

            # Normalize to [-1, 1]
            center = (min_coords + max_coords) / 2
            scale = range_coords / 2

            normalized_poses[t, :, :2] = (poses[t, :, :2] - center) / scale

        return normalized_poses

    def extract_pose_features(self, poses: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract additional features from pose sequences.

        Args:
            poses: Pose sequence (T, N, D)

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Velocity features
        features['velocity'] = self._compute_velocity(poses)

        # Acceleration features
        features['acceleration'] = self._compute_acceleration(poses)

        # Bone length features (if we have enough keypoints)
        if self.num_keypoints >= 33:  # MediaPipe pose has 33 landmarks
            features['bone_lengths'] = self._compute_bone_lengths(poses)

        # Pose angles
        features['angles'] = self._compute_pose_angles(poses)

        return features

    def _compute_velocity(self, poses: np.ndarray) -> np.ndarray:
        """Compute velocity features.

        Args:
            poses: Pose sequence (T, N, D)

        Returns:
            Velocity features (T-1, N, 2)
        """
        if poses.shape[0] < 2:
            return np.zeros((0, poses.shape[1], 2))

        # Compute frame-to-frame differences
        velocity = np.diff(poses[:, :, :2], axis=0)

        return velocity

    def _compute_acceleration(self, poses: np.ndarray) -> np.ndarray:
        """Compute acceleration features.

        Args:
            poses: Pose sequence (T, N, D)

        Returns:
            Acceleration features (T-2, N, 2)
        """
        velocity = self._compute_velocity(poses)

        if velocity.shape[0] < 2:
            return np.zeros((0, poses.shape[1], 2))

        # Compute acceleration as velocity differences
        acceleration = np.diff(velocity, axis=0)

        return acceleration

    def _compute_bone_lengths(self, poses: np.ndarray) -> np.ndarray:
        """Compute bone length features for MediaPipe pose.

        Args:
            poses: Pose sequence (T, N, D)

        Returns:
            Bone length features (T, num_bones)
        """
        # MediaPipe pose connections (simplified)
        connections = [
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (24, 26), (25, 27), (26, 28),  # Legs
            (27, 29), (28, 30), (29, 31), (30, 32)  # Feet
        ]

        T = poses.shape[0]
        num_bones = len(connections)
        bone_lengths = np.zeros((T, num_bones))

        for t in range(T):
            frame_poses = poses[t]  # (N, D)

            for i, (joint1, joint2) in enumerate(connections):
                if joint1 < poses.shape[1] and joint2 < poses.shape[1]:
                    p1 = frame_poses[joint1, :2]
                    p2 = frame_poses[joint2, :2]

                    # Compute Euclidean distance
                    bone_lengths[t, i] = np.linalg.norm(p2 - p1)

        return bone_lengths

    def _compute_pose_angles(self, poses: np.ndarray) -> np.ndarray:
        """Compute pose angle features.

        Args:
            poses: Pose sequence (T, N, D)

        Returns:
            Angle features (T, num_angles)
        """
        # Define angle triplets (joint1, vertex, joint2)
        angle_triplets = [
            (13, 11, 23),  # Left shoulder angle
            (14, 12, 24),  # Right shoulder angle
            (11, 13, 15),  # Left elbow angle
            (12, 14, 16),  # Right elbow angle
            (25, 23, 11),  # Left hip angle
            (26, 24, 12),  # Right hip angle
            (23, 25, 27),  # Left knee angle
            (24, 26, 28),  # Right knee angle
        ]

        T = poses.shape[0]
        num_angles = len(angle_triplets)
        angles = np.zeros((T, num_angles))

        for t in range(T):
            frame_poses = poses[t]  # (N, D)

            for i, (j1, vertex, j2) in enumerate(angle_triplets):
                if all(j < poses.shape[1] for j in [j1, vertex, j2]):
                    p1 = frame_poses[j1, :2]
                    pv = frame_poses[vertex, :2]
                    p2 = frame_poses[j2, :2]

                    # Compute angle
                    v1 = p1 - pv
                    v2 = p2 - pv

                    # Avoid division by zero
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)

                    if norm1 > 1e-6 and norm2 > 1e-6:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angles[t, i] = np.arccos(cos_angle)

        return angles


def create_data_splits(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """Create train/val/test splits for pose dataset.

    Args:
        data_path: Path to dataset directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducible splits
    """
    data_path = Path(data_path)

    # Find all npz files
    all_files = list(data_path.glob("**/*.npz"))

    if not all_files:
        raise ValueError(f"No .npz files found in {data_path}")

    # Set random seed
    np.random.seed(seed)

    # Shuffle files
    np.random.shuffle(all_files)

    # Calculate split indices
    n_files = len(all_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)

    # Create splits
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    # Save split files
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        split_file = data_path / f"{split_name}_files.txt"
        with open(split_file, 'w') as f:
            for file_path in files:
                # Store relative path
                rel_path = file_path.relative_to(data_path)
                f.write(f"{rel_path}\n")

    print(f"Created splits: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")