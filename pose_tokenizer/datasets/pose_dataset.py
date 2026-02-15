"""Base dataset class for pose sequences."""

import os
import numpy as np
import torch
import pickle
import hashlib
import time
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json


class PoseDataset(Dataset):
    """Base dataset class for pose sequences.

    This class handles loading pose sequences from npz files and provides
    basic preprocessing functionality.
    """

    def __init__(
        self,
        data_path: str = None,
        sequence_length: int = 64,
        clip_mode: str = "random_window",
        window_T: int = None,
        num_keypoints: int = 133,
        keypoint_dim: int = 3,
        normalize: bool = True,
        augment: bool = False,
        split: str = "train",
        data_dir: str = None,
        normalize_by_wh: bool = True,
        log_load_every: int = 0,
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
        if data_path is None:
            data_path = data_dir
        if data_path is None:
            raise ValueError("data_path (or data_dir) must be provided")
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.clip_mode = clip_mode
        self.window_T = window_T or sequence_length
        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.normalize = normalize
        self.augment = augment
        self.split = split
        self.normalize_by_wh = normalize_by_wh
        self.log_load_every = log_load_every
        self._load_counter = 0

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
        rank, world_size = self._get_dist_info()
        cache_root = Path(os.environ.get("POSE_FILE_LIST_CACHE_DIR", str(self.data_path)))
        dataset_id = hashlib.md5(str(self.data_path).encode("utf-8")).hexdigest()[:12]
        cache_file = cache_root / f"{dataset_id}_{self.split}_files.txt"
        all_cache_file = cache_root / f"{dataset_id}_all_files.txt"

        if split_file.exists():
            # Load from split file
            files = self._read_file_list(split_file)
        elif cache_file.exists():
            files = self._read_file_list(cache_file)
        else:
            all_files = self._load_all_files(all_cache_file, rank, world_size)
            files = self._split_files(all_files)
            self._write_file_list(cache_file, files)

        return files

    @staticmethod
    def _stable_split_bucket(filename: str) -> int:
        """Return a stable bucket id in [0, 9] for split assignment."""
        digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 10

    @staticmethod
    def _get_dist_info() -> Tuple[int, int]:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    @staticmethod
    def _wait_for_file(file_path: Path, timeout_s: int = 900, interval_s: float = 1.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if file_path.exists():
                return True
            time.sleep(interval_s)
        return file_path.exists()

    def _read_file_list(self, list_path: Path) -> List[Path]:
        with open(list_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        return [self.data_path / line for line in lines]

    def _write_file_list(self, list_path: Path, files: List[Path]) -> bool:
        try:
            os.makedirs(list_path.parent, exist_ok=True)
            tmp_path = list_path.with_suffix(list_path.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                for file_path in files:
                    try:
                        rel = file_path.relative_to(self.data_path)
                    except ValueError:
                        rel = file_path
                    f.write(f"{rel}\n")
            os.replace(tmp_path, list_path)
            return True
        except OSError:
            return False

    def _discover_and_split_files(self) -> List[Path]:
        files = self._scan_all_files()
        return self._split_files(files)

    def _scan_all_files(self) -> List[Path]:
        files = list(self.data_path.glob("**/*.npz"))
        files += list(self.data_path.glob("**/*.pkl"))
        return sorted(files)

    def _split_files(self, files: List[Path]) -> List[Path]:
        if self.split == "train":
            files = [f for f in files if self._stable_split_bucket(f.name) < 8]
        elif self.split == "val":
            files = [f for f in files if self._stable_split_bucket(f.name) == 8]
        else:
            files = [f for f in files if self._stable_split_bucket(f.name) == 9]
        return files

    def _load_all_files(self, all_cache_file: Path, rank: int, world_size: int) -> List[Path]:
        if all_cache_file.exists():
            return self._read_file_list(all_cache_file)

        if world_size > 1 and rank != 0:
            if self._wait_for_file(all_cache_file, timeout_s=120):
                return self._read_file_list(all_cache_file)
            return self._scan_all_files()

        files = self._scan_all_files()
        self._write_file_list(all_cache_file, files)
        return files

    def _compute_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Compute normalization statistics from training data."""
        import os

        stats_file = os.environ.get(
            "NORM_STATS_PATH", self.data_path / "norm_stats.json"
        )
        if isinstance(stats_file, str):
            stats_file = Path(stats_file)

        if stats_file.exists():
            # Load precomputed stats
            with open(stats_file, 'r') as f:
                stats_dict = json.load(f)
                return {
                    'mean': np.array(stats_dict['mean']),
                    'std': np.array(stats_dict['std'])
                }

        rank, world_size = self._get_dist_info()
        if world_size > 1 and rank != 0:
            if self._wait_for_file(stats_file, timeout_s=1800):
                with open(stats_file, "r") as f:
                    stats_dict = json.load(f)
                return {
                    "mean": np.array(stats_dict["mean"]),
                    "std": np.array(stats_dict["std"]),
                }

        # Compute stats from data
        sample_limit = int(os.environ.get("NORM_STATS_SAMPLE_LIMIT", "100"))
        sample_limit = max(sample_limit, 1)
        all_poses = []
        for file_path in self.data_files[:sample_limit]:  # Sample subset for efficiency
            try:
                pose_seq, _ = self._load_pose_file(file_path)
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
            os.makedirs(stats_file.parent, exist_ok=True)
            tmp_stats = stats_file.with_suffix(stats_file.suffix + ".tmp")
            with open(tmp_stats, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            os.replace(tmp_stats, stats_file)

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
            pose_seq, extra_meta = self._load_pose_file(file_path)

            # Get metadata if available
            metadata = {}
            if extra_meta:
                metadata.update(extra_meta)
            metadata['file_path'] = str(file_path)

            # Process sequence
            pose_seq, mask = self._process_sequence(pose_seq)
            pose_seq = self._sanitize_pose_seq(pose_seq)

            # Apply normalization
            if self.normalize and self.norm_stats is not None:
                pose_seq = self._normalize_pose(pose_seq)

            # Apply augmentation
            if self.augment and self.split == "train":
                pose_seq = self._augment_pose(pose_seq, mask)
            pose_seq = self._sanitize_pose_seq(pose_seq)

            return {
                'poses': torch.from_numpy(pose_seq).float(),
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

        if self.clip_mode == "full":
            output_seq = pose_seq.astype(np.float32)
            mask = np.ones(T, dtype=bool)
            return output_seq, mask

        window_T = self.window_T
        output_seq = np.zeros((window_T, N, D), dtype=np.float32)
        mask = np.zeros(window_T, dtype=bool)

        if T >= window_T:
            if self.split == "train":
                start_idx = np.random.randint(0, T - window_T + 1)
            else:
                start_idx = (T - window_T) // 2
            output_seq = pose_seq[start_idx:start_idx + window_T]
            mask[:] = True
        else:
            output_seq[:T] = pose_seq
            mask[:T] = True

        return output_seq.astype(np.float32), mask

    def _load_pose_file(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load pose sequence from npz or pkl file."""
        if file_path.suffix == ".pkl":
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            keypoints = np.asarray(data["keypoints"])
            # Squeeze possible singleton dims: (1,T,133,2) or (T,1,133,2)
            if keypoints.ndim == 4:
                if keypoints.shape[0] == 1:
                    keypoints = keypoints[0]
                elif keypoints.shape[1] == 1:
                    keypoints = keypoints[:, 0]
                elif keypoints.shape[2] == 1:
                    keypoints = keypoints[:, :, 0]
            if keypoints.ndim != 3:
                raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")
            scores = data.get("scores", None)
            w_h = data.get("w_h", None)

            if self.normalize_by_wh and w_h is not None:
                w, h = float(w_h[0]), float(w_h[1])
                if w > 0 and h > 0:
                    keypoints = keypoints.astype(np.float32)
                    keypoints[..., 0] = keypoints[..., 0] / w
                    keypoints[..., 1] = keypoints[..., 1] / h

            if scores is None:
                pose_seq = keypoints.astype(np.float32)
            else:
                scores = np.asarray(scores)
                # Squeeze singleton dims
                if scores.ndim > 1:
                    scores = np.squeeze(scores)
                # Normalize shape to (T, J)
                if scores.ndim == 1:
                    if scores.shape[0] != keypoints.shape[0]:
                        raise ValueError(
                            f"Scores length {scores.shape[0]} != T {keypoints.shape[0]}"
                        )
                    scores = np.repeat(scores[:, None], keypoints.shape[1], axis=1)
                elif scores.ndim == 2:
                    if (
                        scores.shape[0] == keypoints.shape[0]
                        and scores.shape[1] == keypoints.shape[1]
                    ):
                        pass
                    elif (
                        scores.shape[1] == keypoints.shape[0]
                        and scores.shape[0] == keypoints.shape[1]
                    ):
                        scores = scores.T
                    elif scores.shape[0] == keypoints.shape[0] and scores.shape[1] == 1:
                        scores = np.repeat(scores, keypoints.shape[1], axis=1)
                    else:
                        raise ValueError(f"Unexpected scores shape: {scores.shape}")
                else:
                    raise ValueError(f"Unexpected scores shape: {scores.shape}")

                scores = scores[..., None]
                pose_seq = np.concatenate([keypoints, scores], axis=-1).astype(
                    np.float32
                )

            self._maybe_log_load(file_path)
            return pose_seq, {"w_h": w_h}

        # npz fallback
        data = np.load(file_path)
        if 'pose' in data:
            pose_seq = data['pose']
        else:
            pose_seq = data['pose_sequence']  # Shape: (T, N, D)

        if 'conf' in data and pose_seq.shape[-1] == 2:
            conf = data['conf']
            if conf.ndim == 2:
                conf = conf[..., None]
            pose_seq = np.concatenate([pose_seq, conf], axis=-1)

        meta = {}
        if 'metadata' in data:
            meta = data['metadata'].item()
        return self._sanitize_pose_seq(pose_seq), meta

    def _maybe_log_load(self, file_path: Path) -> None:
        """Log pkl loading progress every N samples."""
        if not self.log_load_every or self.log_load_every <= 0:
            return
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.id != 0:
            return
        self._load_counter += 1
        if self._load_counter % self.log_load_every == 0:
            print(
                f"[PoseDataset] Loaded {self._load_counter} pkl files. Latest: {file_path.name}"
            )

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

    def _sanitize_pose_seq(self, pose_seq: np.ndarray) -> np.ndarray:
        """Replace invalid values and clamp extreme coordinates."""
        pose_seq = np.asarray(pose_seq, dtype=np.float32)

        if not np.isfinite(pose_seq).all():
            pose_seq = np.nan_to_num(pose_seq, nan=0.0, posinf=0.0, neginf=0.0)

        if pose_seq.shape[-1] >= 2:
            pose_seq[..., :2] = np.clip(pose_seq[..., :2], -20.0, 20.0)
        if pose_seq.shape[-1] >= 3:
            pose_seq[..., 2] = np.clip(pose_seq[..., 2], 0.0, 1.0)

        return pose_seq

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
            'poses': torch.zeros(self.sequence_length, self.num_keypoints, self.keypoint_dim),
            'mask': torch.zeros(self.sequence_length, dtype=torch.bool),
            'metadata': {'file_path': 'dummy', 'error': True}
        }
