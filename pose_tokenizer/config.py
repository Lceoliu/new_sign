"""Configuration utilities for pose tokenizer."""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List
from pathlib import Path
import yaml


@dataclass
class PoseTokenizerConfig:
    """Configuration schema for pose tokenizer training/evaluation."""

    # Model Architecture
    num_keypoints: int = 133
    keypoint_dim: int = 3
    sequence_length: int = 64
    hidden_dim: int = 256
    num_layers: int = 4

    # ST-GCN Encoder
    stgcn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    stgcn_temporal_kernels: List[int] = field(default_factory=lambda: [9, 5, 3])
    stgcn_dropout: float = 0.1

    # RVQ Tokenizer
    num_quantizers: int = 4
    codebook_size: int = 1024
    commitment_cost: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5
    dead_code_threshold: float = 1e-5

    # Decoder
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64])
    decoder_dropout: float = 0.1

    # Training Parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

    # Loss Weights
    reconstruction_weight: float = 1.0
    quantization_weight: float = 1.0
    perceptual_weight: float = 0.1
    temporal_weight: float = 0.1
    adversarial_weight: float = 0.01
    w_pos: float = 1.0
    w_vel: float = 0.5
    w_acc: float = 0.25
    w_bone: float = 0.5
    w_vq: float = 1.0

    # Optimizer
    optimizer_type: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: float = 0.9

    # Scheduler
    scheduler_type: str = "warmup_cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1

    # Data
    data_dir: str = "data/CSL-News/pose_format"
    normalize_poses: bool = True
    augment_data: bool = True
    num_workers: int = 4
    clip_mode: str = "random_window"
    window_T: int = 256
    normalize_by_wh: bool = True
    load_log_every: int = 0

    # Training Settings
    use_amp: bool = True
    distributed: bool = False
    val_interval: int = 1
    log_interval: int = 100
    save_interval: int = 10

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    skeleton_def: str = ""

    # Evaluation
    eval_batch_size: int = 64
    eval_num_samples: int = 1000

    # Export Settings
    export_format: str = "onnx"
    export_dynamic_axes: bool = True
    export_opset_version: int = 11

    def __post_init__(self):
        """Coerce numeric/bool fields when YAML provides string values (e.g. 1e-6)."""
        int_fields = {
            "num_keypoints",
            "keypoint_dim",
            "sequence_length",
            "hidden_dim",
            "num_layers",
            "num_quantizers",
            "codebook_size",
            "batch_size",
            "num_epochs",
            "warmup_epochs",
            "step_size",
            "num_workers",
            "window_T",
            "val_interval",
            "log_interval",
            "save_interval",
            "eval_batch_size",
            "eval_num_samples",
            "export_opset_version",
        }
        float_fields = {
            "stgcn_dropout",
            "commitment_cost",
            "decay",
            "epsilon",
            "dead_code_threshold",
            "decoder_dropout",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "reconstruction_weight",
            "quantization_weight",
            "perceptual_weight",
            "temporal_weight",
            "adversarial_weight",
            "w_pos",
            "w_vel",
            "w_acc",
            "w_bone",
            "w_vq",
            "beta1",
            "beta2",
            "momentum",
            "min_lr",
            "gamma",
        }
        bool_fields = {
            "normalize_poses",
            "augment_data",
            "normalize_by_wh",
            "use_amp",
            "distributed",
            "export_dynamic_axes",
        }

        for name in int_fields:
            value = getattr(self, name, None)
            if isinstance(value, str):
                try:
                    setattr(self, name, int(float(value)))
                except ValueError as e:
                    raise ValueError(f"Invalid integer value for {name}: {value}") from e

        for name in float_fields:
            value = getattr(self, name, None)
            if isinstance(value, str):
                try:
                    setattr(self, name, float(value))
                except ValueError as e:
                    raise ValueError(f"Invalid float value for {name}: {value}") from e

        for name in bool_fields:
            value = getattr(self, name, None)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "y", "on"}:
                    setattr(self, name, True)
                elif lowered in {"0", "false", "no", "n", "off"}:
                    setattr(self, name, False)
                else:
                    raise ValueError(f"Invalid boolean value for {name}: {value}")

    @classmethod
    def from_yaml(cls, path: str) -> "PoseTokenizerConfig":
        """Load configuration from YAML file."""
        config_path = Path(path)
        with config_path.open("r") as f:
            raw = yaml.safe_load(f) or {}

        # Filter only known fields
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
