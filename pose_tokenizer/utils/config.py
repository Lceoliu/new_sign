"""Configuration management for pose tokenizer."""

import os
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str = "data/pose_sequences"
    batch_size: int = 32
    num_workers: int = 4
    sequence_length: int = 64
    keypoint_dim: int = 3  # x, y, confidence
    num_keypoints: int = 133  # MediaPipe pose landmarks
    normalize: bool = True
    augment: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class ModelConfig:
    """Model configuration."""
    # Encoder config
    encoder_type: str = "stgcn"
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1

    # Quantizer config
    codebook_size: int = 1024
    num_quantizers: int = 8
    commitment_cost: float = 0.25

    # Decoder config
    decoder_type: str = "stgcn"
    output_dim: int = 399  # 133 * 3


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Loss weights
    reconstruction_weight: float = 1.0
    vq_weight: float = 1.0

    # Checkpointing
    save_every: int = 10
    eval_every: int = 5

    # Logging
    log_every: int = 100
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "pose-tokenizer"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    # General settings
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
    experiment_name: str = "pose_tokenizer"

    def __post_init__(self):
        """Post-initialization validation."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate splits sum to 1.0
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    # Create config objects
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))

    # Extract general settings
    general_settings = {k: v for k, v in config_dict.items()
                       if k not in ['data', 'model', 'training']}

    return Config(
        data=data_config,
        model=model_config,
        training=training_config,
        **general_settings
    )


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        'data': asdict(config.data),
        'model': asdict(config.model),
        'training': asdict(config.training),
        'seed': config.seed,
        'device': config.device,
        'output_dir': config.output_dir,
        'experiment_name': config.experiment_name
    }

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig()
    )