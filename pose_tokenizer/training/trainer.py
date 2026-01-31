"""Training utilities for pose tokenizer."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Tuple, List, Any
import math
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm

from .losses import PoseTokenizerLoss


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model parameters
    num_keypoints: int = 133
    keypoint_dim: int = 3
    hidden_dim: int = 256
    num_quantizers: int = 4
    codebook_size: int = 1024

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

    # Loss weights
    reconstruction_weight: float = 1.0
    quantization_weight: float = 1.0
    perceptual_weight: float = 0.1
    temporal_weight: float = 0.1
    adversarial_weight: float = 0.01

    # Scheduler parameters
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'exponential', 'warmup_cosine'
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1

    # Optimization
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: float = 0.9

    # Validation and logging
    val_interval: int = 1
    log_interval: int = 100
    save_interval: int = 10

    # Paths
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    # Mixed precision
    use_amp: bool = True

    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing scheduler with warmup."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class EMAModel:
    """Exponential Moving Average model."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize EMA model.

        Args:
            model: Model to track
            decay: EMA decay rate
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        """Apply shadow parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class MetricsTracker:
    """Track training metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.history = {}

    def update(self, metrics: Dict[str, float]):
        """Update metrics.

        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.history[key] = []

            self.metrics[key].append(value)
            self.history[key].append(value)

    def get_average(self, key: str, window: int = None) -> float:
        """Get average of metric.

        Args:
            key: Metric key
            window: Window size for moving average

        Returns:
            Average value
        """
        if key not in self.metrics or not self.metrics[key]:
            return 0.0

        values = self.metrics[key]
        if window is not None:
            values = values[-window:]

        return sum(values) / len(values)

    def reset(self):
        """Reset current metrics (keep history)."""
        self.metrics = {key: [] for key in self.metrics}

    def get_summary(self) -> Dict[str, float]:
        """Get summary of current metrics."""
        return {key: self.get_average(key) for key in self.metrics}


class PoseTokenizerTrainer:
    """Trainer for pose tokenizer."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device = None,
        logger: logging.Logger = None
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            device: Training device
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize loss function
        self.criterion = PoseTokenizerLoss(
            reconstruction_weight=config.reconstruction_weight,
            quantization_weight=config.quantization_weight,
            perceptual_weight=config.perceptual_weight,
            temporal_weight=config.temporal_weight,
            adversarial_weight=config.adversarial_weight,
            use_temporal=True
        )

        # Initialize EMA
        self.ema = EMAModel(model, decay=0.999)

        # Initialize metrics tracker
        self.metrics = MetricsTracker()

        # Initialize mixed precision scaler
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

    def _create_scheduler(self) -> _LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type == 'warmup_cosine':
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.num_epochs,
                min_lr=self.config.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary of losses
        """
        self.model.train()

        # Move batch to device
        poses = batch['poses'].to(self.device)  # (B, T, N, D)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(self.device)

        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Encode
                encoded = self.model.encoder(poses, mask)

                # Quantize
                quantized, indices, quant_losses = self.model.tokenizer(encoded)

                # Decode
                reconstructed = self.model.decoder(quantized, poses.shape[1:])

                # Compute losses
                losses = self.criterion(
                    reconstructed, poses, quant_losses, mask
                )
        else:
            # Encode
            encoded = self.model.encoder(poses, mask)

            # Quantize
            quantized, indices, quant_losses = self.model.tokenizer(encoded)

            # Decode
            reconstructed = self.model.decoder(quantized, poses.shape[1:])

            # Compute losses
            losses = self.criterion(
                reconstructed, poses, quant_losses, mask
            )

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(losses['total_loss']).backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            self.optimizer.step()

        # Update EMA
        self.ema.update(self.model)

        # Convert losses to float
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in losses.items()}

        return loss_dict

    def validate(self, val_loader) -> Dict[str, float]:
        """Validation loop.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = MetricsTracker()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                # Move batch to device
                poses = batch['poses'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)

                # Forward pass
                encoded = self.model.encoder(poses, mask)
                quantized, indices, quant_losses = self.model.tokenizer(encoded)
                reconstructed = self.model.decoder(quantized, poses.shape[1:])

                # Compute losses
                losses = self.criterion(
                    reconstructed, poses, quant_losses, mask
                )

                # Update metrics
                loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in losses.items()}
                val_metrics.update(loss_dict)

        return val_metrics.get_summary()

    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        self.metrics.reset()

        # Training loop
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Training step
            losses = self.train_step(batch)
            self.metrics.update(losses)
            self.global_step += 1

            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'recon': f"{losses.get('reconstruction_loss', 0):.4f}",
                    'quant': f"{losses.get('quant_vq_loss', 0):.4f}",
                    'lr': f"{current_lr:.2e}"
                })

        # Get training metrics
        train_metrics = self.metrics.get_summary()

        # Validation
        val_metrics = {}
        if val_loader is not None and self.epoch % self.config.val_interval == 0:
            val_metrics = self.validate(val_loader)

            # Check for best model
            val_loss = val_metrics.get('total_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best.pth')

        # Update scheduler
        self.scheduler.step()

        # Combine metrics
        epoch_metrics = {
            'epoch': self.epoch,
            'lr': self.optimizer.param_groups[0]['lr'],
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }

        return epoch_metrics

    def fit(self, train_loader, val_loader=None):
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Train epoch
            start_time = time.time()
            epoch_metrics = self.train_epoch(train_loader, val_loader)
            epoch_time = time.time() - start_time

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"train_loss={epoch_metrics.get('train_total_loss', 0):.4f}, "
                f"val_loss={epoch_metrics.get('val_total_loss', 0):.4f}, "
                f"lr={epoch_metrics['lr']:.2e}, "
                f"time={epoch_time:.1f}s"
            )

            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')

        # Save final checkpoint
        self.save_checkpoint('final.pth')
        self.logger.info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_shadow': self.ema.shadow,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'metrics_history': self.metrics.history
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        # Load optimizer and scheduler
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load EMA
        if 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']

        # Load metrics history
        if 'metrics_history' in checkpoint:
            self.metrics.history = checkpoint['metrics_history']

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs
) -> optim.Optimizer:
    """Create optimizer with different parameter groups.

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    # Separate parameters for different components
    encoder_params = []
    decoder_params = []
    quantizer_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'encoder' in name:
            encoder_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        elif 'quantizer' in name or 'tokenizer' in name:
            quantizer_params.append(param)
        else:
            other_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': encoder_params, 'lr': learning_rate, 'weight_decay': weight_decay},
        {'params': decoder_params, 'lr': learning_rate, 'weight_decay': weight_decay},
        {'params': quantizer_params, 'lr': learning_rate * 0.5, 'weight_decay': weight_decay * 0.1},
        {'params': other_params, 'lr': learning_rate, 'weight_decay': weight_decay}
    ]

    # Filter empty groups
    param_groups = [group for group in param_groups if group['params']]

    if optimizer_type == 'adam':
        return optim.Adam(param_groups, **kwargs)
    elif optimizer_type == 'adamw':
        return optim.AdamW(param_groups, **kwargs)
    elif optimizer_type == 'sgd':
        return optim.SGD(param_groups, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")