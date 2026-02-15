"""Training utilities for pose tokenizer."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
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

from pose_tokenizer.models.losses import PoseTokenizerLoss


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
    w_pos: float = 1.0
    w_vel: float = 0.5
    w_acc: float = 0.25
    w_bone: float = 0.5
    w_vq: float = 1.0

    # Skeleton definition (optional path)
    skeleton_def: str = ""

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

    def __post_init__(self):
        """Normalize common numeric fields that may come from YAML strings."""
        int_fields = {
            "num_keypoints",
            "keypoint_dim",
            "hidden_dim",
            "num_quantizers",
            "codebook_size",
            "batch_size",
            "num_epochs",
            "warmup_epochs",
            "step_size",
            "val_interval",
            "log_interval",
            "save_interval",
            "local_rank",
            "world_size",
        }
        float_fields = {
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
        bool_fields = {"use_amp", "distributed"}

        for name in int_fields:
            value = getattr(self, name, None)
            if isinstance(value, str):
                setattr(self, name, int(float(value)))

        for name in float_fields:
            value = getattr(self, name, None)
            if isinstance(value, str):
                setattr(self, name, float(value))

        for name in bool_fields:
            value = getattr(self, name, None)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "y", "on"}:
                    setattr(self, name, True)
                elif lowered in {"0", "false", "no", "n", "off"}:
                    setattr(self, name, False)


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
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = float(value.item())
            elif isinstance(value, bool):
                value = float(value)
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                continue
            if not math.isfinite(value):
                continue

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
        logger: logging.Logger = None,
        tb_writer: Any = None,
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
        self.is_distributed = bool(config.distributed and config.world_size > 1)
        self.tb_writer = tb_writer

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Load skeleton definition if provided
        bones = None
        if getattr(config, "skeleton_def", ""):
            try:
                with open(config.skeleton_def, "r") as f:
                    skeleton = json.load(f)
                bones = skeleton.get("bones") or skeleton.get("edges")
            except Exception:
                bones = None

        # Initialize loss function
        self.criterion = PoseTokenizerLoss(
            reconstruction_weight=config.reconstruction_weight,
            quantization_weight=config.quantization_weight,
            perceptual_weight=config.perceptual_weight,
            temporal_weight=config.temporal_weight,
            adversarial_weight=config.adversarial_weight,
            w_pos=config.w_pos,
            w_vel=config.w_vel,
            w_acc=config.w_acc,
            w_bone=config.w_bone,
            w_vq=config.w_vq,
            bones=bones,
            use_temporal=False,
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

    def _is_main_process(self) -> bool:
        """Return True for the process allowed to write checkpoints/log summaries."""
        if not self.is_distributed:
            return True
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Average scalar metrics across distributed workers."""
        if not self.is_distributed:
            return metrics
        if not dist.is_available() or not dist.is_initialized():
            return metrics
        if not metrics:
            return metrics

        keys = sorted(metrics.keys())
        values = torch.tensor(
            [float(metrics[k]) for k in keys],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values = values / float(dist.get_world_size())
        return {k: float(v) for k, v in zip(keys, values.tolist())}

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        lr = float(self.config.learning_rate)
        weight_decay = float(self.config.weight_decay)

        if self.config.optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=weight_decay
            )
        elif self.config.optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=weight_decay,
            )
        elif self.config.optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

    def _create_scheduler(self) -> _LRScheduler:
        """Create learning rate scheduler."""
        t_max = int(self.config.num_epochs)
        eta_min = float(self.config.min_lr)
        step_size = int(self.config.step_size)
        gamma = float(self.config.gamma)

        if self.config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min
            )
        elif self.config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif self.config.scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
        elif self.config.scheduler_type == 'warmup_cosine':
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=int(self.config.warmup_epochs),
                total_epochs=t_max,
                min_lr=eta_min
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def _core_model(self) -> nn.Module:
        """Return the underlying model for direct component access."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    @staticmethod
    def _has_module_prefix(state_dict: Dict[str, torch.Tensor]) -> bool:
        if not state_dict:
            return False
        first_key = next(iter(state_dict.keys()))
        return first_key.startswith("module.")

    def _align_state_dict_for_current_model(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Match checkpoint key-prefix style (module./non-module.) to current model."""
        target_state = self.model.state_dict()
        target_has_module = self._has_module_prefix(target_state)
        src_has_module = self._has_module_prefix(state_dict)

        if src_has_module == target_has_module:
            return state_dict
        if src_has_module and not target_has_module:
            return {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
        return {f"module.{k}": v for k, v in state_dict.items()}

    def _tb_add_scalar(self, tag: str, value: Any, step: int) -> None:
        """Safely add a scalar to TensorBoard."""
        if self.tb_writer is None:
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
            value = float(value.item())
        elif isinstance(value, bool):
            value = float(value)
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            return

        if not math.isfinite(value):
            return
        self.tb_writer.add_scalar(tag, value, step)

    @staticmethod
    def _sanitize_tensor(x: torch.Tensor, clamp: float = 1e4) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp)
        return torch.clamp(x, min=-clamp, max=clamp)

    @staticmethod
    def _sanitize_quant_losses(losses: Dict[str, Any]) -> Dict[str, Any]:
        if losses is None:
            return {}
        clean = {}
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    if torch.isfinite(value):
                        clean[key] = value
                    else:
                        clean[key] = torch.zeros_like(value)
                else:
                    clean[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                clean[key] = value
        return clean

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
        core_model = self._core_model()
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                encoded = core_model.encoder(poses, mask)
            with torch.cuda.amp.autocast(enabled=False):
                encoded = self._sanitize_tensor(encoded.float())
                quantized, indices, quant_losses = core_model.tokenizer(encoded)
                quantized = self._sanitize_tensor(quantized.float())
                reconstructed = core_model.decoder(quantized, poses.shape[1:])
                reconstructed = self._sanitize_tensor(reconstructed.float())
                quant_losses = self._sanitize_quant_losses(quant_losses)
                losses = self.criterion(
                    reconstructed, poses.float(), quant_losses, mask
                )
        else:
            # Encode
            encoded = core_model.encoder(poses, mask)
            encoded = self._sanitize_tensor(encoded.float())

            # Quantize
            quantized, indices, quant_losses = core_model.tokenizer(encoded)
            quantized = self._sanitize_tensor(quantized.float())
            quant_losses = self._sanitize_quant_losses(quant_losses)

            # Decode
            reconstructed = core_model.decoder(quantized, poses.shape[1:])
            reconstructed = self._sanitize_tensor(reconstructed.float())

            # Compute losses
            losses = self.criterion(
                reconstructed, poses.float(), quant_losses, mask
            )

        total_loss = losses.get("total_loss")
        if isinstance(total_loss, torch.Tensor) and not torch.isfinite(total_loss):
            self.optimizer.zero_grad(set_to_none=True)
            if self._is_main_process():
                bad_keys = []
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1 and not torch.isfinite(value):
                        bad_keys.append(key)
                self.logger.warning(
                    "Skip non-finite batch at epoch=%d step=%d total_loss=%s bad_keys=%s pose_abs_max=%.4f enc_abs_max=%.4f q_abs_max=%.4f rec_abs_max=%.4f",
                    self.epoch,
                    self.global_step + 1,
                    str(total_loss.detach().float().cpu().item()) if total_loss.numel() == 1 else "non-scalar",
                    bad_keys,
                    float(poses.detach().abs().max().item()),
                    float(encoded.detach().abs().max().item()),
                    float(quantized.detach().abs().max().item()),
                    float(reconstructed.detach().abs().max().item()),
                )
            loss_dict = {"total_loss": 0.0}
            for key in ("pos_loss", "vel_loss", "acc_loss", "bone_loss", "quant_vq_loss"):
                value = losses.get(key, 0.0)
                if isinstance(value, torch.Tensor) and value.numel() == 1 and torch.isfinite(value):
                    loss_dict[key] = float(value.item())
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    loss_dict[key] = float(value)
                else:
                    loss_dict[key] = 0.0
            loss_dict["skipped_non_finite"] = 1.0
            self._tb_add_scalar("train_step/skipped_non_finite", 1.0, self.global_step + 1)
            return loss_dict

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
            core_model = self._core_model()
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                # Move batch to device
                poses = batch['poses'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)

                # Forward pass
                encoded = core_model.encoder(poses, mask)
                encoded = self._sanitize_tensor(encoded.float())
                quantized, indices, quant_losses = core_model.tokenizer(encoded)
                quantized = self._sanitize_tensor(quantized.float())
                reconstructed = core_model.decoder(quantized, poses.shape[1:])
                reconstructed = self._sanitize_tensor(reconstructed.float())
                quant_losses = self._sanitize_quant_losses(quant_losses)

                # Compute losses
                losses = self.criterion(
                    reconstructed, poses.float(), quant_losses, mask
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
        log_allowed = getattr(self.config, "local_rank", 0) == 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch}",
            disable=not log_allowed,
        )
        start_time = time.time()
        last_log_time = start_time
        steps_per_epoch = len(train_loader)
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
                    'pos': f"{losses.get('pos_loss', 0):.4f}",
                    'vel': f"{losses.get('vel_loss', 0):.4f}",
                    'bone': f"{losses.get('bone_loss', 0):.4f}",
                    'vq': f"{losses.get('quant_vq_loss', 0):.4f}",
                    'lr': f"{current_lr:.2e}"
                })

                perplexity = losses.get('quant_perplexity', None)
                if log_allowed and perplexity is not None and perplexity < 1.2:
                    self.logger.warning(
                        f"Codebook perplexity is low ({perplexity:.2f}); possible collapse risk."
                    )

                if log_allowed:
                    now = time.time()
                    iter_time = (now - last_log_time) / max(self.config.log_interval, 1)
                    elapsed = now - start_time
                    eta = (steps_per_epoch - batch_idx - 1) * iter_time
                    self.logger.info(
                        "Epoch %d [%d/%d] step=%d loss=%.4f pos=%.4f vel=%.4f acc=%.4f bone=%.4f vq=%.4f lr=%.2e "
                        "iter=%.3fs elapsed=%.1fs eta=%.1fs",
                        self.epoch,
                        batch_idx + 1,
                        steps_per_epoch,
                        self.global_step,
                        losses.get('total_loss', 0.0),
                        losses.get('pos_loss', 0.0),
                        losses.get('vel_loss', 0.0),
                        losses.get('acc_loss', 0.0),
                        losses.get('bone_loss', 0.0),
                        losses.get('quant_vq_loss', 0.0),
                        current_lr,
                        iter_time,
                        elapsed,
                        eta,
                    )
                    last_log_time = now
                    self._tb_add_scalar("train_step/lr", current_lr, self.global_step)
                    for key, value in losses.items():
                        if key == "total_loss":
                            self._tb_add_scalar("train_step/loss_total", value, self.global_step)
                        else:
                            self._tb_add_scalar(f"train_step/{key}", value, self.global_step)

        # Get training metrics
        train_metrics = self.metrics.get_summary()

        # Validation
        val_metrics = {}
        is_best = False
        if val_loader is not None and self.epoch % self.config.val_interval == 0:
            val_metrics = self.validate(val_loader)
            val_metrics = self._sync_metrics(val_metrics)

            # Check for best model
            val_loss = val_metrics.get('total_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True

        # Update scheduler
        self.scheduler.step()

        # Save best checkpoint after scheduler step so optimizer/scheduler states are consistent.
        if is_best and self._is_main_process():
            self.save_checkpoint('best.pth')

        # Combine metrics
        epoch_metrics = {
            'epoch': self.epoch,
            'lr': self.optimizer.param_groups[0]['lr'],
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        for key, value in epoch_metrics.items():
            if key == "epoch":
                continue
            self._tb_add_scalar(f"epoch/{key}", value, self.epoch)

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
            if epoch % self.config.save_interval == 0 and self._is_main_process():
                self.save_checkpoint(f'epoch_{epoch}.pth')

        # Save final checkpoint
        if self._is_main_process():
            self.save_checkpoint('final.pth')
        self.logger.info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if not self._is_main_process():
            return

        checkpoint = {
            'epoch': self.epoch,
            'next_epoch': self.epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self._core_model().state_dict(),
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
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            raw_state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            # Compatibility: allow loading raw state_dict checkpoints.
            raw_state_dict = checkpoint
            checkpoint = {}
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

        # Load model state (DDP/non-DDP compatible).
        state_dict = self._align_state_dict_for_current_model(raw_state_dict)
        self.model.load_state_dict(state_dict, strict=True)

        loaded_epoch = int(checkpoint.get('epoch', -1))
        if load_optimizer:
            self.epoch = int(checkpoint.get('next_epoch', loaded_epoch + 1))
        else:
            self.epoch = loaded_epoch
        self.global_step = int(checkpoint.get('global_step', 0))
        self.best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))

        # Load optimizer and scheduler states if present.
        if load_optimizer:
            opt_state = checkpoint.get('optimizer_state_dict', None)
            sch_state = checkpoint.get('scheduler_state_dict', None)
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
            if sch_state is not None:
                self.scheduler.load_state_dict(sch_state)

                # If checkpoint was saved before epoch-end scheduler.step (e.g., best.pth),
                # avoid resuming with a stale scheduler epoch.
                if getattr(self.scheduler, "last_epoch", -1) < loaded_epoch:
                    self.scheduler.last_epoch = loaded_epoch

            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load EMA (align key style with current model keys when possible).
        if 'ema_shadow' in checkpoint and isinstance(checkpoint['ema_shadow'], dict):
            ema_shadow = checkpoint['ema_shadow']
            target_has_module = self._has_module_prefix(dict(self.model.named_parameters()))
            src_has_module = self._has_module_prefix(ema_shadow)
            if src_has_module != target_has_module:
                if src_has_module and not target_has_module:
                    ema_shadow = {
                        (k[7:] if k.startswith("module.") else k): v
                        for k, v in ema_shadow.items()
                    }
                else:
                    ema_shadow = {f"module.{k}": v for k, v in ema_shadow.items()}
            self.ema.shadow = ema_shadow

        # Load metrics history
        if 'metrics_history' in checkpoint and isinstance(checkpoint['metrics_history'], dict):
            self.metrics.history = checkpoint['metrics_history']

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(
            "Resumed at epoch=%d (loaded_epoch=%d), step=%d",
            self.epoch,
            loaded_epoch,
            self.global_step,
        )


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
