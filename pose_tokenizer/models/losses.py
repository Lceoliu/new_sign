"""Loss functions for pose tokenizer training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for pose sequences."""

    def __init__(
        self,
        loss_type: str = 'mse',
        keypoint_weights: Optional[torch.Tensor] = None,
        confidence_weight: float = 1.0,
        position_weight: float = 1.0
    ):
        """Initialize reconstruction loss.

        Args:
            loss_type: Type of loss ('mse', 'l1', 'smooth_l1', 'huber')
            keypoint_weights: Per-keypoint weights (N,)
            confidence_weight: Weight for confidence loss
            position_weight: Weight for position loss
        """
        super().__init__()

        self.loss_type = loss_type
        self.confidence_weight = confidence_weight
        self.position_weight = position_weight

        if keypoint_weights is not None:
            self.register_buffer('keypoint_weights', keypoint_weights)
        else:
            self.keypoint_weights = None

        # Select loss function
        if loss_type == 'mse':
            self.loss_fn = F.mse_loss
        elif loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'smooth_l1':
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == 'huber':
            self.loss_fn = F.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pred: Predicted poses (B, T, N, D)
            target: Target poses (B, T, N, D)
            mask: Valid frame mask (B, T)

        Returns:
            Dictionary of losses
        """
        B, T, N, D = pred.shape

        # Split position and confidence
        if D == 3:
            pred_pos = pred[..., :2]  # (B, T, N, 2)
            pred_conf = pred[..., 2]  # (B, T, N)
            target_pos = target[..., :2]
            target_conf = target[..., 2]
        else:
            pred_pos = pred
            pred_conf = None
            target_pos = target
            target_conf = None

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            pred_pos = pred_pos * mask_expanded
            target_pos = target_pos * mask_expanded

            if pred_conf is not None:
                mask_kp = mask.unsqueeze(-1)  # (B, T, 1)
                pred_conf = pred_conf * mask_kp
                target_conf = target_conf * mask_kp

        # Position loss
        pos_loss = self.loss_fn(pred_pos, target_pos, reduction='none')  # (B, T, N, 2)

        # Apply keypoint weights if provided
        if self.keypoint_weights is not None:
            weights = self.keypoint_weights.view(1, 1, N, 1)  # (1, 1, N, 1)
            pos_loss = pos_loss * weights

        # Reduce position loss
        pos_loss = pos_loss.mean()

        losses = {'position_loss': pos_loss * self.position_weight}

        # Confidence loss
        if pred_conf is not None and target_conf is not None:
            conf_loss = self.loss_fn(pred_conf, target_conf, reduction='mean')
            losses['confidence_loss'] = conf_loss * self.confidence_weight

        # Total reconstruction loss
        total_loss = sum(losses.values())
        losses['reconstruction_loss'] = total_loss

        return losses


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained pose features."""

    def __init__(
        self,
        feature_layers: List[str] = ['layer_2', 'layer_4'],
        weights: List[float] = [1.0, 1.0]
    ):
        """Initialize perceptual loss.

        Args:
            feature_layers: List of layer names to extract features from
            weights: Weights for each feature layer
        """
        super().__init__()

        self.feature_layers = feature_layers
        self.weights = weights

        # Note: In practice, you would load a pre-trained pose model here
        # For now, we'll use a simple feature extractor
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self):
        """Build a simple feature extractor."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: Predicted poses (B, T, N, D)
            target: Target poses (B, T, N, D)

        Returns:
            Perceptual loss
        """
        # For simplicity, we'll compute a basic perceptual loss
        # In practice, you would extract features from multiple layers
        B, T, N, D = pred.shape

        # Reshape for feature extraction: (B*T, D, N, 1)
        pred_reshaped = pred.view(B*T, D, N, 1)
        target_reshaped = target.view(B*T, D, N, 1)

        # Extract features
        pred_features = self.feature_extractor(pred_reshaped)
        target_features = self.feature_extractor(target_reshaped)

        # Compute perceptual loss
        perceptual_loss = F.mse_loss(pred_features, target_features)

        return perceptual_loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for smooth motion."""

    def __init__(
        self,
        loss_type: str = 'mse',
        order: int = 1
    ):
        """Initialize temporal consistency loss.

        Args:
            loss_type: Type of loss ('mse', 'l1')
            order: Order of temporal difference (1 for velocity, 2 for acceleration)
        """
        super().__init__()

        self.loss_type = loss_type
        self.order = order

        if loss_type == 'mse':
            self.loss_fn = F.mse_loss
        elif loss_type == 'l1':
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _compute_temporal_diff(self, x: torch.Tensor, order: int) -> torch.Tensor:
        """Compute temporal differences.

        Args:
            x: Input tensor (B, T, ...)
            order: Order of difference

        Returns:
            Temporal differences
        """
        diff = x
        for _ in range(order):
            diff = diff[:, 1:] - diff[:, :-1]
        return diff

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: Predicted poses (B, T, N, D)
            target: Target poses (B, T, N, D)
            mask: Valid frame mask (B, T)

        Returns:
            Temporal consistency loss
        """
        # Compute temporal differences
        pred_diff = self._compute_temporal_diff(pred, self.order)
        target_diff = self._compute_temporal_diff(target, self.order)

        # Apply mask if provided
        if mask is not None:
            # Adjust mask for temporal differences
            diff_mask = mask
            for _ in range(self.order):
                diff_mask = diff_mask[:, 1:] & diff_mask[:, :-1]

            # Expand mask
            diff_mask = diff_mask.unsqueeze(-1).unsqueeze(-1)
            pred_diff = pred_diff * diff_mask
            target_diff = target_diff * diff_mask

        # Compute loss
        temporal_loss = self.loss_fn(pred_diff, target_diff)

        return temporal_loss


class AdversarialLoss(nn.Module):
    """Adversarial loss for realistic pose generation."""

    def __init__(
        self,
        discriminator: nn.Module,
        loss_type: str = 'hinge'
    ):
        """Initialize adversarial loss.

        Args:
            discriminator: Discriminator network
            loss_type: Type of adversarial loss ('hinge', 'bce', 'wgan')
        """
        super().__init__()

        self.discriminator = discriminator
        self.loss_type = loss_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mode: str = 'generator'
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: Predicted poses (B, T, N, D)
            target: Target poses (B, T, N, D)
            mode: 'generator' or 'discriminator'

        Returns:
            Adversarial loss
        """
        if mode == 'generator':
            # Generator loss: fool the discriminator
            fake_logits = self.discriminator(pred)

            if self.loss_type == 'hinge':
                gen_loss = -fake_logits.mean()
            elif self.loss_type == 'bce':
                gen_loss = F.binary_cross_entropy_with_logits(
                    fake_logits, torch.ones_like(fake_logits)
                )
            elif self.loss_type == 'wgan':
                gen_loss = -fake_logits.mean()
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            return gen_loss

        elif mode == 'discriminator':
            # Discriminator loss: distinguish real from fake
            real_logits = self.discriminator(target.detach())
            fake_logits = self.discriminator(pred.detach())

            if self.loss_type == 'hinge':
                real_loss = F.relu(1.0 - real_logits).mean()
                fake_loss = F.relu(1.0 + fake_logits).mean()
                disc_loss = real_loss + fake_loss
            elif self.loss_type == 'bce':
                real_loss = F.binary_cross_entropy_with_logits(
                    real_logits, torch.ones_like(real_logits)
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_logits, torch.zeros_like(fake_logits)
                )
                disc_loss = real_loss + fake_loss
            elif self.loss_type == 'wgan':
                disc_loss = fake_logits.mean() - real_logits.mean()
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            return disc_loss

        else:
            raise ValueError(f"Unknown mode: {mode}")


class PoseTokenizerLoss(nn.Module):
    """Combined loss for pose tokenizer training."""

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        quantization_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        temporal_weight: float = 0.1,
        adversarial_weight: float = 0.01,
        use_perceptual: bool = False,
        use_temporal: bool = True,
        use_adversarial: bool = False,
        **kwargs
    ):
        """Initialize combined loss.

        Args:
            reconstruction_weight: Weight for reconstruction loss
            quantization_weight: Weight for quantization loss
            perceptual_weight: Weight for perceptual loss
            temporal_weight: Weight for temporal consistency loss
            adversarial_weight: Weight for adversarial loss
            use_perceptual: Whether to use perceptual loss
            use_temporal: Whether to use temporal consistency loss
            use_adversarial: Whether to use adversarial loss
            **kwargs: Additional arguments for loss components
        """
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.quantization_weight = quantization_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
        self.adversarial_weight = adversarial_weight

        # Reconstruction loss
        self.reconstruction_loss = ReconstructionLoss(**kwargs)

        # Optional losses
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

        if use_temporal:
            self.temporal_loss = TemporalConsistencyLoss()
        else:
            self.temporal_loss = None

        if use_adversarial:
            # Note: discriminator should be passed in kwargs
            discriminator = kwargs.get('discriminator')
            if discriminator is not None:
                self.adversarial_loss = AdversarialLoss(discriminator)
            else:
                self.adversarial_loss = None
        else:
            self.adversarial_loss = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        quantization_losses: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        mode: str = 'generator'
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pred: Predicted poses (B, T, N, D)
            target: Target poses (B, T, N, D)
            quantization_losses: Losses from quantization
            mask: Valid frame mask (B, T)
            mode: 'generator' or 'discriminator' (for adversarial training)

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Reconstruction loss
        recon_losses = self.reconstruction_loss(pred, target, mask)
        for key, value in recon_losses.items():
            losses[key] = value * self.reconstruction_weight

        # Quantization losses
        for key, value in quantization_losses.items():
            losses[f'quant_{key}'] = value * self.quantization_weight

        # Perceptual loss
        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred, target)
            losses['perceptual_loss'] = perceptual * self.perceptual_weight

        # Temporal consistency loss
        if self.temporal_loss is not None:
            temporal = self.temporal_loss(pred, target, mask)
            losses['temporal_loss'] = temporal * self.temporal_weight

        # Adversarial loss
        if self.adversarial_loss is not None:
            adversarial = self.adversarial_loss(pred, target, mode)
            if mode == 'generator':
                losses['adversarial_gen_loss'] = adversarial * self.adversarial_weight
            else:
                losses['adversarial_disc_loss'] = adversarial

        # Total loss
        if mode == 'generator':
            total_loss = sum(v for k, v in losses.items() if not k.startswith('adversarial_disc'))
        else:
            total_loss = losses.get('adversarial_disc_loss', torch.tensor(0.0))

        losses['total_loss'] = total_loss

        return losses


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in pose classification."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """Initialize focal loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Predicted logits (B, C)
            targets: Target labels (B,)

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning pose representations."""

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0
    ):
        """Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for softmax
            margin: Margin for negative pairs
        """
        super().__init__()

        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Feature representations (B, D)
            labels: Labels for contrastive learning (B,)

        Returns:
            Contrastive loss
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1 - positive_mask

        # Remove diagonal (self-similarity)
        mask = torch.eye(features.size(0), device=features.device)
        positive_mask = positive_mask * (1 - mask)
        negative_mask = negative_mask * (1 - mask)

        # Compute contrastive loss
        exp_sim = torch.exp(similarity)
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        negative_sum = (exp_sim * negative_mask).sum(dim=1)

        loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))
        loss = loss[positive_mask.sum(dim=1) > 0]  # Only consider samples with positives

        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=features.device)