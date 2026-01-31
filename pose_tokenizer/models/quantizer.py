"""Residual Vector Quantization (RVQ) for pose tokenizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import numpy as np
from einops import rearrange, repeat


class VectorQuantizer(nn.Module):
    """Vector Quantization layer."""

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        """Initialize vector quantizer.

        Args:
            codebook_size: Number of vectors in the codebook
            embedding_dim: Dimension of each vector
            commitment_cost: Weight for commitment loss
            decay: Decay rate for exponential moving average
            epsilon: Small constant for numerical stability
        """
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize codebook
        self.register_buffer('embedding', torch.randn(codebook_size, embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', self.embedding.clone())

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            inputs: Input tensor (..., embedding_dim)

        Returns:
            quantized: Quantized tensor
            indices: Codebook indices
            losses: Dictionary of losses
        """
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.t()))

        # Get closest codebook entries
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(indices.shape[0], self.codebook_size, device=inputs.device)
        encodings.scatter_(1, indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding).view(input_shape)

        # Update codebook with exponential moving average
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                torch.sum(encodings, 0), alpha=1 - self.decay)

            n = torch.sum(self.cluster_size.data)
            self.cluster_size.data.add_(self.epsilon).div_(n + self.codebook_size * self.epsilon).mul_(n)

            dw = torch.matmul(encodings.t(), flat_input)
            self.embed_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            self.embedding.data.copy_(self.embed_avg / self.cluster_size.unsqueeze(1))

        # Calculate losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()

        losses = {
            'vq_loss': q_latent_loss + commitment_loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': q_latent_loss
        }

        return quantized, indices.view(input_shape[:-1]), losses


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (RVQ)."""

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        shared_codebook: bool = False
    ):
        """Initialize RVQ.

        Args:
            num_quantizers: Number of quantization layers
            codebook_size: Size of each codebook
            embedding_dim: Dimension of embeddings
            commitment_cost: Weight for commitment loss
            decay: Decay rate for EMA
            shared_codebook: Whether to share codebook across layers
        """
        super().__init__()

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.shared_codebook = shared_codebook

        # Create quantizers
        if shared_codebook:
            # Single shared quantizer
            self.quantizer = VectorQuantizer(
                codebook_size=codebook_size,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay
            )
        else:
            # Separate quantizers for each layer
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    codebook_size=codebook_size,
                    embedding_dim=embedding_dim,
                    commitment_cost=commitment_cost,
                    decay=decay
                ) for _ in range(num_quantizers)
            ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (..., embedding_dim)

        Returns:
            quantized: Final quantized tensor
            indices: Codebook indices for each layer (num_quantizers, ...)
            losses: Dictionary of losses
        """
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices = []
        all_losses = {'vq_loss': 0, 'commitment_loss': 0, 'codebook_loss': 0}

        for i in range(self.num_quantizers):
            if self.shared_codebook:
                quantized, indices, losses = self.quantizer(residual)
            else:
                quantized, indices, losses = self.quantizers[i](residual)

            # Accumulate quantized output
            quantized_out = quantized_out + quantized

            # Update residual
            residual = residual - quantized

            # Store indices and losses
            all_indices.append(indices)
            for key in all_losses:
                all_losses[key] = all_losses[key] + losses[key]

        # Stack indices: (num_quantizers, ...)
        all_indices = torch.stack(all_indices, dim=0)

        return quantized_out, all_indices, all_losses

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to indices.

        Args:
            x: Input tensor (..., embedding_dim)

        Returns:
            indices: Codebook indices (num_quantizers, ...)
        """
        _, indices, _ = self.forward(x)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to vectors.

        Args:
            indices: Codebook indices (num_quantizers, ...)

        Returns:
            decoded: Decoded tensor (..., embedding_dim)
        """
        quantized_out = torch.zeros(
            *indices.shape[1:], self.embedding_dim,
            device=indices.device, dtype=torch.float32
        )

        for i in range(self.num_quantizers):
            layer_indices = indices[i]  # (...)

            if self.shared_codebook:
                # Use shared codebook
                flat_indices = layer_indices.view(-1)
                quantized = F.embedding(flat_indices, self.quantizer.embedding)
                quantized = quantized.view(*layer_indices.shape, self.embedding_dim)
            else:
                # Use layer-specific codebook
                flat_indices = layer_indices.view(-1)
                quantized = F.embedding(flat_indices, self.quantizers[i].embedding)
                quantized = quantized.view(*layer_indices.shape, self.embedding_dim)

            quantized_out = quantized_out + quantized

        return quantized_out

    def get_codebook_usage(self) -> Dict[str, torch.Tensor]:
        """Get codebook usage statistics.

        Returns:
            Dictionary with usage statistics for each quantizer
        """
        usage_stats = {}

        if self.shared_codebook:
            usage_stats['shared'] = self.quantizer.cluster_size.clone()
        else:
            for i, quantizer in enumerate(self.quantizers):
                usage_stats[f'layer_{i}'] = quantizer.cluster_size.clone()

        return usage_stats


class PoseTokenizer(nn.Module):
    """Complete pose tokenizer with encoder and RVQ."""

    def __init__(
        self,
        encoder_dim: int = 256,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        commitment_cost: float = 0.25,
        shared_codebook: bool = False
    ):
        """Initialize pose tokenizer.

        Args:
            encoder_dim: Dimension of encoder output
            num_quantizers: Number of RVQ layers
            codebook_size: Size of each codebook
            commitment_cost: Weight for commitment loss
            shared_codebook: Whether to share codebook across layers
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        # Pre-quantization projection
        self.pre_quant_proj = nn.Linear(encoder_dim, encoder_dim)

        # RVQ quantizer
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            embedding_dim=encoder_dim,
            commitment_cost=commitment_cost,
            shared_codebook=shared_codebook
        )

        # Post-quantization projection
        self.post_quant_proj = nn.Linear(encoder_dim, encoder_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Encoded features (..., encoder_dim)

        Returns:
            quantized: Quantized features
            indices: Codebook indices
            losses: Quantization losses
        """
        # Pre-quantization projection
        x = self.pre_quant_proj(x)

        # Quantize
        quantized, indices, losses = self.quantizer(x)

        # Post-quantization projection
        quantized = self.post_quant_proj(quantized)

        return quantized, indices, losses

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to discrete tokens.

        Args:
            x: Encoded features (..., encoder_dim)

        Returns:
            tokens: Discrete tokens (num_quantizers, ...)
        """
        x = self.pre_quant_proj(x)
        return self.quantizer.encode(x)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode from discrete tokens.

        Args:
            tokens: Discrete tokens (num_quantizers, ...)

        Returns:
            decoded: Decoded features (..., encoder_dim)
        """
        quantized = self.quantizer.decode(tokens)
        return self.post_quant_proj(quantized)

    def get_codebook_usage(self) -> Dict[str, torch.Tensor]:
        """Get codebook usage statistics."""
        return self.quantizer.get_codebook_usage()


class GroupedResidualVectorQuantizer(nn.Module):
    """Grouped RVQ for handling different pose components separately."""

    def __init__(
        self,
        num_groups: int = 3,  # e.g., pose, left_hand, right_hand
        group_dims: List[int] = [256, 128, 128],
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        commitment_cost: float = 0.25
    ):
        """Initialize grouped RVQ.

        Args:
            num_groups: Number of groups (pose components)
            group_dims: Dimension for each group
            num_quantizers: Number of quantization layers per group
            codebook_size: Size of each codebook
            commitment_cost: Weight for commitment loss
        """
        super().__init__()

        self.num_groups = num_groups
        self.group_dims = group_dims
        self.num_quantizers = num_quantizers

        # Create RVQ for each group
        self.group_quantizers = nn.ModuleList([
            ResidualVectorQuantizer(
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                embedding_dim=dim,
                commitment_cost=commitment_cost,
                shared_codebook=False
            ) for dim in group_dims
        ])

        # Group projections
        total_dim = sum(group_dims)
        self.group_projections = nn.ModuleList([
            nn.Linear(total_dim, dim) for dim in group_dims
        ])

        # Reconstruction projection
        self.recon_projection = nn.Linear(sum(group_dims), total_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input features (..., total_dim)

        Returns:
            quantized: Reconstructed features
            indices_list: List of indices for each group
            losses: Combined losses
        """
        # Project to groups
        group_features = []
        for proj in self.group_projections:
            group_features.append(proj(x))

        # Quantize each group
        quantized_groups = []
        indices_list = []
        total_losses = {'vq_loss': 0, 'commitment_loss': 0, 'codebook_loss': 0}

        for i, (features, quantizer) in enumerate(zip(group_features, self.group_quantizers)):
            quantized, indices, losses = quantizer(features)
            quantized_groups.append(quantized)
            indices_list.append(indices)

            # Accumulate losses
            for key in total_losses:
                total_losses[key] = total_losses[key] + losses[key]

        # Concatenate and reconstruct
        concatenated = torch.cat(quantized_groups, dim=-1)
        reconstructed = self.recon_projection(concatenated)

        return reconstructed, indices_list, total_losses

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode to discrete tokens for each group."""
        group_features = []
        for proj in self.group_projections:
            group_features.append(proj(x))

        indices_list = []
        for features, quantizer in zip(group_features, self.group_quantizers):
            indices = quantizer.encode(features)
            indices_list.append(indices)

        return indices_list

    def decode(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """Decode from discrete tokens."""
        quantized_groups = []
        for indices, quantizer in zip(indices_list, self.group_quantizers):
            quantized = quantizer.decode(indices)
            quantized_groups.append(quantized)

        # Concatenate and reconstruct
        concatenated = torch.cat(quantized_groups, dim=-1)
        reconstructed = self.recon_projection(concatenated)

        return reconstructed