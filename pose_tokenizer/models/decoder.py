"""Decoder module for pose tokenizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import sys
import os

# Add the parent directory to path to import existing ST-GCN modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../stgcn_layers'))
from stgcn_block import STGCN_block

from .encoder import MediaPipeGraph


class STGCNDecoder(nn.Module):
    """ST-GCN decoder for pose reconstruction."""

    def __init__(
        self,
        num_keypoints: int = 133,
        keypoint_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        kernel_size: Tuple[int, int] = (3, 3),
        dropout: float = 0.1,
        adaptive: bool = True
    ):
        """Initialize ST-GCN decoder.

        Args:
            num_keypoints: Number of keypoints per frame
            keypoint_dim: Dimension of each keypoint (x, y, confidence)
            hidden_dim: Hidden dimension size
            num_layers: Number of ST-GCN layers
            kernel_size: Temporal and spatial kernel sizes
            dropout: Dropout rate
            adaptive: Whether to use adaptive adjacency matrix
        """
        super().__init__()

        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create pose graph
        self.graph = MediaPipeGraph(num_keypoints)
        self.A = self.graph.A

        # Input projection from hidden_dim to initial decoder dimension
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # ST-GCN layers (reverse of encoder)
        self.st_gcn_layers = nn.ModuleList()

        # Layer configurations: (in_channels, out_channels, stride)
        # Reverse order of encoder
        layer_configs = [
            (hidden_dim, 256, 1),
            (256, 256, 1),
            (256, 128, 1),
            (128, 128, 1),
            (128, 64, 1),
            (64, 64, 1)
        ]

        for i in range(min(num_layers, len(layer_configs))):
            in_ch, out_ch, stride = layer_configs[i]

            self.st_gcn_layers.append(
                STGCN_block(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    A=self.A.clone(),
                    adaptive=adaptive,
                    stride=stride,
                    dropout=dropout,
                    residual=True
                )
            )

        # Output projection to keypoint coordinates
        self.output_proj = nn.Conv2d(64, keypoint_dim, kernel_size=1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        target_shape: Tuple[int, int, int] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, hidden_dim) or (B, T, hidden_dim)
            target_shape: Target output shape (T, N, D)

        Returns:
            Reconstructed poses (B, T, N, D)
        """
        B = x.shape[0]

        # Handle different input shapes
        if len(x.shape) == 2:
            # Global features (B, hidden_dim) -> expand to sequence
            if target_shape is None:
                T, N, D = 64, self.num_keypoints, self.keypoint_dim
            else:
                T, N, D = target_shape

            # Expand to sequence: (B, hidden_dim) -> (B, T, hidden_dim)
            x = x.unsqueeze(1).expand(-1, T, -1)
        else:
            # Sequence features (B, T, hidden_dim)
            T = x.shape[1]
            N, D = self.num_keypoints, self.keypoint_dim

        # Input projection
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Expand to spatial dimensions: (B, T, hidden_dim) -> (B, hidden_dim, T, N)
        x = x.unsqueeze(-1).expand(-1, -1, -1, N)  # (B, T, hidden_dim, N)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, hidden_dim, T, N)

        # ST-GCN layers
        for layer in self.st_gcn_layers:
            x = layer(x)

        # Output projection
        x = self.output_proj(x)  # (B, keypoint_dim, T, N)

        # Reshape to output format: (B, T, N, D)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x


class TemporalDecoder(nn.Module):
    """Temporal decoder using 1D transposed convolutions."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """Initialize temporal decoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of temporal layers
            kernel_size: Temporal kernel size
            dropout: Dropout rate
        """
        super().__init__()

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            nn.Sequential(
                nn.ConvTranspose1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            )

        # Final layer
        if num_layers > 1:
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(input_dim),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, T, D)

        Returns:
            Decoded features (B, T, D)
        """
        # Transpose for 1D conv: (B, D, T)
        x = x.transpose(1, 2)

        # Apply layers
        for layer in self.layers:
            residual = x
            x = layer(x)

            # Residual connection (if dimensions match)
            if x.shape == residual.shape:
                x = x + residual

        # Transpose back: (B, T, D)
        x = x.transpose(1, 2)

        return x


class PoseDecoder(nn.Module):
    """Complete pose decoder combining temporal and spatial decoding."""

    def __init__(
        self,
        num_keypoints: int = 133,
        keypoint_dim: int = 3,
        hidden_dim: int = 256,
        spatial_layers: int = 6,
        temporal_layers: int = 4,
        dropout: float = 0.1,
        use_temporal: bool = True
    ):
        """Initialize pose decoder.

        Args:
            num_keypoints: Number of keypoints per frame
            keypoint_dim: Dimension of each keypoint
            hidden_dim: Hidden dimension
            spatial_layers: Number of spatial ST-GCN layers
            temporal_layers: Number of temporal layers
            dropout: Dropout rate
            use_temporal: Whether to use temporal decoding
        """
        super().__init__()

        self.use_temporal = use_temporal

        # Temporal decoder (optional)
        if use_temporal:
            self.temporal_decoder = TemporalDecoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=temporal_layers,
                dropout=dropout
            )

        # Spatial decoder (ST-GCN)
        self.spatial_decoder = STGCNDecoder(
            num_keypoints=num_keypoints,
            keypoint_dim=keypoint_dim,
            hidden_dim=hidden_dim,
            num_layers=spatial_layers,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        target_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, hidden_dim) or (B, T, hidden_dim)
            target_shape: Target output shape (T, N, D)

        Returns:
            Reconstructed poses (B, T, N, D)
        """
        if self.use_temporal and len(x.shape) == 3:
            # Temporal decoding for sequence features
            x = self.temporal_decoder(x)  # (B, T, hidden_dim)

        # Spatial decoding
        poses = self.spatial_decoder(x, target_shape)  # (B, T, N, D)

        return poses


class MultiScaleDecoder(nn.Module):
    """Multi-scale decoder for hierarchical pose reconstruction."""

    def __init__(
        self,
        num_keypoints: int = 133,
        keypoint_dim: int = 3,
        hidden_dim: int = 256,
        scales: List[int] = [1, 2, 4],
        dropout: float = 0.1
    ):
        """Initialize multi-scale decoder.

        Args:
            num_keypoints: Number of keypoints per frame
            keypoint_dim: Dimension of each keypoint
            hidden_dim: Hidden dimension
            scales: List of temporal scales
            dropout: Dropout rate
        """
        super().__init__()

        self.scales = scales
        self.num_scales = len(scales)

        # Create decoders for each scale
        self.scale_decoders = nn.ModuleList([
            PoseDecoder(
                num_keypoints=num_keypoints,
                keypoint_dim=keypoint_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_temporal=True
            ) for _ in scales
        ])

        # Scale fusion
        self.scale_fusion = nn.Conv1d(
            self.num_scales * keypoint_dim,
            keypoint_dim,
            kernel_size=1
        )

    def forward(
        self,
        x: torch.Tensor,
        target_length: int = 64
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, hidden_dim) or (B, T, hidden_dim)
            target_length: Target sequence length

        Returns:
            Multi-scale reconstructed poses (B, T, N, D)
        """
        B = x.shape[0]
        scale_outputs = []

        for i, (scale, decoder) in enumerate(zip(self.scales, self.scale_decoders)):
            # Determine target shape for this scale
            scale_length = target_length // scale
            target_shape = (scale_length, decoder.spatial_decoder.num_keypoints,
                          decoder.spatial_decoder.keypoint_dim)

            # Decode at this scale
            scale_output = decoder(x, target_shape)  # (B, scale_T, N, D)

            # Upsample to target length if needed
            if scale_length != target_length:
                # Reshape for interpolation: (B, N*D, scale_T)
                B, scale_T, N, D = scale_output.shape
                scale_output = scale_output.view(B, scale_T, -1).transpose(1, 2)

                # Interpolate
                scale_output = F.interpolate(
                    scale_output, size=target_length, mode='linear', align_corners=False
                )

                # Reshape back: (B, T, N, D)
                scale_output = scale_output.transpose(1, 2).view(B, target_length, N, D)

            scale_outputs.append(scale_output)

        # Fuse scales
        # Concatenate along feature dimension: (B, T, N, num_scales * D)
        fused = torch.cat(scale_outputs, dim=-1)

        # Reshape for 1D conv: (B, T*N, num_scales * D) -> (B, num_scales * D, T*N)
        B, T, N, multi_D = fused.shape
        fused = fused.view(B, T*N, multi_D).transpose(1, 2)

        # Apply fusion conv
        fused = self.scale_fusion(fused)  # (B, D, T*N)

        # Reshape back: (B, T, N, D)
        fused = fused.transpose(1, 2).view(B, T, N, -1)

        return fused


class AdaptiveDecoder(nn.Module):
    """Adaptive decoder that adjusts based on input sequence length."""

    def __init__(
        self,
        num_keypoints: int = 133,
        keypoint_dim: int = 3,
        hidden_dim: int = 256,
        min_layers: int = 3,
        max_layers: int = 8,
        dropout: float = 0.1
    ):
        """Initialize adaptive decoder.

        Args:
            num_keypoints: Number of keypoints per frame
            keypoint_dim: Dimension of each keypoint
            hidden_dim: Hidden dimension
            min_layers: Minimum number of layers
            max_layers: Maximum number of layers
            dropout: Dropout rate
        """
        super().__init__()

        self.min_layers = min_layers
        self.max_layers = max_layers

        # Create decoders with different complexities
        self.decoders = nn.ModuleDict({
            str(num_layers): PoseDecoder(
                num_keypoints=num_keypoints,
                keypoint_dim=keypoint_dim,
                hidden_dim=hidden_dim,
                spatial_layers=num_layers,
                dropout=dropout,
                use_temporal=True
            ) for num_layers in range(min_layers, max_layers + 1)
        })

    def _select_decoder(self, sequence_length: int) -> PoseDecoder:
        """Select appropriate decoder based on sequence length.

        Args:
            sequence_length: Input sequence length

        Returns:
            Selected decoder
        """
        # Simple heuristic: more layers for longer sequences
        if sequence_length <= 16:
            num_layers = self.min_layers
        elif sequence_length <= 32:
            num_layers = self.min_layers + 1
        elif sequence_length <= 64:
            num_layers = self.min_layers + 2
        else:
            num_layers = self.max_layers

        return self.decoders[str(num_layers)]

    def forward(
        self,
        x: torch.Tensor,
        target_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, hidden_dim) or (B, T, hidden_dim)
            target_shape: Target output shape (T, N, D)

        Returns:
            Reconstructed poses (B, T, N, D)
        """
        # Determine sequence length
        if len(x.shape) == 3:
            sequence_length = x.shape[1]
        elif target_shape is not None:
            sequence_length = target_shape[0]
        else:
            sequence_length = 64  # Default

        # Select and use appropriate decoder
        decoder = self._select_decoder(sequence_length)
        return decoder(x, target_shape)