"""ST-GCN encoder for pose tokenizer."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import sys
import os

# Add the parent directory to path to import existing ST-GCN modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../stgcn_layers'))
from stgcn_block import STGCN_block, GCN_unit


class MediaPipeGraph:
    """MediaPipe pose graph structure for ST-GCN."""

    def __init__(self, num_keypoints: int = 133):
        """Initialize MediaPipe graph.

        Args:
            num_keypoints: Number of keypoints (133 for MediaPipe holistic)
        """
        self.num_keypoints = num_keypoints
        self.edges = self._get_edges()
        self.A = self._get_adjacency_matrix()

    def _get_edges(self):
        """Get edge connections for MediaPipe pose landmarks."""
        # MediaPipe pose connections (33 landmarks: 0-32)
        pose_edges = [
            # Face outline
            (10, 9), (9, 8), (8, 7), (7, 6), (6, 5), (5, 4), (4, 3), (3, 2), (2, 1), (1, 0),
            # Body connections
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (24, 26), (25, 27), (26, 28),  # Legs
            (27, 29), (28, 30), (29, 31), (30, 32),  # Feet
            (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # Hands
            (17, 19), (18, 20), (27, 29), (27, 31), (28, 30), (28, 32)  # Feet details
        ]

        # For full MediaPipe holistic (133 keypoints), we add hand and face connections
        # Hand connections (21 landmarks each for left and right hands)
        left_hand_offset = 33  # Left hand starts at index 33
        right_hand_offset = 54  # Right hand starts at index 54

        hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]

        # Add left hand edges
        for edge in hand_edges:
            pose_edges.append((edge[0] + left_hand_offset, edge[1] + left_hand_offset))

        # Add right hand edges
        for edge in hand_edges:
            pose_edges.append((edge[0] + right_hand_offset, edge[1] + right_hand_offset))

        # Face connections (58 landmarks: 75-132)
        face_offset = 75
        # Simplified face connections (outline and major features)
        face_edges = [
            # Face outline
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
            (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
            # Eyes
            (17, 18), (18, 19), (19, 20), (20, 21), (21, 17),  # Left eye
            (22, 23), (23, 24), (24, 25), (25, 26), (26, 22),  # Right eye
            # Nose
            (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
            # Mouth
            (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 48)
        ]

        # Add face edges (only add if within range)
        for edge in face_edges:
            if edge[0] + face_offset < self.num_keypoints and edge[1] + face_offset < self.num_keypoints:
                pose_edges.append((edge[0] + face_offset, edge[1] + face_offset))

        return pose_edges

    def _get_adjacency_matrix(self):
        """Get adjacency matrix for the pose graph."""
        A = np.zeros((3, self.num_keypoints, self.num_keypoints))

        # Self-connections (identity)
        for i in range(self.num_keypoints):
            A[0, i, i] = 1

        # Neighbor connections
        for edge in self.edges:
            i, j = edge
            if i < self.num_keypoints and j < self.num_keypoints:
                A[1, i, j] = 1
                A[1, j, i] = 1

        # Distant connections (2-hop neighbors)
        A_squared = np.matmul(A[1], A[1])
        A[2] = (A_squared > 0).astype(float) - A[0] - A[1]
        A[2] = np.maximum(A[2], 0)

        # Normalize adjacency matrices
        for i in range(3):
            A_sum = A[i].sum(axis=1, keepdims=True)
            A_sum[A_sum == 0] = 1  # Avoid division by zero
            A[i] = A[i] / A_sum

        return torch.from_numpy(A).float()


class STGCNEncoder(nn.Module):
    """ST-GCN encoder for pose sequences."""

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
        """Initialize ST-GCN encoder.

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

        # Input projection
        self.input_proj = nn.Conv2d(
            keypoint_dim, 64, kernel_size=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(64)
        self.input_relu = nn.ReLU(inplace=True)

        # ST-GCN layers
        self.st_gcn_layers = nn.ModuleList()

        # Layer configurations: (in_channels, out_channels, stride)
        layer_configs = [
            (64, 64, 1),
            (64, 128, 1),
            (128, 128, 1),
            (128, 256, 1),
            (256, 256, 1),
            (256, hidden_dim, 1)
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

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input pose sequence (B, T, N, D)
            mask: Valid frame mask (B, T)

        Returns:
            Encoded features (B, hidden_dim)
        """
        B, T, N, D = x.shape

        # Reshape for ST-GCN: (B, D, T, N)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match input dimensions
            mask_expanded = mask.unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
            mask_expanded = mask_expanded.expand(-1, D, -1, N)  # (B, D, T, N)
            x = x * mask_expanded.float()

        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_relu(x)

        # ST-GCN layers
        for layer in self.st_gcn_layers:
            x = layer(x)

        # Global pooling: (B, C, T, N) -> (B, C, 1, 1)
        x = self.global_pool(x)
        x = x.view(B, -1)  # (B, C)

        # Output projection
        x = self.output_proj(x)

        return x

    def get_feature_maps(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> list:
        """Get intermediate feature maps from all layers.

        Args:
            x: Input pose sequence (B, T, N, D)
            mask: Valid frame mask (B, T)

        Returns:
            List of feature maps from each layer
        """
        B, T, N, D = x.shape

        # Reshape for ST-GCN: (B, D, T, N)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
            mask_expanded = mask_expanded.expand(-1, D, -1, N)  # (B, D, T, N)
            x = x * mask_expanded.float()

        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_relu(x)

        feature_maps = [x]

        # ST-GCN layers
        for layer in self.st_gcn_layers:
            x = layer(x)
            feature_maps.append(x)

        return feature_maps


class TemporalEncoder(nn.Module):
    """Temporal encoder for pose sequences using 1D convolutions."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """Initialize temporal encoder.

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
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            )

        # Final layer
        if num_layers > 1:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, T, D)
            mask: Valid frame mask (B, T)

        Returns:
            Encoded features (B, T, D)
        """
        # Transpose for 1D conv: (B, D, T)
        x = x.transpose(1, 2)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # (B, 1, T)
            x = x * mask_expanded.float()

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


class PoseEncoder(nn.Module):
    """Complete pose encoder combining spatial and temporal encoding."""

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
        """Initialize pose encoder.

        Args:
            num_keypoints: Number of keypoints per frame
            keypoint_dim: Dimension of each keypoint
            hidden_dim: Hidden dimension
            spatial_layers: Number of spatial ST-GCN layers
            temporal_layers: Number of temporal layers
            dropout: Dropout rate
            use_temporal: Whether to use temporal encoding
        """
        super().__init__()

        self.use_temporal = use_temporal

        # Spatial encoder (ST-GCN)
        self.spatial_encoder = STGCNEncoder(
            num_keypoints=num_keypoints,
            keypoint_dim=keypoint_dim,
            hidden_dim=hidden_dim,
            num_layers=spatial_layers,
            dropout=dropout
        )

        # Temporal encoder (optional)
        if use_temporal:
            self.temporal_encoder = TemporalEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=temporal_layers,
                dropout=dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input pose sequence (B, T, N, D)
            mask: Valid frame mask (B, T)

        Returns:
            Encoded features (B, hidden_dim) or (B, T, hidden_dim)
        """
        if self.use_temporal:
            # Frame-wise encoding for temporal modeling
            B, T, N, D = x.shape

            # Process each frame independently
            frame_features = []
            for t in range(T):
                frame = x[:, t:t+1]  # (B, 1, N, D)
                frame_mask = mask[:, t:t+1] if mask is not None else None

                # Encode frame
                feat = self.spatial_encoder(frame, frame_mask)  # (B, hidden_dim)
                frame_features.append(feat)

            # Stack frame features
            frame_features = torch.stack(frame_features, dim=1)  # (B, T, hidden_dim)

            # Temporal encoding
            features = self.temporal_encoder(frame_features, mask)  # (B, T, hidden_dim)

            return features
        else:
            # Global sequence encoding
            features = self.spatial_encoder(x, mask)  # (B, hidden_dim)
            return features