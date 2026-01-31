#!/usr/bin/env python3
"""Basic tests for Pose Tokenizer components."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from pose_tokenizer.config import PoseTokenizerConfig
from pose_tokenizer.models.model import PoseTokenizerModel
from pose_tokenizer.models.encoder import STGCNEncoder
from pose_tokenizer.models.tokenizer import RVQTokenizer
from pose_tokenizer.models.decoder import PoseDecoder
from pose_tokenizer.data.dataset import PoseDataset


class TestPoseTokenizerConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = PoseTokenizerConfig()
        assert config.num_keypoints == 133
        assert config.keypoint_dim == 3
        assert config.sequence_length == 64
        assert config.hidden_dim == 256

    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        config_dict = {
            'num_keypoints': 21,
            'keypoint_dim': 2,
            'sequence_length': 32,
            'hidden_dim': 128
        }
        config = PoseTokenizerConfig(**config_dict)
        assert config.num_keypoints == 21
        assert config.keypoint_dim == 2
        assert config.sequence_length == 32
        assert config.hidden_dim == 128


class TestSTGCNEncoder:
    """Test ST-GCN encoder."""

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        config = PoseTokenizerConfig(
            num_keypoints=21,
            keypoint_dim=3,
            sequence_length=32,
            hidden_dim=64
        )

        encoder = STGCNEncoder(config)

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, config.sequence_length, config.num_keypoints, config.keypoint_dim)

        # Forward pass
        output = encoder(x)

        # Check output shape
        expected_shape = (batch_size, config.sequence_length, config.hidden_dim)
        assert output.shape == expected_shape

    def test_encoder_with_mask(self):
        """Test encoder with mask."""
        config = PoseTokenizerConfig(
            num_keypoints=21,
            keypoint_dim=3,
            sequence_length=32,
            hidden_dim=64
        )

        encoder = STGCNEncoder(config)

        # Create dummy input with mask
        batch_size = 2
        x = torch.randn(batch_size, config.sequence_length, config.num_keypoints, config.keypoint_dim)
        mask = torch.ones(batch_size, config.sequence_length, dtype=torch.bool)
        mask[:, -5:] = False  # Mask last 5 frames

        # Forward pass
        output = encoder(x, mask)

        # Check output shape
        expected_shape = (batch_size, config.sequence_length, config.hidden_dim)
        assert output.shape == expected_shape


class TestRVQTokenizer:
    """Test RVQ tokenizer."""

    def test_tokenizer_forward(self):
        """Test tokenizer forward pass."""
        config = PoseTokenizerConfig(
            hidden_dim=64,
            num_quantizers=2,
            codebook_size=256
        )

        tokenizer = RVQTokenizer(config)

        # Create dummy input
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Forward pass
        quantized, indices, losses = tokenizer(x)

        # Check output shapes
        assert quantized.shape == x.shape
        assert indices.shape == (config.num_quantizers, batch_size, seq_len)
        assert len(losses) == config.num_quantizers

    def test_tokenizer_encode_decode(self):
        """Test tokenizer encode/decode consistency."""
        config = PoseTokenizerConfig(
            hidden_dim=64,
            num_quantizers=2,
            codebook_size=256
        )

        tokenizer = RVQTokenizer(config)

        # Create dummy input
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Encode
        indices = tokenizer.encode(x)

        # Decode
        decoded = tokenizer.decode(indices)

        # Check shapes
        assert indices.shape == (config.num_quantizers, batch_size, seq_len)
        assert decoded.shape == x.shape


class TestPoseDecoder:
    """Test pose decoder."""

    def test_decoder_forward(self):
        """Test decoder forward pass."""
        config = PoseTokenizerConfig(
            num_keypoints=21,
            keypoint_dim=3,
            sequence_length=32,
            hidden_dim=64
        )

        decoder = PoseDecoder(config)

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, config.sequence_length, config.hidden_dim)

        # Forward pass
        output = decoder(x)

        # Check output shape
        expected_shape = (batch_size, config.sequence_length, config.num_keypoints, config.keypoint_dim)
        assert output.shape == expected_shape


class TestPoseTokenizerModel:
    """Test complete pose tokenizer model."""

    def test_model_forward(self):
        """Test model forward pass."""
        config = PoseTokenizerConfig(
            num_keypoints=21,
            keypoint_dim=3,
            sequence_length=32,
            hidden_dim=64,
            num_quantizers=2,
            codebook_size=256
        )

        model = PoseTokenizerModel(config)

        # Create dummy input
        batch_size = 2
        poses = torch.randn(batch_size, config.sequence_length, config.num_keypoints, config.keypoint_dim)

        # Forward pass
        reconstructed, indices, quantized = model(poses)

        # Check output shapes
        assert reconstructed.shape == poses.shape
        assert indices.shape == (config.num_quantizers, batch_size, config.sequence_length)
        assert quantized.shape == (batch_size, config.sequence_length, config.hidden_dim)

    def test_model_with_mask(self):
        """Test model with mask."""
        config = PoseTokenizerConfig(
            num_keypoints=21,
            keypoint_dim=3,
            sequence_length=32,
            hidden_dim=64,
            num_quantizers=2,
            codebook_size=256
        )

        model = PoseTokenizerModel(config)

        # Create dummy input with mask
        batch_size = 2
        poses = torch.randn(batch_size, config.sequence_length, config.num_keypoints, config.keypoint_dim)
        mask = torch.ones(batch_size, config.sequence_length, dtype=torch.bool)
        mask[:, -5:] = False  # Mask last 5 frames

        # Forward pass
        reconstructed, indices, quantized = model(poses, mask)

        # Check output shapes
        assert reconstructed.shape == poses.shape
        assert indices.shape == (config.num_quantizers, batch_size, config.sequence_length)
        assert quantized.shape == (batch_size, config.sequence_length, config.hidden_dim)


class TestPoseDataset:
    """Test pose dataset."""

    def create_dummy_data(self, data_dir: Path, num_samples: int = 5):
        """Create dummy data for testing."""
        data_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            # Create dummy pose data
            seq_len = np.random.randint(20, 50)
            poses = np.random.randn(seq_len, 21, 3)
            mask = np.ones(seq_len, dtype=bool)

            # Save as npz
            np.savez(data_dir / f"sample_{i:03d}.npz", poses=poses, mask=mask)

    def test_dataset_loading(self):
        """Test dataset loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            self.create_dummy_data(data_dir)

            # Create dataset
            dataset = PoseDataset(
                data_dir=str(data_dir),
                sequence_length=32,
                num_keypoints=21,
                keypoint_dim=3
            )

            # Check dataset length
            assert len(dataset) == 5

            # Check sample
            sample = dataset[0]
            assert 'poses' in sample
            assert sample['poses'].shape == (32, 21, 3)

    def test_dataset_with_augmentation(self):
        """Test dataset with augmentation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            self.create_dummy_data(data_dir)

            # Create dataset with augmentation
            dataset = PoseDataset(
                data_dir=str(data_dir),
                sequence_length=32,
                num_keypoints=21,
                keypoint_dim=3,
                augment=True
            )

            # Get two samples of the same data
            sample1 = dataset[0]
            sample2 = dataset[0]

            # They should be different due to augmentation
            assert not torch.equal(sample1['poses'], sample2['poses'])


if __name__ == '__main__':
    pytest.main([__file__])