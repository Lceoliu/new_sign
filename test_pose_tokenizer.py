#!/usr/bin/env python3
"""
Test script for Pose Tokenizer integration in Uni-Sign project.
"""

import torch
import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Uni_Sign, PoseTokenizer, VectorQuantizer, ResidualVectorQuantizer
from config import POSE_TOKENIZER_CONFIG


def test_vector_quantizer():
    """Test basic VectorQuantizer functionality."""
    print("Testing VectorQuantizer...")

    # Create test data
    batch_size, seq_len, dim = 2, 10, 256
    test_input = torch.randn(batch_size, seq_len, dim)

    # Create quantizer
    vq = VectorQuantizer(num_embeddings=1024, embedding_dim=dim)

    # Forward pass
    quantized, loss, perplexity, indices = vq(test_input)

    # Check outputs
    assert quantized.shape == test_input.shape, f"Shape mismatch: {quantized.shape} vs {test_input.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative: {loss.item()}"
    assert perplexity.item() > 0, f"Perplexity should be positive: {perplexity.item()}"
    assert indices.shape == (batch_size * seq_len,), f"Indices shape mismatch: {indices.shape}"

    print(f"✓ VectorQuantizer test passed")
    print(f"  - Input shape: {test_input.shape}")
    print(f"  - Output shape: {quantized.shape}")
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Perplexity: {perplexity.item():.2f}")
    print()


def test_residual_vector_quantizer():
    """Test ResidualVectorQuantizer functionality."""
    print("Testing ResidualVectorQuantizer...")

    # Create test data
    batch_size, seq_len, dim = 2, 10, 256
    test_input = torch.randn(batch_size, seq_len, dim)

    # Create RVQ
    rvq = ResidualVectorQuantizer(
        num_quantizers=4,
        num_embeddings=1024,
        embedding_dim=dim
    )

    # Forward pass
    quantized, loss, perplexity, indices_list = rvq(test_input)

    # Check outputs
    assert quantized.shape == test_input.shape, f"Shape mismatch: {quantized.shape} vs {test_input.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative: {loss.item()}"
    assert perplexity.item() > 0, f"Perplexity should be positive: {perplexity.item()}"
    assert len(indices_list) == 4, f"Should have 4 quantizer indices: {len(indices_list)}"

    print(f"✓ ResidualVectorQuantizer test passed")
    print(f"  - Input shape: {test_input.shape}")
    print(f"  - Output shape: {quantized.shape}")
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Perplexity: {perplexity.item():.2f}")
    print(f"  - Number of quantizers: {len(indices_list)}")
    print()


def test_pose_tokenizer():
    """Test PoseTokenizer functionality."""
    print("Testing PoseTokenizer...")

    # Create test data (simulating concatenated ST-GCN features)
    batch_size, seq_len, input_dim = 2, 10, 1024  # 256*4 from 4 body parts
    test_input = torch.randn(batch_size, seq_len, input_dim)

    # Create pose tokenizer
    tokenizer = PoseTokenizer(
        input_dim=input_dim,
        hidden_dim=256,
        num_quantizers=4,
        num_embeddings=1024
    )

    # Forward pass
    decoded, vq_loss, perplexity, indices_list = tokenizer(test_input)

    # Check outputs
    assert decoded.shape == test_input.shape, f"Shape mismatch: {decoded.shape} vs {test_input.shape}"
    assert vq_loss.item() >= 0, f"VQ loss should be non-negative: {vq_loss.item()}"
    assert perplexity.item() > 0, f"Perplexity should be positive: {perplexity.item()}"
    assert len(indices_list) == 4, f"Should have 4 quantizer indices: {len(indices_list)}"

    print(f"✓ PoseTokenizer test passed")
    print(f"  - Input shape: {test_input.shape}")
    print(f"  - Output shape: {decoded.shape}")
    print(f"  - VQ Loss: {vq_loss.item():.4f}")
    print(f"  - Perplexity: {perplexity.item():.2f}")
    print(f"  - Reconstruction error: {torch.mean((decoded - test_input)**2).item():.6f}")
    print()


def test_uni_sign_integration():
    """Test Uni_Sign model with pose tokenizer integration."""
    print("Testing Uni_Sign with pose tokenizer integration...")

    # Create mock arguments
    class MockArgs:
        def __init__(self):
            self.hidden_dim = 256
            self.dataset = "CSL_News"
            self.label_smoothing = 0.1

            # Pose tokenizer settings
            self.use_pose_tokenizer = True
            self.tokenizer_hidden_dim = 256
            self.num_quantizers = 4
            self.codebook_size = 1024
            self.commitment_cost = 0.25
            self.vq_loss_weight = 1.0

    args = MockArgs()

    # Create model
    try:
        model = Uni_Sign(args)
        print(f"✓ Uni_Sign model created successfully")
        print(f"  - Use pose tokenizer: {model.use_pose_tokenizer}")

        if model.use_pose_tokenizer:
            print(f"  - Pose tokenizer input dim: {model.pose_tokenizer.pre_quant_conv.in_channels}")
            print(f"  - Pose tokenizer hidden dim: {model.pose_tokenizer.pre_quant_conv.out_channels}")
            print(f"  - Number of quantizers: {model.pose_tokenizer.quantizer.num_quantizers}")
            print(f"  - Codebook size: {model.pose_tokenizer.quantizer.quantizers[0].num_embeddings}")

        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        if model.use_pose_tokenizer:
            tokenizer_params = sum(p.numel() for p in model.pose_tokenizer.parameters())
            print(f"  - Pose tokenizer parameters: {tokenizer_params:,}")

        print()

    except Exception as e:
        print(f"✗ Uni_Sign model creation failed: {e}")
        return False

    return True


def test_config_integration():
    """Test configuration integration."""
    print("Testing configuration integration...")

    # Check if config is properly loaded
    assert 'use_pose_tokenizer' in POSE_TOKENIZER_CONFIG
    assert 'tokenizer_hidden_dim' in POSE_TOKENIZER_CONFIG
    assert 'num_quantizers' in POSE_TOKENIZER_CONFIG
    assert 'codebook_size' in POSE_TOKENIZER_CONFIG

    print(f"✓ Configuration integration test passed")
    print(f"  - Available config keys: {list(POSE_TOKENIZER_CONFIG.keys())}")
    print(f"  - Default use_pose_tokenizer: {POSE_TOKENIZER_CONFIG['use_pose_tokenizer']}")
    print(f"  - Default codebook_size: {POSE_TOKENIZER_CONFIG['codebook_size']}")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Pose Tokenizer Integration Test Suite")
    print("=" * 60)
    print()

    try:
        # Test individual components
        test_vector_quantizer()
        test_residual_vector_quantizer()
        test_pose_tokenizer()

        # Test configuration
        test_config_integration()

        # Test full integration
        success = test_uni_sign_integration()

        if success:
            print("=" * 60)
            print("🎉 All tests passed! Pose tokenizer integration is working correctly.")
            print("=" * 60)
            print()
            print("Next steps:")
            print("1. Run training with --use_pose_tokenizer=True")
            print("2. Monitor VQ loss and perplexity during training")
            print("3. Evaluate codebook utilization")
            print()
        else:
            print("=" * 60)
            print("❌ Some tests failed. Please check the errors above.")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)