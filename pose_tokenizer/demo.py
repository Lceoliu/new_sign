#!/usr/bin/env python3
"""
Simple example demonstrating how to use the Pose Tokenizer.

This script shows:
1. How to create synthetic pose data
2. How to train a simple model
3. How to evaluate and visualize results
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose_tokenizer.config import PoseTokenizerConfig
from pose_tokenizer.models.model import PoseTokenizerModel
from pose_tokenizer.data.dataset import PoseDataset
from pose_tokenizer.training.trainer import PoseTokenizerTrainer
from pose_tokenizer.utils.evaluation import PoseVisualizer


def create_synthetic_data(output_dir: str, num_samples: int = 100):
    """Create synthetic pose data for demonstration."""
    print(f"Creating {num_samples} synthetic pose samples...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create train/val/test splits
    splits = {
        'train': int(0.7 * num_samples),
        'val': int(0.15 * num_samples),
        'test': int(0.15 * num_samples)
    }

    sample_idx = 0
    for split, count in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)

        for i in range(count):
            # Generate synthetic pose sequence
            sequence_length = np.random.randint(32, 128)
            num_keypoints = 133

            # Create realistic pose motion (simple walking pattern)
            t = np.linspace(0, 4 * np.pi, sequence_length)

            poses = np.zeros((sequence_length, num_keypoints, 3))

            # Add some basic motion patterns
            for kp in range(num_keypoints):
                # X coordinate: slight oscillation
                poses[:, kp, 0] = 0.5 + 0.1 * np.sin(t + kp * 0.1)
                # Y coordinate: vertical movement
                poses[:, kp, 1] = 0.5 + 0.05 * np.cos(2 * t + kp * 0.2)
                # Confidence: random but mostly high
                poses[:, kp, 2] = np.random.uniform(0.7, 1.0, sequence_length)

            # Add some noise
            poses[:, :, :2] += np.random.normal(0, 0.01, poses[:, :, :2].shape)

            # Create mask (all frames valid)
            mask = np.ones(sequence_length, dtype=bool)

            # Save as npz
            sample_path = split_dir / f"sample_{sample_idx:06d}.npz"
            np.savez(sample_path, poses=poses, mask=mask)

            sample_idx += 1

    print(f"Synthetic data created in {output_dir}")
    return str(output_dir)


def run_quick_training_demo():
    """Run a quick training demonstration."""
    print("=" * 60)
    print("POSE TOKENIZER DEMO")
    print("=" * 60)

    # Create synthetic data
    data_dir = create_synthetic_data("demo_data", num_samples=50)

    # Create a small configuration for quick demo
    config = PoseTokenizerConfig(
        # Model
        num_keypoints=133,
        keypoint_dim=3,
        sequence_length=64,
        hidden_dim=64,
        num_layers=2,

        # Tokenizer
        num_quantizers=2,
        codebook_size=256,

        # Training
        batch_size=4,
        num_epochs=5,
        learning_rate=1e-3,

        # Data
        data_dir=data_dir,
        normalize_poses=True,
        augment_data=False,
        num_workers=0,

        # Paths
        checkpoint_dir="demo_checkpoints",
        log_dir="demo_logs",
        output_dir="demo_outputs"
    )

    print(f"Configuration created:")
    print(f"- Model size: {config.hidden_dim}D, {config.num_layers} layers")
    print(f"- Tokenizer: {config.num_quantizers} quantizers, {config.codebook_size} codes each")
    print(f"- Training: {config.num_epochs} epochs, batch size {config.batch_size}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PoseTokenizerModel(config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} trainable, {total_params:,} total")

    # Create trainer
    trainer = PoseTokenizerTrainer(config, model, device)

    # Quick training
    print("\nStarting training...")
    trainer.train()

    # Load best model for evaluation
    best_checkpoint = Path(config.checkpoint_dir) / "best.pth"
    if best_checkpoint.exists():
        print(f"\nLoading best model from {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Quick evaluation
    print("\nRunning evaluation...")
    model.eval()

    # Load test data
    from torch.utils.data import DataLoader
    test_dataset = PoseDataset(
        data_dir=config.data_dir,
        split='test',
        sequence_length=config.sequence_length,
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        augment=False,
        normalize=config.normalize_poses
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Evaluate on a few samples
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 3:  # Only evaluate a few batches
                break

            poses = batch['poses'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass
            reconstructed, indices, quantized = model(poses, mask)

            # Compute MSE
            if poses.shape[-1] == 3:
                # Compare positions only (ignore confidence)
                mse = torch.mean((reconstructed[..., :2] - poses[..., :2]) ** 2)
            else:
                mse = torch.mean((reconstructed - poses) ** 2)

            total_mse += mse.item() * poses.shape[0]
            total_samples += poses.shape[0]

    avg_mse = total_mse / total_samples if total_samples > 0 else 0
    print(f"Average MSE on test set: {avg_mse:.6f}")

    # Create a simple visualization
    print("\nCreating visualization...")
    try:
        # Get one sample for visualization
        sample_batch = next(iter(test_loader))
        poses = sample_batch['poses'][:1].to(device)  # Take first sample
        mask = sample_batch.get('mask', None)
        if mask is not None:
            mask = mask[:1].to(device)

        with torch.no_grad():
            reconstructed, _, _ = model(poses, mask)

        # Convert to numpy
        original = poses[0].cpu().numpy()
        recon = reconstructed[0].cpu().numpy()

        # Extract positions (ignore confidence)
        if original.shape[-1] == 3:
            original_pos = original[:, :, :2]
            recon_pos = recon[:, :, :2]
        else:
            original_pos = original
            recon_pos = recon

        # Simple visualization using matplotlib
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot first frame of original
        ax1.scatter(original_pos[0, :, 0], original_pos[0, :, 1], alpha=0.6, s=20)
        ax1.set_title('Original Pose (Frame 0)')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Plot first frame of reconstruction
        ax2.scatter(recon_pos[0, :, 0], recon_pos[0, :, 1], alpha=0.6, s=20, color='red')
        ax2.set_title('Reconstructed Pose (Frame 0)')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save visualization
        vis_dir = Path(config.output_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / "demo_comparison.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to {vis_path}")

    except Exception as e:
        print(f"Visualization failed: {e}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print(f"Results saved in:")
    print(f"- Checkpoints: {config.checkpoint_dir}")
    print(f"- Logs: {config.log_dir}")
    print(f"- Outputs: {config.output_dir}")
    print("\nTo run full training, use:")
    print("python scripts/train.py --config configs/default.yaml")


if __name__ == '__main__':
    run_quick_training_demo()