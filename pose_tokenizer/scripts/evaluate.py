#!/usr/bin/env python3
"""Evaluation script for pose tokenizer."""

import os
import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose_tokenizer.config import PoseTokenizerConfig
from pose_tokenizer.data.dataset import PoseDataset
from pose_tokenizer.models.model import PoseTokenizerModel
from pose_tokenizer.utils.evaluation import EvaluationSuite


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = log_dir / 'evaluation.log'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def load_model(config: PoseTokenizerConfig, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    model = PoseTokenizerModel(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP training)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def create_dataloader(config: PoseTokenizerConfig, split: str = 'test'):
    """Create evaluation dataloader."""
    dataset = PoseDataset(
        data_dir=config.data_dir,
        split=split,
        sequence_length=config.sequence_length,
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        augment=False,  # No augmentation for evaluation
        normalize=config.normalize_poses
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader


def evaluate_reconstruction_quality(model, dataloader, device, logger):
    """Evaluate reconstruction quality metrics."""
    model.eval()

    total_samples = 0
    total_mse = 0.0
    total_mae = 0.0
    total_pck = 0.0

    reconstruction_errors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            poses = batch['poses'].to(device)  # (B, T, N, D)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass
            encoded = model.encoder(poses, mask)
            quantized, indices, _ = model.tokenizer(encoded)
            reconstructed = model.decoder(quantized, poses.shape[1:])

            # Compute metrics
            if poses.shape[-1] == 3:
                # Extract positions (ignore confidence)
                pred_pos = reconstructed[..., :2]
                target_pos = poses[..., :2]
            else:
                pred_pos = reconstructed
                target_pos = poses

            # Apply mask if available
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
                pred_pos = pred_pos * mask_expanded
                target_pos = target_pos * mask_expanded
                valid_samples = torch.sum(mask).item()
            else:
                valid_samples = poses.shape[0] * poses.shape[1]

            # MSE
            mse = torch.mean((pred_pos - target_pos) ** 2).item()
            total_mse += mse * valid_samples

            # MAE
            mae = torch.mean(torch.abs(pred_pos - target_pos)).item()
            total_mae += mae * valid_samples

            # PCK (Percentage of Correct Keypoints)
            distances = torch.sqrt(torch.sum((pred_pos - target_pos) ** 2, dim=-1))
            pck_threshold = 0.05  # 5% of normalized coordinates
            correct_keypoints = distances < pck_threshold
            if mask is not None:
                correct_keypoints = correct_keypoints * mask.unsqueeze(-1)
            pck = torch.mean(correct_keypoints.float()).item()
            total_pck += pck * valid_samples

            # Store per-sample reconstruction errors
            sample_errors = torch.mean(distances, dim=(1, 2)).cpu().numpy()
            reconstruction_errors.extend(sample_errors)

            total_samples += valid_samples

            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Compute final metrics
    metrics = {
        'mse': total_mse / total_samples,
        'mae': total_mae / total_samples,
        'rmse': np.sqrt(total_mse / total_samples),
        'pck': total_pck / total_samples,
        'reconstruction_error_mean': np.mean(reconstruction_errors),
        'reconstruction_error_std': np.std(reconstruction_errors),
        'reconstruction_error_median': np.median(reconstruction_errors)
    }

    return metrics


def evaluate_tokenization_quality(model, dataloader, device, logger):
    """Evaluate tokenization quality metrics."""
    model.eval()

    codebook_usage = {}
    quantization_errors = []
    compression_ratios = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            poses = batch['poses'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass
            encoded = model.encoder(poses, mask)
            quantized, indices, quant_losses = model.tokenizer(encoded)

            # Quantization error
            quant_error = torch.mean((encoded - quantized) ** 2).item()
            quantization_errors.append(quant_error)

            # Codebook usage
            indices_np = indices.detach().cpu().numpy()
            for q in range(indices_np.shape[0]):  # num_quantizers
                if q not in codebook_usage:
                    codebook_usage[q] = {}

                layer_indices = indices_np[q].flatten()
                for idx in layer_indices:
                    idx = int(idx)
                    codebook_usage[q][idx] = codebook_usage[q].get(idx, 0) + 1

            # Compression ratio (original size vs tokenized size)
            original_size = encoded.numel() * 4  # float32 = 4 bytes
            tokenized_size = indices.numel() * 2  # assume 16-bit indices
            compression_ratio = original_size / tokenized_size
            compression_ratios.append(compression_ratio)

            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Compute codebook utilization metrics
    utilization_metrics = {}
    for q, usage in codebook_usage.items():
        total_usage = sum(usage.values())
        num_used_codes = len(usage)
        codebook_size = model.tokenizer.codebook_size

        utilization = num_used_codes / codebook_size

        # Entropy of usage distribution
        usage_probs = np.array(list(usage.values())) / total_usage
        entropy = -np.sum(usage_probs * np.log(usage_probs + 1e-8))

        utilization_metrics[f'codebook_{q}_utilization'] = utilization
        utilization_metrics[f'codebook_{q}_entropy'] = entropy
        utilization_metrics[f'codebook_{q}_num_used'] = num_used_codes

    # Compute final metrics
    metrics = {
        'avg_quantization_error': np.mean(quantization_errors),
        'std_quantization_error': np.std(quantization_errors),
        'avg_compression_ratio': np.mean(compression_ratios),
        'std_compression_ratio': np.std(compression_ratios),
        **utilization_metrics
    }

    return metrics, codebook_usage


def generate_visualizations(model, dataloader, device, save_dir, logger, num_samples=5):
    """Generate visualization samples."""
    from pose_tokenizer.utils.evaluation import PoseVisualizer

    visualizer = PoseVisualizer(model.config.num_keypoints)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    samples_generated = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if samples_generated >= num_samples:
                break

            poses = batch['poses'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass
            encoded = model.encoder(poses, mask)
            quantized, indices, _ = model.tokenizer(encoded)
            reconstructed = model.decoder(quantized, poses.shape[1:])

            # Convert to numpy
            poses_np = poses.cpu().numpy()
            reconstructed_np = reconstructed.cpu().numpy()

            # Generate visualizations for each sample in batch
            batch_size = poses_np.shape[0]
            for i in range(min(batch_size, num_samples - samples_generated)):
                sample_idx = samples_generated + i

                # Extract positions (ignore confidence if present)
                if poses_np.shape[-1] == 3:
                    original_pos = poses_np[i, :, :, :2]
                    recon_pos = reconstructed_np[i, :, :, :2]
                else:
                    original_pos = poses_np[i]
                    recon_pos = reconstructed_np[i]

                # Plot comparison
                fig = visualizer.plot_pose_comparison(
                    original_pos, recon_pos, frame_idx=0,
                    save_path=save_dir / f'comparison_sample_{sample_idx}.png'
                )

                # Plot sequence
                fig = visualizer.plot_pose_sequence(
                    original_pos,
                    title=f'Original Sequence {sample_idx}',
                    save_path=save_dir / f'original_sequence_{sample_idx}.png'
                )

                fig = visualizer.plot_pose_sequence(
                    recon_pos,
                    title=f'Reconstructed Sequence {sample_idx}',
                    save_path=save_dir / f'reconstructed_sequence_{sample_idx}.png'
                )

                logger.info(f"Generated visualizations for sample {sample_idx}")

            samples_generated += min(batch_size, num_samples - samples_generated)

    logger.info(f"Generated {samples_generated} visualization samples")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate pose tokenizer')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization samples')
    parser.add_argument('--num-vis-samples', type=int, default=5,
                       help='Number of samples to visualize')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting pose tokenizer evaluation")

    # Load configuration
    config = PoseTokenizerConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(config, args.checkpoint, device)
    logger.info(f"Model loaded successfully")

    # Create dataloader
    logger.info(f"Creating dataloader for {args.split} split")
    dataloader = create_dataloader(config, args.split)
    logger.info(f"Dataset size: {len(dataloader.dataset)} samples")

    # Evaluate reconstruction quality
    logger.info("Evaluating reconstruction quality...")
    reconstruction_metrics = evaluate_reconstruction_quality(model, dataloader, device, logger)
    logger.info("Reconstruction evaluation completed")

    # Evaluate tokenization quality
    logger.info("Evaluating tokenization quality...")
    tokenization_metrics, codebook_usage = evaluate_tokenization_quality(model, dataloader, device, logger)
    logger.info("Tokenization evaluation completed")

    # Combine results
    results = {
        'config': config.__dict__,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'evaluation_time': datetime.now().isoformat(),
        'reconstruction_metrics': reconstruction_metrics,
        'tokenization_metrics': tokenization_metrics,
        'codebook_usage': {str(k): v for k, v in codebook_usage.items()}
    }

    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Generate report
    report_lines = [
        "# Pose Tokenizer Evaluation Report",
        f"",
        f"**Evaluation Time:** {results['evaluation_time']}",
        f"**Checkpoint:** {args.checkpoint}",
        f"**Dataset Split:** {args.split}",
        f"**Dataset Size:** {len(dataloader.dataset)} samples",
        f"",
        "## Reconstruction Metrics",
        ""
    ]

    for metric, value in reconstruction_metrics.items():
        if isinstance(value, float):
            report_lines.append(f"- **{metric.upper()}**: {value:.6f}")

    report_lines.extend([
        "",
        "## Tokenization Metrics",
        ""
    ])

    for metric, value in tokenization_metrics.items():
        if isinstance(value, float):
            if 'utilization' in metric:
                report_lines.append(f"- **{metric}**: {value:.2%}")
            else:
                report_lines.append(f"- **{metric}**: {value:.6f}")

    report = "\n".join(report_lines)

    # Save report
    report_file = output_dir / 'evaluation_report.md'
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"MSE: {reconstruction_metrics['mse']:.6f}")
    print(f"MAE: {reconstruction_metrics['mae']:.6f}")
    print(f"RMSE: {reconstruction_metrics['rmse']:.6f}")
    print(f"PCK: {reconstruction_metrics['pck']:.4f}")
    print(f"Quantization Error: {tokenization_metrics['avg_quantization_error']:.6f}")
    print(f"Compression Ratio: {tokenization_metrics['avg_compression_ratio']:.2f}x")

    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        vis_dir = output_dir / 'visualizations'
        generate_visualizations(model, dataloader, device, vis_dir, logger, args.num_vis_samples)
        logger.info(f"Visualizations saved to {vis_dir}")

    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()