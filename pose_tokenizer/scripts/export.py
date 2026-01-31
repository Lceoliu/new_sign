#!/usr/bin/env python3
"""Export trained pose tokenizer model to various formats."""

import os
import sys
import argparse
import logging
import torch
import torch.onnx
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose_tokenizer.config import PoseTokenizerConfig
from pose_tokenizer.models.model import PoseTokenizerModel


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
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


def export_onnx(
    model: torch.nn.Module,
    config: PoseTokenizerConfig,
    output_path: str,
    dynamic_axes: bool = True,
    opset_version: int = 11
):
    """Export model to ONNX format."""
    logger = logging.getLogger(__name__)
    logger.info(f"Exporting model to ONNX: {output_path}")

    # Create dummy input
    batch_size = 1
    dummy_poses = torch.randn(
        batch_size,
        config.sequence_length,
        config.num_keypoints,
        config.keypoint_dim
    )
    dummy_mask = torch.ones(batch_size, config.sequence_length, dtype=torch.bool)

    # Define input and output names
    input_names = ['poses', 'mask']
    output_names = ['reconstructed', 'indices', 'quantized']

    # Define dynamic axes
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'poses': {0: 'batch_size', 1: 'sequence_length'},
            'mask': {0: 'batch_size', 1: 'sequence_length'},
            'reconstructed': {0: 'batch_size', 1: 'sequence_length'},
            'indices': {1: 'batch_size', 2: 'sequence_length'},
            'quantized': {0: 'batch_size', 1: 'sequence_length'}
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_poses, dummy_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        verbose=False
    )

    logger.info(f"ONNX export completed: {output_path}")

    # Save metadata
    metadata = {
        'model_type': 'pose_tokenizer',
        'config': config.__dict__,
        'input_names': input_names,
        'output_names': output_names,
        'input_shapes': {
            'poses': [batch_size, config.sequence_length, config.num_keypoints, config.keypoint_dim],
            'mask': [batch_size, config.sequence_length]
        },
        'opset_version': opset_version,
        'dynamic_axes': dynamic_axes_dict is not None
    }

    metadata_path = output_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved: {metadata_path}")


def export_torchscript(
    model: torch.nn.Module,
    config: PoseTokenizerConfig,
    output_path: str,
    method: str = 'trace'
):
    """Export model to TorchScript format."""
    logger = logging.getLogger(__name__)
    logger.info(f"Exporting model to TorchScript ({method}): {output_path}")

    if method == 'trace':
        # Create dummy input for tracing
        dummy_poses = torch.randn(
            1,
            config.sequence_length,
            config.num_keypoints,
            config.keypoint_dim
        )
        dummy_mask = torch.ones(1, config.sequence_length, dtype=torch.bool)

        # Trace the model
        traced_model = torch.jit.trace(model, (dummy_poses, dummy_mask))
        traced_model.save(output_path)

    elif method == 'script':
        # Script the model
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)

    else:
        raise ValueError(f"Unknown TorchScript export method: {method}")

    logger.info(f"TorchScript export completed: {output_path}")

    # Save metadata
    metadata = {
        'model_type': 'pose_tokenizer',
        'config': config.__dict__,
        'export_method': method,
        'input_shapes': {
            'poses': [1, config.sequence_length, config.num_keypoints, config.keypoint_dim],
            'mask': [1, config.sequence_length]
        }
    }

    metadata_path = output_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved: {metadata_path}")


def validate_onnx_export(onnx_path: str, model: torch.nn.Module, config: PoseTokenizerConfig):
    """Validate ONNX export by comparing outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        logging.getLogger(__name__).warning("ONNXRuntime not available, skipping validation")
        return

    logger = logging.getLogger(__name__)
    logger.info("Validating ONNX export...")

    # Create test input
    test_poses = torch.randn(1, config.sequence_length, config.num_keypoints, config.keypoint_dim)
    test_mask = torch.ones(1, config.sequence_length, dtype=torch.bool)

    # Get PyTorch output
    model.eval()
    with torch.no_grad():
        torch_output = model(test_poses, test_mask)

    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    onnx_inputs = {
        'poses': test_poses.numpy(),
        'mask': test_mask.numpy()
    }
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # Compare outputs
    torch_recon = torch_output[0].numpy()
    onnx_recon = onnx_outputs[0]

    max_diff = np.abs(torch_recon - onnx_recon).max()
    mean_diff = np.abs(torch_recon - onnx_recon).mean()

    logger.info(f"ONNX validation - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    if max_diff < 1e-4:
        logger.info("ONNX export validation passed!")
    else:
        logger.warning(f"ONNX export validation failed - large difference: {max_diff}")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export pose tokenizer model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'both'],
                       default='onnx', help='Export format')
    parser.add_argument('--torchscript-method', type=str, choices=['trace', 'script'],
                       default='trace', help='TorchScript export method')
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--no-dynamic-axes', action='store_true',
                       help='Disable dynamic axes in ONNX export')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported model')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("Starting model export")

    # Load configuration
    config = PoseTokenizerConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(config, args.checkpoint, device)
    logger.info("Model loaded successfully")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export model
    if args.format in ['onnx', 'both']:
        onnx_path = str(output_path.with_suffix('.onnx'))
        export_onnx(
            model=model,
            config=config,
            output_path=onnx_path,
            dynamic_axes=not args.no_dynamic_axes,
            opset_version=args.opset_version
        )

        if args.validate:
            validate_onnx_export(onnx_path, model, config)

    if args.format in ['torchscript', 'both']:
        ts_path = str(output_path.with_suffix('.pt'))
        export_torchscript(
            model=model,
            config=config,
            output_path=ts_path,
            method=args.torchscript_method
        )

    logger.info("Export completed successfully!")


if __name__ == '__main__':
    main()