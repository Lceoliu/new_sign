"""Model export utilities."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import json
import numpy as np


def export_tokenizer(
    model: nn.Module,
    config: Dict[str, Any],
    export_path: str,
    format: str = "pytorch"
) -> None:
    """Export trained pose tokenizer model.

    Args:
        model: Trained pose tokenizer model
        config: Model configuration
        export_path: Path to save exported model
        format: Export format ('pytorch', 'onnx', 'torchscript')
    """
    export_path = Path(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    if format == "pytorch":
        _export_pytorch(model, config, export_path)
    elif format == "onnx":
        _export_onnx(model, config, export_path)
    elif format == "torchscript":
        _export_torchscript(model, config, export_path)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _export_pytorch(
    model: nn.Module,
    config: Dict[str, Any],
    export_path: Path
) -> None:
    """Export model in PyTorch format."""
    # Save model state dict
    torch.save(
        model.state_dict(),
        export_path / "model.pth"
    )

    # Save full model (for easier loading)
    torch.save(
        model,
        export_path / "model_full.pth"
    )

    # Save configuration
    with open(export_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save model info
    model_info = {
        "format": "pytorch",
        "model_type": "pose_tokenizer",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    with open(export_path / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)


def _export_onnx(
    model: nn.Module,
    config: Dict[str, Any],
    export_path: Path
) -> None:
    """Export model in ONNX format."""
    try:
        import onnx
        import onnxruntime
    except ImportError:
        raise ImportError("ONNX export requires 'onnx' and 'onnxruntime' packages")

    model.eval()

    # Create dummy input
    batch_size = 1
    sequence_length = config.get("sequence_length", 64)
    num_keypoints = config.get("num_keypoints", 133)
    keypoint_dim = config.get("keypoint_dim", 3)

    dummy_input = torch.randn(
        batch_size, sequence_length, num_keypoints, keypoint_dim
    )

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        export_path / "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["pose_sequence"],
        output_names=["tokens", "reconstructed_pose"],
        dynamic_axes={
            "pose_sequence": {0: "batch_size", 1: "sequence_length"},
            "tokens": {0: "batch_size"},
            "reconstructed_pose": {0: "batch_size", 1: "sequence_length"}
        }
    )

    # Save configuration
    with open(export_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _export_torchscript(
    model: nn.Module,
    config: Dict[str, Any],
    export_path: Path
) -> None:
    """Export model in TorchScript format."""
    model.eval()

    # Create dummy input for tracing
    batch_size = 1
    sequence_length = config.get("sequence_length", 64)
    num_keypoints = config.get("num_keypoints", 133)
    keypoint_dim = config.get("keypoint_dim", 3)

    dummy_input = torch.randn(
        batch_size, sequence_length, num_keypoints, keypoint_dim
    )

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)

    # Save traced model
    traced_model.save(export_path / "model_traced.pt")

    # Try scripting as well (more robust but may fail for complex models)
    try:
        scripted_model = torch.jit.script(model)
        scripted_model.save(export_path / "model_scripted.pt")
    except Exception as e:
        print(f"Warning: Could not script model: {e}")

    # Save configuration
    with open(export_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_exported_model(export_path: str, format: str = "pytorch") -> tuple:
    """Load exported model.

    Args:
        export_path: Path to exported model
        format: Export format ('pytorch', 'onnx', 'torchscript')

    Returns:
        Tuple of (model, config)
    """
    export_path = Path(export_path)

    # Load configuration
    with open(export_path / "config.json", "r") as f:
        config = json.load(f)

    if format == "pytorch":
        model = torch.load(export_path / "model_full.pth")
        return model, config
    elif format == "torchscript":
        model = torch.jit.load(export_path / "model_traced.pt")
        return model, config
    elif format == "onnx":
        try:
            import onnxruntime
            session = onnxruntime.InferenceSession(str(export_path / "model.onnx"))
            return session, config
        except ImportError:
            raise ImportError("ONNX loading requires 'onnxruntime' package")
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get model summary statistics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": model_size_mb,
        "modules": len(list(model.modules())),
        "layers": len([m for m in model.modules() if len(list(m.children())) == 0])
    }