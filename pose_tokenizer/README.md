# Pose Tokenizer

A PyTorch implementation of a pose tokenizer for human pose sequence compression and reconstruction using Residual Vector Quantization (RVQ) and ST-GCN.

## Overview

This project implements a pose tokenizer that can compress human pose sequences into discrete tokens and reconstruct them with high fidelity. The system uses:

- **ST-GCN Encoder**: Spatial-temporal graph convolutional network for pose feature extraction
- **Residual Vector Quantization (RVQ)**: Multi-layer quantization for efficient compression
- **Decoder**: Reconstructs pose sequences from quantized features

## Features

- 🎯 High-quality pose sequence reconstruction
- 🗜️ Efficient compression with configurable compression ratios
- 📊 Comprehensive evaluation metrics and visualizations
- 🚀 Distributed training support
- 📈 Integration with Weights & Biases for experiment tracking
- 🔧 Flexible configuration system
- 📦 Easy export to ONNX and TorchScript formats

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Uni-Sign/pose_tokenizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Prepare Data

Organize your pose data in the following structure:
```
data/
├── processed/
│   ├── train/
│   │   ├── sample_001.npz
│   │   ├── sample_002.npz
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
```

Each `.npz` file should contain:
- `poses`: Array of shape (T, N, D) where T=time, N=keypoints, D=dimensions
- `mask`: Optional boolean mask of shape (T,) for valid frames

### 2. Training

Train with default configuration:
```bash
python scripts/train.py --config configs/default.yaml
```

Train with custom configuration:
```bash
python scripts/train.py --config configs/small.yaml --wandb-project my-project
```

For distributed training:
```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/large.yaml
```

### 3. Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pth \
    --split test \
    --visualize
```

### 4. Export Model

Export to ONNX format:
```bash
python scripts/export.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pth \
    --format onnx \
    --output exports/model.onnx
```

## Configuration

The system uses YAML configuration files. Key parameters:

### Model Architecture
```yaml
num_keypoints: 133        # Number of pose keypoints
keypoint_dim: 3          # Keypoint dimensions (x, y, confidence)
sequence_length: 64      # Input sequence length
hidden_dim: 256          # Hidden feature dimensions
num_quantizers: 4        # Number of RVQ layers
codebook_size: 1024      # Size of each codebook
```

### Training Parameters
```yaml
batch_size: 32
num_epochs: 100
learning_rate: 1e-4
optimizer_type: "adamw"
scheduler_type: "warmup_cosine"
```

### Loss Weights
```yaml
reconstruction_weight: 1.0
quantization_weight: 1.0
perceptual_weight: 0.1
temporal_weight: 0.1
```

See `configs/` directory for example configurations.

## Model Architecture

### ST-GCN Encoder
- Processes pose sequences using spatial-temporal graph convolutions
- Captures both spatial relationships between keypoints and temporal dynamics
- Outputs feature representations for quantization

### Residual Vector Quantization (RVQ)
- Multi-layer quantization for hierarchical compression
- Each layer refines the quantization residual
- Configurable number of quantizers and codebook sizes

### Decoder
- Reconstructs pose sequences from quantized features
- Uses transposed convolutions and upsampling
- Outputs pose coordinates with same dimensions as input

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Reconstruction Quality
- **MSE**: Mean Squared Error between original and reconstructed poses
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **PCK**: Percentage of Correct Keypoints (within threshold)

### Quantization Quality
- **Codebook Utilization**: Percentage of codebook entries used
- **Quantization Error**: Error introduced by quantization
- **Compression Ratio**: Original size vs. compressed size

### Visualizations
- Pose sequence comparisons
- Per-keypoint error analysis
- Codebook usage statistics
- Interactive pose animations

## Directory Structure

```
pose_tokenizer/
├── configs/                 # Configuration files
│   ├── default.yaml
│   ├── small.yaml
│   └── large.yaml
├── pose_tokenizer/          # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── data/               # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── dataloader.py
│   ├── models/             # Model implementations
│   │   ├── __init__.py
│   │   ├── encoder.py      # ST-GCN encoder
│   │   ├── tokenizer.py    # RVQ tokenizer
│   │   ├── decoder.py      # Decoder
│   │   └── model.py        # Complete model
│   ├── training/           # Training utilities
│   │   ├── __init__.py
│   │   ├── losses.py       # Loss functions
│   │   └── trainer.py      # Training loop
│   └── utils/              # Utilities
│       ├── __init__.py
│       └── evaluation.py   # Evaluation and visualization
├── scripts/                # Training and evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── requirements.txt
└── README.md
```

## Advanced Usage

### Custom Data Loading

Implement custom dataset by extending `PoseDataset`:

```python
from pose_tokenizer.data.dataset import PoseDataset

class CustomPoseDataset(PoseDataset):
    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, **kwargs)

    def load_sample(self, file_path):
        # Custom loading logic
        return poses, mask
```

### Custom Loss Functions

Add custom loss components:

```python
from pose_tokenizer.training.losses import PoseTokenizerLoss

class CustomLoss(PoseTokenizerLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_custom_loss(self, pred, target):
        # Custom loss computation
        return loss
```

### Model Customization

Modify model architecture:

```python
from pose_tokenizer.models.model import PoseTokenizerModel

class CustomModel(PoseTokenizerModel):
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers

    def forward(self, x, mask=None):
        # Custom forward pass
        return output
```

## Performance Tips

### Training Optimization
- Use mixed precision training (`use_amp: true`)
- Enable distributed training for multiple GPUs
- Adjust batch size based on GPU memory
- Use gradient accumulation for large effective batch sizes

### Memory Optimization
- Reduce sequence length for memory-constrained environments
- Use smaller model configurations (see `configs/small.yaml`)
- Enable gradient checkpointing for very deep models

### Speed Optimization
- Use more workers for data loading (`num_workers`)
- Enable pin memory (`pin_memory: true`)
- Use compiled models with `torch.compile()` (PyTorch 2.0+)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow training**
   - Increase number of data loading workers
   - Use distributed training
   - Check data loading bottlenecks

3. **Poor reconstruction quality**
   - Increase model capacity (hidden_dim, num_layers)
   - Adjust loss weights
   - Use more quantizers or larger codebooks

4. **Low codebook utilization**
   - Adjust commitment cost
   - Use codebook reset strategies
   - Increase model capacity

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Use smaller configurations for debugging:
```bash
python scripts/train.py --config configs/small.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pose-tokenizer,
  title={Pose Tokenizer: Efficient Human Pose Sequence Compression},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/pose-tokenizer}
}
```

## Acknowledgments

- ST-GCN implementation based on [ST-GCN paper](https://arxiv.org/abs/1801.07455)
- Vector Quantization based on [VQ-VAE](https://arxiv.org/abs/1711.00937)
- Residual Vector Quantization based on [RVQ paper](https://arxiv.org/abs/2107.03312)