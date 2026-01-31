"""Evaluation metrics and visualization for pose tokenizer."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class PoseMetrics:
    """Evaluation metrics for pose reconstruction and tokenization."""

    def __init__(self, num_keypoints: int = 133):
        """Initialize pose metrics.

        Args:
            num_keypoints: Number of keypoints
        """
        self.num_keypoints = num_keypoints
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.total_samples = 0
        self.total_mse = 0.0
        self.total_mae = 0.0
        self.total_pck = 0.0
        self.keypoint_errors = np.zeros(self.num_keypoints)
        self.keypoint_counts = np.zeros(self.num_keypoints)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """Update metrics with batch predictions.

        Args:
            pred: Predicted poses (B, T, N, D)
            target: Target poses (B, T, N, D)
            mask: Valid frame mask (B, T)
        """
        B, T, N, D = pred.shape

        # Convert to numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.ones((B, T))

        # Extract positions (ignore confidence if present)
        if D == 3:
            pred_pos = pred_np[..., :2]
            target_pos = target_np[..., :2]
        else:
            pred_pos = pred_np
            target_pos = target_np

        # Apply mask
        valid_mask = mask_np[..., None, None]  # (B, T, 1, 1)
        pred_pos = pred_pos * valid_mask
        target_pos = target_pos * valid_mask

        # Compute MSE
        mse = np.mean((pred_pos - target_pos) ** 2)
        self.total_mse += mse * np.sum(mask_np)

        # Compute MAE
        mae = np.mean(np.abs(pred_pos - target_pos))
        self.total_mae += mae * np.sum(mask_np)

        # Compute PCK (Percentage of Correct Keypoints)
        distances = np.sqrt(np.sum((pred_pos - target_pos) ** 2, axis=-1))  # (B, T, N)
        pck_threshold = 0.05  # 5% of image size (normalized coordinates)
        correct_keypoints = distances < pck_threshold
        pck = np.mean(correct_keypoints * mask_np[..., None])
        self.total_pck += pck * np.sum(mask_np)

        # Per-keypoint errors
        for n in range(N):
            keypoint_error = np.mean(distances[..., n] * mask_np)
            keypoint_count = np.sum(mask_np)
            self.keypoint_errors[n] += keypoint_error * keypoint_count
            self.keypoint_counts[n] += keypoint_count

        self.total_samples += np.sum(mask_np)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary of metrics
        """
        if self.total_samples == 0:
            return {}

        metrics = {
            'mse': self.total_mse / self.total_samples,
            'mae': self.total_mae / self.total_samples,
            'pck': self.total_pck / self.total_samples,
            'rmse': np.sqrt(self.total_mse / self.total_samples)
        }

        # Per-keypoint metrics
        valid_keypoints = self.keypoint_counts > 0
        if np.any(valid_keypoints):
            avg_keypoint_errors = np.zeros(self.num_keypoints)
            avg_keypoint_errors[valid_keypoints] = (
                self.keypoint_errors[valid_keypoints] / self.keypoint_counts[valid_keypoints]
            )
            metrics['keypoint_errors'] = avg_keypoint_errors.tolist()
            metrics['avg_keypoint_error'] = np.mean(avg_keypoint_errors[valid_keypoints])

        return metrics


class QuantizationMetrics:
    """Metrics for vector quantization quality."""

    def __init__(self, num_quantizers: int, codebook_size: int):
        """Initialize quantization metrics.

        Args:
            num_quantizers: Number of quantization layers
            codebook_size: Size of each codebook
        """
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.total_samples = 0
        self.codebook_usage = np.zeros((self.num_quantizers, self.codebook_size))
        self.quantization_errors = []

    def update(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor,
        indices: torch.Tensor
    ):
        """Update metrics with batch data.

        Args:
            original: Original features before quantization
            quantized: Quantized features
            indices: Quantization indices (num_quantizers, B, ...)
        """
        B = original.shape[0]

        # Quantization error
        quant_error = torch.mean((original - quantized) ** 2).item()
        self.quantization_errors.append(quant_error)

        # Codebook usage
        indices_np = indices.detach().cpu().numpy()
        for q in range(self.num_quantizers):
            layer_indices = indices_np[q].flatten()
            for idx in layer_indices:
                if 0 <= idx < self.codebook_size:
                    self.codebook_usage[q, idx] += 1

        self.total_samples += B

    def compute(self) -> Dict[str, Any]:
        """Compute final metrics.

        Returns:
            Dictionary of metrics
        """
        if self.total_samples == 0:
            return {}

        metrics = {
            'avg_quantization_error': np.mean(self.quantization_errors),
            'std_quantization_error': np.std(self.quantization_errors)
        }

        # Codebook utilization
        for q in range(self.num_quantizers):
            usage = self.codebook_usage[q]
            total_usage = np.sum(usage)
            if total_usage > 0:
                # Percentage of codebook used
                used_codes = np.sum(usage > 0)
                utilization = used_codes / self.codebook_size

                # Entropy of usage distribution
                prob_dist = usage / total_usage
                prob_dist = prob_dist[prob_dist > 0]  # Remove zeros
                entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-8))

                metrics[f'codebook_{q}_utilization'] = utilization
                metrics[f'codebook_{q}_entropy'] = entropy
                metrics[f'codebook_{q}_usage'] = usage.tolist()

        return metrics


class PoseVisualizer:
    """Visualization utilities for pose data."""

    def __init__(self, num_keypoints: int = 133):
        """Initialize visualizer.

        Args:
            num_keypoints: Number of keypoints
        """
        self.num_keypoints = num_keypoints
        self.connections = self._get_pose_connections()

    def _get_pose_connections(self) -> List[Tuple[int, int]]:
        """Get pose skeleton connections."""
        # MediaPipe pose connections (simplified)
        connections = [
            # Face outline (first 10 points)
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
            # Body connections
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32)
        ]

        # Filter connections that are within keypoint range
        valid_connections = [
            (i, j) for i, j in connections
            if i < self.num_keypoints and j < self.num_keypoints
        ]

        return valid_connections

    def plot_pose_sequence(
        self,
        poses: np.ndarray,
        title: str = "Pose Sequence",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot pose sequence as animation frames.

        Args:
            poses: Pose sequence (T, N, 2)
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        T, N, _ = poses.shape

        # Create subplot grid
        cols = min(8, T)
        rows = (T + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(title, fontsize=16)

        for t in range(T):
            row = t // cols
            col = t % cols
            ax = axes[row, col]

            # Plot keypoints
            x, y = poses[t, :, 0], poses[t, :, 1]
            ax.scatter(x, y, c='red', s=20, alpha=0.7)

            # Plot connections
            for i, j in self.connections:
                if i < N and j < N:
                    ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.5, linewidth=1)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'Frame {t}')
            ax.axis('off')

        # Hide empty subplots
        for t in range(T, rows * cols):
            row = t // cols
            col = t % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_pose_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        frame_idx: int = 0,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison between original and reconstructed poses.

        Args:
            original: Original poses (T, N, 2)
            reconstructed: Reconstructed poses (T, N, 2)
            frame_idx: Frame index to visualize
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Original pose
        x_orig, y_orig = original[frame_idx, :, 0], original[frame_idx, :, 1]
        ax1.scatter(x_orig, y_orig, c='blue', s=30, alpha=0.7, label='Keypoints')
        for i, j in self.connections:
            if i < len(x_orig) and j < len(x_orig):
                ax1.plot([x_orig[i], x_orig[j]], [y_orig[i], y_orig[j]],
                        'b-', alpha=0.5, linewidth=2)
        ax1.set_title('Original')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Reconstructed pose
        x_recon, y_recon = reconstructed[frame_idx, :, 0], reconstructed[frame_idx, :, 1]
        ax2.scatter(x_recon, y_recon, c='red', s=30, alpha=0.7, label='Keypoints')
        for i, j in self.connections:
            if i < len(x_recon) and j < len(x_recon):
                ax2.plot([x_recon[i], x_recon[j]], [y_recon[i], y_recon[j]],
                        'r-', alpha=0.5, linewidth=2)
        ax2.set_title('Reconstructed')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # Overlay comparison
        ax3.scatter(x_orig, y_orig, c='blue', s=30, alpha=0.7, label='Original')
        ax3.scatter(x_recon, y_recon, c='red', s=30, alpha=0.7, label='Reconstructed')

        # Draw error lines
        for i in range(min(len(x_orig), len(x_recon))):
            ax3.plot([x_orig[i], x_recon[i]], [y_orig[i], y_recon[i]],
                    'gray', alpha=0.5, linewidth=1)

        ax3.set_title('Overlay Comparison')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_aspect('equal')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_keypoint_errors(
        self,
        errors: np.ndarray,
        keypoint_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot per-keypoint reconstruction errors.

        Args:
            errors: Per-keypoint errors (N,)
            keypoint_names: Names of keypoints
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(errors))
        bars = ax.bar(x, errors, alpha=0.7)

        # Color bars by error magnitude
        colors = plt.cm.viridis(errors / np.max(errors))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Keypoint Index')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Per-Keypoint Reconstruction Errors')

        if keypoint_names:
            ax.set_xticks(x[::5])  # Show every 5th label to avoid crowding
            ax.set_xticklabels([keypoint_names[i] for i in x[::5]], rotation=45)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                  norm=plt.Normalize(vmin=0, vmax=np.max(errors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Error Magnitude')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_codebook_usage(
        self,
        usage_stats: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot codebook usage statistics.

        Args:
            usage_stats: Dictionary of usage statistics per quantizer
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        num_quantizers = len(usage_stats)
        fig, axes = plt.subplots(2, (num_quantizers + 1) // 2, figsize=(15, 8))
        if num_quantizers == 1:
            axes = [axes]
        elif num_quantizers <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, (layer_name, usage) in enumerate(usage_stats.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            # Plot usage histogram
            ax.bar(range(len(usage)), usage, alpha=0.7)
            ax.set_title(f'{layer_name} Usage')
            ax.set_xlabel('Codebook Index')
            ax.set_ylabel('Usage Count')

            # Add statistics
            used_codes = np.sum(usage > 0)
            total_codes = len(usage)
            utilization = used_codes / total_codes
            ax.text(0.02, 0.98, f'Utilization: {utilization:.2%}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Hide unused subplots
        for i in range(num_quantizers, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def create_interactive_pose_plot(
        self,
        poses: np.ndarray,
        title: str = "Interactive Pose Sequence",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive pose sequence plot using Plotly.

        Args:
            poses: Pose sequence (T, N, 2)
            title: Plot title
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        T, N, _ = poses.shape

        # Create frames for animation
        frames = []
        for t in range(T):
            frame_data = []

            # Keypoints
            frame_data.append(
                go.Scatter(
                    x=poses[t, :, 0],
                    y=poses[t, :, 1],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name='Keypoints',
                    showlegend=(t == 0)
                )
            )

            # Connections
            for i, j in self.connections:
                if i < N and j < N:
                    frame_data.append(
                        go.Scatter(
                            x=[poses[t, i, 0], poses[t, j, 0]],
                            y=[poses[t, i, 1], poses[t, j, 1]],
                            mode='lines',
                            line=dict(color='blue', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )

            frames.append(go.Frame(data=frame_data, name=str(t)))

        # Create initial plot
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )

        # Add animation controls
        fig.update_layout(
            title=title,
            xaxis=dict(range=[0, 1], title='X'),
            yaxis=dict(range=[0, 1], title='Y', scaleanchor='x'),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[str(t)], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': str(t),
                        'method': 'animate'
                    } for t in range(T)
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Frame: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )

        if save_path:
            fig.write_html(save_path)

        return fig


class EvaluationSuite:
    """Complete evaluation suite for pose tokenizer."""

    def __init__(
        self,
        num_keypoints: int = 133,
        num_quantizers: int = 4,
        codebook_size: int = 1024
    ):
        """Initialize evaluation suite.

        Args:
            num_keypoints: Number of keypoints
            num_quantizers: Number of quantization layers
            codebook_size: Size of each codebook
        """
        self.pose_metrics = PoseMetrics(num_keypoints)
        self.quant_metrics = QuantizationMetrics(num_quantizers, codebook_size)
        self.visualizer = PoseVisualizer(num_keypoints)

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.

        Args:
            model: Model to evaluate
            dataloader: Data loader
            device: Device to run evaluation on
            save_dir: Directory to save visualizations

        Returns:
            Dictionary of evaluation results
        """
        model.eval()

        # Reset metrics
        self.pose_metrics.reset()
        self.quant_metrics.reset()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                poses = batch['poses'].to(device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(device)

                # Forward pass
                encoded = model.encoder(poses, mask)
                quantized, indices, _ = model.tokenizer(encoded)
                reconstructed = model.decoder(quantized, poses.shape[1:])

                # Update metrics
                self.pose_metrics.update(reconstructed, poses, mask)
                self.quant_metrics.update(encoded, quantized, indices)

                # Store samples for visualization
                if batch_idx < 5:  # Save first few batches
                    all_predictions.append(reconstructed.cpu().numpy())
                    all_targets.append(poses.cpu().numpy())

        # Compute final metrics
        pose_results = self.pose_metrics.compute()
        quant_results = self.quant_metrics.compute()

        # Combine results
        results = {
            'pose_metrics': pose_results,
            'quantization_metrics': quant_results
        }

        # Generate visualizations
        if save_dir and all_predictions:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Plot pose comparisons
            for i, (pred, target) in enumerate(zip(all_predictions[:3], all_targets[:3])):
                fig = self.visualizer.plot_pose_comparison(
                    target[0], pred[0], frame_idx=0,
                    save_path=save_path / f'comparison_{i}.png'
                )
                plt.close(fig)

            # Plot keypoint errors
            if 'keypoint_errors' in pose_results:
                fig = self.visualizer.plot_keypoint_errors(
                    np.array(pose_results['keypoint_errors']),
                    save_path=save_path / 'keypoint_errors.png'
                )
                plt.close(fig)

            # Plot codebook usage
            usage_stats = {}
            for key, value in quant_results.items():
                if key.endswith('_usage'):
                    layer_name = key.replace('_usage', '')
                    usage_stats[layer_name] = np.array(value)

            if usage_stats:
                fig = self.visualizer.plot_codebook_usage(
                    usage_stats,
                    save_path=save_path / 'codebook_usage.png'
                )
                plt.close(fig)

            # Save results to JSON
            with open(save_path / 'evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        return results

    def generate_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate evaluation report.

        Args:
            results: Evaluation results
            save_path: Path to save report

        Returns:
            Report string
        """
        report_lines = [
            "# Pose Tokenizer Evaluation Report",
            "",
            "## Pose Reconstruction Metrics",
            ""
        ]

        pose_metrics = results.get('pose_metrics', {})
        for metric, value in pose_metrics.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"- **{metric.upper()}**: {value:.6f}")

        report_lines.extend([
            "",
            "## Quantization Metrics",
            ""
        ])

        quant_metrics = results.get('quantization_metrics', {})
        for metric, value in quant_metrics.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"- **{metric}**: {value:.6f}")

        # Codebook utilization summary
        utilization_metrics = [k for k in quant_metrics.keys() if 'utilization' in k]
        if utilization_metrics:
            report_lines.extend([
                "",
                "## Codebook Utilization",
                ""
            ])
            for metric in utilization_metrics:
                value = quant_metrics[metric]
                report_lines.append(f"- **{metric}**: {value:.2%}")

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report