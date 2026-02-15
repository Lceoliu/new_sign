#!/usr/bin/env python3
"""Generate training visualizations for a pose_tokenizer run directory.

Usage:
  python scripts/visualize_run_results.py --run-dir runs/20260215_005410
"""

import argparse
import json
import random
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")

import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_tokenizer.config import PoseTokenizerConfig
from pose_tokenizer.datasets.pose_dataset import PoseDataset
from pose_tokenizer.models.model import PoseTokenizerModel
from pose_tokenizer.models.encoder import MediaPipeGraph
from body_config import WholeBodyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pose_tokenizer run results")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory, e.g. runs/20260215_005410")
    parser.add_argument("--config", type=str, default=None, help="Optional config path override")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir (default: <run-dir>/visualizations)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Local cache dir for file lists/norm stats (default: <run-dir>/cache)")
    parser.add_argument("--num-samples", type=int, default=15, help="Samples per split (train/test)")
    parser.add_argument("--max-frames", type=int, default=96, help="Max frames per video")
    parser.add_argument("--fps", type=int, default=12, help="Video fps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--invert-y", action="store_true", help="Invert y-axis in visualization")
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for future use")
    return parser.parse_args()


def load_config(run_dir: Path, config_override: Optional[str]) -> PoseTokenizerConfig:
    if config_override:
        return PoseTokenizerConfig.from_yaml(config_override)

    resolved_json = run_dir / "resolved_config.json"
    if resolved_json.exists():
        with resolved_json.open("r") as f:
            raw = json.load(f)
        valid = {f.name for f in fields(PoseTokenizerConfig)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        return PoseTokenizerConfig(**filtered)

    raise FileNotFoundError(
        f"No resolved config found at {resolved_json}. Use --config to provide one."
    )


def resolve_checkpoint(run_dir: Path, checkpoint_override: Optional[str]) -> Optional[Path]:
    if checkpoint_override:
        ckpt = Path(checkpoint_override)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
        return ckpt

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    for name in ("best.pth", "final.pth"):
        p = ckpt_dir / name
        if p.exists():
            return p

    candidates = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_scalars(log_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        events = acc.Scalars(tag)
        if not events:
            continue
        steps = np.array([e.step for e in events], dtype=np.int64)
        values = np.array([e.value for e in events], dtype=np.float64)
        out[tag] = (steps, values)
    return out


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 2500) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, num=max_points).astype(np.int64)
    return x[idx], y[idx]


def plot_training_curves(scalars: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path) -> None:
    if not scalars:
        print("[WARN] No TensorBoard scalar events found. Skipping scalar plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    loss_tags = [
        "epoch/train_total_loss",
        "epoch/val_total_loss",
        "train_step/loss_total",
        "train_step/quant_vq_loss",
        "train_step/pos_loss",
        "train_step/vel_loss",
        "train_step/acc_loss",
        "train_step/bone_loss",
    ]
    cb_tags = [
        "train_step/quant_perplexity",
        "train_step/quant_dead_code_ratio",
        "train_step/skipped_non_finite",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    has_loss = False
    for tag in loss_tags:
        if tag not in scalars:
            continue
        x, y = scalars[tag]
        x, y = downsample_xy(x, y)
        axes[0].plot(x, y, label=tag)
        has_loss = True
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Step / Epoch")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    if has_loss:
        axes[0].legend(loc="best")
    else:
        axes[0].text(0.5, 0.5, "No loss tags found", ha="center", va="center")

    has_cb = False
    for tag in cb_tags:
        if tag not in scalars:
            continue
        x, y = scalars[tag]
        x, y = downsample_xy(x, y)
        axes[1].plot(x, y, label=tag)
        has_cb = True
    axes[1].set_title("Codebook / Stability Curves")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)
    if has_cb:
        axes[1].legend(loc="best")
    else:
        axes[1].text(0.5, 0.5, "No codebook tags found", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    scalar_json = {tag: {"num_points": int(len(vals[0]))} for tag, vals in scalars.items()}
    with (out_dir / "scalar_tags.json").open("w") as f:
        json.dump(scalar_json, f, indent=2)


def load_model(config: PoseTokenizerConfig, checkpoint_path: Path, device: torch.device) -> PoseTokenizerModel:
    model = PoseTokenizerModel(config)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_dataset(config: PoseTokenizerConfig, split: str) -> PoseDataset:
    return PoseDataset(
        data_dir=config.data_dir,
        split=split,
        sequence_length=config.sequence_length,
        clip_mode="full",
        window_T=config.window_T,
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        augment=False,
        normalize=config.normalize_poses,
        normalize_by_wh=config.normalize_by_wh,
        log_load_every=0,
    )


def pick_indices(n_total: int, n_pick: int, rng: random.Random) -> List[int]:
    if n_total <= 0:
        return []
    n_pick = min(n_total, n_pick)
    return rng.sample(range(n_total), n_pick)


def compute_codebook_stats(counts: np.ndarray) -> Dict[str, np.ndarray]:
    # counts: [Q, C]
    totals = counts.sum(axis=1, keepdims=True)
    used = (counts > 0).sum(axis=1)
    codebook_size = counts.shape[1]
    utilization = used / max(codebook_size, 1)
    dead_ratio = 1.0 - utilization

    perplexity = np.zeros(counts.shape[0], dtype=np.float64)
    for q in range(counts.shape[0]):
        if totals[q, 0] <= 0:
            perplexity[q] = 0.0
            continue
        probs = counts[q] / totals[q, 0]
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        perplexity[q] = float(np.exp(entropy))

    return {
        "utilization": utilization,
        "dead_ratio": dead_ratio,
        "perplexity": perplexity,
    }


def plot_codebook_summary(stats_by_split: Dict[str, Dict[str, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = list(stats_by_split.keys())
    if not splits:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["utilization", "dead_ratio", "perplexity"]
    titles = ["Codebook Utilization", "Codebook Dead Ratio", "Codebook Perplexity"]

    for ax, metric, title in zip(axes, metrics, titles):
        for split in splits:
            vals = stats_by_split[split][metric]
            ax.plot(np.arange(len(vals)), vals, marker="o", label=split)
        ax.set_title(title)
        ax.set_xlabel("Quantizer Layer")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_dir / "codebook_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "codebook_summary.json").open("w") as f:
        serializable = {
            split: {k: v.tolist() for k, v in stats.items()} for split, stats in stats_by_split.items()
        }
        json.dump(serializable, f, indent=2)


def _normalize_xy_for_draw(
    orig_xy: np.ndarray, recon_xy: np.ndarray, pad: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.concatenate([orig_xy.reshape(-1, 2), recon_xy.reshape(-1, 2)], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return orig_xy.copy(), recon_xy.copy()

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    min_xy = min_xy - pad * span
    max_xy = max_xy + pad * span
    span = np.maximum(max_xy - min_xy, 1e-6)

    o = (orig_xy - min_xy) / span
    r = (recon_xy - min_xy) / span
    return np.clip(o, 0.0, 1.0), np.clip(r, 0.0, 1.0)


def _draw_pose_panel(
    canvas: np.ndarray,
    pose_xy: np.ndarray,
    connections: List[Tuple[int, int]],
    x0: int,
    panel_w: int,
    panel_h: int,
    color: Tuple[int, int, int],
    invert_y: bool = False,
) -> None:
    n = pose_xy.shape[0]
    px = (pose_xy[:, 0] * (panel_w - 1)).astype(np.int32) + x0
    if invert_y:
        py = ((1.0 - pose_xy[:, 1]) * (panel_h - 1)).astype(np.int32)
    else:
        py = (pose_xy[:, 1] * (panel_h - 1)).astype(np.int32)

    for i, j in connections:
        if i >= n or j >= n:
            continue
        cv2.line(canvas, (int(px[i]), int(py[i])), (int(px[j]), int(py[j])), color, 1, cv2.LINE_AA)
    for i in range(n):
        cv2.circle(canvas, (int(px[i]), int(py[i])), 2, color, -1, cv2.LINE_AA)


def write_comparison_video(
    original_xy: np.ndarray,
    recon_xy: np.ndarray,
    connections: List[Tuple[int, int]],
    out_path: Path,
    fps: int,
    invert_y: bool = False,
) -> None:
    t = min(original_xy.shape[0], recon_xy.shape[0])
    original_xy = original_xy[:t]
    recon_xy = recon_xy[:t]
    original_xy, recon_xy = _normalize_xy_for_draw(original_xy, recon_xy)

    panel_w = 480
    panel_h = 480
    width = panel_w * 2
    height = panel_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    for i in range(t):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        _draw_pose_panel(
            frame, original_xy[i], connections, 0, panel_w, panel_h, (255, 0, 0), invert_y=invert_y
        )
        _draw_pose_panel(
            frame, recon_xy[i], connections, panel_w, panel_w, panel_h, (0, 0, 255), invert_y=invert_y
        )

        cv2.putText(frame, "Original", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, "Reconstructed", (panel_w + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {i + 1}/{t}", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()


def run_reconstruction_visualization(
    model: PoseTokenizerModel,
    config: PoseTokenizerConfig,
    device: torch.device,
    out_dir: Path,
    num_samples: int,
    max_frames: int,
    fps: int,
    seed: int,
    invert_y: bool = False,
) -> None:
    rng = random.Random(seed)
    if int(config.num_keypoints) == 133:
        connections = [tuple(e) for e in WholeBodyConfig().get_full_skeleton()]
    else:
        connections = [tuple(e) for e in MediaPipeGraph(config.num_keypoints).edges]
    connections = [
        (int(i), int(j))
        for i, j in connections
        if 0 <= int(i) < int(config.num_keypoints) and 0 <= int(j) < int(config.num_keypoints)
    ]
    codebook_size = int(config.codebook_size)
    num_quantizers = int(config.num_quantizers)

    stats_by_split: Dict[str, Dict[str, np.ndarray]] = {}
    summary: Dict[str, Dict[str, float]] = {}

    for split in ("train", "test"):
        dataset = build_dataset(config, split)
        indices = pick_indices(len(dataset), num_samples, rng)
        if not indices:
            print(f"[WARN] No samples found for split={split}.")
            continue

        split_dir = out_dir / "videos" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        counts = np.zeros((num_quantizers, codebook_size), dtype=np.int64)

        rec_errors = []
        with torch.no_grad():
            for k, idx in enumerate(indices):
                sample = dataset[idx]
                poses = sample["poses"]  # [T,N,D]
                mask = sample["mask"]  # [T]
                valid_t = int(mask.sum().item()) if mask.numel() > 0 else int(poses.shape[0])
                valid_t = max(valid_t, 1)
                t = min(valid_t, int(max_frames))
                poses = poses[:t]
                mask = mask[:t]

                poses_b = poses.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device)

                outputs = model(poses_b, mask_b)
                recon = outputs["reconstructed"]
                token_idx = outputs["indices"]

                orig_np = poses_b[0, :t].detach().cpu().numpy()
                recon_np = recon[0, :t].detach().cpu().numpy()
                orig_xy = orig_np[..., :2]
                recon_xy = recon_np[..., :2]
                orig_xy = np.nan_to_num(orig_xy, nan=0.0, posinf=1e4, neginf=-1e4)
                recon_xy = np.nan_to_num(recon_xy, nan=0.0, posinf=1e4, neginf=-1e4)

                err = float(np.mean(np.abs(orig_xy - recon_xy)))
                rec_errors.append(err)

                token_idx_np = token_idx.detach().cpu().numpy()  # [Q,B,T] typically
                if token_idx_np.ndim >= 3:
                    token_idx_np = token_idx_np[:, 0, :t]
                elif token_idx_np.ndim == 2:
                    token_idx_np = token_idx_np[:, :t]
                else:
                    token_idx_np = token_idx_np.reshape(num_quantizers, -1)
                for q in range(min(num_quantizers, token_idx_np.shape[0])):
                    ids = token_idx_np[q].reshape(-1)
                    ids = ids[(ids >= 0) & (ids < codebook_size)]
                    if ids.size > 0:
                        binc = np.bincount(ids.astype(np.int64), minlength=codebook_size)
                        counts[q] += binc[:codebook_size]

                vid_name = f"{split}_sample_{k:02d}_idx_{idx}.mp4"
                write_comparison_video(
                    original_xy=orig_xy,
                    recon_xy=recon_xy,
                    connections=connections,
                    out_path=split_dir / vid_name,
                    fps=fps,
                    invert_y=invert_y,
                )
                print(f"[INFO] Saved video: {split_dir / vid_name}")

        stats_by_split[split] = compute_codebook_stats(counts)
        summary[split] = {
            "num_samples": float(len(indices)),
            "mean_abs_reconstruction_error": float(np.mean(rec_errors)) if rec_errors else 0.0,
        }

    if stats_by_split:
        plot_codebook_summary(stats_by_split, out_dir / "plots")
    with (out_dir / "reconstruction_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (run_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Force dataset side artifacts to local writable path.
    os.environ["POSE_FILE_LIST_CACHE_DIR"] = str(cache_dir)
    os.environ["NORM_STATS_PATH"] = str(cache_dir / "norm_stats.json")

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Cache dir: {cache_dir}")

    config = load_config(run_dir, args.config)
    print(f"[INFO] Loaded config. data_dir={config.data_dir}")

    scalars = load_scalars(run_dir / "logs")
    plot_training_curves(scalars, output_dir / "plots")
    print("[INFO] Scalar plots generated.")

    ckpt = resolve_checkpoint(run_dir, args.checkpoint)
    if ckpt is None:
        print("[WARN] No checkpoint found. Skipped reconstruction videos.")
        return

    device = pick_device(args.device)
    model = load_model(config, ckpt, device)
    print(f"[INFO] Loaded checkpoint: {ckpt}")
    print(f"[INFO] Device: {device}")

    run_reconstruction_visualization(
        model=model,
        config=config,
        device=device,
        out_dir=output_dir,
        num_samples=args.num_samples,
        max_frames=args.max_frames,
        fps=args.fps,
        seed=args.seed,
        invert_y=args.invert_y,
    )
    print("[INFO] Visualization complete.")


if __name__ == "__main__":
    main()
