#!/usr/bin/env python3
"""Prepare pose dataset for tokenizer training."""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pose_tokenizer.datasets.preprocessing import PosePreprocessor


def main():
    parser = argparse.ArgumentParser(description="Prepare pose dataset (npz) for tokenizer training")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with raw npz files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write processed npz files")
    parser.add_argument("--root-relative", action="store_true")
    parser.add_argument("--root-joint", type=int, default=0)
    parser.add_argument("--scale-norm", action="store_true")
    parser.add_argument("--scale-mode", type=str, default="shoulder_width", choices=["shoulder_width", "bone_length_mean"])
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--smooth-polyorder", type=int, default=2)
    parser.add_argument("--interp-missing", action="store_true")
    parser.add_argument("--interp-max-gap", type=int, default=3)
    parser.add_argument("--dims", type=int, default=3, choices=[2, 3])

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = PosePreprocessor(keypoint_dim=args.dims)

    lengths = []
    coord_sum = np.zeros(2, dtype=np.float64)
    coord_sq_sum = np.zeros(2, dtype=np.float64)
    coord_count = 0
    conf_vals = []
    nan_fix_count = 0
    total_samples = 0

    for npz_path in input_dir.rglob("*.npz"):
        data = np.load(npz_path)
        if 'pose' in data:
            pose = data['pose']
        else:
            pose = data['pose_sequence']

        if 'conf' in data and pose.shape[-1] == 2:
            conf = data['conf']
            if conf.ndim == 2:
                conf = conf[..., None]
            pose = np.concatenate([pose, conf], axis=-1)

        nan_fix_count += np.sum(~np.isfinite(pose))

        processed = preprocessor.process_raw_poses(
            pose,
            smooth=args.smooth,
            fill_missing=args.interp_missing,
            interp_max_gap=args.interp_max_gap,
            root_relative=args.root_relative,
            root_joint=args.root_joint,
            scale_norm=args.scale_norm,
            scale_mode=args.scale_mode,
            smooth_window=args.smooth_window,
            smooth_polyorder=args.smooth_polyorder,
        )

        lengths.append(processed.shape[0])
        coords = processed[..., :2].reshape(-1, 2)
        coord_sum += coords.sum(axis=0)
        coord_sq_sum += (coords ** 2).sum(axis=0)
        coord_count += coords.shape[0]

        if processed.shape[-1] > 2:
            conf_vals.append(processed[..., 2].reshape(-1))

        out_path = output_dir / npz_path.name
        np.savez_compressed(out_path, pose=processed)
        total_samples += 1

    lengths = np.array(lengths) if lengths else np.array([0])
    mean_coords = (coord_sum / max(coord_count, 1)).tolist()
    var_coords = (coord_sq_sum / max(coord_count, 1) - np.array(mean_coords) ** 2).tolist()

    stats = {
        "num_samples": int(total_samples),
        "mean_T": float(np.mean(lengths)),
        "p95_T": float(np.percentile(lengths, 95)),
        "coord_mean": mean_coords,
        "coord_var": var_coords,
        "nan_fix_count": int(nan_fix_count),
    }

    if conf_vals:
        conf_concat = np.concatenate(conf_vals)
        stats["conf_stats"] = {
            "mean": float(np.mean(conf_concat)),
            "std": float(np.std(conf_concat)),
            "min": float(np.min(conf_concat)),
            "max": float(np.max(conf_concat))
        }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Processed {total_samples} files. Stats saved to {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
