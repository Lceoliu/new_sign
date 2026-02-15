# Uni-Sign Pose Tokenizer Dev Guide

## Data Format
Each sample is an `.npz` file containing:
- `pose`: float32 array of shape `[T, J, D]` (D=2 or 3)
- optional `conf`: float32 array of shape `[T, J]` or `[T, J, 1]`
- optional `fps`: int

During loading, `conf` is concatenated to `pose` if `pose` is 2D.

## Preprocessing Pipeline
`PosePreprocessor.process_raw_poses()` supports the following steps (all optional):
1. NaN/Inf sanitization
2. Missing-value interpolation (`interp_max_gap`)
3. Smoothing (moving average or Savitzky–Golay)
4. Root-relative transform (`root_joint`)
5. Scale normalization (`shoulder_width` or `bone_length_mean`)

## Core Interfaces
- `PoseTokenizerModel.encode(poses, mask)` → `(tokens, lengths)`
- `PoseTokenizerModel.decode(tokens, target_shape)` → `reconstructed_poses`

## Commands
Prepare data:
```bash
python scripts/prepare_dataset.py --input-dir data/raw --output-dir data/processed \
  --root-relative --root-joint 0 --scale-norm --scale-mode shoulder_width \
  --smooth --smooth-window 9 --smooth-polyorder 2 --interp-missing --interp-max-gap 3
```

Train:
```bash
python scripts/train_tokenizer.py --config pose_tokenizer/configs/default.yaml
```

Evaluate:
```bash
python scripts/eval_tokenizer.py --config pose_tokenizer/configs/default.yaml \
  --checkpoint checkpoints/best.pth --split val --output-dir eval_out --visualize
```

Export:
```bash
python scripts/export_tokenizer.py --config pose_tokenizer/configs/default.yaml \
  --checkpoint checkpoints/best.pth --format onnx --output exports/model.onnx
```

## Common Failure Modes
1. Perplexity ~ 1 for a long time (codebook collapse)
   - Reduce LR, increase commitment, ensure EMA updates, check encoder variance.
2. Over-smoothed reconstructions
   - Reduce downsampling/window, increase vel/acc loss weights.
3. 2D tokens encode scale/view instead of motion
   - Ensure root-relative + scale normalization enabled, check smoothing.

## Notes
- `configs/skeleton_definition.json` is used for bone-length loss.
- All loss weights and training hyperparameters should come from config.
