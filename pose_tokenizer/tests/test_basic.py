"""Basic regression tests for pose_tokenizer core logic."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from pose_tokenizer.datasets.pose_dataset import PoseDataset
from pose_tokenizer.models.encoder import PoseEncoder
from pose_tokenizer.models.decoder import PoseDecoder
from pose_tokenizer.models.quantizer import ResidualVectorQuantizer
from pose_tokenizer.training.trainer import PoseTokenizerTrainer, TrainingConfig


def _write_npz_samples(data_dir: Path, num_files: int = 20) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_files):
        arr = np.random.randn(8, 21, 3).astype(np.float32)
        np.savez(data_dir / f"sample_{i:03d}.npz", pose_sequence=arr)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class _DDPStyleWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _FakeCoreModel:
    def encoder(self, poses, mask):
        return torch.zeros(poses.shape[0], poses.shape[1], 4, device=poses.device)

    def tokenizer(self, encoded):
        idx = torch.zeros((1, encoded.shape[0], encoded.shape[1]), dtype=torch.long, device=encoded.device)
        return encoded, idx, {}

    def decoder(self, quantized, target_shape):
        bsz = quantized.shape[0]
        t = target_shape[0]
        return torch.zeros(bsz, t, 21, 3, device=quantized.device)


class TestPoseTokenizerBasics(unittest.TestCase):
    def test_pose_encoder_forward_shape(self) -> None:
        encoder = PoseEncoder(
            num_keypoints=21,
            keypoint_dim=3,
            hidden_dim=32,
            spatial_layers=2,
            temporal_layers=2,
            dropout=0.0,
            use_temporal=True,
        )
        x = torch.randn(2, 8, 21, 3)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, -2:] = False

        y = encoder(x, mask)

        self.assertEqual(y.shape, (2, 8, 32))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_rvq_encode_decode_shapes_and_losses(self) -> None:
        rvq = ResidualVectorQuantizer(
            num_quantizers=2,
            codebook_size=32,
            embedding_dim=16,
            commitment_cost=0.25,
        )
        x = torch.randn(2, 6, 16)

        quantized, indices, losses = rvq(x)
        decoded = rvq.decode(indices)

        self.assertEqual(quantized.shape, x.shape)
        self.assertEqual(decoded.shape, x.shape)
        self.assertEqual(indices.shape, (2, 2, 6))
        self.assertEqual(losses["vq_loss"].ndim, 0)
        self.assertEqual(losses["perplexity"].ndim, 0)
        self.assertEqual(losses["dead_code_ratio"].ndim, 0)

    def test_pose_decoder_channel_alignment_with_fewer_layers(self) -> None:
        decoder = PoseDecoder(
            num_keypoints=133,
            keypoint_dim=3,
            hidden_dim=128,
            spatial_layers=3,
            temporal_layers=2,
            dropout=0.0,
            use_temporal=True,
        )
        x = torch.randn(2, 64, 128)
        y = decoder(x, target_shape=(64, 133, 3))
        self.assertEqual(y.shape, (2, 64, 133, 3))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_pose_dataset_auto_split_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "pose_data"
            _write_npz_samples(data_dir, num_files=30)

            train_a = PoseDataset(
                data_dir=str(data_dir),
                split="train",
                sequence_length=8,
                window_T=8,
                num_keypoints=21,
                keypoint_dim=3,
                normalize=False,
                augment=False,
            )
            train_b = PoseDataset(
                data_dir=str(data_dir),
                split="train",
                sequence_length=8,
                window_T=8,
                num_keypoints=21,
                keypoint_dim=3,
                normalize=False,
                augment=False,
            )
            val = PoseDataset(
                data_dir=str(data_dir),
                split="val",
                sequence_length=8,
                window_T=8,
                num_keypoints=21,
                keypoint_dim=3,
                normalize=False,
                augment=False,
            )
            test = PoseDataset(
                data_dir=str(data_dir),
                split="test",
                sequence_length=8,
                window_T=8,
                num_keypoints=21,
                keypoint_dim=3,
                normalize=False,
                augment=False,
            )

            train_a_files = [p.name for p in train_a.data_files]
            train_b_files = [p.name for p in train_b.data_files]
            val_files = {p.name for p in val.data_files}
            test_files = {p.name for p in test.data_files}

            self.assertEqual(train_a_files, train_b_files)
            self.assertTrue(set(train_a_files).isdisjoint(val_files))
            self.assertTrue(set(train_a_files).isdisjoint(test_files))
            self.assertTrue(val_files.isdisjoint(test_files))

    def test_pose_sanitize_replaces_non_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "pose_data"
            _write_npz_samples(data_dir, num_files=1)
            ds = PoseDataset(
                data_dir=str(data_dir),
                split="train",
                sequence_length=8,
                window_T=8,
                num_keypoints=21,
                keypoint_dim=3,
                normalize=False,
                augment=False,
            )
            arr = np.zeros((2, 21, 3), dtype=np.float32)
            arr[0, 0, 0] = np.nan
            arr[0, 1, 1] = np.inf
            arr[0, 2, 2] = 5.0
            out = ds._sanitize_pose_seq(arr)
            self.assertTrue(np.isfinite(out).all())
            self.assertLessEqual(out[..., 2].max(), 1.0)

    def test_trainer_saves_checkpoint_only_on_main_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = TrainingConfig(
                distributed=True,
                world_size=2,
                local_rank=1,
                use_amp=False,
                checkpoint_dir=str(Path(temp_dir) / "ckpt"),
                log_dir=str(Path(temp_dir) / "logs"),
                num_keypoints=21,
                keypoint_dim=3,
                hidden_dim=16,
                num_quantizers=2,
                codebook_size=32,
            )
            trainer = PoseTokenizerTrainer(
                model=_TinyModel(),
                config=cfg,
                device=torch.device("cpu"),
            )

            with patch("pose_tokenizer.training.trainer.dist.is_available", return_value=True), \
                 patch("pose_tokenizer.training.trainer.dist.is_initialized", return_value=True), \
                 patch("pose_tokenizer.training.trainer.dist.get_rank", return_value=1):
                trainer.save_checkpoint("rank1_should_not_write.pth")

            self.assertFalse((Path(temp_dir) / "ckpt" / "rank1_should_not_write.pth").exists())

            with patch("pose_tokenizer.training.trainer.dist.is_available", return_value=True), \
                 patch("pose_tokenizer.training.trainer.dist.is_initialized", return_value=True), \
                 patch("pose_tokenizer.training.trainer.dist.get_rank", return_value=0):
                trainer.save_checkpoint("rank0_should_write.pth")

            self.assertTrue((Path(temp_dir) / "ckpt" / "rank0_should_write.pth").exists())

    def test_trainer_skips_non_finite_batch(self) -> None:
        cfg = TrainingConfig(
            use_amp=False,
            checkpoint_dir="checkpoints",
            log_dir="logs",
            num_keypoints=21,
            keypoint_dim=3,
            hidden_dim=16,
            num_quantizers=2,
            codebook_size=32,
        )
        trainer = PoseTokenizerTrainer(
            model=_TinyModel(),
            config=cfg,
            device=torch.device("cpu"),
        )
        trainer._core_model = lambda: _FakeCoreModel()
        trainer.criterion = lambda reconstructed, poses, quant_losses, mask: {
            "total_loss": torch.tensor(float("nan"), device=poses.device),
            "pos_loss": torch.tensor(0.0, device=poses.device),
        }
        batch = {
            "poses": torch.zeros(2, 8, 21, 3),
            "mask": torch.ones(2, 8, dtype=torch.bool),
        }
        out = trainer.train_step(batch)
        self.assertEqual(out.get("skipped_non_finite"), 1.0)

    def test_trainer_core_model_handles_ddp_style_wrapper(self) -> None:
        cfg = TrainingConfig(
            use_amp=False,
            checkpoint_dir="checkpoints",
            log_dir="logs",
            num_keypoints=21,
            keypoint_dim=3,
            hidden_dim=16,
            num_quantizers=2,
            codebook_size=32,
        )
        inner = _TinyModel()
        wrapped = _DDPStyleWrapper(inner)
        trainer = PoseTokenizerTrainer(
            model=wrapped,
            config=cfg,
            device=torch.device("cpu"),
        )
        self.assertIs(trainer._core_model(), inner)


if __name__ == "__main__":
    unittest.main()
