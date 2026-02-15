#!/usr/bin/env python3
"""Training script for pose tokenizer."""

import os
import sys
import argparse
import logging
import re
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from pathlib import Path
import json
from datetime import datetime
import time
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose_tokenizer.config import PoseTokenizerConfig
from pose_tokenizer.data.dataset import PoseDataset
from pose_tokenizer.datasets.data_loader import collate_pose_sequences
from pose_tokenizer.models.model import PoseTokenizerModel
from pose_tokenizer.training.trainer import PoseTokenizerTrainer, TrainingConfig
from pose_tokenizer.utils.evaluation import EvaluationSuite


def setup_logging(log_dir: str, rank: int = 0):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = log_dir / f'train_rank_{rank}.log'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        return rank, world_size, local_rank, device
    else:
        return 0, 1, 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _is_default_path(path_value: str, default_name: str) -> bool:
    if path_value is None:
        return True
    normalized = Path(path_value).as_posix().rstrip("/")
    return normalized in ("", default_name, f"./{default_name}")


def _resolve_run_paths(config: PoseTokenizerConfig, rank: int, world_size: int) -> Path:
    """Resolve run-specific output paths, synchronized across DDP ranks."""
    run_timestamp = os.environ.get("POSE_RUN_TIMESTAMP")
    if not run_timestamp and rank == 0:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if world_size > 1:
        obj = [run_timestamp]
        dist.broadcast_object_list(obj, src=0)
        run_timestamp = obj[0]

    run_dir = Path("runs") / str(run_timestamp)
    config.run_dir = str(run_dir)

    if _is_default_path(config.log_dir, "logs"):
        config.log_dir = str(run_dir / "logs")
    if _is_default_path(config.checkpoint_dir, "checkpoints"):
        config.checkpoint_dir = str(run_dir / "checkpoints")
    if _is_default_path(config.output_dir, "outputs"):
        config.output_dir = str(run_dir / "outputs")

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        with (run_dir / "resolved_config.json").open("w") as f:
            json.dump(config.__dict__, f, indent=2)

    if world_size > 1:
        dist.barrier()

    return run_dir


def _configure_local_dataset_cache(run_dir: Path, rank: int, world_size: int) -> None:
    """Force dataset cache/stat files to local writable paths (avoid NAS permission issues)."""
    cache_dir = run_dir / "cache"
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    os.environ.setdefault("POSE_FILE_LIST_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("NORM_STATS_PATH", str(cache_dir / "norm_stats.json"))


def _infer_run_timestamp_from_path(path_str: str) -> Optional[str]:
    path = Path(path_str).expanduser()
    parts = path.parts
    for idx, token in enumerate(parts):
        if token == "runs" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _pick_resume_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None

    latest = ckpt_dir / "latest.pth"
    if latest.exists():
        return latest

    epoch_candidates = []
    for p in ckpt_dir.glob("epoch_*.pth"):
        m = re.match(r"epoch_(\d+)\.pth$", p.name)
        if m:
            epoch_candidates.append((int(m.group(1)), p))
    if epoch_candidates:
        epoch_candidates.sort(key=lambda x: x[0], reverse=True)
        return epoch_candidates[0][1]

    final = ckpt_dir / "final.pth"
    if final.exists():
        return final

    best = ckpt_dir / "best.pth"
    if best.exists():
        return best

    any_ckpt = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return any_ckpt[0] if any_ckpt else None


def _find_latest_run_timestamp() -> Optional[str]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        ckpt = _pick_resume_checkpoint(d / "checkpoints")
        if ckpt is not None:
            candidates.append((d.stat().st_mtime, d.name))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _resolve_resume_checkpoint(resume_arg: Optional[str], run_dir: Path) -> Optional[Path]:
    if resume_arg is None:
        return None

    if resume_arg in {"auto", "latest"}:
        return _pick_resume_checkpoint(run_dir / "checkpoints")

    resume_path = Path(resume_arg).expanduser()
    if resume_path.is_dir():
        found = _pick_resume_checkpoint(resume_path)
        if found is None:
            raise FileNotFoundError(f"No checkpoint found in directory: {resume_path}")
        return found

    if resume_path.exists():
        return resume_path

    candidate = run_dir / "checkpoints" / resume_arg
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Resume checkpoint not found: {resume_arg}")


def create_dataloaders(config: PoseTokenizerConfig, rank: int = 0, world_size: int = 1):
    """Create training and validation dataloaders."""

    # Training dataset
    train_dataset = PoseDataset(
        data_dir=config.data_dir,
        split='train',
        sequence_length=config.sequence_length,
        clip_mode=config.clip_mode,
        window_T=config.window_T,
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        augment=config.augment_data,
        normalize=config.normalize_poses,
        normalize_by_wh=config.normalize_by_wh,
        log_load_every=config.load_log_every
    )

    # Validation dataset
    val_dataset = PoseDataset(
        data_dir=config.data_dir,
        split='val',
        sequence_length=config.sequence_length,
        clip_mode="full",
        window_T=config.window_T,
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        augment=False,
        normalize=config.normalize_poses,
        normalize_by_wh=config.normalize_by_wh,
        log_load_every=config.load_log_every
    )

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

    # Create dataloaders
    eval_batch_size = config.eval_batch_size if config.eval_batch_size > 0 else config.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_pose_sequences,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_pose_sequences,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, train_sampler


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train pose tokenizer')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint path/dir, or "auto"/"latest"')
    parser.add_argument('--run-timestamp', type=str, default=None,
                       help='Run timestamp under runs/<timestamp> (helps deterministic resume)')
    parser.add_argument('--wandb-project', type=str, default='pose-tokenizer',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation')

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()

    # Determine run timestamp before resolving run paths.
    run_timestamp = args.run_timestamp or os.environ.get("POSE_RUN_TIMESTAMP")
    if rank == 0 and run_timestamp is None:
        if args.resume and args.resume not in {"auto", "latest"}:
            run_timestamp = _infer_run_timestamp_from_path(args.resume)
        elif args.resume in {"auto", "latest"}:
            run_timestamp = _find_latest_run_timestamp()
    if world_size > 1:
        ts_box = [run_timestamp if rank == 0 else None]
        dist.broadcast_object_list(ts_box, src=0)
        run_timestamp = ts_box[0]
    if run_timestamp:
        os.environ["POSE_RUN_TIMESTAMP"] = str(run_timestamp)

    # Load configuration
    config = PoseTokenizerConfig.from_yaml(args.config)
    run_dir = _resolve_run_paths(config, rank, world_size)
    _configure_local_dataset_cache(run_dir, rank, world_size)

    # Setup logging
    logger = setup_logging(config.log_dir, rank)
    logger.info(f"Starting training on rank {rank}/{world_size}")
    logger.info(f"Device: {device}")
    if rank == 0:
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"TensorBoard/Logs directory: {config.log_dir}")
        logger.info(f"Checkpoint directory: {config.checkpoint_dir}")

    tb_writer: Optional[object] = None
    if rank == 0:
        if SummaryWriter is None:
            logger.warning("TensorBoard is not available. Install tensorboard to enable event logging.")
        else:
            tb_writer = SummaryWriter(log_dir=config.log_dir)
            logger.info("TensorBoard writer initialized.")

    # Initialize wandb (only on rank 0)
    if not args.no_wandb and rank == 0:
        run_name = args.wandb_run_name or f"pose_tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config.__dict__,
            tags=['pose-tokenizer', 'training'],
            mode="disabled",
        )

    # Create model
    model = PoseTokenizerModel(config)
    model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Wrap model for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create dataloaders
    data_start = time.time()
    train_loader, val_loader, train_sampler = create_dataloaders(config, rank, world_size)
    data_elapsed = time.time() - data_start

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Dataloader setup time: {data_elapsed:.1f}s")

    # Create training configuration
    training_config = TrainingConfig(
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        hidden_dim=config.hidden_dim,
        num_quantizers=config.num_quantizers,
        codebook_size=config.codebook_size,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        reconstruction_weight=config.reconstruction_weight,
        quantization_weight=config.quantization_weight,
        perceptual_weight=config.perceptual_weight,
        temporal_weight=config.temporal_weight,
        adversarial_weight=config.adversarial_weight,
        w_pos=config.w_pos,
        w_vel=config.w_vel,
        w_acc=config.w_acc,
        w_bone=config.w_bone,
        w_vq=config.w_vq,
        beta1=config.beta1,
        beta2=config.beta2,
        momentum=config.momentum,
        scheduler_type=config.scheduler_type,
        warmup_epochs=config.warmup_epochs,
        min_lr=config.min_lr,
        step_size=config.step_size,
        gamma=config.gamma,
        optimizer_type=config.optimizer_type,
        val_interval=config.val_interval,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        use_amp=config.use_amp,
        distributed=world_size > 1,
        local_rank=local_rank,
        world_size=world_size,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        skeleton_def=config.skeleton_def,
    )

    # Create trainer
    trainer = PoseTokenizerTrainer(
        model=model,
        config=training_config,
        device=device,
        logger=logger,
        tb_writer=tb_writer if rank == 0 else None,
    )

    # Resume from checkpoint if specified
    resume_ckpt = _resolve_resume_checkpoint(args.resume, run_dir)
    if resume_ckpt is not None:
        trainer.load_checkpoint(str(resume_ckpt), load_optimizer=True)
        logger.info(f"Resumed training from {resume_ckpt}")
    elif args.resume in {"auto", "latest"} and rank == 0:
        logger.warning("Resume requested (%s) but no checkpoint found in %s", args.resume, run_dir / "checkpoints")

    # Evaluation only mode
    if args.eval_only:
        if rank == 0:
            logger.info("Running evaluation only...")

            # Create evaluation suite
            eval_suite = EvaluationSuite(
                num_keypoints=config.num_keypoints,
                num_quantizers=config.num_quantizers,
                codebook_size=config.codebook_size
            )

            # Run evaluation
            eval_results = eval_suite.evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                save_dir=Path(config.log_dir) / 'evaluation'
            )

            # Generate report
            eval_suite.generate_report(
                eval_results,
                save_path=Path(config.log_dir) / 'evaluation_report.md'
            )

            logger.info("Evaluation completed!")
            logger.info(f"Results saved to {config.log_dir}/evaluation")

            # Log to wandb
            if not args.no_wandb:
                wandb.log(eval_results['pose_metrics'])
                wandb.log(eval_results['quantization_metrics'])

        if world_size > 1:
            dist.barrier()

        if tb_writer is not None:
            tb_writer.flush()

        return

    # Training loop
    try:
        for epoch in range(trainer.epoch, training_config.num_epochs):
            trainer.epoch = epoch

            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Train epoch
            epoch_metrics = trainer.train_epoch(train_loader, val_loader)

            # Log to wandb (only on rank 0)
            if not args.no_wandb and rank == 0:
                wandb.log(epoch_metrics, step=epoch)

            # Log metrics
            if rank == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={epoch_metrics.get('train_total_loss', 0):.4f}, "
                    f"val_loss={epoch_metrics.get('val_total_loss', 0):.4f}, "
                    f"lr={epoch_metrics['lr']:.2e}"
                )
                trainer.save_checkpoint("latest.pth")
                if epoch % training_config.save_interval == 0:
                    trainer.save_checkpoint(f"epoch_{epoch}.pth")

        if rank == 0:
            trainer.save_checkpoint('final.pth')
        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if rank == 0:
            try:
                trainer.save_checkpoint('crash_latest.pth')
                logger.info("Saved crash checkpoint: crash_latest.pth")
            except Exception as save_err:
                logger.error(f"Failed to save crash checkpoint: {save_err}")
        raise

    finally:
        # Cleanup
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

        if world_size > 1:
            dist.destroy_process_group()

        if not args.no_wandb and rank == 0:
            wandb.finish()


if __name__ == '__main__':
    main()
