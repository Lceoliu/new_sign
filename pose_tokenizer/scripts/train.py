#!/usr/bin/env python3
"""Training script for pose tokenizer."""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose_tokenizer.config import PoseTokenizerConfig
from pose_tokenizer.data.dataset import PoseDataset
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


def create_dataloaders(config: PoseTokenizerConfig, rank: int = 0, world_size: int = 1):
    """Create training and validation dataloaders."""

    # Training dataset
    train_dataset = PoseDataset(
        data_dir=config.data_dir,
        split='train',
        sequence_length=config.sequence_length,
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        augment=True,
        normalize=config.normalize_poses
    )

    # Validation dataset
    val_dataset = PoseDataset(
        data_dir=config.data_dir,
        split='val',
        sequence_length=config.sequence_length,
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        augment=False,
        normalize=config.normalize_poses
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
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
                       help='Path to checkpoint to resume from')
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

    # Load configuration
    config = PoseTokenizerConfig.from_yaml(args.config)

    # Setup logging
    logger = setup_logging(config.log_dir, rank)
    logger.info(f"Starting training on rank {rank}/{world_size}")
    logger.info(f"Device: {device}")

    # Initialize wandb (only on rank 0)
    if not args.no_wandb and rank == 0:
        run_name = args.wandb_run_name or f"pose_tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config.__dict__,
            tags=['pose-tokenizer', 'training']
        )

    # Create model
    model = PoseTokenizerModel(config)
    model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Wrap model for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders(config, rank, world_size)

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

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
        scheduler_type=config.scheduler_type,
        warmup_epochs=config.warmup_epochs,
        min_lr=config.min_lr,
        optimizer_type=config.optimizer_type,
        use_amp=config.use_amp,
        distributed=world_size > 1,
        local_rank=local_rank,
        world_size=world_size,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir
    )

    # Create trainer
    trainer = PoseTokenizerTrainer(
        model=model,
        config=training_config,
        device=device,
        logger=logger
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from {args.resume}")

    # Evaluation only mode
    if args.eval_only:
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
        report = eval_suite.generate_report(
            eval_results,
            save_path=Path(config.log_dir) / 'evaluation_report.md'
        )

        logger.info("Evaluation completed!")
        logger.info(f"Results saved to {config.log_dir}/evaluation")

        # Log to wandb
        if not args.no_wandb and rank == 0:
            wandb.log(eval_results['pose_metrics'])
            wandb.log(eval_results['quantization_metrics'])

        return

    # Training loop
    try:
        for epoch in range(trainer.epoch, training_config.num_epochs):
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

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    finally:
        # Cleanup
        if world_size > 1:
            dist.destroy_process_group()

        if not args.no_wandb and rank == 0:
            wandb.finish()


if __name__ == '__main__':
    main()