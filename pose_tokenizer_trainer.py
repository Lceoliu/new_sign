#!/usr/bin/env python3
"""
Pose Tokenizer Trainer for Uni-Sign project.
Specialized trainer for pose tokenization using RVQ.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import numpy as np

from models import Uni_Sign
from datasets import S2T_Dataset_news


class PoseTokenizerTrainer:
    """
    Trainer specifically for pose tokenization mode.
    """

    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device

        # Setup logging
        self.setup_logging()

        # Setup optimizer and scheduler
        self.setup_optimizer()

        # Setup data loaders
        self.setup_data_loaders()

        # Metrics tracking
        self.best_loss = float('inf')
        self.global_step = 0

        # Create checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.args.log_dir, 'pose_tokenizer_training.log'))
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize wandb if enabled
        if getattr(self.args, 'use_wandb', False):
            wandb.init(
                project=getattr(self.args, 'wandb_project', 'uni-sign-pose-tokenizer'),
                name=getattr(self.args, 'wandb_run_name', 'pose_tokenizer_training'),
                config=vars(self.args)
            )

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Filter parameters - only train pose tokenizer components
        if self.args.use_pose_tokenizer:
            tokenizer_params = []
            other_params = []

            for name, param in self.model.named_parameters():
                if 'pose_tokenizer' in name:
                    tokenizer_params.append(param)
                else:
                    other_params.append(param)

            # Different learning rates for tokenizer and other components
            param_groups = [
                {'params': tokenizer_params, 'lr': self.args.learning_rate},
                {'params': other_params, 'lr': self.args.learning_rate * 0.1}  # Lower LR for pre-trained components
            ]
        else:
            param_groups = [{'params': self.model.parameters(), 'lr': self.args.learning_rate}]

        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, 'weight_decay', 0.01),
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        if getattr(self.args, 'use_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.num_epochs,
                eta_min=self.args.learning_rate * 0.01
            )
        else:
            self.scheduler = None

    def setup_data_loaders(self):
        """Setup data loaders for training and validation."""
        from config import train_label_paths, dev_label_paths

        # Training data loader
        train_dataset = S2T_Dataset_news(
            path=train_label_paths[self.args.dataset],
            args=self.args,
            phase='train'
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=getattr(self.args, 'num_workers', 4),
            pin_memory=True,
            drop_last=True
        )

        # Validation data loader
        val_dataset = S2T_Dataset_news(
            path=dev_label_paths[self.args.dataset],
            args=self.args,
            phase='val'
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, 'num_workers', 4),
            pin_memory=True
        )

        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_vq_loss = 0.0
        total_main_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.num_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            src_input = {}
            for key in ['body', 'left', 'right', 'face_all', 'attention_mask']:
                if key in batch:
                    src_input[key] = batch[key].to(self.device)

            tgt_input = {
                'gt_sentence': batch['gt_sentence']
            }

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(src_input, tgt_input)

            loss = outputs['loss']

            # Extract VQ-specific losses if available
            vq_loss = outputs.get('vq_loss', torch.tensor(0.0))
            perplexity = outputs.get('perplexity', torch.tensor(0.0))

            # Backward pass
            loss.backward()

            # Gradient clipping
            if getattr(self.args, 'max_grad_norm', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            if isinstance(vq_loss, torch.Tensor):
                total_vq_loss += vq_loss.item()
            if isinstance(perplexity, torch.Tensor):
                total_perplexity += perplexity.item()

            main_loss = loss.item() - (vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0)
            total_main_loss += main_loss

            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'VQ Loss': f'{vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0:.4f}',
                'Perplexity': f'{perplexity.item() if isinstance(perplexity, torch.Tensor) else 0:.2f}'
            })

            # Log to wandb
            if getattr(self.args, 'use_wandb', False) and self.global_step % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/vq_loss': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0,
                    'train/main_loss': main_loss,
                    'train/perplexity': perplexity.item() if isinstance(perplexity, torch.Tensor) else 0,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches
        avg_main_loss = total_main_loss / num_batches
        avg_perplexity = total_perplexity / num_batches

        self.logger.info(f'Epoch {epoch+1} Training - Loss: {avg_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}, '
                        f'Main Loss: {avg_main_loss:.4f}, Perplexity: {avg_perplexity:.2f}')

        return avg_loss, avg_vq_loss, avg_perplexity

    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0.0
        total_vq_loss = 0.0
        total_main_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                src_input = {}
                for key in ['body', 'left', 'right', 'face_all', 'attention_mask']:
                    if key in batch:
                        src_input[key] = batch[key].to(self.device)

                tgt_input = {
                    'gt_sentence': batch['gt_sentence']
                }

                # Forward pass
                outputs = self.model(src_input, tgt_input)

                loss = outputs['loss']
                vq_loss = outputs.get('vq_loss', torch.tensor(0.0))
                perplexity = outputs.get('perplexity', torch.tensor(0.0))

                # Update metrics
                total_loss += loss.item()
                if isinstance(vq_loss, torch.Tensor):
                    total_vq_loss += vq_loss.item()
                if isinstance(perplexity, torch.Tensor):
                    total_perplexity += perplexity.item()

                main_loss = loss.item() - (vq_loss.item() if isinstance(vq_loss, torch.Tensor) else 0)
                total_main_loss += main_loss

                num_batches += 1

        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches
        avg_main_loss = total_main_loss / num_batches
        avg_perplexity = total_perplexity / num_batches

        self.logger.info(f'Epoch {epoch+1} Validation - Loss: {avg_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}, '
                        f'Main Loss: {avg_main_loss:.4f}, Perplexity: {avg_perplexity:.2f}')

        return avg_loss, avg_vq_loss, avg_perplexity

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'global_step': self.global_step,
            'args': vars(self.args)
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved with loss: {loss:.4f}')

        # Save epoch checkpoint
        epoch_path = os.path.join(self.args.checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_path)

    def train(self):
        """Main training loop."""
        self.logger.info("Starting pose tokenizer training...")
        self.logger.info(f"Training for {self.args.num_epochs} epochs")
        self.logger.info(f"Batch size: {self.args.batch_size}")
        self.logger.info(f"Learning rate: {self.args.learning_rate}")
        self.logger.info(f"Use pose tokenizer: {self.args.use_pose_tokenizer}")

        for epoch in range(self.args.num_epochs):
            # Training
            train_loss, train_vq_loss, train_perplexity = self.train_epoch(epoch)

            # Validation
            val_loss, val_vq_loss, val_perplexity = self.validate_epoch(epoch)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Log epoch metrics
            if getattr(self.args, 'use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'train/epoch_vq_loss': train_vq_loss,
                    'train/epoch_perplexity': train_perplexity,
                    'val/epoch_loss': val_loss,
                    'val/epoch_vq_loss': val_vq_loss,
                    'val/epoch_perplexity': val_perplexity,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

            self.logger.info(f'Epoch {epoch+1} completed. Best validation loss: {self.best_loss:.4f}')

        self.logger.info("Training completed!")

        if getattr(self.args, 'use_wandb', False):
            wandb.finish()

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1

        self.logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

        return start_epoch


def create_pose_tokenizer_trainer(args):
    """Factory function to create pose tokenizer trainer."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = Uni_Sign(args)
    model.to(device)

    # Create trainer
    trainer = PoseTokenizerTrainer(args, model, device)

    return trainer