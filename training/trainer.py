"""
CLIPZyme trainer with callbacks, logging, and validation.

Provides a robust training loop with all modern features.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.amp as amp
import numpy as np
import logging

from .callbacks import CallbackList, Callback
from .logger import TrainingLogger, ConsoleLogger
from .lr_scheduler import get_scheduler


logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for CLIPZyme trainer."""

    # Training settings
    max_epochs: int = 30
    device: str = 'cuda'
    use_amp: bool = True
    gradient_clip: float = 1.0

    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler settings
    scheduler_type: str = 'warmup_cosine'
    warmup_steps: int = 100
    min_lr: float = 1e-6

    # CLIP loss settings
    temperature: float = 0.07
    learnable_temperature: bool = False

    # Validation settings
    val_every_n_epochs: int = 1
    val_at_start: bool = True

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 1

    # Logging
    log_every_n_steps: int = 10
    log_dir: str = 'logs'

    # Misc
    seed: int = 42
    num_workers: int = 4


class CLIPZymeTrainer:
    """
    Complete trainer for CLIPZyme with callbacks and logging.
    """

    def __init__(
        self,
        protein_encoder: nn.Module,
        reaction_encoder: nn.Module,
        config: TrainerConfig,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[TrainingLogger] = None
    ):
        """
        Initialize trainer.

        Args:
            protein_encoder: Protein encoder model
            reaction_encoder: Reaction encoder model
            config: Trainer configuration
            callbacks: List of callbacks
            logger: Training logger
        """
        self.config = config
        self.protein_encoder = protein_encoder.to(config.device)
        self.reaction_encoder = reaction_encoder.to(config.device)

        # Set random seed
        self._set_seed(config.seed)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(protein_encoder.parameters()) + list(reaction_encoder.parameters()),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )

        # Scheduler (will be initialized in fit())
        self.scheduler = None

        # AMP scaler
        self.scaler = amp.GradScaler('cuda') if config.use_amp else None

        # Temperature parameter
        if config.learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(config.temperature))
            )
        else:
            self.register_buffer(
                'log_temperature',
                torch.log(torch.tensor(config.temperature))
            )

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # Logger
        self.logger = logger or ConsoleLogger()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False

        # Best metrics
        self.best_val_loss = float('inf')

        logger.info("Trainer initialized")

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ):
        """
        Train the model.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Calculate total steps
        total_steps = len(train_dataloader) * self.config.max_epochs

        # Initialize scheduler
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.config.scheduler_type,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.min_lr
        )

        # Log hyperparameters
        self.logger.log_hyperparameters(self.config.__dict__)

        # Training begin callback
        self.callbacks.on_train_begin(self)

        # Validation at start if requested
        if self.config.val_at_start and val_dataloader is not None:
            val_metrics = self.validate(val_dataloader)
            logger.info(f"Initial validation: {val_metrics}")

        # Training loop
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            # Epoch begin callback
            self.callbacks.on_epoch_begin(self, epoch)

            # Train one epoch
            train_metrics = self.train_epoch(train_dataloader, epoch)

            # Validation
            val_metrics = {}
            if (val_dataloader is not None and
                (epoch + 1) % self.config.val_every_n_epochs == 0):
                val_metrics = self.validate(val_dataloader)

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Epoch end callback
            self.callbacks.on_epoch_end(self, epoch, all_metrics)

            # Log metrics
            self.logger.log(all_metrics, step=epoch)

            # Check for early stopping
            if self.should_stop:
                logger.info("Early stopping triggered")
                break

        # Training end callback
        self.callbacks.on_train_end(self)

        # Finish logging
        self.logger.finish()

        logger.info("Training completed")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.protein_encoder.train()
        self.reaction_encoder.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Batch begin callback
            self.callbacks.on_batch_begin(self, batch_idx)

            # Forward pass
            loss = self.training_step(batch)

            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()

                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.protein_encoder.parameters()) +
                        list(self.reaction_encoder.parameters()),
                        self.config.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.protein_encoder.parameters()) +
                        list(self.reaction_encoder.parameters()),
                        self.config.gradient_clip
                    )

                self.optimizer.step()

            self.optimizer.zero_grad()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Batch end callback
            self.callbacks.on_batch_end(self, batch_idx, loss.item())

            # Log batch metrics
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                self.logger.log(
                    {
                        'train_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    },
                    step=self.global_step
                )

        # Calculate epoch metrics
        avg_loss = total_loss / num_batches

        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Perform one training step.

        Args:
            batch: Batch of data

        Returns:
            Loss tensor
        """
        # Move batch to device
        sequences = batch['sequences']
        structures = batch.get('structures')  # Optional
        reactions = batch['reactions']

        # Encode proteins
        if structures is not None:
            protein_embeddings = self.protein_encoder(sequences, structures)
        else:
            protein_embeddings = self.protein_encoder(sequences)

        # Encode reactions
        reaction_embeddings = self.reaction_encoder(reactions)

        # CLIP loss
        loss = self.clip_loss(protein_embeddings, reaction_embeddings)

        return loss

    def clip_loss(
        self,
        protein_embeddings: torch.Tensor,
        reaction_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CLIP loss.

        Args:
            protein_embeddings: Protein embeddings [batch_size, dim]
            reaction_embeddings: Reaction embeddings [batch_size, dim]

        Returns:
            CLIP loss
        """
        # Normalize embeddings
        protein_embeddings = torch.nn.functional.normalize(protein_embeddings, dim=-1)
        reaction_embeddings = torch.nn.functional.normalize(reaction_embeddings, dim=-1)

        # Compute similarity matrix
        temperature = torch.exp(self.log_temperature)
        logits = protein_embeddings @ reaction_embeddings.T / temperature

        # Labels: diagonal elements are positive pairs
        batch_size = protein_embeddings.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Cross-entropy loss in both directions
        loss_protein = torch.nn.functional.cross_entropy(logits, labels)
        loss_reaction = torch.nn.functional.cross_entropy(logits.T, labels)

        # Average loss
        loss = (loss_protein + loss_reaction) / 2

        return loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.protein_encoder.eval()
        self.reaction_encoder.eval()

        # Validation begin callback
        self.callbacks.on_validation_begin(self)

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            loss = self.training_step(batch)
            total_loss += loss.item()
            num_batches += 1

        # Calculate metrics
        avg_loss = total_loss / num_batches

        val_metrics = {
            'val_loss': avg_loss
        }

        # Validation end callback
        self.callbacks.on_validation_end(self, val_metrics)

        return val_metrics

    def save_checkpoint(self, path: str, **extra_data):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            **extra_data: Additional data to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'protein_encoder_state_dict': self.protein_encoder.state_dict(),
            'reaction_encoder_state_dict': self.reaction_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            **extra_data
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, strict: bool = True):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
            strict: Strict state dict loading
        """
        checkpoint = torch.load(path, map_location=self.config.device)

        self.protein_encoder.load_state_dict(
            checkpoint['protein_encoder_state_dict'],
            strict=strict
        )
        self.reaction_encoder.load_state_dict(
            checkpoint['reaction_encoder_state_dict'],
            strict=strict
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)

        logger.info(f"Loaded checkpoint from: {path}")


# Export
__all__ = [
    'CLIPZymeTrainer',
    'TrainerConfig',
]
