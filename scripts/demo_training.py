"""
Demo script for CLIPZyme training with callbacks and logging.

Shows how to use the new training infrastructure.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('scripts', 1)[0].rstrip('/\\'))

import torch
from pathlib import Path

# Import training components
from training import (
    CLIPZymeTrainer,
    TrainerConfig,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    MetricsLogger,
    ProgressBar,
    WandbLogger,
    TensorBoardLogger,
    ConsoleLogger,
)

# Import models
from protein_encoder.esm_model import ProteinEncoderESM
from reaction_encoder.dmpnn import ReactionDMPNN


def create_dummy_dataloader(batch_size=4, num_batches=10):
    """Create a dummy dataloader for testing."""
    from torch.utils.data import Dataset, DataLoader

    class DummyDataset(Dataset):
        def __len__(self):
            return num_batches * batch_size

        def __getitem__(self, idx):
            return {
                'sequences': ['MSKQLI' * 10],
                'reactions': ['[C:1]=[O:2]>>[C:1][O:2]']
            }

    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    """Run training demo."""
    print("=" * 60)
    print("CLIPZyme Training Demo")
    print("=" * 60)

    # Configuration
    config = TrainerConfig(
        max_epochs=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-4,
        warmup_steps=10,
        val_every_n_epochs=1,
        log_every_n_steps=5,
        checkpoint_dir='demo_checkpoints',
    )

    print(f"\nDevice: {config.device}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Learning rate: {config.learning_rate}")

    # Create models
    print("\n[1/5] Creating models...")
    protein_encoder = ProteinEncoderESM(
        plm_name='facebook/esm2_t6_8M_UR50D',  # Small model for demo
        pooling='mean',
        proj_dim=128
    )

    reaction_encoder = ReactionDMPNN(
        node_dim=16,
        edge_dim=6,
        hidden_dim=128,
        num_layers=3,
        proj_dim=128
    )

    print("   ✓ Models created")

    # Create dataloaders
    print("\n[2/5] Creating dataloaders...")
    train_dataloader = create_dummy_dataloader(batch_size=4, num_batches=10)
    val_dataloader = create_dummy_dataloader(batch_size=4, num_batches=5)
    print("   ✓ Dataloaders created")

    # Setup callbacks
    print("\n[3/5] Setting up callbacks...")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
            verbose=True
        ),
        ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
        MetricsLogger(
            log_file=Path(config.checkpoint_dir) / 'metrics.log',
            console=True
        ),
        ProgressBar(),
    ]
    print("   ✓ Callbacks configured")

    # Setup logger
    print("\n[4/5] Setting up logger...")
    # Choose logger type (comment/uncomment as needed)

    # Option 1: Console logger (simple)
    logger = ConsoleLogger()

    # Option 2: TensorBoard logger
    # logger = TensorBoardLogger(log_dir='runs/demo')
    # print("   Run: tensorboard --logdir runs/demo")

    # Option 3: WandB logger (requires wandb account)
    # logger = WandbLogger(
    #     project='clipzyme',
    #     name='demo_run',
    #     config=config.__dict__
    # )

    print("   ✓ Logger configured")

    # Create trainer
    print("\n[5/5] Creating trainer...")
    trainer = CLIPZymeTrainer(
        protein_encoder=protein_encoder,
        reaction_encoder=reaction_encoder,
        config=config,
        callbacks=callbacks,
        logger=logger
    )
    print("   ✓ Trainer created")

    # Start training
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    try:
        trainer.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )

        print("\n" + "=" * 60)
        print("Training Completed!")
        print("=" * 60)

        # Show results
        checkpoint_dir = Path(config.checkpoint_dir)
        if checkpoint_dir.exists():
            print(f"\nCheckpoints saved to: {checkpoint_dir}")
            print("Files:")
            for f in sorted(checkpoint_dir.glob("*.pt")):
                print(f"  - {f.name}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        print(f"\n\nTraining failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nDemo completed!")


if __name__ == '__main__':
    main()
