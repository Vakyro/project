"""
Tests for training callbacks.
"""

import pytest
import torch
from pathlib import Path

from training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    MetricsLogger,
    CallbackList
)


class MockTrainer:
    """Mock trainer for testing callbacks."""

    def __init__(self):
        self.should_stop = False
        self.global_step = 0
        self.current_epoch = 0
        self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(10))], lr=0.001)
        self.protein_encoder = torch.nn.Linear(10, 10)
        self.reaction_encoder = torch.nn.Linear(10, 10)
        self.scheduler = None
        self.scaler = None
        self.config = type('Config', (), {'device': 'cpu'})()


class TestEarlyStopping:
    """Test suite for EarlyStopping callback."""

    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')
        trainer = MockTrainer()

        # Improving losses - should not stop
        for epoch in range(5):
            metrics = {'val_loss': 1.0 - epoch * 0.1}
            callback.on_epoch_end(trainer, epoch, metrics)

        assert not trainer.should_stop

    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement."""
        callback = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=False)
        trainer = MockTrainer()

        # No improvement - should stop after patience epochs
        for epoch in range(10):
            metrics = {'val_loss': 1.0}
            callback.on_epoch_end(trainer, epoch, metrics)

            if epoch >= 3:  # After patience
                assert trainer.should_stop
                break

    def test_early_stopping_mode_max(self):
        """Test early stopping in max mode."""
        callback = EarlyStopping(monitor='val_acc', patience=2, mode='max', verbose=False)
        trainer = MockTrainer()

        # Decreasing accuracy - should stop
        for epoch in range(10):
            metrics = {'val_acc': 1.0 - epoch * 0.1}
            callback.on_epoch_end(trainer, epoch, metrics)

            if epoch >= 2:
                assert trainer.should_stop
                break


class TestModelCheckpoint:
    """Test suite for ModelCheckpoint callback."""

    def test_save_checkpoint(self, temp_dir):
        """Test saving checkpoint."""
        callback = ModelCheckpoint(
            checkpoint_dir=str(temp_dir),
            save_best_only=False,
            filename_format='epoch_{epoch}.pt',
            verbose=False
        )

        trainer = MockTrainer()
        metrics = {'val_loss': 0.5}

        callback.on_epoch_end(trainer, epoch=0, metrics=metrics)

        # Check checkpoint was saved
        checkpoint_files = list(temp_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0

    def test_save_best_only(self, temp_dir):
        """Test save best only mode."""
        callback = ModelCheckpoint(
            checkpoint_dir=str(temp_dir),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_last=False,
            verbose=False
        )

        trainer = MockTrainer()

        # First epoch - should save
        callback.on_epoch_end(trainer, 0, {'val_loss': 1.0})

        # Worse loss - should not save
        callback.on_epoch_end(trainer, 1, {'val_loss': 2.0})

        # Better loss - should save
        callback.on_epoch_end(trainer, 2, {'val_loss': 0.5})

        # Best checkpoint should exist
        best_path = temp_dir / 'best.pt'
        assert best_path.exists()


class TestLearningRateMonitor:
    """Test suite for LearningRateMonitor callback."""

    def test_lr_monitoring_epoch(self):
        """Test LR monitoring at epoch level."""
        callback = LearningRateMonitor(logging_interval='epoch')
        trainer = MockTrainer()

        metrics = {'val_loss': 0.5}
        callback.on_epoch_end(trainer, 0, metrics)

        assert 'learning_rate' in metrics


class TestMetricsLogger:
    """Test suite for MetricsLogger callback."""

    def test_log_to_file(self, temp_dir):
        """Test logging metrics to file."""
        log_file = temp_dir / 'metrics.log'

        callback = MetricsLogger(
            log_file=str(log_file),
            console=False
        )

        trainer = MockTrainer()
        metrics = {'loss': 0.5, 'accuracy': 0.9}

        callback.on_epoch_end(trainer, 0, metrics)

        # Check log file exists and has content
        assert log_file.exists()
        content = log_file.read_text()
        assert 'loss' in content
        assert 'accuracy' in content


class TestCallbackList:
    """Test suite for CallbackList."""

    def test_callback_list_execution(self):
        """Test callbacks are executed in order."""
        execution_order = []

        class TrackingCallback:
            def __init__(self, name):
                self.name = name

            def on_epoch_begin(self, trainer, epoch):
                execution_order.append(self.name)

        callbacks = CallbackList([
            TrackingCallback('callback1'),
            TrackingCallback('callback2'),
            TrackingCallback('callback3')
        ])

        trainer = MockTrainer()
        callbacks.on_epoch_begin(trainer, 0)

        assert execution_order == ['callback1', 'callback2', 'callback3']

    def test_add_callback(self):
        """Test adding callback to list."""
        callbacks = CallbackList()

        assert len(callbacks.callbacks) == 0

        callbacks.add(EarlyStopping(monitor='val_loss'))

        assert len(callbacks.callbacks) == 1
