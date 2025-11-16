"""
Pytest configuration and fixtures.

Provides reusable test fixtures for CLIPZyme tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def device():
    """Get compute device for tests."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_protein_sequence():
    """Sample protein sequence for testing."""
    return "MSKQLIVNLLKQNNYKNSGSSAFWEYFNPSNLSQLQKDIPMNPFSEMGFLFQQKGQQMIVLSDLFNLSKKNQKPILK"


@pytest.fixture
def sample_protein_sequences():
    """Multiple protein sequences for batch testing."""
    return [
        "MSKQLIVNLLKQNN",
        "AFGWEKPQLMNSKD",
        "LQKDIPMNPFSEMG"
    ]


@pytest.fixture
def sample_reaction_smiles():
    """Sample reaction SMILES for testing."""
    return "[C:1]=[O:2]>>[C:1][O:2]"


@pytest.fixture
def sample_reactions():
    """Multiple reactions for batch testing."""
    return [
        "[C:1]=[O:2]>>[C:1][O:2]",
        "[N:1]=[N:2]>>[N:1][N:2]",
        "[C:1]=[C:2]>>[C:1][C:2]"
    ]


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_checkpoint_path(temp_dir):
    """Create a sample checkpoint file."""
    checkpoint_path = temp_dir / "test_checkpoint.pt"

    # Create a minimal checkpoint
    checkpoint = {
        'epoch': 5,
        'protein_encoder_state_dict': {},
        'reaction_encoder_state_dict': {},
        'optimizer_state_dict': {},
        'metrics': {'loss': 0.5}
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return {
        'protein': torch.randn(10, 128),
        'reaction': torch.randn(10, 128)
    }


@pytest.fixture
def sample_screening_set(temp_dir):
    """Sample screening set for testing."""
    from screening.screening_set import ScreeningSet

    # Create dummy data
    protein_names = [f"protein_{i}" for i in range(5)]
    embeddings = torch.randn(5, 128)

    screening_set = ScreeningSet(
        protein_names=protein_names,
        embeddings=embeddings
    )

    # Save to temp file
    save_path = temp_dir / "test_screening_set.pkl"
    screening_set.save(save_path)

    return save_path


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    from config.config import CLIPZymeConfig

    config = CLIPZymeConfig()
    config.protein_encoder.proj_dim = 128
    config.reaction_encoder.proj_dim = 128

    return config


@pytest.fixture
def sample_trainer_config():
    """Sample trainer configuration."""
    from training.trainer import TrainerConfig

    return TrainerConfig(
        max_epochs=2,
        device='cpu',
        learning_rate=1e-4,
        warmup_steps=5,
        val_every_n_epochs=1
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture(autouse=True)
def reset_cuda():
    """Reset CUDA state after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Mark GPU tests
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (skip if not available)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
