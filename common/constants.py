"""
Constants and configuration defaults for CLIPZyme.

Centralizes all magic numbers and hardcoded values to improve maintainability.
"""

from typing import List


class ESMConfig:
    """Configuration constants for ESM2 protein language model."""

    # Model names
    MODEL_T12_35M = "facebook/esm2_t12_35M_UR50D"  # 35M parameters
    MODEL_T30_150M = "facebook/esm2_t30_150M_UR50D"  # 150M parameters
    MODEL_T33_650M = "facebook/esm2_t33_650M_UR50D"  # 650M parameters (CLIPZyme default)
    MODEL_T36_3B = "facebook/esm2_t36_3B_UR50D"  # 3B parameters

    # Default settings
    DEFAULT_MODEL = MODEL_T12_35M  # Use smallest for demos
    CLIPZYME_MODEL = MODEL_T33_650M  # Use for production
    MAX_SEQUENCE_LENGTH = 650  # Maximum sequence length for CLIPZyme
    MAX_SEQUENCE_LENGTH_EXTENDED = 1024  # ESM2 absolute max

    # Model dimensions
    HIDDEN_DIM_T12 = 480
    HIDDEN_DIM_T30 = 640
    HIDDEN_DIM_T33 = 1280
    HIDDEN_DIM_T36 = 2560


class EGNNConfig:
    """Configuration constants for E(n)-equivariant GNN."""

    # Graph construction
    K_NEIGHBORS = 30  # Number of nearest neighbors
    DISTANCE_CUTOFF = 10.0  # Angstroms

    # Distance embedding
    THETA = 10000.0  # Sinusoidal encoding period
    NUM_RBFS = 16  # Number of radial basis functions

    # Architecture
    DEFAULT_NUM_LAYERS = 6
    DEFAULT_HIDDEN_DIM = 1280
    DEFAULT_DROPOUT = 0.1


class ChemistryConfig:
    """Configuration constants for chemistry/reaction processing."""

    # Common elements (atomic numbers)
    COMMON_ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I

    # Atom features (CLIPZyme)
    MAX_ATOMIC_NUM = 100
    MAX_DEGREE = 10
    MAX_FORMAL_CHARGE = 10
    MAX_NUM_HS = 8
    MAX_VALENCE = 15

    # Bond types
    BOND_TYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

    # Feature dimensions
    ATOM_FEATURE_DIM_BASIC = 7
    ATOM_FEATURE_DIM_CLIPZYME = 9
    EDGE_FEATURE_DIM_BASIC = 6  # (exists_r, exists_p, formed, broken, unchanged, changed_order)
    EDGE_FEATURE_DIM_CLIPZYME = 6


class TrainingConfig:
    """Default training configuration."""

    # Optimizer
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    # Learning rate schedule
    WARMUP_EPOCHS = 10
    MIN_LR = 1e-6

    # Training
    BATCH_SIZE = 64
    MAX_EPOCHS = 100
    GRADIENT_CLIP = 1.0

    # CLIP loss
    TEMPERATURE = 0.07

    # Checkpointing
    SAVE_EVERY = 5  # epochs
    KEEP_LAST_N = 3  # checkpoints


class ProjectionConfig:
    """Configuration for projection heads."""

    DEFAULT_PROJ_DIM = 512
    DEFAULT_HIDDEN_DIM = 2048
    DEFAULT_DROPOUT = 0.1
    USE_BATCH_NORM = False
    USE_LAYER_NORM = True


class PoolingConfig:
    """Configuration for pooling strategies."""

    AVAILABLE_STRATEGIES = ['attention', 'mean', 'cls', 'max']
    DEFAULT_STRATEGY = 'attention'


class DataConfig:
    """Configuration for data loading and processing."""

    # Paths
    DATA_DIR = "data"
    PROTEINS_CSV = "data/proteins.csv"
    REACTIONS_CSV = "data/reactions.csv"
    REACTIONS_EXTENDED_CSV = "data/reactions_extended.csv"
    ENZYME_REACTIONS_CSV = "data/enzyme_reactions.csv"

    # Processing
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    PIN_MEMORY = True


# Export all configs
__all__ = [
    'ESMConfig',
    'EGNNConfig',
    'ChemistryConfig',
    'TrainingConfig',
    'ProjectionConfig',
    'PoolingConfig',
    'DataConfig',
]
