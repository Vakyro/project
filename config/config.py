"""
Configuration classes for CLIPZyme using dataclasses.

Provides type-safe configuration with YAML serialization support.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
from pathlib import Path
import yaml

from common.constants import (
    ESMConfig,
    EGNNConfig,
    ChemistryConfig,
    TrainingConfig as DefaultTrainingConfig,
    ProjectionConfig,
    PoolingConfig,
    DataConfig as DefaultDataConfig,
)


@dataclass
class ProteinEncoderConfig:
    """Configuration for protein encoder."""

    # Encoder type
    type: Literal['ESM2', 'EGNN'] = 'ESM2'

    # ESM2 settings
    plm_name: str = ESMConfig.DEFAULT_MODEL
    pooling: str = PoolingConfig.DEFAULT_STRATEGY

    # EGNN settings (only used if type='EGNN')
    k_neighbors: int = EGNNConfig.K_NEIGHBORS
    distance_cutoff: float = EGNNConfig.DISTANCE_CUTOFF
    egnn_layers: int = EGNNConfig.DEFAULT_NUM_LAYERS
    egnn_hidden_dim: int = EGNNConfig.DEFAULT_HIDDEN_DIM

    # Projection head
    proj_dim: int = ProjectionConfig.DEFAULT_PROJ_DIM
    proj_hidden_dim: Optional[int] = None
    dropout: float = ProjectionConfig.DEFAULT_DROPOUT

    # Other
    gradient_checkpointing: bool = False
    freeze_plm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ReactionEncoderConfig:
    """Configuration for reaction encoder."""

    # Encoder type
    type: Literal['GNN', 'Enhanced', 'DMPNN', 'DualBranch'] = 'Enhanced'

    # Feature extraction
    feature_type: Literal['basic', 'clipzyme'] = 'basic'
    use_enhanced_features: bool = True

    # GNN architecture
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1

    # For DMPNN
    node_dim: Optional[int] = None  # Auto-detected from features
    edge_dim: Optional[int] = None  # Auto-detected from features
    dmpnn_hidden_dim: int = 1280  # CLIPZyme default

    # Projection head
    proj_dim: int = ProjectionConfig.DEFAULT_PROJ_DIM
    use_attention_pool: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic settings
    batch_size: int = DefaultTrainingConfig.BATCH_SIZE
    max_epochs: int = DefaultTrainingConfig.MAX_EPOCHS
    device: str = 'cuda'

    # Optimizer
    learning_rate: float = DefaultTrainingConfig.LEARNING_RATE
    weight_decay: float = DefaultTrainingConfig.WEIGHT_DECAY
    betas: tuple = DefaultTrainingConfig.BETAS
    eps: float = DefaultTrainingConfig.EPS

    # Learning rate schedule
    use_scheduler: bool = True
    warmup_epochs: int = DefaultTrainingConfig.WARMUP_EPOCHS
    min_lr: float = DefaultTrainingConfig.MIN_LR

    # CLIP loss
    temperature: float = DefaultTrainingConfig.TEMPERATURE
    learnable_temperature: bool = False

    # Regularization
    gradient_clip: float = DefaultTrainingConfig.GRADIENT_CLIP
    use_amp: bool = True  # Automatic mixed precision

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = DefaultTrainingConfig.SAVE_EVERY
    keep_last_n: int = DefaultTrainingConfig.KEEP_LAST_N
    resume_from: Optional[str] = None

    # Logging
    log_every: int = 10
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Paths
    data_dir: str = DefaultDataConfig.DATA_DIR
    proteins_csv: str = DefaultDataConfig.PROTEINS_CSV
    reactions_csv: str = DefaultDataConfig.REACTIONS_EXTENDED_CSV
    enzyme_reactions_csv: str = DefaultDataConfig.ENZYME_REACTIONS_CSV

    # Training data
    train_json: Optional[str] = None
    val_json: Optional[str] = None
    test_json: Optional[str] = None

    # DataLoader settings
    num_workers: int = DefaultDataConfig.NUM_WORKERS
    prefetch_factor: int = DefaultDataConfig.PREFETCH_FACTOR
    pin_memory: bool = DefaultDataConfig.PIN_MEMORY

    # Data processing
    max_protein_length: int = ESMConfig.MAX_SEQUENCE_LENGTH
    use_alphafold_structures: bool = False
    structure_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CLIPZymeConfig:
    """
    Main configuration class for CLIPZyme.

    Combines all sub-configurations into a single object.
    """

    protein_encoder: ProteinEncoderConfig = field(default_factory=ProteinEncoderConfig)
    reaction_encoder: ReactionEncoderConfig = field(default_factory=ReactionEncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment info
    experiment_name: str = 'clipzyme'
    description: str = ''
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> 'CLIPZymeConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            CLIPZymeConfig object
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            protein_encoder=ProteinEncoderConfig(**config_dict.get('protein_encoder', {})),
            reaction_encoder=ReactionEncoderConfig(**config_dict.get('reaction_encoder', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment_name=config_dict.get('experiment_name', 'clipzyme'),
            description=config_dict.get('description', ''),
            random_seed=config_dict.get('random_seed', 42),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CLIPZymeConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            CLIPZymeConfig object
        """
        return cls(
            protein_encoder=ProteinEncoderConfig(**config_dict.get('protein_encoder', {})),
            reaction_encoder=ReactionEncoderConfig(**config_dict.get('reaction_encoder', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment_name=config_dict.get('experiment_name', 'clipzyme'),
            description=config_dict.get('description', ''),
            random_seed=config_dict.get('random_seed', 42),
        )

    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        config_dict = {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'random_seed': self.random_seed,
            'protein_encoder': self.protein_encoder.to_dict(),
            'reaction_encoder': self.reaction_encoder.to_dict(),
            'training': self.training.to_dict(),
            'data': self.data.to_dict(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'random_seed': self.random_seed,
            'protein_encoder': self.protein_encoder.to_dict(),
            'reaction_encoder': self.reaction_encoder.to_dict(),
            'training': self.training.to_dict(),
            'data': self.data.to_dict(),
        }

    @classmethod
    def default(cls) -> 'CLIPZymeConfig':
        """
        Create default configuration.

        Returns:
            CLIPZymeConfig with default settings
        """
        return cls()

    @classmethod
    def clipzyme_faithful(cls) -> 'CLIPZymeConfig':
        """
        Create configuration matching CLIPZyme paper exactly.

        Returns:
            CLIPZymeConfig with paper settings
        """
        config = cls()

        # Protein encoder: ESM2 (650M) + EGNN
        config.protein_encoder.type = 'EGNN'
        config.protein_encoder.plm_name = ESMConfig.CLIPZYME_MODEL
        config.protein_encoder.egnn_layers = 6
        config.protein_encoder.egnn_hidden_dim = 1280
        config.protein_encoder.proj_dim = 512

        # Reaction encoder: Two-stage DMPNN
        config.reaction_encoder.type = 'DMPNN'
        config.reaction_encoder.feature_type = 'clipzyme'
        config.reaction_encoder.dmpnn_hidden_dim = 1280
        config.reaction_encoder.num_layers = 5
        config.reaction_encoder.proj_dim = 512

        # Training
        config.training.batch_size = 64
        config.training.temperature = 0.07

        return config


# Helper functions
def load_config(path: str) -> CLIPZymeConfig:
    """
    Load configuration from file.

    Args:
        path: Path to configuration file (.yaml or .yml)

    Returns:
        CLIPZymeConfig object
    """
    return CLIPZymeConfig.from_yaml(path)


def save_config(config: CLIPZymeConfig, path: str):
    """
    Save configuration to file.

    Args:
        config: CLIPZymeConfig object
        path: Path to save file
    """
    config.to_yaml(path)


# Export
__all__ = [
    'ProteinEncoderConfig',
    'ReactionEncoderConfig',
    'TrainingConfig',
    'DataConfig',
    'CLIPZymeConfig',
    'load_config',
    'save_config',
]
