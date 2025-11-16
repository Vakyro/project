"""
Checkpoint Loader for CLIPZyme Models

Loads checkpoints from various formats:
- Official CLIPZyme PyTorch Lightning checkpoints
- Standard PyTorch state_dict
- This project's checkpoint format

Handles:
- Parameter name mapping
- Model architecture inference
- Compatibility checks
"""

import torch
import pickle
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any
import logging

from models.clipzyme import CLIPZymeModel
from models.builder import CLIPZymeBuilder
from config.config import CLIPZymeConfig

logger = logging.getLogger(__name__)


class CheckpointLoader:
    """
    Universal checkpoint loader for CLIPZyme models.

    Supports multiple checkpoint formats and handles conversion automatically.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize checkpoint loader.

        Args:
            device: Device to load checkpoint to
        """
        self.device = device

    def load(
        self,
        checkpoint_path: Union[str, Path],
        config: Optional[CLIPZymeConfig] = None,
        strict: bool = False
    ) -> CLIPZymeModel:
        """
        Load a checkpoint and return CLIPZyme model.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Optional model config (inferred if not provided)
            strict: If True, strictly match all parameter names

        Returns:
            Loaded CLIPZyme model
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Detect checkpoint format
        checkpoint_format = self._detect_format(checkpoint_path)
        logger.info(f"Detected format: {checkpoint_format}")

        # Load checkpoint based on format
        if checkpoint_format == "pytorch_lightning":
            model = self._load_lightning_checkpoint(checkpoint_path, config, strict)
        elif checkpoint_format == "state_dict":
            model = self._load_state_dict(checkpoint_path, config, strict)
        elif checkpoint_format == "full_model":
            model = self._load_full_model(checkpoint_path)
        elif checkpoint_format == "pickle":
            model = self._load_pickle(checkpoint_path)
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint_format}")

        model = model.to(self.device)
        model.eval()

        logger.info("✓ Checkpoint loaded successfully")
        return model

    def _detect_format(self, checkpoint_path: Path) -> str:
        """
        Detect checkpoint format by examining file structure.

        Returns:
            Format string: 'pytorch_lightning', 'state_dict', 'full_model', or 'pickle'
        """
        try:
            # Try loading as torch checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # PyTorch Lightning format has 'state_dict' key
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    # Check if it's Lightning format (has additional keys)
                    if any(k in checkpoint for k in ['epoch', 'global_step', 'hyper_parameters']):
                        return "pytorch_lightning"
                    return "state_dict"

                # Check if it's a direct state_dict (all keys are model parameters)
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    return "state_dict"

            # If it's a model instance
            if isinstance(checkpoint, torch.nn.Module):
                return "full_model"

            return "pickle"

        except Exception as e:
            logger.warning(f"Could not detect format: {e}")
            return "unknown"

    def _load_lightning_checkpoint(
        self,
        checkpoint_path: Path,
        config: Optional[CLIPZymeConfig],
        strict: bool
    ) -> CLIPZymeModel:
        """Load PyTorch Lightning checkpoint."""
        logger.info("Loading PyTorch Lightning checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state_dict
        state_dict = checkpoint['state_dict']

        # Lightning often prefixes keys with 'model.'
        state_dict = self._remove_prefix(state_dict, 'model.')

        # Get hyperparameters if available
        hparams = checkpoint.get('hyper_parameters', {})
        if hparams and not config:
            logger.info("Inferring config from hyperparameters...")
            config = self._infer_config_from_hparams(hparams)

        # Build model
        if config is None:
            # Try to infer from state_dict
            config = self._infer_config_from_state_dict(state_dict)

        model = self._build_model_from_config(config)

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

        # Log training info
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            logger.info(f"Global step: {checkpoint['global_step']}")

        return model

    def _load_state_dict(
        self,
        checkpoint_path: Path,
        config: Optional[CLIPZymeConfig],
        strict: bool
    ) -> CLIPZymeModel:
        """Load standard state_dict checkpoint."""
        logger.info("Loading state_dict checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state_dict if wrapped
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Infer config if not provided
        if config is None:
            config = self._infer_config_from_state_dict(state_dict)

        # Build model
        model = self._build_model_from_config(config)

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

        return model

    def _load_full_model(self, checkpoint_path: Path) -> CLIPZymeModel:
        """Load full model (torch.save(model, ...))."""
        logger.info("Loading full model checkpoint...")

        model = torch.load(checkpoint_path, map_location='cpu')

        if not isinstance(model, CLIPZymeModel):
            logger.warning(
                f"Loaded model is {type(model)}, not CLIPZymeModel. "
                "May cause compatibility issues."
            )

        return model

    def _load_pickle(self, checkpoint_path: Path) -> CLIPZymeModel:
        """Load pickle checkpoint."""
        logger.info("Loading pickle checkpoint...")

        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        if isinstance(checkpoint, CLIPZymeModel):
            return checkpoint
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            return checkpoint['model']
        else:
            raise ValueError("Pickle file does not contain a CLIPZyme model")

    def _remove_prefix(self, state_dict: Dict, prefix: str) -> Dict:
        """Remove prefix from state_dict keys."""
        if not any(k.startswith(prefix) for k in state_dict.keys()):
            return state_dict

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
            else:
                new_state_dict[k] = v

        logger.info(f"Removed prefix '{prefix}' from state_dict keys")
        return new_state_dict

    def _infer_config_from_hparams(self, hparams: Dict) -> CLIPZymeConfig:
        """Infer CLIPZymeConfig from Lightning hyperparameters."""
        logger.info("Inferring config from hyperparameters...")

        # This will depend on how the official model stores hparams
        # We'll use defaults and override with any found values

        config = CLIPZymeConfig.get_preset('clipzyme_faithful')

        # Update from hparams if available
        if 'temperature' in hparams:
            config.training.temperature = hparams['temperature']

        if 'learning_rate' in hparams:
            config.training.learning_rate = hparams['learning_rate']

        # Add more mappings as needed...

        return config

    def _infer_config_from_state_dict(self, state_dict: Dict) -> CLIPZymeConfig:
        """Infer CLIPZymeConfig from state_dict structure."""
        logger.info("Inferring config from state_dict structure...")

        # Start with faithful preset
        config = CLIPZymeConfig.get_preset('clipzyme_faithful')

        # Try to infer dimensions and architecture details
        # Look for specific layer shapes to determine architecture

        # Check protein encoder type
        if any('egnn' in k.lower() for k in state_dict.keys()):
            config.protein_encoder.encoder_type = 'egnn'
            logger.info("Detected EGNN protein encoder")
        elif any('esm' in k.lower() for k in state_dict.keys()):
            config.protein_encoder.encoder_type = 'esm2'
            logger.info("Detected ESM2 protein encoder")

        # Check reaction encoder type
        if any('dmpnn' in k.lower() for k in state_dict.keys()):
            config.reaction_encoder.encoder_type = 'dmpnn'
            logger.info("Detected DMPNN reaction encoder")

        # Infer embedding dimensions from projection layers
        for key, tensor in state_dict.items():
            if 'projection' in key.lower() and 'weight' in key:
                if tensor.dim() == 2:
                    out_dim, in_dim = tensor.shape
                    logger.info(f"Inferred projection: {in_dim} -> {out_dim}")
                    config.protein_encoder.projection_dim = out_dim
                    config.reaction_encoder.projection_dim = out_dim
                    break

        return config

    def _build_model_from_config(self, config: CLIPZymeConfig) -> CLIPZymeModel:
        """Build CLIPZyme model from config."""
        logger.info("Building model from config...")

        builder = CLIPZymeBuilder()
        model = builder.with_config(config).build()

        logger.info(f"Built model: {model.get_config()}")
        return model


def load_official_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    config: Optional[CLIPZymeConfig] = None
) -> CLIPZymeModel:
    """
    Load official CLIPZyme checkpoint from Zenodo.

    This is the main function to use for loading official checkpoints.

    Args:
        checkpoint_path: Path to checkpoint file (.ckpt, .pt, .pth, .pkl)
        device: Device to load model to
        config: Optional config (auto-inferred if not provided)

    Returns:
        Loaded CLIPZyme model ready for inference

    Example:
        >>> model = load_official_checkpoint("data/checkpoints/clipzyme_model.ckpt")
        >>> embeddings = model.encode_reactions(["[C:1]=[O:2]>>[C:1]-[O:2]"])
    """
    loader = CheckpointLoader(device=device)
    model = loader.load(checkpoint_path, config=config, strict=False)

    logger.info("Official checkpoint loaded successfully!")
    return model


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    **kwargs
) -> CLIPZymeModel:
    """
    General checkpoint loader (alias for load_official_checkpoint).

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load to
        **kwargs: Additional arguments for CheckpointLoader

    Returns:
        Loaded CLIPZyme model
    """
    return load_official_checkpoint(checkpoint_path, device, **kwargs)


def load_pretrained(
    model_name: str = "clipzyme",
    cache_dir: str = "data/checkpoints",
    device: str = "cpu",
    download_if_missing: bool = True
) -> CLIPZymeModel:
    """
    Load a pretrained CLIPZyme model by name.

    Args:
        model_name: Model name ('clipzyme', 'clipzyme_faithful')
        cache_dir: Directory to cache downloaded models
        device: Device to load to
        download_if_missing: If True, download if not found locally

    Returns:
        Pretrained CLIPZyme model

    Example:
        >>> model = load_pretrained("clipzyme", device="cuda")
        >>> # Model is automatically downloaded if needed
    """
    cache_dir = Path(cache_dir)

    # Look for checkpoint
    checkpoint_patterns = [
        cache_dir / "clipzyme_model.ckpt",
        cache_dir / "clipzyme_model" / "*.ckpt",
        cache_dir / "*.ckpt",
    ]

    checkpoint_path = None
    for pattern in checkpoint_patterns:
        matches = list(cache_dir.glob(str(pattern.name)))
        if matches:
            checkpoint_path = matches[0]
            break

    # Download if missing
    if checkpoint_path is None or not checkpoint_path.exists():
        if download_if_missing:
            logger.info("Checkpoint not found locally, downloading from Zenodo...")
            from checkpoints.downloader import download_clipzyme_checkpoint

            checkpoint_dir = download_clipzyme_checkpoint(
                output_dir=str(cache_dir),
                extract=True
            )

            # Find .ckpt file in extracted directory
            checkpoint_path = list(Path(checkpoint_dir).glob("**/*.ckpt"))[0]
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {cache_dir} and download_if_missing=False"
            )

    # Load checkpoint
    logger.info(f"Loading pretrained model from {checkpoint_path}")
    model = load_official_checkpoint(checkpoint_path, device=device)

    return model


__all__ = [
    'CheckpointLoader',
    'load_official_checkpoint',
    'load_checkpoint',
    'load_pretrained',
]
