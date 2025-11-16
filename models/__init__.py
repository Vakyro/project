"""
CLIPZyme models and builders.

Contains the unified CLIPZyme model and builder patterns.
"""

from .clipzyme import CLIPZymeModel
from .builder import CLIPZymeBuilder


# Convenient checkpoint loading functions
def load_pretrained(
    model_name: str = "clipzyme",
    cache_dir: str = "data/checkpoints",
    device: str = "cpu",
    download_if_missing: bool = True
):
    """
    Load a pretrained CLIPZyme model.

    Args:
        model_name: Model name ('clipzyme', 'clipzyme_faithful')
        cache_dir: Directory to cache downloaded models
        device: Device to load to ('cpu', 'cuda', etc.)
        download_if_missing: Auto-download from Zenodo if not found

    Returns:
        Pretrained CLIPZyme model

    Example:
        >>> from models import load_pretrained
        >>> model = load_pretrained("clipzyme", device="cuda")
        >>> # Model automatically downloaded from Zenodo if needed
        >>> embeddings = model.encode_reactions(["[C:1]=[O:2]>>[C:1]-[O:2]"])
    """
    from checkpoints.loader import load_pretrained as _load_pretrained
    return _load_pretrained(model_name, cache_dir, device, download_if_missing)


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """
    Load a checkpoint file (official or local format).

    Args:
        checkpoint_path: Path to checkpoint file (.ckpt, .pt, .pth)
        device: Device to load to

    Returns:
        Loaded CLIPZyme model

    Example:
        >>> from models import load_checkpoint
        >>> model = load_checkpoint("data/checkpoints/clipzyme_model.ckpt")
        >>> model.eval()
    """
    from checkpoints.loader import load_official_checkpoint
    return load_official_checkpoint(checkpoint_path, device)


__all__ = [
    'CLIPZymeModel',
    'CLIPZymeBuilder',
    'load_pretrained',
    'load_checkpoint',
]
