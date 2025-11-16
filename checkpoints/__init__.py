"""
CLIPZyme Checkpoint Management

Tools for downloading, loading, and converting CLIPZyme model checkpoints.

Supports:
- Official CLIPZyme checkpoints from Zenodo
- PyTorch Lightning checkpoints
- Standard PyTorch state_dict
- Checkpoint conversion between formats
"""

from checkpoints.downloader import ZenodoDownloader, download_clipzyme_checkpoint
from checkpoints.loader import (
    load_official_checkpoint,
    load_checkpoint,
    CheckpointLoader,
)
from checkpoints.converter import (
    convert_official_to_local,
    StateDict Converter,
)
from checkpoints.validator import validate_checkpoint, compare_checkpoints

__all__ = [
    'ZenodoDownloader',
    'download_clipzyme_checkpoint',
    'load_official_checkpoint',
    'load_checkpoint',
    'CheckpointLoader',
    'convert_official_to_local',
    'StateDictConverter',
    'validate_checkpoint',
    'compare_checkpoints',
]
