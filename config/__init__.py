"""
Configuration system for CLIPZyme.

Provides typed configuration classes using dataclasses with YAML support.
"""

from .config import (
    ProteinEncoderConfig,
    ReactionEncoderConfig,
    TrainingConfig,
    DataConfig,
    CLIPZymeConfig,
    load_config,
    save_config,
)

__all__ = [
    'ProteinEncoderConfig',
    'ReactionEncoderConfig',
    'TrainingConfig',
    'DataConfig',
    'CLIPZymeConfig',
    'load_config',
    'save_config',
]
