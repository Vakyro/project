"""
Common utilities and shared components for CLIPZyme.

This module contains shared components used across both protein and reaction encoders:
- Projection heads
- MLP modules
- Constants
- Interfaces
- Factory patterns
"""

from .modules import ProjectionHead, MLP
from .constants import ESMConfig, EGNNConfig, ChemistryConfig, TrainingConfig
from .interfaces import ProteinEncoder, ReactionEncoder

__all__ = [
    'ProjectionHead',
    'MLP',
    'ESMConfig',
    'EGNNConfig',
    'ChemistryConfig',
    'TrainingConfig',
    'ProteinEncoder',
    'ReactionEncoder',
]
