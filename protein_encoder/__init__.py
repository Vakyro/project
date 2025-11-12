"""
Protein encoder module for CLIPZyme.

Provides ESM2-based protein encoding with attention pooling and projection.
"""

from .esm_model import ProteinEncoderESM2, ProjectionHead
from .pooling import AttentionPool, mean_pool, cls_pool
from .utils import chunk_sequence, encode_long_sequence

__all__ = [
    'ProteinEncoderESM2',
    'ProjectionHead',
    'AttentionPool',
    'mean_pool',
    'cls_pool',
    'chunk_sequence',
    'encode_long_sequence'
]
