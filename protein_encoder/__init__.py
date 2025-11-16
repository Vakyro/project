"""
Protein encoder module for CLIPZyme.

Production implementation: ProteinEncoderEGNN (ESM2 + E(n)-equivariant GNN)
matching the CLIPZyme paper architecture exactly.
"""

from .egnn import ProteinEncoderEGNN
from .pooling import AttentionPool, mean_pool, cls_pool
from .utils import chunk_sequence, encode_long_sequence

__all__ = [
    'ProteinEncoderEGNN',
    'AttentionPool',
    'mean_pool',
    'cls_pool',
    'chunk_sequence',
    'encode_long_sequence'
]
