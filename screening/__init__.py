"""
CLIPZyme Screening Module

Provides tools for virtual screening of enzymes against chemical reactions:
- ScreeningSet: Manage large collections of pre-embedded proteins
- InteractiveScreener: Screen single reactions
- BatchedScreener: High-throughput screening with multi-GPU support
- Ranking metrics: BEDROC, Top-K accuracy, etc.
"""

from screening.screening_set import ScreeningSet, ProteinDatabase
from screening.interactive_mode import InteractiveScreener
from screening.batched_mode import BatchedScreener
from screening.ranking import (
    compute_bedroc,
    compute_topk_accuracy,
    compute_enrichment_factor,
    rank_proteins_for_reaction,
)
from screening.cache import EmbeddingCache

__all__ = [
    'ScreeningSet',
    'ProteinDatabase',
    'InteractiveScreener',
    'BatchedScreener',
    'compute_bedroc',
    'compute_topk_accuracy',
    'compute_enrichment_factor',
    'rank_proteins_for_reaction',
    'EmbeddingCache',
]
