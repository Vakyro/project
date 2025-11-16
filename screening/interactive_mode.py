"""
Interactive Screening Mode

For screening single reactions against a protein database.
Memory-efficient, suitable for exploratory analysis and small-scale screening.

Usage:
    screener = InteractiveScreener(model, screening_set)
    results = screener.screen_reaction(reaction_smiles, top_k=100)
"""

import torch
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
import logging

from screening.screening_set import ScreeningSet, ProteinDatabase
from screening.ranking import (
    ScreeningResult,
    rank_proteins_for_reaction,
    evaluate_screening
)

logger = logging.getLogger(__name__)


@dataclass
class InteractiveScreeningConfig:
    """Configuration for interactive screening."""
    top_k: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    return_embeddings: bool = False
    evaluate: bool = False


class InteractiveScreener:
    """
    Interactive screening mode for single reactions.

    This mode is memory-efficient and suitable for:
    - Exploratory analysis
    - Single-reaction screening
    - Interactive Jupyter notebook usage
    - Small-scale screening (<1000 reactions)

    For large-scale screening (>1000 reactions), use BatchedScreener instead.
    """

    def __init__(
        self,
        model,
        screening_set: ScreeningSet,
        protein_database: Optional[ProteinDatabase] = None,
        config: Optional[InteractiveScreeningConfig] = None
    ):
        """
        Initialize interactive screener.

        Args:
            model: CLIPZyme model or tuple of (protein_encoder, reaction_encoder)
            screening_set: ScreeningSet with pre-embedded proteins
            protein_database: Optional database with protein sequences/metadata
            config: Screening configuration
        """
        self.config = config or InteractiveScreeningConfig()

        # Handle model types
        if isinstance(model, tuple):
            self.protein_encoder, self.reaction_encoder = model
            self.model = None
        else:
            self.model = model
            self.protein_encoder = model.protein_encoder if hasattr(model, 'protein_encoder') else None
            self.reaction_encoder = model.reaction_encoder if hasattr(model, 'reaction_encoder') else None

        self.screening_set = screening_set
        self.protein_database = protein_database

        # Move models to device
        if self.model is not None:
            self.model = self.model.to(self.config.device)
            self.model.eval()
        if self.reaction_encoder is not None:
            self.reaction_encoder = self.reaction_encoder.to(self.config.device)
            self.reaction_encoder.eval()

        logger.info(
            f"Initialized InteractiveScreener with {len(screening_set)} proteins "
            f"on {self.config.device}"
        )

    def encode_reaction(self, reaction_smiles: str) -> torch.Tensor:
        """
        Encode a single reaction to embedding.

        Args:
            reaction_smiles: Reaction SMILES (must be atom-mapped)

        Returns:
            Reaction embedding tensor (embedding_dim,)
        """
        with torch.no_grad():
            if self.model is not None and hasattr(self.model, 'encode_reactions'):
                # Use full model
                embedding = self.model.encode_reactions(
                    [reaction_smiles],
                    device=self.config.device
                )
            elif self.reaction_encoder is not None:
                # Use reaction encoder directly
                embedding = self.reaction_encoder.encode(
                    [reaction_smiles],
                    device=self.config.device
                )
            else:
                raise RuntimeError("No reaction encoder available")

            # Return single embedding
            if embedding.dim() == 2:
                embedding = embedding.squeeze(0)

            return embedding

    def screen_reaction(
        self,
        reaction_smiles: str,
        top_k: Optional[int] = None,
        true_positives: Optional[List[str]] = None,
        reaction_id: Optional[str] = None
    ) -> ScreeningResult:
        """
        Screen a single reaction against the protein database.

        Args:
            reaction_smiles: Reaction SMILES (atom-mapped)
            top_k: Number of top matches to return (default: config.top_k)
            true_positives: Optional list of true positive protein IDs for evaluation
            reaction_id: Optional identifier for this reaction

        Returns:
            ScreeningResult with ranked proteins and metrics
        """
        top_k = top_k or self.config.top_k
        reaction_id = reaction_id or reaction_smiles

        # Encode reaction
        try:
            reaction_embedding = self.encode_reaction(reaction_smiles)
        except Exception as e:
            logger.error(f"Failed to encode reaction {reaction_id}: {e}")
            raise

        # Rank proteins
        ranked_protein_ids, scores, _ = rank_proteins_for_reaction(
            reaction_embedding=reaction_embedding,
            screening_set=self.screening_set,
            top_k=top_k,
            return_all_scores=False
        )

        # Build result
        result = ScreeningResult(
            reaction_id=reaction_id,
            ranked_protein_ids=ranked_protein_ids,
            scores=scores,
            true_positives=true_positives
        )

        # Evaluate if true positives provided
        if true_positives and self.config.evaluate:
            metrics = evaluate_screening(
                ranked_protein_ids=ranked_protein_ids,
                scores=scores,
                true_positives=true_positives
            )
            result.metrics = metrics

            # Log primary metric
            if 'BEDROC_20' in metrics:
                logger.info(
                    f"Screened {reaction_id}: "
                    f"BEDROC_20={metrics['BEDROC_20']:.3f}, "
                    f"Top1={metrics.get('Top1_Accuracy', 0):.3f}"
                )

        return result

    def screen_reactions(
        self,
        reaction_smiles_list: List[str],
        top_k: Optional[int] = None,
        true_positives_list: Optional[List[List[str]]] = None,
        reaction_ids: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> List[ScreeningResult]:
        """
        Screen multiple reactions sequentially.

        Args:
            reaction_smiles_list: List of reaction SMILES
            top_k: Number of top matches per reaction
            true_positives_list: Optional list of true positives for each reaction
            reaction_ids: Optional identifiers for reactions
            show_progress: Show progress bar

        Returns:
            List of ScreeningResult objects
        """
        from tqdm import tqdm

        if reaction_ids is None:
            reaction_ids = [f"reaction_{i}" for i in range(len(reaction_smiles_list))]

        if true_positives_list is None:
            true_positives_list = [None] * len(reaction_smiles_list)

        results = []

        iterator = zip(reaction_smiles_list, reaction_ids, true_positives_list)
        if show_progress:
            iterator = tqdm(
                list(iterator),
                desc="Screening reactions"
            )

        for rxn_smiles, rxn_id, true_pos in iterator:
            try:
                result = self.screen_reaction(
                    reaction_smiles=rxn_smiles,
                    top_k=top_k,
                    true_positives=true_pos,
                    reaction_id=rxn_id
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to screen {rxn_id}: {e}")
                continue

        return results

    def get_protein_info(self, protein_id: str) -> Optional[Dict]:
        """
        Get information about a protein from the database.

        Args:
            protein_id: Protein identifier

        Returns:
            Dictionary with protein info or None if not found
        """
        if self.protein_database is None:
            return None

        if protein_id not in self.protein_database:
            return None

        entry = self.protein_database[protein_id]
        info = {
            'protein_id': entry.protein_id,
            'sequence': entry.sequence,
            'sequence_length': len(entry.sequence),
            **entry.metadata
        }

        # Add embedding if available
        if protein_id in self.screening_set:
            info['has_embedding'] = True
        else:
            info['has_embedding'] = False

        return info

    def compare_reactions(
        self,
        reaction_smiles_1: str,
        reaction_smiles_2: str,
        top_k: int = 10
    ) -> Dict:
        """
        Compare two reactions by their screening results.

        Args:
            reaction_smiles_1: First reaction SMILES
            reaction_smiles_2: Second reaction SMILES
            top_k: Number of top matches to compare

        Returns:
            Dictionary with comparison results
        """
        # Screen both reactions
        result1 = self.screen_reaction(reaction_smiles_1, top_k=top_k)
        result2 = self.screen_reaction(reaction_smiles_2, top_k=top_k)

        # Find overlaps
        set1 = set(result1.ranked_protein_ids)
        set2 = set(result2.ranked_protein_ids)

        overlap = set1 & set2
        unique1 = set1 - set2
        unique2 = set2 - set1

        # Compute embedding similarity
        emb1 = self.encode_reaction(reaction_smiles_1)
        emb2 = self.encode_reaction(reaction_smiles_2)
        reaction_similarity = (emb1 @ emb2).item()

        return {
            'reaction_1': reaction_smiles_1,
            'reaction_2': reaction_smiles_2,
            'reaction_similarity': reaction_similarity,
            'overlap_count': len(overlap),
            'overlap_proteins': list(overlap),
            'unique_to_1': list(unique1),
            'unique_to_2': list(unique2),
            'jaccard_index': len(overlap) / len(set1 | set2) if (set1 | set2) else 0,
        }

    def find_similar_proteins(
        self,
        protein_id: str,
        top_k: int = 10
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Find proteins similar to a given protein in embedding space.

        Args:
            protein_id: Query protein ID
            top_k: Number of similar proteins to return

        Returns:
            List of similar protein IDs and their similarity scores
        """
        if protein_id not in self.screening_set:
            raise ValueError(f"Protein {protein_id} not found in screening set")

        # Get query embedding
        query_embedding = self.screening_set.get_embedding(protein_id)

        # Compute similarities with all proteins
        scores, indices, protein_ids = self.screening_set.compute_similarity(
            query_embedding,
            top_k=top_k + 1,  # +1 because query itself will be included
            return_scores=True
        )

        # Remove query protein from results
        filtered_ids = []
        filtered_scores = []
        for pid, score in zip(protein_ids, scores):
            if pid != protein_id:
                filtered_ids.append(pid)
                filtered_scores.append(score)

        filtered_scores = torch.stack(filtered_scores[:top_k])
        filtered_ids = filtered_ids[:top_k]

        return filtered_ids, filtered_scores

    def __repr__(self) -> str:
        return (
            f"InteractiveScreener("
            f"num_proteins={len(self.screening_set)}, "
            f"device={self.config.device})"
        )


__all__ = [
    'InteractiveScreeningConfig',
    'InteractiveScreener',
]
