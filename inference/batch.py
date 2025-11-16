"""
Batch inference utilities for CLIPZyme.

Provides optimized batch processing for large-scale screening.
"""

from typing import List, Optional, Callable
from pathlib import Path
import torch
from tqdm import tqdm
import logging

from .predictor import CLIPZymePredictor, ScreeningResult, PredictorConfig


logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    Batch predictor for efficient large-scale screening.

    Optimized for processing thousands of reactions.
    """

    def __init__(
        self,
        predictor: CLIPZymePredictor,
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Initialize batch predictor.

        Args:
            predictor: CLIPZymePredictor instance
            batch_size: Batch size for processing
            show_progress: Show progress bar
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.show_progress = show_progress

    def screen_reactions(
        self,
        reactions: List[str],
        top_k: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ScreeningResult]:
        """
        Screen multiple reactions in batches.

        Args:
            reactions: List of reaction SMILES
            top_k: Number of top proteins per reaction
            progress_callback: Optional callback(current, total)

        Returns:
            List of ScreeningResults
        """
        results = []
        total = len(reactions)

        iterator = tqdm(
            range(0, total, self.batch_size),
            desc="Screening reactions",
            disable=not self.show_progress
        )

        for start_idx in iterator:
            end_idx = min(start_idx + self.batch_size, total)
            batch_reactions = reactions[start_idx:end_idx]

            # Process batch
            batch_results = self._process_batch(batch_reactions, top_k)
            results.extend(batch_results)

            # Progress callback
            if progress_callback:
                progress_callback(end_idx, total)

        return results

    def _process_batch(
        self,
        batch_reactions: List[str],
        top_k: Optional[int]
    ) -> List[ScreeningResult]:
        """Process a batch of reactions."""
        batch_results = []

        # Encode all reactions at once
        with torch.no_grad():
            reaction_embs = self.predictor.model.encode_reactions(batch_reactions)

        # Screen each reaction
        protein_embs = self.predictor.screening_set.embeddings

        for i, reaction_smiles in enumerate(batch_reactions):
            reaction_emb = reaction_embs[i:i+1]

            # Compute similarities
            similarities = (reaction_emb @ protein_embs.T).squeeze(0)

            # Get top-k
            k = top_k or self.predictor.config.top_k
            top_k_scores, top_k_indices = torch.topk(
                similarities,
                k=min(k, len(self.predictor.screening_set))
            )

            # Get protein names
            top_proteins = [
                self.predictor.screening_set.protein_names[idx]
                for idx in top_k_indices.cpu().tolist()
            ]

            # Convert scores
            scores = top_k_scores.cpu().tolist()

            result = ScreeningResult(
                reaction_smiles=reaction_smiles,
                top_proteins=top_proteins,
                scores=scores,
                metadata={'batch_processed': True}
            )

            batch_results.append(result)

        return batch_results


def batch_screen_reactions(
    checkpoint_path: str,
    reactions: List[str],
    screening_set_path: str,
    top_k: int = 100,
    batch_size: int = 32,
    device: str = 'cuda',
    output_path: Optional[Path] = None
) -> List[ScreeningResult]:
    """
    Convenience function for batch screening.

    Args:
        checkpoint_path: Path to model checkpoint
        reactions: List of reaction SMILES
        screening_set_path: Path to screening set
        top_k: Number of top proteins
        batch_size: Batch size
        device: Device to use
        output_path: Optional path to save results

    Returns:
        List of ScreeningResults
    """
    logger.info("Starting batch screening...")

    # Create predictor
    config = PredictorConfig(
        device=device,
        batch_size=batch_size,
        top_k=top_k
    )

    predictor = CLIPZymePredictor.from_checkpoint(checkpoint_path, config)
    predictor.load_screening_set(screening_set_path)

    # Create batch predictor
    batch_predictor = BatchPredictor(
        predictor=predictor,
        batch_size=batch_size,
        show_progress=True
    )

    # Screen reactions
    results = batch_predictor.screen_reactions(reactions, top_k=top_k)

    logger.info(f"Screened {len(reactions)} reactions")

    # Save results if requested
    if output_path:
        logger.info(f"Saving results to {output_path}")
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

    return results


# Export
__all__ = [
    'BatchPredictor',
    'batch_screen_reactions',
]
