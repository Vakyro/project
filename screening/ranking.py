"""
Ranking and Evaluation Metrics for Virtual Screening

Implements metrics used in CLIPZyme paper:
- BEDROC (Boltzmann-Enhanced Discrimination of ROC)
- Top-K accuracy
- Enrichment Factor
- Area Under ROC Curve (AUROC)
- Average Precision

BEDROC is the primary metric reported in CLIPZyme paper.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """Results from a screening run."""
    reaction_id: str
    ranked_protein_ids: List[str]
    scores: torch.Tensor
    true_positives: Optional[List[str]] = None
    metrics: Optional[Dict[str, float]] = None


def compute_bedroc(
    scores: Union[torch.Tensor, np.ndarray, List[float]],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    alpha: float = 20.0,
    decreasing: bool = True
) -> float:
    """
    Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    BEDROC emphasizes early recognition (top of ranked list) using an exponential
    weighting function. It is bounded between 0 and 1.

    Args:
        scores: Predicted scores for each sample (higher = more likely positive)
        labels: True binary labels (1 = positive, 0 = negative)
        alpha: Early recognition parameter (default: 20.0 as in CLIPZyme)
               Higher alpha = more emphasis on early recognition
        decreasing: If True, sort scores in decreasing order (default: True)

    Returns:
        BEDROC score in [0, 1]

    Reference:
        Truchon & Bayly (2007) "Evaluating Virtual Screening Methods:
        Good and Bad Metrics for the 'Early Recognition' Problem"
    """
    # Convert to numpy
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if len(scores) != len(labels):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(labels)} labels")

    # Sort by scores
    sorted_indices = np.argsort(scores)
    if decreasing:
        sorted_indices = sorted_indices[::-1]

    sorted_labels = labels[sorted_indices]

    # Number of positives and negatives
    n_positives = int(np.sum(labels))
    n_negatives = len(labels) - n_positives

    if n_positives == 0 or n_negatives == 0:
        logger.warning("Cannot compute BEDROC: only one class present")
        return 0.0

    # Calculate BEDROC
    n = len(labels)
    ra = n_positives / n  # Random hit rate

    # Calculate exponentially weighted sum
    exp_sum = 0.0
    for i, label in enumerate(sorted_labels):
        rank = i + 1
        exp_sum += label * np.exp(-alpha * rank / n)

    # Normalization factor
    numerator = (np.sinh(alpha / 2) / np.sinh(alpha)) * exp_sum / n_positives

    # Expected value for random ranking
    random_sum = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)

    # BEDROC formula
    bedroc = (numerator - random_sum) / (1 - random_sum)

    # Clamp to [0, 1] (numerical stability)
    bedroc = max(0.0, min(1.0, bedroc))

    return float(bedroc)


def compute_topk_accuracy(
    ranked_protein_ids: List[str],
    true_positives: List[str],
    k: int
) -> float:
    """
    Compute Top-K accuracy: fraction of true positives in top-K predictions.

    Args:
        ranked_protein_ids: Protein IDs ranked by score (best first)
        true_positives: Set of true positive protein IDs
        k: Number of top predictions to consider

    Returns:
        Accuracy in [0, 1]
    """
    if not true_positives:
        return 0.0

    k = min(k, len(ranked_protein_ids))
    top_k = set(ranked_protein_ids[:k])
    true_set = set(true_positives)

    hits = len(top_k & true_set)
    accuracy = hits / len(true_set)

    return accuracy


def compute_enrichment_factor(
    ranked_protein_ids: List[str],
    true_positives: List[str],
    fraction: float = 0.01
) -> float:
    """
    Compute Enrichment Factor at a given fraction of the dataset.

    EF(x%) = (Hits at x%) / (Expected hits at x%)

    Args:
        ranked_protein_ids: Protein IDs ranked by score (best first)
        true_positives: Set of true positive protein IDs
        fraction: Fraction of dataset to consider (default: 0.01 = 1%)

    Returns:
        Enrichment factor (> 1 means better than random)
    """
    if not true_positives:
        return 0.0

    n_total = len(ranked_protein_ids)
    n_positives = len(true_positives)
    n_check = max(1, int(fraction * n_total))

    top_n = set(ranked_protein_ids[:n_check])
    true_set = set(true_positives)

    hits = len(top_n & true_set)
    expected = n_positives * fraction

    if expected == 0:
        return 0.0

    ef = hits / expected
    return ef


def compute_auroc(
    scores: Union[torch.Tensor, np.ndarray, List[float]],
    labels: Union[torch.Tensor, np.ndarray, List[int]]
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        scores: Predicted scores
        labels: True binary labels

    Returns:
        AUROC in [0, 1]
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("sklearn not available, cannot compute AUROC")
        return 0.0

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        logger.warning("Cannot compute AUROC: only one class present")
        return 0.0

    return float(roc_auc_score(labels, scores))


def compute_average_precision(
    scores: Union[torch.Tensor, np.ndarray, List[float]],
    labels: Union[torch.Tensor, np.ndarray, List[int]]
) -> float:
    """
    Compute Average Precision (area under precision-recall curve).

    Args:
        scores: Predicted scores
        labels: True binary labels

    Returns:
        Average Precision in [0, 1]
    """
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        logger.warning("sklearn not available, cannot compute AP")
        return 0.0

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        logger.warning("Cannot compute AP: only one class present")
        return 0.0

    return float(average_precision_score(labels, scores))


def rank_proteins_for_reaction(
    reaction_embedding: torch.Tensor,
    screening_set,
    top_k: Optional[int] = None,
    return_all_scores: bool = False
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
    """
    Rank all proteins in screening set by similarity to a reaction.

    Args:
        reaction_embedding: Reaction embedding tensor (embedding_dim,)
        screening_set: ScreeningSet with protein embeddings
        top_k: Return only top-k (default: None = all)
        return_all_scores: If True, return scores for all proteins (not just top-k)

    Returns:
        ranked_protein_ids: List of protein IDs (best first)
        top_scores: Scores for returned proteins
        all_scores: All scores if return_all_scores=True, else None
    """
    # Compute similarities
    if top_k is not None:
        scores, indices, protein_ids = screening_set.compute_similarity(
            reaction_embedding,
            top_k=top_k,
            return_scores=True
        )
        all_scores = None
    else:
        # Get all
        all_scores = screening_set.compute_similarity(
            reaction_embedding,
            return_scores=False
        )
        scores, indices = torch.sort(all_scores, descending=True)
        protein_ids = [screening_set.protein_ids[idx] for idx in indices.cpu().tolist()]

    if not return_all_scores:
        all_scores = None

    return protein_ids, scores, all_scores


def evaluate_screening(
    ranked_protein_ids: List[str],
    scores: torch.Tensor,
    true_positives: List[str],
    alpha: float = 20.0,
    top_k_values: List[int] = [1, 5, 10, 50, 100],
    ef_fractions: List[float] = [0.01, 0.05, 0.10]
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a screening result.

    Args:
        ranked_protein_ids: Protein IDs ranked by score (best first)
        scores: Similarity scores for ranked proteins
        true_positives: List of true positive protein IDs
        alpha: BEDROC alpha parameter (default: 20.0)
        top_k_values: K values for Top-K accuracy
        ef_fractions: Fractions for Enrichment Factor

    Returns:
        Dictionary with all metrics
    """
    # Create binary labels
    true_set = set(true_positives)
    labels = [1 if pid in true_set else 0 for pid in ranked_protein_ids]

    metrics = {}

    # BEDROC (primary metric for CLIPZyme)
    if len(set(labels)) > 1:  # Need both classes
        # For BEDROC at different alpha values
        for a in [20.0, 50.0, 85.0]:
            bedroc = compute_bedroc(scores.cpu().numpy(), labels, alpha=a)
            metrics[f'BEDROC_{int(a)}'] = bedroc

    # Top-K accuracy
    for k in top_k_values:
        if k <= len(ranked_protein_ids):
            acc = compute_topk_accuracy(ranked_protein_ids, true_positives, k)
            metrics[f'Top{k}_Accuracy'] = acc

    # Enrichment Factor
    for frac in ef_fractions:
        ef = compute_enrichment_factor(ranked_protein_ids, true_positives, frac)
        metrics[f'EF_{int(frac*100)}%'] = ef

    # AUROC and AP
    if len(set(labels)) > 1:
        auroc = compute_auroc(scores, labels)
        avg_prec = compute_average_precision(scores, labels)
        metrics['AUROC'] = auroc
        metrics['AvgPrecision'] = avg_prec

    # Basic stats
    metrics['num_true_positives'] = len(true_positives)
    metrics['num_predictions'] = len(ranked_protein_ids)

    return metrics


def batch_evaluate_screening(
    results: List[ScreeningResult],
    alpha: float = 20.0
) -> Dict[str, float]:
    """
    Evaluate multiple screening results and aggregate metrics.

    Args:
        results: List of ScreeningResult objects
        alpha: BEDROC alpha parameter

    Returns:
        Dictionary with averaged metrics across all results
    """
    all_metrics = []

    for result in results:
        if result.true_positives is None:
            logger.warning(f"Skipping {result.reaction_id}: no true positives provided")
            continue

        metrics = evaluate_screening(
            ranked_protein_ids=result.ranked_protein_ids,
            scores=result.scores,
            true_positives=result.true_positives,
            alpha=alpha
        )
        all_metrics.append(metrics)

    if not all_metrics:
        return {}

    # Average all metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        if key.startswith('num_'):
            # Sum for counts
            aggregated[key] = sum(m[key] for m in all_metrics)
        else:
            # Average for metrics
            aggregated[key] = np.mean([m[key] for m in all_metrics])

    aggregated['num_reactions'] = len(all_metrics)

    return aggregated


__all__ = [
    'ScreeningResult',
    'compute_bedroc',
    'compute_topk_accuracy',
    'compute_enrichment_factor',
    'compute_auroc',
    'compute_average_precision',
    'rank_proteins_for_reaction',
    'evaluate_screening',
    'batch_evaluate_screening',
]
