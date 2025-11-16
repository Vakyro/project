"""
Evaluation Metrics for CLIPZyme

Implements all metrics from the CLIPZyme paper:
- BEDROC₈₅: Primary metric (α=85 emphasizes early recognition)
- BEDROC₂₀, BEDROC₅₀: Additional BEDROC variants
- Top-K Accuracy: Hit rate in top K predictions
- Enrichment Factor: Enrichment at different percentiles
- AUROC: Area under ROC curve
- Average Precision: Area under PR curve
- Hit Rate @ N: Fraction of actives in top N

Paper Reference:
Mikhael et al. (2024) "CLIPZyme: Reaction-Conditioned Virtual Screening of Enzymes"
Reports BEDROC₈₅ as primary metric.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""

    # BEDROC variants (primary metric for CLIPZyme)
    bedroc_85: float = 0.0  # α=85 (paper's primary metric)
    bedroc_50: float = 0.0  # α=50
    bedroc_20: float = 0.0  # α=20 (standard)

    # Top-K Accuracy
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    top10_accuracy: float = 0.0
    top50_accuracy: float = 0.0
    top100_accuracy: float = 0.0

    # Enrichment Factor
    ef_1pct: float = 0.0   # 1% of database
    ef_5pct: float = 0.0   # 5% of database
    ef_10pct: float = 0.0  # 10% of database

    # Area under curves
    auroc: float = 0.0
    auprc: float = 0.0  # Average Precision

    # Hit Rate @ N
    hit_rate_10: float = 0.0
    hit_rate_50: float = 0.0
    hit_rate_100: float = 0.0

    # Additional statistics
    num_actives: int = 0
    num_total: int = 0
    active_fraction: float = 0.0

    # Optional metadata
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            # BEDROC (α values)
            'BEDROC_85': self.bedroc_85,
            'BEDROC_50': self.bedroc_50,
            'BEDROC_20': self.bedroc_20,

            # Top-K
            'Top1': self.top1_accuracy,
            'Top5': self.top5_accuracy,
            'Top10': self.top10_accuracy,
            'Top50': self.top50_accuracy,
            'Top100': self.top100_accuracy,

            # Enrichment
            'EF_1%': self.ef_1pct,
            'EF_5%': self.ef_5pct,
            'EF_10%': self.ef_10pct,

            # AUC
            'AUROC': self.auroc,
            'AUPRC': self.auprc,

            # Hit Rate
            'HitRate@10': self.hit_rate_10,
            'HitRate@50': self.hit_rate_50,
            'HitRate@100': self.hit_rate_100,

            # Stats
            'num_actives': self.num_actives,
            'num_total': self.num_total,
            'active_fraction': self.active_fraction,
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationMetrics(\n"
            f"  BEDROC_85={self.bedroc_85:.4f} [PRIMARY],\n"
            f"  BEDROC_50={self.bedroc_50:.4f},\n"
            f"  BEDROC_20={self.bedroc_20:.4f},\n"
            f"  Top1={self.top1_accuracy:.4f},\n"
            f"  AUROC={self.auroc:.4f},\n"
            f"  Actives: {self.num_actives}/{self.num_total}\n"
            f")"
        )


class CLIPZymeMetrics:
    """
    Compute evaluation metrics exactly as reported in CLIPZyme paper.

    The paper uses BEDROC₈₅ as the primary metric for early recognition.
    """

    @staticmethod
    def compute_bedroc(
        scores: np.ndarray,
        labels: np.ndarray,
        alpha: float = 85.0,
        decreasing: bool = True
    ) -> float:
        """
        Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).

        BEDROC emphasizes early recognition using exponential weighting.
        Higher α values give more weight to early hits.

        Args:
            scores: Predicted scores (higher = more likely active)
            labels: True binary labels (1 = active, 0 = inactive)
            alpha: Exponential weighting parameter
                  - α=20: Standard, moderate early emphasis
                  - α=50: Stronger early emphasis
                  - α=85: Very strong early emphasis (CLIPZyme paper)
            decreasing: Sort scores in decreasing order

        Returns:
            BEDROC score in [0, 1], where 1 is perfect

        Reference:
            Truchon & Bayly (2007) "Evaluating Virtual Screening Methods"
        """
        scores = np.asarray(scores)
        labels = np.asarray(labels)

        if len(scores) != len(labels):
            raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(labels)} labels")

        # Sort by scores
        sorted_indices = np.argsort(scores)
        if decreasing:
            sorted_indices = sorted_indices[::-1]

        sorted_labels = labels[sorted_indices]

        # Count actives and inactives
        n_actives = int(np.sum(labels))
        n_total = len(labels)

        if n_actives == 0 or n_actives == n_total:
            logger.warning("Cannot compute BEDROC: only one class present")
            return 0.0

        # Random hit rate
        ra = n_actives / n_total

        # Calculate exponentially weighted sum
        exp_sum = 0.0
        for i, label in enumerate(sorted_labels):
            rank = i + 1
            exp_sum += label * np.exp(-alpha * rank / n_total)

        # Normalization factor
        numerator = (np.sinh(alpha / 2) / np.sinh(alpha)) * exp_sum / n_actives

        # Expected value for random ranking
        random_sum = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n_total) - 1)

        # BEDROC formula
        bedroc = (numerator - random_sum) / (1 - random_sum)

        # Clamp to [0, 1]
        bedroc = max(0.0, min(1.0, bedroc))

        return float(bedroc)

    @staticmethod
    def compute_top_k_accuracy(
        ranked_ids: List[str],
        active_ids: List[str],
        k: int
    ) -> float:
        """
        Compute Top-K accuracy: fraction of actives in top K.

        Args:
            ranked_ids: IDs ranked by score (best first)
            active_ids: Set of active (positive) IDs
            k: Number of top predictions to consider

        Returns:
            Fraction of actives found in top K
        """
        if not active_ids:
            return 0.0

        k = min(k, len(ranked_ids))
        top_k = set(ranked_ids[:k])
        active_set = set(active_ids)

        hits = len(top_k & active_set)
        recall = hits / len(active_set)

        return recall

    @staticmethod
    def compute_enrichment_factor(
        ranked_ids: List[str],
        active_ids: List[str],
        fraction: float
    ) -> float:
        """
        Compute Enrichment Factor at given fraction of database.

        EF = (Hits @ fraction) / (Expected hits @ fraction)

        Args:
            ranked_ids: IDs ranked by score
            active_ids: Set of active IDs
            fraction: Fraction of database (e.g., 0.01 for 1%)

        Returns:
            Enrichment factor (>1 means better than random)
        """
        if not active_ids:
            return 0.0

        n_total = len(ranked_ids)
        n_actives = len(active_ids)
        n_check = max(1, int(fraction * n_total))

        top_n = set(ranked_ids[:n_check])
        active_set = set(active_ids)

        hits = len(top_n & active_set)
        expected = n_actives * fraction

        if expected == 0:
            return 0.0

        return hits / expected

    @staticmethod
    def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute Area Under ROC Curve."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            logger.warning("sklearn not available, returning 0")
            return 0.0

        if len(np.unique(labels)) < 2:
            logger.warning("Only one class present, cannot compute AUROC")
            return 0.0

        return float(roc_auc_score(labels, scores))

    @staticmethod
    def compute_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute Average Precision (area under PR curve)."""
        try:
            from sklearn.metrics import average_precision_score
        except ImportError:
            logger.warning("sklearn not available, returning 0")
            return 0.0

        if len(np.unique(labels)) < 2:
            logger.warning("Only one class present, cannot compute AP")
            return 0.0

        return float(average_precision_score(labels, scores))

    @staticmethod
    def compute_hit_rate_at_n(
        ranked_ids: List[str],
        active_ids: List[str],
        n: int
    ) -> float:
        """
        Compute hit rate @ N: fraction of actives in top N.

        Similar to Top-K accuracy but more commonly used terminology.
        """
        return CLIPZymeMetrics.compute_top_k_accuracy(ranked_ids, active_ids, n)


def compute_all_metrics(
    ranked_ids: List[str],
    scores: Union[np.ndarray, torch.Tensor, List[float]],
    active_ids: List[str],
    alpha_values: List[float] = [85.0, 50.0, 20.0],
    top_k_values: List[int] = [1, 5, 10, 50, 100],
    ef_fractions: List[float] = [0.01, 0.05, 0.10]
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics at once.

    This is the main function to use for comprehensive evaluation.

    Args:
        ranked_ids: IDs ranked by score (best first)
        scores: Corresponding scores for each ID
        active_ids: Set of active (ground truth) IDs
        alpha_values: Alpha values for BEDROC
        top_k_values: K values for Top-K accuracy
        ef_fractions: Fractions for Enrichment Factor

    Returns:
        EvaluationMetrics object with all computed metrics

    Example:
        >>> metrics = compute_all_metrics(
        ...     ranked_ids=result.ranked_protein_ids,
        ...     scores=result.scores,
        ...     active_ids=["P12345", "P67890"]
        ... )
        >>> print(f"BEDROC_85: {metrics.bedroc_85:.4f}")
    """
    # Convert scores to numpy
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    scores = np.asarray(scores)

    # Create binary labels
    active_set = set(active_ids)
    labels = np.array([1 if rid in active_set else 0 for rid in ranked_ids])

    # Initialize metrics
    metrics = EvaluationMetrics()

    metrics.num_actives = len(active_ids)
    metrics.num_total = len(ranked_ids)
    metrics.active_fraction = metrics.num_actives / metrics.num_total if metrics.num_total > 0 else 0.0

    # Compute BEDROC for different α values
    if len(np.unique(labels)) > 1:
        for alpha in alpha_values:
            bedroc = CLIPZymeMetrics.compute_bedroc(scores, labels, alpha=alpha)

            if alpha == 85.0:
                metrics.bedroc_85 = bedroc
            elif alpha == 50.0:
                metrics.bedroc_50 = bedroc
            elif alpha == 20.0:
                metrics.bedroc_20 = bedroc

    # Compute Top-K accuracy
    for k in top_k_values:
        if k <= len(ranked_ids):
            acc = CLIPZymeMetrics.compute_top_k_accuracy(ranked_ids, active_ids, k)

            if k == 1:
                metrics.top1_accuracy = acc
            elif k == 5:
                metrics.top5_accuracy = acc
            elif k == 10:
                metrics.top10_accuracy = acc
            elif k == 50:
                metrics.top50_accuracy = acc
            elif k == 100:
                metrics.top100_accuracy = acc

    # Compute Enrichment Factor
    for frac in ef_fractions:
        ef = CLIPZymeMetrics.compute_enrichment_factor(ranked_ids, active_ids, frac)

        if abs(frac - 0.01) < 1e-6:
            metrics.ef_1pct = ef
        elif abs(frac - 0.05) < 1e-6:
            metrics.ef_5pct = ef
        elif abs(frac - 0.10) < 1e-6:
            metrics.ef_10pct = ef

    # Compute AUROC and AUPRC
    if len(np.unique(labels)) > 1:
        metrics.auroc = CLIPZymeMetrics.compute_auroc(scores, labels)
        metrics.auprc = CLIPZymeMetrics.compute_average_precision(scores, labels)

    # Compute Hit Rate @ N
    metrics.hit_rate_10 = CLIPZymeMetrics.compute_hit_rate_at_n(ranked_ids, active_ids, 10)
    metrics.hit_rate_50 = CLIPZymeMetrics.compute_hit_rate_at_n(ranked_ids, active_ids, 50)
    metrics.hit_rate_100 = CLIPZymeMetrics.compute_hit_rate_at_n(ranked_ids, active_ids, 100)

    return metrics


def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
    """
    Aggregate metrics across multiple evaluations.

    Computes mean of each metric.

    Args:
        metrics_list: List of EvaluationMetrics objects

    Returns:
        EvaluationMetrics with averaged values
    """
    if not metrics_list:
        return EvaluationMetrics()

    aggregated = EvaluationMetrics()

    # Average all numeric fields
    numeric_fields = [
        'bedroc_85', 'bedroc_50', 'bedroc_20',
        'top1_accuracy', 'top5_accuracy', 'top10_accuracy', 'top50_accuracy', 'top100_accuracy',
        'ef_1pct', 'ef_5pct', 'ef_10pct',
        'auroc', 'auprc',
        'hit_rate_10', 'hit_rate_50', 'hit_rate_100',
        'active_fraction'
    ]

    for field in numeric_fields:
        values = [getattr(m, field) for m in metrics_list]
        setattr(aggregated, field, np.mean(values))

    # Sum counts
    aggregated.num_actives = sum(m.num_actives for m in metrics_list)
    aggregated.num_total = sum(m.num_total for m in metrics_list)

    # Add metadata
    aggregated.metadata['num_evaluations'] = len(metrics_list)
    aggregated.metadata['std_bedroc_85'] = np.std([m.bedroc_85 for m in metrics_list])

    return aggregated


__all__ = [
    'EvaluationMetrics',
    'CLIPZymeMetrics',
    'compute_all_metrics',
    'aggregate_metrics',
]
