"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np
from evaluation.metrics import (
    bedroc_score,
    top_k_accuracy,
    enrichment_factor
)


class TestBEDROC:
    """Test suite for BEDROC metric."""

    def test_perfect_ranking(self):
        """Test BEDROC with perfect ranking."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([6, 5, 4, 3, 2, 1])  # Perfect ranking

        bedroc = bedroc_score(y_true, scores, alpha=20.0)

        assert bedroc > 0.9, "Perfect ranking should have high BEDROC"

    def test_random_ranking(self):
        """Test BEDROC with random ranking."""
        np.random.seed(42)
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.random.rand(6)

        bedroc = bedroc_score(y_true, scores, alpha=20.0)

        # Random ranking should be around 0.5
        assert 0.0 <= bedroc <= 1.0

    def test_worst_ranking(self):
        """Test BEDROC with worst ranking."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([1, 2, 3, 4, 5, 6])  # Worst ranking

        bedroc = bedroc_score(y_true, scores, alpha=20.0)

        assert bedroc < 0.1, "Worst ranking should have low BEDROC"


class TestTopKAccuracy:
    """Test suite for Top-K accuracy."""

    def test_top_1_accuracy_perfect(self):
        """Test top-1 accuracy with perfect prediction."""
        y_true = np.array([1, 0, 0, 0])
        scores = np.array([4, 3, 2, 1])

        acc = top_k_accuracy(y_true, scores, k=1)

        assert acc == 1.0

    def test_top_3_accuracy(self):
        """Test top-3 accuracy."""
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([5, 4, 3, 2, 1])

        # Both true positives in top 3
        acc = top_k_accuracy(y_true, scores, k=3)

        assert acc == 1.0

    def test_top_k_no_positives(self):
        """Test top-k when no positives in top-k."""
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([1, 2, 5, 4, 3])  # Positives ranked last

        acc = top_k_accuracy(y_true, scores, k=2)

        assert acc == 0.0


class TestEnrichmentFactor:
    """Test suite for Enrichment Factor."""

    def test_enrichment_factor_perfect(self):
        """Test EF with perfect enrichment."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([6, 5, 4, 3, 2, 1])

        ef = enrichment_factor(y_true, scores, fraction=0.5)

        # Perfect enrichment should be 2.0 at 50%
        assert ef == pytest.approx(2.0, abs=0.1)

    def test_enrichment_factor_random(self):
        """Test EF with random ranking."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        np.random.seed(42)
        scores = np.random.rand(6)

        ef = enrichment_factor(y_true, scores, fraction=0.5)

        # Random should be around 1.0
        assert 0.5 <= ef <= 1.5
