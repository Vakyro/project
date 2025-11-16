"""
Benchmark Evaluation for CLIPZyme

Compare model performance against CLIPZyme paper results.

Paper Results (from Mikhael et al. 2024):
- BEDROC₈₅: 44.69% (without EC information)
- BEDROC₈₅: 75.57% (with EC2 prediction)
- Test set: EnzymeMap reactions
- Screening set: 260,197 enzymes

This module provides tools to:
- Run standard benchmarks
- Compare to paper results
- Evaluate on different test sets
- Generate comparison reports
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import json

from evaluation.metrics import EvaluationMetrics, compute_all_metrics, aggregate_metrics
from screening import InteractiveScreener, BatchedScreener, ScreeningSet

logger = logging.getLogger(__name__)


class CLIPZymePaperResults:
    """
    Published results from CLIPZyme paper.

    Reference:
    Mikhael et al. (2024) "CLIPZyme: Reaction-Conditioned Virtual Screening of Enzymes"
    """

    # Main results (Table 1 in paper)
    BEDROC_85_NO_EC = 0.4469  # Without EC information
    BEDROC_85_WITH_EC2 = 0.7557  # With EC2 prediction

    # Additional metrics (from paper)
    TOP1_ACCURACY = 0.25  # Approximate
    TOP10_ACCURACY = 0.60  # Approximate

    # Experimental setup
    SCREENING_SET_SIZE = 260197  # Number of enzymes
    TEST_SET = "EnzymeMap"
    ALPHA = 85.0  # Primary BEDROC parameter

    @classmethod
    def get_baseline_metrics(cls) -> EvaluationMetrics:
        """Get baseline metrics from paper."""
        metrics = EvaluationMetrics()
        metrics.bedroc_85 = cls.BEDROC_85_NO_EC
        metrics.metadata = {
            'source': 'CLIPZyme Paper',
            'dataset': cls.TEST_SET,
            'screening_set_size': cls.SCREENING_SET_SIZE
        }
        return metrics

    @classmethod
    def get_with_ec_metrics(cls) -> EvaluationMetrics:
        """Get metrics with EC prediction from paper."""
        metrics = EvaluationMetrics()
        metrics.bedroc_85 = cls.BEDROC_85_WITH_EC2
        metrics.metadata = {
            'source': 'CLIPZyme Paper + EC2',
            'dataset': cls.TEST_SET,
            'screening_set_size': cls.SCREENING_SET_SIZE
        }
        return metrics


class BenchmarkEvaluator:
    """
    Run benchmark evaluations and compare to paper results.

    Example:
        >>> evaluator = BenchmarkEvaluator(model, screening_set)
        >>> results = evaluator.run_benchmark(test_reactions, test_labels)
        >>> comparison = evaluator.compare_to_paper(results)
    """

    def __init__(
        self,
        model,
        screening_set: ScreeningSet,
        device: str = "cuda"
    ):
        """
        Initialize benchmark evaluator.

        Args:
            model: CLIPZyme model
            screening_set: Pre-embedded protein screening set
            device: Device for inference
        """
        self.model = model
        self.screening_set = screening_set
        self.device = device

        self.screener = InteractiveScreener(
            model=model,
            screening_set=screening_set
        )

    def run_benchmark(
        self,
        test_reactions: List[str],
        true_labels: List[List[str]],
        reaction_ids: Optional[List[str]] = None,
        top_k: int = 100,
        show_progress: bool = True
    ) -> Dict:
        """
        Run benchmark evaluation on test set.

        Args:
            test_reactions: List of reaction SMILES
            true_labels: List of true positive protein IDs for each reaction
            reaction_ids: Optional reaction identifiers
            top_k: Number of top predictions to consider
            show_progress: Show progress bar

        Returns:
            Dictionary with results and metrics
        """
        logger.info("Running benchmark evaluation...")

        if reaction_ids is None:
            reaction_ids = [f"rxn_{i}" for i in range(len(test_reactions))]

        # Screen all reactions
        results = self.screener.screen_reactions(
            reaction_smiles_list=test_reactions,
            reaction_ids=reaction_ids,
            true_positives_list=true_labels,
            top_k=top_k,
            show_progress=show_progress
        )

        # Compute metrics for each reaction
        all_metrics = []
        for result in results:
            if result.true_positives:
                metrics = compute_all_metrics(
                    ranked_ids=result.ranked_protein_ids,
                    scores=result.scores,
                    active_ids=result.true_positives
                )
                all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics)

        logger.info(f"\nBenchmark Results:")
        logger.info(f"  BEDROC_85: {aggregated.bedroc_85:.4f}")
        logger.info(f"  Top-1 Acc: {aggregated.top1_accuracy:.4f}")
        logger.info(f"  AUROC: {aggregated.auroc:.4f}")

        return {
            'individual_results': results,
            'individual_metrics': all_metrics,
            'aggregated_metrics': aggregated,
            'num_reactions': len(test_reactions),
            'screening_set_size': len(self.screening_set)
        }

    def compare_to_paper(
        self,
        benchmark_results: Dict,
        use_ec: bool = False
    ) -> Dict:
        """
        Compare benchmark results to CLIPZyme paper.

        Args:
            benchmark_results: Results from run_benchmark()
            use_ec: Compare to EC-enhanced results if True

        Returns:
            Comparison dictionary
        """
        our_metrics = benchmark_results['aggregated_metrics']

        if use_ec:
            paper_metrics = CLIPZymePaperResults.get_with_ec_metrics()
            comparison_type = "With EC2 Prediction"
        else:
            paper_metrics = CLIPZymePaperResults.get_baseline_metrics()
            comparison_type = "Without EC Information"

        comparison = {
            'comparison_type': comparison_type,
            'our_bedroc_85': our_metrics.bedroc_85,
            'paper_bedroc_85': paper_metrics.bedroc_85,
            'difference': our_metrics.bedroc_85 - paper_metrics.bedroc_85,
            'relative_performance': our_metrics.bedroc_85 / paper_metrics.bedroc_85 if paper_metrics.bedroc_85 > 0 else 0,
            'our_metrics': our_metrics,
            'paper_metrics': paper_metrics,
        }

        logger.info(f"\n=== Comparison to CLIPZyme Paper ({comparison_type}) ===")
        logger.info(f"Our BEDROC_85:   {comparison['our_bedroc_85']:.4f}")
        logger.info(f"Paper BEDROC_85: {comparison['paper_bedroc_85']:.4f}")
        logger.info(f"Difference:      {comparison['difference']:+.4f}")
        logger.info(f"Relative:        {comparison['relative_performance']:.2%}")

        return comparison

    def evaluate_by_ec_class(
        self,
        test_reactions: List[str],
        true_labels: List[List[str]],
        ec_classes: List[str],
        top_k: int = 100
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate performance stratified by EC class.

        Args:
            test_reactions: List of reaction SMILES
            true_labels: True positive protein IDs
            ec_classes: EC class for each reaction (e.g., "1.1.1.1")
            top_k: Top K to consider

        Returns:
            Dictionary mapping EC class to metrics
        """
        logger.info("Evaluating by EC class...")

        # Group by EC class (first digit)
        ec_groups = {}
        for rxn, labels, ec in zip(test_reactions, true_labels, ec_classes):
            ec_main = ec.split('.')[0] if ec else "unknown"
            if ec_main not in ec_groups:
                ec_groups[ec_main] = []
            ec_groups[ec_main].append((rxn, labels))

        # Evaluate each group
        results = {}
        for ec_class, reactions_labels in ec_groups.items():
            rxns = [r[0] for r in reactions_labels]
            labels = [r[1] for r in reactions_labels]

            if len(rxns) < 5:  # Skip small groups
                continue

            logger.info(f"  EC {ec_class}: {len(rxns)} reactions")

            group_results = self.run_benchmark(
                test_reactions=rxns,
                true_labels=labels,
                top_k=top_k,
                show_progress=False
            )

            results[f"EC_{ec_class}"] = group_results['aggregated_metrics']

        return results


def run_benchmark(
    model,
    screening_set: ScreeningSet,
    test_reactions: List[str],
    true_labels: List[List[str]],
    output_dir: Optional[str] = None,
    compare_to_paper: bool = True
) -> Dict:
    """
    Convenience function to run complete benchmark evaluation.

    Args:
        model: CLIPZyme model
        screening_set: Screening set
        test_reactions: Test reaction SMILES
        true_labels: True positive protein IDs
        output_dir: Directory to save results
        compare_to_paper: Compare to paper results

    Returns:
        Dictionary with all results

    Example:
        >>> from models import load_pretrained
        >>> from screening import ScreeningSet
        >>> from evaluation import run_benchmark
        >>>
        >>> model = load_pretrained("clipzyme")
        >>> screening_set = ScreeningSet().load_from_pickle("screening_set.p")
        >>>
        >>> results = run_benchmark(
        ...     model, screening_set,
        ...     test_reactions, true_labels,
        ...     output_dir="results/benchmark"
        ... )
    """
    evaluator = BenchmarkEvaluator(model, screening_set)

    # Run benchmark
    results = evaluator.run_benchmark(
        test_reactions=test_reactions,
        true_labels=true_labels,
        show_progress=True
    )

    # Compare to paper
    if compare_to_paper:
        comparison = evaluator.compare_to_paper(results)
        results['paper_comparison'] = comparison

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = output_dir / "benchmark_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results['aggregated_metrics'].to_dict(), f, indent=2)

        # Save comparison
        if compare_to_paper:
            comparison_file = output_dir / "paper_comparison.json"
            comparison_data = {
                'our_bedroc_85': comparison['our_bedroc_85'],
                'paper_bedroc_85': comparison['paper_bedroc_85'],
                'difference': comparison['difference'],
                'relative_performance': comparison['relative_performance']
            }
            with open(comparison_file, 'w') as f:
                json.dump(comparison_data, f, indent=2)

        logger.info(f"Results saved to {output_dir}")

    return results


def compare_to_paper_results(our_metrics: EvaluationMetrics) -> None:
    """
    Print comparison to paper results.

    Args:
        our_metrics: Computed metrics
    """
    paper_baseline = CLIPZymePaperResults.get_baseline_metrics()
    paper_with_ec = CLIPZymePaperResults.get_with_ec_metrics()

    print("\n" + "=" * 70)
    print("COMPARISON TO CLIPZYME PAPER RESULTS")
    print("=" * 70)

    print("\nOur Results:")
    print(f"  BEDROC_85: {our_metrics.bedroc_85:.4f}")
    print(f"  BEDROC_50: {our_metrics.bedroc_50:.4f}")
    print(f"  BEDROC_20: {our_metrics.bedroc_20:.4f}")
    print(f"  Top-1 Acc: {our_metrics.top1_accuracy:.4f}")
    print(f"  AUROC:     {our_metrics.auroc:.4f}")

    print("\nPaper Results (Baseline):")
    print(f"  BEDROC_85: {paper_baseline.bedroc_85:.4f}")
    print(f"  Dataset: {paper_baseline.metadata['dataset']}")

    print("\nPaper Results (With EC2):")
    print(f"  BEDROC_85: {paper_with_ec.bedroc_85:.4f}")

    print("\nComparison:")
    diff_baseline = our_metrics.bedroc_85 - paper_baseline.bedroc_85
    diff_ec = our_metrics.bedroc_85 - paper_with_ec.bedroc_85

    print(f"  vs Baseline: {diff_baseline:+.4f} ({diff_baseline/paper_baseline.bedroc_85:+.1%})")
    print(f"  vs With EC2: {diff_ec:+.4f} ({diff_ec/paper_with_ec.bedroc_85:+.1%})")

    print("\n" + "=" * 70)


__all__ = [
    'CLIPZymePaperResults',
    'BenchmarkEvaluator',
    'run_benchmark',
    'compare_to_paper_results',
]
