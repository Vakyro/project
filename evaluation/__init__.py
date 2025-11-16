"""
CLIPZyme Evaluation Module

Complete evaluation system for enzyme screening models.

Implements all metrics from CLIPZyme paper:
- BEDROC (α=20, 50, 85) - Primary metric
- Top-K Accuracy (K=1, 5, 10, 50, 100)
- Enrichment Factor (1%, 5%, 10%)
- AUROC and Average Precision
- Hit Rate @ N

Plus additional analysis:
- Statistical significance tests
- ROC and PR curve plotting
- Performance stratification (by EC class, etc.)
- Benchmark comparisons
"""

from evaluation.metrics import (
    EvaluationMetrics,
    compute_all_metrics,
    CLIPZymeMetrics,
)
from evaluation.benchmark import (
    BenchmarkEvaluator,
    run_benchmark,
    compare_to_paper_results,
)
from evaluation.visualization import (
    plot_roc_curve,
    plot_pr_curve,
    plot_bedroc_comparison,
    plot_top_k_accuracy,
    create_evaluation_report,
)
from evaluation.statistics import (
    compute_confidence_intervals,
    significance_test,
    bootstrap_metrics,
)

__all__ = [
    'EvaluationMetrics',
    'compute_all_metrics',
    'CLIPZymeMetrics',
    'BenchmarkEvaluator',
    'run_benchmark',
    'compare_to_paper_results',
    'plot_roc_curve',
    'plot_pr_curve',
    'plot_bedroc_comparison',
    'plot_top_k_accuracy',
    'create_evaluation_report',
    'compute_confidence_intervals',
    'significance_test',
    'bootstrap_metrics',
]
