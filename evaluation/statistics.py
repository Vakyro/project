"""
Statistical Analysis for Evaluation

Provides statistical tools for robust evaluation:
- Confidence intervals via bootstrap
- Significance testing
- Effect size computation
- Multiple testing correction
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from scipy import stats
import logging

from evaluation.metrics import EvaluationMetrics, compute_all_metrics

logger = logging.getLogger(__name__)


def bootstrap_metrics(
    ranked_ids_list: List[List[str]],
    scores_list: List[np.ndarray],
    active_ids_list: List[List[str]],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    metric_name: str = 'bedroc_85',
    random_seed: Optional[int] = 42
) -> Dict:
    """
    Compute confidence intervals via bootstrap resampling.

    Args:
        ranked_ids_list: List of ranked IDs for each test case
        scores_list: List of scores for each test case
        active_ids_list: List of active IDs for each test case
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        metric_name: Metric to bootstrap (attribute of EvaluationMetrics)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with bootstrap statistics

    Example:
        >>> ci = bootstrap_metrics(ranked_ids, scores, active_ids)
        >>> print(f"BEDROC_85: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = len(ranked_ids_list)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Compute metric on bootstrap sample
        metrics_list = []
        for idx in indices:
            metrics = compute_all_metrics(
                ranked_ids=ranked_ids_list[idx],
                scores=scores_list[idx],
                active_ids=active_ids_list[idx]
            )
            metrics_list.append(getattr(metrics, metric_name))

        # Average across bootstrap sample
        bootstrap_values.append(np.mean(metrics_list))

    bootstrap_values = np.array(bootstrap_values)

    # Compute statistics
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_values, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_values, (1 - alpha/2) * 100)

    return {
        'mean': np.mean(bootstrap_values),
        'std': np.std(bootstrap_values),
        'median': np.median(bootstrap_values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'metric': metric_name
    }


def compute_confidence_intervals(
    metrics_list: List[EvaluationMetrics],
    confidence_level: float = 0.95
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute confidence intervals for all metrics.

    Uses normal approximation (parametric).

    Args:
        metrics_list: List of EvaluationMetrics
        confidence_level: Confidence level

    Returns:
        Dictionary mapping metric names to (mean, ci_lower, ci_upper)
    """
    if not metrics_list:
        return {}

    # Extract numeric fields
    numeric_fields = [
        'bedroc_85', 'bedroc_50', 'bedroc_20',
        'top1_accuracy', 'top5_accuracy', 'top10_accuracy',
        'ef_1pct', 'ef_5pct', 'ef_10pct',
        'auroc', 'auprc'
    ]

    results = {}

    for field in numeric_fields:
        values = [getattr(m, field) for m in metrics_list]
        values = np.array(values)

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(len(values))

        # Compute confidence interval
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        margin = z_score * sem

        ci_lower = mean - margin
        ci_upper = mean + margin

        results[field] = (mean, ci_lower, ci_upper)

    return results


def significance_test(
    metrics_a: List[EvaluationMetrics],
    metrics_b: List[EvaluationMetrics],
    metric_name: str = 'bedroc_85',
    test_type: str = 'paired_t'
) -> Dict:
    """
    Test if two sets of metrics are significantly different.

    Args:
        metrics_a: First set of metrics
        metrics_b: Second set of metrics
        metric_name: Metric to compare
        test_type: Type of test ('paired_t', 'unpaired_t', 'wilcoxon', 'mann_whitney')

    Returns:
        Dictionary with test results

    Example:
        >>> result = significance_test(baseline_metrics, improved_metrics)
        >>> print(f"p-value: {result['p_value']:.4f}")
        >>> print(f"Significant: {result['significant']}")
    """
    values_a = np.array([getattr(m, metric_name) for m in metrics_a])
    values_b = np.array([getattr(m, metric_name) for m in metrics_b])

    if test_type == 'paired_t':
        if len(values_a) != len(values_b):
            raise ValueError("Paired test requires equal sample sizes")
        statistic, p_value = stats.ttest_rel(values_a, values_b)

    elif test_type == 'unpaired_t':
        statistic, p_value = stats.ttest_ind(values_a, values_b)

    elif test_type == 'wilcoxon':
        if len(values_a) != len(values_b):
            raise ValueError("Wilcoxon test requires equal sample sizes")
        statistic, p_value = stats.wilcoxon(values_a, values_b)

    elif test_type == 'mann_whitney':
        statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')

    else:
        raise ValueError(f"Unknown test type: {test_type}")

    # Effect size (Cohen's d for t-tests)
    if 't' in test_type:
        pooled_std = np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2)
        cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = None

    return {
        'test_type': test_type,
        'metric': metric_name,
        'mean_a': np.mean(values_a),
        'mean_b': np.mean(values_b),
        'difference': np.mean(values_a) - np.mean(values_b),
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'n_a': len(values_a),
        'n_b': len(values_b)
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Bonferroni correction for multiple testing.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        Dictionary with corrected results
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = [p < corrected_alpha for p in p_values]

    return {
        'n_tests': n_tests,
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'p_values': p_values,
        'significant': significant,
        'n_significant': sum(significant)
    }


def compute_effect_size(
    metrics_a: List[EvaluationMetrics],
    metrics_b: List[EvaluationMetrics],
    metric_name: str = 'bedroc_85'
) -> float:
    """
    Compute Cohen's d effect size.

    Effect size interpretation:
    - Small: d = 0.2
    - Medium: d = 0.5
    - Large: d = 0.8

    Args:
        metrics_a: First set of metrics
        metrics_b: Second set of metrics
        metric_name: Metric to compare

    Returns:
        Cohen's d effect size
    """
    values_a = np.array([getattr(m, metric_name) for m in metrics_a])
    values_b = np.array([getattr(m, metric_name) for m in metrics_b])

    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)

    pooled_std = np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2)

    if pooled_std == 0:
        return 0.0

    cohens_d = (mean_a - mean_b) / pooled_std
    return cohens_d


def statistical_summary(
    metrics_list: List[EvaluationMetrics],
    metric_name: str = 'bedroc_85'
) -> Dict:
    """
    Compute comprehensive statistical summary.

    Args:
        metrics_list: List of metrics
        metric_name: Metric to summarize

    Returns:
        Dictionary with statistical summary
    """
    values = np.array([getattr(m, metric_name) for m in metrics_list])

    return {
        'metric': metric_name,
        'n': len(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values, ddof=1),
        'sem': stats.sem(values),
        'min': np.min(values),
        'max': np.max(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'skewness': stats.skew(values),
        'kurtosis': stats.kurtosis(values)
    }


__all__ = [
    'bootstrap_metrics',
    'compute_confidence_intervals',
    'significance_test',
    'bonferroni_correction',
    'compute_effect_size',
    'statistical_summary',
]
