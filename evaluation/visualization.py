"""
Visualization Tools for Evaluation Results

Creates publication-quality plots for:
- ROC curves
- Precision-Recall curves
- BEDROC comparisons
- Top-K accuracy plots
- Enrichment plots
- Evaluation reports
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

from evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Args:
        scores: Predicted scores
        labels: True binary labels
        title: Plot title
        save_path: Path to save figure
        show: If True, display the plot

    Returns:
        Matplotlib figure
    """
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        logger.error("sklearn required for ROC curve plotting")
        return None

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_pr_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot Precision-Recall curve.

    Args:
        scores: Predicted scores
        labels: True binary labels
        title: Plot title
        save_path: Path to save figure
        show: If True, display the plot

    Returns:
        Matplotlib figure
    """
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        logger.error("sklearn required for PR curve plotting")
        return None

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')

    # Baseline
    baseline = np.sum(labels) / len(labels)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2,
            linestyle='--', label=f'Random (baseline = {baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PR curve to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_bedroc_comparison(
    metrics_dict: Dict[str, EvaluationMetrics],
    title: str = "BEDROC Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot BEDROC comparison across different models or settings.

    Args:
        metrics_dict: Dictionary mapping names to EvaluationMetrics
        title: Plot title
        save_path: Path to save figure
        show: If True, display the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(metrics_dict.keys())
    bedroc_85 = [metrics_dict[name].bedroc_85 for name in names]
    bedroc_50 = [metrics_dict[name].bedroc_50 for name in names]
    bedroc_20 = [metrics_dict[name].bedroc_20 for name in names]

    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, bedroc_85, width, label='BEDROC₈₅ (Primary)', color='#e74c3c')
    ax.bar(x, bedroc_50, width, label='BEDROC₅₀', color='#3498db')
    ax.bar(x + width, bedroc_20, width, label='BEDROC₂₀', color='#2ecc71')

    ax.set_ylabel('BEDROC Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved BEDROC comparison to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_top_k_accuracy(
    metrics_dict: Dict[str, EvaluationMetrics],
    k_values: List[int] = [1, 5, 10, 50, 100],
    title: str = "Top-K Accuracy",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot Top-K accuracy across different K values.

    Args:
        metrics_dict: Dictionary mapping names to EvaluationMetrics
        k_values: K values to plot
        title: Plot title
        save_path: Path to save figure
        show: If True, display the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, metrics in metrics_dict.items():
        accuracies = []
        for k in k_values:
            if k == 1:
                accuracies.append(metrics.top1_accuracy)
            elif k == 5:
                accuracies.append(metrics.top5_accuracy)
            elif k == 10:
                accuracies.append(metrics.top10_accuracy)
            elif k == 50:
                accuracies.append(metrics.top50_accuracy)
            elif k == 100:
                accuracies.append(metrics.top100_accuracy)

        ax.plot(k_values, accuracies, marker='o', linewidth=2, label=name)

    ax.set_xlabel('K (Top-K)')
    ax.set_ylabel('Accuracy (Recall)')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Top-K accuracy plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_enrichment_factor(
    metrics_dict: Dict[str, EvaluationMetrics],
    fractions: List[float] = [0.01, 0.05, 0.10],
    title: str = "Enrichment Factor",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot Enrichment Factor at different database fractions.

    Args:
        metrics_dict: Dictionary mapping names to EvaluationMetrics
        fractions: Database fractions to plot
        title: Plot title
        save_path: Path to save figure
        show: If True, display the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(metrics_dict.keys())
    x = np.arange(len(names))
    width = 0.25

    ef_1pct = [metrics_dict[name].ef_1pct for name in names]
    ef_5pct = [metrics_dict[name].ef_5pct for name in names]
    ef_10pct = [metrics_dict[name].ef_10pct for name in names]

    ax.bar(x - width, ef_1pct, width, label='EF 1%', color='#e74c3c')
    ax.bar(x, ef_5pct, width, label='EF 5%', color='#3498db')
    ax.bar(x + width, ef_10pct, width, label='EF 10%', color='#2ecc71')

    # Add random baseline
    ax.axhline(y=1.0, color='gray', linestyle='--', label='Random (EF=1)')

    ax.set_ylabel('Enrichment Factor')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved enrichment factor plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_evaluation_report(
    metrics: EvaluationMetrics,
    scores: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    name: str = "evaluation"
) -> Dict[str, Path]:
    """
    Create comprehensive evaluation report with all plots.

    Args:
        metrics: Computed evaluation metrics
        scores: Predicted scores
        labels: True labels
        output_dir: Directory to save plots
        name: Name prefix for files

    Returns:
        Dictionary mapping plot types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # ROC curve
    roc_path = output_dir / f"{name}_roc_curve.png"
    plot_roc_curve(scores, labels, save_path=str(roc_path), show=False)
    saved_files['roc'] = roc_path

    # PR curve
    pr_path = output_dir / f"{name}_pr_curve.png"
    plot_pr_curve(scores, labels, save_path=str(pr_path), show=False)
    saved_files['pr'] = pr_path

    # Metrics summary plot
    summary_path = output_dir / f"{name}_summary.png"
    _plot_metrics_summary(metrics, save_path=str(summary_path))
    saved_files['summary'] = summary_path

    # Save metrics to text file
    metrics_path = output_dir / f"{name}_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Metrics: {name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("PRIMARY METRIC (CLIPZyme Paper):\n")
        f.write(f"  BEDROC_85: {metrics.bedroc_85:.4f}\n\n")

        f.write("BEDROC Variants:\n")
        f.write(f"  BEDROC_85: {metrics.bedroc_85:.4f}\n")
        f.write(f"  BEDROC_50: {metrics.bedroc_50:.4f}\n")
        f.write(f"  BEDROC_20: {metrics.bedroc_20:.4f}\n\n")

        f.write("Top-K Accuracy:\n")
        f.write(f"  Top-1:   {metrics.top1_accuracy:.4f}\n")
        f.write(f"  Top-5:   {metrics.top5_accuracy:.4f}\n")
        f.write(f"  Top-10:  {metrics.top10_accuracy:.4f}\n")
        f.write(f"  Top-50:  {metrics.top50_accuracy:.4f}\n")
        f.write(f"  Top-100: {metrics.top100_accuracy:.4f}\n\n")

        f.write("Enrichment Factor:\n")
        f.write(f"  EF 1%:  {metrics.ef_1pct:.2f}\n")
        f.write(f"  EF 5%:  {metrics.ef_5pct:.2f}\n")
        f.write(f"  EF 10%: {metrics.ef_10pct:.2f}\n\n")

        f.write("Area Under Curves:\n")
        f.write(f"  AUROC: {metrics.auroc:.4f}\n")
        f.write(f"  AUPRC: {metrics.auprc:.4f}\n\n")

        f.write("Statistics:\n")
        f.write(f"  Actives: {metrics.num_actives}\n")
        f.write(f"  Total: {metrics.num_total}\n")
        f.write(f"  Active Fraction: {metrics.active_fraction:.4f}\n")

    saved_files['metrics_txt'] = metrics_path

    logger.info(f"Created evaluation report in {output_dir}")
    return saved_files


def _plot_metrics_summary(
    metrics: EvaluationMetrics,
    save_path: str
):
    """Internal function to plot metrics summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # BEDROC comparison
    ax = axes[0, 0]
    bedroc_values = [metrics.bedroc_85, metrics.bedroc_50, metrics.bedroc_20]
    bedroc_labels = ['BEDROC₈₅', 'BEDROC₅₀', 'BEDROC₂₀']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    ax.bar(bedroc_labels, bedroc_values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title('BEDROC Variants')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)

    # Top-K accuracy
    ax = axes[0, 1]
    k_values = [1, 5, 10, 50, 100]
    top_k_values = [
        metrics.top1_accuracy,
        metrics.top5_accuracy,
        metrics.top10_accuracy,
        metrics.top50_accuracy,
        metrics.top100_accuracy
    ]
    ax.plot(k_values, top_k_values, marker='o', linewidth=2, color='#3498db')
    ax.set_xlabel('K')
    ax.set_ylabel('Accuracy')
    ax.set_title('Top-K Accuracy')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Enrichment Factor
    ax = axes[1, 0]
    ef_values = [metrics.ef_1pct, metrics.ef_5pct, metrics.ef_10pct]
    ef_labels = ['1%', '5%', '10%']
    ax.bar(ef_labels, ef_values, color='#2ecc71')
    ax.axhline(y=1.0, color='gray', linestyle='--', label='Random')
    ax.set_ylabel('Enrichment Factor')
    ax.set_title('Enrichment at Different %')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # AUC metrics
    ax = axes[1, 1]
    auc_values = [metrics.auroc, metrics.auprc]
    auc_labels = ['AUROC', 'AUPRC']
    ax.bar(auc_labels, auc_values, color=['#9b59b6', '#f39c12'])
    ax.set_ylabel('Score')
    ax.set_title('Area Under Curve Metrics')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Evaluation Metrics Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


__all__ = [
    'plot_roc_curve',
    'plot_pr_curve',
    'plot_bedroc_comparison',
    'plot_top_k_accuracy',
    'plot_enrichment_factor',
    'create_evaluation_report',
]
