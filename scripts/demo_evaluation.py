#!/usr/bin/env python3
"""
Demo: CLIPZyme Evaluation System

Demonstrates how to use the evaluation module with BEDROC₈₅ and other metrics.

This demo shows:
1. Computing all metrics for a screening result
2. Running benchmark evaluation
3. Comparing to CLIPZyme paper results
4. Creating visualizations
5. Statistical analysis with bootstrap

Usage:
    python scripts/demo_evaluation.py
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_metrics():
    """Demo 1: Compute basic metrics from screening results."""
    from evaluation import compute_all_metrics

    logger.info("\n" + "="*70)
    logger.info("DEMO 1: Computing Evaluation Metrics")
    logger.info("="*70)

    # Simulate a screening result
    # In practice, these would come from model.screen_reaction()
    np.random.seed(42)

    # 100 proteins in screening set
    protein_ids = [f"P{i:05d}" for i in range(100)]

    # Similarity scores (higher = more similar)
    scores = np.random.rand(100)

    # Known active enzymes (ground truth)
    active_ids = ["P00005", "P00012", "P00023", "P00056", "P00089"]

    # Sort by scores (descending)
    sorted_indices = np.argsort(scores)[::-1]
    ranked_ids = [protein_ids[i] for i in sorted_indices]
    sorted_scores = scores[sorted_indices]

    logger.info(f"\nScreening set size: {len(protein_ids)}")
    logger.info(f"Known active enzymes: {len(active_ids)}")

    # Compute all metrics
    metrics = compute_all_metrics(
        ranked_ids=ranked_ids,
        scores=sorted_scores,
        active_ids=active_ids
    )

    logger.info("\n📊 Evaluation Metrics:")
    logger.info(f"  BEDROC_85 (Primary): {metrics.bedroc_85:.4f}")
    logger.info(f"  BEDROC_50:           {metrics.bedroc_50:.4f}")
    logger.info(f"  BEDROC_20:           {metrics.bedroc_20:.4f}")
    logger.info(f"  Top-1 Accuracy:      {metrics.top1_accuracy:.4f}")
    logger.info(f"  Top-5 Accuracy:      {metrics.top5_accuracy:.4f}")
    logger.info(f"  Top-10 Accuracy:     {metrics.top10_accuracy:.4f}")
    logger.info(f"  EF 1%:               {metrics.ef_1pct:.2f}")
    logger.info(f"  EF 5%:               {metrics.ef_5pct:.2f}")
    logger.info(f"  AUROC:               {metrics.auroc:.4f}")
    logger.info(f"  AUPRC:               {metrics.auprc:.4f}")

    return metrics


def demo_paper_comparison():
    """Demo 2: Compare to CLIPZyme paper results."""
    from evaluation import compare_to_paper_results, CLIPZymePaperResults

    logger.info("\n" + "="*70)
    logger.info("DEMO 2: Comparing to CLIPZyme Paper Results")
    logger.info("="*70)

    # Get paper results
    paper_baseline = CLIPZymePaperResults.get_baseline_metrics()
    paper_with_ec = CLIPZymePaperResults.get_with_ec_metrics()

    logger.info("\n📄 Published Results from CLIPZyme Paper:")
    logger.info(f"  Baseline (no EC):   BEDROC_85 = {paper_baseline.bedroc_85:.4f}")
    logger.info(f"  With EC2 prediction: BEDROC_85 = {paper_with_ec.bedroc_85:.4f}")
    logger.info(f"  Dataset: {paper_baseline.metadata['dataset']}")
    logger.info(f"  Screening set: {paper_baseline.metadata['screening_set_size']:,} enzymes")

    # Simulate our results
    from evaluation import EvaluationMetrics
    our_metrics = EvaluationMetrics()
    our_metrics.bedroc_85 = 0.4650  # Example: slightly better than baseline
    our_metrics.bedroc_50 = 0.4210
    our_metrics.bedroc_20 = 0.3890
    our_metrics.top1_accuracy = 0.2600
    our_metrics.auroc = 0.8450

    logger.info("\n📊 Our Results:")
    logger.info(f"  BEDROC_85: {our_metrics.bedroc_85:.4f}")

    # Compare
    compare_to_paper_results(our_metrics)


def demo_visualization():
    """Demo 3: Create evaluation visualizations."""
    from evaluation import plot_roc_curve, plot_pr_curve, create_evaluation_report

    logger.info("\n" + "="*70)
    logger.info("DEMO 3: Creating Visualizations")
    logger.info("="*70)

    # Simulate scores and labels
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic data with some signal
    true_labels = np.random.binomial(1, 0.1, n_samples)  # 10% positives
    scores = np.random.rand(n_samples)
    # Add signal: positives tend to have higher scores
    scores[true_labels == 1] += 0.3
    scores = np.clip(scores, 0, 1)

    logger.info(f"\nGenerating plots for {n_samples} samples...")
    logger.info(f"  Positives: {true_labels.sum()}")
    logger.info(f"  Negatives: {(1 - true_labels).sum()}")

    output_dir = project_root / "results" / "demo_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ROC curve
        logger.info("\n📈 Creating ROC curve...")
        plot_roc_curve(
            scores=scores,
            labels=true_labels,
            title="ROC Curve - Demo",
            save_path=str(output_dir / "demo_roc_curve.png"),
            show=False
        )
        logger.info(f"  ✓ Saved to {output_dir / 'demo_roc_curve.png'}")

        # PR curve
        logger.info("\n📈 Creating Precision-Recall curve...")
        plot_pr_curve(
            scores=scores,
            labels=true_labels,
            title="Precision-Recall Curve - Demo",
            save_path=str(output_dir / "demo_pr_curve.png"),
            show=False
        )
        logger.info(f"  ✓ Saved to {output_dir / 'demo_pr_curve.png'}")

        # Complete report
        logger.info("\n📊 Creating complete evaluation report...")
        from evaluation import compute_all_metrics

        # Convert to ranked format
        sorted_indices = np.argsort(scores)[::-1]
        ranked_ids = [f"P{i:05d}" for i in sorted_indices]
        active_ids = [f"P{i:05d}" for i in np.where(true_labels == 1)[0]]

        metrics = compute_all_metrics(
            ranked_ids=ranked_ids,
            scores=scores[sorted_indices],
            active_ids=active_ids
        )

        create_evaluation_report(
            metrics=metrics,
            scores=scores,
            labels=true_labels,
            output_dir=str(output_dir),
            name="demo_evaluation"
        )
        logger.info(f"  ✓ Complete report saved to {output_dir}")

    except ImportError as e:
        logger.warning(f"\n⚠️  Visualization requires matplotlib: {e}")
        logger.warning("Install with: pip install matplotlib seaborn scikit-learn")


def demo_statistical_analysis():
    """Demo 4: Statistical analysis with bootstrap."""
    from evaluation import bootstrap_metrics, compute_all_metrics

    logger.info("\n" + "="*70)
    logger.info("DEMO 4: Statistical Analysis")
    logger.info("="*70)

    # Simulate multiple screening results
    np.random.seed(42)
    n_reactions = 20

    logger.info(f"\nSimulating {n_reactions} screening results...")

    ranked_ids_list = []
    scores_list = []
    active_ids_list = []

    for i in range(n_reactions):
        # Each reaction screened against 100 proteins
        protein_ids = [f"P{j:05d}" for j in range(100)]
        scores = np.random.rand(100)

        # Random actives (2-5 per reaction)
        n_actives = np.random.randint(2, 6)
        active_idx = np.random.choice(100, n_actives, replace=False)
        active_ids = [protein_ids[idx] for idx in active_idx]

        # Rank by scores
        sorted_indices = np.argsort(scores)[::-1]
        ranked_ids = [protein_ids[idx] for idx in sorted_indices]

        ranked_ids_list.append(ranked_ids)
        scores_list.append(scores[sorted_indices])
        active_ids_list.append(active_ids)

    logger.info("\n🔬 Computing bootstrap confidence intervals...")
    logger.info("  This may take a moment...")

    try:
        # Bootstrap for BEDROC_85
        ci = bootstrap_metrics(
            ranked_ids_list=ranked_ids_list,
            scores_list=scores_list,
            active_ids_list=active_ids_list,
            n_bootstrap=100,  # Use 1000+ in practice
            metric_name='bedroc_85'
        )

        logger.info(f"\n📊 BEDROC_85 Bootstrap Results:")
        logger.info(f"  Mean:     {ci['mean']:.4f}")
        logger.info(f"  Median:   {ci['median']:.4f}")
        logger.info(f"  Std Dev:  {ci['std']:.4f}")
        logger.info(f"  95% CI:   [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        logger.info(f"  Iterations: {ci['n_bootstrap']}")

    except ImportError as e:
        logger.warning(f"\n⚠️  Statistical analysis requires scipy: {e}")
        logger.warning("Install with: pip install scipy scikit-learn")


def demo_bedroc_variants():
    """Demo 5: Compare different BEDROC alpha values."""
    from evaluation import CLIPZymeMetrics

    logger.info("\n" + "="*70)
    logger.info("DEMO 5: BEDROC Variants (Different Alpha Values)")
    logger.info("="*70)

    # Create synthetic ranking with early actives
    np.random.seed(42)
    n_total = 1000
    n_actives = 50

    # Create labels (1 for active, 0 for inactive)
    labels = np.zeros(n_total)
    labels[:n_actives] = 1  # First 50 are active

    # Shuffle to simulate ranking
    shuffle_idx = np.random.permutation(n_total)
    labels_shuffled = labels[shuffle_idx]
    scores = np.random.rand(n_total)

    logger.info(f"\nDataset: {n_total} compounds, {n_actives} actives")
    logger.info(f"Alpha values interpret early recognition differently:")
    logger.info("  α=20:  Standard (moderate early emphasis)")
    logger.info("  α=50:  Strong early emphasis")
    logger.info("  α=85:  Very strong early emphasis (CLIPZyme paper)")

    try:
        # Compute BEDROC with different alpha values
        alphas = [20.0, 50.0, 85.0]
        bedroc_values = []

        for alpha in alphas:
            bedroc = CLIPZymeMetrics.compute_bedroc(
                scores=scores,
                labels=labels_shuffled,
                alpha=alpha
            )
            bedroc_values.append(bedroc)
            logger.info(f"\n  BEDROC_{int(alpha)}: {bedroc:.4f}")

        logger.info("\n💡 Higher alpha = more emphasis on early retrieval")
        logger.info("   CLIPZyme uses α=85 to focus on top predictions")

    except ImportError as e:
        logger.warning(f"\n⚠️  BEDROC computation requires sklearn: {e}")
        logger.warning("Install with: pip install scikit-learn")


def main():
    """Run all demos."""
    logger.info("\n" + "#"*70)
    logger.info("# CLIPZyme Evaluation System Demo")
    logger.info("#"*70)

    try:
        # Demo 1: Basic metrics
        demo_basic_metrics()

        # Demo 2: Paper comparison
        demo_paper_comparison()

        # Demo 3: Visualization
        demo_visualization()

        # Demo 4: Statistical analysis
        demo_statistical_analysis()

        # Demo 5: BEDROC variants
        demo_bedroc_variants()

        logger.info("\n" + "="*70)
        logger.info("✅ All demos completed!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("  1. Install dependencies: pip install -r requirements.txt")
        logger.info("  2. Run full evaluation: python scripts/run_evaluation.py --help")
        logger.info("  3. See evaluation/README.md for detailed documentation")

    except Exception as e:
        logger.error(f"\n❌ Error in demo: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
