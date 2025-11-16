#!/usr/bin/env python3
"""
CLIPZyme Evaluation Script

Run comprehensive evaluation with BEDROC₈₅ and other metrics.

Usage:
    # Run benchmark evaluation
    python scripts/run_evaluation.py \
        --model data/checkpoints/clipzyme_model.ckpt \
        --screening-set data/screening_set.p \
        --test-data data/test_reactions.csv \
        --output results/evaluation

    # Compare to paper results
    python scripts/run_evaluation.py \
        --model data/checkpoints/clipzyme_model.ckpt \
        --screening-set data/screening_set.p \
        --test-data data/test_reactions.csv \
        --compare-to-paper \
        --output results/evaluation
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import load_checkpoint, load_pretrained
from screening import ScreeningSet
from evaluation import (
    run_benchmark,
    compute_all_metrics,
    compare_to_paper_results,
    create_evaluation_report,
    bootstrap_metrics,
    plot_roc_curve,
    plot_pr_curve,
    plot_bedroc_comparison
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(
    csv_path: str,
    reaction_column: str = "reaction_smiles",
    id_column: str = "reaction_id",
    labels_column: str = "known_enzymes"
) -> tuple:
    """Load test data from CSV."""
    logger.info(f"Loading test data from {csv_path}")

    df = pd.read_csv(csv_path)

    reactions = df[reaction_column].tolist()

    if id_column in df.columns:
        ids = df[id_column].tolist()
    else:
        ids = [f"rxn_{i}" for i in range(len(reactions))]

    if labels_column in df.columns:
        # Parse comma-separated labels
        labels = []
        for label_str in df[labels_column]:
            if pd.notna(label_str):
                labels.append(str(label_str).split(','))
            else:
                labels.append([])
    else:
        labels = [[]] * len(reactions)

    logger.info(f"Loaded {len(reactions)} reactions")
    return reactions, ids, labels


def main():
    parser = argparse.ArgumentParser(description="CLIPZyme Evaluation")

    # Model and data
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to model checkpoint or "clipzyme" for pretrained'
    )
    parser.add_argument(
        '--screening-set', '-s',
        type=str,
        required=True,
        help='Path to screening set pickle file'
    )
    parser.add_argument(
        '--test-data', '-t',
        type=str,
        required=True,
        help='Path to test data CSV'
    )

    # Data columns
    parser.add_argument(
        '--reaction-column',
        type=str,
        default='reaction_smiles',
        help='Column name for reaction SMILES'
    )
    parser.add_argument(
        '--id-column',
        type=str,
        default='reaction_id',
        help='Column name for reaction IDs'
    )
    parser.add_argument(
        '--labels-column',
        type=str,
        default='known_enzymes',
        help='Column name for known enzymes (comma-separated)'
    )

    # Evaluation options
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top predictions to consider'
    )
    parser.add_argument(
        '--compare-to-paper',
        action='store_true',
        help='Compare results to CLIPZyme paper'
    )
    parser.add_argument(
        '--bootstrap',
        action='store_true',
        help='Compute bootstrap confidence intervals'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap iterations'
    )

    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference (cuda or cpu)'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CLIPZyme Evaluation with BEDROC₈₅")
    logger.info("=" * 70)

    # Load model
    logger.info(f"\n1. Loading model...")
    if args.model == "clipzyme":
        model = load_pretrained("clipzyme", device=args.device)
    else:
        model = load_checkpoint(args.model, device=args.device)

    logger.info("✓ Model loaded")

    # Load screening set
    logger.info(f"\n2. Loading screening set...")
    screening_set = ScreeningSet(device=args.device)
    screening_set.load_from_pickle(args.screening_set)
    logger.info(f"✓ Loaded {len(screening_set)} proteins")

    # Load test data
    logger.info(f"\n3. Loading test data...")
    test_reactions, reaction_ids, true_labels = load_test_data(
        csv_path=args.test_data,
        reaction_column=args.reaction_column,
        id_column=args.id_column,
        labels_column=args.labels_column
    )

    # Filter reactions with labels
    has_labels = [len(labels) > 0 for labels in true_labels]
    test_reactions = [r for r, has in zip(test_reactions, has_labels) if has]
    reaction_ids = [rid for rid, has in zip(reaction_ids, has_labels) if has]
    true_labels = [l for l, has in zip(true_labels, has_labels) if has]

    logger.info(f"✓ {len(test_reactions)} reactions with known enzymes")

    if len(test_reactions) == 0:
        logger.error("No reactions with known enzymes found!")
        return 1

    # Run benchmark
    logger.info(f"\n4. Running benchmark evaluation...")
    results = run_benchmark(
        model=model,
        screening_set=screening_set,
        test_reactions=test_reactions,
        true_labels=true_labels,
        output_dir=args.output,
        compare_to_paper=args.compare_to_paper
    )

    metrics = results['aggregated_metrics']

    # Print results
    logger.info(f"\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)

    logger.info(f"\n🎯 PRIMARY METRIC (CLIPZyme Paper):")
    logger.info(f"  BEDROC_85: {metrics.bedroc_85:.4f}")

    logger.info(f"\n📊 BEDROC Variants:")
    logger.info(f"  BEDROC_85: {metrics.bedroc_85:.4f}")
    logger.info(f"  BEDROC_50: {metrics.bedroc_50:.4f}")
    logger.info(f"  BEDROC_20: {metrics.bedroc_20:.4f}")

    logger.info(f"\n🔝 Top-K Accuracy:")
    logger.info(f"  Top-1:   {metrics.top1_accuracy:.4f}")
    logger.info(f"  Top-5:   {metrics.top5_accuracy:.4f}")
    logger.info(f"  Top-10:  {metrics.top10_accuracy:.4f}")
    logger.info(f"  Top-50:  {metrics.top50_accuracy:.4f}")
    logger.info(f"  Top-100: {metrics.top100_accuracy:.4f}")

    logger.info(f"\n📈 Enrichment Factor:")
    logger.info(f"  EF 1%:  {metrics.ef_1pct:.2f}")
    logger.info(f"  EF 5%:  {metrics.ef_5pct:.2f}")
    logger.info(f"  EF 10%: {metrics.ef_10pct:.2f}")

    logger.info(f"\n📉 Area Under Curves:")
    logger.info(f"  AUROC: {metrics.auroc:.4f}")
    logger.info(f"  AUPRC: {metrics.auprc:.4f}")

    # Compare to paper
    if args.compare_to_paper:
        logger.info(f"\n5. Comparing to CLIPZyme paper...")
        compare_to_paper_results(metrics)

    # Bootstrap confidence intervals
    if args.bootstrap:
        logger.info(f"\n6. Computing bootstrap confidence intervals...")

        ranked_ids_list = [r.ranked_protein_ids for r in results['individual_results']]
        scores_list = [r.scores.cpu().numpy() for r in results['individual_results']]
        active_ids_list = [r.true_positives for r in results['individual_results']]

        ci = bootstrap_metrics(
            ranked_ids_list=ranked_ids_list,
            scores_list=scores_list,
            active_ids_list=active_ids_list,
            n_bootstrap=args.n_bootstrap,
            metric_name='bedroc_85'
        )

        logger.info(f"\n✓ Bootstrap Results (n={args.n_bootstrap}):")
        logger.info(f"  BEDROC_85: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        logger.info(f"  Confidence Level: {ci['confidence_level']*100:.0f}%")

    # Create plots
    logger.info(f"\n7. Creating evaluation plots...")

    output_dir = Path(args.output)

    # Aggregate all scores and labels
    all_scores = []
    all_labels = []
    for result in results['individual_results']:
        if result.true_positives:
            active_set = set(result.true_positives)
            labels = [1 if pid in active_set else 0 for pid in result.ranked_protein_ids]
            all_scores.extend(result.scores.cpu().numpy())
            all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # ROC curve
    plot_roc_curve(
        all_scores, all_labels,
        title="ROC Curve - CLIPZyme Evaluation",
        save_path=str(output_dir / "roc_curve.png"),
        show=False
    )

    # PR curve
    plot_pr_curve(
        all_scores, all_labels,
        title="Precision-Recall Curve - CLIPZyme Evaluation",
        save_path=str(output_dir / "pr_curve.png"),
        show=False
    )

    # Full evaluation report
    create_evaluation_report(
        metrics=metrics,
        scores=all_scores,
        labels=all_labels,
        output_dir=str(output_dir),
        name="clipzyme_evaluation"
    )

    logger.info(f"✓ Plots saved to {output_dir}")

    logger.info(f"\n" + "=" * 70)
    logger.info("✓ Evaluation complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
