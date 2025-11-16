"""
Evaluation workflow definitions.

Provides pre-built workflows for screening evaluation.
"""

from pathlib import Path
from typing import Optional, List

from ..core.workflow import WorkflowBuilder, Workflow
from ..core.task import TaskConfig
from ..tasks.evaluation import (
    EvaluateScreeningTask,
    GenerateReportTask
)


def create_evaluation_workflow(
    ground_truth_csv: str,
    output_dir: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
    include_plots: bool = True
) -> Workflow:
    """
    Create evaluation workflow.

    This workflow:
    1. Evaluates screening results against ground truth
    2. Generates report with visualizations

    Args:
        ground_truth_csv: Path to ground truth CSV
        output_dir: Output directory
        metrics: List of metrics to compute
        include_plots: Whether to include plots

    Returns:
        Configured workflow
    """
    output_dir = output_dir or Path("results/evaluation")

    builder = WorkflowBuilder(
        name="evaluation_pipeline",
        description="Screening evaluation and reporting"
    )

    # Task 1: Evaluate
    evaluate_task = EvaluateScreeningTask(
        config=TaskConfig(
            name="evaluate_screening",
            description="Evaluate screening results"
        ),
        ground_truth_csv=ground_truth_csv,
        output_dir=output_dir,
        metrics=metrics
    )

    builder.add_task(evaluate_task)

    # Task 2: Generate report
    report_task = GenerateReportTask(
        config=TaskConfig(
            name="generate_report",
            description="Generate evaluation report",
            depends_on=["evaluate_screening"]
        ),
        output_dir=output_dir,
        include_plots=include_plots
    )

    builder.add_task(report_task)

    return builder.build()


def create_full_pipeline(
    checkpoint_name: str,
    reactions: List[str],
    proteins_csv: str,
    ground_truth_csv: str,
    top_k: int = 100,
    output_dir: Optional[Path] = None,
    device: str = "cuda"
) -> Workflow:
    """
    Create full screening + evaluation pipeline.

    This workflow:
    1. Downloads and loads checkpoint
    2. Builds screening set
    3. Runs screening
    4. Evaluates results
    5. Generates report

    Args:
        checkpoint_name: Checkpoint name
        reactions: List of reaction SMILES
        proteins_csv: Path to proteins CSV
        ground_truth_csv: Path to ground truth CSV
        top_k: Number of top proteins
        output_dir: Output directory
        device: Device for computation

    Returns:
        Configured workflow
    """
    from .screening import create_screening_workflow

    output_dir = output_dir or Path("results/full_pipeline")

    # Create combined workflow
    builder = WorkflowBuilder(
        name="full_pipeline",
        description="Complete screening and evaluation pipeline"
    )

    # Import screening tasks
    from ..tasks.checkpoint import (
        DownloadCheckpointTask,
        ValidateCheckpointTask,
        LoadCheckpointTask
    )
    from ..tasks.screening import (
        BuildScreeningSetTask,
        RunScreeningTask
    )

    # Add checkpoint tasks
    download_task = DownloadCheckpointTask(
        config=TaskConfig(name="download_checkpoint"),
        checkpoint_name=checkpoint_name
    )
    builder.add_task(download_task)

    validate_task = ValidateCheckpointTask(
        config=TaskConfig(
            name="validate_checkpoint",
            depends_on=["download_checkpoint"]
        )
    )
    builder.add_task(validate_task)

    load_task = LoadCheckpointTask(
        config=TaskConfig(
            name="load_checkpoint",
            depends_on=["validate_checkpoint"],
            gpu_required=True
        ),
        device=device
    )
    builder.add_task(load_task)

    # Add screening tasks
    build_set_task = BuildScreeningSetTask(
        config=TaskConfig(
            name="build_screening_set",
            depends_on=["load_checkpoint"],
            gpu_required=True
        ),
        proteins_csv=proteins_csv,
        device=device
    )
    builder.add_task(build_set_task)

    screen_task = RunScreeningTask(
        config=TaskConfig(
            name="run_screening",
            depends_on=["build_screening_set"],
            gpu_required=True
        ),
        reactions=reactions,
        output_dir=output_dir / "screening",
        top_k=top_k
    )
    builder.add_task(screen_task)

    # Add evaluation tasks
    evaluate_task = EvaluateScreeningTask(
        config=TaskConfig(
            name="evaluate_screening",
            depends_on=["run_screening"]
        ),
        ground_truth_csv=ground_truth_csv,
        output_dir=output_dir / "evaluation"
    )
    builder.add_task(evaluate_task)

    report_task = GenerateReportTask(
        config=TaskConfig(
            name="generate_report",
            depends_on=["evaluate_screening"]
        ),
        output_dir=output_dir / "report"
    )
    builder.add_task(report_task)

    return builder.build()


# Export
__all__ = [
    'create_evaluation_workflow',
    'create_full_pipeline',
]
