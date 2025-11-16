"""
Screening workflow definitions.

Provides pre-built workflows for virtual screening.
"""

from pathlib import Path
from typing import Optional, List

from ..core.workflow import WorkflowBuilder, Workflow
from ..core.task import TaskConfig, TaskPriority
from ..tasks.checkpoint import (
    DownloadCheckpointTask,
    ValidateCheckpointTask,
    LoadCheckpointTask
)
from ..tasks.screening import (
    BuildScreeningSetTask,
    RunScreeningTask
)


def create_screening_workflow(
    checkpoint_name: str,
    reactions: List[str],
    proteins_csv: Optional[str] = None,
    top_k: int = 100,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    mode: str = "interactive",
    build_screening_set: bool = True
) -> Workflow:
    """
    Create a complete screening workflow.

    This workflow:
    1. Downloads checkpoint (if needed)
    2. Validates checkpoint
    3. Loads checkpoint into model
    4. Builds screening set (if requested)
    5. Runs screening

    Args:
        checkpoint_name: Name of checkpoint to download
        reactions: List of reaction SMILES
        proteins_csv: Path to proteins CSV (for building screening set)
        top_k: Number of top proteins to return
        output_dir: Output directory
        device: Device for computation
        mode: Screening mode (interactive or batched)
        build_screening_set: Whether to build screening set

    Returns:
        Configured workflow
    """
    output_dir = output_dir or Path("results/screening")

    # Create workflow
    builder = WorkflowBuilder(
        name="screening_pipeline",
        description=f"Virtual screening with {checkpoint_name}"
    )

    # Configure workflow
    builder.set_max_parallel(1)  # Sequential execution
    builder.set_fail_fast(True)

    # Task 1: Download checkpoint
    download_task = DownloadCheckpointTask(
        config=TaskConfig(
            name="download_checkpoint",
            description="Download CLIPZyme checkpoint",
            priority=TaskPriority.HIGH,
            max_retries=3
        ),
        checkpoint_name=checkpoint_name
    )

    builder.add_task(download_task)

    # Task 2: Validate checkpoint
    validate_task = ValidateCheckpointTask(
        config=TaskConfig(
            name="validate_checkpoint",
            description="Validate checkpoint integrity",
            depends_on=["download_checkpoint"]
        )
    )

    builder.add_task(validate_task)

    # Task 3: Load checkpoint
    load_task = LoadCheckpointTask(
        config=TaskConfig(
            name="load_checkpoint",
            description="Load checkpoint into model",
            depends_on=["validate_checkpoint"],
            gpu_required=True,
            min_gpus=1
        ),
        device=device
    )

    builder.add_task(load_task)

    # Task 4: Build screening set (optional)
    dependencies = ["load_checkpoint"]

    if build_screening_set:
        if proteins_csv is None:
            raise ValueError("proteins_csv required when build_screening_set=True")

        build_set_task = BuildScreeningSetTask(
            config=TaskConfig(
                name="build_screening_set",
                description="Build protein screening set",
                depends_on=["load_checkpoint"],
                gpu_required=True
            ),
            proteins_csv=proteins_csv,
            output_path=output_dir / "screening_set.pkl",
            device=device
        )

        builder.add_task(build_set_task)
        dependencies = ["build_screening_set"]

    # Task 5: Run screening
    screen_task = RunScreeningTask(
        config=TaskConfig(
            name="run_screening",
            description="Screen reactions against proteins",
            depends_on=dependencies,
            gpu_required=True
        ),
        reactions=reactions,
        output_dir=output_dir,
        top_k=top_k,
        mode=mode
    )

    builder.add_task(screen_task)

    # Build workflow
    return builder.build()


def create_simple_screening_workflow(
    checkpoint_path: str,
    screening_set_path: str,
    reactions: List[str],
    top_k: int = 100,
    output_dir: Optional[Path] = None,
    device: str = "cuda"
) -> Workflow:
    """
    Create a simple screening workflow (checkpoint and screening set already exist).

    Args:
        checkpoint_path: Path to checkpoint file
        screening_set_path: Path to screening set file
        reactions: List of reaction SMILES
        top_k: Number of top proteins
        output_dir: Output directory
        device: Device for computation

    Returns:
        Configured workflow
    """
    output_dir = output_dir or Path("results/screening")

    builder = WorkflowBuilder(
        name="simple_screening",
        description="Simple screening workflow"
    )

    # Load checkpoint
    load_task = LoadCheckpointTask(
        config=TaskConfig(
            name="load_checkpoint",
            description="Load checkpoint",
            gpu_required=True
        ),
        checkpoint_path=Path(checkpoint_path),
        device=device
    )

    builder.add_task(load_task)

    # Run screening
    screen_task = RunScreeningTask(
        config=TaskConfig(
            name="run_screening",
            description="Run screening",
            depends_on=["load_checkpoint"],
            gpu_required=True
        ),
        reactions=reactions,
        screening_set_path=Path(screening_set_path),
        output_dir=output_dir,
        top_k=top_k
    )

    builder.add_task(screen_task)

    return builder.build()


# Export
__all__ = [
    'create_screening_workflow',
    'create_simple_screening_workflow',
]
