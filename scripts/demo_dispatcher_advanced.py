"""
Advanced dispatcher demo.

Demonstrates:
- Custom task creation
- Manual workflow building
- Progress tracking
- Multiple concurrent jobs
- Resource monitoring
"""

import sys
sys.path.insert(0, str(__file__).rsplit('scripts', 1)[0].rstrip('/\\'))

from dispatcher import (
    DispatcherAPI,
    WorkflowBuilder,
    TaskConfig,
    TaskPriority,
    configure_logging,
)
from dispatcher.tasks import (
    LoadCheckpointTask,
    RunScreeningTask,
)
import logging
from pathlib import Path


def main():
    """Run advanced dispatcher demo."""
    configure_logging(level=logging.INFO)

    print("=" * 60)
    print("CLIPZyme Dispatcher - Advanced Demo")
    print("=" * 60)

    # Create dispatcher with custom config
    config = {
        'scheduler': {
            'max_concurrent_jobs': 3,
            'max_workers_per_job': 4,
        },
        'logging': {
            'level': 'INFO',
        }
    }

    print("\n[1] Starting dispatcher...")
    dispatcher = DispatcherAPI(config=config, auto_start=True)

    # Create multiple workflows
    workflows = []

    print("\n[2] Creating workflows...")

    # Workflow 1: Quick screening (high priority)
    print("   Creating workflow 1 (high priority)...")
    builder1 = WorkflowBuilder(
        name="quick_screening",
        description="Quick screening with high priority"
    )

    load_task1 = LoadCheckpointTask(
        config=TaskConfig(
            name="load_checkpoint",
            priority=TaskPriority.HIGH,
            gpu_required=True
        ),
        checkpoint_path=Path("checkpoints/clipzyme.pt"),
        device="cuda"
    )
    builder1.add_task(load_task1)

    screen_task1 = RunScreeningTask(
        config=TaskConfig(
            name="run_screening",
            depends_on=["load_checkpoint"],
            priority=TaskPriority.HIGH
        ),
        reactions=["CC(=O)O>>CCO"],
        screening_set_path=Path("data/screening_set.pkl"),
        top_k=10
    )
    builder1.add_task(screen_task1)

    workflows.append(("workflow_1", builder1.build(), TaskPriority.HIGH))

    # Workflow 2: Normal screening
    print("   Creating workflow 2 (normal priority)...")
    builder2 = WorkflowBuilder(name="normal_screening")

    load_task2 = LoadCheckpointTask(
        config=TaskConfig(name="load_checkpoint_2"),
        checkpoint_path=Path("checkpoints/clipzyme.pt"),
        device="cuda"
    )
    builder2.add_task(load_task2)

    screen_task2 = RunScreeningTask(
        config=TaskConfig(
            name="run_screening_2",
            depends_on=["load_checkpoint_2"]
        ),
        reactions=["CC(C)CC(N)C(=O)O>>CC(C)CC(=O)C(=O)O"] * 5,
        top_k=20
    )
    builder2.add_task(screen_task2)

    workflows.append(("workflow_2", builder2.build(), TaskPriority.NORMAL))

    # Submit jobs
    print("\n[3] Submitting jobs...")
    job_ids = []

    for name, workflow, priority in workflows:
        job_id = dispatcher.submit_workflow(
            workflow,
            name=name,
            priority=priority
        )
        job_ids.append((job_id, name))
        print(f"   Submitted: {name} (job_id={job_id}, priority={priority.name})")

    # Monitor progress
    print("\n[4] Monitoring execution...")
    print("   Press Ctrl+C to stop monitoring\n")

    try:
        import time
        completed = set()

        while len(completed) < len(job_ids):
            # Get stats
            stats = dispatcher.get_stats()

            # Print summary
            print("\r" + " " * 80, end="")  # Clear line
            print(
                f"\rQueued: {stats['queue']['queued']} | "
                f"Running: {stats['queue']['running']} | "
                f"Completed: {len(completed)}/{len(job_ids)}",
                end="",
                flush=True
            )

            # Check job completion
            for job_id, name in job_ids:
                if job_id not in completed:
                    result = dispatcher.get_job_result(job_id)
                    if result is not None:
                        completed.add(job_id)
                        print(f"\n   ✓ {name} completed ({result.status.value})")

            time.sleep(1)

        print("\n\nAll jobs completed!")

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted!")

    # Show final stats
    print("\n[5] Final Statistics")
    print("=" * 60)

    stats = dispatcher.get_stats()

    # Queue stats
    print("Queue:")
    print(f"  Total jobs: {stats['queue']['total_jobs']}")
    print(f"  Completed: {stats['queue']['completed']}")

    # Resource stats
    if 'resources' in stats and 'gpu' in stats['resources']:
        print("\nGPU:")
        gpu_stats = stats['resources']['gpu']
        print(f"  Total GPUs: {gpu_stats.get('total_gpus', 0)}")

    if 'memory' in stats['resources']:
        mem_stats = stats['resources']['memory']
        if 'system_memory' in mem_stats:
            sys_mem = mem_stats['system_memory']
            print("\nMemory:")
            print(
                f"  System: {sys_mem.get('available_mb', 0):.0f}MB available / "
                f"{sys_mem.get('total_mb', 0):.0f}MB total"
            )

    print("\n" + "=" * 60)

    # Cleanup
    print("\nStopping dispatcher...")
    dispatcher.stop()

    print("\nDemo completed!")


if __name__ == '__main__':
    main()
