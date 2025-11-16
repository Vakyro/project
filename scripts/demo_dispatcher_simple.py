"""
Simple dispatcher demo.

Demonstrates basic workflow submission and execution.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('scripts', 1)[0].rstrip('/\\'))

from dispatcher import (
    DispatcherAPI,
    create_screening_workflow,
    configure_logging
)
import logging


def main():
    """Run simple dispatcher demo."""
    # Configure logging
    configure_logging(level=logging.INFO, structured=False)

    print("=" * 60)
    print("CLIPZyme Dispatcher - Simple Demo")
    print("=" * 60)

    # Example reactions
    reactions = [
        "CC(=O)Oc1ccccc1C(=O)O>>CC(=O)O.O=C(O)c1ccccc1O",  # Aspirin hydrolysis
        "CC(C)CC(N)C(=O)O>>CC(C)CC(=O)C(=O)O",  # Leucine deamination
    ]

    print(f"\nScreening {len(reactions)} reactions...")

    # Create workflow
    print("\n[1/3] Creating screening workflow...")
    workflow = create_screening_workflow(
        checkpoint_name="clipzyme_official_v1",
        reactions=reactions,
        proteins_csv="data/proteins.csv",
        top_k=10,
        device="cuda" if __import__('torch').cuda.is_available() else "cpu"
    )

    print(f"   Workflow: {workflow.config.name}")
    print(f"   Tasks: {workflow.get_task_count()}")

    # Create dispatcher
    print("\n[2/3] Starting dispatcher...")
    dispatcher = DispatcherAPI(auto_start=True)

    # Submit job
    print("\n[3/3] Submitting job...")
    job_id = dispatcher.submit_workflow(
        workflow,
        name="demo_screening"
    )

    print(f"\nJob submitted: {job_id}")
    print("Waiting for completion...")

    # Wait for completion
    try:
        result = dispatcher.wait_for_job(job_id, timeout=600)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Status: {result.status.value.upper()}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Completed tasks: {len(result.completed_tasks)}/{len(result.task_results)}")

        if result.failed_tasks:
            print(f"Failed tasks: {', '.join(result.failed_tasks)}")

        print("\nTask breakdown:")
        for task_name, task_result in result.task_results.items():
            status = "✓" if task_result.success else "✗"
            duration = f"{task_result.duration:.2f}s" if task_result.duration else "N/A"
            print(f"  {status} {task_name}: {duration}")

    except TimeoutError:
        print("\nTimeout waiting for job completion!")
        print(f"Current status: {dispatcher.get_job_status(job_id)}")

    finally:
        # Stop dispatcher
        print("\nStopping dispatcher...")
        dispatcher.stop()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
