"""
Command-line interface for dispatcher.

Provides CLI commands for dispatcher management.
"""

import click
from pathlib import Path
import json
import sys


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.pass_context
def cli(ctx, config):
    """CLIPZyme Dispatcher - Workflow orchestration for virtual screening."""
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = Path(config) if config else None


@cli.command()
@click.option('--max-jobs', '-j', type=int, default=2, help='Maximum concurrent jobs')
@click.option('--workers', '-w', type=int, default=4, help='Workers per job')
@click.pass_context
def start(ctx, max_jobs, workers):
    """Start the dispatcher scheduler."""
    from .python_api import DispatcherAPI

    config = {
        'scheduler': {
            'max_concurrent_jobs': max_jobs,
            'max_workers_per_job': workers,
        }
    }

    dispatcher = DispatcherAPI(
        config_file=ctx.obj.get('config_file'),
        config=config
    )

    click.echo(f"Dispatcher started (jobs={max_jobs}, workers={workers})")
    click.echo("Press Ctrl+C to stop...")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping dispatcher...")
        dispatcher.stop()
        click.echo("Dispatcher stopped")


@cli.command()
@click.argument('job_id')
@click.pass_context
def status(ctx, job_id):
    """Get job status."""
    from .python_api import DispatcherAPI

    dispatcher = DispatcherAPI(
        config_file=ctx.obj.get('config_file'),
        auto_start=False
    )

    status = dispatcher.get_job_status(job_id)

    if status:
        click.echo(f"Job {job_id}: {status}")
    else:
        click.echo(f"Job {job_id} not found", err=True)
        sys.exit(1)


@cli.command()
@click.option('--status', '-s', help='Filter by status')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def list_jobs(ctx, status, output_json):
    """List jobs."""
    from .python_api import DispatcherAPI

    dispatcher = DispatcherAPI(
        config_file=ctx.obj.get('config_file'),
        auto_start=False
    )

    jobs = dispatcher.list_jobs(status=status)

    if output_json:
        click.echo(json.dumps(jobs, indent=2))
    else:
        if not jobs:
            click.echo("No jobs found")
        else:
            click.echo(f"{'Job ID':<40} {'Name':<30} {'Status':<15} {'Created'}")
            click.echo("-" * 100)

            for job in jobs:
                click.echo(
                    f"{job['job_id']:<40} "
                    f"{job['name']:<30} "
                    f"{job['status']:<15} "
                    f"{job['created_at']}"
                )


@cli.command()
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def stats(ctx, output_json):
    """Show dispatcher statistics."""
    from .python_api import DispatcherAPI
    from ..monitoring import get_reporter

    dispatcher = DispatcherAPI(
        config_file=ctx.obj.get('config_file'),
        auto_start=False
    )

    stats = dispatcher.get_stats()

    if output_json:
        click.echo(json.dumps(stats, indent=2))
    else:
        reporter = get_reporter()
        click.echo(reporter.format_full_status(stats))


@cli.command()
@click.argument('job_id')
@click.pass_context
def cancel(ctx, job_id):
    """Cancel a job."""
    from .python_api import DispatcherAPI

    dispatcher = DispatcherAPI(
        config_file=ctx.obj.get('config_file'),
        auto_start=False
    )

    if dispatcher.cancel_job(job_id):
        click.echo(f"Job {job_id} cancelled")
    else:
        click.echo(f"Failed to cancel job {job_id}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('checkpoint_name')
@click.argument('reactions_file', type=click.Path(exists=True))
@click.option('--proteins-csv', type=click.Path(exists=True), help='Proteins CSV file')
@click.option('--top-k', type=int, default=100, help='Number of top proteins')
@click.option('--output-dir', type=click.Path(), default='results', help='Output directory')
@click.pass_context
def screen(ctx, checkpoint_name, reactions_file, proteins_csv, top_k, output_dir):
    """Run screening workflow."""
    from .python_api import DispatcherAPI

    click.echo(f"Submitting screening workflow...")
    click.echo(f"  Checkpoint: {checkpoint_name}")
    click.echo(f"  Reactions: {reactions_file}")
    click.echo(f"  Top-K: {top_k}")

    # Load reactions
    with open(reactions_file, 'r') as f:
        reactions = [line.strip() for line in f if line.strip()]

    click.echo(f"  Loaded {len(reactions)} reactions")

    # Create workflow
    from ..workflows.screening import create_screening_workflow

    workflow = create_screening_workflow(
        checkpoint_name=checkpoint_name,
        reactions=reactions,
        proteins_csv=proteins_csv,
        top_k=top_k,
        output_dir=Path(output_dir)
    )

    # Submit job
    dispatcher = DispatcherAPI(config_file=ctx.obj.get('config_file'))

    job_id = dispatcher.submit_workflow(workflow, name=f"screening-{checkpoint_name}")

    click.echo(f"\nJob submitted: {job_id}")
    click.echo("Use 'dispatcher status <job_id>' to check progress")


if __name__ == '__main__':
    cli()


# Export
__all__ = [
    'cli',
]
