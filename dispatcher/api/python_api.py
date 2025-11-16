"""
Python API for dispatcher.

Provides high-level API for submitting and managing jobs.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

from ..scheduler import Scheduler, JobQueue
from ..core.workflow import Workflow, WorkflowBuilder, WorkflowResult
from ..core.task import Task, TaskPriority, TaskConfig
from ..config import ConfigResolver, ConfigValidator
from ..monitoring import configure_logging, get_logger


logger = logging.getLogger(__name__)


class DispatcherAPI:
    """
    High-level API for dispatcher system.

    Provides simple interface for job submission and management.
    """

    def __init__(
        self,
        config_file: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        auto_start: bool = True
    ):
        """
        Initialize dispatcher API.

        Args:
            config_file: Optional configuration file
            config: Optional configuration dictionary
            auto_start: Automatically start scheduler
        """
        # Resolve configuration
        resolver = ConfigResolver()
        self.config = resolver.resolve(config_file=config_file, overrides=config)

        # Validate configuration
        validator = ConfigValidator()
        validator.validate(self.config)

        # Configure logging
        log_config = self.config.get('logging', {})
        configure_logging(
            level=getattr(logging, log_config.get('level', 'INFO')),
            console=log_config.get('console', True),
            file=log_config.get('file', True),
            structured=log_config.get('structured', False)
        )

        self.logger = get_logger()

        # Create scheduler
        scheduler_config = self.config.get('scheduler', {})
        self.scheduler = Scheduler(
            max_concurrent_jobs=scheduler_config.get('max_concurrent_jobs', 2),
            max_workers_per_job=scheduler_config.get('max_workers_per_job', 4),
            poll_interval=scheduler_config.get('poll_interval', 1.0)
        )

        # Start scheduler if requested
        if auto_start:
            self.start()

    def start(self):
        """Start the scheduler."""
        self.scheduler.start()
        self.logger.info("Dispatcher started")

    def stop(self, wait: bool = True):
        """
        Stop the scheduler.

        Args:
            wait: Wait for current jobs to complete
        """
        self.scheduler.stop(wait=wait)
        self.logger.info("Dispatcher stopped")

    def submit_workflow(
        self,
        workflow: Workflow,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a workflow for execution.

        Args:
            workflow: Workflow to execute
            name: Job name (optional)
            priority: Job priority
            metadata: Job metadata

        Returns:
            Job ID
        """
        job_id = self.scheduler.submit_job(
            workflow=workflow,
            name=name,
            priority=priority,
            metadata=metadata
        )

        self.logger.info(f"Workflow submitted: {name or workflow.config.name} (job_id={job_id})")

        return job_id

    def submit_task(
        self,
        task: Task,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Submit a single task as a workflow.

        Args:
            task: Task to execute
            name: Job name
            priority: Job priority

        Returns:
            Job ID
        """
        # Create simple workflow with single task
        from ..core.workflow import WorkflowConfig

        workflow_config = WorkflowConfig(
            name=name or task.name,
            max_parallel_tasks=1
        )

        workflow = Workflow(workflow_config)
        workflow.add_task(task)

        return self.submit_workflow(workflow, name=name, priority=priority)

    def get_job_status(self, job_id: str) -> Optional[str]:
        """
        Get job status.

        Args:
            job_id: Job ID

        Returns:
            Job status or None
        """
        return self.scheduler.get_job_status(job_id)

    def get_job_result(self, job_id: str) -> Optional[WorkflowResult]:
        """
        Get job result.

        Args:
            job_id: Job ID

        Returns:
            Workflow result or None
        """
        return self.scheduler.get_job_result(job_id)

    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> WorkflowResult:
        """
        Wait for job to complete.

        Args:
            job_id: Job ID
            timeout: Timeout in seconds

        Returns:
            Workflow result

        Raises:
            TimeoutError: If timeout reached
        """
        import time

        start_time = time.time()

        while True:
            result = self.get_job_result(job_id)

            if result is not None:
                return result

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            time.sleep(0.5)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled
        """
        return self.scheduler.cancel_job(job_id)

    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List jobs.

        Args:
            status: Filter by status (optional)

        Returns:
            List of job information
        """
        from ..scheduler.queue import JobStatus

        status_filter = JobStatus(status) if status else None

        jobs = self.scheduler.queue.list_jobs(status=status_filter)

        return [job.to_dict() for job in jobs]

    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        return self.scheduler.get_stats()

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self.scheduler.is_running

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Export
__all__ = [
    'DispatcherAPI',
]
