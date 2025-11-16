"""
Job scheduler for executing workflows from the queue.

Provides multi-threaded job execution with resource management.
"""

from typing import Optional, Callable, Dict, Any
from threading import Thread, Event
import time
import logging

from .queue import JobQueue, Job, JobStatus
from ..core.executor import WorkflowExecutor
from ..core.workflow import WorkflowResult
from ..resources.gpu import get_gpu_manager
from ..resources.memory import get_memory_manager


logger = logging.getLogger(__name__)


class Scheduler:
    """
    Job scheduler.

    Manages job execution from queue with resource allocation and monitoring.
    """

    def __init__(
        self,
        queue: Optional[JobQueue] = None,
        max_concurrent_jobs: int = 2,
        max_workers_per_job: int = 4,
        poll_interval: float = 1.0
    ):
        """
        Initialize scheduler.

        Args:
            queue: Job queue (creates new if None)
            max_concurrent_jobs: Maximum concurrent jobs
            max_workers_per_job: Maximum workers per job
            poll_interval: Queue polling interval (seconds)
        """
        self.queue = queue or JobQueue()
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_workers_per_job = max_workers_per_job
        self.poll_interval = poll_interval

        # Executor
        self.executor = WorkflowExecutor(max_workers=max_workers_per_job)

        # Resource managers
        self.gpu_manager = get_gpu_manager()
        self.memory_manager = get_memory_manager()

        # Scheduler state
        self._running = False
        self._scheduler_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Callbacks
        self.on_job_start: Optional[Callable[[Job], None]] = None
        self.on_job_complete: Optional[Callable[[Job, WorkflowResult], None]] = None
        self.on_job_failed: Optional[Callable[[Job, WorkflowResult], None]] = None

        self.logger = logging.getLogger(f"{__name__}.Scheduler")

    def start(self):
        """Start the scheduler."""
        if self._running:
            self.logger.warning("Scheduler already running")
            return

        self._running = True
        self._stop_event.clear()

        # Start scheduler thread
        self._scheduler_thread = Thread(
            target=self._scheduler_loop,
            name="DispatcherScheduler",
            daemon=True
        )
        self._scheduler_thread.start()

        self.logger.info("Scheduler started")

    def stop(self, wait: bool = True):
        """
        Stop the scheduler.

        Args:
            wait: Wait for current jobs to complete
        """
        if not self._running:
            return

        self.logger.info("Stopping scheduler...")

        self._running = False
        self._stop_event.set()

        if wait and self._scheduler_thread:
            self._scheduler_thread.join(timeout=30)

        self.logger.info("Scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        self.logger.info("Scheduler loop started")

        while self._running:
            try:
                # Check if we can run more jobs
                if self.queue.get_running_count() >= self.max_concurrent_jobs:
                    time.sleep(self.poll_interval)
                    continue

                # Try to get next job
                job = self.queue.get(block=True, timeout=self.poll_interval)

                if job is None:
                    continue

                # Execute job in separate thread
                job_thread = Thread(
                    target=self._execute_job,
                    args=(job,),
                    name=f"Job-{job.job_id[:8]}",
                    daemon=True
                )
                job_thread.start()

            except Exception as e:
                self.logger.error(f"Scheduler loop error: {str(e)}", exc_info=True)
                time.sleep(self.poll_interval)

        self.logger.info("Scheduler loop exited")

    def _execute_job(self, job: Job):
        """
        Execute a job.

        Args:
            job: Job to execute
        """
        self.logger.info(f"Executing job: {job.name} (id={job.job_id})")

        # Callback: job start
        if self.on_job_start:
            try:
                self.on_job_start(job)
            except Exception as e:
                self.logger.warning(f"Job start callback error: {str(e)}")

        try:
            # Allocate resources
            self._allocate_resources(job)

            # Execute workflow
            result = self.executor.execute(job.workflow)

            # Complete job
            self.queue.complete(job.job_id, result)

            # Callback: job complete/failed
            if result.success:
                if self.on_job_complete:
                    try:
                        self.on_job_complete(job, result)
                    except Exception as e:
                        self.logger.warning(f"Job complete callback error: {str(e)}")
            else:
                if self.on_job_failed:
                    try:
                        self.on_job_failed(job, result)
                    except Exception as e:
                        self.logger.warning(f"Job failed callback error: {str(e)}")

        except Exception as e:
            self.logger.error(
                f"Job execution error: {job.name} (id={job.job_id}) - {str(e)}",
                exc_info=True
            )

            # Create error result
            from ..core.workflow import WorkflowStatus
            result = WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=e
            )

            self.queue.complete(job.job_id, result)

        finally:
            # Release resources
            self._release_resources(job)

    def _allocate_resources(self, job: Job):
        """
        Allocate resources for job.

        Args:
            job: Job
        """
        try:
            # Check GPU requirements from workflow config
            # For now, we don't allocate automatically
            # Tasks will request GPUs as needed

            pass

        except Exception as e:
            self.logger.warning(f"Resource allocation error: {str(e)}")

    def _release_resources(self, job: Job):
        """
        Release resources for job.

        Args:
            job: Job
        """
        try:
            # Release GPU allocations
            self.gpu_manager.release_gpus(job.job_id)

            # Release memory allocations
            self.memory_manager.release_memory(job.job_id)

        except Exception as e:
            self.logger.warning(f"Resource release error: {str(e)}")

    def submit_job(
        self,
        workflow,
        name: Optional[str] = None,
        priority=None,
        metadata: Optional[Dict[str, Any]] = None,
        owner: Optional[str] = None
    ) -> str:
        """
        Submit a job to the queue.

        Args:
            workflow: Workflow to execute
            name: Job name
            priority: Job priority
            metadata: Job metadata
            owner: Job owner

        Returns:
            Job ID
        """
        from ..core.task import TaskPriority
        priority = priority or TaskPriority.NORMAL

        return self.queue.submit(
            workflow=workflow,
            name=name,
            priority=priority,
            metadata=metadata,
            owner=owner
        )

    def get_job_status(self, job_id: str) -> Optional[str]:
        """
        Get job status.

        Args:
            job_id: Job ID

        Returns:
            Job status or None
        """
        job = self.queue.get_job(job_id)
        return job.status.value if job else None

    def get_job_result(self, job_id: str) -> Optional[WorkflowResult]:
        """
        Get job result.

        Args:
            job_id: Job ID

        Returns:
            Workflow result or None
        """
        job = self.queue.get_job(job_id)
        return job.result if job else None

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled
        """
        return self.queue.cancel(job_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            'scheduler': {
                'running': self._running,
                'max_concurrent_jobs': self.max_concurrent_jobs,
            },
            'queue': self.queue.get_stats(),
            'resources': {
                'gpu': self.gpu_manager.get_allocation_summary(),
                'memory': self.memory_manager.get_allocation_summary(),
            }
        }

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Export
__all__ = [
    'Scheduler',
]
