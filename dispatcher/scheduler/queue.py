"""
Job queue for managing workflow execution requests.

Provides thread-safe queue with priority support and persistence.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import queue
import threading
import uuid
import logging

from ..core.workflow import Workflow, WorkflowResult, WorkflowStatus
from ..core.task import TaskPriority


logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """
    Job represents a workflow execution request.
    """

    # Identification
    job_id: str
    name: str
    workflow: Workflow

    # Priority
    priority: TaskPriority = TaskPriority.NORMAL

    # Status
    status: JobStatus = JobStatus.QUEUED

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Result
    result: Optional[WorkflowResult] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Owner/requester
    owner: Optional[str] = None

    @property
    def wait_time(self) -> Optional[float]:
        """Time spent waiting in queue (seconds)."""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def execution_time(self) -> Optional[float]:
        """Time spent executing (seconds)."""
        if not self.started_at:
            return None

        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    @property
    def total_time(self) -> float:
        """Total time from creation to completion (seconds)."""
        end_time = self.completed_at or datetime.now()
        return (end_time - self.created_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'wait_time': self.wait_time,
            'execution_time': self.execution_time,
            'total_time': self.total_time,
            'metadata': self.metadata,
            'owner': self.owner,
        }


class JobQueue:
    """
    Priority queue for jobs.

    Thread-safe queue with priority support and job tracking.
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize job queue.

        Args:
            maxsize: Maximum queue size (0 for unlimited)
        """
        # Priority queue (lower number = higher priority)
        # Items are tuples: (priority_value, timestamp, job)
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=maxsize)

        # Job tracking
        self._jobs: Dict[str, Job] = {}  # All jobs
        self._queued_jobs: Dict[str, Job] = {}  # Jobs in queue
        self._running_jobs: Dict[str, Job] = {}  # Jobs currently running
        self._completed_jobs: Dict[str, Job] = {}  # Completed jobs

        # Thread safety
        self._lock = threading.RLock()

        # Counter for maintaining queue order for same priority
        self._counter = 0

        self.logger = logging.getLogger(f"{__name__}.JobQueue")

    def submit(
        self,
        workflow: Workflow,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        owner: Optional[str] = None
    ) -> str:
        """
        Submit a job to the queue.

        Args:
            workflow: Workflow to execute
            name: Job name (optional)
            priority: Job priority
            metadata: Job metadata
            owner: Job owner

        Returns:
            Job ID
        """
        with self._lock:
            # Create job
            job = Job(
                job_id=str(uuid.uuid4()),
                name=name or workflow.config.name,
                workflow=workflow,
                priority=priority,
                status=JobStatus.QUEUED,
                metadata=metadata or {},
                owner=owner
            )

            # Add to tracking
            self._jobs[job.job_id] = job
            self._queued_jobs[job.job_id] = job

            # Add to priority queue
            # Use negative priority value so higher priority = lower number
            priority_value = -priority.value
            self._counter += 1
            self._queue.put((priority_value, self._counter, job))

            self.logger.info(
                f"Job submitted: {job.name} (id={job.job_id}, priority={priority.name})"
            )

            return job.job_id

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Job]:
        """
        Get next job from queue.

        Args:
            block: Block until job available
            timeout: Timeout in seconds

        Returns:
            Job or None
        """
        try:
            _, _, job = self._queue.get(block=block, timeout=timeout)

            with self._lock:
                # Update status
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()

                # Move to running
                self._queued_jobs.pop(job.job_id, None)
                self._running_jobs[job.job_id] = job

            self.logger.info(
                f"Job dequeued: {job.name} (id={job.job_id}, wait={job.wait_time:.2f}s)"
            )

            return job

        except queue.Empty:
            return None

    def complete(self, job_id: str, result: WorkflowResult):
        """
        Mark job as completed.

        Args:
            job_id: Job ID
            result: Workflow result
        """
        with self._lock:
            job = self._jobs.get(job_id)

            if not job:
                self.logger.warning(f"Unknown job ID: {job_id}")
                return

            # Update job
            job.status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
            job.completed_at = datetime.now()
            job.result = result

            # Move to completed
            self._running_jobs.pop(job_id, None)
            self._completed_jobs[job_id] = job

            self.logger.info(
                f"Job completed: {job.name} "
                f"(id={job_id}, status={job.status.value}, "
                f"exec_time={job.execution_time:.2f}s)"
            )

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled
        """
        with self._lock:
            job = self._jobs.get(job_id)

            if not job:
                self.logger.warning(f"Unknown job ID: {job_id}")
                return False

            # Can only cancel queued jobs
            if job.status != JobStatus.QUEUED:
                self.logger.warning(
                    f"Cannot cancel job {job_id} with status {job.status.value}"
                )
                return False

            # Update status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()

            # Remove from queued
            self._queued_jobs.pop(job_id, None)
            self._completed_jobs[job_id] = job

            self.logger.info(f"Job cancelled: {job.name} (id={job_id})")

            return True

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job or None
        """
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        owner: Optional[str] = None
    ) -> List[Job]:
        """
        List jobs.

        Args:
            status: Filter by status
            owner: Filter by owner

        Returns:
            List of jobs
        """
        with self._lock:
            jobs = list(self._jobs.values())

            if status:
                jobs = [j for j in jobs if j.status == status]

            if owner:
                jobs = [j for j in jobs if j.owner == owner]

            return jobs

    def get_queued_count(self) -> int:
        """Get number of queued jobs."""
        return len(self._queued_jobs)

    def get_running_count(self) -> int:
        """Get number of running jobs."""
        return len(self._running_jobs)

    def get_completed_count(self) -> int:
        """Get number of completed jobs."""
        return len(self._completed_jobs)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()

    def clear_completed(self, keep_count: int = 100):
        """
        Clear old completed jobs.

        Args:
            keep_count: Number of recent completed jobs to keep
        """
        with self._lock:
            if len(self._completed_jobs) <= keep_count:
                return

            # Sort by completion time
            sorted_jobs = sorted(
                self._completed_jobs.values(),
                key=lambda j: j.completed_at or datetime.min,
                reverse=True
            )

            # Keep recent, remove old
            to_keep = sorted_jobs[:keep_count]
            to_remove = sorted_jobs[keep_count:]

            for job in to_remove:
                self._completed_jobs.pop(job.job_id, None)
                self._jobs.pop(job.job_id, None)

            self.logger.info(f"Cleared {len(to_remove)} old completed jobs")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                'total_jobs': len(self._jobs),
                'queued': len(self._queued_jobs),
                'running': len(self._running_jobs),
                'completed': len(self._completed_jobs),
                'queue_size': self.size(),
            }


# Export
__all__ = [
    'Job',
    'JobStatus',
    'JobQueue',
]
