"""
Priority strategies for job scheduling.

Provides different strategies for determining job priority.
"""

from typing import Protocol, Dict, Any
from datetime import datetime

from .queue import Job
from ..core.task import TaskPriority


class PriorityStrategy(Protocol):
    """Protocol for priority strategies."""

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """
        Calculate priority for a job.

        Args:
            job: Job to prioritize
            context: Additional context

        Returns:
            Priority value (higher = more important)
        """
        ...


class FIFOPriority:
    """First-In-First-Out priority (submission order)."""

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """Calculate FIFO priority (earlier = higher priority)."""
        # Return negative timestamp so earlier jobs have higher priority
        return -job.created_at.timestamp()


class LIFOPriority:
    """Last-In-First-Out priority (reverse submission order)."""

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """Calculate LIFO priority (later = higher priority)."""
        return job.created_at.timestamp()


class UserDefinedPriority:
    """User-defined priority based on job priority attribute."""

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """Calculate priority based on job's priority attribute."""
        return job.priority.value


class WeightedPriority:
    """
    Weighted priority combining multiple factors.

    Combines user priority, wait time, and job size.
    """

    def __init__(
        self,
        priority_weight: float = 0.5,
        wait_time_weight: float = 0.3,
        size_weight: float = 0.2
    ):
        """
        Initialize weighted priority.

        Args:
            priority_weight: Weight for user-defined priority
            wait_time_weight: Weight for wait time
            size_weight: Weight for job size (smaller = higher priority)
        """
        self.priority_weight = priority_weight
        self.wait_time_weight = wait_time_weight
        self.size_weight = size_weight

        # Normalize weights
        total = priority_weight + wait_time_weight + size_weight
        self.priority_weight /= total
        self.wait_time_weight /= total
        self.size_weight /= total

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """Calculate weighted priority."""
        # User-defined priority (0-3, normalized to 0-1)
        priority_score = job.priority.value / 3.0

        # Wait time (normalized to 0-1, capped at 1 hour)
        wait_time = job.wait_time or 0
        wait_score = min(wait_time / 3600.0, 1.0)

        # Job size (number of tasks, normalized)
        # Smaller jobs have higher priority
        job_size = job.workflow.get_task_count()
        max_size = context.get('max_job_size', 100)
        size_score = 1.0 - min(job_size / max_size, 1.0)

        # Weighted combination
        total_score = (
            self.priority_weight * priority_score +
            self.wait_time_weight * wait_score +
            self.size_weight * size_score
        )

        return total_score


class ShortestJobFirst:
    """Shortest Job First (SJF) priority."""

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """Calculate SJF priority (smaller jobs = higher priority)."""
        job_size = job.workflow.get_task_count()
        # Return negative size so smaller jobs have higher priority
        return -job_size


class DeadlinePriority:
    """
    Earliest Deadline First (EDF) priority.

    Jobs with earlier deadlines have higher priority.
    """

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """Calculate EDF priority."""
        deadline = job.metadata.get('deadline')

        if deadline is None:
            # No deadline, use low priority
            return 0.0

        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)

        # Time until deadline
        time_remaining = (deadline - datetime.now()).total_seconds()

        # Negative so earlier deadlines have higher priority
        return -time_remaining


class FairSharePriority:
    """
    Fair-share priority.

    Balances priorities across different users/owners.
    """

    def __init__(self):
        """Initialize fair-share priority."""
        self._user_job_counts: Dict[str, int] = {}

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """Calculate fair-share priority."""
        owner = job.owner or 'default'

        # Count jobs by owner
        job_count = self._user_job_counts.get(owner, 0)

        # Users with fewer running jobs get higher priority
        # Negative so fewer jobs = higher priority
        return -job_count

    def update_counts(self, jobs: list):
        """
        Update job counts per owner.

        Args:
            jobs: List of currently running jobs
        """
        self._user_job_counts.clear()

        for job in jobs:
            owner = job.owner or 'default'
            self._user_job_counts[owner] = self._user_job_counts.get(owner, 0) + 1


class AdaptivePriority:
    """
    Adaptive priority that adjusts based on system load.

    Prioritizes smaller jobs under high load, larger jobs under low load.
    """

    def calculate_priority(self, job: Job, context: Dict[str, Any]) -> float:
        """Calculate adaptive priority."""
        system_load = context.get('system_load', 0.5)  # 0-1
        job_size = job.workflow.get_task_count()
        max_size = context.get('max_job_size', 100)

        # Under high load, prefer smaller jobs
        # Under low load, prefer larger jobs (better utilization)
        size_score = job_size / max_size

        if system_load > 0.7:
            # High load: prefer smaller jobs
            priority = 1.0 - size_score
        elif system_load < 0.3:
            # Low load: prefer larger jobs
            priority = size_score
        else:
            # Medium load: FIFO
            priority = -job.created_at.timestamp() / 1e10  # Normalized

        # Incorporate user priority
        user_priority = job.priority.value / 3.0

        return 0.7 * priority + 0.3 * user_priority


# Factory for creating priority strategies
def create_priority_strategy(strategy_name: str) -> PriorityStrategy:
    """
    Create a priority strategy by name.

    Args:
        strategy_name: Strategy name

    Returns:
        Priority strategy instance
    """
    strategies = {
        'fifo': FIFOPriority,
        'lifo': LIFOPriority,
        'priority': UserDefinedPriority,
        'weighted': WeightedPriority,
        'sjf': ShortestJobFirst,
        'edf': DeadlinePriority,
        'fair_share': FairSharePriority,
        'adaptive': AdaptivePriority,
    }

    strategy_class = strategies.get(strategy_name.lower())

    if strategy_class is None:
        raise ValueError(
            f"Unknown priority strategy: {strategy_name}. "
            f"Available: {', '.join(strategies.keys())}"
        )

    return strategy_class()


# Export
__all__ = [
    'PriorityStrategy',
    'FIFOPriority',
    'LIFOPriority',
    'UserDefinedPriority',
    'WeightedPriority',
    'ShortestJobFirst',
    'DeadlinePriority',
    'FairSharePriority',
    'AdaptivePriority',
    'create_priority_strategy',
]
