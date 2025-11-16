"""
Task abstraction for the dispatcher system.

Defines base classes for tasks, task results, and task contexts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable
from enum import Enum
from datetime import datetime
import uuid


class TaskStatus(Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Priority levels for task execution."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskContext:
    """Context information passed to tasks during execution."""

    # Task identification
    task_id: str
    task_name: str

    # Execution context
    workflow_id: Optional[str] = None
    parent_task_id: Optional[str] = None

    # Resource allocation
    gpu_ids: List[int] = field(default_factory=list)
    memory_limit_mb: Optional[int] = None

    # Runtime state
    start_time: Optional[datetime] = None
    attempt: int = 0

    # Shared data between tasks
    shared_state: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Callbacks
    progress_callback: Optional[Callable[[float, str], None]] = None
    metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class TaskResult:
    """Result of a task execution."""

    # Status
    status: TaskStatus

    # Output data
    output: Optional[Any] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error information
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success(self) -> bool:
        """Whether the task succeeded."""
        return self.status == TaskStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'output': self.output,
            'metadata': self.metadata,
            'error': str(self.error) if self.error else None,
            'error_traceback': self.error_traceback,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'metrics': self.metrics,
        }


@dataclass
class TaskConfig:
    """Configuration for a task."""

    # Task identification
    name: str
    description: str = ""

    # Execution settings
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[int] = None  # seconds

    # Retry settings
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    retry_on_errors: List[type] = field(default_factory=lambda: [Exception])

    # Resource requirements
    gpu_required: bool = False
    min_gpus: int = 0
    max_gpus: int = 1
    memory_mb: Optional[int] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Caching
    cache_enabled: bool = False
    cache_key_fn: Optional[Callable[..., str]] = None

    # Task-specific config
    task_params: Dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """
    Base class for all tasks.

    Tasks are the fundamental units of work in the dispatcher system.
    Each task performs a specific operation and can have dependencies on other tasks.
    """

    def __init__(self, config: TaskConfig):
        """
        Initialize task.

        Args:
            config: Task configuration
        """
        self.config = config
        self.task_id = str(uuid.uuid4())
        self._result: Optional[TaskResult] = None

    @abstractmethod
    def execute(self, context: TaskContext) -> TaskResult:
        """
        Execute the task.

        Args:
            context: Execution context

        Returns:
            Task result
        """
        pass

    def validate_inputs(self, context: TaskContext) -> bool:
        """
        Validate task inputs before execution.

        Args:
            context: Execution context

        Returns:
            True if inputs are valid
        """
        return True

    def cleanup(self, context: TaskContext):
        """
        Cleanup after task execution.

        Args:
            context: Execution context
        """
        pass

    def estimate_duration(self, context: TaskContext) -> Optional[float]:
        """
        Estimate task duration in seconds.

        Args:
            context: Execution context

        Returns:
            Estimated duration or None if unknown
        """
        return None

    def estimate_memory(self, context: TaskContext) -> Optional[int]:
        """
        Estimate memory requirements in MB.

        Args:
            context: Execution context

        Returns:
            Estimated memory or None if unknown
        """
        return self.config.memory_mb

    @property
    def name(self) -> str:
        """Task name."""
        return self.config.name

    @property
    def result(self) -> Optional[TaskResult]:
        """Last execution result."""
        return self._result

    def __repr__(self) -> str:
        return f"Task(name={self.name}, id={self.task_id})"


class FunctionTask(Task):
    """
    Task that wraps a simple function.

    Useful for creating tasks from functions without subclassing.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[TaskContext], Any],
        config: Optional[TaskConfig] = None
    ):
        """
        Initialize function task.

        Args:
            name: Task name
            func: Function to execute
            config: Task configuration (optional)
        """
        if config is None:
            config = TaskConfig(name=name)
        else:
            config.name = name

        super().__init__(config)
        self.func = func

    def execute(self, context: TaskContext) -> TaskResult:
        """Execute the wrapped function."""
        start_time = datetime.now()

        try:
            output = self.func(context)

            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=output,
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            import traceback
            return TaskResult(
                status=TaskStatus.FAILED,
                error=e,
                error_traceback=traceback.format_exc(),
                start_time=start_time,
                end_time=datetime.now()
            )


# Export
__all__ = [
    'Task',
    'TaskConfig',
    'TaskContext',
    'TaskResult',
    'TaskStatus',
    'TaskPriority',
    'FunctionTask',
]
