"""
Workflow orchestration engine with dependency graph management.

Provides workflow definition, DAG construction, and execution orchestration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict, deque
import uuid
from datetime import datetime
from enum import Enum

from .task import Task, TaskContext, TaskResult, TaskStatus


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""

    # Workflow identification
    name: str
    description: str = ""

    # Execution settings
    max_parallel_tasks: int = 4
    continue_on_error: bool = False
    fail_fast: bool = True

    # Retry settings
    max_workflow_retries: int = 0
    retry_failed_tasks: bool = True

    # Timeout
    timeout: Optional[int] = None  # seconds

    # Callbacks
    on_task_complete: Optional[Callable[[str, TaskResult], None]] = None
    on_task_failed: Optional[Callable[[str, TaskResult], None]] = None
    on_workflow_complete: Optional[Callable[['WorkflowResult'], None]] = None


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    # Status
    status: WorkflowStatus

    # Task results
    task_results: Dict[str, TaskResult] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Error information
    error: Optional[Exception] = None
    failed_tasks: List[str] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success(self) -> bool:
        """Whether workflow succeeded."""
        return self.status == WorkflowStatus.COMPLETED

    @property
    def completed_tasks(self) -> List[str]:
        """List of completed task names."""
        return [
            name for name, result in self.task_results.items()
            if result.status == TaskStatus.COMPLETED
        ]

    def get_task_result(self, task_name: str) -> Optional[TaskResult]:
        """Get result for a specific task."""
        return self.task_results.get(task_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'task_results': {
                name: result.to_dict()
                for name, result in self.task_results.items()
            },
            'metadata': self.metadata,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'failed_tasks': self.failed_tasks,
        }


class DependencyGraph:
    """
    Directed Acyclic Graph (DAG) for task dependencies.

    Manages task dependencies and provides topological ordering.
    """

    def __init__(self):
        """Initialize dependency graph."""
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)  # task -> dependencies
        self._dependents: Dict[str, Set[str]] = defaultdict(set)    # task -> dependents
        self._tasks: Dict[str, Task] = {}

    def add_task(self, task: Task, depends_on: Optional[List[str]] = None):
        """
        Add task to graph.

        Args:
            task: Task to add
            depends_on: List of task names this task depends on
        """
        task_name = task.name

        if task_name in self._tasks:
            raise ValueError(f"Task '{task_name}' already exists in graph")

        self._tasks[task_name] = task

        if depends_on:
            for dep in depends_on:
                self.add_dependency(task_name, dep)

    def add_dependency(self, task_name: str, depends_on: str):
        """
        Add dependency between tasks.

        Args:
            task_name: Name of dependent task
            depends_on: Name of task to depend on
        """
        if task_name not in self._tasks:
            raise ValueError(f"Task '{task_name}' not in graph")

        # Note: We allow forward references (depends_on not yet in graph)
        # They will be validated before execution

        self._dependencies[task_name].add(depends_on)
        self._dependents[depends_on].add(task_name)

    def get_dependencies(self, task_name: str) -> Set[str]:
        """Get all dependencies of a task."""
        return self._dependencies.get(task_name, set())

    def get_dependents(self, task_name: str) -> Set[str]:
        """Get all tasks that depend on this task."""
        return self._dependents.get(task_name, set())

    def get_task(self, task_name: str) -> Optional[Task]:
        """Get task by name."""
        return self._tasks.get(task_name)

    def has_task(self, task_name: str) -> bool:
        """Check if task exists in graph."""
        return task_name in self._tasks

    def validate(self) -> bool:
        """
        Validate graph for cycles and missing dependencies.

        Returns:
            True if graph is valid

        Raises:
            ValueError: If graph has cycles or missing dependencies
        """
        # Check for missing dependencies
        for task_name, deps in self._dependencies.items():
            for dep in deps:
                if dep not in self._tasks:
                    raise ValueError(
                        f"Task '{task_name}' depends on unknown task '{dep}'"
                    )

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for dep in self._dependencies.get(node, set()):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for task_name in self._tasks:
            if task_name not in visited:
                if has_cycle(task_name):
                    raise ValueError(
                        f"Circular dependency detected involving task '{task_name}'"
                    )

        return True

    def topological_sort(self) -> List[str]:
        """
        Get topological ordering of tasks.

        Returns:
            List of task names in execution order

        Raises:
            ValueError: If graph has cycles
        """
        self.validate()

        # Kahn's algorithm
        in_degree = {task: len(self._dependencies.get(task, set())) for task in self._tasks}
        queue = deque([task for task, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            task = queue.popleft()
            result.append(task)

            for dependent in self._dependents.get(task, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._tasks):
            raise ValueError("Graph has cycles - cannot perform topological sort")

        return result

    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """
        Get tasks ready to execute (all dependencies completed).

        Args:
            completed_tasks: Set of completed task names

        Returns:
            List of task names ready to execute
        """
        ready = []

        for task_name in self._tasks:
            if task_name in completed_tasks:
                continue

            dependencies = self._dependencies.get(task_name, set())
            if dependencies.issubset(completed_tasks):
                ready.append(task_name)

        return ready

    def get_execution_levels(self) -> List[List[str]]:
        """
        Get tasks grouped by execution level (tasks in same level can run in parallel).

        Returns:
            List of levels, each containing task names
        """
        self.validate()

        levels = []
        remaining = set(self._tasks.keys())
        completed = set()

        while remaining:
            # Find tasks with all dependencies completed
            ready = [
                task for task in remaining
                if self._dependencies.get(task, set()).issubset(completed)
            ]

            if not ready:
                raise ValueError("No ready tasks - graph may have cycles")

            levels.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)

        return levels

    def __len__(self) -> int:
        """Number of tasks in graph."""
        return len(self._tasks)

    def __contains__(self, task_name: str) -> bool:
        """Check if task is in graph."""
        return task_name in self._tasks


class Workflow:
    """
    Workflow definition and execution.

    A workflow is a collection of tasks with dependencies,
    executed according to the dependency graph.
    """

    def __init__(self, config: WorkflowConfig):
        """
        Initialize workflow.

        Args:
            config: Workflow configuration
        """
        self.config = config
        self.workflow_id = str(uuid.uuid4())
        self.graph = DependencyGraph()
        self._result: Optional[WorkflowResult] = None

    def add_task(self, task: Task, depends_on: Optional[List[str]] = None):
        """
        Add task to workflow.

        Args:
            task: Task to add
            depends_on: List of task names this task depends on
        """
        # Use task config dependencies if not explicitly provided
        if depends_on is None:
            depends_on = task.config.depends_on

        self.graph.add_task(task, depends_on)

    def add_tasks(self, tasks: List[Task]):
        """
        Add multiple tasks to workflow.

        Args:
            tasks: List of tasks to add
        """
        for task in tasks:
            self.add_task(task)

    def validate(self) -> bool:
        """
        Validate workflow.

        Returns:
            True if workflow is valid
        """
        return self.graph.validate()

    def get_execution_plan(self) -> List[List[str]]:
        """
        Get execution plan (tasks grouped by level).

        Returns:
            List of execution levels
        """
        return self.graph.get_execution_levels()

    def get_task(self, task_name: str) -> Optional[Task]:
        """Get task by name."""
        return self.graph.get_task(task_name)

    def get_task_count(self) -> int:
        """Get total number of tasks."""
        return len(self.graph)

    @property
    def result(self) -> Optional[WorkflowResult]:
        """Get workflow result."""
        return self._result

    def __repr__(self) -> str:
        return f"Workflow(name={self.config.name}, tasks={self.get_task_count()}, id={self.workflow_id})"


# Builder for fluent workflow construction
class WorkflowBuilder:
    """Builder for constructing workflows fluently."""

    def __init__(self, name: str, description: str = ""):
        """
        Initialize builder.

        Args:
            name: Workflow name
            description: Workflow description
        """
        self._config = WorkflowConfig(name=name, description=description)
        self._tasks: List[tuple[Task, Optional[List[str]]]] = []

    def set_max_parallel(self, count: int) -> 'WorkflowBuilder':
        """Set max parallel tasks."""
        self._config.max_parallel_tasks = count
        return self

    def set_continue_on_error(self, value: bool) -> 'WorkflowBuilder':
        """Set continue on error."""
        self._config.continue_on_error = value
        return self

    def set_fail_fast(self, value: bool) -> 'WorkflowBuilder':
        """Set fail fast."""
        self._config.fail_fast = value
        return self

    def set_timeout(self, seconds: int) -> 'WorkflowBuilder':
        """Set workflow timeout."""
        self._config.timeout = seconds
        return self

    def add_task(self, task: Task, depends_on: Optional[List[str]] = None) -> 'WorkflowBuilder':
        """Add task to workflow."""
        self._tasks.append((task, depends_on))
        return self

    def on_task_complete(self, callback: Callable) -> 'WorkflowBuilder':
        """Set task complete callback."""
        self._config.on_task_complete = callback
        return self

    def on_task_failed(self, callback: Callable) -> 'WorkflowBuilder':
        """Set task failed callback."""
        self._config.on_task_failed = callback
        return self

    def on_workflow_complete(self, callback: Callable) -> 'WorkflowBuilder':
        """Set workflow complete callback."""
        self._config.on_workflow_complete = callback
        return self

    def build(self) -> Workflow:
        """Build the workflow."""
        workflow = Workflow(self._config)

        for task, depends_on in self._tasks:
            workflow.add_task(task, depends_on)

        workflow.validate()

        return workflow


# Export
__all__ = [
    'Workflow',
    'WorkflowBuilder',
    'WorkflowConfig',
    'WorkflowResult',
    'WorkflowStatus',
    'DependencyGraph',
]
