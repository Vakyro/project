"""
Core dispatcher components.

Provides task abstraction, workflow orchestration, execution engine, and state management.
"""

from .task import (
    Task,
    TaskConfig,
    TaskContext,
    TaskResult,
    TaskStatus,
    TaskPriority,
    FunctionTask,
)

from .registry import (
    TaskRegistry,
    register_task,
    register_function_task,
    create_task,
    get_registry,
    task,
)

from .workflow import (
    Workflow,
    WorkflowBuilder,
    WorkflowConfig,
    WorkflowResult,
    WorkflowStatus,
    DependencyGraph,
)

from .executor import (
    TaskExecutor,
    WorkflowExecutor,
    ExecutionContext,
)

from .state import (
    StateManager,
    TaskState,
    WorkflowState,
)


__all__ = [
    # Task
    'Task',
    'TaskConfig',
    'TaskContext',
    'TaskResult',
    'TaskStatus',
    'TaskPriority',
    'FunctionTask',
    # Registry
    'TaskRegistry',
    'register_task',
    'register_function_task',
    'create_task',
    'get_registry',
    'task',
    # Workflow
    'Workflow',
    'WorkflowBuilder',
    'WorkflowConfig',
    'WorkflowResult',
    'WorkflowStatus',
    'DependencyGraph',
    # Executor
    'TaskExecutor',
    'WorkflowExecutor',
    'ExecutionContext',
    # State
    'StateManager',
    'TaskState',
    'WorkflowState',
]
