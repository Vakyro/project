"""
Task registry for managing task types and instances.

Provides central registration and lookup of task types.
"""

from typing import Dict, Type, Optional, Callable, Any
from .task import Task, TaskConfig, FunctionTask
import threading


class TaskRegistry:
    """
    Registry for task types.

    Allows registration and creation of tasks by name.
    Thread-safe singleton pattern.
    """

    _instance: Optional['TaskRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry."""
        if self._initialized:
            return

        self._task_types: Dict[str, Type[Task]] = {}
        self._task_factories: Dict[str, Callable[..., Task]] = {}
        self._task_configs: Dict[str, TaskConfig] = {}
        self._lock_registry = threading.RLock()
        self._initialized = True

    def register(
        self,
        name: str,
        task_type: Optional[Type[Task]] = None,
        factory: Optional[Callable[..., Task]] = None,
        config: Optional[TaskConfig] = None
    ):
        """
        Register a task type or factory.

        Args:
            name: Task type name
            task_type: Task class (optional if factory provided)
            factory: Factory function (optional if task_type provided)
            config: Default task configuration
        """
        with self._lock_registry:
            if task_type is None and factory is None:
                raise ValueError("Either task_type or factory must be provided")

            if task_type is not None:
                self._task_types[name] = task_type

            if factory is not None:
                self._task_factories[name] = factory

            if config is not None:
                self._task_configs[name] = config

    def register_function(
        self,
        name: str,
        func: Callable,
        config: Optional[TaskConfig] = None
    ):
        """
        Register a function as a task.

        Args:
            name: Task name
            func: Function to wrap
            config: Task configuration
        """
        def factory(**kwargs):
            task_config = config or TaskConfig(name=name)
            # Merge kwargs into task_params
            if kwargs:
                task_config.task_params.update(kwargs)
            return FunctionTask(name=name, func=func, config=task_config)

        self.register(name=name, factory=factory, config=config)

    def create(
        self,
        name: str,
        config: Optional[TaskConfig] = None,
        **kwargs
    ) -> Task:
        """
        Create a task instance.

        Args:
            name: Task type name
            config: Task configuration (overrides default)
            **kwargs: Additional arguments passed to task constructor

        Returns:
            Task instance
        """
        with self._lock_registry:
            # Check if factory exists
            if name in self._task_factories:
                return self._task_factories[name](**kwargs)

            # Check if task type exists
            if name not in self._task_types:
                raise KeyError(f"Task type '{name}' not registered")

            task_type = self._task_types[name]

            # Get config
            if config is None:
                config = self._task_configs.get(name)

            if config is None:
                raise ValueError(f"No config provided for task '{name}'")

            # Create task
            return task_type(config=config, **kwargs)

    def get_task_type(self, name: str) -> Optional[Type[Task]]:
        """
        Get task type by name.

        Args:
            name: Task type name

        Returns:
            Task type or None
        """
        return self._task_types.get(name)

    def list_tasks(self) -> list[str]:
        """
        List all registered task names.

        Returns:
            List of task names
        """
        with self._lock_registry:
            all_names = set(self._task_types.keys()) | set(self._task_factories.keys())
            return sorted(all_names)

    def unregister(self, name: str):
        """
        Unregister a task type.

        Args:
            name: Task type name
        """
        with self._lock_registry:
            self._task_types.pop(name, None)
            self._task_factories.pop(name, None)
            self._task_configs.pop(name, None)

    def clear(self):
        """Clear all registered tasks."""
        with self._lock_registry:
            self._task_types.clear()
            self._task_factories.clear()
            self._task_configs.clear()


# Global registry instance
_global_registry = TaskRegistry()


def register_task(
    name: str,
    task_type: Optional[Type[Task]] = None,
    factory: Optional[Callable[..., Task]] = None,
    config: Optional[TaskConfig] = None
):
    """
    Register a task type in the global registry.

    Args:
        name: Task type name
        task_type: Task class
        factory: Factory function
        config: Default configuration
    """
    _global_registry.register(name, task_type, factory, config)


def register_function_task(
    name: str,
    func: Callable,
    config: Optional[TaskConfig] = None
):
    """
    Register a function as a task in the global registry.

    Args:
        name: Task name
        func: Function to wrap
        config: Task configuration
    """
    _global_registry.register_function(name, func, config)


def create_task(
    name: str,
    config: Optional[TaskConfig] = None,
    **kwargs
) -> Task:
    """
    Create a task from the global registry.

    Args:
        name: Task type name
        config: Task configuration
        **kwargs: Additional arguments

    Returns:
        Task instance
    """
    return _global_registry.create(name, config, **kwargs)


def get_registry() -> TaskRegistry:
    """Get the global task registry."""
    return _global_registry


# Decorator for registering tasks
def task(
    name: Optional[str] = None,
    config: Optional[TaskConfig] = None
):
    """
    Decorator for registering task classes or functions.

    Usage:
        @task(name="my_task")
        class MyTask(Task):
            ...

        @task(name="my_func_task")
        def my_function(context):
            ...
    """
    def decorator(obj):
        task_name = name or (obj.__name__ if hasattr(obj, '__name__') else str(obj))

        if isinstance(obj, type) and issubclass(obj, Task):
            # Registering a task class
            _global_registry.register(task_name, task_type=obj, config=config)
        elif callable(obj):
            # Registering a function
            _global_registry.register_function(task_name, obj, config)
        else:
            raise TypeError(f"Cannot register {type(obj)} as task")

        return obj

    return decorator


# Export
__all__ = [
    'TaskRegistry',
    'register_task',
    'register_function_task',
    'create_task',
    'get_registry',
    'task',
]
