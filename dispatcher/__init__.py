"""
CLIPZyme Dispatcher System

A complete workflow orchestration system for CLIPZyme virtual screening.

Features:
- Task abstraction and dependency management
- Workflow orchestration with DAG execution
- Resource management (GPU, memory, cache)
- Job scheduling with priorities
- Monitoring and progress tracking
- Python API and CLI

Quick Start:
    >>> from dispatcher import DispatcherAPI
    >>> from dispatcher.workflows import create_screening_workflow
    >>>
    >>> # Create dispatcher
    >>> dispatcher = DispatcherAPI()
    >>>
    >>> # Create workflow
    >>> workflow = create_screening_workflow(
    ...     checkpoint_name="clipzyme_official_v1",
    ...     reactions=["CC(=O)O>>CCO"],
    ...     proteins_csv="data/proteins.csv"
    ... )
    >>>
    >>> # Submit job
    >>> job_id = dispatcher.submit_workflow(workflow)
    >>>
    >>> # Wait for completion
    >>> result = dispatcher.wait_for_job(job_id)
    >>> print(result.status)

CLI Usage:
    # Start dispatcher
    $ python -m dispatcher.api.cli start

    # Run screening
    $ python -m dispatcher.api.cli screen clipzyme_v1 reactions.txt

    # Check status
    $ python -m dispatcher.api.cli stats
"""

__version__ = "1.0.0"

# Core components
from .core import (
    Task,
    TaskConfig,
    TaskContext,
    TaskResult,
    TaskStatus,
    TaskPriority,
    Workflow,
    WorkflowBuilder,
    WorkflowConfig,
    WorkflowResult,
    WorkflowStatus,
    TaskExecutor,
    WorkflowExecutor,
    register_task,
    create_task,
)

# Scheduler
from .scheduler import (
    Scheduler,
    JobQueue,
    Job,
    JobStatus,
)

# Resources
from .resources import (
    GPUManager,
    MemoryManager,
    get_gpu_manager,
    get_memory_manager,
)

# Monitoring
from .monitoring import (
    configure_logging,
    get_logger,
    get_metrics_collector,
    ProgressTracker,
)

# Configuration
from .config import (
    ConfigResolver,
    ConfigValidator,
)

# API
from .api import (
    DispatcherAPI,
    cli,
)

# Tasks
from .tasks import (
    DownloadCheckpointTask,
    ValidateCheckpointTask,
    LoadCheckpointTask,
    BuildScreeningSetTask,
    RunScreeningTask,
    EvaluateScreeningTask,
    GenerateReportTask,
)

# Workflows
from .workflows import (
    create_screening_workflow,
    create_simple_screening_workflow,
    create_evaluation_workflow,
    create_full_pipeline,
)


__all__ = [
    # Version
    '__version__',

    # Core
    'Task',
    'TaskConfig',
    'TaskContext',
    'TaskResult',
    'TaskStatus',
    'TaskPriority',
    'Workflow',
    'WorkflowBuilder',
    'WorkflowConfig',
    'WorkflowResult',
    'WorkflowStatus',
    'TaskExecutor',
    'WorkflowExecutor',
    'register_task',
    'create_task',

    # Scheduler
    'Scheduler',
    'JobQueue',
    'Job',
    'JobStatus',

    # Resources
    'GPUManager',
    'MemoryManager',
    'get_gpu_manager',
    'get_memory_manager',

    # Monitoring
    'configure_logging',
    'get_logger',
    'get_metrics_collector',
    'ProgressTracker',

    # Configuration
    'ConfigResolver',
    'ConfigValidator',

    # API
    'DispatcherAPI',
    'cli',

    # Tasks
    'DownloadCheckpointTask',
    'ValidateCheckpointTask',
    'LoadCheckpointTask',
    'BuildScreeningSetTask',
    'RunScreeningTask',
    'EvaluateScreeningTask',
    'GenerateReportTask',

    # Workflows
    'create_screening_workflow',
    'create_simple_screening_workflow',
    'create_evaluation_workflow',
    'create_full_pipeline',
]
