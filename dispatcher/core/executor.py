"""
Execution engine for tasks and workflows.

Orchestrates task execution based on dependency graphs with retry logic,
resource management, and monitoring integration.
"""

from typing import Dict, Optional, Set, List, Any
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime
import traceback
import time
import logging

from .task import Task, TaskContext, TaskResult, TaskStatus, TaskPriority
from .workflow import Workflow, WorkflowResult, WorkflowStatus
from .state import StateManager


logger = logging.getLogger(__name__)


class ExecutionContext:
    """
    Context for workflow execution.

    Manages shared state, resource allocation, and execution tracking.
    """

    def __init__(self, workflow: Workflow):
        """
        Initialize execution context.

        Args:
            workflow: Workflow to execute
        """
        self.workflow = workflow
        self.state_manager = StateManager()

        # Execution state
        self.completed_tasks: Set[str] = set()
        self.running_tasks: Dict[str, Future] = {}
        self.failed_tasks: Set[str] = set()
        self.task_results: Dict[str, TaskResult] = {}

        # Retry tracking
        self.retry_counts: Dict[str, int] = {}

        # Timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def is_task_ready(self, task_name: str) -> bool:
        """Check if task is ready to execute."""
        if task_name in self.completed_tasks or task_name in self.running_tasks:
            return False

        dependencies = self.workflow.graph.get_dependencies(task_name)
        return dependencies.issubset(self.completed_tasks)

    def get_ready_tasks(self) -> List[str]:
        """Get all tasks ready to execute."""
        return [
            task_name
            for task_name in self.workflow.graph._tasks.keys()
            if self.is_task_ready(task_name)
        ]

    def mark_completed(self, task_name: str, result: TaskResult):
        """Mark task as completed."""
        self.completed_tasks.add(task_name)
        self.task_results[task_name] = result

        if task_name in self.running_tasks:
            del self.running_tasks[task_name]

    def mark_failed(self, task_name: str, result: TaskResult):
        """Mark task as failed."""
        self.failed_tasks.add(task_name)
        self.task_results[task_name] = result

        if task_name in self.running_tasks:
            del self.running_tasks[task_name]

    def should_retry(self, task_name: str) -> bool:
        """Check if task should be retried."""
        task = self.workflow.get_task(task_name)
        if not task:
            return False

        retry_count = self.retry_counts.get(task_name, 0)
        return retry_count < task.config.max_retries

    def increment_retry(self, task_name: str):
        """Increment retry count for task."""
        self.retry_counts[task_name] = self.retry_counts.get(task_name, 0) + 1


class TaskExecutor:
    """
    Executor for individual tasks.

    Handles task execution with retry logic, timeout, and error handling.
    """

    def __init__(self):
        """Initialize task executor."""
        self.logger = logging.getLogger(f"{__name__}.TaskExecutor")

    def execute(
        self,
        task: Task,
        workflow_id: str,
        shared_state: Dict[str, Any],
        attempt: int = 0
    ) -> TaskResult:
        """
        Execute a single task.

        Args:
            task: Task to execute
            workflow_id: ID of parent workflow
            shared_state: Shared state dictionary
            attempt: Retry attempt number

        Returns:
            Task result
        """
        start_time = datetime.now()

        # Create task context
        context = TaskContext(
            task_id=task.task_id,
            task_name=task.name,
            workflow_id=workflow_id,
            start_time=start_time,
            attempt=attempt,
            shared_state=shared_state,
            config=task.config.task_params
        )

        try:
            # Validate inputs
            self.logger.info(f"Validating inputs for task: {task.name}")
            if not task.validate_inputs(context):
                raise ValueError(f"Input validation failed for task: {task.name}")

            # Execute task
            self.logger.info(f"Executing task: {task.name} (attempt {attempt + 1})")
            result = task.execute(context)

            # Update timing
            result.start_time = start_time
            if result.end_time is None:
                result.end_time = datetime.now()

            # Log result
            if result.success:
                self.logger.info(
                    f"Task completed: {task.name} "
                    f"(duration: {result.duration:.2f}s)"
                )
            else:
                self.logger.error(
                    f"Task failed: {task.name} - {result.error}"
                )

            return result

        except Exception as e:
            # Handle execution error
            self.logger.error(
                f"Task execution error: {task.name} - {str(e)}\n"
                f"{traceback.format_exc()}"
            )

            return TaskResult(
                status=TaskStatus.FAILED,
                error=e,
                error_traceback=traceback.format_exc(),
                start_time=start_time,
                end_time=datetime.now()
            )

        finally:
            # Cleanup
            try:
                task.cleanup(context)
            except Exception as e:
                self.logger.warning(
                    f"Task cleanup error: {task.name} - {str(e)}"
                )

    def execute_with_retry(
        self,
        task: Task,
        workflow_id: str,
        shared_state: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> TaskResult:
        """
        Execute task with retry logic.

        Args:
            task: Task to execute
            workflow_id: Workflow ID
            shared_state: Shared state
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)

        Returns:
            Task result
        """
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.info(
                    f"Retrying task: {task.name} "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(retry_delay)

            result = self.execute(
                task=task,
                workflow_id=workflow_id,
                shared_state=shared_state,
                attempt=attempt
            )

            if result.success:
                return result

            # Check if error is retryable
            if result.error and not any(
                isinstance(result.error, err_type)
                for err_type in task.config.retry_on_errors
            ):
                self.logger.info(
                    f"Error not retryable for task: {task.name} - "
                    f"{type(result.error).__name__}"
                )
                break

        # Mark as failed after all retries
        result.status = TaskStatus.FAILED
        return result


class WorkflowExecutor:
    """
    Executor for workflows.

    Orchestrates workflow execution based on dependency graph.
    Supports parallel execution with resource management.
    """

    def __init__(
        self,
        max_workers: int = 4,
        resource_manager: Optional[Any] = None
    ):
        """
        Initialize workflow executor.

        Args:
            max_workers: Maximum parallel workers
            resource_manager: Resource manager instance (optional)
        """
        self.max_workers = max_workers
        self.resource_manager = resource_manager
        self.task_executor = TaskExecutor()
        self.logger = logging.getLogger(f"{__name__}.WorkflowExecutor")

    def execute(self, workflow: Workflow) -> WorkflowResult:
        """
        Execute workflow.

        Args:
            workflow: Workflow to execute

        Returns:
            Workflow result
        """
        self.logger.info(f"Starting workflow execution: {workflow.config.name}")

        # Validate workflow
        try:
            workflow.validate()
        except Exception as e:
            self.logger.error(f"Workflow validation failed: {str(e)}")
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=e,
                start_time=datetime.now(),
                end_time=datetime.now()
            )

        # Create execution context
        ctx = ExecutionContext(workflow)
        ctx.start_time = datetime.now()

        # Execute workflow
        try:
            result = self._execute_workflow(workflow, ctx)
            result.start_time = ctx.start_time
            result.end_time = datetime.now()

            self.logger.info(
                f"Workflow completed: {workflow.config.name} "
                f"(status: {result.status.value}, duration: {result.duration:.2f}s)"
            )

            # Call completion callback
            if workflow.config.on_workflow_complete:
                try:
                    workflow.config.on_workflow_complete(result)
                except Exception as e:
                    self.logger.warning(
                        f"Workflow completion callback error: {str(e)}"
                    )

            return result

        except Exception as e:
            self.logger.error(
                f"Workflow execution error: {str(e)}\n"
                f"{traceback.format_exc()}"
            )

            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=e,
                task_results=ctx.task_results,
                failed_tasks=list(ctx.failed_tasks),
                start_time=ctx.start_time,
                end_time=datetime.now()
            )

    def _execute_workflow(
        self,
        workflow: Workflow,
        ctx: ExecutionContext
    ) -> WorkflowResult:
        """
        Internal workflow execution.

        Args:
            workflow: Workflow to execute
            ctx: Execution context

        Returns:
            Workflow result
        """
        total_tasks = workflow.get_task_count()
        shared_state = {}

        # Use thread pool for parallel execution
        max_parallel = min(
            workflow.config.max_parallel_tasks,
            self.max_workers
        )

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            while len(ctx.completed_tasks) + len(ctx.failed_tasks) < total_tasks:
                # Get ready tasks
                ready_tasks = ctx.get_ready_tasks()

                if not ready_tasks and not ctx.running_tasks:
                    # No ready tasks and nothing running - check for failures
                    if ctx.failed_tasks:
                        break
                    else:
                        raise RuntimeError(
                            "Workflow deadlock: no ready tasks but workflow incomplete"
                        )

                # Submit ready tasks
                for task_name in ready_tasks:
                    if len(ctx.running_tasks) >= max_parallel:
                        break

                    task = workflow.get_task(task_name)
                    if not task:
                        continue

                    # Submit task for execution
                    future = executor.submit(
                        self.task_executor.execute_with_retry,
                        task=task,
                        workflow_id=workflow.workflow_id,
                        shared_state=shared_state,
                        max_retries=task.config.max_retries,
                        retry_delay=task.config.retry_delay
                    )

                    ctx.running_tasks[task_name] = future
                    self.logger.debug(f"Submitted task: {task_name}")

                # Wait for at least one task to complete
                if ctx.running_tasks:
                    completed = as_completed(
                        ctx.running_tasks.values(),
                        timeout=1.0
                    )

                    for future in completed:
                        # Find task name for this future
                        task_name = None
                        for name, f in ctx.running_tasks.items():
                            if f == future:
                                task_name = name
                                break

                        if not task_name:
                            continue

                        # Get result
                        try:
                            result = future.result()
                        except Exception as e:
                            self.logger.error(
                                f"Task future error: {task_name} - {str(e)}"
                            )
                            result = TaskResult(
                                status=TaskStatus.FAILED,
                                error=e,
                                error_traceback=traceback.format_exc()
                            )

                        # Handle result
                        if result.success:
                            ctx.mark_completed(task_name, result)

                            # Call success callback
                            if workflow.config.on_task_complete:
                                try:
                                    workflow.config.on_task_complete(task_name, result)
                                except Exception as e:
                                    self.logger.warning(
                                        f"Task complete callback error: {str(e)}"
                                    )

                        else:
                            ctx.mark_failed(task_name, result)

                            # Call failure callback
                            if workflow.config.on_task_failed:
                                try:
                                    workflow.config.on_task_failed(task_name, result)
                                except Exception as e:
                                    self.logger.warning(
                                        f"Task failed callback error: {str(e)}"
                                    )

                            # Check if should continue
                            if workflow.config.fail_fast:
                                self.logger.error(
                                    f"Failing fast due to task failure: {task_name}"
                                )
                                # Cancel remaining tasks
                                for f in ctx.running_tasks.values():
                                    f.cancel()
                                break

                        # Break after processing one completion
                        break

                    # Check for timeout
                    if workflow.config.timeout:
                        elapsed = (datetime.now() - ctx.start_time).total_seconds()
                        if elapsed > workflow.config.timeout:
                            self.logger.error(
                                f"Workflow timeout after {elapsed:.2f}s"
                            )
                            # Cancel running tasks
                            for f in ctx.running_tasks.values():
                                f.cancel()
                            break

                else:
                    # No running tasks, wait a bit
                    time.sleep(0.1)

        # Build result
        if ctx.failed_tasks and not workflow.config.continue_on_error:
            status = WorkflowStatus.FAILED
        elif len(ctx.completed_tasks) == total_tasks:
            status = WorkflowStatus.COMPLETED
        else:
            status = WorkflowStatus.FAILED

        return WorkflowResult(
            status=status,
            task_results=ctx.task_results,
            failed_tasks=list(ctx.failed_tasks),
            metadata={
                'total_tasks': total_tasks,
                'completed_tasks': len(ctx.completed_tasks),
                'failed_tasks': len(ctx.failed_tasks),
            }
        )


# Export
__all__ = [
    'TaskExecutor',
    'WorkflowExecutor',
    'ExecutionContext',
]
