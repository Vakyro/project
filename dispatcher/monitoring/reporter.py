"""
Status reporting for dispatcher.

Provides formatted status reports for workflows, tasks, and resources.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import json


class StatusReporter:
    """Formats and reports dispatcher status."""

    def __init__(self):
        """Initialize status reporter."""
        pass

    def format_workflow_status(self, workflow_result: Any) -> str:
        """
        Format workflow status report.

        Args:
            workflow_result: Workflow result

        Returns:
            Formatted status string
        """
        lines = []
        lines.append(f"Workflow Status: {workflow_result.status.value.upper()}")
        lines.append(f"Duration: {workflow_result.duration:.2f}s")
        lines.append(f"Tasks completed: {len(workflow_result.completed_tasks)}")

        if workflow_result.failed_tasks:
            lines.append(f"Tasks failed: {len(workflow_result.failed_tasks)}")
            for task_name in workflow_result.failed_tasks:
                lines.append(f"  - {task_name}")

        return "\n".join(lines)

    def format_task_status(self, task_result: Any) -> str:
        """Format task status report."""
        lines = []
        lines.append(f"Task Status: {task_result.status.value.upper()}")

        if task_result.duration:
            lines.append(f"Duration: {task_result.duration:.2f}s")

        if task_result.error:
            lines.append(f"Error: {task_result.error}")

        return "\n".join(lines)

    def format_job_queue_status(self, queue_stats: Dict[str, Any]) -> str:
        """Format job queue status."""
        lines = []
        lines.append("Job Queue Status:")
        lines.append(f"  Queued: {queue_stats.get('queued', 0)}")
        lines.append(f"  Running: {queue_stats.get('running', 0)}")
        lines.append(f"  Completed: {queue_stats.get('completed', 0)}")
        lines.append(f"  Total: {queue_stats.get('total_jobs', 0)}")

        return "\n".join(lines)

    def format_resource_status(self, resource_stats: Dict[str, Any]) -> str:
        """Format resource status."""
        lines = []
        lines.append("Resource Status:")

        # GPU
        if 'gpu' in resource_stats:
            gpu_info = resource_stats['gpu']
            lines.append(f"  GPUs: {gpu_info.get('total_gpus', 0)} total")

            if 'gpu_info' in gpu_info:
                for gpu_id, info in gpu_info['gpu_info'].items():
                    lines.append(
                        f"    GPU {gpu_id}: {info.get('memory_free', 0)}MB free / "
                        f"{info.get('memory_total', 0)}MB total"
                    )

        # Memory
        if 'memory' in resource_stats:
            mem_info = resource_stats['memory']
            if 'system_memory' in mem_info:
                sys_mem = mem_info['system_memory']
                lines.append(
                    f"  System Memory: {sys_mem.get('available_mb', 0):.0f}MB available / "
                    f"{sys_mem.get('total_mb', 0):.0f}MB total "
                    f"({sys_mem.get('percent', 0):.1f}% used)"
                )

        return "\n".join(lines)

    def format_full_status(self, scheduler_stats: Dict[str, Any]) -> str:
        """Format complete dispatcher status."""
        lines = []
        lines.append("=" * 60)
        lines.append("DISPATCHER STATUS")
        lines.append("=" * 60)

        # Queue status
        if 'queue' in scheduler_stats:
            lines.append("")
            lines.append(self.format_job_queue_status(scheduler_stats['queue']))

        # Resource status
        if 'resources' in scheduler_stats:
            lines.append("")
            lines.append(self.format_resource_status(scheduler_stats['resources']))

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_json(self, data: Dict[str, Any]) -> str:
        """Convert data to JSON."""
        return json.dumps(data, indent=2, default=str)


# Global reporter
_global_reporter: Optional[StatusReporter] = None


def get_reporter() -> StatusReporter:
    """Get global status reporter."""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = StatusReporter()
    return _global_reporter


# Export
__all__ = [
    'StatusReporter',
    'get_reporter',
]
