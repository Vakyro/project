"""
State management for workflows and tasks.

Provides persistent state tracking, caching, and recovery.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import threading
import pickle


@dataclass
class TaskState:
    """State information for a task."""

    task_id: str
    task_name: str
    status: str
    attempt: int = 0

    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Output
    output: Optional[Any] = None
    error: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskState':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class WorkflowState:
    """State information for a workflow."""

    workflow_id: str
    workflow_name: str
    status: str

    # Task states
    task_states: Dict[str, TaskState] = field(default_factory=dict)

    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert task states
        data['task_states'] = {
            name: state.to_dict() if isinstance(state, TaskState) else state
            for name, state in self.task_states.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create from dictionary."""
        # Convert task states
        task_states = {
            name: TaskState.from_dict(state) if isinstance(state, dict) else state
            for name, state in data.get('task_states', {}).items()
        }
        data['task_states'] = task_states
        return cls(**data)


class StateManager:
    """
    Manager for workflow and task state.

    Provides in-memory and persistent state storage with thread-safety.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for persistent state storage
        """
        self.state_dir = state_dir or Path.home() / '.clipzyme' / 'dispatcher' / 'state'
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._workflow_states: Dict[str, WorkflowState] = {}
        self._task_states: Dict[str, TaskState] = {}

        # Cache
        self._cache: Dict[str, Any] = {}

        # Thread safety
        self._lock = threading.RLock()

    def save_workflow_state(self, state: WorkflowState, persist: bool = True):
        """
        Save workflow state.

        Args:
            state: Workflow state
            persist: Whether to persist to disk
        """
        with self._lock:
            self._workflow_states[state.workflow_id] = state

            if persist:
                self._persist_workflow_state(state)

    def load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Load workflow state.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow state or None
        """
        with self._lock:
            # Check memory first
            if workflow_id in self._workflow_states:
                return self._workflow_states[workflow_id]

            # Try loading from disk
            return self._load_workflow_state(workflow_id)

    def save_task_state(self, state: TaskState):
        """
        Save task state.

        Args:
            state: Task state
        """
        with self._lock:
            self._task_states[state.task_id] = state

    def load_task_state(self, task_id: str) -> Optional[TaskState]:
        """
        Load task state.

        Args:
            task_id: Task ID

        Returns:
            Task state or None
        """
        with self._lock:
            return self._task_states.get(task_id)

    def list_workflow_states(self, status: Optional[str] = None) -> List[WorkflowState]:
        """
        List workflow states.

        Args:
            status: Filter by status (optional)

        Returns:
            List of workflow states
        """
        with self._lock:
            states = list(self._workflow_states.values())

            if status:
                states = [s for s in states if s.status == status]

            return states

    def delete_workflow_state(self, workflow_id: str):
        """
        Delete workflow state.

        Args:
            workflow_id: Workflow ID
        """
        with self._lock:
            self._workflow_states.pop(workflow_id, None)

            # Delete from disk
            state_file = self.state_dir / f"{workflow_id}.json"
            if state_file.exists():
                state_file.unlink()

    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set cache value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (seconds, not implemented yet)
        """
        with self._lock:
            self._cache[key] = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'ttl': ttl
            }

    def cache_get(self, key: str) -> Optional[Any]:
        """
        Get cache value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            cached = self._cache.get(key)
            if cached:
                return cached['value']
            return None

    def cache_delete(self, key: str):
        """
        Delete cache entry.

        Args:
            key: Cache key
        """
        with self._lock:
            self._cache.pop(key, None)

    def cache_clear(self):
        """Clear all cache."""
        with self._lock:
            self._cache.clear()

    def _persist_workflow_state(self, state: WorkflowState):
        """Persist workflow state to disk."""
        state_file = self.state_dir / f"{state.workflow_id}.json"

        try:
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
        except Exception as e:
            # Log error but don't fail
            import logging
            logging.warning(f"Failed to persist workflow state: {str(e)}")

    def _load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state from disk."""
        state_file = self.state_dir / f"{workflow_id}.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            return WorkflowState.from_dict(data)
        except Exception as e:
            import logging
            logging.warning(f"Failed to load workflow state: {str(e)}")
            return None

    def cleanup_old_states(self, days: int = 7):
        """
        Clean up old workflow states.

        Args:
            days: Delete states older than this many days
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)

        with self._lock:
            # Clean up memory
            to_delete = []
            for workflow_id, state in self._workflow_states.items():
                if state.end_time:
                    end_time = datetime.fromisoformat(state.end_time)
                    if end_time.timestamp() < cutoff:
                        to_delete.append(workflow_id)

            for workflow_id in to_delete:
                self.delete_workflow_state(workflow_id)

            # Clean up disk
            for state_file in self.state_dir.glob("*.json"):
                if state_file.stat().st_mtime < cutoff:
                    state_file.unlink()


# Export
__all__ = [
    'StateManager',
    'TaskState',
    'WorkflowState',
]
