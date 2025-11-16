"""
Memory resource management.

Provides memory monitoring and allocation tracking.
"""

from typing import Optional, Dict
from dataclasses import dataclass
import psutil
import threading
import logging


logger = logging.getLogger(__name__)


@dataclass
class MemoryInfo:
    """Memory usage information."""

    total: int        # Total memory (MB)
    available: int    # Available memory (MB)
    used: int         # Used memory (MB)
    percent: float    # Usage percentage (0-100)

    @property
    def available_ratio(self) -> float:
        """Ratio of available memory (0-1)."""
        if self.total == 0:
            return 0.0
        return self.available / self.total


class MemoryManager:
    """
    Manager for memory resource monitoring and allocation.

    Provides memory usage tracking and allocation recommendations.
    Thread-safe singleton pattern.
    """

    _instance: Optional['MemoryManager'] = None
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
        """Initialize memory manager."""
        if self._initialized:
            return

        self.logger = logging.getLogger(f"{__name__}.MemoryManager")

        # Memory allocation tracking
        self._allocations: Dict[str, int] = {}  # task_id -> allocated_mb

        # Thread safety
        self._allocation_lock = threading.RLock()

        self._initialized = True

    def get_memory_info(self) -> MemoryInfo:
        """
        Get current memory information.

        Returns:
            Memory information
        """
        try:
            mem = psutil.virtual_memory()
            return MemoryInfo(
                total=mem.total // (1024 * 1024),      # Convert to MB
                available=mem.available // (1024 * 1024),
                used=mem.used // (1024 * 1024),
                percent=mem.percent
            )
        except Exception as e:
            self.logger.error(f"Error getting memory info: {str(e)}")
            return MemoryInfo(total=0, available=0, used=0, percent=0.0)

    def allocate_memory(self, task_id: str, memory_mb: int) -> bool:
        """
        Allocate memory for a task.

        Args:
            task_id: Task ID
            memory_mb: Memory to allocate (MB)

        Returns:
            True if allocation successful
        """
        with self._allocation_lock:
            # Check if already allocated
            if task_id in self._allocations:
                self.logger.warning(
                    f"Task {task_id} already has memory allocated"
                )
                return True

            # Check if enough memory available
            mem_info = self.get_memory_info()
            total_allocated = sum(self._allocations.values())
            available = mem_info.available - total_allocated

            if available < memory_mb:
                self.logger.warning(
                    f"Not enough memory for task {task_id}. "
                    f"Requested: {memory_mb}MB, Available: {available}MB"
                )
                return False

            # Allocate memory
            self._allocations[task_id] = memory_mb
            self.logger.info(
                f"Allocated {memory_mb}MB memory to task {task_id}"
            )

            return True

    def release_memory(self, task_id: str):
        """
        Release memory allocated to a task.

        Args:
            task_id: Task ID
        """
        with self._allocation_lock:
            if task_id in self._allocations:
                memory_mb = self._allocations[task_id]
                del self._allocations[task_id]
                self.logger.info(
                    f"Released {memory_mb}MB memory from task {task_id}"
                )

    def get_allocated_memory(self, task_id: str) -> int:
        """
        Get memory allocated to a task.

        Args:
            task_id: Task ID

        Returns:
            Allocated memory (MB)
        """
        with self._allocation_lock:
            return self._allocations.get(task_id, 0)

    def get_total_allocated(self) -> int:
        """
        Get total allocated memory across all tasks.

        Returns:
            Total allocated memory (MB)
        """
        with self._allocation_lock:
            return sum(self._allocations.values())

    def get_available_memory(self) -> int:
        """
        Get available memory for allocation.

        Returns:
            Available memory (MB)
        """
        mem_info = self.get_memory_info()
        total_allocated = self.get_total_allocated()
        return mem_info.available - total_allocated

    def can_allocate(self, memory_mb: int) -> bool:
        """
        Check if memory can be allocated.

        Args:
            memory_mb: Memory to allocate (MB)

        Returns:
            True if allocation is possible
        """
        available = self.get_available_memory()
        return available >= memory_mb

    def suggest_batch_size(
        self,
        item_memory_mb: int,
        max_batch_size: int = 1024,
        memory_reserve_ratio: float = 0.2
    ) -> int:
        """
        Suggest optimal batch size based on available memory.

        Args:
            item_memory_mb: Memory per item (MB)
            max_batch_size: Maximum batch size
            memory_reserve_ratio: Ratio of memory to keep reserved (0-1)

        Returns:
            Suggested batch size
        """
        mem_info = self.get_memory_info()

        # Available memory for batching (after reserve)
        available = mem_info.available * (1 - memory_reserve_ratio)

        # Calculate batch size
        if item_memory_mb == 0:
            return max_batch_size

        suggested = int(available / item_memory_mb)

        # Clamp to max batch size
        suggested = min(suggested, max_batch_size)

        # Ensure at least 1
        suggested = max(suggested, 1)

        return suggested

    def get_allocation_summary(self) -> Dict[str, any]:
        """Get summary of memory allocations."""
        mem_info = self.get_memory_info()

        with self._allocation_lock:
            return {
                'system_memory': {
                    'total_mb': mem_info.total,
                    'available_mb': mem_info.available,
                    'used_mb': mem_info.used,
                    'percent': mem_info.percent,
                },
                'allocations': {
                    'total_allocated_mb': self.get_total_allocated(),
                    'task_count': len(self._allocations),
                    'tasks': self._allocations.copy(),
                },
                'available_for_allocation_mb': self.get_available_memory(),
            }


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


# Export
__all__ = [
    'MemoryManager',
    'MemoryInfo',
    'get_memory_manager',
]
