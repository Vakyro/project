"""
GPU resource management.

Provides intelligent GPU allocation and monitoring for tasks.
"""

from typing import List, Optional, Dict, Set
from dataclasses import dataclass
import threading
import subprocess
import re
import logging


logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    id: int
    name: str
    memory_total: int  # MB
    memory_used: int   # MB
    memory_free: int   # MB
    utilization: float  # 0-100%
    temperature: Optional[int] = None  # Celsius

    @property
    def memory_available_ratio(self) -> float:
        """Ratio of available memory (0-1)."""
        if self.memory_total == 0:
            return 0.0
        return self.memory_free / self.memory_total

    @property
    def is_available(self) -> bool:
        """Check if GPU is available for use."""
        return self.memory_available_ratio > 0.1  # At least 10% free


class GPUManager:
    """
    Manager for GPU resource allocation.

    Provides GPU discovery, allocation, and monitoring.
    Thread-safe singleton pattern.
    """

    _instance: Optional['GPUManager'] = None
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
        """Initialize GPU manager."""
        if self._initialized:
            return

        self.logger = logging.getLogger(f"{__name__}.GPUManager")

        # GPU state
        self._gpu_info: Dict[int, GPUInfo] = {}
        self._allocated_gpus: Dict[str, Set[int]] = {}  # task_id -> gpu_ids

        # Thread safety
        self._allocation_lock = threading.RLock()

        # Initialize
        self._discover_gpus()
        self._initialized = True

    def _discover_gpus(self):
        """Discover available GPUs."""
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                self.logger.info(f"Discovered {num_gpus} CUDA GPUs")

                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    mem_total = props.total_memory // (1024 * 1024)  # Convert to MB

                    # Get current memory usage
                    try:
                        mem_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)
                    except:
                        mem_allocated = 0

                    self._gpu_info[i] = GPUInfo(
                        id=i,
                        name=props.name,
                        memory_total=mem_total,
                        memory_used=mem_allocated,
                        memory_free=mem_total - mem_allocated,
                        utilization=0.0
                    )

            else:
                self.logger.warning("No CUDA GPUs available")

        except ImportError:
            self.logger.warning("PyTorch not available - cannot detect GPUs")
        except Exception as e:
            self.logger.error(f"Error discovering GPUs: {str(e)}")

    def refresh_gpu_info(self):
        """Refresh GPU information."""
        try:
            import torch
            if not torch.cuda.is_available():
                return

            for i in range(torch.cuda.device_count()):
                if i in self._gpu_info:
                    # Update memory usage
                    mem_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)
                    mem_cached = torch.cuda.memory_reserved(i) // (1024 * 1024)
                    mem_total = self._gpu_info[i].memory_total

                    self._gpu_info[i].memory_used = mem_allocated + mem_cached
                    self._gpu_info[i].memory_free = mem_total - self._gpu_info[i].memory_used

                    # Try to get utilization via nvidia-smi
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=utilization.gpu',
                             '--format=csv,noheader,nounits', f'--id={i}'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            util = float(result.stdout.strip())
                            self._gpu_info[i].utilization = util
                    except:
                        pass

        except Exception as e:
            self.logger.warning(f"Error refreshing GPU info: {str(e)}")

    def get_available_gpus(self, min_memory_mb: Optional[int] = None) -> List[int]:
        """
        Get list of available GPU IDs.

        Args:
            min_memory_mb: Minimum free memory required (MB)

        Returns:
            List of GPU IDs
        """
        self.refresh_gpu_info()

        available = []
        for gpu_id, info in self._gpu_info.items():
            if not info.is_available:
                continue

            if min_memory_mb and info.memory_free < min_memory_mb:
                continue

            available.append(gpu_id)

        # Sort by available memory (descending)
        available.sort(
            key=lambda gpu_id: self._gpu_info[gpu_id].memory_free,
            reverse=True
        )

        return available

    def allocate_gpus(
        self,
        task_id: str,
        num_gpus: int = 1,
        min_memory_mb: Optional[int] = None
    ) -> List[int]:
        """
        Allocate GPUs for a task.

        Args:
            task_id: Task ID
            num_gpus: Number of GPUs to allocate
            min_memory_mb: Minimum free memory per GPU (MB)

        Returns:
            List of allocated GPU IDs

        Raises:
            RuntimeError: If not enough GPUs available
        """
        with self._allocation_lock:
            # Check if already allocated
            if task_id in self._allocated_gpus:
                self.logger.warning(f"Task {task_id} already has GPUs allocated")
                return list(self._allocated_gpus[task_id])

            # Get available GPUs
            available = self.get_available_gpus(min_memory_mb)

            # Filter out already allocated GPUs
            already_allocated = set()
            for allocated_set in self._allocated_gpus.values():
                already_allocated.update(allocated_set)

            available = [gpu for gpu in available if gpu not in already_allocated]

            if len(available) < num_gpus:
                raise RuntimeError(
                    f"Not enough GPUs available. Requested: {num_gpus}, "
                    f"Available: {len(available)}"
                )

            # Allocate GPUs
            allocated = available[:num_gpus]
            self._allocated_gpus[task_id] = set(allocated)

            self.logger.info(
                f"Allocated GPUs {allocated} to task {task_id}"
            )

            return allocated

    def release_gpus(self, task_id: str):
        """
        Release GPUs allocated to a task.

        Args:
            task_id: Task ID
        """
        with self._allocation_lock:
            if task_id in self._allocated_gpus:
                gpu_ids = self._allocated_gpus[task_id]
                self.logger.info(
                    f"Released GPUs {list(gpu_ids)} from task {task_id}"
                )
                del self._allocated_gpus[task_id]

    def get_gpu_info(self, gpu_id: int) -> Optional[GPUInfo]:
        """
        Get information about a specific GPU.

        Args:
            gpu_id: GPU ID

        Returns:
            GPU information or None
        """
        self.refresh_gpu_info()
        return self._gpu_info.get(gpu_id)

    def get_all_gpu_info(self) -> Dict[int, GPUInfo]:
        """Get information about all GPUs."""
        self.refresh_gpu_info()
        return self._gpu_info.copy()

    def get_gpu_count(self) -> int:
        """Get total number of GPUs."""
        return len(self._gpu_info)

    def get_allocated_gpus(self, task_id: str) -> List[int]:
        """
        Get GPUs allocated to a task.

        Args:
            task_id: Task ID

        Returns:
            List of GPU IDs
        """
        with self._allocation_lock:
            return list(self._allocated_gpus.get(task_id, set()))

    def is_gpu_available(self) -> bool:
        """Check if any GPU is available."""
        return len(self._gpu_info) > 0

    def get_best_gpu(self, min_memory_mb: Optional[int] = None) -> Optional[int]:
        """
        Get the best available GPU (most free memory).

        Args:
            min_memory_mb: Minimum free memory required (MB)

        Returns:
            GPU ID or None
        """
        available = self.get_available_gpus(min_memory_mb)
        return available[0] if available else None

    def clear_cache(self, gpu_id: Optional[int] = None):
        """
        Clear GPU cache.

        Args:
            gpu_id: GPU ID (None for all GPUs)
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return

            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                self.logger.info(f"Cleared cache for GPU {gpu_id}")
            else:
                torch.cuda.empty_cache()
                self.logger.info("Cleared cache for all GPUs")

            # Refresh info
            self.refresh_gpu_info()

        except Exception as e:
            self.logger.warning(f"Error clearing GPU cache: {str(e)}")

    def get_allocation_summary(self) -> Dict[str, any]:
        """Get summary of GPU allocations."""
        with self._allocation_lock:
            return {
                'total_gpus': len(self._gpu_info),
                'allocated_tasks': len(self._allocated_gpus),
                'allocations': {
                    task_id: list(gpu_ids)
                    for task_id, gpu_ids in self._allocated_gpus.items()
                },
                'gpu_info': {
                    gpu_id: {
                        'name': info.name,
                        'memory_total': info.memory_total,
                        'memory_free': info.memory_free,
                        'utilization': info.utilization,
                    }
                    for gpu_id, info in self._gpu_info.items()
                }
            }


# Global GPU manager instance
_global_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _global_gpu_manager
    if _global_gpu_manager is None:
        _global_gpu_manager = GPUManager()
    return _global_gpu_manager


# Export
__all__ = [
    'GPUManager',
    'GPUInfo',
    'get_gpu_manager',
]
