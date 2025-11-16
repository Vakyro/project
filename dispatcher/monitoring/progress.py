"""
Progress tracking for tasks and workflows.

Provides real-time progress updates and ETA calculation.
"""

from typing import Optional, Dict, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading


@dataclass
class ProgressInfo:
    """Progress information."""

    current: int
    total: int
    percentage: float
    message: str
    eta: Optional[timedelta] = None
    rate: Optional[float] = None  # items per second

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'current': self.current,
            'total': self.total,
            'percentage': self.percentage,
            'message': self.message,
            'eta_seconds': self.eta.total_seconds() if self.eta else None,
            'rate': self.rate,
        }


class ProgressTracker:
    """
    Tracks progress for a task or workflow.

    Provides progress updates, ETA calculation, and callbacks.
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        callback: Optional[Callable[[ProgressInfo], None]] = None
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Progress description
            callback: Progress callback function
        """
        self.total = total
        self.description = description
        self.callback = callback

        self.current = 0
        self.start_time = datetime.now()
        self.last_update_time = self.start_time

        self._lock = threading.Lock()

    def update(self, n: int = 1, message: Optional[str] = None):
        """
        Update progress.

        Args:
            n: Number of items completed
            message: Progress message
        """
        with self._lock:
            self.current = min(self.current + n, self.total)
            self.last_update_time = datetime.now()

            if self.callback:
                info = self.get_info(message)
                try:
                    self.callback(info)
                except:
                    pass

    def set_progress(self, current: int, message: Optional[str] = None):
        """
        Set absolute progress.

        Args:
            current: Current progress
            message: Progress message
        """
        with self._lock:
            self.current = min(current, self.total)
            self.last_update_time = datetime.now()

            if self.callback:
                info = self.get_info(message)
                try:
                    self.callback(info)
                except:
                    pass

    def get_info(self, message: Optional[str] = None) -> ProgressInfo:
        """Get current progress info."""
        with self._lock:
            percentage = (self.current / self.total * 100) if self.total > 0 else 0

            # Calculate ETA
            eta = None
            rate = None
            if self.current > 0:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                rate = self.current / elapsed if elapsed > 0 else 0

                if rate > 0:
                    remaining = self.total - self.current
                    eta_seconds = remaining / rate
                    eta = timedelta(seconds=eta_seconds)

            return ProgressInfo(
                current=self.current,
                total=self.total,
                percentage=percentage,
                message=message or self.description,
                eta=eta,
                rate=rate
            )

    @property
    def is_complete(self) -> bool:
        """Check if progress is complete."""
        return self.current >= self.total


# Export
__all__ = [
    'ProgressInfo',
    'ProgressTracker',
]
