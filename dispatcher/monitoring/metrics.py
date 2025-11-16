"""
Metrics collection and tracking.

Provides metrics collection for tasks, workflows, and system resources.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading
from collections import defaultdict


@dataclass
class Metric:
    """A single metric measurement."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


class MetricsCollector:
    """
    Collector for system and task metrics.

    Thread-safe metrics aggregation and storage.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.RLock()

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: Optional[str] = None):
        """
        Record a metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Metric tags
            unit: Measurement unit
        """
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                tags=tags or {},
                unit=unit
            )
            self._metrics[name].append(metric)

    def increment(self, name: str, value: int = 1):
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """Get gauge value."""
        return self._gauges.get(name, 0.0)

    def get_metrics(self, name: str) -> List[Metric]:
        """Get all metrics for a name."""
        with self._lock:
            return self._metrics.get(name, []).copy()

    def get_latest(self, name: str) -> Optional[Metric]:
        """Get latest metric value."""
        metrics = self.get_metrics(name)
        return metrics[-1] if metrics else None

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        metrics = self.get_metrics(name)
        if not metrics:
            return {}

        values = [m.value for m in metrics]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'sum': sum(values),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all metrics statistics."""
        with self._lock:
            return {
                'metrics': {
                    name: self.get_stats(name)
                    for name in self._metrics.keys()
                },
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
            }

    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()


# Global metrics collector
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


# Export
__all__ = [
    'Metric',
    'MetricsCollector',
    'get_metrics_collector',
]
