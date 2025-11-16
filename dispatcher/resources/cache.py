"""
Cache management for dispatcher.

Provides intelligent caching for embeddings, results, and intermediate data.
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import pickle
import hashlib
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the cache."""

    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl: Optional[int] = None  # seconds

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False

        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class LRUCache:
    """
    Least Recently Used (LRU) cache.

    Thread-safe in-memory cache with size limits and TTL support.
    """

    def __init__(
        self,
        max_size_mb: int = 1024,
        max_entries: int = 10000,
        default_ttl: Optional[int] = None
    ):
        """
        Initialize LRU cache.

        Args:
            max_size_mb: Maximum cache size (MB)
            max_entries: Maximum number of entries
            default_ttl: Default TTL for entries (seconds)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl

        self._cache: Dict[str, CacheEntry] = {}
        self._current_size_bytes = 0
        self._lock = threading.RLock()

        self.logger = logging.getLogger(f"{__name__}.LRUCache")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            # Check expiration
            if entry.is_expired:
                self.delete(key)
                return None

            # Update access info
            entry.accessed_at = datetime.now()
            entry.access_count += 1

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (seconds, optional)
        """
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0

            # Check if we need to evict
            while (
                (len(self._cache) >= self.max_entries or
                 self._current_size_bytes + size_bytes > self.max_size_bytes) and
                len(self._cache) > 0
            ):
                self._evict_lru()

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size_bytes -= old_entry.size_bytes

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=0,
                ttl=ttl or self.default_ttl
            )

            self._cache[key] = entry
            self._current_size_bytes += size_bytes

    def delete(self, key: str):
        """
        Delete entry from cache.

        Args:
            key: Cache key
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                del self._cache[key]

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].accessed_at
        )

        self.logger.debug(f"Evicting LRU entry: {lru_key}")
        self.delete(lru_key)

    def cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired:
                self.delete(key)

            if expired:
                self.logger.info(f"Cleaned up {len(expired)} expired entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'entries': len(self._cache),
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_entries': self.max_entries,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': len(self._cache) / max(self.max_entries, 1),
            }


class DiskCache:
    """
    Disk-based cache for large objects.

    Persists cache entries to disk for durability and large storage.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_mb: int = 10240,  # 10 GB default
        default_ttl: Optional[int] = None
    ):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size (MB)
            default_ttl: Default TTL (seconds)
        """
        self.cache_dir = cache_dir or (
            Path.home() / '.clipzyme' / 'dispatcher' / 'cache'
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl

        self._metadata: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

        self.logger = logging.getLogger(f"{__name__}.DiskCache")

        # Load metadata
        self._load_metadata()

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash key for filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _get_metadata_path(self) -> Path:
        """Get metadata file path."""
        return self.cache_dir / 'metadata.pkl'

    def _load_metadata(self):
        """Load cache metadata."""
        metadata_path = self._get_metadata_path()

        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    self._metadata = pickle.load(f)
                self.logger.info(
                    f"Loaded cache metadata: {len(self._metadata)} entries"
                )
            except Exception as e:
                self.logger.warning(f"Error loading cache metadata: {str(e)}")
                self._metadata = {}

    def _save_metadata(self):
        """Save cache metadata."""
        metadata_path = self._get_metadata_path()

        try:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self._metadata, f)
        except Exception as e:
            self.logger.warning(f"Error saving cache metadata: {str(e)}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._metadata.get(key)

            if entry is None:
                return None

            # Check expiration
            if entry.is_expired:
                self.delete(key)
                return None

            # Load from disk
            cache_path = self._get_cache_path(key)

            if not cache_path.exists():
                self.logger.warning(f"Cache file missing: {key}")
                self.delete(key)
                return None

            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)

                # Update access info
                entry.accessed_at = datetime.now()
                entry.access_count += 1
                self._save_metadata()

                return value

            except Exception as e:
                self.logger.error(f"Error loading cache entry {key}: {str(e)}")
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        with self._lock:
            cache_path = self._get_cache_path(key)

            # Save to disk
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)

                size_bytes = cache_path.stat().st_size

                # Create metadata entry
                entry = CacheEntry(
                    key=key,
                    value=None,  # Don't store in metadata
                    size_bytes=size_bytes,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    access_count=0,
                    ttl=ttl or self.default_ttl
                )

                self._metadata[key] = entry
                self._save_metadata()

                # Check size limits
                self._enforce_size_limit()

            except Exception as e:
                self.logger.error(f"Error saving cache entry {key}: {str(e)}")

    def delete(self, key: str):
        """Delete entry from cache."""
        with self._lock:
            if key in self._metadata:
                del self._metadata[key]
                self._save_metadata()

            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Delete all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.name != 'metadata.pkl':
                    cache_file.unlink()

            self._metadata.clear()
            self._save_metadata()

    def _enforce_size_limit(self):
        """Enforce cache size limit by removing old entries."""
        # Calculate current size
        current_size = sum(entry.size_bytes for entry in self._metadata.values())

        if current_size <= self.max_size_bytes:
            return

        # Sort by access time (oldest first)
        sorted_keys = sorted(
            self._metadata.keys(),
            key=lambda k: self._metadata[k].accessed_at
        )

        # Remove oldest until under limit
        for key in sorted_keys:
            if current_size <= self.max_size_bytes:
                break

            entry = self._metadata[key]
            current_size -= entry.size_bytes
            self.delete(key)
            self.logger.debug(f"Evicted cache entry: {key}")

    def cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired = [
                key for key, entry in self._metadata.items()
                if entry.is_expired
            ]

            for key in expired:
                self.delete(key)

            if expired:
                self.logger.info(f"Cleaned up {len(expired)} expired entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self._metadata.values())

            return {
                'entries': len(self._metadata),
                'size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': total_size / max(self.max_size_bytes, 1),
                'cache_dir': str(self.cache_dir),
            }


class TwoLevelCache:
    """
    Two-level cache (memory + disk).

    Combines LRU cache (fast, small) with disk cache (slow, large).
    """

    def __init__(
        self,
        memory_size_mb: int = 512,
        disk_size_mb: int = 10240,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize two-level cache.

        Args:
            memory_size_mb: Memory cache size (MB)
            disk_size_mb: Disk cache size (MB)
            cache_dir: Directory for disk cache
        """
        self.l1_cache = LRUCache(max_size_mb=memory_size_mb)
        self.l2_cache = DiskCache(cache_dir=cache_dir, max_size_mb=disk_size_mb)

        self.logger = logging.getLogger(f"{__name__}.TwoLevelCache")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 then L2)."""
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(key, value)
            return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in both caches."""
        self.l1_cache.set(key, value, ttl)
        self.l2_cache.set(key, value, ttl)

    def delete(self, key: str):
        """Delete from both caches."""
        self.l1_cache.delete(key)
        self.l2_cache.delete(key)

    def clear(self):
        """Clear both caches."""
        self.l1_cache.clear()
        self.l2_cache.clear()

    def cleanup_expired(self):
        """Clean up expired entries in both caches."""
        self.l1_cache.cleanup_expired()
        self.l2_cache.cleanup_expired()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for both caches."""
        return {
            'l1_memory': self.l1_cache.get_stats(),
            'l2_disk': self.l2_cache.get_stats(),
        }


# Export
__all__ = [
    'LRUCache',
    'DiskCache',
    'TwoLevelCache',
    'CacheEntry',
]
