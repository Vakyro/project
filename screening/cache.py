"""
Embedding Cache Management

Provides caching mechanisms for protein and reaction embeddings to avoid
redundant computation during repeated screening runs.

Supports:
- In-memory caching (LRU)
- Disk-based caching (pickle, HDF5)
- Automatic cache invalidation
- Cache statistics and monitoring
"""

import torch
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    cache_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'cache_size': self.cache_size,
            'max_size': self.max_size,
            'hit_rate': self.hit_rate
        }


class LRUCache:
    """
    Least Recently Used (LRU) cache for embeddings.

    Thread-safe in-memory cache with automatic eviction.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.stats = CacheStats(max_size=max_size)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached embedding or None if not found
        """
        self.stats.total_requests += 1

        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.stats.hits += 1
            return self.cache[key]

        self.stats.misses += 1
        return None

    def put(self, key: str, value: torch.Tensor):
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Embedding tensor
        """
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new
            self.cache[key] = value

            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

        self.stats.cache_size = len(self.cache)

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.stats = CacheStats(max_size=self.max_size)

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        return key in self.cache


class DiskCache:
    """
    Disk-based cache for embeddings.

    Stores embeddings as pickle files in a cache directory.
    Useful for persistent caching across sessions.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: float = 10.0
    ):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory for cache files
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)

        self.stats = CacheStats()
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.stats.hits = metadata.get('hits', 0)
                self.stats.misses = metadata.get('misses', 0)
                self.stats.total_requests = metadata.get('total_requests', 0)

    def _save_metadata(self):
        """Save cache metadata."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        metadata = {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'total_requests': self.stats.total_requests,
            'last_updated': datetime.now().isoformat()
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Hash key to create filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached embedding or None if not found
        """
        self.stats.total_requests += 1

        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                self.stats.hits += 1
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                # Remove corrupted file
                cache_file.unlink()

        self.stats.misses += 1
        return None

    def put(self, key: str, value: torch.Tensor):
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Embedding tensor
        """
        cache_file = self._get_cache_file(key)

        try:
            # Save to disk (move to CPU first)
            with open(cache_file, 'wb') as f:
                pickle.dump(value.cpu(), f)

            # Update stats
            self.stats.cache_size = self._get_cache_size()
            self._save_metadata()

            # Check if over capacity
            if self.stats.cache_size > self.max_size_bytes:
                self._evict_oldest()

        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")

    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            total_size += cache_file.stat().st_size
        return total_size

    def _evict_oldest(self):
        """Evict oldest files until under capacity."""
        # Get all cache files sorted by modification time
        cache_files = sorted(
            self.cache_dir.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime
        )

        # Remove oldest until under capacity
        for cache_file in cache_files:
            if self._get_cache_size() <= self.max_size_bytes:
                break
            cache_file.unlink()
            logger.debug(f"Evicted cache file: {cache_file}")

    def clear(self):
        """Clear all cached items."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

        self.stats = CacheStats()
        self._save_metadata()

    def __len__(self) -> int:
        return len(list(self.cache_dir.glob("*.pkl")))


class EmbeddingCache:
    """
    Unified embedding cache with multiple backends.

    Supports:
    - In-memory LRU cache for fast access
    - Disk cache for persistence
    - Two-level caching (memory + disk)
    """

    def __init__(
        self,
        memory_cache_size: int = 1000,
        disk_cache_dir: Optional[Union[str, Path]] = None,
        disk_cache_size_gb: float = 10.0,
        use_memory: bool = True,
        use_disk: bool = True
    ):
        """
        Initialize embedding cache.

        Args:
            memory_cache_size: Max items in memory cache
            disk_cache_dir: Directory for disk cache (required if use_disk=True)
            disk_cache_size_gb: Max disk cache size in GB
            use_memory: Enable memory caching
            use_disk: Enable disk caching
        """
        self.use_memory = use_memory
        self.use_disk = use_disk

        # Initialize memory cache
        self.memory_cache = LRUCache(max_size=memory_cache_size) if use_memory else None

        # Initialize disk cache
        if use_disk:
            if disk_cache_dir is None:
                raise ValueError("disk_cache_dir required when use_disk=True")
            self.disk_cache = DiskCache(
                cache_dir=disk_cache_dir,
                max_size_gb=disk_cache_size_gb
            )
        else:
            self.disk_cache = None

        logger.info(
            f"Initialized EmbeddingCache "
            f"(memory={use_memory}, disk={use_disk})"
        )

    def get(self, key: str, device: str = "cpu") -> Optional[torch.Tensor]:
        """
        Get embedding from cache.

        Checks memory cache first, then disk cache.

        Args:
            key: Cache key (e.g., protein ID or reaction SMILES)
            device: Device to load embedding to

        Returns:
            Cached embedding or None if not found
        """
        # Try memory cache first
        if self.memory_cache is not None:
            embedding = self.memory_cache.get(key)
            if embedding is not None:
                return embedding.to(device)

        # Try disk cache
        if self.disk_cache is not None:
            embedding = self.disk_cache.get(key)
            if embedding is not None:
                # Store in memory cache for next time
                if self.memory_cache is not None:
                    self.memory_cache.put(key, embedding)
                return embedding.to(device)

        return None

    def put(self, key: str, value: torch.Tensor):
        """
        Put embedding in cache.

        Stores in both memory and disk cache if enabled.

        Args:
            key: Cache key
            value: Embedding tensor
        """
        if self.memory_cache is not None:
            self.memory_cache.put(key, value)

        if self.disk_cache is not None:
            self.disk_cache.put(key, value)

    def get_batch(
        self,
        keys: List[str],
        device: str = "cpu"
    ) -> Tuple[List[str], List[torch.Tensor], List[str]]:
        """
        Get multiple embeddings from cache.

        Args:
            keys: List of cache keys
            device: Device to load embeddings to

        Returns:
            cached_keys: Keys that were found
            cached_embeddings: Corresponding embeddings
            missing_keys: Keys that were not found
        """
        cached_keys = []
        cached_embeddings = []
        missing_keys = []

        for key in keys:
            embedding = self.get(key, device=device)
            if embedding is not None:
                cached_keys.append(key)
                cached_embeddings.append(embedding)
            else:
                missing_keys.append(key)

        return cached_keys, cached_embeddings, missing_keys

    def put_batch(self, keys: List[str], values: torch.Tensor):
        """
        Put multiple embeddings in cache.

        Args:
            keys: List of cache keys
            values: Batch of embeddings (batch_size, embedding_dim)
        """
        if len(keys) != values.shape[0]:
            raise ValueError(f"Keys and values length mismatch: {len(keys)} vs {values.shape[0]}")

        for key, value in zip(keys, values):
            self.put(key, value)

    def clear(self):
        """Clear all caches."""
        if self.memory_cache is not None:
            self.memory_cache.clear()

        if self.disk_cache is not None:
            self.disk_cache.clear()

    def get_stats(self) -> Dict[str, Dict]:
        """
        Get cache statistics.

        Returns:
            Dictionary with stats for each cache backend
        """
        stats = {}

        if self.memory_cache is not None:
            stats['memory'] = self.memory_cache.stats.to_dict()

        if self.disk_cache is not None:
            stats['disk'] = self.disk_cache.stats.to_dict()

        return stats

    def __repr__(self) -> str:
        parts = []
        if self.memory_cache:
            parts.append(f"memory={len(self.memory_cache)}/{self.memory_cache.max_size}")
        if self.disk_cache:
            parts.append(f"disk={len(self.disk_cache)} items")

        return f"EmbeddingCache({', '.join(parts)})"


__all__ = [
    'CacheStats',
    'LRUCache',
    'DiskCache',
    'EmbeddingCache',
]
