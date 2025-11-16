"""
Configuration resolver for dispatcher.

Handles merging, resolution, and validation of configurations.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import copy


class ConfigResolver:
    """
    Resolves and merges configurations from multiple sources.

    Priority (highest to lowest):
    1. Runtime overrides
    2. User config file
    3. Environment variables
    4. Default config
    """

    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize config resolver.

        Args:
            default_config: Default configuration
        """
        self.default_config = default_config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'scheduler': {
                'max_concurrent_jobs': 2,
                'max_workers_per_job': 4,
                'poll_interval': 1.0,
            },
            'resources': {
                'gpu': {
                    'auto_allocate': True,
                    'min_free_memory_mb': 1000,
                },
                'memory': {
                    'reserve_ratio': 0.2,
                },
                'cache': {
                    'l1_size_mb': 512,
                    'l2_size_mb': 10240,
                },
            },
            'tasks': {
                'checkpoint': {
                    'device': 'cuda',
                },
                'screening': {
                    'mode': 'interactive',
                    'batch_size': 8,
                    'top_k': 100,
                },
                'evaluation': {
                    'metrics': ['bedroc', 'topk_accuracy', 'enrichment_factor'],
                },
            },
            'logging': {
                'level': 'INFO',
                'console': True,
                'file': True,
                'structured': False,
            },
            'workflows': {
                'screening_pipeline': {
                    'continue_on_error': False,
                    'fail_fast': True,
                },
            },
        }

    def load_from_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def merge_configs(
        self,
        *configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge multiple configurations.

        Later configs override earlier ones.

        Args:
            *configs: Configuration dictionaries

        Returns:
            Merged configuration
        """
        result = {}

        for config in configs:
            result = self._deep_merge(result, config)

        return result

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def resolve(
        self,
        config_file: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve final configuration.

        Args:
            config_file: Optional config file path
            overrides: Optional runtime overrides

        Returns:
            Resolved configuration
        """
        # Start with defaults
        config = copy.deepcopy(self.default_config)

        # Merge config file
        if config_file:
            file_config = self.load_from_file(config_file)
            config = self._deep_merge(config, file_config)

        # Merge overrides
        if overrides:
            config = self._deep_merge(config, overrides)

        return config

    def get_nested(self, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Get nested configuration value.

        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., 'scheduler.max_concurrent_jobs')
            default: Default value if not found

        Returns:
            Configuration value
        """
        keys = path.split('.')
        value = config

        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]

        return value

    def set_nested(self, config: Dict[str, Any], path: str, value: Any):
        """
        Set nested configuration value.

        Args:
            config: Configuration dictionary
            path: Dot-separated path
            value: Value to set
        """
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value


# Export
__all__ = [
    'ConfigResolver',
]
