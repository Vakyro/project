"""
Configuration management.

Provides configuration resolution and validation.
"""

from .resolver import ConfigResolver
from .validator import ConfigValidator, ValidationError


__all__ = [
    'ConfigResolver',
    'ConfigValidator',
    'ValidationError',
]
