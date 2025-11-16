"""
Pre-defined workflows.

Provides ready-to-use workflows for common tasks.
"""

from .screening import (
    create_screening_workflow,
    create_simple_screening_workflow,
)

from .evaluation import (
    create_evaluation_workflow,
    create_full_pipeline,
)


__all__ = [
    # Screening
    'create_screening_workflow',
    'create_simple_screening_workflow',
    # Evaluation
    'create_evaluation_workflow',
    'create_full_pipeline',
]
