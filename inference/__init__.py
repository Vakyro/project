"""
Inference API for CLIPZyme.

Provides simple APIs for model inference and prediction.
"""

from .predictor import (
    CLIPZymePredictor,
    PredictorConfig,
    ScreeningResult,
)

from .batch import (
    BatchPredictor,
    batch_screen_reactions,
)


__all__ = [
    # Predictor
    'CLIPZymePredictor',
    'PredictorConfig',
    'ScreeningResult',
    # Batch
    'BatchPredictor',
    'batch_screen_reactions',
]
