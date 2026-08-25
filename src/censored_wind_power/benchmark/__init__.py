"""Extensible benchmark framework for potential available power forecasting."""

from .core import (
    BenchmarkContext,
    BenchmarkError,
    ConfigurationError,
    FitResult,
    Forecast,
    ModelAdapter,
    create_model,
    list_models,
    register_model,
)
from .data import FitData, ObservableSplit, PredictionData

__all__ = [
    "BenchmarkContext",
    "BenchmarkError",
    "ConfigurationError",
    "FitData",
    "FitResult",
    "Forecast",
    "ModelAdapter",
    "ObservableSplit",
    "PredictionData",
    "create_model",
    "list_models",
    "register_model",
]
