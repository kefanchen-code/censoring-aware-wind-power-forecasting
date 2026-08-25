"""Public model API and registry for the PAP benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import numpy as np


class BenchmarkError(RuntimeError):
    """Base error raised by the benchmark framework."""


class ConfigurationError(ValueError):
    """Raised when a benchmark configuration is invalid."""


class ForecastValidationError(ValueError):
    """Raised when a model emits an invalid forecast artifact."""


@dataclass(frozen=True)
class BenchmarkContext:
    """Immutable protocol information supplied to each model adapter."""

    scenario: str
    protocol_id: str
    output_dir: Path
    device: str
    config: Mapping[str, Any]
    quantile_levels: np.ndarray
    bin_edges: np.ndarray
    bin_centers: np.ndarray


@dataclass
class Forecast:
    """Standard forecast representation returned by every model.

    Quantiles are mandatory because WIS is the cross-model primary score. A
    model may additionally provide a PMF on the protocol grid for exact CRPS.
    """

    quantiles: np.ndarray
    pmf: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, n_samples: int, context: BenchmarkContext) -> None:
        """Validate shape, finiteness, monotonicity, and probability mass."""

        expected_quantiles = (n_samples, len(context.quantile_levels))
        if self.quantiles.shape != expected_quantiles:
            raise ForecastValidationError(
                "quantiles shape %s does not match expected %s"
                % (self.quantiles.shape, expected_quantiles)
            )
        if not np.isfinite(self.quantiles).all():
            raise ForecastValidationError("quantiles contain NaN or infinity")
        if np.any(np.diff(self.quantiles, axis=1) < -1e-7):
            raise ForecastValidationError("quantiles are not monotone by level")
        support_min = float(context.config["forecast"]["support_min_pu"])
        support_max = float(context.config["forecast"]["support_max_pu"])
        if np.any(self.quantiles < support_min - 1e-7) or np.any(
            self.quantiles > support_max + 1e-7
        ):
            raise ForecastValidationError("quantiles lie outside declared support")
        if self.pmf is None:
            return
        expected_pmf = (n_samples, len(context.bin_centers))
        if self.pmf.shape != expected_pmf:
            raise ForecastValidationError(
                "pmf shape %s does not match expected %s"
                % (self.pmf.shape, expected_pmf)
            )
        if not np.isfinite(self.pmf).all() or np.any(self.pmf < -1e-8):
            raise ForecastValidationError("pmf contains invalid probability mass")
        mass = self.pmf.sum(axis=1)
        if not np.allclose(mass, 1.0, atol=1e-5):
            raise ForecastValidationError("pmf rows do not sum to one")
        cumulative = np.cumsum(self.pmf, axis=1)
        implied = np.empty_like(self.quantiles)
        for column, level in enumerate(context.quantile_levels):
            indices = np.sum(cumulative < level, axis=1)
            implied[:, column] = context.bin_centers[
                np.clip(indices, 0, len(context.bin_centers) - 1)
            ]
        if not np.allclose(self.quantiles, implied, atol=1e-6):
            raise ForecastValidationError("quantiles are inconsistent with supplied pmf")


@dataclass
class FitResult:
    """Fitted estimator plus auditable training metadata."""

    estimator: Any
    history: List[Dict[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Interface implemented by every benchmark model.

    Adapters may wrap PyTorch, scikit-learn, statistical, physical, or remote
    models. They receive the same prepared scenario data and must return the
    benchmark's common quantile representation.
    """

    name: str = ""
    version: str = "1"
    stochastic: bool = True
    # Only adapters that declare this flag receive the latent-truth label view
    # (``ScenarioData.oracle_fit_view``); every other adapter is restricted to
    # observable labels by construction.
    requires_latent_truth: bool = False
    # Only adapters that declare this flag receive the reconstruction view
    # (``ScenarioData.reconstruction_fit_view``): frame-level training
    # observables plus target-time standardized wind, all of which remain
    # measurements and never contain the latent truth. The runner refuses any
    # adapter that declares both views.
    requires_reconstruction_inputs: bool = False

    @abstractmethod
    def fit(self, data: Any, context: BenchmarkContext, seed: Optional[int]) -> FitResult:
        """Fit the model using only train and validation splits."""

    @abstractmethod
    def predict(self, fitted: FitResult, data: Any, context: BenchmarkContext) -> Forecast:
        """Predict the test split without accessing test truth."""

    def save_fit(self, fitted: FitResult, output_dir: Path) -> Dict[str, str]:
        """Optionally persist fitted parameters and return named file paths.

        External adapters may override this hook. Forecast artifacts remain
        sufficient for model-independent comparison when a backend does not
        support portable estimator serialization.
        """

        return {}


_MODEL_REGISTRY: Dict[str, Type[ModelAdapter]] = {}


def register_model(
    model_class: Optional[Type[ModelAdapter]] = None,
    *,
    name: Optional[str] = None,
) -> Union[
    Type[ModelAdapter], Callable[[Type[ModelAdapter]], Type[ModelAdapter]]
]:
    """Register a model adapter class.

    The decorator supports ``@register_model`` and
    ``@register_model(name="custom_name")`` forms.
    """

    def decorator(cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
        model_name = name or cls.name
        if not model_name:
            raise ConfigurationError("registered model must define a non-empty name")
        if model_name in _MODEL_REGISTRY and _MODEL_REGISTRY[model_name] is not cls:
            raise ConfigurationError("model %r is already registered" % model_name)
        cls.name = model_name
        _MODEL_REGISTRY[model_name] = cls
        return cls

    if model_class is None:
        return decorator
    return decorator(model_class)


def create_model(name: str) -> ModelAdapter:
    """Instantiate a registered adapter by stable model name."""

    try:
        return _MODEL_REGISTRY[name]()
    except KeyError as error:
        raise ConfigurationError(
            "unknown model %r; registered models: %s"
            % (name, ", ".join(sorted(_MODEL_REGISTRY)))
        ) from error


def list_models() -> List[str]:
    """Return registered model names in deterministic order."""

    return sorted(_MODEL_REGISTRY)
