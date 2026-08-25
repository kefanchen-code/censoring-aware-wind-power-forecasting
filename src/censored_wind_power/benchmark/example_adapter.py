"""Minimal external-adapter example; import it through ``model_modules``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .core import BenchmarkContext, FitResult, Forecast, ModelAdapter, register_model
from .data import FitData, PredictionData


@register_model
class ResidualPersistenceExample(ModelAdapter):
    """Persistence plus training-only empirical residual quantiles.

    This small baseline demonstrates the complete extension API without
    coupling the benchmark runner to a particular machine-learning library.
    """

    name = "example_residual_persistence"
    version = "1"
    stochastic = False

    def fit(
        self,
        data: FitData,
        context: BenchmarkContext,
        seed: Optional[int],
    ) -> FitResult:
        last_observed = data.train.features[:, 0, -1]
        residual = data.train.observed - last_observed
        residual_quantiles = np.quantile(residual, context.quantile_levels)
        return FitResult(
            estimator=residual_quantiles.astype(np.float32),
            metadata={"source": "training-window persistence residuals only"},
        )

    def predict(
        self,
        fitted: FitResult,
        data: PredictionData,
        context: BenchmarkContext,
    ) -> Forecast:
        quantiles = data.persistence[:, None] + fitted.estimator[None, :]
        quantiles = np.clip(
            quantiles,
            float(context.config["forecast"]["support_min_pu"]),
            float(context.config["forecast"]["support_max_pu"]),
        ).astype(np.float32)
        return Forecast(quantiles=quantiles)

    def save_fit(self, fitted: FitResult, output_dir: Path) -> Dict[str, str]:
        path = output_dir / "fitted_parameters.npz"
        temporary = output_dir / "fitted_parameters.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, residual_quantiles=fitted.estimator)
        temporary.replace(path)
        return {"fitted_parameters": path.name}
