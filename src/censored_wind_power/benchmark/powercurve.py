"""Power-curve truth-reconstruction baseline on the shared TCN backbone.

This adapter implements the classical available-power route: fit an empirical
power curve on *uncensored* training rows only, use it to replace the labels
of censored windows (reconstructed from the model-visible target-time wind
reading), and then train the ordinary point-label likelihood on the relabeled
data. It brings the reconstruction school into the same arena as the
censored-likelihood models under a known-truth protocol, where the quality of
the reconstructed channel can finally be scored directly.

The module intentionally lives outside ``models.py``: model identity is bound
to the adapter source file, so adding the baseline in a separate module keeps
every previously frozen artifact reusable. The training protocol is the shared
TCN driver itself; the only varied factor is the label channel.

Information-set statement: every array consumed here is a measurement
(frame-level wind/power/censoring flag from the training split, and the
target-time wind reading). The latent truth never enters the reconstruction
view, and the runner dispatches that view only to adapters declaring
``requires_reconstruction_inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .core import (
    BenchmarkContext,
    BenchmarkError,
    FitResult,
    register_model,
)
from .data import FitData, ObservableSplit, ReconstructionFitData
from .models import _TorchTCNAdapter


@dataclass(frozen=True)
class PowerCurve:
    """Binned empirical power curve with flat extrapolation."""

    bin_width_ms: float
    bin_left_edges: np.ndarray
    bin_centers: np.ndarray
    bin_power_mean: np.ndarray
    bin_counts: np.ndarray
    knot_wind_ms: np.ndarray
    knot_power_pu: np.ndarray

    def evaluate(self, wind_ms: np.ndarray) -> np.ndarray:
        """Interpolate the curve; np.interp clamps outside the knot range."""

        values = np.interp(
            np.asarray(wind_ms, dtype=np.float64),
            self.knot_wind_ms,
            self.knot_power_pu,
        )
        return np.atleast_1d(values)

    def to_payload(self) -> Dict[str, Any]:
        """JSON-safe description persisted in training.json."""

        return {
            "bin_width_ms": float(self.bin_width_ms),
            "n_bins_total": int(len(self.bin_centers)),
            "n_bins_used_as_knots": int(len(self.knot_wind_ms)),
            "knot_wind_range_ms": [
                float(self.knot_wind_ms[0]),
                float(self.knot_wind_ms[-1]),
            ],
            "knot_power_range_pu": [
                float(self.knot_power_pu[0]),
                float(self.knot_power_pu[-1]),
            ],
        }


def fit_power_curve(
    wind_ms: np.ndarray,
    power_pu: np.ndarray,
    censored: np.ndarray,
    bin_width_ms: float,
    min_bin_count: int,
) -> PowerCurve:
    """Fit the empirical power curve on uncensored rows only.

    Bins are anchored at zero so the curve is fully determined by the
    training data; bins below ``min_bin_count`` are dropped from the knot
    set and the interpolation bridges across them.
    """

    wind_ms = np.asarray(wind_ms, dtype=np.float64)
    power_pu = np.asarray(power_pu, dtype=np.float64)
    clean = np.asarray(censored, dtype=np.float64) < 0.5
    if not np.any(clean):
        raise BenchmarkError(
            "power-curve reconstruction has no uncensored training rows"
        )
    if bin_width_ms <= 0:
        raise BenchmarkError("bin_width_ms must be positive")
    wind_clean = wind_ms[clean]
    power_clean = power_pu[clean]
    n_bins = max(2, int(np.ceil(float(wind_clean.max()) / bin_width_ms)) + 1)
    edges = bin_width_ms * np.arange(n_bins + 1, dtype=np.float64)
    index = np.clip(
        np.floor(wind_clean / bin_width_ms).astype(np.int64), 0, n_bins - 1
    )
    counts = np.bincount(index, minlength=n_bins).astype(np.int64)
    power_sum = np.bincount(index, weights=power_clean, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, power_sum / np.maximum(counts, 1), np.nan)
    usable = counts >= int(min_bin_count)
    if not np.any(usable):
        raise BenchmarkError(
            "no power-curve bin reaches min_bin_count=%d clean rows"
            % int(min_bin_count)
        )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return PowerCurve(
        bin_width_ms=float(bin_width_ms),
        bin_left_edges=edges[:-1],
        bin_centers=centers,
        bin_power_mean=means,
        bin_counts=counts,
        knot_wind_ms=centers[usable],
        knot_power_pu=means[usable],
    )


def relabel_split(
    split: ObservableSplit,
    target_wind_ms: np.ndarray,
    curve: PowerCurve,
    support_min: float,
    support_max: float,
) -> "tuple[ObservableSplit, int]":
    """Replace labels of censored windows with reconstructed values.

    Uncensored windows keep their exact observed labels; censored windows get
    ``curve(wind_at_target)`` clipped to the forecast support. Returns the
    relabeled split and the number of replaced windows.
    """

    if len(split) != len(target_wind_ms):
        raise BenchmarkError(
            "target wind length %d does not match split length %d"
            % (len(target_wind_ms), len(split))
        )
    observed = split.observed.astype(np.float32, copy=True)
    mask = split.censored >= 0.5
    replaced = int(mask.sum())
    if replaced:
        reconstructed = np.clip(
            curve.evaluate(target_wind_ms[mask]), support_min, support_max
        ).astype(np.float32)
        observed[mask] = reconstructed
    return (
        ObservableSplit(
            features=split.features,
            observed=observed,
            cap=split.cap,
            censored=split.censored,
        ),
        replaced,
    )


@register_model
class B2ReconAdapter(_TorchTCNAdapter):
    """Point-label TCN trained on power-curve-reconstructed censored labels.

    Single varied factor relative to B4 (PL): censored training labels are
    replaced by the empirical power curve evaluated at the model-visible
    target-time wind reading, exactly the recipe of the reconstruction
    school. Inputs, backbone, and training driver are identical.
    """

    name = "B2_recon"
    mode = "B4"
    version = "1"
    requires_reconstruction_inputs = True

    def _settings(self, context: BenchmarkContext) -> Dict[str, Any]:
        settings = context.config.get("model_settings", {}).get(self.name, {})
        return {
            "bin_width_ms": float(settings.get("bin_width_ms", 0.5)),
            "min_bin_count": int(settings.get("min_bin_count", 30)),
        }

    def fit(
        self,
        data: ReconstructionFitData,
        context: BenchmarkContext,
        seed: Optional[int],
    ) -> FitResult:
        if not isinstance(data, ReconstructionFitData):
            raise BenchmarkError(
                "%s requires the declared reconstruction fit view" % self.name
            )
        settings = self._settings(context)
        support_min = float(context.config["forecast"]["support_min_pu"])
        support_max = float(context.config["forecast"]["support_max_pu"])
        # Power-curve fitting consumes only uncensored training rows, so the
        # contaminated wind readings of censored frames never enter the curve.
        curve = fit_power_curve(
            data.train_wind_ms_rows,
            data.train_observed_rows,
            data.train_censored_rows,
            settings["bin_width_ms"],
            settings["min_bin_count"],
        )
        train_relabeled, train_replaced = relabel_split(
            data.train, data.train_target_wind_ms, curve, support_min, support_max
        )
        validation_relabeled, validation_replaced = relabel_split(
            data.validation,
            data.validation_target_wind_ms,
            curve,
            support_min,
            support_max,
        )
        self._fitted_curve = curve
        self._relabel_audit = {
            "train_censored_windows_replaced": train_replaced,
            "validation_censored_windows_replaced": validation_replaced,
        }
        synthesized = FitData(
            train=train_relabeled,
            validation=validation_relabeled,
            train_observed_rows=data.train_observed_rows,
            feature_names=data.feature_names,
        )
        fitted = super().fit(synthesized, context, seed)
        fitted.metadata.update(
            {
                "label_source": "observable_power_curve_reconstruction",
                "reconstruction": {
                    "recipe": (
                        "empirical binned power curve fitted on uncensored"
                        " training rows; censored window labels replaced by"
                        " curve(model-visible target-time wind) clipped to the"
                        " forecast support"
                    ),
                    "min_bin_count": settings["min_bin_count"],
                    "clean_rows_used": int(
                        (np.asarray(data.train_censored_rows) < 0.5).sum()
                    ),
                    "power_curve": curve.to_payload(),
                    **self._relabel_audit,
                },
            }
        )
        return fitted

    def save_fit(self, fitted: FitResult, output_dir: Path) -> Dict[str, str]:
        """Persist the TCN checkpoint plus the fitted power curve."""

        files = super().save_fit(fitted, output_dir)
        curve: PowerCurve = getattr(self, "_fitted_curve", None)
        if curve is not None:
            path = output_dir / "power_curve.npz"
            temporary = output_dir / "power_curve.npz.tmp"
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    bin_centers=curve.bin_centers,
                    bin_power_mean=curve.bin_power_mean,
                    bin_counts=curve.bin_counts,
                    knot_wind_ms=curve.knot_wind_ms,
                    knot_power_pu=curve.knot_power_pu,
                )
            temporary.replace(path)
            files["power_curve"] = path.name
        return files
