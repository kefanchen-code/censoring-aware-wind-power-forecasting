"""Data contracts, immutable segment splits, and benchmark windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .core import ConfigurationError
from .protocol import stable_json_hash, validate_contamination


REQUIRED_COLUMNS = (
    "A_true",
    "Y_syn",
    "C_syn",
    "wind",
    "is_censored",
    "policy_active",
    "segment_id",
)


@dataclass(frozen=True)
class WindowSplit:
    """Materialized, model-independent windows for one data split.

    ``target_wind_ms`` and ``history_censor_frac`` are evaluation-only
    stratification fields; ``ObservableSplit`` never exposes them to models.
    """

    features: np.ndarray
    truth: np.ndarray
    observed: np.ndarray
    cap: np.ndarray
    censored: np.ndarray
    segment_id: np.ndarray
    persistence: np.ndarray
    target_row: np.ndarray
    target_wind_ms: np.ndarray
    history_censor_frac: np.ndarray
    # Model-visible sensor reading at the target instant. Unlike the
    # evaluation-only ``target_wind_ms`` (kept clean in contaminated runs for
    # comparable strata), this column always carries exactly what a modeller
    # would read from the sensor, and is exposed only through the declared
    # reconstruction view.
    target_wind_visible_ms: np.ndarray

    def __len__(self) -> int:
        return int(len(self.truth))


@dataclass(frozen=True)
class ScenarioData:
    """Prepared train/validation/test data shared by every model."""

    scenario: str
    train: WindowSplit
    validation: WindowSplit
    test: WindowSplit
    train_observed_rows: np.ndarray
    feature_names: Tuple[str, ...]
    split_summary: Mapping[str, Any]
    data_signature: str
    # Frame-level training observables used only by the declared reconstruction
    # view; they are measurements, never the latent truth.
    train_wind_ms_rows: np.ndarray
    train_censored_rows: np.ndarray

    def fit_view(self) -> "FitData":
        """Return training data with latent truth removed from the interface."""

        return FitData(
            train=ObservableSplit.from_window_split(self.train),
            validation=ObservableSplit.from_window_split(self.validation),
            train_observed_rows=self.train_observed_rows,
            feature_names=self.feature_names,
        )

    def oracle_fit_view(self) -> "FitData":
        """Latent-truth label view reserved for declared oracle adapters.

        Labels are the latent truth and the censoring indicator is cleared,
        so a point-label loss on this view equals full-truth training. The
        runner dispatches it only to adapters with ``requires_latent_truth``.
        """

        return FitData(
            train=ObservableSplit.oracle_from_window_split(self.train),
            validation=ObservableSplit.oracle_from_window_split(self.validation),
            train_observed_rows=self.train_observed_rows,
            feature_names=self.feature_names,
        )

    def reconstruction_fit_view(self) -> "ReconstructionFitData":
        """Observable-only extras reserved for declared reconstruction adapters.

        Beyond the ordinary fit view this exposes (i) frame-level training
        observables for fitting an empirical power curve on uncensored rows
        and (ii) the model-visible target-time wind for every train and
        validation window, so censored labels can be replaced by a
        reconstructed value. Every exposed array is a measurement; the latent
        truth never enters this view. The runner dispatches it only to
        adapters with ``requires_reconstruction_inputs``.
        """

        return ReconstructionFitData(
            train=ObservableSplit.from_window_split(self.train),
            validation=ObservableSplit.from_window_split(self.validation),
            train_observed_rows=self.train_observed_rows,
            feature_names=self.feature_names,
            train_wind_ms_rows=self.train_wind_ms_rows,
            train_censored_rows=self.train_censored_rows,
            train_target_wind_ms=self.train.target_wind_visible_ms,
            validation_target_wind_ms=self.validation.target_wind_visible_ms,
        )

    def prediction_view(self) -> "PredictionData":
        """Return test inputs with targets and cluster labels withheld."""

        return PredictionData(
            features=self.test.features,
            persistence=self.test.persistence,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True)
class ObservableSplit:
    """Training or validation split containing only observable labels."""

    features: np.ndarray
    observed: np.ndarray
    cap: np.ndarray
    censored: np.ndarray

    def __len__(self) -> int:
        return int(len(self.observed))

    @classmethod
    def from_window_split(cls, split: WindowSplit) -> "ObservableSplit":
        """Strip latent truth and evaluation-only identifiers."""

        return cls(
            features=split.features,
            observed=split.observed,
            cap=split.cap,
            censored=split.censored,
        )

    @classmethod
    def oracle_from_window_split(cls, split: WindowSplit) -> "ObservableSplit":
        """Latent-truth labels with the censoring indicator cleared."""

        return cls(
            features=split.features,
            observed=split.truth,
            cap=split.cap,
            censored=np.zeros_like(split.censored),
        )


@dataclass(frozen=True)
class FitData:
    """Leakage-controlled input supplied to ``ModelAdapter.fit``."""

    train: ObservableSplit
    validation: ObservableSplit
    train_observed_rows: np.ndarray
    feature_names: Tuple[str, ...]


@dataclass(frozen=True)
class ReconstructionFitData(FitData):
    """Fit view extended with observables for truth-channel reconstruction.

    ``train_wind_ms_rows`` / ``train_censored_rows`` are frame-level training
    readings for power-curve fitting on uncensored rows only; the
    ``*_target_wind_ms`` arrays hold the model-visible sensor reading at each
    window's target instant so censored labels can be relabeled. None of these
    arrays contain the latent truth.
    """

    train_wind_ms_rows: np.ndarray = None
    train_censored_rows: np.ndarray = None
    train_target_wind_ms: np.ndarray = None
    validation_target_wind_ms: np.ndarray = None


@dataclass(frozen=True)
class PredictionData:
    """Leakage-controlled input supplied to ``ModelAdapter.predict``."""

    features: np.ndarray
    persistence: np.ndarray
    feature_names: Tuple[str, ...]

    def __len__(self) -> int:
        return int(len(self.features))


def validate_semisynthetic_frame(
    frame: pd.DataFrame,
    rated_power_kw: float,
    sampling_interval_minutes: int,
    support_min_pu: float,
    support_max_pu: float,
) -> None:
    """Validate the semisynthetic observation contract before model access."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ConfigurationError("semisynthetic data lack columns: %s" % missing)
    if frame.empty:
        raise ConfigurationError("semisynthetic data are empty")
    numeric = frame[list(REQUIRED_COLUMNS[:4])]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ConfigurationError("semisynthetic numeric fields contain NaN or infinity")
    if frame["segment_id"].isna().any():
        raise ConfigurationError("segment_id contains missing values")
    if rated_power_kw <= 0:
        raise ConfigurationError("rated_power_kw must be positive")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ConfigurationError("semisynthetic data require a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ConfigurationError("timestamps must be strictly increasing and unique")

    segment_values = frame["segment_id"].to_numpy()
    transition_segments = segment_values[
        np.r_[True, segment_values[1:] != segment_values[:-1]]
    ]
    if len(transition_segments) != len(np.unique(segment_values)):
        raise ConfigurationError("each segment_id must occupy one contiguous time block")
    expected_delta = pd.Timedelta(minutes=sampling_interval_minutes)
    deltas = frame.index.to_series().diff().to_numpy()
    same_segment = segment_values[1:] == segment_values[:-1]
    if not np.all(deltas[1:][same_segment] == expected_delta):
        raise ConfigurationError(
            "timestamps within each segment must follow the declared sampling interval"
        )

    truth_pu = frame["A_true"].to_numpy(dtype=float) / rated_power_kw
    if np.any(truth_pu < support_min_pu - 1e-8) or np.any(
        truth_pu > support_max_pu + 1e-8
    ):
        raise ConfigurationError("latent truth lies outside the forecast support")

    expected_observed = np.minimum(
        frame["A_true"].to_numpy(dtype=float), frame["C_syn"].to_numpy(dtype=float)
    )
    if not np.allclose(frame["Y_syn"].to_numpy(dtype=float), expected_observed, atol=1e-5):
        raise ConfigurationError("Y_syn is inconsistent with min(A_true, C_syn)")
    expected_censoring = frame["policy_active"].astype(bool).to_numpy() & (
        frame["A_true"].to_numpy(dtype=float) >= frame["C_syn"].to_numpy(dtype=float)
    )
    if not np.array_equal(frame["is_censored"].astype(bool).to_numpy(), expected_censoring):
        raise ConfigurationError("is_censored violates policy_active and A_true >= C_syn")


def segment_signature(frame: pd.DataFrame) -> str:
    """Hash ordered segment sizes to detect incompatible scenario inputs."""

    sizes = frame.groupby("segment_id", sort=False).size()
    payload = [[int(segment_id), int(size)] for segment_id, size in sizes.items()]
    return stable_json_hash(payload)


def make_segment_assignment(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> Dict[int, str]:
    """Assign complete chronological segments by cumulative row midpoint."""

    sizes = frame.groupby("segment_id", sort=False).size()
    total = float(sizes.sum())
    train_cut = train_fraction * total
    validation_cut = (train_fraction + validation_fraction) * total
    assignment: Dict[int, str] = {}
    cumulative = 0.0
    for raw_segment_id, raw_size in sizes.items():
        segment_id = int(raw_segment_id)
        size = float(raw_size)
        midpoint = cumulative + 0.5 * size
        if midpoint <= train_cut:
            split = "train"
        elif midpoint <= validation_cut:
            split = "validation"
        else:
            split = "test"
        assignment[segment_id] = split
        cumulative += size
    if set(assignment.values()) != {"train", "validation", "test"}:
        raise ConfigurationError("segment assignment produced an empty split")
    return assignment


def apply_segment_assignment(
    frame: pd.DataFrame,
    assignment: Mapping[int, str],
) -> Dict[str, pd.DataFrame]:
    """Apply a frozen assignment and prove that no segment crosses splits."""

    observed_segments = {int(value) for value in frame["segment_id"].unique()}
    assigned_segments = {int(value) for value in assignment}
    if observed_segments != assigned_segments:
        missing = sorted(observed_segments - assigned_segments)
        extra = sorted(assigned_segments - observed_segments)
        raise ConfigurationError(
            "split manifest and data segments differ; missing=%s extra=%s" % (missing, extra)
        )
    labels = frame["segment_id"].map({int(key): value for key, value in assignment.items()})
    if labels.isna().any():
        raise ConfigurationError("one or more rows have no split assignment")
    parts = {
        split: frame.loc[labels.eq(split)].copy().reset_index(drop=False)
        for split in ("train", "validation", "test")
    }
    segment_sets = {split: set(part["segment_id"]) for split, part in parts.items()}
    if not segment_sets["train"].isdisjoint(segment_sets["validation"]):
        raise ConfigurationError("train and validation share a segment")
    if not segment_sets["train"].isdisjoint(segment_sets["test"]):
        raise ConfigurationError("train and test share a segment")
    if not segment_sets["validation"].isdisjoint(segment_sets["test"]):
        raise ConfigurationError("validation and test share a segment")
    return parts


def _valid_anchors(segment_id: np.ndarray, window: int, horizon: int) -> np.ndarray:
    anchors = []
    for anchor in range(window, len(segment_id) - horizon + 1):
        target = anchor + horizon - 1
        current_segment = segment_id[anchor - 1]
        if segment_id[anchor - window] != current_segment:
            continue
        if segment_id[target] != current_segment:
            continue
        anchors.append(anchor)
    return np.asarray(anchors, dtype=np.int64)


DEFAULT_CONTAMINATION_BANDS_MS = (0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 99.0)


def apply_input_contamination(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Contaminate the input wind channel on censored frames, or pass through.

    Implements the mechanism measured on real curtailment segments (feathering
    lowers rotor thrust, so the nacelle anemometer reads closer to free stream):
    only rows where the cap actually binds are perturbed, and the multiplicative
    offset is applied to the model-visible ``wind`` column while the untouched
    readings are preserved in ``wind_clean``.

    The schedule variant looks up its per-band offset with the *clean* wind speed
    so the injected magnitude never depends on the injection itself.

    Returns the (possibly new) frame and an injection audit, or ``(frame, None)``
    when contamination is disabled.
    """

    contamination = config.get("contamination", {})
    if not contamination.get("enabled", False):
        return frame, None
    validate_contamination(contamination)

    mask = frame["is_censored"].astype(bool).to_numpy()
    wind_clean = frame["wind"].to_numpy(dtype=float)
    factor = float(contamination.get("intensity_factor", 1.0))
    relative = np.zeros(len(wind_clean), dtype=float)
    mode = str(contamination["mode"])
    if mode == "relative":
        relative[mask] = float(contamination["relative_delta"]) * factor
        band_edges = np.asarray(
            contamination.get("schedule_bins_ms", DEFAULT_CONTAMINATION_BANDS_MS),
            dtype=float,
        )
    else:
        band_edges = np.asarray(contamination["schedule_bins_ms"], dtype=float)
        deltas = np.asarray(contamination["schedule_relative_delta"], dtype=float)
        band = np.clip(
            np.searchsorted(band_edges, wind_clean, side="right") - 1,
            0,
            len(deltas) - 1,
        )
        relative[mask] = deltas[band[mask]] * factor

    wind_contaminated = wind_clean * (1.0 + relative)
    contaminated = frame.copy()
    contaminated["wind"] = wind_contaminated
    contaminated["wind_clean"] = wind_clean

    audit: Dict[str, Any] = {
        key: contamination[key]
        for key in sorted(contamination)
        if key in {
            "enabled",
            "channel",
            "trigger",
            "mode",
            "relative_delta",
            "intensity_factor",
            "schedule_bins_ms",
            "schedule_relative_delta",
            "measurement_provenance",
        }
    }
    audit.update(_contamination_statistics(contaminated))
    delta = wind_contaminated - wind_clean
    bands = []
    for index in range(len(band_edges) - 1):
        low = float(band_edges[index])
        high = float(band_edges[index + 1])
        in_band = mask & (wind_clean >= low) & (wind_clean < high)
        bands.append(
            {
                "clean_wind_ms_low": low,
                "clean_wind_ms_high": high,
                "contaminated_rows": int(in_band.sum()),
                "delta_ms_mean": float(delta[in_band].mean()) if in_band.any() else None,
                "relative_delta_applied": (
                    float(relative[in_band].mean()) if in_band.any() else None
                ),
            }
        )
    audit["by_clean_wind_band"] = bands
    return contaminated, audit


def _contamination_statistics(frame: pd.DataFrame) -> Dict[str, Any]:
    """Summarize the realized wind-channel perturbation of one frame."""

    clean = frame["wind_clean"].to_numpy(dtype=float)
    dirty = frame["wind"].to_numpy(dtype=float)
    delta = dirty - clean
    touched = delta != 0.0
    censored = frame["is_censored"].astype(bool).to_numpy()
    if np.any(touched & ~censored):
        raise ConfigurationError("contamination touched a non-censored frame")
    statistics: Dict[str, Any] = {
        "rows": int(len(frame)),
        "contaminated_rows": int(touched.sum()),
        "contaminated_fraction": float(touched.mean()) if len(frame) else 0.0,
        "censored_rows": int(censored.sum()),
        "clean_wind_max_ms": float(clean.max()) if len(frame) else None,
        "contaminated_wind_max_ms": float(dirty.max()) if len(frame) else None,
    }
    if touched.any():
        statistics.update(
            {
                "delta_ms_mean": float(delta[touched].mean()),
                "delta_ms_median": float(np.median(delta[touched])),
                "delta_ms_max": float(delta[touched].max()),
                "relative_delta_mean": float(
                    (delta[touched] / clean[touched]).mean()
                ),
            }
        )
    else:
        statistics.update(
            {
                "delta_ms_mean": None,
                "delta_ms_median": None,
                "delta_ms_max": None,
                "relative_delta_mean": None,
            }
        )
    return statistics


def build_window_split(
    frame: pd.DataFrame,
    rated_power_kw: float,
    wind_mean: float,
    wind_std: float,
    window: int,
    horizon: int,
    wind_reference: Optional[np.ndarray] = None,
) -> WindowSplit:
    """Materialize the exact feature and target arrays exposed to models.

    ``wind_reference`` overrides the source of the evaluation-only
    ``target_wind_ms`` stratification field and defaults to the model-visible
    wind column. Contaminated runs pass the clean readings so their strata stay
    comparable with the clean run; model features always use ``frame['wind']``.
    """

    power = frame["Y_syn"].to_numpy(dtype=np.float32) / rated_power_kw
    wind_raw = frame["wind"].to_numpy(dtype=np.float32)
    wind = (wind_raw - wind_mean) / wind_std
    if wind_reference is None:
        stratification_wind = wind_raw
    else:
        stratification_wind = np.asarray(wind_reference, dtype=np.float32)
        if stratification_wind.shape != wind_raw.shape:
            raise ConfigurationError("wind_reference must match the frame length")
    cap = frame["C_syn"].to_numpy(dtype=np.float32) / rated_power_kw
    censored = frame["is_censored"].to_numpy(dtype=np.float32)
    truth = frame["A_true"].to_numpy(dtype=np.float32) / rated_power_kw
    segment = frame["segment_id"].to_numpy(dtype=np.int64)
    anchors = _valid_anchors(segment, window, horizon)
    if anchors.size == 0:
        raise ConfigurationError("a split has no valid forecast windows")
    target_rows = anchors + horizon - 1
    features = np.empty((len(anchors), 4, window), dtype=np.float32)
    history_censor_frac = np.empty(len(anchors), dtype=np.float32)
    for output_index, anchor in enumerate(anchors):
        history = slice(anchor - window, anchor)
        features[output_index] = np.stack(
            (power[history], wind[history], cap[history], censored[history]), axis=0
        )
        history_censor_frac[output_index] = censored[history].mean()
    return WindowSplit(
        features=features,
        truth=truth[target_rows],
        observed=power[target_rows],
        cap=cap[target_rows],
        censored=censored[target_rows],
        segment_id=segment[target_rows],
        persistence=power[anchors - 1],
        target_row=target_rows,
        target_wind_ms=stratification_wind[target_rows],
        history_censor_frac=history_censor_frac,
        target_wind_visible_ms=wind_raw[target_rows],
    )


def _split_statistics(frame: pd.DataFrame, rated_power_kw: float) -> Dict[str, Any]:
    active = frame["policy_active"].astype(bool)
    active_caps = frame.loc[active, "C_syn"] / rated_power_kw
    statistics = {
        "rows": int(len(frame)),
        "segments": int(frame["segment_id"].nunique()),
        "wind_mean_ms": float(frame["wind"].mean()),
        "wind_q90_ms": float(frame["wind"].quantile(0.9)),
        "truth_mean_pu": float((frame["A_true"] / rated_power_kw).mean()),
        "truth_q90_pu": float((frame["A_true"] / rated_power_kw).quantile(0.9)),
        "policy_rate": float(active.mean()),
        "binding_rate": float(frame["is_censored"].mean()),
        "active_cap_mean_pu": float(active_caps.mean()) if len(active_caps) else None,
    }
    if "wind_clean" in frame.columns:
        # ``wind_mean_ms`` above describes the contaminated readings the models
        # actually see; the clean pair is kept alongside so the two are never
        # confused when a contaminated run is compared with the clean run.
        statistics["wind_clean_mean_ms"] = float(frame["wind_clean"].mean())
        statistics["wind_clean_q90_ms"] = float(frame["wind_clean"].quantile(0.9))
        statistics["contamination"] = _contamination_statistics(frame)
    return statistics


def prepare_scenario(
    scenario: str,
    frame: pd.DataFrame,
    assignment: Mapping[int, str],
    config: Mapping[str, Any],
) -> ScenarioData:
    """Validate, split, standardize, and materialize one scenario."""

    rated_power_kw = float(config["data"]["rated_power_kw"])
    validate_semisynthetic_frame(
        frame,
        rated_power_kw,
        int(config["data"]["sampling_interval_minutes"]),
        float(config["forecast"]["support_min_pu"]),
        float(config["forecast"]["support_max_pu"]),
    )
    # Contamination is injected only after the untouched frame has cleared the
    # data contract, so a perturbed input can never bypass validation.
    frame, contamination_audit = apply_input_contamination(frame, config)
    parts = apply_segment_assignment(frame, assignment)
    # Standardization deliberately uses the contaminated training readings: a
    # modeller in the field only ever sees the perturbed sensor, so borrowing the
    # clean statistics would leak information that does not exist.
    wind_mean = float(parts["train"]["wind"].mean())
    wind_std = float(parts["train"]["wind"].std())
    if not np.isfinite(wind_std) or wind_std < 1e-8:
        wind_std = 1.0
    window = int(config["forecast"]["window_minutes"])
    horizon = int(config["forecast"]["horizon_minutes"])
    windows = {
        split: build_window_split(
            part,
            rated_power_kw,
            wind_mean,
            wind_std,
            window,
            horizon,
            wind_reference=(
                part["wind_clean"].to_numpy(dtype=np.float32)
                if "wind_clean" in part.columns
                else None
            ),
        )
        for split, part in parts.items()
    }
    summary: Dict[str, Any] = {
        "wind_standardization_train_only": {"mean": wind_mean, "std": wind_std},
        "splits": {
            split: {
                **_split_statistics(part, rated_power_kw),
                "valid_windows": len(windows[split]),
                "window_segments": int(np.unique(windows[split].segment_id).size),
            }
            for split, part in parts.items()
        },
    }
    if contamination_audit is not None:
        summary["contamination"] = contamination_audit
    return ScenarioData(
        scenario=scenario,
        train=windows["train"],
        validation=windows["validation"],
        test=windows["test"],
        train_observed_rows=parts["train"]["Y_syn"].to_numpy(dtype=np.float32)
        / rated_power_kw,
        feature_names=(
            "observed_power_pu",
            "wind_speed_train_standardized",
            "cap_pu",
            "censoring_indicator",
        ),
        split_summary=summary,
        data_signature=segment_signature(frame),
        train_wind_ms_rows=parts["train"]["wind"].to_numpy(dtype=np.float32),
        train_censored_rows=parts["train"]["is_censored"].to_numpy(dtype=np.float32),
    )
