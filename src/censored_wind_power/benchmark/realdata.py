"""Real-data view for the Altahullion T11 release-event validation layer.

Builds an observable-only windowing contract from the real 1-min SCADA record
using the canonical audit classification (U free rows, R right-censored rows;
I/X excluded). No latent truth exists on this track and nothing in this module
fabricates one: censored labels carry the observed power together with the
binding cap (PowerRef), exactly the contract the semi-synthetic benchmark
hands to its learned baselines.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
RATED_POWER_KW = 1330.0
NON_BINDING_CAP_PU = 1.05
WINDOW_MINUTES = 60
HORIZON_MINUTES = 15
RELEASE_EVAL_HORIZONS_MIN = (1, 5, 15)


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "altahullion_audit", PROJECT_DIR / "scripts" / "10_altahullion_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_real_frame() -> pd.DataFrame:
    """Classified real record kept on the natural one-minute clock.

    Columns: observed_pu, cap_pu, censored, wind_ms, label, strict_r, usable,
    with a DatetimeIndex covering every recorded minute. The reconstructed
    protocol v1.3 preserves the executable v1.2 semantics: every
    active curtailment minute (PowerRed>0; audit labels R, I and X alike) is
    right-censored at PowerRef, while U rows are free labels. Minutes missing
    PowerRef, inactive minutes and record gaps are unusable. ``strict_r``
    marks the conservative strict-R subset for the sensitivity stratum.
    Training windows draw only from contiguous usable blocks; event feature
    stacks read this frame directly because the release instant is the minute
    AFTER the last censored minute and only exists on the raw clock.
    """

    audit = _load_audit_module()
    frame = audit.load_frame()
    frame, _ = audit.classify_observations(frame)
    active = frame["active"].to_numpy()
    curtailment = frame["curtail_flag"].to_numpy()
    reference_kw = frame["power_ref"].to_numpy(dtype=float)
    reference_finite = np.isfinite(reference_kw)

    free = frame["label"].eq("U").to_numpy()
    censored = active & curtailment & reference_finite
    usable = free | censored

    power_kw = frame["power"].to_numpy(dtype=float)
    observed_pu = np.where(
        usable & active, power_kw / RATED_POWER_KW, np.nan
    ).astype(np.float32)
    cap_pu = np.where(
        censored,
        reference_kw / RATED_POWER_KW,
        NON_BINDING_CAP_PU,
    ).astype(np.float32)

    real = pd.DataFrame(
        {
            "observed_pu": observed_pu,
            "cap_pu": cap_pu,
            "censored": censored.astype(np.float32),
            "wind_ms": frame["wind"].to_numpy(dtype=np.float32),
            "label": frame["label"].to_numpy(),
            "strict_r": frame["label"].eq("R").to_numpy(),
            "usable": usable,
        },
        index=frame.index,
    )
    real.index.name = "timestamp"
    return real


def usable_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Contiguous usable (U/R) blocks with running segment identifiers."""

    usable = frame["usable"].to_numpy()
    starts = np.r_[True, usable[1:] & ~usable[:-1]]
    block_ids = np.cumsum(usable & starts)
    block_frame = frame.loc[usable, [
        "observed_pu", "cap_pu", "censored", "wind_ms", "label"
    ]].copy()
    block_frame["segment_id"] = (block_ids[usable] - 1).astype(np.int64)
    return block_frame


def assign_segments(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> Dict[int, str]:
    """Chronological whole-segment assignment by cumulative row midpoint."""

    sizes = frame.groupby("segment_id", sort=False).size()
    total = float(sizes.sum())
    train_cut = train_fraction * total
    validation_cut = (train_fraction + validation_fraction) * total
    assignment: Dict[int, str] = {}
    cumulative = 0.0
    for raw_segment_id, raw_size in sizes.items():
        midpoint = cumulative + 0.5 * float(raw_size)
        if midpoint <= train_cut:
            split = "train"
        elif midpoint <= validation_cut:
            split = "validation"
        else:
            split = "test"
        assignment[int(raw_segment_id)] = split
        cumulative += float(raw_size)
    return assignment


@dataclass(frozen=True)
class RealWindowSplit:
    """Windows for one split at one evaluation horizon (minutes ahead)."""

    features: np.ndarray
    observed: np.ndarray
    cap: np.ndarray
    censored: np.ndarray
    segment_id: np.ndarray
    persistence: np.ndarray
    target_time: np.ndarray

    def __len__(self) -> int:
        return int(len(self.observed))


def build_real_windows(
    frame: pd.DataFrame,
    assignment: Mapping[int, str],
    horizon_minutes: int,
    window_minutes: int = WINDOW_MINUTES,
) -> Tuple[Dict[str, RealWindowSplit], Dict[str, float]]:
    """Materialize per-split windows for a single forecast horizon.

    A window is valid only when both the full history and the target row lie
    in the same contiguous U/R segment, so no window crosses an I/X gap or a
    split boundary. Wind standardization uses training-side readings only.
    Returns the split windows and the training-side standardization constants.
    """

    segment = frame["segment_id"].to_numpy()
    power = frame["observed_pu"].to_numpy(dtype=np.float32)
    cap = frame["cap_pu"].to_numpy(dtype=np.float32)
    censored = frame["censored"].to_numpy(dtype=np.float32)
    wind = frame["wind_ms"].to_numpy(dtype=np.float32)
    times = frame.index

    train_mask = np.isin(segment, [s for s, label in assignment.items() if label == "train"])
    train_wind = frame.loc[train_mask, "wind_ms"]
    wind_mean = float(train_wind.mean())
    wind_std = float(train_wind.std())
    if not np.isfinite(wind_std) or wind_std < 1e-8:
        wind_std = 1.0
    wind_standardized = ((wind - wind_mean) / wind_std).astype(np.float32)

    splits: Dict[str, dict] = {
        name: {"arrays": []} for name in ("train", "validation", "test")
    }
    for position in range(window_minutes, len(frame) - horizon_minutes + 1):
        target = position + horizon_minutes - 1
        current = segment[position - 1]
        if segment[position - window_minutes] != current or segment[target] != current:
            continue
        split_name = assignment.get(int(current))
        if split_name is None:
            continue
        history = slice(position - window_minutes, position)
        feature = np.stack(
            (
                power[history],
                wind_standardized[history],
                cap[history],
                censored[history],
            ),
            axis=0,
        )
        splits[split_name]["arrays"].append(
            (
                feature,
                power[target],
                cap[target],
                censored[target],
                current,
                power[position - 1],
                times[target],
            )
        )

    standardization = {"wind_mean_ms": wind_mean, "wind_std_ms": wind_std}
    windows: Dict[str, RealWindowSplit] = {}
    for name, bucket in splits.items():
        if not bucket["arrays"]:
            raise ValueError("split %r produced no valid windows" % name)
        (
            features,
            observed,
            caps,
            censored_rows,
            segments,
            persistence,
            target_times,
        ) = zip(*bucket["arrays"])
        windows[name] = RealWindowSplit(
            features=np.stack(features).astype(np.float32),
            observed=np.asarray(observed, dtype=np.float32),
            cap=np.asarray(caps, dtype=np.float32),
            censored=np.asarray(censored_rows, dtype=np.float32),
            segment_id=np.asarray(segments, dtype=np.int64),
            persistence=np.asarray(persistence, dtype=np.float32),
            target_time=np.asarray(target_times),
        )
    return windows, standardization


def load_release_events() -> pd.DataFrame:
    """Frozen outcome-blind release-event list from the audit."""

    path = (
        PROJECT_DIR
        / "results"
        / "altahullion_audit"
        / "qualified_release_events.csv"
    )
    events = pd.read_csv(path, parse_dates=["release_time"])
    return events.sort_values("release_time").reset_index(drop=True)


def event_windows(
    frame: pd.DataFrame,
    release_times: pd.Series,
    window_minutes: int = WINDOW_MINUTES,
    wind_standardization: Mapping[str, float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature stacks ending at each release instant (history-only, no future).

    ``frame`` is the full natural-clock record from ``load_real_frame``. An
    event is valid only when the release instant exists in the record and the
    preceding 60 minutes are complete, active, usable (U/R) minutes; the
    release instant itself may be any recorded minute since only its history
    is consumed.
    """

    times = frame.index
    power = frame["observed_pu"].to_numpy(dtype=np.float32)
    cap = frame["cap_pu"].to_numpy(dtype=np.float32)
    censored = frame["censored"].to_numpy(dtype=np.float32)
    wind = frame["wind_ms"].to_numpy(dtype=np.float32)
    usable = frame["usable"].to_numpy()
    position_by_time = pd.Series(np.arange(len(frame)), index=times)
    wind_finite = np.isfinite(wind)
    if wind_standardization is None:
        wind_mean = float(np.nanmean(wind))
        wind_std = float(np.nanstd(wind))
    else:
        wind_mean = float(wind_standardization["wind_mean_ms"])
        wind_std = float(wind_standardization["wind_std_ms"])
    wind_standardized = np.where(
        wind_finite, ((wind - wind_mean) / wind_std), 0.0
    ).astype(np.float32)

    features = np.zeros((len(release_times), 4, window_minutes), dtype=np.float32)
    persistence = np.zeros(len(release_times), dtype=np.float32)
    valid = np.zeros(len(release_times), dtype=bool)
    for event_index, release_time in enumerate(release_times):
        if release_time not in position_by_time.index:
            continue
        position = int(position_by_time.loc[release_time])
        if position < window_minutes:
            continue
        history = slice(position - window_minutes, position)
        expected = pd.date_range(
            end=times[position - 1], periods=window_minutes, freq="1min"
        )
        if not times[history].equals(expected):
            continue
        if not usable[history].all():
            continue
        history_power = power[history]
        if not np.isfinite(history_power).all():
            continue
        features[event_index] = np.stack(
            (
                history_power,
                wind_standardized[history],
                cap[history],
                censored[history],
            ),
            axis=0,
        )
        persistence[event_index] = power[position - 1]
        valid[event_index] = True
    return features, persistence, valid


def observed_after_release(
    raw_frame: pd.DataFrame, release_time, horizon_minutes: int
) -> float:
    """Measured power h minutes after release (no truth assumption)."""

    target = release_time + pd.Timedelta(minutes=horizon_minutes)
    if target in raw_frame.index:
        value = raw_frame.loc[target, "power"]
        if pd.notna(value):
            return float(value) / RATED_POWER_KW
    return float("nan")
