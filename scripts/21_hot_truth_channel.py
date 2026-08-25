"""Rebuild the archived Hill of Towie T11 truth-channel inputs.

This is a provenance-preserving replacement for a historical script that was
not retained.  The routing rule has been recovered exactly from the published
input files and archived result counts:

* usable generation: observed power > 50 kW and nacelle wind speed >= 4 m/s;
* curtailed: usable generation and PowerRef < 0.95 x 2300 kW;
* free usable rows use measured power (``pap_source=actual``);
* curtailed rows use a 0.5 m/s binned-median power curve
  (``pap_source=power_curve``);
* all remaining rows use ``pap_source=none``.

The historical neighbour-turbine route had zero coverage in the archived
outputs.  It is therefore recorded as unavailable rather than re-created with
unverifiable assumptions.  The resulting Parquet files reproduce the exact
route-composition counts used by the manuscript's bias audit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ONE_MINUTE = (
    PROJECT_DIR
    / "data"
    / "hill_of_towie"
    / "converted"
    / "fl_df_HOT_T11_1min.parquet"
)
DEFAULT_TEN_MINUTE_ARCHIVE = PROJECT_DIR / "data" / "hill_of_towie" / "2026.zip"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "hot_truth_channel"

STATION_ID = 2_304_520
RATED_POWER_KW = 2300.0
ACTIVE_POWER_KW = 50.0
MIN_WIND_MS = 4.0
FREE_REFERENCE_FRACTION = 0.95
POWER_CURVE_BIN_MS = 0.5
POWER_CURVE_MIN_COUNT = 10
TEMPORAL_TRAIN_FRACTION = 0.70
MONTHS = ("2026_01", "2026_02", "2026_03", "2026_04")

EXPECTED_COMPOSITION = {
    "truth_channel_T11_1min.parquet": {
        "rows": 172_800,
        "actual": 96_843,
        "power_curve": 47_244,
        "none": 28_713,
    },
    "truth_channel_T11_10min.parquet": {
        "rows": 17_281,
        "actual": 10_294,
        "power_curve": 3_947,
        "none": 3_040,
    },
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_one_minute(path: Path) -> pd.DataFrame:
    """Load T11 and restore the complete published January--April clock."""

    raw = pd.read_parquet(path)
    if isinstance(raw.columns, pd.MultiIndex):
        station_columns = {str(column[0]) for column in raw.columns}
        if station_columns != {str(STATION_ID)}:
            raise ValueError("expected only station %s in %s" % (STATION_ID, path))
        raw.columns = [column[1] for column in raw.columns]
    required = {"ActPower_Value", "AcWindSp_AcWindSp", "PowerRef_PowerRef"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError("missing one-minute columns: %s" % sorted(missing))
    raw = raw.sort_index()
    return raw.rename(
        columns={
            "ActPower_Value": "power_obs_kW",
            "AcWindSp_AcWindSp": "wind_ms",
            "PowerRef_PowerRef": "pref_kW",
            "PowerRed_PowerRed": "power_red",
        }
    )[["power_obs_kW", "wind_ms", "pref_kW", "power_red"]]


def load_ten_minute(path: Path) -> pd.DataFrame:
    """Read the published 10-minute T11 channels directly from ``2026.zip``."""

    frames = []
    with zipfile.ZipFile(path) as archive:
        for month in MONTHS:
            turbine = pd.read_csv(
                io.BytesIO(archive.read("tblSCTurbine_%s.csv" % month)),
                usecols=[
                    "TimeStamp",
                    "StationId",
                    "wtc_PowerRef_endvalue",
                    "wtc_AcWindSp_mean",
                ],
            )
            grid = pd.read_csv(
                io.BytesIO(archive.read("tblSCTurGrid_%s.csv" % month)),
                usecols=["TimeStamp", "StationId", "wtc_ActPower_mean"],
            )
            digital = pd.read_csv(
                io.BytesIO(archive.read("tblSCTurDigiIn_%s.csv" % month)),
                usecols=["TimeStamp", "StationId", "wtc_PowerRed_endvalue"],
            )
            frame = turbine.merge(grid, on=["TimeStamp", "StationId"], how="inner")
            frame = frame.merge(digital, on=["TimeStamp", "StationId"], how="inner")
            frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["TimeStamp"] = pd.to_datetime(result["TimeStamp"])
    complete_clock = pd.DatetimeIndex(result["TimeStamp"].drop_duplicates().sort_values())
    # Match the public-data loader used by the original analysis: its
    # ``pivot_table`` averages duplicate (timestamp, station) records.
    result = (
        result.loc[result["StationId"] == STATION_ID]
        .drop(columns=["StationId"])
        .groupby("TimeStamp", sort=True, as_index=True)
        .mean(numeric_only=True)
        .reindex(complete_clock)
    )
    result.index.name = None
    return result.rename(
        columns={
            "wtc_ActPower_mean": "power_obs_kW",
            "wtc_AcWindSp_mean": "wind_ms",
            "wtc_PowerRef_endvalue": "pref_kW",
            "wtc_PowerRed_endvalue": "power_red",
        }
    )[["power_obs_kW", "wind_ms", "pref_kW", "power_red"]]


def fit_power_curve(wind: np.ndarray, power: np.ndarray) -> Tuple[Callable, list, list]:
    """Fit the recovered 0.5 m/s binned-median and linear interpolation rule."""

    finite = np.isfinite(wind) & np.isfinite(power)
    wind = np.asarray(wind, dtype=float)[finite]
    power = np.asarray(power, dtype=float)[finite]
    edges = np.arange(0.0, 30.0 + POWER_CURVE_BIN_MS, POWER_CURVE_BIN_MS)
    labels = pd.cut(wind, edges, labels=edges[:-1] + POWER_CURVE_BIN_MS / 2.0)
    grouped = pd.DataFrame({"bin": labels, "power": power}).groupby(
        "bin", observed=True
    )["power"]
    summary = grouped.agg(["median", "size"])
    summary = summary.loc[summary["size"] >= POWER_CURVE_MIN_COUNT]
    if len(summary) < 2:
        raise ValueError("too few populated wind bins to fit a power curve")
    centers = summary.index.astype(float).to_numpy()
    values = summary["median"].to_numpy(dtype=float)

    def predict(values_to_predict):
        values_to_predict = np.asarray(values_to_predict, dtype=float)
        return np.clip(
            np.interp(
                values_to_predict,
                centers,
                values,
                left=0.0,
                right=values[-1],
            ),
            0.0,
            RATED_POWER_KW,
        )

    return predict, centers.tolist(), values.tolist()


def build_truth_channel(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply the recovered routing rule and return a transparent audit record."""

    frame = frame.copy()
    eligible = (
        frame["power_obs_kW"].gt(ACTIVE_POWER_KW)
        & frame["wind_ms"].ge(MIN_WIND_MS)
    )
    curtailed = eligible & frame["pref_kW"].lt(
        FREE_REFERENCE_FRACTION * RATED_POWER_KW
    )
    actual = eligible & ~curtailed

    actual_positions = np.flatnonzero(actual.to_numpy())
    train_count = int(np.floor(TEMPORAL_TRAIN_FRACTION * len(actual_positions)))
    train_positions = actual_positions[:train_count]
    holdout_positions = actual_positions[train_count:]
    curve, centers, values = fit_power_curve(
        frame["wind_ms"].to_numpy()[train_positions],
        frame["power_obs_kW"].to_numpy()[train_positions],
    )
    holdout_prediction = curve(frame["wind_ms"].to_numpy()[holdout_positions])
    holdout_observed = frame["power_obs_kW"].to_numpy()[holdout_positions]
    holdout_mae = float(np.mean(np.abs(holdout_prediction - holdout_observed)))

    source = np.full(len(frame), "none", dtype=object)
    source[actual.to_numpy()] = "actual"
    source[curtailed.to_numpy()] = "power_curve"
    curve_prediction = np.full(len(frame), np.nan, dtype=float)
    reconstructable = curtailed.to_numpy() & np.isfinite(frame["wind_ms"].to_numpy())
    curve_prediction[reconstructable] = curve(
        frame["wind_ms"].to_numpy()[reconstructable]
    )
    pap = np.full(len(frame), np.nan, dtype=float)
    pap[actual.to_numpy()] = frame["power_obs_kW"].to_numpy()[actual.to_numpy()]
    pap[reconstructable] = np.maximum(
        curve_prediction[reconstructable],
        frame["power_obs_kW"].to_numpy()[reconstructable],
    )

    output = frame.copy()
    output["curtailed"] = curtailed.to_numpy(dtype=bool)
    output["pap_source"] = source
    output["pap_power_curve_kW"] = curve_prediction
    output["pap_neighbor_kW"] = np.nan
    output["pap_kW"] = pap
    output.index.name = "timestamp"

    audit = {
        "rows": int(len(output)),
        "curtailed_rows": int(curtailed.sum()),
        "pap_source_counts": {
            key: int(value) for key, value in output["pap_source"].value_counts().items()
        },
        "neighbor_route_coverage": 0,
        "power_curve": {
            "bin_width_ms": POWER_CURVE_BIN_MS,
            "minimum_bin_count": POWER_CURVE_MIN_COUNT,
            "centers_ms": centers,
            "median_power_kW": values,
            "temporal_train_fraction": TEMPORAL_TRAIN_FRACTION,
            "train_rows": int(train_count),
            "holdout_rows": int(len(holdout_positions)),
            "holdout_mae_kW": holdout_mae,
            "holdout_mae_fraction_rated": holdout_mae / RATED_POWER_KW,
        },
    }
    return output, audit


def verify_composition(name: str, audit: Dict) -> None:
    expected = EXPECTED_COMPOSITION[name]
    actual = {
        "rows": audit["rows"],
        **{
            key: audit["pap_source_counts"].get(key, 0)
            for key in ("actual", "power_curve", "none")
        },
    }
    if actual != expected:
        raise RuntimeError(
            "truth-channel composition differs from the archived contract for %s:\n"
            "expected %r\nactual   %r" % (name, expected, actual)
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-minute", type=Path, default=DEFAULT_ONE_MINUTE)
    parser.add_argument("--ten-minute-archive", type=Path, default=DEFAULT_TEN_MINUTE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-composition-check",
        action="store_true",
        help="allow non-v2 or partial inputs whose counts differ from the archived run",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    for path in (arguments.one_minute, arguments.ten_minute_archive):
        if not path.exists():
            raise SystemExit("missing input: %s" % path)

    ten_minute = load_ten_minute(arguments.ten_minute_archive)
    one_minute = flatten_one_minute(arguments.one_minute)
    # The standard archive includes a final boundary at 2026-05-01 00:00.
    # Restore the complete minute clock but keep the fully missing T11 day NaN.
    minute_clock = pd.date_range(ten_minute.index.min(), ten_minute.index.max(), freq="1min")[:-1]
    one_minute = one_minute.reindex(minute_clock)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "reconstructed replacement for missing historical script",
        "routing_contract": {
            "eligible": "power_obs_kW > 50 and wind_ms >= 4",
            "curtailed": "eligible and pref_kW < 0.95 * 2300",
            "actual_route": "eligible and not curtailed",
            "power_curve_route": "all curtailed rows",
            "neighbor_route": "unavailable; archived coverage was zero",
        },
        "historical_reference_holdout": {
            "mae_kW_rounded": 65.0,
            "fraction_rated_rounded": 0.028,
            "note": "legacy prose value; the manifest below contains the recomputed value",
        },
        "inputs": {
            "one_minute": "data/hill_of_towie/converted/fl_df_HOT_T11_1min.parquet",
            "ten_minute_archive": "data/hill_of_towie/2026.zip",
        },
        "outputs": {},
    }
    for name, frame in (
        ("truth_channel_T11_10min.parquet", ten_minute),
        ("truth_channel_T11_1min.parquet", one_minute),
    ):
        truth, audit = build_truth_channel(frame)
        if not arguments.skip_composition_check:
            verify_composition(name, audit)
        destination = arguments.output_dir / name
        truth.to_parquet(destination)
        audit["path"] = destination.name
        audit["sha256"] = sha256_of(destination)
        manifest["outputs"][name] = audit
        print(
            "wrote %s: rows=%d actual=%d power_curve=%d none=%d holdout_MAE=%.1f kW"
            % (
                destination,
                audit["rows"],
                audit["pap_source_counts"].get("actual", 0),
                audit["pap_source_counts"].get("power_curve", 0),
                audit["pap_source_counts"].get("none", 0),
                audit["power_curve"]["holdout_mae_kW"],
            )
        )

    (arguments.output_dir / "truth_channel_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
