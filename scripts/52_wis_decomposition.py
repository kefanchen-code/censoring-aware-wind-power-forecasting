# -*- coding: utf-8 -*-
"""Decompose WIS into dispersion / overprediction / underprediction penalties.

Pure post-processing: the script only reads the frozen ``forecast.npz`` artifacts
written by ``run_pap_benchmark.py`` and never retrains anything.  For every
scenario/model/seed it recomputes the Bracher et al. (2021) three-way
decomposition, asserts that the three components sum back to the stored WIS, and
aggregates by (run, scenario, model, window type) where the window type is
``censored`` / ``clean`` according to ``target_censored``.

Outputs (written into each benchmark output directory):

- ``wis_components_by_run.csv``: one row per run/scenario/model/seed/window type;
- ``wis_components.csv``:        seed mean and sample standard deviation;
- ``wis_components_contrast.csv``: challenger minus reference component deltas
  for every pair declared in the config ``comparisons`` block.

The contrast table is the diagnostic the manuscript needs: if the B4 -> B6 gain
sits in the underprediction penalty, the improvement comes from no longer
learning the suppressed observations; if ``dispersion`` rises while the total
falls, the gain is calibration rather than a sharpness trick.

Usage:
    python scripts/52_wis_decomposition.py --config configs/pap_benchmark_semisyn.json
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from censored_wind_power.benchmark.scoring import (
    wis_components_per_sample,
    wis_per_sample,
)

COMPONENTS = ("dispersion", "overprediction", "underprediction")
WINDOW_TYPES = ("all", "censored", "clean")
# Stored quantiles are float32 while the stored WIS was computed in float64, so
# the identity check needs a float32-level tolerance rather than exact equality.
IDENTITY_ATOL = 1e-6


def decompose_run(forecast_path, levels, alphas):
    """Return per-window-type component means for a single frozen forecast."""

    with np.load(forecast_path) as arrays:
        quantiles = arrays["quantiles"].astype(np.float64)
        targets = arrays["target"].astype(np.float64)
        censored = arrays["target_censored"].astype(np.float64) >= 0.5
        stored_wis = arrays["wis"].astype(np.float64)

    components = wis_components_per_sample(quantiles, targets, levels, alphas)
    total = sum(components[name] for name in COMPONENTS)
    recomputed = wis_per_sample(quantiles, targets, levels, alphas)
    identity_gap = float(np.max(np.abs(total - recomputed)))
    if identity_gap > IDENTITY_ATOL:
        raise RuntimeError(
            "decomposition identity violated by %.3e in %s"
            % (identity_gap, forecast_path)
        )
    stored_gap = float(np.max(np.abs(total - stored_wis)))
    if stored_gap > IDENTITY_ATOL:
        raise RuntimeError(
            "decomposition disagrees with stored WIS by %.3e in %s"
            % (stored_gap, forecast_path)
        )

    masks = {
        "all": np.ones_like(censored, dtype=bool),
        "censored": censored,
        "clean": ~censored,
    }
    results = {}
    for window_type in WINDOW_TYPES:
        mask = masks[window_type]
        entry = {
            "n_windows": int(mask.sum()),
            "identity_max_abs_gap": stored_gap,
        }
        if entry["n_windows"] == 0:
            for name in COMPONENTS:
                entry[name] = float("nan")
                entry["%s_share" % name] = float("nan")
            entry["wis"] = float("nan")
            results[window_type] = entry
            continue
        wis_mean = float(total[mask].mean())
        entry["wis"] = wis_mean
        for name in COMPONENTS:
            value = float(components[name][mask].mean())
            entry[name] = value
            entry["%s_share" % name] = value / wis_mean if wis_mean > 0 else float("nan")
        results[window_type] = entry
    return results


def load_runs(run_name, output_dir, scenarios, models, levels, alphas):
    rows = []
    for scenario in scenarios:
        for model in models:
            model_dir = output_dir / "artifacts" / scenario / model
            if not model_dir.exists():
                raise FileNotFoundError("missing artifact directory: %s" % model_dir)
            for run_dir in sorted(model_dir.iterdir()):
                forecast_path = run_dir / "forecast.npz"
                artifact_path = run_dir / "artifact.json"
                if not forecast_path.exists() or not artifact_path.exists():
                    continue
                with artifact_path.open("r", encoding="utf-8") as handle:
                    artifact = json.load(handle)
                decomposed = decompose_run(forecast_path, levels, alphas)
                for window_type in WINDOW_TYPES:
                    entry = decomposed[window_type]
                    row = {
                        "run": run_name,
                        "scenario": scenario,
                        "model": model,
                        "seed": artifact.get("seed"),
                        "window_type": window_type,
                        "n_windows": entry["n_windows"],
                        "wis_mean": entry["wis"],
                    }
                    for name in COMPONENTS:
                        row["%s_mean" % name] = entry[name]
                        row["%s_share" % name] = entry["%s_share" % name]
                    row["identity_max_abs_gap"] = entry["identity_max_abs_gap"]
                    rows.append(row)
    return rows


def summarize(rows):
    metric_keys = ["wis_mean"]
    for name in COMPONENTS:
        metric_keys.extend(["%s_mean" % name, "%s_share" % name])
    grouped = {}
    for row in rows:
        key = (row["run"], row["scenario"], row["model"], row["window_type"])
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (run_name, scenario, model, window_type), runs in grouped.items():
        entry = {
            "run": run_name,
            "scenario": scenario,
            "model": model,
            "window_type": window_type,
            "n_runs": len(runs),
            "n_windows": runs[0]["n_windows"],
        }
        for key in metric_keys:
            values = [run[key] for run in runs]
            if any(value != value for value in values):  # NaN guard
                entry["%s_seedmean" % key] = float("nan")
                entry["%s_seedsd" % key] = float("nan")
                continue
            entry["%s_seedmean" % key] = statistics.mean(values)
            entry["%s_seedsd" % key] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary_rows.append(entry)
    return summary_rows


def contrast(summary_rows, comparisons):
    index = {
        (row["run"], row["scenario"], row["model"], row["window_type"]): row
        for row in summary_rows
    }
    keys = ["wis_mean"] + ["%s_mean" % name for name in COMPONENTS]
    contrast_rows = []
    for comparison in comparisons:
        reference = comparison["reference"]
        challenger = comparison["challenger"]
        for (run_name, scenario, model, window_type), row in sorted(index.items()):
            if model != challenger:
                continue
            base = index.get((run_name, scenario, reference, window_type))
            if base is None:
                continue
            entry = {
                "run": run_name,
                "family": comparison.get("family", ""),
                "scenario": scenario,
                "reference": reference,
                "challenger": challenger,
                "window_type": window_type,
                "n_windows": row["n_windows"],
            }
            for key in keys:
                name = key[: -len("_mean")] if key.endswith("_mean") else key
                entry["delta_%s" % name] = (
                    row["%s_seedmean" % key] - base["%s_seedmean" % key]
                )
            total_delta = entry["delta_wis"]
            for name in COMPONENTS:
                entry["fraction_of_delta_%s" % name] = (
                    entry["delta_%s" % name] / total_delta
                    if total_delta != 0.0
                    else float("nan")
                )
            contrast_rows.append(entry)
    return contrast_rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("refusing to write empty table: %s" % path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_config(project_dir, config_path, allow_incomplete):
    with (project_dir / config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    output_dir = project_dir / config["output_dir"]
    run_name = output_dir.name

    status_path = output_dir / "benchmark_status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    complete = bool(status.get("complete") and status.get("valid_for_final_comparison"))
    if not complete and not allow_incomplete:
        raise RuntimeError(
            "benchmark_status.json is not valid for final comparison (%s); "
            "pass --allow-incomplete to decompose anyway" % status_path
        )

    evaluation = config["evaluation"]
    levels = np.asarray(evaluation["quantiles"], dtype=np.float64)
    alphas = [float(value) for value in evaluation["interval_alphas"]]

    rows = load_runs(
        run_name, output_dir, config["scenarios"], config["models"], levels, alphas
    )
    summary_rows = summarize(rows)
    contrast_rows = contrast(summary_rows, config.get("comparisons", []))

    by_run_path = output_dir / "wis_components_by_run.csv"
    summary_path = output_dir / "wis_components.csv"
    contrast_path = output_dir / "wis_components_contrast.csv"
    write_csv(by_run_path, rows)
    write_csv(summary_path, summary_rows)
    if contrast_rows:
        write_csv(contrast_path, contrast_rows)

    print("run:", run_name)
    print("  protocol_id:", status.get("protocol_id"))
    print("  complete:", complete)
    print("  forecast files decomposed:", len(rows) // len(WINDOW_TYPES))
    print("  max identity gap:", max(row["identity_max_abs_gap"] for row in rows))
    print("  wrote:", by_run_path)
    print("  wrote:", summary_path)
    if contrast_rows:
        print("  wrote:", contrast_path)
    return summary_rows, contrast_rows


def report_primary(contrast_rows, reference, challenger):
    selected = [
        row
        for row in contrast_rows
        if row["reference"] == reference and row["challenger"] == challenger
    ]
    if not selected:
        return
    print()
    print("%s -> %s component deltas (seed means, pu):" % (reference, challenger))
    header = "  %-18s %-9s %10s %10s %10s %10s" % (
        "scenario",
        "windows",
        "dWIS",
        "dDisp",
        "dOver",
        "dUnder",
    )
    print(header)
    for row in selected:
        print(
            "  %-18s %-9s %10.5f %10.5f %10.5f %10.5f"
            % (
                row["scenario"],
                row["window_type"],
                row["delta_wis"],
                row["delta_dispersion"],
                row["delta_overprediction"],
                row["delta_underprediction"],
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/pap_benchmark_semisyn.json"],
        type=Path,
        help="one or more benchmark configs; each output_dir gets its own tables",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="decompose runs whose benchmark_status.json is not final",
    )
    parser.add_argument(
        "--primary-reference",
        default="B4",
        help="reference model for the printed contrast summary",
    )
    parser.add_argument(
        "--primary-challenger",
        default="B6",
        help="challenger model for the printed contrast summary",
    )
    args = parser.parse_args()
    project_dir = PROJECT_DIR

    for config_path in args.config:
        _, contrast_rows = process_config(
            project_dir, config_path, args.allow_incomplete
        )
        report_primary(contrast_rows, args.primary_reference, args.primary_challenger)


if __name__ == "__main__":
    main()
