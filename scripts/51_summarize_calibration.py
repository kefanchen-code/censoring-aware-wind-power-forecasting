# -*- coding: utf-8 -*-
"""Aggregate calibration diagnostics from frozen-protocol benchmark artifacts.

Read-only with respect to the frozen protocol: this script only reads
``artifact.json`` files produced by ``run_pap_benchmark.py`` and writes two
new CSV files into the benchmark output directory:

- ``calibration_by_run.csv``:   one row per scenario/model/seed;
- ``calibration_summary.csv``:  seed mean and sample standard deviation per
  scenario/model, intended as the single data source for the manuscript
  calibration table.

Usage:
    python scripts/51_summarize_calibration.py --config configs/pap_benchmark_v2.json
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

COVERAGE_KEYS = ("40", "60", "80", "90")
SCALAR_KEYS = (
    "wis_mean",
    "crps_mean",
    "median_mae",
    "median_bias",
    "pit_mean",
    "pit_std",
)


def load_runs(output_dir, scenarios, models):
    rows = []
    for scenario in scenarios:
        for model in models:
            model_dir = output_dir / "artifacts" / scenario / model
            if not model_dir.exists():
                raise FileNotFoundError("missing artifact directory: %s" % model_dir)
            for run_dir in sorted(model_dir.iterdir()):
                artifact_path = run_dir / "artifact.json"
                if not artifact_path.exists():
                    continue
                with artifact_path.open("r", encoding="utf-8") as handle:
                    artifact = json.load(handle)
                summary = artifact["score_summary"]
                row = {
                    "scenario": scenario,
                    "model": model,
                    "seed": artifact.get("seed"),
                    "crps_kind": summary["crps_kind"],
                    "n_test_windows": summary["n_samples"],
                }
                for key in SCALAR_KEYS:
                    row[key] = summary[key]
                for key in COVERAGE_KEYS:
                    row["coverage_%s" % key] = summary["coverage"][key]
                    row["coverage_error_%s" % key] = summary["coverage_error"][key]
                    row["interval_width_%s" % key] = summary["interval_width_mean"][key]
                rows.append(row)
    return rows


def summarize(rows):
    metric_keys = list(SCALAR_KEYS)
    for key in COVERAGE_KEYS:
        metric_keys.extend(
            ["coverage_%s" % key, "coverage_error_%s" % key, "interval_width_%s" % key]
        )
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["model"]), []).append(row)
    summary_rows = []
    for (scenario, model), runs in grouped.items():
        entry = {
            "scenario": scenario,
            "model": model,
            "n_runs": len(runs),
            "crps_kind": runs[0]["crps_kind"],
        }
        for key in metric_keys:
            values = [run[key] for run in runs]
            entry["%s_mean" % key] = statistics.mean(values)
            entry["%s_sd" % key] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary_rows.append(entry)
    return summary_rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/pap_benchmark_v2.json", type=Path
    )
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    with (project_dir / args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    output_dir = project_dir / config["output_dir"]

    status_path = output_dir / "benchmark_status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if not (status.get("complete") and status.get("valid_for_final_comparison")):
        raise RuntimeError(
            "benchmark_status.json is not valid for final comparison; refusing to summarize"
        )

    rows = load_runs(output_dir, config["scenarios"], config["models"])
    expected = 0
    for model in config["models"]:
        expected += len(config["seeds"]) if model not in ("B0_persistence", "B0_climatology") else 1
    expected *= len(config["scenarios"])
    if len(rows) != expected:
        raise RuntimeError(
            "expected %d artifact rows, found %d" % (expected, len(rows))
        )

    by_run_path = output_dir / "calibration_by_run.csv"
    summary_path = output_dir / "calibration_summary.csv"
    write_csv(by_run_path, rows)
    write_csv(summary_path, summarize(rows))
    print("protocol_id:", status["protocol_id"])
    print("rows:", len(rows))
    print("wrote:", by_run_path)
    print("wrote:", summary_path)


if __name__ == "__main__":
    main()
