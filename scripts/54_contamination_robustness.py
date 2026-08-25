# -*- coding: utf-8 -*-
"""Cross-run robustness summary for the input-contamination variants.

The runner compares models only inside a single run, so the question "does the
B6 advantage survive a contaminated wind channel?" needs a separate pass over
several output directories.  This script pairs the clean run
(``results/pap_benchmark_semisyn``, the 0x control) with every contaminated run
that already exists and reports:

- ``contamination_wis_degradation.csv``: per model WIS(contam) - WIS(clean),
  split by censored / clean windows;
- ``contamination_advantage.csv``: the paired cluster inference for each
  configured comparison under each contamination intensity, i.e. whether the
  advantage is retained;
- ``contamination_mechanism.csv``: the Bracher component deltas collected from
  every run (requires ``52_wis_decomposition.py`` to have been run first), which
  is where the mechanism prediction is checked -- lifting the input wind should
  shift part of the gain into the overprediction penalty;
- ``contamination_invariance.json``: the bitwise self-check on the wind-free
  baselines.  ``B0_persistence`` and ``B0_climatology`` never touch the wind
  channel, so their forecasts must be identical to the clean run.  Any
  difference means the injection leaked outside the wind column.

Inference reuses the runner's cluster bootstrap and sign-flip settings so the
clean-run numbers reproduce ``comparisons.json`` exactly (verified against it
when present).  These results are declared robustness analyses: they are *not*
part of the confirmatory hypothesis family and carry no multiplicity
correction.  The primary hypothesis remains B4 versus B6 on the clean run.

Usage:
    python scripts/54_contamination_robustness.py
    python scripts/54_contamination_robustness.py --contaminated-config \
        configs/pap_benchmark_semisyn_contam100.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from censored_wind_power.benchmark.protocol import stable_json_hash
from censored_wind_power.benchmark.scoring import cluster_inference

WINDOW_TYPES = ("all", "censored", "clean")
WIND_FREE_MODELS = ("B0_persistence", "B0_climatology")
DEFAULT_CONTAMINATED = (
    "configs/pap_benchmark_semisyn_contam100.json",
    "configs/pap_benchmark_semisyn_contam050.json",
    "configs/pap_benchmark_semisyn_contam150.json",
    "configs/pap_benchmark_semisyn_contam_sched.json",
)


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("refusing to write empty table: %s" % path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_run_config(project_dir, config_path, allow_incomplete):
    """Return (config, output_dir, run_name, status) for one benchmark config."""

    config = read_json(project_dir / config_path)
    output_dir = project_dir / config["output_dir"]
    status_path = output_dir / "benchmark_status.json"
    if not status_path.exists():
        return None
    status = read_json(status_path)
    complete = bool(status.get("complete") and status.get("valid_for_final_comparison"))
    if not complete and not allow_incomplete:
        raise RuntimeError(
            "%s is not valid for final comparison; pass --allow-incomplete to"
            " summarize it anyway" % status_path
        )
    contamination = config.get("contamination", {})
    return {
        "config": config,
        "config_path": str(config_path),
        "output_dir": output_dir,
        "run": output_dir.name,
        "status": status,
        "complete": complete,
        "contamination": contamination,
        "intensity": (
            float(contamination.get("intensity_factor", 1.0))
            if contamination.get("enabled", False)
            else 0.0
        ),
        "mode": contamination.get("mode", "none") if contamination.get("enabled") else "none",
    }


def load_model(output_dir, scenario, model):
    """Load per-window arrays for every seed of one scenario/model."""

    model_dir = output_dir / "artifacts" / scenario / model
    if not model_dir.exists():
        return None
    seeds = []
    wis = []
    reference = None
    for run_dir in sorted(model_dir.iterdir()):
        forecast_path = run_dir / "forecast.npz"
        artifact_path = run_dir / "artifact.json"
        if not forecast_path.exists() or not artifact_path.exists():
            continue
        artifact = read_json(artifact_path)
        with np.load(forecast_path) as arrays:
            wis.append(arrays["wis"].astype(np.float64))
            if reference is None:
                reference = {
                    "segment_id": arrays["segment_id"].copy(),
                    "censored": arrays["target_censored"].astype(np.float64) >= 0.5,
                    "target": arrays["target"].copy(),
                    "target_wind_ms": arrays["target_wind_ms"].copy(),
                    "alignment_hash": artifact["alignment_hash"],
                }
        seeds.append(artifact.get("seed"))
    if not wis:
        return None
    entry = dict(reference)
    entry["seeds"] = seeds
    entry["wis"] = np.stack(wis).mean(axis=0)
    return entry


def masks_for(entry):
    censored = entry["censored"]
    return {
        "all": np.ones_like(censored, dtype=bool),
        "censored": censored,
        "clean": ~censored,
    }


def assert_alignment(clean, contaminated, scenario, model):
    """Cross-run pairing is only valid if the evaluation windows are identical."""

    if clean["alignment_hash"] != contaminated["alignment_hash"]:
        raise RuntimeError(
            "alignment hash differs between runs for %s/%s: contamination must not"
            " change the evaluation windows" % (scenario, model)
        )
    if not np.array_equal(clean["target"], contaminated["target"]):
        raise RuntimeError("latent truth differs between runs for %s/%s" % (scenario, model))
    if not np.array_equal(clean["target_wind_ms"], contaminated["target_wind_ms"]):
        raise RuntimeError(
            "target_wind_ms differs between runs for %s/%s: the stratification"
            " field must stay on the clean readings" % (scenario, model)
        )


def degradation_rows(run, scenario, model, clean, contaminated):
    rows = []
    masks = masks_for(clean)
    for window_type in WINDOW_TYPES:
        mask = masks[window_type]
        if not mask.any():
            continue
        clean_wis = float(clean["wis"][mask].mean())
        contaminated_wis = float(contaminated["wis"][mask].mean())
        rows.append(
            {
                "run": run["run"],
                "contamination_mode": run["mode"],
                "intensity_factor": run["intensity"],
                "scenario": scenario,
                "model": model,
                "window_type": window_type,
                "n_windows": int(mask.sum()),
                "wis_clean": clean_wis,
                "wis_contaminated": contaminated_wis,
                "wis_degradation": contaminated_wis - clean_wis,
                "wis_degradation_relative": (
                    (contaminated_wis - clean_wis) / clean_wis
                    if clean_wis > 0
                    else float("nan")
                ),
            }
        )
    return rows


def inference_seed(base_seed, scenario, comparison_index, window_type):
    """Reproduce the runner's seed rule; window subsets get their own stream."""

    seed = base_seed ^ int(stable_json_hash([scenario, comparison_index])[:8], 16)
    if window_type != "all":
        seed ^= int(stable_json_hash([window_type])[:8], 16)
    return seed


def advantage_rows(run, scenario, specification, comparison_index, reference, challenger, inference_config):
    rows = []
    masks = masks_for(reference)
    for window_type in WINDOW_TYPES:
        mask = masks[window_type]
        if int(mask.sum()) < 2:
            continue
        result = cluster_inference(
            reference["wis"][mask],
            challenger["wis"][mask],
            reference["segment_id"][mask],
            inference_seed(
                int(inference_config["seed"]),
                scenario,
                comparison_index,
                window_type,
            ),
            int(inference_config["cluster_bootstrap_draws"]),
            int(inference_config["sign_flip_draws"]),
            int(inference_config["exhaustive_max_clusters"]),
        )
        ci_low, ci_high = result["cluster_bootstrap_ci_95"]
        rows.append(
            {
                "run": run["run"],
                "contamination_mode": run["mode"],
                "intensity_factor": run["intensity"],
                "inference_role": "robustness_no_multiplicity_correction",
                "family": specification.get(
                    "family",
                    "%s_vs_%s_wis" % (specification["challenger"], specification["reference"]),
                ),
                "scenario": scenario,
                "reference": specification["reference"],
                "challenger": specification["challenger"],
                "window_type": window_type,
                "n_windows": int(mask.sum()),
                "n_clusters": result["n_clusters"],
                "difference_reference_minus_challenger": result[
                    "difference_reference_minus_challenger"
                ],
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "sign_flip_p_two_sided": result["sign_flip_p_two_sided"],
                "advantage_retained": bool(float(ci_low) > 0.0),
            }
        )
    return rows


def check_wind_free_invariance(clean_dir, contaminated_dir, scenarios):
    """B0 baselines ignore wind, so their forecasts must be bit-identical."""

    checks = []
    for scenario in scenarios:
        for model in WIND_FREE_MODELS:
            clean_model = clean_dir / "artifacts" / scenario / model
            if not clean_model.exists():
                continue
            for run_dir in sorted(clean_model.iterdir()):
                clean_path = run_dir / "forecast.npz"
                contaminated_path = (
                    contaminated_dir
                    / "artifacts"
                    / scenario
                    / model
                    / run_dir.name
                    / "forecast.npz"
                )
                if not clean_path.exists() or not contaminated_path.exists():
                    continue
                with np.load(clean_path) as left, np.load(contaminated_path) as right:
                    identical = bool(
                        np.array_equal(left["quantiles"], right["quantiles"])
                    )
                    max_gap = float(
                        np.max(
                            np.abs(
                                left["quantiles"].astype(np.float64)
                                - right["quantiles"].astype(np.float64)
                            )
                        )
                    )
                checks.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "run": run_dir.name,
                        "bitwise_identical": identical,
                        "max_abs_quantile_gap": max_gap,
                    }
                )
    return checks


def collect_mechanism(runs):
    rows = []
    for run in runs:
        path = run["output_dir"] / "wis_components_contrast.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["contamination_mode"] = run["mode"]
                row["intensity_factor"] = run["intensity"]
                rows.append(row)
    return rows


def summarize(project_dir, clean_config_path, contaminated_config_paths, output_dir, allow_incomplete):
    clean_run = load_run_config(project_dir, clean_config_path, allow_incomplete)
    if clean_run is None:
        raise RuntimeError("the clean control run has no benchmark_status.json")
    if clean_run["contamination"].get("enabled", False):
        raise RuntimeError("the clean control config must not enable contamination")
    config = clean_run["config"]
    scenarios = config["scenarios"]
    models = config["models"]
    inference_config = config["inference"]

    contaminated_runs = []
    for path in contaminated_config_paths:
        run = load_run_config(project_dir, path, allow_incomplete)
        if run is None:
            print("skipping (not run yet):", path)
            continue
        if not run["contamination"].get("enabled", False):
            raise RuntimeError("%s does not enable contamination" % path)
        contaminated_runs.append(run)
    if not contaminated_runs:
        raise RuntimeError("no contaminated run is available yet")

    degradation = []
    advantage = []
    invariance = {}
    for run in contaminated_runs:
        invariance[run["run"]] = check_wind_free_invariance(
            clean_run["output_dir"], run["output_dir"], scenarios
        )
        for scenario in scenarios:
            clean_models = {}
            contaminated_models = {}
            for model in models:
                clean_entry = load_model(clean_run["output_dir"], scenario, model)
                contaminated_entry = load_model(run["output_dir"], scenario, model)
                if clean_entry is None or contaminated_entry is None:
                    continue
                assert_alignment(clean_entry, contaminated_entry, scenario, model)
                clean_models[model] = clean_entry
                contaminated_models[model] = contaminated_entry
                degradation.extend(
                    degradation_rows(run, scenario, model, clean_entry, contaminated_entry)
                )
            for index, specification in enumerate(config.get("comparisons", [])):
                if str(specification["metric"]) != "wis":
                    continue
                reference = specification["reference"]
                challenger = specification["challenger"]
                if reference not in clean_models or challenger not in clean_models:
                    continue
                advantage.extend(
                    advantage_rows(
                        clean_run,
                        scenario,
                        specification,
                        index,
                        clean_models[reference],
                        clean_models[challenger],
                        inference_config,
                    )
                )
                advantage.extend(
                    advantage_rows(
                        run,
                        scenario,
                        specification,
                        index,
                        contaminated_models[reference],
                        contaminated_models[challenger],
                        inference_config,
                    )
                )

    # The clean rows are recomputed once per contaminated run; keep one copy.
    seen = set()
    unique_advantage = []
    for row in advantage:
        key = (
            row["run"],
            row["scenario"],
            row["family"],
            row["reference"],
            row["challenger"],
            row["window_type"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_advantage.append(row)

    mechanism = collect_mechanism([clean_run] + contaminated_runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "contamination_wis_degradation.csv", degradation)
    write_csv(output_dir / "contamination_advantage.csv", unique_advantage)
    if mechanism:
        write_csv(output_dir / "contamination_mechanism.csv", mechanism)
    with (output_dir / "contamination_invariance.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "claim": (
                    "B0_persistence and B0_climatology never consume the wind"
                    " channel, so contamination must leave their forecasts"
                    " bitwise unchanged"
                ),
                "clean_run": clean_run["run"],
                "checks": invariance,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return clean_run, contaminated_runs, degradation, unique_advantage, invariance


def verify_against_comparisons(clean_run, advantage):
    """The clean 'all windows' rows must reproduce the run's own comparisons.json."""

    path = clean_run["output_dir"] / "comparisons.json"
    if not path.exists():
        return
    stored = {
        (item["scenario"], item["reference"], item["challenger"]): item
        for item in read_json(path)["comparisons"]
        if item["metric"] == "wis"
    }
    worst = 0.0
    for row in advantage:
        if row["run"] != clean_run["run"] or row["window_type"] != "all":
            continue
        item = stored.get((row["scenario"], row["reference"], row["challenger"]))
        if item is None:
            continue
        worst = max(
            worst,
            abs(
                item["difference_reference_minus_challenger"]
                - row["difference_reference_minus_challenger"]
            ),
        )
    print("  clean rows versus comparisons.json, max abs gap: %.3e" % worst)
    if worst > 1e-12:
        raise RuntimeError(
            "recomputed clean differences do not reproduce comparisons.json"
        )


def report(clean_run, contaminated_runs, degradation, advantage, invariance):
    print()
    print("wind-free baseline invariance (B0_persistence / B0_climatology):")
    for run_name, checks in invariance.items():
        failures = [check for check in checks if not check["bitwise_identical"]]
        print(
            "  %-42s %3d checks, %d differ" % (run_name, len(checks), len(failures))
        )
        for failure in failures:
            print(
                "    DIFFERS %s/%s/%s max gap %.3e"
                % (
                    failure["scenario"],
                    failure["model"],
                    failure["run"],
                    failure["max_abs_quantile_gap"],
                )
            )

    print()
    print("B4 -> B6 advantage under contamination (positive means B6 wins):")
    print(
        "  %-42s %-18s %-9s %10s %10s %10s %8s"
        % ("run", "scenario", "windows", "diff", "ci_low", "ci_high", "kept")
    )
    for row in advantage:
        if row["reference"] != "B4" or row["challenger"] != "B6":
            continue
        print(
            "  %-42s %-18s %-9s %10.5f %10.5f %10.5f %8s"
            % (
                row["run"],
                row["scenario"],
                row["window_type"],
                row["difference_reference_minus_challenger"],
                row["ci_low"],
                row["ci_high"],
                row["advantage_retained"],
            )
        )

    print()
    print("WIS degradation by model (all windows, mean over scenarios):")
    grouped = {}
    for row in degradation:
        if row["window_type"] != "all":
            continue
        grouped.setdefault((row["run"], row["model"]), []).append(
            row["wis_degradation"]
        )
    for (run_name, model), values in sorted(grouped.items()):
        print(
            "  %-42s %-16s %+10.5f" % (run_name, model, sum(values) / len(values))
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-config",
        type=Path,
        default=Path("configs/pap_benchmark_semisyn.json"),
        help="the 0x control config",
    )
    parser.add_argument(
        "--contaminated-config",
        nargs="+",
        type=Path,
        default=[Path(item) for item in DEFAULT_CONTAMINATED],
        help="contaminated configs; those without results are skipped",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/contamination_robustness"),
        help="directory for the cross-run tables",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="summarize runs whose benchmark_status.json is not final",
    )
    args = parser.parse_args()

    clean_run, contaminated_runs, degradation, advantage, invariance = summarize(
        PROJECT_DIR,
        args.clean_config,
        args.contaminated_config,
        PROJECT_DIR / args.output_dir,
        args.allow_incomplete,
    )
    print("clean control:", clean_run["run"])
    verify_against_comparisons(clean_run, advantage)
    print("contaminated runs:", ", ".join(run["run"] for run in contaminated_runs))
    report(clean_run, contaminated_runs, degradation, advantage, invariance)
    print()
    print("wrote tables into:", PROJECT_DIR / args.output_dir)


if __name__ == "__main__":
    main()
