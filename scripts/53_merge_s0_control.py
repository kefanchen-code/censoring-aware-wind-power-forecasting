# -*- coding: utf-8 -*-
"""Merge the S0 negative control into the main semi-synthetic result tables.

S0 has no curtailment, hence no censoring, so the censored likelihood degenerates
to the plain likelihood and B6 must be *statistically equivalent* to B4.  This is
a pre-registered placebo check, not a superiority test, and the merged tables
label it as such so the manuscript cannot silently read it as extra evidence for
the primary hypothesis.

Pure post-processing: reads ``metrics_summary.csv`` and ``comparisons.json`` from
both benchmark output directories and writes, into the main output directory:

- ``metrics_summary_with_s0_control.csv``
- ``comparisons_with_s0_control.csv``

The script refuses to run unless the S0 config declares an ``equivalence_margin``
for the B4 vs B6 pair, because without that margin the S0 run answers a different
question (absence of a significant difference is not equivalence).

Usage:
    python scripts/53_merge_s0_control.py
"""

import argparse
import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

METRIC_COLUMNS = (
    "n_runs",
    "crps_kind",
    "wis_mean_mean",
    "wis_mean_sd",
    "crps_mean_mean",
    "crps_mean_sd",
    "median_mae_mean",
    "median_mae_sd",
    "median_bias_mean",
    "median_bias_sd",
)

CONFIRMATORY = "confirmatory_semisynthetic"
NEGATIVE_CONTROL = "negative_control_no_curtailment"


def load_config(config_path):
    with (PROJECT_DIR / config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_valid_status(output_dir):
    status_path = output_dir / "benchmark_status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if not (status.get("complete") and status.get("valid_for_final_comparison")):
        raise RuntimeError("not valid for final comparison: %s" % status_path)
    return status


def check_equivalence_declared(config, reference, challenger):
    for comparison in config.get("comparisons", []):
        if (
            comparison.get("reference") == reference
            and comparison.get("challenger") == challenger
            and comparison.get("equivalence_margin") is not None
        ):
            return float(comparison["equivalence_margin"])
    raise RuntimeError(
        "the S0 config declares no equivalence_margin for %s vs %s; add one and "
        "re-run `run_pap_benchmark.py aggregate` (no retraining needed) before "
        "merging the negative control" % (reference, challenger)
    )


def read_metrics(output_dir, role, run_name):
    path = output_dir / "metrics_summary.csv"
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            row = {
                "role": role,
                "run": run_name,
                "scenario": record["scenario"],
                "model": record["model"],
            }
            for column in METRIC_COLUMNS:
                row[column] = record[column]
            rows.append(row)
    return rows


def read_comparisons(output_dir, role, run_name):
    path = output_dir / "comparisons.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for comparison in payload["comparisons"]:
        margin = comparison.get("equivalence_margin")
        ci = comparison.get("cluster_bootstrap_ci_95", [None, None])
        rows.append(
            {
                "role": role,
                "run": run_name,
                "test_type": "equivalence" if margin is not None else "superiority",
                "family": comparison.get("family"),
                "scenario": comparison.get("scenario"),
                "metric": comparison.get("metric"),
                "reference": comparison.get("reference"),
                "challenger": comparison.get("challenger"),
                "difference_reference_minus_challenger": comparison.get(
                    "difference_reference_minus_challenger"
                ),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "n_clusters": comparison.get("n_clusters"),
                "sign_flip_p_two_sided": comparison.get("sign_flip_p_two_sided"),
                "holm_adjusted_p_within_family": comparison.get(
                    "holm_adjusted_p_within_family"
                ),
                "equivalence_margin": margin,
                "equivalent_within_margin": comparison.get("equivalent_within_margin"),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("refusing to write empty table: %s" % path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-config", default="configs/pap_benchmark_semisyn.json", type=Path
    )
    parser.add_argument(
        "--control-config", default="configs/pap_benchmark_s0_control.json", type=Path
    )
    parser.add_argument("--reference", default="B4")
    parser.add_argument("--challenger", default="B6")
    args = parser.parse_args()

    main_config = load_config(args.main_config)
    control_config = load_config(args.control_config)
    main_dir = PROJECT_DIR / main_config["output_dir"]
    control_dir = PROJECT_DIR / control_config["output_dir"]

    main_status = require_valid_status(main_dir)
    control_status = require_valid_status(control_dir)
    margin = check_equivalence_declared(
        control_config, args.reference, args.challenger
    )

    metrics = read_metrics(main_dir, CONFIRMATORY, main_dir.name)
    metrics += read_metrics(control_dir, NEGATIVE_CONTROL, control_dir.name)
    comparisons = read_comparisons(main_dir, CONFIRMATORY, main_dir.name)
    comparisons += read_comparisons(control_dir, NEGATIVE_CONTROL, control_dir.name)

    metrics_path = main_dir / "metrics_summary_with_s0_control.csv"
    comparisons_path = main_dir / "comparisons_with_s0_control.csv"
    write_csv(metrics_path, metrics)
    write_csv(comparisons_path, comparisons)

    print("main protocol_id:   ", main_status["protocol_id"])
    print("control protocol_id:", control_status["protocol_id"])
    if main_status["protocol_id"] != control_status["protocol_id"]:
        print(
            "  note: protocol ids differ, so the two runs are not under one frozen"
            " protocol; report them as separate experiments"
        )
    print("metric rows:", len(metrics), "comparison rows:", len(comparisons))
    print("wrote:", metrics_path)
    print("wrote:", comparisons_path)

    print()
    print(
        "negative control, %s vs %s on S0 (equivalence margin %.4f pu):"
        % (args.reference, args.challenger, margin)
    )
    for row in comparisons:
        if row["role"] != NEGATIVE_CONTROL:
            continue
        if row["reference"] != args.reference or row["challenger"] != args.challenger:
            continue
        print(
            "  diff(ref-chal) = %+.6f  CI95 [%+.6f, %+.6f]  equivalent=%s"
            % (
                row["difference_reference_minus_challenger"],
                row["ci95_low"],
                row["ci95_high"],
                row["equivalent_within_margin"],
            )
        )

    print()
    print("confirmatory %s vs %s for contrast:" % (args.reference, args.challenger))
    for row in comparisons:
        if row["role"] != CONFIRMATORY:
            continue
        if row["reference"] != args.reference or row["challenger"] != args.challenger:
            continue
        print(
            "  %-18s diff = %+.6f  CI95 [%+.6f, %+.6f]  holm_p = %s"
            % (
                row["scenario"],
                row["difference_reference_minus_challenger"],
                row["ci95_low"],
                row["ci95_high"],
                row["holm_adjusted_p_within_family"],
            )
        )


if __name__ == "__main__":
    main()
