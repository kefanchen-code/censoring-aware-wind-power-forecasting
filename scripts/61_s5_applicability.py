# -*- coding: utf-8 -*-
"""S5 applicability-domain analysis (pillar 4 of the revision plan).

Stratifies free-sample support by wind-speed bin for every semi-synthetic
scenario and for the real Altahullion T11 record, then classifies each
censoring mechanism into three tiers:

  tier 1  independent censoring          (no censoring / mask independent of
                                          outcome and covariates)
  tier 2  conditionally ignorable        (known cap threshold, mask independent
                                          of wind; free support above the cap
                                          remains positive in every wind bin)
  tier 3  positivity violation           (mask triggered by wind; free support
                                          collapses exactly where potential
                                          power is largest)

Real anchors: Altahullion free/curtailed support recomputed from the raw
1-min record with the canonical audit classification rules, and the Hill of
Towie mechanism composition (external command share, synchronous share) from
results/hot_truth_channel_bias/hot_truth_bias.json.

Outputs (results/s5_applicability/):
  s5_applicability.json   full numerics, tier classification, anchors
  support_by_wind_bin.csv long table of per-bin free support
  support_heatmap.{png,svg}  wind-stratified free-support heatmap
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "altahullion_audit"
HOT_BIAS_PATH = ROOT / "results" / "hot_truth_channel_bias" / "hot_truth_bias.json"
OUT_DIR = ROOT / "results" / "s5_applicability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCENARIOS = [
    "S0_no_curtailment",
    "S1_fixed_50pct",
    "S2_fixed_70pct",
    "S3_fixed_85pct",
    "S4_random_uniform",
    "S5_wind_triggered",
]
BIN_WIDTH_MS = 1.0
MIN_BIN_COUNT = 50          # bins with fewer records are reported, not judged
POSITIVITY_FLOOR = 0.05     # free support below this counts as collapsed
S5_WIND_THRESH = 12.0       # must match scripts/10_altahullion_audit.py

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


def load_audit_module():
    """Reuse the canonical classification rules without duplicating them."""

    spec = importlib.util.spec_from_file_location(
        "altahullion_audit", ROOT / "scripts" / "10_altahullion_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wind_bin_edges(wind: pd.Series) -> np.ndarray:
    upper = float(np.nanmax(wind)) + BIN_WIDTH_MS
    return np.arange(0.0, upper + BIN_WIDTH_MS, BIN_WIDTH_MS)


def bin_support(
    wind: np.ndarray, free_mask: np.ndarray, edges: np.ndarray
) -> list[dict]:
    """Per-bin record counts and free-sample support rate."""

    bin_index = np.clip(
        np.searchsorted(edges, wind, side="right") - 1, 0, len(edges) - 2
    )
    rows = []
    for position in range(len(edges) - 1):
        in_bin = bin_index == position
        n_total = int(in_bin.sum())
        n_free = int((in_bin & free_mask).sum())
        rows.append(
            {
                "bin_left_ms": float(edges[position]),
                "bin_right_ms": float(edges[position + 1]),
                "n_records": n_total,
                "n_free": n_free,
                "free_support": (n_free / n_total) if n_total else float("nan"),
            }
        )
    return rows


def scenario_summary(frame: pd.DataFrame, edges: np.ndarray) -> dict:
    """Wind-stratified support plus mechanism diagnostics for one scenario."""

    wind = frame["wind"].to_numpy()
    censored = frame["is_censored"].to_numpy()
    policy_active = frame["policy_active"].to_numpy()

    rows = bin_support(wind, ~censored, edges)
    judged = [r for r in rows if r["n_records"] >= MIN_BIN_COUNT]
    min_support = min((r["free_support"] for r in judged), default=float("nan"))
    n_zero_bins = int(sum(1 for r in judged if r["free_support"] == 0.0))

    high_wind = wind >= S5_WIND_THRESH
    free_support_high_wind = float(
        np.mean(~censored[high_wind]) if high_wind.any() else float("nan")
    )
    if policy_active.any() and (~policy_active).any():
        policy_wind_corr = float(
            np.corrcoef(policy_active.astype(float), wind)[0, 1]
        )
    else:
        policy_wind_corr = 0.0

    return {
        "n_records": int(len(frame)),
        "binding_rate": float(censored.mean()),
        "policy_coverage": float(policy_active.mean()),
        "policy_wind_correlation": policy_wind_corr,
        "free_support_wind_ge_12ms": free_support_high_wind,
        "min_bin_free_support": min_support,
        "n_judged_bins_with_zero_support": n_zero_bins,
        "bins": rows,
    }


def classify_tier(summary: dict) -> tuple[str, str]:
    """Rule-based tier from the computed evidence (no hard-coded labels)."""

    if summary["binding_rate"] == 0.0:
        return (
            "tier1_independent_censoring",
            "no censoring present; every record is a free observation",
        )
    wind_coupled = abs(summary["policy_wind_correlation"]) >= 0.30
    support_collapsed = (
        summary["min_bin_free_support"] < POSITIVITY_FLOOR
        or summary["n_judged_bins_with_zero_support"] > 0
    )
    if wind_coupled or support_collapsed:
        return (
            "tier3_positivity_violation",
            "mask coupled to wind speed or judged bins lose free support "
            "entirely; the censored tail cannot be identified from free "
            "observations in the affected strata",
        )
    return (
        "tier2_conditionally_ignorable",
        "mask independent of wind and free support stays positive in every "
        "judged bin; the known cap threshold makes the right-censoring "
        "likelihood identifiable",
    )


def real_alta2_anchor(audit, edges: np.ndarray) -> dict:
    """Free/curtailed support of the real record under audit rules."""

    frame = audit.load_frame()
    frame, _ = audit.classify_observations(frame)
    active = frame["active"]
    free = frame["label"].eq("U") & active
    curtailed = frame["label"].eq("R") & active
    wind = frame["wind"].to_numpy()

    # Support rate among the two comparable classes (free vs binding-curtailed).
    comparable = free.to_numpy() | curtailed.to_numpy()
    rows = bin_support(
        wind[comparable], free.to_numpy()[comparable], edges
    )
    high_wind = wind >= S5_WIND_THRESH
    share_curtail_high_wind = float(curtailed.to_numpy()[high_wind].sum() / max(curtailed.sum(), 1))
    share_free_high_wind = float(free.to_numpy()[high_wind].sum() / max(free.sum(), 1))

    judged = [r for r in rows if r["n_records"] >= MIN_BIN_COUNT]
    return {
        "dataset": "Altahullion T11 real record (1-min, audit U/R classes)",
        "n_free": int(free.sum()),
        "n_curtailed": int(curtailed.sum()),
        "overall_free_support": float(free.sum() / max(free.sum() + curtailed.sum(), 1)),
        "curtailment_share_at_wind_ge_12ms": share_curtail_high_wind,
        "free_share_at_wind_ge_12ms": share_free_high_wind,
        "mean_wind_curtailed_ms": float(frame.loc[curtailed, "wind"].mean()),
        "mean_wind_free_ms": float(frame.loc[free, "wind"].mean()),
        "min_bin_free_support": min(
            (r["free_support"] for r in judged), default=float("nan")
        ),
        "bins": rows,
    }


def hot_anchor() -> dict:
    payload = json.loads(HOT_BIAS_PATH.read_text(encoding="utf-8"))
    source = payload["did_source"]
    return {
        "dataset": "Hill of Towie 2026 (10-min, 21 turbines, DID audit)",
        "external_command_share_of_treated": source[
            "external_command_share_of_treated"
        ],
        "synchronous_curtailment_share": source["synchronous_curtailment_share"],
        "provenance": "results/hot_truth_channel_bias/hot_truth_bias.json",
    }


def render_heatmap(matrix, row_labels, edges, out_stem: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 2.8))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        interpolation="nearest",
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    tick_positions = centers[::2]
    axis.set_xticks(range(len(centers))[::2])
    axis.set_xticklabels(["%g" % centers[2 * i] for i in range(len(tick_positions))])
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels)
    threshold_position = S5_WIND_THRESH - edges[0] - 0.5
    axis.axvline(threshold_position, color="white", lw=0.8, ls="--")
    axis.text(
        threshold_position + 0.4,
        len(row_labels) - 0.35,
        "S5 trigger 12 m/s",
        color="white",
        fontsize=6.5,
    )
    axis.set_xlabel("wind speed (m/s)")
    axis.set_ylabel("")
    figure.colorbar(image, ax=axis, label="free-sample support", shrink=0.9)
    figure.tight_layout()
    figure.savefig(out_stem.with_suffix(".png"), dpi=300)
    figure.savefig(out_stem.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    scenario_frames = {}
    wind_max = 0.0
    for name in SCENARIOS:
        frame = pd.read_parquet(AUDIT_DIR / ("synthetic_%s.parquet" % name))
        scenario_frames[name] = frame
        wind_max = max(wind_max, float(frame["wind"].max()))

    audit = load_audit_module()
    real_frame = audit.load_frame()
    real_frame, _ = audit.classify_observations(real_frame)
    wind_max = max(wind_max, float(real_frame["wind"].max()))
    edges = wind_bin_edges(pd.Series([wind_max]))

    results = {"scenarios": {}, "wind_bin_width_ms": BIN_WIDTH_MS}
    for name in SCENARIOS:
        summary = scenario_summary(scenario_frames[name], edges)
        tier, rationale = classify_tier(summary)
        summary["tier"] = tier
        summary["tier_rationale"] = rationale
        results["scenarios"][name] = summary

    results["real_alta2"] = real_alta2_anchor(audit, edges)
    results["hot_mechanism_anchor"] = hot_anchor()
    results["interpretation"] = {
        "s5_role": (
            "S5 is retained as the mechanism-form scenario: it demonstrates "
            "what wind-coupled informative censoring does to estimators, not "
            "the dominant real-farm curtailment form."
        ),
        "real_farm_evidence": (
            "HOT curtailment is dominated by external dispatch commands "
            "(87.3% of treated frames) with only 2.0% synchronous; the real "
            "regime is informative censoring by dispatch, which the "
            "likelihood-based models target."
        ),
    }

    with (OUT_DIR / "s5_applicability.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    long_rows = []
    for name in SCENARIOS:
        for row in results["scenarios"][name]["bins"]:
            long_rows.append({"dataset": name, **row})
    for row in results["real_alta2"]["bins"]:
        long_rows.append({"dataset": "ALTA2_real", **row})
    pd.DataFrame(long_rows).to_csv(
        OUT_DIR / "support_by_wind_bin.csv", index=False
    )

    # Heatmap over a common truncated wind range for readability.
    n_bins = min(
        len(results["scenarios"]["S0_no_curtailment"]["bins"]),
        len(results["real_alta2"]["bins"]),
    )
    keep = 0
    for position in range(n_bins):
        if results["scenarios"]["S0_no_curtailment"]["bins"][position]["n_records"] >= MIN_BIN_COUNT:
            keep = position + 1
    row_labels = []
    matrix_rows = []
    for name in SCENARIOS:
        row_labels.append(name.replace("_", " "))
        matrix_rows.append(
            [
                r["free_support"] if r["n_records"] else np.nan
                for r in results["scenarios"][name]["bins"][:keep]
            ]
        )
    row_labels.append("ALTA2 real (U vs R)")
    matrix_rows.append(
        [
            r["free_support"] if r["n_records"] else np.nan
            for r in results["real_alta2"]["bins"][:keep]
        ]
    )
    render_heatmap(
        np.asarray(matrix_rows, dtype=float),
        row_labels,
        edges[: keep + 1],
        OUT_DIR / "support_heatmap",
    )

    print(json.dumps(
        {
            name: {
                "tier": results["scenarios"][name]["tier"],
                "policy_wind_correlation": results["scenarios"][name][
                    "policy_wind_correlation"
                ],
                "min_bin_free_support": results["scenarios"][name][
                    "min_bin_free_support"
                ],
                "free_support_wind_ge_12ms": results["scenarios"][name][
                    "free_support_wind_ge_12ms"
                ],
            }
            for name in SCENARIOS
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("real ALTA2:", {
        key: results["real_alta2"][key]
        for key in (
            "overall_free_support",
            "curtailment_share_at_wind_ge_12ms",
            "mean_wind_curtailed_ms",
            "mean_wind_free_ms",
        )
    })
    print("output:", OUT_DIR)


if __name__ == "__main__":
    main()
