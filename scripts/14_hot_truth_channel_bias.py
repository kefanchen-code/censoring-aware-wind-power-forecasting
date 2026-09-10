# -*- coding: utf-8 -*-
"""Step 14: quantify systematic bias in the Hill of Towie truth channel.

The reconstructed truth channel has two intended routes: a free-generation
power curve driven by nacelle wind and a contemporaneous-neighbour route.
Archived outputs show that curtailed windows have 100% power-curve coverage
and zero neighbour coverage because no free neighbour is available during
synchronous farm-level curtailment.

Script 13 measures control contamination in the nacelle-wind input. This
script estimates the resulting bias as ``pc(v_obs) - pc(v_obs - delta_v)``
using cap-depth-specific shifts. The clean-window hold-out MAE from script 21
and this curtailed-window systematic bias describe different error sources and
are reported side by side, then compared with the forecasting effect size.

Inputs are the script 13 outputs and reproducible truth-channel Parquet files;
outputs are written under ``results/hot_truth_channel_bias/``.

Usage:
    python scripts/14_hot_truth_channel_bias.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DID_DIR = PROJECT_DIR / "results" / "hot_contamination_did"
TRUTH_DIR = PROJECT_DIR / "results" / "hot_truth_channel"
OUT_DIR = PROJECT_DIR / "results" / "hot_truth_channel_bias"

RATED = 2300.0
PC_BIN = 0.5      # 与脚本 21 的功率曲线分箱宽度一致
MIN_BIN_COUNT = 10

# 运行时由 scripts/21_hot_truth_channel.py 的可复现 manifest 覆盖。下面的
# 64.4 kW 只是完整 v2 数据上的预期近似值，用于直接调用 process() 时兜底。
REPORTED_HOLDOUT = {
    "route": "power_curve",
    "mae_kw": 64.4,
    "mae_frac_rated": 64.4 / RATED,
    "validated_on": "chronological final 30% of clean actual-route rows",
    "provenance": "results/hot_truth_channel/truth_channel_manifest.json",
}

# 第一层真实数据上的待检测效应量，同一份汇总第 26 行与预测层结果。
EFFECT_SIZE = {
    "delta_wis": 0.00383,
    "delta_wis_relative": 0.077,
    "note": "B4 versus B6 primary-score difference in the HOT real-data layer",
}


def load_recomputed_holdout():
    """Load the power-curve validation error produced by the rebuilt pipeline."""

    path = TRUTH_DIR / "truth_channel_manifest.json"
    if not path.exists():
        raise SystemExit("missing %s; run scripts/21_hot_truth_channel.py first" % path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    curve = manifest["outputs"]["truth_channel_T11_1min.parquet"]["power_curve"]
    return {
        "route": "power_curve",
        "mae_kw": float(curve["holdout_mae_kW"]),
        "mae_frac_rated": float(curve["holdout_mae_fraction_rated"]),
        "validated_on": "chronological final 30% of clean actual-route rows",
        "train_rows": int(curve["train_rows"]),
        "holdout_rows": int(curve["holdout_rows"]),
        "provenance": path.relative_to(PROJECT_DIR).as_posix(),
    }


def load_did():
    """Load measured delta-v overall, by cap depth, and by wind band."""

    headline_path = DID_DIR / "hot_did_headline.json"
    if not headline_path.exists():
        raise SystemExit(
            "missing %s; run scripts/13_hot_contamination_did.py first" % headline_path
        )
    headline = json.loads(headline_path.read_text(encoding="utf-8"))
    by_depth = pd.read_csv(DID_DIR / "hot_did_by_depth.csv")
    by_band = pd.read_csv(DID_DIR / "hot_did_by_wind_band.csv")
    return headline, by_depth, by_band


def fit_power_curve(wind, power):
    """Rebuild the script 21 binned-median power curve.

    Clean windows from the truth-channel file reproduce the curve used for PAP
    reconstruction instead of borrowing the farm-level curve from script 13.
    """

    finite = np.isfinite(wind) & np.isfinite(power)
    wind, power = wind[finite], power[finite]
    edges = np.arange(0, 30 + PC_BIN, PC_BIN)
    labels = pd.cut(wind, edges, labels=edges[:-1] + PC_BIN / 2)
    frame = pd.DataFrame({"label": labels, "power": power})
    grouped = frame.groupby("label", observed=True)["power"]
    median = grouped.median()
    counts = grouped.size()
    median = median[counts >= MIN_BIN_COUNT]
    centers = median.index.astype(float).to_numpy()
    values = median.to_numpy()

    def curve(value):
        return np.clip(
            np.interp(np.asarray(value, dtype=float), centers, values,
                      left=0.0, right=values[-1]),
            0.0,
            RATED,
        )

    return curve, centers, values


def depth_delta_lookup(by_depth):
    """Map cap-depth bins to delta-v using the script 13 boundaries."""

    edges = [0.0, 0.25, 0.4, 0.6, 0.95]
    table = by_depth.set_index("depth_bin_pu")["delta_v_ms"].to_dict()
    labels = list(by_depth["depth_bin_pu"])

    def lookup(depth):
        depth = np.asarray(depth, dtype=float)
        out = np.full(depth.shape, np.nan)
        for index, label in enumerate(labels):
            low, high = edges[index], edges[index + 1]
            mask = (depth > low) & (depth <= high)
            out[mask] = table[label]
        # 两端外推：档位≤0（指令为零的完全限功）归最深档，
        # >0.95（几乎未限功）归最浅档。
        out[np.isnan(out) & np.isfinite(depth) & (depth <= edges[0])] = table[labels[0]]
        out[np.isnan(out) & np.isfinite(depth) & (depth > edges[-1])] = table[labels[-1]]
        return out

    return lookup


def summarise_bias(label, wind, bias_kw, depth=None):
    absolute = np.abs(bias_kw)
    record = {
        "sample": label,
        "n_frames": int(bias_kw.size),
        "mean_bias_kw": float(np.mean(bias_kw)),
        "median_bias_kw": float(np.median(bias_kw)),
        "mean_abs_bias_kw": float(np.mean(absolute)),
        "p05_bias_kw": float(np.quantile(bias_kw, 0.05)),
        "p95_bias_kw": float(np.quantile(bias_kw, 0.95)),
        "mean_bias_frac_rated": float(np.mean(bias_kw) / RATED),
        "mean_abs_bias_frac_rated": float(np.mean(absolute) / RATED),
        "mean_wind_ms": float(np.mean(wind)),
    }
    if depth is not None:
        record["mean_depth_pu"] = float(np.mean(depth))
    return record


def process(path, headline, by_depth):
    """Quantify one truth-channel file and return summaries and route audit."""

    frame = pd.read_parquet(path)
    curtailed = frame["curtailed"].astype(bool).to_numpy()
    source = frame["pap_source"].astype(str)

    composition = {
        "file": path.name,
        "n_rows": int(len(frame)),
        "n_curtailed": int(curtailed.sum()),
        "pap_source_counts": {
            key: int(value) for key, value in source.value_counts().items()
        },
        "curtailed_pap_source_counts": {
            key: int(value)
            for key, value in source[curtailed].value_counts().items()
        },
    }
    curtailed_sources = composition["curtailed_pap_source_counts"]
    n_curtailed = composition["n_curtailed"]
    composition["power_curve_share_of_curtailed"] = (
        curtailed_sources.get("power_curve", 0) / max(n_curtailed, 1)
    )
    composition["neighbor_share_of_curtailed"] = (
        curtailed_sources.get("neighbor", 0) / max(n_curtailed, 1)
    )

    clean = ~curtailed & np.isfinite(frame["power_obs_kW"].to_numpy())
    curve, centers, values = fit_power_curve(
        frame["wind_ms"].to_numpy()[clean], frame["power_obs_kW"].to_numpy()[clean]
    )

    # 只在真的由功率曲线路重建的限功帧上量化——这是被污染输入唯一进入真值的路径
    target = curtailed & (source == "power_curve").to_numpy()
    wind_obs = frame["wind_ms"].to_numpy()[target]
    depth = frame["pref_kW"].to_numpy()[target] / RATED
    valid = np.isfinite(wind_obs) & np.isfinite(depth)
    wind_obs, depth = wind_obs[valid], depth[valid]
    if wind_obs.size == 0:
        raise SystemExit("no reconstructable curtailed frame in %s" % path.name)

    lookup = depth_delta_lookup(by_depth)
    delta_by_depth = lookup(depth)
    covered = np.isfinite(delta_by_depth)
    composition["n_power_curve_curtailed"] = int(wind_obs.size)
    composition["n_excluded_uncovered_depth"] = int((~covered).sum())
    wind_obs, depth, delta_by_depth = (
        wind_obs[covered],
        depth[covered],
        delta_by_depth[covered],
    )
    delta_overall = float(headline["delta_v_ms"])

    # 偏置 = 用被污染读数重建 − 用干净读数重建（精确差分，非线性化）
    bias_depth = curve(wind_obs) - curve(wind_obs - delta_by_depth)
    bias_overall = curve(wind_obs) - curve(wind_obs - delta_overall)
    slope = np.gradient(values, centers)
    local_slope = np.interp(wind_obs, centers, slope)
    bias_linear = local_slope * delta_by_depth

    summary_rows = [
        summarise_bias("%s / depth-specific Δv" % path.name, wind_obs, bias_depth, depth),
        summarise_bias("%s / overall Δv" % path.name, wind_obs, bias_overall, depth),
        summarise_bias("%s / linearised dP/dv·Δv" % path.name, wind_obs, bias_linear, depth),
    ]

    depth_rows = []
    edges = [0.0, 0.25, 0.4, 0.6, 0.95]
    labels = list(by_depth["depth_bin_pu"])
    for index, label in enumerate(labels):
        low, high = edges[index], edges[index + 1]
        mask = (depth > low) & (depth <= high)
        if not mask.any():
            continue
        record = summarise_bias(
            "%s / %s" % (path.name, label), wind_obs[mask], bias_depth[mask], depth[mask]
        )
        record["depth_bin_pu"] = label
        record["delta_v_ms"] = float(delta_by_depth[mask][0])
        record["ratio_to_reported_mae"] = (
            record["mean_abs_bias_kw"] / REPORTED_HOLDOUT["mae_kw"]
        )
        depth_rows.append(record)

    for row in summary_rows:
        row["ratio_to_reported_mae"] = (
            row["mean_abs_bias_kw"] / REPORTED_HOLDOUT["mae_kw"]
        )
    return summary_rows, depth_rows, composition


def main():
    global REPORTED_HOLDOUT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTED_HOLDOUT = load_recomputed_holdout()
    headline, by_depth, by_band = load_did()

    print("=== Delta-v measured by script 13 (primary specification) ===")
    print(
        "Overall delta-v = %+.3f m/s  95%%CI [%+.3f, %+.3f]  relative %+.1f%%"
        % (
            headline["delta_v_ms"],
            headline["delta_v_ci_low_ms"],
            headline["delta_v_ci_high_ms"],
            100 * headline["relative_delta"],
        )
    )
    print("By cap depth (the sign changes with depth, so lookup is frame-specific):")
    for _, row in by_depth.iterrows():
        print(
            "   %-10s Δv = %+.3f m/s (n_treat=%d)"
            % (row["depth_bin_pu"], row["delta_v_ms"], row["n_treated_frames"])
        )

    summary_rows, depth_rows, compositions = [], [], []
    for name in ("truth_channel_T11_10min.parquet", "truth_channel_T11_1min.parquet"):
        path = TRUTH_DIR / name
        if not path.exists():
            print("\n[skip] missing %s" % path)
            continue
        summaries, depths, composition = process(path, headline, by_depth)
        summary_rows.extend(summaries)
        depth_rows.extend(depths)
        compositions.append(composition)

    if not summary_rows:
        raise SystemExit("no truth-channel parquet was available")

    print("\n" + "=" * 78)
    print("=== Truth-channel reconstruction routes in curtailed windows ===")
    print("=" * 78)
    for composition in compositions:
        print(
            "%-34s total=%6d curtailed=%5d  power-curve share=%.1f%%  neighbour share=%.1f%%"
            % (
                composition["file"],
                composition["n_rows"],
                composition["n_curtailed"],
                100 * composition["power_curve_share_of_curtailed"],
                100 * composition["neighbor_share_of_curtailed"],
            )
        )
    print(
        "\nThe neighbour route has zero coverage in curtailed windows. Their reconstructed\n"
        "truth therefore inherits the contaminated nacelle-wind channel without an independent cross-check."
    )

    print("\n" + "=" * 78)
    print("=== Systematic PAP-truth bias in curtailed windows ===")
    print("=" * 78)
    print(
        "%-46s %8s %10s %10s %8s"
        % ("sample / delta-v basis", "n", "mean kW", "|mean| kW", "% rated")
    )
    for row in summary_rows:
        print(
            "%-46s %8d %+10.1f %10.1f %+8.2f"
            % (
                row["sample"],
                row["n_frames"],
                row["mean_bias_kw"],
                row["mean_abs_bias_kw"],
                100 * row["mean_bias_frac_rated"],
            )
        )
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "hot_truth_bias_summary.csv", index=False, encoding="utf-8-sig"
    )

    print("\n--- By cap depth (bias sign changes with depth) ---")
    print(
        "%-40s %8s %9s %10s %10s %8s"
        % ("sample / depth", "n", "delta-v", "mean kW", "|mean| kW", "vs MAE")
    )
    for row in depth_rows:
        print(
            "%-40s %8d %+9.3f %+10.1f %10.1f %8.2fx"
            % (
                row["sample"],
                row["n_frames"],
                row["delta_v_ms"],
                row["mean_bias_kw"],
                row["mean_abs_bias_kw"],
                row["ratio_to_reported_mae"],
            )
        )
    pd.DataFrame(depth_rows).to_csv(
        OUT_DIR / "hot_truth_bias_by_depth.csv", index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 78)
    print("=== Comparison with hold-out MAE (distinct error sources, not subtractive) ===")
    print("=" * 78)
    primary = summary_rows[0]
    print(
        "Reported hold-out MAE (power-curve route): %.0f kW = %.1f%% rated; validation set = %s"
        % (
            REPORTED_HOLDOUT["mae_kw"],
            100 * REPORTED_HOLDOUT["mae_frac_rated"],
            REPORTED_HOLDOUT["validated_on"],
        )
    )
    print(
        "Systematic bias quantified here (curtailed windows, depth-specific delta-v): "
        "mean %+.1f kW, |mean| %.1f kW = %.2f%% rated"
        % (
            primary["mean_bias_kw"],
            primary["mean_abs_bias_kw"],
            100 * primary["mean_abs_bias_frac_rated"],
        )
    )
    worst = max(depth_rows, key=lambda row: row["mean_abs_bias_kw"])
    print(
        "Deepest cap bin %s: |mean| %.1f kW = %.2f%% rated = %.2f times the reported MAE"
        % (
            worst["depth_bin_pu"],
            worst["mean_abs_bias_kw"],
            100 * worst["mean_abs_bias_frac_rated"],
            worst["ratio_to_reported_mae"],
        )
    )
    print(
        "\nThe hold-out MAE is evaluated only on clean windows, which are uncontaminated by\n"
        "definition. It therefore measures random error and does not bound the systematic bias above."
    )
    print(
        "\nSignal-to-noise comparison: delta-WIS = %.5f (relative %.1f%%), while systematic\n"
        "truth bias in curtailed windows reaches %.2f%% rated. If label bias is comparable\n"
        "to or larger than the effect, a ranking reversal cannot identify model quality."
        % (
            EFFECT_SIZE["delta_wis"],
            100 * EFFECT_SIZE["delta_wis_relative"],
            100 * primary["mean_abs_bias_frac_rated"],
        )
    )

    payload = {
        "did_source": headline,
        "reported_holdout": REPORTED_HOLDOUT,
        "effect_size": EFFECT_SIZE,
        "route_composition": compositions,
        "bias_summary": summary_rows,
        "bias_by_depth": depth_rows,
        "provenance": "scripts/14_hot_truth_channel_bias.py",
    }
    (OUT_DIR / "hot_truth_bias.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved to %s" % OUT_DIR)


if __name__ == "__main__":
    main()
