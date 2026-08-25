# -*- coding: utf-8 -*-
"""Step 68: A2 分箱敏感性三格对比表（bins=53/106/212）。

从三套结果目录读取 B4/B6/ORACLE 的 WIS/CRPS 汇总（各 3 种子均值与标准差），
拼成 场景×模型×分箱数 的敏感性对比表，用于支撑"分箱宽度不改变结论"的稳健性论证。

数据源：
- results/pap_benchmark_semisyn_bins53/   （n_bins=53）
- results/pap_benchmark_semisyn/          （n_bins=106，主协议，仅取 B4/B6/ORACLE）
- results/pap_benchmark_semisyn_bins212/  （n_bins=212）

输出：results/bins_sensitivity/bins_sensitivity_table.csv + bins_sensitivity.json
用法：python scripts/68_bins_sensitivity_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "results" / "bins_sensitivity"

SOURCES = [
    (53, "results/pap_benchmark_semisyn_bins53"),
    (106, "results/pap_benchmark_semisyn"),
    (212, "results/pap_benchmark_semisyn_bins212"),
]
MODELS = ["B4", "B6", "ORACLE"]
SCENARIOS = [
    "S1_fixed_50pct",
    "S2_fixed_70pct",
    "S3_fixed_85pct",
    "S4_random_uniform",
    "S5_wind_triggered",
]


def load_summary(relative: str) -> pd.DataFrame:
    path = PROJECT_DIR / relative / "metrics_summary.csv"
    if not path.exists():
        raise SystemExit("missing metrics summary: %s" % path)
    frame = pd.read_csv(path)
    return frame.loc[frame["model"].isin(MODELS)].copy()


def main() -> int:
    rows = []
    for n_bins, relative in SOURCES:
        summary = load_summary(relative)
        for scenario in SCENARIOS:
            for model in MODELS:
                hit = summary[(summary["scenario"] == scenario) & (summary["model"] == model)]
                if hit.empty:
                    continue
                record = hit.iloc[0]
                rows.append(
                    {
                        "n_bins": n_bins,
                        "scenario": scenario,
                        "model": model,
                        "n_seeds": int(record["n_runs"]),
                        "wis_mean": float(record["wis_mean_mean"]),
                        "wis_sd": float(record["wis_mean_sd"]),
                        "crps_mean": float(record["crps_mean_mean"]),
                        "crps_sd": float(record["crps_mean_sd"]),
                        "median_bias": float(record["median_bias_mean"]),
                    }
                )
    table = pd.DataFrame(rows)
    expected = len(SOURCES) * len(SCENARIOS) * len(MODELS)
    missing = expected - len(table)
    if missing:
        print("WARNING: %d/%d cells missing (training not finished?)" % (missing, expected))

    # 派生量：B6 相对 B4 的改善与 ORACLE 差距，逐分箱逐场景
    derived = []
    for n_bins, _ in SOURCES:
        for scenario in SCENARIOS:
            cell = {
                model: table[
                    (table["n_bins"] == n_bins)
                    & (table["scenario"] == scenario)
                    & (table["model"] == model)
                ]
                for model in MODELS
            }
            if any(part.empty for part in cell.values()):
                continue
            b4 = float(cell["B4"].iloc[0]["wis_mean"])
            b6 = float(cell["B6"].iloc[0]["wis_mean"])
            oracle = float(cell["ORACLE"].iloc[0]["wis_mean"])
            derived.append(
                {
                    "n_bins": n_bins,
                    "scenario": scenario,
                    "b6_minus_b4_wis": b6 - b4,
                    "relative_improvement": (b4 - b6) / b4 if b4 > 0 else None,
                    "b6_minus_oracle_wis": b6 - oracle,
                }
            )
    derived_table = pd.DataFrame(derived)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "bins_sensitivity_table.csv", index=False)
    derived_table.to_csv(OUT_DIR / "bins_sensitivity_derived.csv", index=False)
    payload = {
        "sources": {n_bins: relative for n_bins, relative in SOURCES},
        "cells_found": len(table),
        "cells_expected": expected,
        "complete": missing == 0,
        "derived": derived_table.to_dict(orient="records"),
    }
    (OUT_DIR / "bins_sensitivity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("cells:", len(table), "/", expected)
    if not derived_table.empty:
        pivot = derived_table.pivot(
            index="scenario", columns="n_bins", values="b6_minus_b4_wis"
        )
        print(pivot.round(4).to_string())
    print("output:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
