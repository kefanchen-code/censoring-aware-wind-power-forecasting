# -*- coding: utf-8 -*-
"""Step 65: audit Altahullion 10-min SCADA for multi-turbine validation (C2).

The outcome-blind screening criteria are at least 90% row coverage, active
minutes above 50 kW, non-empty PowerRed and PowerRef channels, median power
above 30% rated at wind speeds over 12 m/s, and power standard deviation above
10 kW.

Output: ``results/multi_turbine_validation/alta2_health_audit.json`` and the
healthy-turbine list consumed by the reduced multi-turbine protocol.

Usage:
    python scripts/65_alta2_multi_health_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCADA_PATH = (
    PROJECT_DIR
    / "data"
    / "turbine_data"
    / "scada_df_ALTA2_20250904_20260301.parquet"
)
OUT_DIR = PROJECT_DIR / "results" / "multi_turbine_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RATED_KW = 1330.0
ACTIVE_THRESH_KW = 50.0
COVERAGE_FLOOR = 0.90
WIND_HIGH_THRESH = 12.0
HIGH_WIND_POWER_FLOOR = 0.30 * RATED_KW
POWER_STD_FLOOR_KW = 10.0


def main() -> int:
    frame = pd.read_parquet(SCADA_PATH)
    stations = sorted({station for station, _ in frame.columns})
    expected_rows = len(frame)
    report = {}
    healthy = []
    for station in stations:
        try:
            sub = frame.xs(station, level=0, axis=1)
        except KeyError:
            continue
        columns = set(sub.columns)
        power = sub.get("ActPower_Value_mean")
        wind = sub.get("AcWindSp_AcWindSp_mean")
        power_red = sub.get("PowerRed_PowerRed_timeon")
        power_ref = sub.get("PowerRef_PowerRef_mean")
        entry = {"station": station, "n_signals": len(columns)}
        ok = True
        reasons = []
        if power is None or wind is None:
            ok = False
            reasons.append("missing power/wind channel")
            entry["coverage"] = None
        else:
            coverage = float(power.notna().mean())
            entry["coverage"] = round(coverage, 4)
            if coverage < COVERAGE_FLOOR:
                ok = False
                reasons.append("coverage %.3f < %.2f" % (coverage, COVERAGE_FLOOR))
            active = power.gt(ACTIVE_THRESH_KW).fillna(False)
            entry["active_share"] = round(float(active.mean()), 4)
            if not active.any():
                ok = False
                reasons.append("no active minutes")
            if power_red is None or power_ref is None:
                ok = False
                reasons.append("missing command channels")
                entry["command_channels"] = False
            else:
                entry["command_channels"] = True
                entry["curtailment_share_active"] = round(
                    float(power_red.gt(0).fillna(False)[active].mean())
                    if active.any()
                    else float("nan"),
                    4,
                )
            high_wind = wind.gt(WIND_HIGH_THRESH).fillna(False) & power.notna()
            if high_wind.any():
                median_high = float(power[high_wind].median())
                entry["median_power_kw_wind_gt_12"] = round(median_high, 1)
                if median_high < HIGH_WIND_POWER_FLOOR:
                    ok = False
                    reasons.append(
                        "high-wind median power %.0f < %.0f"
                        % (median_high, HIGH_WIND_POWER_FLOOR)
                    )
            else:
                entry["median_power_kw_wind_gt_12"] = None
            power_std = float(power.std())
            entry["power_std_kw"] = round(power_std, 1)
            if power_std < POWER_STD_FLOOR_KW:
                ok = False
                reasons.append("near-constant power")
        entry["healthy"] = ok
        entry["rejection_reasons"] = reasons
        report[station] = entry
        if ok:
            healthy.append(station)

    summary = {
        "source": "data/turbine_data/scada_df_ALTA2_20250904_20260301.parquet",
        "sampling": "10-min",
        "n_rows": expected_rows,
        "criteria": {
            "coverage_floor": COVERAGE_FLOOR,
            "active_thresh_kw": ACTIVE_THRESH_KW,
            "high_wind_thresh_ms": WIND_HIGH_THRESH,
            "high_wind_power_floor_kw": HIGH_WIND_POWER_FLOOR,
            "power_std_floor_kw": POWER_STD_FLOOR_KW,
        },
        "healthy_turbines": healthy,
        "n_healthy": len(healthy),
        "per_turbine": report,
    }
    (OUT_DIR / "alta2_health_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("stations:", len(stations), "healthy:", healthy)
    for station, entry in report.items():
        print(
            station,
            "healthy=%s" % entry["healthy"],
            "cov=%s" % entry.get("coverage"),
            "curt=%s" % entry.get("curtailment_share_active"),
            entry.get("rejection_reasons", []),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
