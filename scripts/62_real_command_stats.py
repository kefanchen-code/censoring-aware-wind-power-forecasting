# -*- coding: utf-8 -*-
"""Step 62: summarize real control commands at two wind farms for S6 (C3).

Data tracks
-----------
- ALTA2 T11: one-minute SCADA loaded through ``load_frame`` in script 10.
  Curtailment requires ``PowerRed > 0`` while the turbine is active, matching
  the release-event screening rule.
- Eight HOT turbines (T01-T05, T07, T13, T14): converted one-minute files.
  T11 is excluded because of anomalous control semantics documented in the
  truth-channel bias audit. Curtailment requires ``PowerRed_PowerRed > 0`` and
  effective cap depth is obtained from ``ActLimit_Power``.

For each event, report duration, depth ``1 - cap/rated``, onset wind speed, and
onset hour. Also report farm-level simultaneity for HOT.

Outputs under ``results/real_command_stats/`` include the JSON summary,
event-level CSV files, and empirical duration/depth figures.

Usage:
    python scripts/62_real_command_stats.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HOT_DIR = PROJECT_DIR / "data" / "hill_of_towie" / "converted"
OUT_DIR = PROJECT_DIR / "results" / "real_command_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALTA2_RATED_KW = 1330.0
HOT_RATED_KW = 2300.0
HOT_TURBINES = ("T01", "T02", "T03", "T04", "T05", "T07", "T13", "T14")
MIN_DURATION_MIN = 2          # 短于该值的不构成限功事件（与审计同族）
MAX_GAP_MIN = 2               # 事件内部允许的最大记录缺口
SYNC_SHARE_THRESH = 0.5       # ≥半数在报机组同时限功 => 同步限功分钟


def quantile_table(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    qs = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def extract_events(
    times: pd.DatetimeIndex,
    curtailed: np.ndarray,
    cap_kw: np.ndarray,
    wind_ms: np.ndarray,
    rated_kw: float,
    source: str,
    turbine: str,
) -> pd.DataFrame:
    """Contiguous curtailment runs -> one row per event."""

    curtailed = np.asarray(curtailed, dtype=bool)
    if curtailed.any():
        starts = np.flatnonzero(curtailed & ~np.r_[False, curtailed[:-1]])
        ends = np.flatnonzero(curtailed & ~np.r_[curtailed[1:], False])
    else:
        starts = ends = np.array([], dtype=int)
    rows = []
    for start, end in zip(starts, ends + 1):
        block_times = times[start:end]
        gaps = block_times.to_series().diff().dt.total_seconds().div(60.0)
        largest_gap = gaps.max()
        if np.isfinite(largest_gap) and largest_gap > MAX_GAP_MIN:
            continue  # record gap inside: not one command episode
        duration = int(end - start)
        if duration < MIN_DURATION_MIN:
            continue
        caps = cap_kw[start:end]
        finite_caps = caps[np.isfinite(caps)]
        depth = (
            float(1.0 - finite_caps.min() / rated_kw) if finite_caps.size else float("nan")
        )
        rows.append(
            {
                "farm": source,
                "turbine": turbine,
                "start_utc": block_times[0],
                "end_utc": block_times[-1],
                "duration_min": duration,
                "depth_pu": depth,
                "mean_cap_pu": (
                    float(finite_caps.mean() / rated_kw) if finite_caps.size else float("nan")
                ),
                "trigger_wind_ms": float(wind_ms[start])
                if np.isfinite(wind_ms[start])
                else float("nan"),
                "trigger_hour": int(block_times[0].hour),
            }
        )
    return pd.DataFrame(rows)


def alta2_events() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location(
        "altahullion_audit", PROJECT_DIR / "scripts" / "10_altahullion_audit.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    frame = audit.load_frame()
    frame, _ = audit.classify_observations(frame)
    curtailed = (frame["curtail_flag"] & frame["active"]).to_numpy()
    return extract_events(
        frame.index,
        curtailed,
        frame["power_ref"].to_numpy(dtype=float),
        frame["wind"].to_numpy(dtype=float),
        ALTA2_RATED_KW,
        "ALTA2",
        "T11",
    )


def hot_events() -> tuple[pd.DataFrame, dict]:
    """Per-turbine events plus farm-level synchrony statistics."""

    pieces = []
    per_minute = {}
    for turbine in HOT_TURBINES:
        path = HOT_DIR / ("fl_df_HOT_%s_1min.parquet" % turbine)
        frame = pd.read_parquet(path)
        frame.columns = [column[1] for column in frame.columns]
        curtailed = frame["PowerRed_PowerRed"].gt(0).to_numpy()
        per_minute[turbine] = pd.Series(curtailed, index=frame.index)
        pieces.append(
            extract_events(
                frame.index,
                curtailed,
                frame["ActLimit_Power"].to_numpy(dtype=float),
                frame["AcWindSp_AcWindSp"].to_numpy(dtype=float),
                HOT_RATED_KW,
                "HOT",
                turbine,
            )
        )
    events = pd.concat(pieces, ignore_index=True)

    aligned = pd.DataFrame(per_minute)
    reporting = aligned.notna().sum(axis=1)
    share = aligned.fillna(False).sum(axis=1) / reporting.clip(lower=1)
    curtail_minutes = aligned.fillna(False).any(axis=1)
    synchrony = {
        "curtailment_minutes_any_turbine": int(curtail_minutes.sum()),
        "synchronous_minutes_share_ge_half": float(
            (share[curtail_minutes] >= SYNC_SHARE_THRESH).mean()
        )
        if curtail_minutes.any()
        else float("nan"),
        "turbine_curtailed_share_at_sync_minutes": float(
            share[curtail_minutes & (share >= SYNC_SHARE_THRESH)].mean()
        )
        if (curtail_minutes & (share >= SYNC_SHARE_THRESH)).any()
        else float("nan"),
    }
    return events, synchrony


def main() -> int:
    alta2 = alta2_events()
    hot, synchrony = hot_events()
    alta2.to_csv(OUT_DIR / "events_alta2.csv", index=False, encoding="utf-8-sig")
    hot.to_csv(OUT_DIR / "events_hot.csv", index=False, encoding="utf-8-sig")

    def farm_block(events: pd.DataFrame) -> dict:
        return {
            "n_events": int(len(events)),
            "duration_min": quantile_table(events["duration_min"]),
            "depth_pu": quantile_table(events["depth_pu"]),
            "trigger_wind_ms": quantile_table(events["trigger_wind_ms"]),
            "trigger_hour_histogram": {
                str(hour): int(count)
                for hour, count in events["trigger_hour"].value_counts()
                .sort_index()
                .items()
            },
            "minutes_total": int(events["duration_min"].sum()) if len(events) else 0,
        }

    summary = {
        "criteria": {
            "alta2": "PowerRed>0 and active; depth from PowerRef",
            "hot": "PowerRed_PowerRed>0; depth from ActLimit_Power",
            "min_duration_min": MIN_DURATION_MIN,
            "max_internal_gap_min": MAX_GAP_MIN,
            "hot_excluded_turbine": "T11 (control-semantics anomaly; hot_truth_bias audit)",
        },
        "alta2_T11": farm_block(alta2),
        "hot_8turbines": farm_block(hot),
        "hot_synchrony": synchrony,
        "per_turbine_event_counts_hot": {
            turbine: int(count)
            for turbine, count in hot["turbine"].value_counts().sort_index().items()
        },
    }
    (OUT_DIR / "real_command_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    render_figure(alta2, hot)
    print(json.dumps(summary["alta2_T11"]["duration_min"], indent=2))
    print(json.dumps(summary["hot_8turbines"]["duration_min"], indent=2))
    print("synchrony:", synchrony)
    print("output:", OUT_DIR)
    return 0


def render_figure(alta2: pd.DataFrame, hot: pd.DataFrame) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

    panel = axes[0]
    for events, label, color in (
        (alta2, "ALTA2 T11 (1-min)", "tab:blue"),
        (hot, "HOT 8 turbines (1-min)", "tab:orange"),
    ):
        durations = np.sort(events["duration_min"].to_numpy(dtype=float))
        panel.step(
            durations,
            np.arange(1, len(durations) + 1) / max(1, len(durations)),
            where="post",
            color=color,
            lw=1.2,
            label="%s, n=%d" % (label, len(durations)),
        )
    panel.set_xscale("log")
    panel.set_xlabel("event duration (min)")
    panel.set_ylabel("ECDF")
    panel.legend(fontsize=6)

    panel = axes[1]
    for events, label, color in (
        (alta2, "ALTA2", "tab:blue"),
        (hot, "HOT", "tab:orange"),
    ):
        depth = events["depth_pu"].dropna().to_numpy()
        panel.hist(
            depth,
            bins=np.linspace(0, 1, 21),
            histtype="step",
            color=color,
            lw=1.2,
            label=label,
            density=True,
        )
    panel.set_xlabel("curtailment depth (1 - C / rated)")
    panel.set_ylabel("density")
    panel.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(OUT_DIR / "Fig_real_command_stats.png", dpi=300)
    figure.savefig(OUT_DIR / "Fig_real_command_stats.svg")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
