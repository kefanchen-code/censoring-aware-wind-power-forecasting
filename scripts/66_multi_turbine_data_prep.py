# -*- coding: utf-8 -*-
"""Step 66: C2 多机组双场站数据准备（结果盲的数据层）。

为两组机组各自构建与主基准同族的半合成场景底本：
- ALTA2 健康 5 台（1301253-1301257，10-min SCADA，名单来自
  scripts/65_alta2_multi_health_audit.py 的 alta2_health_audit.json）；
- HOT 8 台 1-min（T01-T05, T07, T13, T14；排除 T11，控制语义异常，
  引用 results/hot_truth_channel_bias/hot_truth_bias.json 审计）。

每台机组：真实自由分钟即底本（A_true 已知），限功分钟不参与底本；
注入场景 S0/S1/S4/S5 与主协议同参数（S1=50% 定深 60% 时长占比 120-min 事件、
S4=均匀随机深度、S5=风速>12 触发 60% 定深），时长以分钟计、与采样步长无关。

输出：results/multi_turbine_validation/data/{farm}_{turbine}/
    base_info.json + synthetic_{S0,S1,S4,S5}.parquet
用法：
    python scripts/66_multi_turbine_data_prep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_DIR / "results" / "multi_turbine_validation"
DATA_ROOT = OUT_ROOT / "data"

ALTA2_SCADA = (
    PROJECT_DIR / "data" / "turbine_data" / "scada_df_ALTA2_20250904_20260301.parquet"
)
HOT_DIR = PROJECT_DIR / "data" / "hill_of_towie" / "converted"
HEALTH_PATH = OUT_ROOT / "alta2_health_audit.json"

ALTA2_RATED_KW = 1330.0
HOT_RATED_KW = 2300.0
ACTIVE_THRESH_KW = 50.0
MIN_SEGMENT_MIN = 60
EVENT_SEED = 20260804
CAP_SEED = EVENT_SEED + 100
NON_BINDING_FACTOR = 1.05

SCENARIOS = {
    "S0_no_curtailment": {"cap_type": "none"},
    "S1_fixed_50pct": {
        "cap_type": "fixed",
        "cap_frac": 0.50,
        "policy_rate": 0.60,
        "duration_min": 120,
    },
    "S4_random_uniform": {
        "cap_type": "random_event",
        "cap_low_frac": 0.30,
        "cap_high_frac": 0.90,
        "policy_rate": 0.60,
        "duration_min": 120,
    },
    "S5_wind_triggered": {
        "cap_type": "wind_triggered",
        "wind_thresh": 12.0,
        "cap_frac": 0.60,
    },
}


def assign_event_blocks(segment_id, policy_rate, duration_steps, rng):
    """120-min persistent non-overlapping blocks per segment (audit semantics)."""

    active = np.zeros(len(segment_id), dtype=bool)
    event_id = np.zeros(len(segment_id), dtype=np.int64)
    next_event = 0
    for seg in np.unique(segment_id):
        positions = np.flatnonzero(segment_id == seg)
        target = int(round(policy_rate * len(positions)))
        if target <= 0:
            continue
        possible = max(1, len(positions) - duration_steps + 1)
        assigned = 0
        for local_start in rng.permutation(possible):
            if assigned >= target:
                break
            block = positions[local_start : min(local_start + duration_steps, len(positions))]
            block = block[~active[block]][: target - assigned]
            if block.size == 0:
                continue
            next_event += 1
            active[block] = True
            event_id[block] = next_event
            assigned += int(block.size)
    return active, event_id


def build_base(power_kw, wind_ms, times, step_min) -> pd.DataFrame:
    """Free minutes only, split into contiguous >=60-min segments."""

    power = np.asarray(power_kw, dtype=float)
    wind = np.asarray(wind_ms, dtype=float)
    usable = np.isfinite(power) & np.isfinite(wind) & (power > ACTIVE_THRESH_KW)
    keep = np.zeros(len(power), dtype=bool)
    segment_id = np.zeros(len(power), dtype=np.int64)
    run_start = None
    next_seg = 0
    for i in range(len(power) + 1):
        run_end = i >= len(power) or not usable[i] or (
            i > 0
            and (times[i] - times[i - 1]).total_seconds() > step_min * 60 * 1.5
        )
        if run_end and run_start is not None:
            length = i - run_start
            if length * step_min >= MIN_SEGMENT_MIN:
                keep[run_start:i] = True
                next_seg += 1
                segment_id[run_start:i] = next_seg
            run_start = None
        elif not run_end and run_start is None:
            run_start = i
    base = pd.DataFrame(
        {
            "A_true": power[keep],
            "wind": wind[keep],
            "segment_id": segment_id[keep],
        },
        index=times[keep],
    )
    return base


def construct_scenario(base, name, config, rated_kw, step_min, event_seed, cap_seed):
    scenario = base.copy()
    scenario["C_syn"] = rated_kw * NON_BINDING_FACTOR
    scenario["policy_active"] = False
    scenario["event_id"] = 0
    if config["cap_type"] in {"fixed", "random_event"}:
        duration_steps = max(1, int(round(config["duration_min"] / step_min)))
        active, event_id = assign_event_blocks(
            scenario["segment_id"].to_numpy(),
            config["policy_rate"],
            duration_steps,
            np.random.default_rng(event_seed),
        )
        scenario["policy_active"] = active
        scenario["event_id"] = event_id
        if config["cap_type"] == "fixed":
            scenario.loc[active, "C_syn"] = config["cap_frac"] * rated_kw
        else:
            cap_rng = np.random.default_rng(cap_seed)
            for current in np.unique(event_id[event_id > 0]):
                frac = cap_rng.uniform(config["cap_low_frac"], config["cap_high_frac"])
                scenario.loc[event_id == current, "C_syn"] = frac * rated_kw
    elif config["cap_type"] == "wind_triggered":
        active = scenario["wind"].gt(config["wind_thresh"]).to_numpy()
        scenario["policy_active"] = active
        starts = active & ~np.r_[False, active[:-1]]
        scenario["event_id"] = np.where(active, np.cumsum(starts), 0)
        scenario.loc[active, "C_syn"] = config["cap_frac"] * rated_kw
    scenario["Y_syn"] = np.minimum(scenario["A_true"], scenario["C_syn"])
    scenario["is_censored"] = scenario["policy_active"] & (
        scenario["A_true"] >= scenario["C_syn"]
    )
    return scenario


def load_alta2_turbines():
    health = json.loads(HEALTH_PATH.read_text(encoding="utf-8"))
    stations = health["healthy_turbines"]
    frame = pd.read_parquet(ALTA2_SCADA)
    records = {}
    for station in stations:
        sub = frame.xs(station, level=0, axis=1)
        records[("ALTA2", station)] = {
            "power_kw": sub["ActPower_Value_mean"].to_numpy(dtype=float),
            "wind_ms": sub["AcWindSp_AcWindSp_mean"].to_numpy(dtype=float),
            "times": frame.index,
            "step_min": 10,
            "rated_kw": ALTA2_RATED_KW,
        }
    return records


def load_hot_turbines():
    records = {}
    for turbine in ("T01", "T02", "T03", "T04", "T05", "T07", "T13", "T14"):
        frame = pd.read_parquet(HOT_DIR / ("fl_df_HOT_%s_1min.parquet" % turbine))
        frame.columns = [column[1] for column in frame.columns]
        free = frame["PowerRed_PowerRed"].le(0) | frame["PowerRed_PowerRed"].isna()
        records[("HOT", turbine)] = {
            "power_kw": frame["ActPower_Value"].where(free).to_numpy(dtype=float),
            "wind_ms": frame["AcWindSp_AcWindSp"].to_numpy(dtype=float),
            "times": frame.index,
            "step_min": 1,
            "rated_kw": HOT_RATED_KW,
        }
    return records


def main() -> int:
    if not HEALTH_PATH.exists():
        raise SystemExit("run scripts/65_alta2_multi_health_audit.py first")
    records = {}
    records.update(load_alta2_turbines())
    records.update(load_hot_turbines())

    manifest = {}
    for (farm, turbine), record in records.items():
        base = build_base(
            record["power_kw"], record["wind_ms"], record["times"], record["step_min"]
        )
        out_dir = DATA_ROOT / ("%s_%s" % (farm, turbine))
        out_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "farm": farm,
            "turbine": turbine,
            "step_min": record["step_min"],
            "rated_kw": record["rated_kw"],
            "n_free_rows": int(len(base)),
            "n_segments": int(base["segment_id"].nunique()),
            "event_seed": EVENT_SEED,
            "cap_seed": CAP_SEED,
            "scenarios": {},
        }
        if len(base) == 0:
            print(farm, turbine, "NO USABLE FREE RECORD - skipped")
            continue
        for name, config in SCENARIOS.items():
            scenario = construct_scenario(
                base, name, config, record["rated_kw"], record["step_min"],
                EVENT_SEED, CAP_SEED,
            )
            scenario.to_parquet(out_dir / ("synthetic_%s.parquet" % name))
            info["scenarios"][name] = {
                "n_events": int(scenario["event_id"].max()),
                "policy_share": round(float(scenario["policy_active"].mean()), 4),
                "n_censored": int(scenario["is_censored"].sum()),
            }
        (out_dir / "base_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest["%s_%s" % (farm, turbine)] = info
        print(
            farm, turbine,
            "free=%d segs=%d" % (info["n_free_rows"], info["n_segments"]),
            {k: v["policy_share"] for k, v in info["scenarios"].items()},
        )
    (OUT_ROOT / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("output:", DATA_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
