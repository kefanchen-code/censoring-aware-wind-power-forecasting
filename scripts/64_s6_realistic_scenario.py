# -*- coding: utf-8 -*-
"""Step 64: 生成 S6_realistic 经验指令注入场景（C3）。

底本：与 S1-S5 完全同源的 342 段自由发电记录（scripts/10_altahullion_audit.py
的 eligible_u_segments，真值 A_true 已知）。注入的限功事件时长与深度不再取
构造值，而是从 scripts/62_real_command_stats.py 统计的 ALTA2 T11 真实指令
经验分布中抽样：

- 目标限功分钟占比 = ALTA2 实测活跃限功分钟 / (自由 + 活跃限功) ≈ 42.7%，
  取代 S1-S4 的构造值 60%；
- 事件时长 ~ ALTA2 经验时长分布（中位 6 min、p95 ≈ 377 min、重尾）；
- 事件深度 ~ ALTA2 经验深度分布（中位 0.943，即多数指令接近深度限制），
  cap = 额定 × (1 - 深度)，下限 2% 额定；
- 事件在段内不重叠、非持久（逐事件独立起止），与真实指令的突发形态一致。

HOT 的同步性（≥半机同步占限功分钟 57.8%）无法在单机半合成中表达，
只在协议生态效度一节作为场级证据引用，不进入本场景。

输出：results/s6_realistic/synthetic_S6_realistic.parquet + s6_scenario_summary.json

用法：
    python scripts/64_s6_realistic_scenario.py
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
OUT_DIR = PROJECT_DIR / "results" / "s6_realistic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_NAME = "S6_realistic"
SCENARIO_SEED = 20260804
CAP_SEED = SCENARIO_SEED + 100
RATED_KW = 1330.0
CAP_FLOOR_KW = 0.02 * RATED_KW
MIN_DURATION_MIN = 2
STATS_PATH = PROJECT_DIR / "results" / "real_command_stats" / "real_command_stats.json"
EVENTS_CSV = PROJECT_DIR / "results" / "real_command_stats" / "events_alta2.csv"


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "altahullion_audit", PROJECT_DIR / "scripts" / "10_altahullion_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empirical_target_rate() -> float:
    """ALTA2 active-curtailment share over (free + curtailed) active minutes."""

    audit = load_audit_module()
    frame = audit.load_frame()
    frame, _ = audit.classify_observations(frame)
    free = int(frame["label"].eq("U").sum())
    curtailed = int((frame["curtail_flag"] & frame["active"]).sum())
    return curtailed / (free + curtailed)


def place_events(
    segment_id: np.ndarray,
    target_rate: float,
    durations: np.ndarray,
    depths: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy non-overlapping placement until the minute budget is spent."""

    n = len(segment_id)
    budget = int(round(target_rate * n))
    policy_active = np.zeros(n, dtype=bool)
    event_id = np.zeros(n, dtype=np.int64)
    cap_kw = np.full(n, RATED_KW * 1.05)
    placed = 0
    next_event = 0
    for seg in np.unique(segment_id):
        positions = np.flatnonzero(segment_id == seg)
        seg_budget = int(round(target_rate * len(positions)))
        if seg_budget <= 0:
            continue
        taken = np.zeros(len(positions), dtype=bool)
        order = rng.permutation(len(positions))
        seg_used = 0
        for start in order:
            if seg_used >= seg_budget:
                break
            duration = int(durations[rng.integers(len(durations))])
            duration = min(duration, len(positions) - start)
            if duration < MIN_DURATION_MIN:
                continue
            window = slice(start, start + duration)
            if taken[window].any():
                continue
            depth = float(depths[rng.integers(len(depths))])
            cap = max(CAP_FLOOR_KW, RATED_KW * (1.0 - depth))
            taken[window] = True
            next_event += 1
            placed += duration
            seg_used += duration
            absolute = positions[window]
            policy_active[absolute] = True
            event_id[absolute] = next_event
            cap_kw[absolute] = cap
    print("budget minutes:", budget, "placed:", placed, "events:", next_event)
    return policy_active, event_id, cap_kw


def main() -> int:
    if not STATS_PATH.exists() or not EVENTS_CSV.exists():
        raise SystemExit("run scripts/62_real_command_stats.py first")
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    events = pd.read_csv(EVENTS_CSV)
    durations = events["duration_min"].to_numpy(dtype=float)
    depths = np.clip(events["depth_pu"].dropna().to_numpy(dtype=float), 0.0, 0.98)

    audit = load_audit_module()
    frame = audit.load_frame()
    frame, _ = audit.classify_observations(frame)
    base = audit.eligible_u_segments(frame)
    target_rate = empirical_target_rate()

    scenario = base[["power", "wind", "pitch", "gen_rpm", "power_ref", "segment_id"]].copy()
    scenario = scenario.rename(columns={"power": "A_true"})
    rng = np.random.default_rng(SCENARIO_SEED)
    policy_active, event_id, cap_kw = place_events(
        scenario["segment_id"].to_numpy(), target_rate, durations, depths, rng
    )
    scenario["C_syn"] = cap_kw
    scenario["policy_active"] = policy_active
    scenario["event_id"] = event_id
    scenario["Y_syn"] = np.minimum(scenario["A_true"], scenario["C_syn"])
    scenario["is_censored"] = policy_active & (scenario["A_true"] >= scenario["C_syn"])
    scenario.to_parquet(OUT_DIR / ("synthetic_" + SCENARIO_NAME + ".parquet"))

    summary = {
        "scenario": SCENARIO_NAME,
        "scenario_seed": SCENARIO_SEED,
        "cap_seed": CAP_SEED,
        "design": {
            "base": "342 eligible U segments (same base as S1-S5, truth known)",
            "duration_distribution": "empirical ALTA2 T11 (62_real_command_stats)",
            "depth_distribution": "empirical ALTA2 T11, clipped to [0, 0.98]",
            "target_curtailment_rate": round(target_rate, 4),
            "rate_source": "ALTA2 active curtailment / (free + curtailed)",
            "cap_floor_kw": CAP_FLOOR_KW,
            "hot_synchrony_note": (
                "HOT farm-level synchrony (57.8% of curtailment minutes with"
                " >=half turbines curtailed) cannot be represented in a"
                " single-turbine semisynthetic; cited as farm-level evidence only."
            ),
        },
        "realized": {
            "n_rows": int(len(scenario)),
            "n_events": int(scenario["event_id"].max()),
            "policy_minutes": int(policy_active.sum()),
            "policy_share": round(float(policy_active.mean()), 4),
            "n_censored": int(scenario["is_censored"].sum()),
            "censored_share_of_policy": round(
                float(scenario["is_censored"].sum() / max(1, policy_active.sum())), 4
            ),
            "cap_mean_kw": round(float(scenario.loc[policy_active, "C_syn"].mean()), 1),
            "duration_quantiles": {
                key: stats["alta2_T11"]["duration_min"][key]
                for key in ("median", "p75", "p95")
            },
        },
    }
    (OUT_DIR / "s6_scenario_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["realized"], indent=2, ensure_ascii=False))
    print("output:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
