"""Altahullion T11 data audit and leakage-controlled semisynthetic construction.

The script is the canonical data-preparation entry point for the manuscript.
It uses all eligible continuous U segments, treats a missing one-minute sample
as a gap, screens release events without using the post-release outcome, and
injects persistent cap events rather than independent minute-wise caps.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJ_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJ_DIR
    / "data"
    / "turbine_data"
    / "fl_df_ALTA2_T11_20250904_to_20260228.parquet"
)
OUT_DIR = PROJ_DIR / "results" / "altahullion_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SID = "1301257"
P_RATED = 1330.0
P_ACTIVE_THRESH = 60.0
TRACK_TOL = 0.08
MIN_CURTAIL_BEFORE_RELEASE = 15
MIN_NORMAL_AFTER_RELEASE = 10
EXPECTED_STEP_MIN = 1.0
GAP_TOL_MIN = 0.1
SCENARIO_SEED = 42
# S1-S4 share one event mask stream so that curtailment depth is the only
# difference between them; random cap depths use an independent stream.
EVENT_SEED = SCENARIO_SEED
CAP_SEED = SCENARIO_SEED + 100


SCENARIOS = {
    "S0_no_curtailment": {
        "cap_type": "none",
    },
    "S1_fixed_50pct": {
        "cap_type": "fixed",
        "cap_value": P_RATED * 0.50,
        "policy_rate": 0.60,
        "duration": 120,
    },
    "S2_fixed_70pct": {
        "cap_type": "fixed",
        "cap_value": P_RATED * 0.70,
        "policy_rate": 0.60,
        "duration": 120,
    },
    "S3_fixed_85pct": {
        "cap_type": "fixed",
        "cap_value": P_RATED * 0.85,
        "policy_rate": 0.60,
        "duration": 120,
    },
    "S4_random_uniform": {
        "cap_type": "random_event",
        "cap_low": P_RATED * 0.30,
        "cap_high": P_RATED * 0.90,
        "policy_rate": 0.60,
        "duration": 120,
    },
    "S5_wind_triggered": {
        "cap_type": "wind_triggered",
        "wind_thresh": 12.0,
        "cap_value": P_RATED * 0.60,
    },
}


def load_frame():
    raw = pd.read_parquet(DATA_PATH)
    columns = {
        "power": "ActPower_Value",
        "power_ref": "PowerRef_PowerRef",
        "wind": "AcWindSp_AcWindSp",
        "pitch": "PitcPosA_Value",
        "gen_rpm": "GenRpm_Value",
        "power_red": "PowerRed_PowerRed",
        "power_max": "max_ActPower_Value",
        "power_min": "min_ActPower_Value",
    }
    frame = pd.concat(
        [raw[(SID, source)].rename(target) for target, source in columns.items()],
        axis=1,
    )
    frame.index.name = "timestamp"
    return frame


def classify_observations(frame):
    frame = frame.copy()
    frame["active"] = frame["power"].gt(P_ACTIVE_THRESH) & frame["power"].notna()
    frame["curtail_flag"] = frame["power_red"].gt(0)
    gap_minutes = frame.index.to_series().diff().dt.total_seconds().div(60.0)
    frame["gap_before"] = gap_minutes.fillna(EXPECTED_STEP_MIN)
    frame["has_gap"] = (
        frame["gap_before"].sub(EXPECTED_STEP_MIN).abs().gt(GAP_TOL_MIN)
    )
    frame["label"] = "X"

    condition_u = (
        frame["active"]
        & ~frame["curtail_flag"]
        & ~frame["has_gap"]
        & frame["power_ref"].ge(P_RATED * 0.95)
    )
    frame.loc[condition_u, "label"] = "U"

    tracking_error = frame["power"].sub(frame["power_ref"]).abs()
    tracking_bound = frame["power_ref"].mul(TRACK_TOL).add(20.0)
    condition_r = (
        frame["active"]
        & frame["curtail_flag"]
        & ~frame["has_gap"]
        & frame["power_ref"].between(P_ACTIVE_THRESH, P_RATED * 0.95, inclusive="neither")
        & tracking_error.le(tracking_bound)
        & frame["wind"].gt(5.0)
    )
    frame.loc[condition_r, "label"] = "R"

    condition_i = (
        frame["active"]
        & frame["curtail_flag"]
        & ~frame["has_gap"]
        & frame["label"].eq("X")
        & frame["power_ref"].between(P_ACTIVE_THRESH, P_RATED * 0.95, inclusive="neither")
    )
    frame.loc[condition_i, "label"] = "I"
    return frame, tracking_error


def screen_release_events(frame):
    """Pre-outcome screen; post-release power is retained only as an endpoint."""
    curtail_binary = (frame["curtail_flag"] & frame["active"]).astype(int)
    release_times = curtail_binary.diff().loc[lambda x: x.eq(-1)].index
    records = []
    for release_time in release_times:
        index = frame.index.get_loc(release_time)
        if not isinstance(index, (int, np.integer)):
            continue
        pre_start = index - MIN_CURTAIL_BEFORE_RELEASE
        post_end = index + MIN_NORMAL_AFTER_RELEASE
        if pre_start < 0 or post_end >= len(frame):
            continue
        pre = frame.iloc[pre_start:index]
        post = frame.iloc[index:post_end]
        window = frame.iloc[pre_start:post_end]
        if pre["curtail_flag"].mean() < 0.80:
            continue
        if post["curtail_flag"].mean() > 0.20:
            continue
        if post["power"].gt(P_ACTIVE_THRESH).mean() < 0.80:
            continue
        if window["has_gap"].any():
            continue
        if pre["wind"].mean() < 6.0:
            continue
        pre_power = float(pre["power"].mean())
        post_power = float(post["power"].mean())
        records.append(
            {
                "release_time": release_time,
                "pre_power_mean_kw": round(pre_power, 1),
                "post_power_mean_kw": round(post_power, 1),
                "power_jump_ratio": round(post_power / max(pre_power, 1.0), 2),
                "post_exceeds_pre_by_10pct": bool(post_power > 1.1 * pre_power),
                "pre_wind_mean_ms": round(float(pre["wind"].mean()), 2),
                "pre_power_ref_mean_kw": round(float(pre["power_ref"].mean()), 1),
                "pre_curtail_fraction": round(float(pre["curtail_flag"].mean()), 3),
                "post_curtail_fraction": round(float(post["curtail_flag"].mean()), 3),
            }
        )
    return pd.DataFrame(records), len(release_times)


def eligible_u_segments(frame):
    mask = frame["label"].eq("U")
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    segments = []
    for segment_id, (_, segment) in enumerate(frame.loc[mask].groupby(groups.loc[mask])):
        if len(segment) < 60:
            continue
        tagged = segment.copy()
        tagged["segment_id"] = segment_id
        segments.append(tagged)
    if not segments:
        raise RuntimeError("No continuous U segment contains at least 60 records")
    return pd.concat(segments).sort_index()


def assign_event_blocks(segment_ids, policy_rate, duration, rng):
    active = np.zeros(len(segment_ids), dtype=bool)
    event_id = np.zeros(len(segment_ids), dtype=np.int64)
    next_event = 0
    for segment_id in np.unique(segment_ids):
        positions = np.flatnonzero(segment_ids == segment_id)
        target = int(round(policy_rate * len(positions)))
        if target <= 0:
            continue
        possible = max(1, len(positions) - duration + 1)
        assigned = 0
        for local_start in rng.permutation(possible):
            if assigned >= target:
                break
            block = positions[local_start : min(local_start + duration, len(positions))]
            block = block[~active[block]][: target - assigned]
            if block.size == 0:
                continue
            next_event += 1
            active[block] = True
            event_id[block] = next_event
            assigned += int(block.size)
    return active, event_id


def construct_scenario(base, name, config, event_seed=EVENT_SEED, cap_seed=CAP_SEED):
    scenario = base[["power", "wind", "pitch", "gen_rpm", "power_ref", "segment_id"]].copy()
    scenario = scenario.rename(columns={"power": "A_true"})
    scenario["C_syn"] = P_RATED * 1.05
    scenario["policy_active"] = False
    scenario["event_id"] = 0

    if config["cap_type"] in {"fixed", "random_event"}:
        # A fresh generator per scenario with the shared EVENT_SEED makes the
        # S1-S4 event masks bitwise identical.
        active, event_id = assign_event_blocks(
            scenario["segment_id"].to_numpy(),
            config["policy_rate"],
            config["duration"],
            np.random.default_rng(event_seed),
        )
        scenario["policy_active"] = active
        scenario["event_id"] = event_id
        if config["cap_type"] == "fixed":
            scenario.loc[active, "C_syn"] = config["cap_value"]
        else:
            cap_rng = np.random.default_rng(cap_seed)
            for current_event in np.unique(event_id[event_id > 0]):
                cap = cap_rng.uniform(config["cap_low"], config["cap_high"])
                scenario.loc[event_id == current_event, "C_syn"] = cap
    elif config["cap_type"] == "wind_triggered":
        active = scenario["wind"].gt(config["wind_thresh"])
        starts = active & ~active.shift(fill_value=False)
        event_id = starts.cumsum().where(active, 0).astype(np.int64)
        scenario.loc[active, "C_syn"] = config["cap_value"]
        scenario["policy_active"] = active.to_numpy()
        scenario["event_id"] = event_id.to_numpy()
    elif config["cap_type"] != "none":
        raise ValueError("unknown cap_type: %r" % config["cap_type"])

    scenario["Y_syn"] = np.minimum(scenario["A_true"], scenario["C_syn"])
    scenario["is_censored"] = (
        scenario["policy_active"] & scenario["A_true"].ge(scenario["C_syn"])
    )
    summary = {
        "binding_rate": round(float(scenario["is_censored"].mean()), 6),
        "policy_coverage": round(float(scenario["policy_active"].mean()), 6),
        "n_records": int(len(scenario)),
        "n_censored": int(scenario["is_censored"].sum()),
        "n_events": int(scenario["event_id"].max()),
        "cap_mean_kw": round(float(scenario["C_syn"].mean()), 1),
        "event_seed": int(event_seed),
        "cap_seed": int(cap_seed),
    }
    scenario.to_parquet(OUT_DIR / ("synthetic_" + name + ".parquet"))
    return summary, scenario


def main():
    frame = load_frame()
    frame, tracking_error = classify_observations(frame)
    label_counts = frame["label"].value_counts()
    releases, raw_release_count = screen_release_events(frame)
    releases.to_csv(OUT_DIR / "qualified_release_events.csv", index=False)
    base = eligible_u_segments(frame)

    scenario_summaries = {}
    scenario_frames = {}
    for name, config in SCENARIOS.items():
        scenario_summaries[name], scenario_frames[name] = construct_scenario(
            base, name, config
        )

    shared_names = [
        "S1_fixed_50pct",
        "S2_fixed_70pct",
        "S3_fixed_85pct",
        "S4_random_uniform",
    ]
    reference_frame = scenario_frames[shared_names[0]]
    for other in shared_names[1:]:
        other_frame = scenario_frames[other]
        if not (
            np.array_equal(
                reference_frame["policy_active"].to_numpy(),
                other_frame["policy_active"].to_numpy(),
            )
            and np.array_equal(
                reference_frame["event_id"].to_numpy(),
                other_frame["event_id"].to_numpy(),
            )
        ):
            raise RuntimeError(
                "shared event mask violated: %s differs from %s"
                % (other, shared_names[0])
            )

    release_successes = int(
        releases.get("post_exceeds_pre_by_10pct", pd.Series(dtype=bool)).sum()
    )
    summary = {
        "dataset": "Altahullion T11 (Station 1301257)",
        "source": "Zenodo DOI: 10.5281/zenodo.19948235",
        "license": "CC-BY-4.0",
        "resolution": "1 min",
        "time_span": "%s to %s" % (frame.index[0].isoformat(), frame.index[-1].isoformat()),
        "total_records": int(len(frame)),
        "rated_power_kw": P_RATED,
        "label_distribution": {key: int(value) for key, value in label_counts.items()},
        "active_records": int(frame["active"].sum()),
        "R_records": int(frame["label"].eq("R").sum()),
        "U_records": int(frame["label"].eq("U").sum()),
        "raw_release_transitions": int(raw_release_count),
        "qualified_release_events_pre_outcome": int(len(releases)),
        "release_events_with_post_power_rise_gt_10pct": release_successes,
        "normal_segments_for_synthetic": int(base["segment_id"].nunique()),
        "normal_records_for_synthetic": int(len(base)),
        "mean_R_tracking_error_kw": float(tracking_error.loc[frame["label"].eq("R")].mean()),
        "synthetic_scenarios": scenario_summaries,
        "classification_rules": {
            "U": "PowerRed=0, active, exact one-minute continuity, PowerRef>=95% rated",
            "R": "PowerRed>0, active, exact one-minute continuity, PowerRef<95% rated, power tracks PowerRef within 8%+20kW, wind>5m/s",
            "I": "PowerRed>0 and active but the right-censoring binding condition is not secure",
            "X": "inactive, discontinuous, startup/shutdown, or unclassifiable",
        },
        "shared_event_mask_S1_S4": True,
        "event_seed": EVENT_SEED,
        "cap_seed": CAP_SEED,
        "semisynthetic_protocol": {
            "base_records": "all eligible U records in continuous segments of at least 60 min",
            "shared_event_mask": "S1-S4 use identical policy_active/event_id generated from EVENT_SEED; S4 cap depths come from the independent CAP_SEED stream; verified element-wise before writing the summary",
            "censoring_semantics": "right-censored when policy_active and A_true >= C_syn",
            "fixed_and_random_caps": "120-min persistent non-overlapping events; random caps are constant within event",
            "outcome_blind_release_screen": True,
            "primary_predictors": "power, wind, cap, and censoring flag; pitch and generator speed are excluded from the primary model",
        },
    }
    with (OUT_DIR / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Altahullion audit complete")
    print("U/R/I/X:", {key: int(value) for key, value in label_counts.items()})
    print("Eligible U records:", len(base), "segments:", base["segment_id"].nunique())
    print("Release events (pre-outcome):", len(releases), "post-rise successes:", release_successes)
    for name, values in scenario_summaries.items():
        print(name, values)
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()
