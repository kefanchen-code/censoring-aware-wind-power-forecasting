# -*- coding: utf-8 -*-
"""Step 70: C2 跨机组/跨场站泛化（LOTO/LOFO，池化训练）。

runner 仅支持单数据集，故本脚本直接复用 benchmark 包的数据契约与 adapter
接口实现池化训练：

统一网格：所有折叠均在 10-min 网格上（HOT 1-min 帧按段内等间隔降采样 10 倍，
段长不足 window+horizon 的丢弃并重建段号）；window=60 min、horizon=10 min。

折叠设计：
- LOTO-ALTA2：5 折，留出 1 台 ALTA2，其余 4 台池化训练；
- LOTO-HOT  ：8 折，留出 1 台 HOT（10-min 网格），其余 7 台池化训练；
- LOFO ALTA2→HOT：全部 5 台 ALTA2 池化训练，逐台 HOT 测试；
- LOFO HOT→ALTA2：全部 8 台 HOT 池化训练，逐台 ALTA2 测试。

场景 S1_fixed_50pct / S4_random_uniform；模型 B4/B6/CLQR；种子 42/123。
另记录 persistence 参照（无训练）。信息隔离与主协议一致：fit 仅观测视图，
predict 仅特征；标准化统计只取池化训练侧。

输出：results/loto_lofo/{loto_lofo_protocol.json, loto_lofo_results.csv}
（CSV 逐行追加，重跑自动跳过已完成单元）。
用法：
    python scripts/70_loto_lofo.py --smoke
    python scripts/70_loto_lofo.py --all
    python scripts/70_loto_lofo.py --folds loto_alta2 lofo_alta2_to_hot
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from censored_wind_power.benchmark import models as _builtin_models  # noqa: F401,E402
from censored_wind_power.benchmark import censored_qr as _clqr  # noqa: F401,E402
from censored_wind_power.benchmark.core import (  # noqa: E402
    BenchmarkContext,
    Forecast,
    create_model,
)
from censored_wind_power.benchmark.data import (  # noqa: E402
    ScenarioData,
    build_window_split,
    validate_semisynthetic_frame,
)
from censored_wind_power.benchmark.protocol import stable_json_hash  # noqa: E402
from censored_wind_power.benchmark.scoring import score_forecast  # noqa: E402

DATA_ROOT = PROJECT_DIR / "results" / "multi_turbine_validation" / "data"
OUT_DIR = PROJECT_DIR / "results" / "loto_lofo"
RESULTS_CSV = OUT_DIR / "loto_lofo_results.csv"

STEP_MIN = 10
WINDOW_STEPS = 6      # 60 min
HORIZON_STEPS = 1     # 10 min
MIN_SEGMENT_ROWS = WINDOW_STEPS + HORIZON_STEPS
SCENARIOS = ["S1_fixed_50pct", "S4_random_uniform"]
MODELS = ["B4", "B6", "CLQR"]
SEEDS = [42, 123]
QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
ALTA2_KEYS = ["ALTA2_%d" % t for t in (1301253, 1301254, 1301255, 1301256, 1301257)]
HOT_KEYS = ["HOT_%s" % t for t in ("T01", "T02", "T03", "T04", "T05", "T07", "T13", "T14")]
RATED = {"ALTA2": 1330.0, "HOT": 2300.0}
SOURCE_STEP = {"ALTA2": 10, "HOT": 1}

CSV_COLUMNS = [
    "fold", "held_out", "scenario", "model", "seed", "n_test_windows",
    "test_censored_share", "wis_mean", "wis_median", "fit_seconds", "utc",
]


def farm_of(key: str) -> str:
    return key.split("_")[0]


def rated_of(key: str) -> float:
    return RATED[farm_of(key)]


def base_config(rated_kw: float, smoke: bool) -> Dict:
    training = {
        "min_epochs": 8, "max_epochs": 40, "patience": 6, "min_delta": 0.0001,
        "batch_size": 512, "learning_rate": 0.001, "weight_decay": 0.0001,
        "gradient_clip": 1.0, "fail_on_nonconvergence": True,
    }
    if smoke:
        training.update(
            {"min_epochs": 1, "max_epochs": 1, "patience": 1,
             "fail_on_nonconvergence": False}
        )
    return {
        "schema_version": 2,
        "data": {"rated_power_kw": rated_kw, "sampling_interval_minutes": STEP_MIN},
        "split": {"strategy": "chronological_whole_segment",
                  "train_fraction": 0.70, "validation_fraction": 0.15},
        "forecast": {"window_minutes": WINDOW_STEPS * STEP_MIN,
                     "horizon_minutes": HORIZON_STEPS * STEP_MIN,
                     "support_min_pu": 0.0, "support_max_pu": 1.06, "n_bins": 106},
        "training": training,
        "model": {"hidden_channels": 32, "layers": 4, "kernel_size": 3, "dropout": 0.10},
        "evaluation": {"primary_score": "WIS", "quantiles": QUANTILES,
                       "interval_alphas": [0.10, 0.20, 0.40, 0.60]},
        "execution": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
    }


def load_frame(key: str, scenario: str) -> pd.DataFrame:
    path = DATA_ROOT / key / ("synthetic_%s.parquet" % scenario)
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame["timestamp"])
    return frame


def resample_to_grid(frame: pd.DataFrame, source_step: int) -> pd.DataFrame:
    """Stride-sample rows within each segment onto the common 10-min grid."""

    stride = STEP_MIN // source_step
    if stride == 1:
        kept = frame
    else:
        parts = []
        segment = frame["segment_id"].to_numpy()
        for seg in np.unique(segment):
            positions = np.flatnonzero(segment == seg)[::stride]
            parts.append(frame.iloc[positions])
        kept = pd.concat(parts)
    # rebuild contiguous segment ids, dropping segments too short for windows
    segment = kept["segment_id"].to_numpy()
    new_id = np.zeros(len(kept), dtype=np.int64)
    next_seg = 0
    run_start = 0
    for i in range(1, len(segment) + 1):
        boundary = i == len(segment) or segment[i] != segment[i - 1]
        if boundary:
            if i - run_start >= MIN_SEGMENT_ROWS:
                next_seg += 1
                new_id[run_start:i] = next_seg
            run_start = i
    kept = kept.loc[new_id > 0].copy()
    kept["segment_id"] = new_id[new_id > 0]
    kept.index = pd.date_range("2020-01-01", periods=len(kept), freq="%dmin" % STEP_MIN)
    return kept


def pool_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-turbine frames with globally renumbered segments."""

    pieces = []
    offset = 0
    for frame in frames:
        piece = frame.copy()
        piece["segment_id"] = piece["segment_id"].to_numpy() + offset
        offset = int(piece["segment_id"].max())
        pieces.append(piece)
    pooled = pd.concat(pieces)
    pooled.index = pd.date_range(
        "2020-01-01", periods=len(pooled), freq="%dmin" % STEP_MIN
    )
    return pooled


def assign_two_way(frame: pd.DataFrame, train_fraction: float) -> Dict[int, str]:
    """Chronological whole-segment assignment into train/validation only."""

    sizes = frame.groupby("segment_id", sort=False).size()
    total = float(sizes.sum())
    train_cut = train_fraction * total
    assignment: Dict[int, str] = {}
    cumulative = 0.0
    for raw_segment_id, raw_size in sizes.items():
        midpoint = cumulative + 0.5 * float(raw_size)
        assignment[int(raw_segment_id)] = (
            "train" if midpoint <= train_cut else "validation"
        )
        cumulative += float(raw_size)
    if set(assignment.values()) != {"train", "validation"}:
        raise RuntimeError("pooled split produced an empty side")
    return assignment


def prepare_pooled(scenario: str, frame: pd.DataFrame, config: Dict) -> ScenarioData:
    """Validate, split (train/validation), and materialize the pooled frame.

    The runner's ``prepare_scenario`` requires a non-empty test split; pooled
    training evaluates on held-out turbines instead, so the test slot carries
    a small validation subset and is never used for scoring.
    """

    rated = float(config["data"]["rated_power_kw"])
    validate_semisynthetic_frame(
        frame, rated, STEP_MIN,
        float(config["forecast"]["support_min_pu"]),
        float(config["forecast"]["support_max_pu"]),
    )
    assignment = assign_two_way(frame, 0.70)
    labels = frame["segment_id"].map(assignment)
    parts = {
        split: frame.loc[labels.eq(split)].copy().reset_index(drop=False)
        for split in ("train", "validation")
    }
    wind_mean = float(parts["train"]["wind"].mean())
    wind_std = float(parts["train"]["wind"].std())
    if not np.isfinite(wind_std) or wind_std < 1e-8:
        wind_std = 1.0
    windows = {
        split: build_window_split(
            part, rated, wind_mean, wind_std, WINDOW_STEPS, HORIZON_STEPS
        )
        for split, part in parts.items()
    }
    summary = {
        "wind_standardization_train_only": {"mean": wind_mean, "std": wind_std},
        "splits": {split: {"valid_windows": len(w)} for split, w in windows.items()},
    }
    return ScenarioData(
        scenario=scenario,
        train=windows["train"],
        validation=windows["validation"],
        test=windows["validation"],
        train_observed_rows=parts["train"]["Y_syn"].to_numpy(dtype=np.float32) / rated,
        feature_names=(
            "observed_power_pu",
            "wind_speed_train_standardized",
            "cap_pu",
            "censoring_indicator",
        ),
        split_summary=summary,
        data_signature="",
        train_wind_ms_rows=parts["train"]["wind"].to_numpy(dtype=np.float32),
        train_censored_rows=parts["train"]["is_censored"].to_numpy(dtype=np.float32),
    )


def completed_keys() -> set:
    if not RESULTS_CSV.exists():
        return set()
    done = pd.read_csv(RESULTS_CSV)
    return {
        (r.fold, r.held_out, r.scenario, r.model, str(r.seed))
        for r in done.itertuples()
    }


def append_row(row: Dict) -> None:
    frame = pd.DataFrame([row])[CSV_COLUMNS]
    frame.to_csv(
        RESULTS_CSV, mode="a", index=False, header=not RESULTS_CSV.exists()
    )


def build_folds() -> List[Dict]:
    folds = []
    for held_out in ALTA2_KEYS:
        folds.append({
            "fold": "loto_alta2", "held_out": held_out,
            "train_keys": [k for k in ALTA2_KEYS if k != held_out],
            "train_rated": RATED["ALTA2"],
        })
    for held_out in HOT_KEYS:
        folds.append({
            "fold": "loto_hot", "held_out": held_out,
            "train_keys": [k for k in HOT_KEYS if k != held_out],
            "train_rated": RATED["HOT"],
        })
    folds.append({
        "fold": "lofo_alta2_to_hot", "held_out": "HOT_all",
        "train_keys": list(ALTA2_KEYS), "train_rated": RATED["ALTA2"],
        "test_keys": list(HOT_KEYS),
    })
    folds.append({
        "fold": "lofo_hot_to_alta2", "held_out": "ALTA2_all",
        "train_keys": list(HOT_KEYS), "train_rated": RATED["HOT"],
        "test_keys": list(ALTA2_KEYS),
    })
    return folds


def run_unit(
    fold: Dict, held_out_key: str, scenario: str, config: Dict,
    pooled_scenario_data, context_dir: Path, smoke: bool,
) -> None:
    device = config["execution"]["device"]
    quantiles = np.asarray(QUANTILES, dtype=np.float64)
    edges = np.linspace(0.0, 1.06, 107, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    held_out_frame = resample_to_grid(
        load_frame(held_out_key, scenario), SOURCE_STEP[farm_of(held_out_key)]
    )
    standardization = pooled_scenario_data.split_summary[
        "wind_standardization_train_only"
    ]
    held_out_split = build_window_split(
        held_out_frame, rated_of(held_out_key),
        float(standardization["mean"]), float(standardization["std"]),
        WINDOW_STEPS, HORIZON_STEPS,
    )
    scenario_data = dataclasses.replace(pooled_scenario_data, test=held_out_split)

    # persistence reference (no training), recorded once per unit group
    persistence_forecast = Forecast(
        quantiles=np.tile(
            held_out_split.persistence[:, None], (1, len(QUANTILES))
        ).astype(np.float64)
    )
    context = BenchmarkContext(
        scenario=scenario, protocol_id="loto-lofo-v1", output_dir=context_dir,
        device=device, config=config, quantile_levels=quantiles,
        bin_edges=edges, bin_centers=centers,
    )
    persistence_score = score_forecast(
        persistence_forecast, held_out_split.truth, context
    )
    append_row({
        "fold": fold["fold"], "held_out": held_out_key, "scenario": scenario,
        "model": "B0_persistence", "seed": "fixed",
        "n_test_windows": len(held_out_split),
        "test_censored_share": float(held_out_split.censored.mean()),
        "wis_mean": float(np.mean(persistence_score.wis)),
        "wis_median": float(np.median(persistence_score.wis)),
        "fit_seconds": 0.0,
        "utc": pd.Timestamp.utcnow().isoformat(),
    })

    prediction_view = scenario_data.prediction_view()
    for model_name in MODELS:
        adapter = create_model(model_name)
        fit_input = scenario_data.fit_view()
        for seed in SEEDS:
            started = time.perf_counter()
            fitted = adapter.fit(fit_input, context, seed)
            fit_seconds = time.perf_counter() - started
            forecast = adapter.predict(fitted, prediction_view, context)
            forecast.validate(len(scenario_data.test), context)
            score = score_forecast(forecast, held_out_split.truth, context)
            append_row({
                "fold": fold["fold"], "held_out": held_out_key,
                "scenario": scenario, "model": model_name, "seed": str(seed),
                "n_test_windows": len(held_out_split),
                "test_censored_share": float(held_out_split.censored.mean()),
                "wis_mean": float(np.mean(score.wis)),
                "wis_median": float(np.median(score.wis)),
                "fit_seconds": round(fit_seconds, 1),
                "utc": pd.Timestamp.utcnow().isoformat(),
            })
            print(
                "  ", fold["fold"], held_out_key, scenario, model_name, seed,
                "wis=%.4f (%.1fs fit)" % (np.mean(score.wis), fit_seconds),
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", nargs="*", default=None)
    arguments = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = build_folds()
    design = {
        "fold_design": folds,
        "step_min": STEP_MIN, "window_steps": WINDOW_STEPS,
        "horizon_steps": HORIZON_STEPS, "scenarios": SCENARIOS,
        "models": MODELS, "seeds": SEEDS, "quantiles": QUANTILES,
        "note": "HOT 1-min frames stride-sampled x10 onto the common 10-min grid",
    }
    design["design_hash"] = stable_json_hash(design)
    (OUT_DIR / "loto_lofo_protocol.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    selected = folds
    if arguments.smoke:
        selected = [folds[0]]
    elif arguments.folds:
        selected = [f for f in folds if f["fold"] in arguments.folds]
        if not selected:
            raise SystemExit("no matching folds")
    elif not arguments.all:
        parser.error("choose --smoke, --all, or --folds")

    done = completed_keys()
    for fold in selected:
        test_keys = fold.get("test_keys", [fold["held_out"]])
        for scenario in SCENARIOS:
            config = base_config(fold["train_rated"], smoke=arguments.smoke)
            train_frames = [
                resample_to_grid(load_frame(key, scenario), SOURCE_STEP[farm_of(key)])
                for key in fold["train_keys"]
            ]
            pooled = pool_frames(train_frames)
            pooled_data = prepare_pooled(scenario, pooled, config)
            print(
                fold["fold"], scenario, "pooled windows train/val/test:",
                len(pooled_data.train), len(pooled_data.validation),
                len(pooled_data.test), flush=True,
            )
            for held_out_key in test_keys:
                needs_work = any(
                    (fold["fold"], held_out_key, scenario, model, str(seed)) not in done
                    for model in MODELS + ["B0_persistence"]
                    for seed in (SEEDS if model != "B0_persistence" else ["fixed"])
                )
                if not needs_work:
                    print(fold["fold"], held_out_key, scenario, "already done")
                    continue
                context_dir = OUT_DIR / "fits" / (
                    "%s_%s_%s" % (fold["fold"], held_out_key, scenario)
                )
                context_dir.mkdir(parents=True, exist_ok=True)
                print(fold["fold"], held_out_key, scenario, flush=True)
                run_unit(fold, held_out_key, scenario, config, pooled_data,
                         context_dir, arguments.smoke)
    print("results:", RESULTS_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
