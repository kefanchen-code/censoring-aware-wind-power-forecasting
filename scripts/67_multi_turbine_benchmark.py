# -*- coding: utf-8 -*-
"""Step 67: run the reduced two-farm multi-turbine protocol (C2).

Generate one benchmark configuration per prepared turbine for scenarios
S0/S1/S4/S5 and models B0_persistence, B2_recon, B4, B6, CLQR, and ORACLE.
ALTA2 uses a 10-min horizon and three seeds; HOT uses a 15-min horizon and one
seed. Both use a 60-min history. Turbine-specific protocol identifiers keep
these runs separate from the frozen main benchmark.

Usage:
    python scripts/67_multi_turbine_benchmark.py --list
    python scripts/67_multi_turbine_benchmark.py --smoke ALTA2_1301253
    python scripts/67_multi_turbine_benchmark.py ALTA2_1301253 HOT_T01 ...
    python scripts/67_multi_turbine_benchmark.py --all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_DIR / "results" / "multi_turbine_validation" / "data"
RUN_ROOT = PROJECT_DIR / "results" / "multi_turbine_validation" / "runs"
CONFIG_DIR = PROJECT_DIR / "configs" / "multi_turbine"
RUNNER = PROJECT_DIR / "scripts" / "run_pap_benchmark.py"

SEEDS = [42, 123, 256]
# HOT 1-min 单机全量耗时过长（2026-08-05 用户裁定）：HOT 子集降为单种子，
# ALTA2 五台维持三种子（已完成）；该不对称写入协议文件与论文局限条款。
SEEDS_HOT = [42]
MODELS = ["B0_persistence", "B2_recon", "B4", "B6", "CLQR", "ORACLE"]
SCENARIOS = ["S0_no_curtailment", "S1_fixed_50pct", "S4_random_uniform", "S5_wind_triggered"]
COMPARISONS = [
    {"family": "MT_censored_likelihood_primary", "metric": "wis",
     "reference": "B4", "challenger": "B6"},
    {"family": "MT_censored_likelihood_vs_free_only", "metric": "wis",
     "reference": "B4", "challenger": "B6"},
    {"family": "MT_secondary_censored_quantile_CLQR", "metric": "wis",
     "reference": "B4", "challenger": "CLQR"},
    {"family": "MT_secondary_censored_quantile_CLQR", "metric": "wis",
     "reference": "B6", "challenger": "CLQR"},
    {"family": "oracle_gap_report_not_confirmatory", "metric": "wis",
     "reference": "B6", "challenger": "ORACLE", "equivalence_margin": 0.005},
]


def turbine_keys() -> list[str]:
    return sorted(path.name for path in DATA_ROOT.iterdir() if path.is_dir())


def horizon_for(key: str) -> int:
    info = json.loads((DATA_ROOT / key / "base_info.json").read_text(encoding="utf-8"))
    return 10 if info["step_min"] == 10 else 15


def sampling_for(key: str) -> tuple[int, float]:
    info = json.loads((DATA_ROOT / key / "base_info.json").read_text(encoding="utf-8"))
    return info["step_min"], info["rated_kw"]


def write_config(key: str) -> Path:
    step_min, rated_kw = sampling_for(key)
    # ALTA2 10-min 机组窗口数小约一个量级，多分位 pinball 收敛更慢：
    # 首次全量在 S0/CLQR/seed42 于 40 epoch 未触发 patience（best 仍在更新），
    # 故 ALTA2 配置单独放宽 max_epochs（protocol_id 随之变化，仅影响 ALTA2 折）。
    max_epochs = 80 if step_min == 10 else 40
    seeds = SEEDS if key.startswith("ALTA2") else SEEDS_HOT
    config = {
        "schema_version": 2,
        "output_dir": "results/multi_turbine_validation/runs/%s" % key,
        "model_modules": [
            "censored_wind_power.benchmark.tobit",
            "censored_wind_power.benchmark.powercurve",
            "censored_wind_power.benchmark.censored_qr",
        ],
        "scenarios": SCENARIOS,
        "models": MODELS,
        "model_settings": {"B2_recon": {"bin_width_ms": 0.5, "min_bin_count": 30}},
        "seeds": seeds,
        "data": {
            "directory": "results/multi_turbine_validation/data/%s" % key,
            "filename_pattern": "synthetic_{scenario}.parquet",
            "rated_power_kw": rated_kw,
            "sampling_interval_minutes": step_min,
        },
        "split": {
            "strategy": "chronological_whole_segment",
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
        },
        "forecast": {
            "window_minutes": 60,
            "horizon_minutes": 10 if step_min == 10 else 15,
            "support_min_pu": 0.0,
            "support_max_pu": 1.06,
            "n_bins": 106,
        },
        "training": {
            "min_epochs": 8,
            "max_epochs": max_epochs,
            "patience": 6,
            "min_delta": 0.0001,
            "batch_size": 512,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "fail_on_nonconvergence": True,
        },
        "model": {
            "hidden_channels": 32,
            "layers": 4,
            "kernel_size": 3,
            "dropout": 0.10,
        },
        "evaluation": {
            "primary_score": "WIS",
            "quantiles": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95],
            "interval_alphas": [0.10, 0.20, 0.40, 0.60],
        },
        "inference": {
            "seed": 20260726,
            "cluster_bootstrap_draws": 5000,
            "sign_flip_draws": 100000,
            "exhaustive_max_clusters": 16,
        },
        "comparisons": COMPARISONS,
        "execution": {"device": "auto"},
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / ("%s.json" % key)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*", help="turbine keys like ALTA2_1301253")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()

    keys = turbine_keys()
    if arguments.list:
        for key in keys:
            step, rated = sampling_for(key)
            print(key, "step=%dmin horizon=%dmin rated=%.0f" % (step, horizon_for(key), rated))
        return 0
    selected = keys if arguments.all else arguments.keys
    if not selected:
        parser.error("select turbine keys, --all, or --list")
    for key in selected:
        if key not in keys:
            raise SystemExit("unknown turbine key: %s" % key)
    for key in selected:
        config_path = write_config(key)
        command = [sys.executable, str(RUNNER), "--config", str(config_path), "run"]
        if arguments.smoke:
            command.append("--smoke")
        log_path = RUN_ROOT / ("%s.log" % key)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        print("running", key, "->", log_path, flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            print("FAILED", key, "rc=%d" % completed.returncode)
            return completed.returncode
        print("done", key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
