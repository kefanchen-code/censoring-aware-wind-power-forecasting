# -*- coding: utf-8 -*-
"""Step 69: C2 聚合敏感性（预测层三档聚合，回应 6.66 MW 缩放质疑）。

用 HOT 9 台 1-min 数据（T01-T05/T07/T11/T13/T14）的自由发电分钟：
1) 估计经验空间相关矩阵（原功率 Pearson 相关 + 去除各自风速-功率曲线
   均值效应后的残差相关，后者对应预测误差的空间耦合强度）；
2) 在预测层构建场站级聚合分布：每台机组的预测分布取自由分钟内
   按风速分箱的经验条件分布（oracle 型预测分布），场站聚合对比三档
   耦合假设——独立抽样 / 经验相关高斯 copula / 完全相关(comonotonic)；
3) 报告聚合分布分位（p10/p50/p90）与缺额概率 P(聚合 < 阈值×场站额定)。

不重跑 Simulink；价值层聚合重跑列为后续可选项（写入论文局限）。
输出：results/aggregation_sensitivity/{spatial_correlation.json,
aggregation_summary.csv, aggregation_sensitivity.json}
用法：python scripts/69_aggregation_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
HOT_DIR = PROJECT_DIR / "data" / "hill_of_towie" / "converted"
OUT_DIR = PROJECT_DIR / "results" / "aggregation_sensitivity"

TURBINES = ["T01", "T02", "T03", "T04", "T05", "T07", "T11", "T13", "T14"]
RATED_KW = 2300.0
ACTIVE_THRESH_KW = 50.0
WIND_BINS = [(0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 99)]
SHORTFALL_FRACTIONS = [0.3, 0.5, 0.7]
COPULA_DRAWS = 4000
SEED = 20260805
TEST_FRACTION = 0.20  # 时间尾部留出作评估分钟，与训练侧分箱估计隔离


def load_turbine(turbine: str) -> pd.DataFrame:
    frame = pd.read_parquet(HOT_DIR / ("fl_df_HOT_%s_1min.parquet" % turbine))
    frame.columns = [column[1] for column in frame.columns]
    power = pd.to_numeric(frame["ActPower_Value"], errors="coerce")
    wind = pd.to_numeric(frame["AcWindSp_AcWindSp"], errors="coerce")
    red = pd.to_numeric(frame["PowerRed_PowerRed"], errors="coerce")
    free = (red.le(0) | red.isna()) & power.gt(ACTIVE_THRESH_KW) & wind.notna()
    out = pd.DataFrame({"power_kw": power, "wind_ms": wind}, index=frame.index)
    return out.loc[free]


def bin_index(wind: np.ndarray) -> np.ndarray:
    idx = np.zeros(len(wind), dtype=int)
    for b, (lo, hi) in enumerate(WIND_BINS):
        idx[(wind >= lo) & (wind < hi)] = b
    return idx


def main() -> int:
    frames = {}
    for turbine in TURBINES:
        frames[turbine] = load_turbine(turbine)
        print(turbine, "free minutes:", len(frames[turbine]))

    power = pd.DataFrame({t: f["power_kw"] for t, f in frames.items()})
    wind = pd.DataFrame({t: f["wind_ms"] for t, f in frames.items()})
    aligned = power.dropna(how="any")
    wind = wind.loc[aligned.index].dropna(how="any")
    common = aligned.index.intersection(wind.index)
    power = (aligned.loc[common] / RATED_KW).astype(float)
    wind = wind.loc[common].astype(float)
    print("common free minutes (all 9 turbines):", len(common))
    if len(common) < 2000:
        raise SystemExit("insufficient common minutes for correlation estimation")

    # --- 1) 经验空间相关 ---
    raw_corr = power.corr().to_numpy()
    # 残差：逐机去除同风速分箱的功率均值（曲线效应），保留湍流/预测误差成分
    residual = power.copy()
    for j, turbine in enumerate(TURBINES):
        column = power[turbine].to_numpy()
        turbine_bins = bin_index(wind[turbine].to_numpy())
        means = np.array(
            [column[(turbine_bins == b)].mean() if (turbine_bins == b).any() else np.nan
             for b in range(len(WIND_BINS))]
        )
        residual[turbine] = column - means[turbine_bins]
    residual = residual.dropna()
    resid_corr = residual.corr().to_numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "spatial_correlation.json").write_text(
        json.dumps(
            {
                "turbines": TURBINES,
                "n_common_minutes": int(len(common)),
                "n_residual_minutes": int(len(residual)),
                "rated_kw": RATED_KW,
                "raw_power_corr_mean_offdiag": float(
                    raw_corr[~np.eye(len(TURBINES), dtype=bool)].mean()
                ),
                "residual_corr_mean_offdiag": float(
                    resid_corr[~np.eye(len(TURBINES), dtype=bool)].mean()
                ),
                "raw_power_corr": raw_corr.tolist(),
                "residual_corr": resid_corr.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- 2) 训练侧经验条件分布（前 80% 分钟分箱估计） ---
    split = int(len(common) * (1.0 - TEST_FRACTION))
    train_power, train_wind = power.iloc[:split], wind.iloc[:split]
    test_power, test_wind = power.iloc[split:], wind.iloc[split:]
    empirical: dict[str, list[np.ndarray]] = {}
    for j, turbine in enumerate(TURBINES):
        column = train_power[turbine].to_numpy()
        turbine_bins = bin_index(train_wind[turbine].to_numpy())
        empirical[turbine] = [
            np.sort(column[turbine_bins == b]) for b in range(len(WIND_BINS))
        ]
    print("test minutes:", len(test_power))

    # --- 3) 三档聚合：独立 / 经验相关 copula / 完全相关 ---
    rng = np.random.default_rng(SEED)
    # 保证残差相关矩阵正定（投影到最近相关矩阵的简化处理：特征值截断）
    eigenvalues, eigenvectors = np.linalg.eigh(resid_corr)
    clipped = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
    scale = np.sqrt(np.diag(clipped))
    copula_corr = clipped / np.outer(scale, scale)
    chol = np.linalg.cholesky(copula_corr)

    n_t = len(TURBINES)
    records = []
    for regime in ("independent", "empirical_copula", "perfect_correlation"):
        quantile_rows = []
        shortfall_counts = {frac: 0 for frac in SHORTFALL_FRACTIONS}
        for start in range(0, len(test_power), COPULA_DRAWS):
            chunk = test_wind.iloc[start : start + COPULA_DRAWS]
            n_draws = len(chunk)
            if n_draws == 0:
                continue
            if regime == "perfect_correlation":
                uniforms = np.tile(rng.uniform(size=(n_draws, 1)), (1, n_t))
            elif regime == "independent":
                uniforms = rng.uniform(size=(n_draws, n_t))
            else:
                normals = rng.multivariate_normal(
                    np.zeros(n_t), copula_corr, size=n_draws
                )
                uniforms = pd.DataFrame(normals).rank(pct=True).to_numpy()
            aggregate = np.zeros(n_draws)
            for j, turbine in enumerate(TURBINES):
                support_bins = bin_index(chunk.to_numpy()[:, j])
                for b in range(len(WIND_BINS)):
                    members = support_bins == b
                    if not members.any():
                        continue
                    samples = empirical[turbine][b]
                    if samples.size == 0:
                        continue
                    positions = np.clip(
                        (uniforms[members, j] * samples.size).astype(int),
                        0,
                        samples.size - 1,
                    )
                    aggregate[members] += samples[positions]
            aggregate /= n_t  # 场站均值化（pu/机），等价于聚合后除以场站额定
            quantile_rows.append(
                [start, np.quantile(aggregate, 0.10), np.quantile(aggregate, 0.50),
                 np.quantile(aggregate, 0.90)]
            )
            for frac in SHORTFALL_FRACTIONS:
                shortfall_counts[frac] += int((aggregate < frac).sum())
        quantiles_all = np.array(quantile_rows)
        total_minutes = float(len(test_power))
        records.append(
            {
                "regime": regime,
                "agg_q10_mean": float(quantiles_all[:, 1].mean()),
                "agg_q50_mean": float(quantiles_all[:, 2].mean()),
                "agg_q90_mean": float(quantiles_all[:, 3].mean()),
                **{
                    "shortfall_prob_%.1f" % frac: shortfall_counts[frac] / total_minutes
                    for frac in SHORTFALL_FRACTIONS
                },
            }
        )
        print(regime, records[-1])

    summary = pd.DataFrame(records)
    summary.to_csv(OUT_DIR / "aggregation_summary.csv", index=False)
    (OUT_DIR / "aggregation_sensitivity.json").write_text(
        json.dumps(
            {
                "turbines": TURBINES,
                "n_test_minutes": int(len(test_power)),
                "copula_draws_per_block": COPULA_DRAWS,
                "seed": SEED,
                "wind_bins": WIND_BINS,
                "shortfall_fractions": SHORTFALL_FRACTIONS,
                "note": "aggregate expressed in pu per turbine (farm mean); "
                "value-layer rerun out of scope (limitation)",
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("output:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
