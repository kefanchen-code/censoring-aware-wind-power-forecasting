# -*- coding: utf-8 -*-
"""Step 4: 实测控制诱导删失对输入通道的污染幅度。

识别策略（双重差分, DiD）
------------------------
真实限功在本风场 55% 的时刻是场级同步的，此时全部机组读数同时被污染，
无法互为参照。因此只使用"部分限功"时刻：同一时刻既有限功发电机组 i，
又有未限功发电机组 j。对每个机对 (i, j)：

    污染量 = E[v_i - v_j | i 限功] - E[v_i - v_j | i,j 均未限功]

后一项吸收了机位固有的风速差（尾流、地形、风速计标定），前一项额外包含
限功导致的读数偏移。分层按参照机 j 的风速区间进行（j 未被污染），
并可选按风向扇区分层。

同一框架同时测量桨距角与转速的偏移，以及污染经功率曲线放大后的功率含义。

设计有效性检验
--------------
1. 安慰剂通道：环境温度不受限功影响，其 DiD 应统计不显著；若显著则说明
   处理组与对照组时刻本身不可比（识别失败）。
2. 事件研究交叉验证：只用限功投入/解除瞬间前后各 K 帧，机组自身作参照，
   再减去同期未限功机组的风速变化。投入与解除应给出等量反号的估计。

落盘
----
除 stdout 外，全部结果同时写入 results/alta2_contamination_did/（命名与
13_hot_contamination_did.py 的产物对齐，便于双风场对比）。本脚本的数字是
主协议污染注入强度（相对 +16.7%）的唯一实测来源，因此必须可复现地落盘。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PATH = "data/turbine_data/scada_df_ALTA2_20250904_20260301.parquet"
RATED = 1330.0
HEALTHY = ["1301253", "1301254", "1301255", "1301256", "1301257"]
RNG = np.random.default_rng(42)
N_BOOT = 5000
OUT_DIR = Path("results/alta2_contamination_did")

# 落盘累加器（仅收集已打印的量，不额外消耗 RNG）
OUT = {"sample_base": [], "channels": [], "wind_band": [], "depth": [],
       "power_curve": [], "power_implication": [], "event_study": []}

df = pd.read_parquet(PATH)


def col(t, s):
    return df[(t, s)]


def generating(t):
    return ((col(t, "TurbinOK_TurbinOK_timeon") > 599)
            & (col(t, "ScInOper_ScInOper_timeon") > 599)
            & (col(t, "ActPower_Value_mean") > 0.05 * RATED)
            & (col(t, "GenRpm_Value_mean") > 800))


# ---------- 状态矩阵 ----------
gen = pd.DataFrame({t: generating(t) for t in HEALTHY})
ref = pd.DataFrame({t: col(t, "PowerRef_PowerRef_mean") / RATED for t in HEALTHY})
pwr = pd.DataFrame({t: col(t, "ActPower_Value_mean") / RATED for t in HEALTHY})
wnd = pd.DataFrame({t: col(t, "AcWindSp_AcWindSp_mean") for t in HEALTHY})
pit = pd.DataFrame({t: col(t, "PitcPosA_Value_mean") for t in HEALTHY})
rpm = pd.DataFrame({t: col(t, "GenRpm_Value_mean") for t in HEALTHY})
ti = pd.DataFrame({t: col(t, "AcWindSp_AcWindSp_stddev") / col(t, "AcWindSp_AcWindSp_mean")
                   for t in HEALTHY})
amb = pd.DataFrame({t: col(t, "AmbieTmp_Value_mean") for t in HEALTHY})   # 安慰剂通道
yaw = pd.DataFrame({t: col(t, "YawPos_Value_mean") for t in HEALTHY})    # 设计检验
# 测风塔风向仅挂在 1301257 上，作为场级唯一风向参照
wdir = col("1301257", "FTAnem1_WndDirec_mean")

# 限功且指令被跟踪（确认功率真的被钳住，构成右删失）
capped = gen & (ref < 0.95) & (pwr > 0.90 * ref) & (pwr < 1.08 * ref)
free = gen & (ref > 0.99)

print("=== 样本基础 ===")
for t in HEALTHY:
    print("%-9s 发电帧=%5d  其中限功(已跟踪)=%5d (%.1f%%)  未限功=%5d" % (
        t, int(gen[t].sum()), int(capped[t].sum()),
        100 * capped[t].sum() / max(gen[t].sum(), 1), int(free[t].sum())))
    OUT["sample_base"].append({
        "turbine": t, "n_generating": int(gen[t].sum()),
        "n_capped_tracked": int(capped[t].sum()),
        "capped_share_of_generating": float(capped[t].sum() / max(gen[t].sum(), 1)),
        "n_free": int(free[t].sum()),
    })

n_cap = capped.sum(axis=1)
n_free = free.sum(axis=1)
usable = (n_cap >= 1) & (n_free >= 1)
print("\n可用于 DiD 的时刻（同时存在限功机与未限功机）: %d 帧 (%.1f%% of all)" % (
    int(usable.sum()), 100 * usable.mean()))
print("  其中限功机数分布:", dict(n_cap[usable].value_counts().sort_index().astype(int)))

# ---------- 构造配对长表 ----------
records = []
idx = df.index
for i in HEALTHY:
    for j in HEALTHY:
        if i == j:
            continue
        # 处理组：i 限功、j 未限功
        m_treat = capped[i] & free[j]
        # 对照组：i、j 均未限功
        m_base = free[i] & free[j]
        for mask, kind in ((m_treat, "treat"), (m_base, "base")):
            if not mask.any():
                continue
            sub = pd.DataFrame({
                "time": idx[mask.values],
                "pair": "%s>%s" % (i, j),
                "kind": kind,
                "dv": (wnd[i] - wnd[j])[mask].values,
                "dv_rel": ((wnd[i] - wnd[j]) / wnd[j])[mask].values,
                "dpitch": (pit[i] - pit[j])[mask].values,
                "drpm": (rpm[i] - rpm[j])[mask].values,
                "dti": (ti[i] - ti[j])[mask].values,
                "damb": (amb[i] - amb[j])[mask].values,
                "dyaw": (yaw[i] - yaw[j])[mask].values,
                "v_ref": wnd[j][mask].values,          # 参照机风速（未污染）
                "depth": ref[i][mask].values,           # i 的限功档位 (pu)
                "wdir": wdir[mask].values,
            })
            records.append(sub)

long = pd.concat(records, ignore_index=True)
long["v_bin"] = pd.cut(long["v_ref"], [0, 5, 6, 7, 8, 9, 10, 12, 30])
long["day"] = pd.to_datetime(long["time"]).dt.floor("D")
print("\n配对样本: treat=%d, base=%d, 机对数=%d" % (
    (long.kind == "treat").sum(), (long.kind == "base").sum(), long.pair.nunique()))

OUT["design"] = {
    "farm": "Altahullion (ALTA2)", "turbine_type_kw": RATED,
    "n_reference_turbines": len(HEALTHY), "basis_minutes": 10,
    "data_file": PATH,
    "identification": "E[v_i - v_j | i capped, j free] - E[v_i - v_j | i,j both free]",
    "strata": ["pair", "v_bin"],
    "note_no_direction_stratum": (
        "wind direction is available only from the single met mast (1301257), "
        "not per turbine, so no direction sector stratum here; the Hill of Towie "
        "replication in 13_hot_contamination_did.py adds a 30 deg sector stratum"
    ),
    "min_treat_frames_per_stratum": 5, "min_base_frames_per_stratum": 20,
    "bootstrap": "day-block, B=%d (main dv) / B=800 (companion channels)" % N_BOOT,
    "rng_seed": 42,
    "n_usable_timestamps": int(usable.sum()),
    "usable_share_of_all": float(usable.mean()),
    "n_treat_rows": int((long.kind == "treat").sum()),
    "n_base_rows": int((long.kind == "base").sum()),
    "n_pairs": int(long.pair.nunique()),
}


# ---------- DiD 估计 ----------
def did_estimate(frame, value="dv", strat=("pair", "v_bin")):
    """按 (机对 x 风速区间) 分层做 DiD，再以处理组样本量加权汇总。"""
    t = frame[frame.kind == "treat"].groupby(list(strat), observed=True)[value].agg(["mean", "size"])
    b = frame[frame.kind == "base"].groupby(list(strat), observed=True)[value].agg(["mean", "size"])
    joined = t.join(b, lsuffix="_t", rsuffix="_b", how="inner")
    joined = joined[(joined["size_t"] >= 5) & (joined["size_b"] >= 20)]
    if joined.empty:
        return np.nan, 0, 0
    diff = joined["mean_t"] - joined["mean_b"]
    w = joined["size_t"]
    return float((diff * w).sum() / w.sum()), int(w.sum()), len(joined)


def block_bootstrap_ci(frame, value="dv", n_boot=N_BOOT):
    """按天分块重采样（时间自相关 + 场级同步限功的聚类结构）。"""
    days = frame["day"].unique()
    out = []
    for _ in range(n_boot):
        pick = RNG.choice(days, size=len(days), replace=True)
        # 用 merge 复制被抽中的天（含重复）
        cnt = pd.Series(pick).value_counts()
        rep = frame[frame["day"].isin(cnt.index)].copy()
        rep["w"] = rep["day"].map(cnt).astype(int)
        rep = rep.loc[rep.index.repeat(rep["w"])]
        est, _, _ = did_estimate(rep, value)
        if np.isfinite(est):
            out.append(est)
    a = np.asarray(out)
    return np.quantile(a, [0.025, 0.5, 0.975]), a


print("\n" + "=" * 78)
print("=== 主结果 A: 机舱风速污染 Δv (m/s，正=限功时读数偏高) ===")
print("=" * 78)
for value, unit, label in (("dv", "m/s", "机舱风速"),
                           ("dv_rel", "-", "风速相对偏差"),
                           ("dpitch", "deg", "桨距角A"),
                           ("drpm", "rpm", "发电机转速"),
                           ("dti", "-", "湍流强度"),
                           ("damb", "degC", "环境温度[安慰剂]"),
                           ("dyaw", "deg", "偏航位置[设计检验]")):
    est, n, k = did_estimate(long, value)
    ci, boots = block_bootstrap_ci(long, value, n_boot=800)
    star = "  <-- 显著" if (ci[0] > 0) or (ci[2] < 0) else "  (不显著)"
    print("%-16s DiD = %+8.3f %-5s  95%%CI [%+.3f, %+.3f]  n=%5d L=%3d%s" % (
        label, est, unit, ci[0], ci[2], n, k, star))
    OUT["channels"].append({
        "channel": value, "label": label, "unit": unit, "did": float(est),
        "ci_low": float(ci[0]), "ci_high": float(ci[2]),
        "n_treat_frames": int(n), "n_strata": int(k),
        "significant": bool((ci[0] > 0) or (ci[2] < 0)),
        "role": {"damb": "placebo", "dyaw": "design_check"}.get(value, "outcome"),
        "n_boot": 800,
    })

ci, boots = block_bootstrap_ci(long, "dv")
print("\n机舱风速 Δv 的日块 bootstrap 95%% CI: [%+.3f, %+.3f]  中位数 %+.3f m/s (B=%d)" % (
    ci[0], ci[2], ci[1], len(boots)))
print("  p(Δv>0) = %.4f" % float((boots > 0).mean()))

_dv_est, _dv_n, _dv_k = did_estimate(long, "dv")
_rel_est, _, _ = did_estimate(long, "dv_rel")
OUT["headline"] = {
    "delta_v_ms": float(_dv_est),
    "delta_v_ci_low_ms": float(ci[0]), "delta_v_ci_high_ms": float(ci[2]),
    "delta_v_boot_median_ms": float(ci[1]),
    "delta_v_n_boot": int(len(boots)),
    "prob_delta_v_positive": float((boots > 0).mean()),
    "relative_delta": float(_rel_est),
    "n_treat_frames": int(_dv_n), "n_strata": int(_dv_k),
    "adopted_by_main_protocol": "relative_delta -> contamination.relative_delta = 0.167",
}

print("\n--- 按参照机风速分层：加性偏差 vs 乘性偏差 ---")
print("%-12s %8s %10s %10s" % ("v_ref区间", "n_treat", "Δv(m/s)", "Δv/v"))
for lvl in long["v_bin"].cat.categories:
    sub = long[long.v_bin == lvl]
    if (sub.kind == "treat").sum() < 30:
        continue
    e_a, n, _ = did_estimate(sub, "dv", strat=("pair",))
    e_r, _, _ = did_estimate(sub, "dv_rel", strat=("pair",))
    print("%-12s %8d %+10.3f %+10.3f" % (str(lvl), n, e_a, e_r))
    OUT["wind_band"].append({"v_ref_band_ms": str(lvl), "n_treat_frames": int(n),
                             "delta_v_ms": float(e_a), "relative_delta": float(e_r)})

print("\n" + "=" * 78)
print("=== 主结果 B: 污染幅度随限功深度的变化（机制检验）===")
print("=" * 78)
long["depth_bin"] = pd.cut(long["depth"], [0, 0.25, 0.4, 0.6, 0.95],
                           labels=["<0.25", "0.25-0.4", "0.4-0.6", "0.6-0.95"])
print("%-12s %8s %10s %10s %10s" % ("限功档位pu", "n_treat", "Δv(m/s)", "Δpitch", "Δrpm"))
for lvl in ["<0.25", "0.25-0.4", "0.4-0.6", "0.6-0.95"]:
    sub = long[(long.kind == "base") | (long.depth_bin == lvl)]
    e_v, n, _ = did_estimate(sub, "dv")
    e_p, _, _ = did_estimate(sub, "dpitch")
    e_r, _, _ = did_estimate(sub, "drpm")
    print("%-12s %8d %+10.3f %+10.2f %+10.1f" % (lvl, n, e_v, e_p, e_r))
    OUT["depth"].append({"cap_depth_pu": lvl, "n_treat_frames": int(n),
                         "delta_v_ms": float(e_v), "delta_pitch_deg": float(e_p),
                         "delta_rpm": float(e_r)})

print("\n" + "=" * 78)
print("=== 主结果 C: 污染的功率含义（经未限功功率曲线放大）===")
print("=" * 78)
# 用全部未限功发电样本拟合经验功率曲线 P(v)，取局部斜率
fv, fp = [], []
for t in HEALTHY:
    m = free[t]
    fv.append(wnd[t][m].values)
    fp.append(pwr[t][m].values)
fv, fp = np.concatenate(fv), np.concatenate(fp)
edges = np.arange(3, 16.5, 0.5)
centers, curve = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (fv >= lo) & (fv < hi)
    if m.sum() >= 50:
        centers.append((lo + hi) / 2)
        curve.append(np.median(fp[m]))
centers, curve = np.asarray(centers), np.asarray(curve)
slope = np.gradient(curve, centers)  # dP/dv, pu per (m/s)
print("经验功率曲线（未限功样本 n=%d）局部斜率 dP/dv:" % len(fv))
for c, p, s in zip(centers, curve, slope):
    if 4 <= c <= 13:
        print("   v=%4.1f m/s  P=%.3f pu  dP/dv=%.4f pu/(m/s)" % (c, p, s))
    OUT["power_curve"].append({"v_ms": float(c), "power_pu": float(p),
                               "dP_dv_pu_per_ms": float(s)})
OUT["design"]["n_free_samples_for_power_curve"] = int(len(fv))

dv_est, _, _ = did_estimate(long, "dv")
rel_est, _, _ = did_estimate(long, "dv_rel")
print("\n若把被污染的机舱风速直接送入 PAP 模型，等效功率偏差 = dP/dv x Δv：")
print("   （加性口径 Δv=%+.3f m/s；乘性口径 Δv=%+.1f%% x v）" % (dv_est, 100 * rel_est))
for target in (6.0, 7.0, 8.0, 9.0, 10.0, 11.0):
    k = int(np.argmin(np.abs(centers - target)))
    c, s = centers[k], slope[k]
    add, mul = s * dv_est, s * rel_est * c
    print("   v≈%4.1f m/s (dP/dv=%.4f): 加性 %+.4f pu (%+6.1f kW) | 乘性 %+.4f pu (%+6.1f kW)"
          % (c, s, add, add * RATED, mul, mul * RATED))
    OUT["power_implication"].append({
        "v_ms": float(c), "dP_dv_pu_per_ms": float(s),
        "additive_bias_pu": float(add), "additive_bias_kw": float(add * RATED),
        "multiplicative_bias_pu": float(mul),
        "multiplicative_bias_kw": float(mul * RATED),
    })

print("\n" + "=" * 78)
print("=== 主结果 D: 事件研究交叉验证（机组自身作参照）===")
print("=" * 78)
K = 3   # 事件前后各取 3 帧（30 min）


def event_study(direction):
    """direction='onset': free->capped; 'release': capped->free。返回每事件的 DiD 列表。"""
    est = []
    for i in HEALTHY:
        a, b = (free[i], capped[i]) if direction == "onset" else (capped[i], free[i])
        av, bv = a.values, b.values
        for p in np.flatnonzero(av[:-1] & bv[1:]):
            lo, hi = p - K + 1, p + 1 + K
            if lo < 0 or hi > len(av):
                continue
            if not (av[lo:p + 1].all() and bv[p + 1:hi].all()):
                continue
            d_i = wnd[i].values[p + 1:hi].mean() - wnd[i].values[lo:p + 1].mean()
            # 同期全程未限功的参照机
            for j in HEALTHY:
                if j == i or not free[j].values[lo:hi].all():
                    continue
                d_j = wnd[j].values[p + 1:hi].mean() - wnd[j].values[lo:p + 1].mean()
                est.append(d_i - d_j)
    return np.asarray(est)


for direction, label, sign in (("onset", "限功投入", "应为正"), ("release", "限功解除", "应为负")):
    e = event_study(direction)
    if len(e) < 10:
        print("%-8s 样本不足 (n=%d)" % (label, len(e)))
        OUT["event_study"].append({
            "direction": direction, "label": label, "expected_sign": sign,
            "n_event_pairs": int(len(e)), "reported": False,
            "delta_v_ms": None, "ci_low_ms": None, "ci_high_ms": None,
        })
        continue
    boot = np.array([RNG.choice(e, len(e), replace=True).mean() for _ in range(2000)])
    print("%-8s Δv = %+.3f m/s  95%%CI [%+.3f, %+.3f]  (事件对 n=%d, %s)" % (
        label, e.mean(), np.quantile(boot, 0.025), np.quantile(boot, 0.975), len(e), sign))
    OUT["event_study"].append({
        "direction": direction, "label": label, "expected_sign": sign,
        "n_event_pairs": int(len(e)), "reported": True,
        "delta_v_ms": float(e.mean()),
        "ci_low_ms": float(np.quantile(boot, 0.025)),
        "ci_high_ms": float(np.quantile(boot, 0.975)),
        "n_boot": 2000, "frames_each_side": K,
    })


# ---------- 落盘 ----------
OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, key in (("alta2_sample_base.csv", "sample_base"),
                  ("alta2_did_channels.csv", "channels"),
                  ("alta2_did_by_wind_band.csv", "wind_band"),
                  ("alta2_did_by_depth.csv", "depth"),
                  ("alta2_power_curve.csv", "power_curve"),
                  ("alta2_power_implication.csv", "power_implication"),
                  ("alta2_event_study.csv", "event_study")):
    pd.DataFrame(OUT[key]).to_csv(OUT_DIR / name, index=False)

with (OUT_DIR / "alta2_did_headline.json").open("w", encoding="utf-8") as fh:
    json.dump({"design": OUT["design"], "headline": OUT["headline"],
               "channels": OUT["channels"], "event_study": OUT["event_study"],
               "by_wind_band": OUT["wind_band"], "by_depth": OUT["depth"]},
              fh, ensure_ascii=False, indent=2)

print("\n已落盘: %s (7 CSV + 1 JSON)" % OUT_DIR)
