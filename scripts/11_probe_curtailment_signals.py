# -*- coding: utf-8 -*-
"""Step 3: 确认 PowerRef 语义（外部指令 vs 停机默认值）并刻画限功事件时间结构。"""
import numpy as np
import pandas as pd

PATH = "data/turbine_data/scada_df_ALTA2_20250904_20260301.parquet"
RATED = 1330.0
HEALTHY = ["1301253", "1301254", "1301255", "1301256", "1301257"]  # 1301252 已排除
df = pd.read_parquet(PATH)


def col(t, s):
    return df[(t, s)]


def generating(t):
    """明确处于发电状态：机组 OK 全窗、并网运行、有实际出力、转速正常。"""
    return ((col(t, "TurbinOK_TurbinOK_timeon") > 599)
            & (col(t, "ScInOper_ScInOper_timeon") > 599)
            & (col(t, "ActPower_Value_mean") > 0.05 * RATED)
            & (col(t, "GenRpm_Value_mean") > 800))


print("=== A. 仅限明确发电样本：PowerRef/rated 随风速的分布 ===")
print("   若 PowerRef 是停机默认值伪影，则此处低风速段不应再恒为 1.00")
bins = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 25)]
for t in HEALTHY:
    g = generating(t)
    v = col(t, "AcWindSp_AcWindSp_mean")
    r = col(t, "PowerRef_PowerRef_mean") / RATED
    out = []
    for lo, hi in bins:
        m = g & (v >= lo) & (v < hi)
        if m.sum() < 30:
            out.append("      -      ")
            continue
        q = r[m]
        out.append("%.2f/%.2f/%.2f" % (q.quantile(0.1), q.median(), q.quantile(0.9)))
    print("%-9s %s" % (t, "  ".join(out)))
print("         %s" % "  ".join("%-13s" % ("%d-%d m/s" % b) for b in bins))

print("\n=== B. 限功状态的时间连续性（run length, 单位=10min 帧） ===")
print("   外部调度指令通常成块持续；逐帧跳变更像内部控制")
for t in HEALTHY:
    g = generating(t)
    capped = (g & (col(t, "PowerRef_PowerRef_mean") < 0.95 * RATED)).astype(int)
    d = capped.diff().fillna(0)
    starts = np.flatnonzero(d.values == 1)
    ends = np.flatnonzero(d.values == -1)
    if len(starts) and len(ends):
        if ends[0] < starts[0]:
            ends = ends[1:]
        n = min(len(starts), len(ends))
        runs = ends[:n] - starts[:n]
        print("%-9s 事件数=%4d  run长度 p50=%.0f p90=%.0f max=%.0f (帧)  "
              "总限功帧=%d" % (t, len(runs), np.median(runs),
                             np.quantile(runs, 0.9), runs.max(), capped.sum()))

print("\n=== C. 跨机组同时性（电网级限功 vs 单机降功率） ===")
print("   统计每个时刻处于限功的健康机组数量分布")
capped_mat = pd.DataFrame({
    t: (generating(t) & (col(t, "PowerRef_PowerRef_mean") < 0.95 * RATED)).astype(int)
    for t in HEALTHY})
gen_mat = pd.DataFrame({t: generating(t).astype(int) for t in HEALTHY})
both = gen_mat.sum(axis=1) >= 4  # 至少4台同时在发电，才有可比性
n_capped = capped_mat[both].sum(axis=1)
print("   同时发电>=4台的时刻数:", int(both.sum()))
print("   其中限功机组数分布:", dict(n_capped.value_counts().sort_index().astype(int)))
print("   全场同时限功(>=4台)占比: %.1f%%" % (100 * (n_capped >= 4).mean()))

print("\n=== D. 限功深度与全场风速的关系（信息性删失的直接证据） ===")
farm_wind = pd.DataFrame({t: col(t, "AcWindSp_AcWindSp_mean") for t in HEALTHY}).mean(axis=1)
for t in HEALTHY:
    g = generating(t)
    r = col(t, "PowerRef_PowerRef_mean") / RATED
    m = g & (r < 0.95)
    if m.sum() < 50:
        continue
    rho = np.corrcoef(farm_wind[m], r[m])[0, 1]
    print("%-9s n=%5d  corr(全场风速, 限功深度比)=%+.3f  "
          "低风(<7m/s)时深度p50=%.2f  高风(>10m/s)时深度p50=%.2f" % (
              t, int(m.sum()), rho,
              r[m & (farm_wind < 7)].median(), r[m & (farm_wind > 10)].median()))
