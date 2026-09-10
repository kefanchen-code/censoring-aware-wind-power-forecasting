# -*- coding: utf-8 -*-
"""Step 3: verify PowerRef semantics and characterize event timing."""
import numpy as np
import pandas as pd

PATH = "data/turbine_data/scada_df_ALTA2_20250904_20260301.parquet"
RATED = 1330.0
HEALTHY = ["1301253", "1301254", "1301255", "1301256", "1301257"]  # 1301252 excluded
df = pd.read_parquet(PATH)


def col(t, s):
    return df[(t, s)]


def generating(t):
    """Require healthy, grid-connected generation with plausible speed."""
    return ((col(t, "TurbinOK_TurbinOK_timeon") > 599)
            & (col(t, "ScInOper_ScInOper_timeon") > 599)
            & (col(t, "ActPower_Value_mean") > 0.05 * RATED)
            & (col(t, "GenRpm_Value_mean") > 800))


print("=== A. PowerRef/rated by wind speed during verified generation ===")
print("   A shutdown-default artifact should not remain fixed at 1.00 at low wind.")
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

print("\n=== B. Curtailment-state persistence (run length in 10-min frames) ===")
print("   External dispatch usually persists in blocks; framewise changes suggest internal control.")
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
        print("%-9s events=%4d  run length p50=%.0f p90=%.0f max=%.0f (frames)  "
              "total capped frames=%d" % (t, len(runs), np.median(runs),
                             np.quantile(runs, 0.9), runs.max(), capped.sum()))

print("\n=== C. Cross-turbine simultaneity (farm-level versus turbine-level) ===")
print("   Distribution of the number of curtailed healthy turbines per timestamp")
capped_mat = pd.DataFrame({
    t: (generating(t) & (col(t, "PowerRef_PowerRef_mean") < 0.95 * RATED)).astype(int)
    for t in HEALTHY})
gen_mat = pd.DataFrame({t: generating(t).astype(int) for t in HEALTHY})
both = gen_mat.sum(axis=1) >= 4  # Require at least four simultaneous generators.
n_capped = capped_mat[both].sum(axis=1)
print("   timestamps with >=4 turbines generating:", int(both.sum()))
print("   curtailed-turbine count distribution:", dict(n_capped.value_counts().sort_index().astype(int)))
print("   simultaneous farm-level curtailment share (>=4 turbines): %.1f%%" % (100 * (n_capped >= 4).mean()))

print("\n=== D. Curtailment depth versus farm wind speed ===")
farm_wind = pd.DataFrame({t: col(t, "AcWindSp_AcWindSp_mean") for t in HEALTHY}).mean(axis=1)
for t in HEALTHY:
    g = generating(t)
    r = col(t, "PowerRef_PowerRef_mean") / RATED
    m = g & (r < 0.95)
    if m.sum() < 50:
        continue
    rho = np.corrcoef(farm_wind[m], r[m])[0, 1]
    print("%-9s n=%5d  corr(farm wind, cap ratio)=%+.3f  "
          "cap-ratio p50 at low wind (<7m/s)=%.2f  high wind (>10m/s)=%.2f" % (
              t, int(m.sum()), rho,
              r[m & (farm_wind < 7)].median(), r[m & (farm_wind > 10)].median()))
