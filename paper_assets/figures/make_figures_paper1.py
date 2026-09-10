"""Generate the six quantitative Paper 1 figures from saved result summaries.

Figure contract:
- Figure 1: censoring and information separation; control limits convert
  latent truth into a lower bound, and latent truth is used only for scoring.
- Figure 2: main results; CL-PMF recovers the loss of PL and approaches ORC,
  while S5 coverage collapses.
- Figure 3: matched-factor chain and WIS-component decomposition; gains arise
  from the likelihood and reduced underprediction.
- Figure 4: S5 coverage curve; coverage collapses at 90%, with partial
  retention by CL-Tobit.
- Figure 5: S0 equivalence and contamination robustness; exact zero remains
  zero and the advantage persists under contamination.
- Figure 6: cross-turbine direction and LOTO/LOFO results, with reversals
  reported explicitly.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJ = Path(__file__).resolve().parents[2]
OUT = PROJ / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = PROJ / "reference_results"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

# 展示名映射
NAME = {"B4": "PL", "B4_plain": "PL-basic", "B6": "CL-PMF", "BT": "CL-Tobit",
        "ORACLE": "ORC", "BQR": "QR", "CLQR": "CL-QR", "B1": "DEL",
        "B2_recon": "REC", "B0_climatology": "Clim.", "B0_persistence": "Pers."}
SCEN = {"S1_fixed_50pct": "S1", "S2_fixed_70pct": "S2", "S3_fixed_85pct": "S3",
        "S4_random_uniform": "S4", "S5_wind_triggered": "S5"}

COL = {"PL": "#4C72B0", "CL-PMF": "#55A868", "CL-Tobit": "#C44E52",
       "ORC": "#4D4D4D", "QR": "#8172B2", "CL-QR": "#64B5CD",
       "PL-basic": "#9AB0C6", "DEL": "#CCB974", "REC": "#DD8452"}


def save_pub(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def set_ticks(ax, ticks, labels, axis="x", **kw):
    """Set tick positions and labels separately for older Matplotlib."""
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, **kw)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, **kw)


# ================== 图 1：删失机制与信息隔离示意 ==================
def fig1():
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)

    # 左：真实底材上的删失结构（半合成 S1 底材取一段）
    ax = fig.add_subplot(gs[0])
    pq = pd.read_parquet(RES / "altahullion_audit" / "synthetic_S1_fixed_50pct.parquet")
    colmap = {c.lower(): c for c in pq.columns}
    a_col = colmap.get("a_true")
    c_col = colmap.get("c_syn")
    y_col = colmap.get("y_syn")
    if a_col and c_col:
        a = pq[a_col].to_numpy(dtype=float)
        c = pq[c_col].to_numpy(dtype=float)
        y = pq[y_col].to_numpy(dtype=float) if y_col else np.minimum(a, c)
        if pq.shape[0] > 900:  # 取一段含删失的窗口
            cen = np.where((a > c) & np.isfinite(c))[0]
            i0 = max(0, int(np.median(cen)) - 450)
            a, c, y = a[i0:i0 + 900], c[i0:i0 + 900], y[i0:i0 + 900]
        t = np.arange(len(a))
        cen_mask = pq["is_censored"].to_numpy(dtype=bool)[i0:i0 + 900] if "is_censored" in pq.columns \
            else (a > c) & np.isfinite(c)
        vmax = np.nanmax(a) * 1.05
        ax.plot(t, a / vmax, color="#999999", lw=0.9, label="Potential power $A_t$ (truth)")
        ax.plot(t, c / vmax, color="#C44E52", lw=0.9, ls="--", label="Control cap $C_t$")
        ax.plot(t, y / vmax, color="#222222", lw=1.1, label="Observed $Y_t=\\min(A_t,C_t)$")
        # 删失窗着色（连续段）
        seg = np.flatnonzero(cen_mask)
        if len(seg):
            starts = np.split(seg, np.where(np.diff(seg) > 1)[0] + 1)
            for s in starts:
                if len(s) >= 10:
                    ax.axvspan(s[0], s[-1], color="#C44E52", alpha=0.10, lw=0)
            s0 = starts[0]
            ax.annotate("censored window\n(observation = lower bound)",
                        xy=(s0[len(s0) // 2], 0.62), xytext=(s0[len(s0) // 2] + 130, 0.78),
                        fontsize=6.5, ha="left",
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88),
                        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))
    else:  # 回退：合成示意
        t = np.arange(600)
        a = 0.5 + 0.35 * np.sin(t / 70) + np.random.default_rng(0).normal(0, 0.04, 600)
        c = np.full(600, np.nan); c[200:420] = 0.55
        y = np.where(np.isnan(c), a, np.minimum(a, c))
        ax.plot(t, a, color="#999999", lw=0.9, label="Potential power $A_t$ (truth)")
        ax.plot(t, c, color="#C44E52", lw=0.9, ls="--", label="Control cap $C_t$")
        ax.plot(t, y, color="#222222", lw=1.1, label="Observed $Y_t=\\min(A_t,C_t)$")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("power (p.u. of rated)")
    ax.set_title("(a) Curtailment turns truth into a lower bound", loc="left", fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=6, loc="upper right", handlelength=1.6)

    # 右：信息隔离
    bx = fig.add_subplot(gs[1]); bx.axis("off")
    bx.set_title("(b) Information separation in the benchmark", loc="left", fontsize=7.5)

    def box(x, y, w, h, text, fc="#F2F2F2", ec="#4D4D4D"):
        bx.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=0.9))
        bx.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.3)

    box(0.02, 0.72, 0.42, 0.20, "Training view\n$Y_t$, wind, $C_t$, $\\delta_t$ (60 min)")
    box(0.56, 0.72, 0.42, 0.20, "Model\n(TCN, any loss)")
    box(0.56, 0.30, 0.42, 0.20, "Probabilistic forecast\n$\\hat{F}_{t+15|t}$")
    box(0.02, 0.06, 0.42, 0.20, "Truth $A_t$ (semi-synthetic)", fc="#E8F0E8", ec="#55A868")
    box(0.56, 0.06, 0.42, 0.20, "Scoring (WIS / CRPS)", fc="#E8F0E8", ec="#55A868")
    bx.annotate("", xy=(0.56, 0.82), xytext=(0.44, 0.82),
                arrowprops=dict(arrowstyle="->", lw=0.9))
    bx.annotate("", xy=(0.77, 0.72), xytext=(0.77, 0.50),
                arrowprops=dict(arrowstyle="->", lw=0.9))
    bx.annotate("", xy=(0.56, 0.16), xytext=(0.44, 0.16),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#55A868"))
    # 禁止箭头：真值不进训练
    bx.annotate("", xy=(0.23, 0.72), xytext=(0.23, 0.26),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#C44E52", ls=(0, (3, 2))))
    bx.plot([0.13, 0.33], [0.49, 0.49], color="#C44E52", lw=2.2)
    bx.text(0.50, 0.55, "truth never enters training", fontsize=6.3, color="#C44E52",
            ha="center", va="center")
    bx.set_xlim(0, 1); bx.set_ylim(0, 1)
    save_pub(fig, "fig1_censoring_mechanism")


# ================== 图 2：主结果 ==================
def fig2():
    ms = pd.read_csv(RES / "pap_benchmark_semisyn" / "metrics_summary.csv")
    st = pd.read_csv(RES / "pap_benchmark_semisyn" / "metrics_stratified.csv")
    scens = ["S1_fixed_50pct", "S2_fixed_70pct", "S3_fixed_85pct",
             "S4_random_uniform", "S5_wind_triggered"]
    xs = np.arange(len(scens))
    models = ["B4", "B6", "ORACLE"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    for k, m in enumerate(models):
        vals = [ms[(ms.scenario == s) & (ms.model == m)].wis_mean_mean.iloc[0] for s in scens]
        style = dict(marker="o", ms=4.5, lw=1.2, color=COL[NAME[m]])
        if m == "ORACLE":
            ax.plot(xs, vals, ls="--", marker="s", ms=3.5, lw=1.0, color=COL["ORC"],
                    label="ORC (full-truth reference)")
        else:
            ax.plot(xs, vals, **style, label=NAME[m])
            for i, v in enumerate(vals):
                if m == "B6":
                    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                                xytext=(0, -9), fontsize=5.6, ha="center", color=COL["CL-PMF"])
    ax.annotate("S5: informative\ncensoring", xy=(4, ms[(ms.scenario == scens[4]) & (ms.model == "B6")].wis_mean_mean.iloc[0]),
                xytext=(2.6, 0.155), fontsize=6.3, color="#C44E52",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#C44E52"))
    set_ticks(ax, xs, [SCEN[s] for s in scens])
    ax.set_xlabel("scenario")
    ax.set_ylabel("WIS (full sample)")
    ax.set_title("(a) Censoring-aware likelihood recovers the loss", loc="left", fontsize=7.5)
    ax.set_ylim(0.05, 0.20)
    ax.legend(fontsize=6, loc="upper left")

    ax = axes[1]
    cstr = st[(st.dimension == "target_censoring") & (st.stratum == "censored")]
    for m in ["B4", "B6", "ORACLE"]:
        vals = [cstr[(cstr.scenario == s) & (cstr.model == m)].coverage_90.iloc[0] for s in scens]
        if m == "ORACLE":
            ax.plot(xs, vals, ls="--", marker="s", ms=3.5, lw=1.0, color=COL["ORC"], label="ORC")
        else:
            ax.plot(xs, vals, marker="o", ms=4.5, lw=1.2, color=COL[NAME[m]], label=NAME[m])
    ax.axhline(0.90, color="#999999", lw=0.8, ls=":")
    ax.text(4.42, 0.90, "nominal 0.90", fontsize=5.8, color="#777777", va="center")
    ax.annotate("collapse\n(0.051)", xy=(4, 0.051), xytext=(2.35, 0.10), fontsize=6.3,
                color="#C44E52", arrowprops=dict(arrowstyle="->", lw=0.7, color="#C44E52"))
    set_ticks(ax, xs, [SCEN[s] for s in scens])
    ax.set_xlabel("scenario")
    ax.set_ylabel("90% coverage on censored targets")
    ax.set_title("(b) Coverage on censored targets", loc="left", fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6, loc="lower right")
    save_pub(fig, "fig2_main_results")


# ================== 图 3：单因子链 + 三分量 ==================
def fig3():
    st = pd.read_csv(RES / "pap_benchmark_semisyn" / "metrics_stratified.csv")
    wc = pd.read_csv(RES / "pap_benchmark_semisyn" / "wis_components_contrast.csv")
    cstr = st[(st.dimension == "target_censoring") & (st.stratum == "censored")]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7),
                             gridspec_kw={"wspace": 0.5})
    ax = axes[0]
    chain = ["B4_plain", "B4", "B6", "BT", "ORACLE"]
    scens = ["S1_fixed_50pct", "S2_fixed_70pct", "S3_fixed_85pct",
             "S4_random_uniform", "S5_wind_triggered"]
    cmap = plt.get_cmap("Greys")
    for i, s in enumerate(scens):
        vals = [cstr[(cstr.scenario == s) & (cstr.model == m)].wis_mean.iloc[0] for m in chain]
        ax.plot(range(5), vals, color=cmap(0.35 + 0.13 * i), lw=0.8, alpha=0.9,
                label=SCEN[s] if i in (0, 4) else None)
        for j, v in enumerate(vals):
            ax.plot(j, v, "o", ms=2.4, color=cmap(0.35 + 0.13 * i))
    ax.axvspan(-0.5, 1.5, color="#4C72B0", alpha=0.06)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "point losses", fontsize=6, ha="center", color="#4C72B0")
    ax.axvspan(1.5, 3.5, color="#55A868", alpha=0.06)
    ax.text(2.5, ax.get_ylim()[1] * 0.97, "censoring-aware likelihood", fontsize=6, ha="center", color="#3D7A4A")
    set_ticks(ax, range(5), [NAME[m] for m in chain])
    ax.set_xlabel("single-factor chain (one factor changes per step)")
    ax.set_ylabel("WIS on censored targets")
    ax.set_title("(a) Attribution chain: adding control features does not help", loc="left", fontsize=7.5)
    ax.legend(fontsize=6, ncol=2, loc="upper right")

    ax = axes[1]
    prim = wc[(wc.family == "RQ1_censored_likelihood_primary") & (wc.window_type == "censored")
              & (wc.reference == "B4") & (wc.challenger == "B6")]
    prim = prim.groupby("scenario").mean(numeric_only=True)
    xs = np.arange(5)
    # delta_wis = challenger - reference（负=CL-PMF 更优）；分量份额翻转不变，仅 ΔWIS 绝对值翻转为改进量
    under = [prim.loc[s, "fraction_of_delta_underprediction"] for s in scens]
    over = [prim.loc[s, "fraction_of_delta_overprediction"] for s in scens]
    disp = [prim.loc[s, "fraction_of_delta_dispersion"] for s in scens]
    dw = [-prim.loc[s, "delta_wis"] for s in scens]
    ax.bar(xs, under, color="#C44E52", width=0.6, label="underprediction")
    ax.bar(xs, over, bottom=under, color="#4C72B0", width=0.6, label="overprediction")
    ax.bar(xs, disp, bottom=np.array(under) + np.array(over), color="#AAAAAA", width=0.6,
           label="dispersion")
    ax.axhline(1.0, color="#444444", lw=0.8, ls=":")
    ax.text(4.45, 1.0, "=100%\nof \u0394WIS", fontsize=5.8, color="#555555", va="center")
    for i, (u, d) in enumerate(zip(under, dw)):
        ax.text(i, u / 2, f"{u * 100:.0f}%", fontsize=5.6, ha="center", va="center", color="white")
        ax.text(i, -0.32, f"\u0394WIS\n{d:+.3f}", fontsize=5.6, ha="center", va="top", color="#333333")
    set_ticks(ax, xs, [SCEN[s] for s in scens])
    ax.set_ylim(-0.55, 1.50)
    ax.set_xlabel("scenario (PL \u2192 CL-PMF, censored windows)")
    ax.set_ylabel("share of \u0394WIS improvement by component")
    ax.set_title("(b) Gain comes from underprediction repair", loc="left", fontsize=7.5)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, borderaxespad=0)
    fig.subplots_adjust(bottom=0.24)
    save_pub(fig, "fig3_attribution")


# ================== 图 4：S5 覆盖率曲线 ==================
def fig4():
    st = pd.read_csv(RES / "pap_benchmark_semisyn" / "metrics_stratified.csv")
    cstr = st[(st.dimension == "target_censoring") & (st.stratum == "censored")
              & (st.scenario == "S5_wind_triggered")]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    nom = [0.40, 0.60, 0.80, 0.90]
    for m, sty in [("B4", dict(ls="-", marker="x")), ("B6", dict(ls="-", marker="o")),
                   ("BT", dict(ls="-", marker="^")), ("ORACLE", dict(ls="--", marker="s"))]:
        row = cstr[cstr.model == m].iloc[0]
        cov = [row[f"coverage_{int(n * 100)}"] for n in nom]
        lbl = NAME[m] + (" (full-truth reference)" if m == "ORACLE" else "")
        ax.plot(nom, cov, color=COL[NAME[m]], lw=1.2, ms=4.5, label=lbl, **sty)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=":")
    ax.annotate("CL-PMF collapses\nat 90%", xy=(0.90, cstr[cstr.model == "B6"].iloc[0]["coverage_90"]),
                xytext=(0.45, 0.62), fontsize=6.3, color="#3D7A4A",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#3D7A4A"))
    ax.annotate("CL-Tobit partially\nretained", xy=(0.90, cstr[cstr.model == "BT"].iloc[0]["coverage_90"]),
                xytext=(0.52, 0.36), fontsize=6.3, color="#C44E52",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#C44E52"))
    ax.set_xlabel("nominal coverage level")
    ax.set_ylabel("empirical coverage (S5, censored targets)")
    ax.set_title("Coverage under informative censoring (S5)", loc="left", fontsize=7.5)
    ax.set_xlim(0.35, 0.95); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6, loc="lower right")
    save_pub(fig, "fig4_s5_coverage")


# ================== 图 5：S0 等价 + 污染稳健性 ==================
def fig5():
    cmp = json.loads((RES / "pap_benchmark_s0_control" / "comparisons.json").read_text(encoding="utf-8"))
    adv = pd.read_csv(RES / "contamination_robustness" / "contamination_advantage.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8),
                             gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    comps = [c for c in cmp["comparisons"] if c["family"] == "S0_negative_control"]
    labels, ys = [], []
    for i, c in enumerate(comps):
        d = c["difference_reference_minus_challenger"]
        lo, hi = c["cluster_bootstrap_ci_95"]
        lbl = f"{NAME.get(c['reference'], c['reference'])} \u2212 {NAME.get(c['challenger'], c['challenger'])}"
        labels.append(lbl); ys.append(i)
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", ms=4,
                    color="#333333", elinewidth=1.0, capsize=2.5)
    margin = comps[0].get("equivalence_margin", 0.005)
    ax.axvspan(-margin, margin, color="#55A868", alpha=0.13)
    ax.axvline(0, color="#444444", lw=0.8)
    set_ticks(ax, ys, labels, axis="y")
    ax.set_xlabel("paired WIS difference (ref \u2212 challenger), S0")
    ax.set_title(f"(a) Negative control: exactly zero\n(equivalence band \u00b1{margin})",
                 loc="left", fontsize=7.5)
    ax.set_ylim(-0.7, len(ys) - 0.3)

    ax = axes[1]
    fam = adv[(adv.family == "RQ1_censored_likelihood_primary") & (adv.window_type == "censored")]
    modes = sorted(fam.contamination_mode.unique())
    base_mode = "none" if "none" in modes else modes[0]
    contam_modes = [m for m in modes if m != base_mode]
    # 取实测污染档（relative，intensity=1.0，锚定 ALTA2 DiD +16.7%）
    cm = contam_modes[0]
    sub_c = fam[fam.contamination_mode == cm]
    ints = sorted(sub_c.intensity_factor.unique())
    target = max(ints)
    sub_c = sub_c[sub_c.intensity_factor == target]
    sub_b = fam[fam.contamination_mode == base_mode]
    scens = [s for s in ["S1_fixed_50pct", "S2_fixed_70pct", "S3_fixed_85pct",
                         "S4_random_uniform", "S5_wind_triggered"]]
    xs = np.arange(len(scens))
    for off, sub, tag, colr in [(-0.12, sub_b, "clean", "#4C72B0"),
                                (0.12, sub_c, "contaminated\n(measured bias \u00d71.167)", "#DD8452")]:
        d = [sub[sub.scenario == s].difference_reference_minus_challenger.iloc[0] for s in scens]
        lo = [sub[sub.scenario == s].ci_low.iloc[0] for s in scens]
        hi = [sub[sub.scenario == s].ci_high.iloc[0] for s in scens]
        ax.errorbar(xs + off, d, yerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)],
                    fmt="o", ms=4, color=colr, elinewidth=1.0, capsize=2.5, label=tag)
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    set_ticks(ax, xs, [SCEN[s] for s in scens])
    ax.set_xlabel("scenario")
    ax.set_ylabel("CL-PMF advantage over PL (\u0394WIS, censored)")
    ax.set_title("(b) Advantage retained under measured\ninput contamination", loc="left", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower right")
    save_pub(fig, "fig5_negative_control_contamination")


# ================== 图 6：多风机 + LOTO/LOFO ==================
def fig6():
    runs_dir = RES / "multi_turbine_validation" / "runs"
    turbines = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    rows = []
    for tb in turbines:
        m = pd.read_csv(runs_dir / tb / "metrics_summary.csv")
        for s in ["S1_fixed_50pct", "S4_random_uniform", "S5_wind_triggered"]:
            g = m[m.scenario == s]
            b4 = g[g.model == "B4"].wis_mean_mean.iloc[0]
            for mm in ["B6", "CLQR"]:
                v = g[g.model == mm].wis_mean_mean.iloc[0]
                rows.append(dict(turbine=tb, scenario=SCEN[s], model=NAME[mm], delta=v - b4))
    mt = pd.DataFrame(rows)

    ll = pd.read_csv(RES / "loto_lofo" / "loto_lofo_results.csv")
    fold_rows = []
    for (exp, sc, fo), g in ll.groupby(["fold", "scenario", "held_out"]):
        b4 = g[g.model == "B4"].wis_mean.mean()
        b6 = g[g.model == "B6"].wis_mean.mean()
        fold_rows.append(dict(exp=exp, scenario=SCEN[sc], fold=fo, delta=b6 - b4))
    lf = pd.DataFrame(fold_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.55})
    ax = axes[0]
    scens = ["S1", "S4", "S5"]
    tb_short = [t.replace("ALTA2_", "A-").replace("HOT_", "H-") for t in turbines]
    ys = np.arange(len(turbines))
    for k, s in enumerate(scens):
        sub = mt[(mt.scenario == s) & (mt.model == "CL-QR")].set_index("turbine")
        vals = [sub.loc[t, "delta"] for t in turbines]
        col = ["#4C72B0", "#55A868", "#C44E52"][k]
        ax.scatter(vals, ys + (k - 1) * 0.22, s=14, color=col, label=f"CL-QR vs PL, {s}")
        n_neg = sum(v < 0 for v in vals)
    ax.axvline(0, color="#444444", lw=0.8)
    set_ticks(ax, ys, tb_short, axis="y", fontsize=5.8)
    ax.axhspan(-0.5, 4.5, color="#4C72B0", alpha=0.05)
    ax.text(ax.get_xlim()[0] if False else -0.118, 4.55, "ALTA2", fontsize=5.8, color="#4C72B0")
    ax.text(-0.118, 5.0 + 2.5, "HOT", fontsize=5.8, color="#DD8452")
    sub5 = mt[(mt.scenario == "S5") & (mt.model == "CL-QR")].set_index("turbine")
    pos_t = [t for t in turbines if sub5.loc[t, "delta"] > 0]
    for t in pos_t:
        i = turbines.index(t)
        ax.annotate("1/13 reversal", xy=(sub5.loc[t, "delta"], i + 0.22),
                    xytext=(0.008, i + 0.4), fontsize=5.6, color="#C44E52",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#C44E52"))
    ax.set_xlabel("\u0394WIS (censoring-aware \u2212 PL); negative = better")
    ax.set_title("(a) 13 turbines, 2 farms (CL-QR vs PL)", loc="left", fontsize=7.5)
    ax.legend(fontsize=5.8, loc="lower right")

    ax = axes[1]
    exps = ["loto_alta2", "loto_hot", "lofo_alta2_to_hot", "lofo_hot_to_alta2"]
    exp_lbl = ["LOTO\nALTA2", "LOTO\nHOT", "LOFO\nALTA2\u2192HOT", "LOFO\nHOT\u2192ALTA2"]
    xs = np.arange(len(exps))
    for k, s in enumerate(["S1", "S4"]):
        means, lows = [], []
        pts_x, pts_y = [], []
        for j, e in enumerate(exps):
            sub = lf[(lf.exp == e) & (lf.scenario == s)]
            d = sub.delta.to_numpy()
            means.append(d.mean())
            pts_x.extend([j + (k - 0.5) * 0.24] * len(d))
            pts_y.extend(d)
        col = ["#4C72B0", "#DD8452"][k]
        ax.scatter(pts_x, pts_y, s=10, color=col, alpha=0.55)
        ax.scatter(xs + (k - 0.5) * 0.24, means, marker="_", s=22, color=col, zorder=5)
        ax.plot([], [], color=col, lw=1.4, label=f"{s} (fold means)")
    ax.axhline(0, color="#444444", lw=0.8)
    set_ticks(ax, xs, exp_lbl, fontsize=5.8)
    ax.set_ylim(bottom=min(lf.delta.min() * 1.25, -0.01), top=max(lf.delta.max() * 1.25, 0.01))
    ax.annotate("advantage reverses\n(0/5 folds in favor)", xy=(3, lf[(lf.exp == "lofo_hot_to_alta2") & (lf.scenario == "S1")].delta.mean()),
                xytext=(1.6, ax.get_ylim()[1] * 0.75), fontsize=5.8, color="#C44E52",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#C44E52"))
    ax.set_ylabel("\u0394WIS (CL-PMF \u2212 PL), held-out test")
    ax.set_title("(b) Leave-one-turbine / leave-one-farm", loc="left", fontsize=7.5)
    ax.legend(fontsize=5.8, loc="upper right")
    save_pub(fig, "fig6_external_validity")


def build_parser():
    """Build command-line options for result and output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJ / "reference_results",
        help="saved result tree (default: reference_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJ / "figures",
        help="directory for SVG, PDF, and PNG outputs",
    )
    return parser


def main():
    """Generate every supported paper figure."""

    global RES, OUT
    arguments = build_parser().parse_args()
    RES = arguments.results_dir.resolve()
    OUT = arguments.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("all figures done ->", OUT)


if __name__ == "__main__":
    main()
