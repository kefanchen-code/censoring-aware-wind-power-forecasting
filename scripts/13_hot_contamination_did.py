# -*- coding: utf-8 -*-
"""Step 13: 在 Hill of Towie 上复现输入通道污染的双重差分实测。

移植 ``scripts/12_measure_censoring_contamination.py``（Altahullion/ALTA2，
1330 kW）到第二个风场（Hill of Towie，SWT-2.3-VS-82，2300 kW），以给出
两风场、两机型的外部有效性。识别策略与判据保持一致：

    污染量 = E[v_i - v_j | i 限功] - E[v_i - v_j | i,j 均未限功]

差异之处（均已核实，写入结果以便论文如实声明）：
1. 状态位可用性更好。HOT 的 ``wtc_ScInOper_timeon`` 与 ALTA2 的
   ``ScInOper_timeon`` 同义，故 ``generating()`` 判据无需退化；转速门限按
   本机型额定发电机转速（~1550 rpm）重新标定，不沿用 ALTA2 的 800 rpm。
2. HOT 的限功不是场级同步的：单机限功占绝大多数（见"样本基础"输出），
   因此可用于 DiD 的时刻比 ALTA2 更充裕，参照机也更多（21 台）。
3. 21 台机给出 420 个有序机对，直接展开配对长表会到千万行量级。这里改为
   按 (机对 x 风速档 x 日 x 处理状态) 预聚合求和与计数，日块 bootstrap 则对
   日权重做加权汇总——与"复制被抽中的天"逐位等价，但内存与耗时可控。

输出 ``results/hot_contamination_did/``：
- ``hot_did_channels.csv``   主结果 A（各通道 DiD 与日块 bootstrap CI）
- ``hot_did_by_wind_band.csv`` 按参照机风速分层的加性/乘性偏差
- ``hot_did_by_depth.csv``   按限功档位分层（机制检验）
- ``hot_power_curve.csv``    未限功样本经验功率曲线与局部斜率
- ``hot_event_study.csv``    事件研究交叉验证
- ``hot_did_headline.json``  供 ``14_hot_truth_channel_bias.py`` 消费
- ``two_farm_contamination.csv`` 与 ALTA2 并列的两风场表

用法：
    python scripts/13_hot_contamination_did.py
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
HOT_ZIP = PROJECT_DIR / "data" / "hill_of_towie" / "2026.zip"
OUT_DIR = PROJECT_DIR / "results" / "hot_contamination_did"

RATED = 2300.0          # kW, SWT-2.3-VS-82
RATED_GEN_RPM = 1550.0  # 观测到的 GenRpm 上界 ~1554 rpm
STATION_OFFSET = 2304509
MONTHS = ["2026_01", "2026_02", "2026_03", "2026_04"]
N_BOOT = 5000
RNG = np.random.default_rng(42)

# 与 ALTA2 一致的判据参数
REF_FREE = 0.99         # PowerRef/额定 > 0.99 视为未限功
REF_CAP = 0.95          # PowerRef/额定 < 0.95 视为有限功指令
TRACK_LOW, TRACK_HIGH = 0.90, 1.08   # 指令被跟踪（功率真被钳住）
MIN_POWER_FRAC = 0.05
MIN_RPM_FRAC = 0.5      # 0.5 x 额定发电机转速；ALTA2 用 800 rpm 的等价标定
V_BINS = [0, 5, 6, 7, 8, 9, 10, 12, 30]
DEPTH_EDGES = [0, 0.25, 0.4, 0.6, 0.95]
DEPTH_LABELS = ["<0.25", "0.25-0.4", "0.4-0.6", "0.6-0.95"]
DIR_SECTOR_DEG = 30    # 风向扇区宽度（设计检验失败时的分层依据）
K_EVENT = 3             # 事件前后各 3 帧（30 min）

STRATA_PLAIN = ("pair", "v_bin")
STRATA_DIR = ("pair", "v_bin", "dir_bin")

# ALTA2 实测锚点（scripts/12_measure_censoring_contamination.py 的主结果，
# 半年 10-min SCADA、5 台健康机）。此处仅用于并列成表，不参与任何估计。
ALTA2_REFERENCE = {
    "farm": "Altahullion (ALTA2)",
    "turbine_type": "1330 kW",
    "n_reference_turbines": 5,
    "basis_minutes": 10,
    "delta_v_ms": 1.45,
    "delta_v_ci_low_ms": np.nan,
    "delta_v_ci_high_ms": np.nan,
    "relative_delta": 0.167,
    "relative_ci_low": 0.132,
    "relative_ci_high": 0.206,
    "placebo_significant": False,
    "event_study_delta_v_ms": 1.02,
    "provenance": "scripts/12_measure_censoring_contamination.py",
}

CHANNELS = (
    ("dv", "m/s", "机舱风速"),
    ("dv_rel", "-", "风速相对偏差"),
    ("dpitch", "deg", "桨距角A"),
    ("drpm", "rpm", "发电机转速"),
    ("dti", "-", "湍流强度"),
    ("damb", "degC", "环境温度[安慰剂]"),
    ("dyaw", "deg", "偏航位置[设计检验]"),
)
VALUES = [name for name, _, _ in CHANNELS]
COUNT_COLS = ["c_" + name for name in VALUES]
SUM_COLS = VALUES + COUNT_COLS + ["n"]


# ------------------------------------------------------------------ 数据装载
def read_month(archive, month):
    """一个月的 10-min 表，按 (TimeStamp, StationId) 内连接所需五张表。"""

    def read(table, columns):
        return pd.read_csv(
            io.BytesIO(archive.read("%s_%s.csv" % (table, month))),
            usecols=["TimeStamp", "StationId"] + columns,
        )

    frame = read(
        "tblSCTurbine",
        [
            "wtc_PowerRef_endvalue",
            "wtc_AcWindSp_mean",
            "wtc_AcWindSp_stddev",
            "wtc_PitcPosA_mean",
            "wtc_GenRpm_mean",
            "wtc_YawPos_mean",
            "wtc_ActualWindDirection_mean",
        ],
    )
    for table, columns in (
        ("tblSCTurGrid", ["wtc_ActPower_mean"]),
        ("tblSCTurFlag", ["wtc_ScInOper_timeon"]),
        ("tblSCTurTemp", ["wtc_AmbieTmp_mean"]),
        ("tblSCTurDigiIn", ["wtc_PowerRed_endvalue"]),
    ):
        frame = frame.merge(read(table, columns), on=["TimeStamp", "StationId"], how="inner")
    return frame


def load_wide():
    """返回宽表字典：每个量一个 (时间 x 机组) DataFrame。"""

    if not HOT_ZIP.exists():
        raise SystemExit("missing %s" % HOT_ZIP)
    with zipfile.ZipFile(HOT_ZIP) as archive:
        frames = [read_month(archive, month) for month in MONTHS]
    long = pd.concat(frames, ignore_index=True)
    long["TimeStamp"] = pd.to_datetime(long["TimeStamp"])
    long["turbine"] = (long["StationId"] - STATION_OFFSET).map("T{:02d}".format)
    long["ti"] = long["wtc_AcWindSp_stddev"] / long["wtc_AcWindSp_mean"]
    long["ref_pu"] = long["wtc_PowerRef_endvalue"] / RATED
    long["power_pu"] = long["wtc_ActPower_mean"] / RATED
    renamed = {
        "wtc_AcWindSp_mean": "wind",
        "wtc_PitcPosA_mean": "pitch",
        "wtc_GenRpm_mean": "rpm",
        "wtc_YawPos_mean": "yaw",
        "wtc_ActualWindDirection_mean": "wdir",
        "wtc_AmbieTmp_mean": "amb",
        "wtc_ScInOper_timeon": "in_oper",
        "wtc_PowerRed_endvalue": "power_red",
    }
    long = long.rename(columns=renamed)
    keys = [
        "wind",
        "pitch",
        "rpm",
        "yaw",
        "wdir",
        "amb",
        "ti",
        "ref_pu",
        "power_pu",
        "in_oper",
        "power_red",
    ]
    wide = {
        key: long.pivot_table(index="TimeStamp", columns="turbine", values=key)
        for key in keys
    }
    return wide


def build_masks(wide):
    """generating / capped / free 三个布尔宽表，判据与 ALTA2 一一对应。"""

    generating = (
        (wide["in_oper"] > 599)
        & (wide["power_pu"] > MIN_POWER_FRAC)
        & (wide["rpm"] > MIN_RPM_FRAC * RATED_GEN_RPM)
    )
    tracked = (wide["power_pu"] > TRACK_LOW * wide["ref_pu"]) & (
        wide["power_pu"] < TRACK_HIGH * wide["ref_pu"]
    )
    capped = generating & (wide["ref_pu"] < REF_CAP) & tracked
    free = generating & (wide["ref_pu"] > REF_FREE)
    return generating.fillna(False), capped.fillna(False), free.fillna(False)


# -------------------------------------------------------------- 配对与预聚合
def pair_aggregate(wide, capped, free, turbines):
    """按 (机对, 风速档, 风向扇区, 日, 处理状态, 限功档位) 预聚合求和与计数。

    与逐行长表在数学上等价：DiD 只用到分层内的均值，而日块 bootstrap 只改变
    每一天的重复次数，因此保留每日的 (sum, count) 即为充分统计量。逐通道单独计数
    （``c_*``），以免某些通道的缺测被当成零拉低均值。
    """

    index = wide["wind"].index
    day = pd.Series(index, index=index).dt.floor("D")
    aggregates = []
    for i in turbines:
        if not capped[i].any():
            continue
        for j in turbines:
            if i == j:
                continue
            treat = (capped[i] & free[j]).to_numpy()
            base = (free[i] & free[j]).to_numpy()
            if not treat.any() or not base.any():
                continue
            for mask, kind in ((treat, "treat"), (base, "base")):
                selected = np.flatnonzero(mask)
                block = pd.DataFrame(
                    {
                        "dv": (wide["wind"][i] - wide["wind"][j]).to_numpy()[selected],
                        "dv_rel": (
                            (wide["wind"][i] - wide["wind"][j]) / wide["wind"][j]
                        ).to_numpy()[selected],
                        "dpitch": (wide["pitch"][i] - wide["pitch"][j]).to_numpy()[selected],
                        "drpm": (wide["rpm"][i] - wide["rpm"][j]).to_numpy()[selected],
                        "dti": (wide["ti"][i] - wide["ti"][j]).to_numpy()[selected],
                        "damb": (wide["amb"][i] - wide["amb"][j]).to_numpy()[selected],
                        "dyaw": (wide["yaw"][i] - wide["yaw"][j]).to_numpy()[selected],
                        "v_ref": wide["wind"][j].to_numpy()[selected],
                        "dir_ref": wide["wdir"][j].to_numpy()[selected],
                        "depth": wide["ref_pu"][i].to_numpy()[selected],
                        "external": wide["power_red"][i].to_numpy()[selected],
                        "day": day.to_numpy()[selected],
                    }
                )
                block["pair"] = "%s>%s" % (i, j)
                block["kind"] = kind
                block["v_bin"] = pd.cut(block["v_ref"], V_BINS)
                # 扇区按参照机 j 的风向取值（j 未被限功，因而未被污染），
                # 与 v_bin 用 v_ref 而非被污染的 v_i 同一道理。
                block["dir_bin"] = np.floor(
                    (block["dir_ref"] % 360.0) / DIR_SECTOR_DEG
                )
                block["depth_bin"] = (
                    pd.cut(block["depth"], DEPTH_EDGES, labels=DEPTH_LABELS)
                    if kind == "treat"
                    else "none"
                )
                block["ext_bin"] = (
                    np.where(block["external"] > 0, "external", "internal")
                    if kind == "treat"
                    else "none"
                )
                grouped = block.groupby(
                    ["pair", "kind", "v_bin", "dir_bin", "depth_bin", "ext_bin", "day"],
                    observed=True,
                )
                summed = grouped[VALUES].sum()
                counts = grouped[VALUES].count()
                counts.columns = COUNT_COLS
                summed = pd.concat([summed, counts], axis=1)
                summed["n"] = grouped.size()
                aggregates.append(summed.reset_index())
    if not aggregates:
        raise SystemExit("no usable treated/control pair was found")
    frame = pd.concat(aggregates, ignore_index=True)
    frame = frame[frame["dir_bin"].notna()]
    return frame


def collapse(aggregate, strata):
    """把不参与本次分层的维度汇总掉，降低后续重采样的行数。"""

    keys = list(strata) + ["kind", "day"]
    grouped = aggregate.groupby(keys, observed=True)[SUM_COLS].sum()
    return grouped.reset_index()


def prepare_design(collapsed, strata):
    """冻结 (分层 x 处理状态) 分组，使每个 bootstrap 抽样只剩 bincount。

    奇位置为处理组、偶位置为对照组，因而一次 bincount 同时得到两组的总量。
    """

    if collapsed.empty:
        return None
    stratum = collapsed.groupby(list(strata), observed=True, sort=False).ngroup().to_numpy()
    is_treat = (collapsed["kind"] == "treat").to_numpy().astype(np.int64)
    day_code, days = pd.factorize(collapsed["day"])
    return {
        "group": stratum * 2 + is_treat,
        "n_groups": 2 * (int(stratum.max()) + 1),
        "day_code": day_code,
        "n_days": len(days),
        "sums": collapsed[VALUES].to_numpy(dtype=float),
        "counts": collapsed[COUNT_COLS].to_numpy(dtype=float),
    }


def did_from_design(design, factor=None, min_treat=5, min_base=20):
    """一次求出全部通道的 DiD：分层内均值差，再以处理组样本量加权汇总。"""

    empty = {value: (float("nan"), 0, 0) for value in VALUES}
    if design is None:
        return empty
    group, n_groups = design["group"], design["n_groups"]
    results = {}
    for position, value in enumerate(VALUES):
        sums = design["sums"][:, position]
        counts = design["counts"][:, position]
        if factor is not None:
            sums = sums * factor
            counts = counts * factor
        total = np.bincount(group, weights=sums, minlength=n_groups)
        size = np.bincount(group, weights=counts, minlength=n_groups)
        base_total, treat_total = total[0::2], total[1::2]
        base_size, treat_size = size[0::2], size[1::2]
        keep = (treat_size >= min_treat) & (base_size >= min_base)
        if not keep.any():
            results[value] = (float("nan"), 0, 0)
            continue
        difference = (
            treat_total[keep] / treat_size[keep] - base_total[keep] / base_size[keep]
        )
        weight = treat_size[keep]
        finite = np.isfinite(difference)
        if not finite.any():
            results[value] = (float("nan"), 0, 0)
            continue
        difference, weight = difference[finite], weight[finite]
        results[value] = (
            float((difference * weight).sum() / weight.sum()),
            int(round(weight.sum())),
            int(finite.sum()),
        )
    return results


def block_bootstrap(design, n_boot):
    """按天分块重采样：抽中的天以重复次数进入加权汇总。

    全部通道共用同一条抽样链，与 12 号脚本逐通道单独抽样等价，但代价只有一遍。
    """

    draws = {value: [] for value in VALUES}
    if design is None:
        return {value: np.asarray([]) for value in VALUES}
    n_days = design["n_days"]
    day_code = design["day_code"]
    for _ in range(n_boot):
        picked = RNG.integers(0, n_days, size=n_days)
        repeats = np.bincount(picked, minlength=n_days).astype(float)
        estimates = did_from_design(design, factor=repeats[day_code])
        for value, (estimate, _, _) in estimates.items():
            if np.isfinite(estimate):
                draws[value].append(estimate)
    return {value: np.asarray(sample) for value, sample in draws.items()}


def summarise(aggregate, strata, n_boot):
    """点估计 + 日块 bootstrap CI，返回逐通道的记录字典。"""

    design = prepare_design(collapse(aggregate, strata), strata)
    point = did_from_design(design)
    samples = block_bootstrap(design, n_boot)
    records = {}
    for value, unit, label in CHANNELS:
        estimate, n_treat, n_cells = point[value]
        sample = samples[value]
        if sample.size:
            low, median, high = np.quantile(sample, [0.025, 0.5, 0.975])
        else:
            low = median = high = float("nan")
        records[value] = {
            "channel": value,
            "label": label,
            "unit": unit,
            "did": estimate,
            "ci_low": float(low),
            "ci_median": float(median),
            "ci_high": float(high),
            "n_treated_frames": n_treat,
            "n_strata": n_cells,
            "bootstrap_draws": int(sample.size),
            "significant": bool(sample.size and (low > 0 or high < 0)),
            "p_greater_than_zero": (
                float((sample > 0).mean()) if sample.size else float("nan")
            ),
        }
    return records


# ------------------------------------------------------------------ 功率曲线
def empirical_power_curve(wide, free):
    winds, powers = [], []
    for turbine in wide["wind"].columns:
        mask = free[turbine].to_numpy()
        winds.append(wide["wind"][turbine].to_numpy()[mask])
        powers.append(wide["power_pu"][turbine].to_numpy()[mask])
    wind = np.concatenate(winds)
    power = np.concatenate(powers)
    finite = np.isfinite(wind) & np.isfinite(power)
    wind, power = wind[finite], power[finite]
    edges = np.arange(3, 16.5, 0.5)
    centers, curve = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (wind >= low) & (wind < high)
        if mask.sum() >= 50:
            centers.append((low + high) / 2.0)
            curve.append(float(np.median(power[mask])))
    centers = np.asarray(centers)
    curve = np.asarray(curve)
    return centers, curve, np.gradient(curve, centers), len(wind)


# ------------------------------------------------------------------ 事件研究
def event_study(wide, capped, free, direction):
    """机组自身作参照，再减去同期全程未限功机组的同步变化。"""

    estimates = []
    wind = wide["wind"]
    for i in wind.columns:
        first, second = (
            (free[i], capped[i]) if direction == "onset" else (capped[i], free[i])
        )
        before, after = first.to_numpy(), second.to_numpy()
        values = wind[i].to_numpy()
        for position in np.flatnonzero(before[:-1] & after[1:]):
            low, high = position - K_EVENT + 1, position + 1 + K_EVENT
            if low < 0 or high > len(before):
                continue
            if not (before[low : position + 1].all() and after[position + 1 : high].all()):
                continue
            shift_i = (
                values[position + 1 : high].mean() - values[low : position + 1].mean()
            )
            for j in wind.columns:
                if j == i or not free[j].to_numpy()[low:high].all():
                    continue
                other = wind[j].to_numpy()
                shift_j = (
                    other[position + 1 : high].mean() - other[low : position + 1].mean()
                )
                if np.isfinite(shift_i) and np.isfinite(shift_j):
                    estimates.append(shift_i - shift_j)
    return np.asarray(estimates)


def write_csv(path, rows):
    if not rows:
        raise SystemExit("refusing to write empty table: %s" % path)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-boot", type=int, default=N_BOOT, help="主规格的日块 bootstrap 次数"
    )
    parser.add_argument(
        "--n-boot-secondary",
        type=int,
        default=800,
        help="辅助规格（未调风向 / 仅外部指令）的抽样次数",
    )
    arguments = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wide = load_wide()
    generating, capped, free = build_masks(wide)
    turbines = list(wide["wind"].columns)

    print("=== 样本基础（Hill of Towie, 2300 kW, 10-min）===")
    base_rows = []
    for turbine in turbines:
        n_gen = int(generating[turbine].sum())
        n_cap = int(capped[turbine].sum())
        n_free = int(free[turbine].sum())
        base_rows.append(
            {
                "turbine": turbine,
                "generating_frames": n_gen,
                "capped_tracked_frames": n_cap,
                "capped_share_of_generating": n_cap / max(n_gen, 1),
                "free_frames": n_free,
            }
        )
        print(
            "%-5s 发电帧=%6d  其中限功(已跟踪)=%5d (%.2f%%)  未限功=%6d"
            % (turbine, n_gen, n_cap, 100 * n_cap / max(n_gen, 1), n_free)
        )
    write_csv(OUT_DIR / "hot_sample_base.csv", base_rows)

    n_cap_row = capped.sum(axis=1)
    n_free_row = free.sum(axis=1)
    usable = (n_cap_row >= 1) & (n_free_row >= 1)
    print(
        "\n可用于 DiD 的时刻（同时存在限功机与未限功机）: %d 帧 (%.2f%% of all)"
        % (int(usable.sum()), 100 * usable.mean())
    )
    print(
        "  其中限功机数分布:",
        dict(n_cap_row[n_cap_row >= 1].value_counts().sort_index().astype(int)),
    )
    synchronous = float((n_cap_row[n_cap_row >= 1] >= 0.5 * len(turbines)).mean())
    print(
        "  场级同步限功（>=50%% 机组同时限功）占限功时刻的 %.1f%%；ALTA2 为 55%%"
        % (100 * synchronous)
    )

    aggregate = pair_aggregate(wide, capped, free, turbines)
    treated_cells = aggregate[aggregate["kind"] == "treat"]
    print(
        "\n预聚合单元: %d 行（treat %d / base %d），机对数=%d，天数=%d"
        % (
            len(aggregate),
            len(treated_cells),
            len(aggregate) - len(treated_cells),
            aggregate["pair"].nunique(),
            aggregate["day"].nunique(),
        )
    )
    print(
        "配对样本帧: treat=%d, base=%d"
        % (
            int(treated_cells["n"].sum()),
            int(aggregate[aggregate["kind"] == "base"]["n"].sum()),
        )
    )
    external_share = float(
        treated_cells[treated_cells["ext_bin"] == "external"]["n"].sum()
        / max(treated_cells["n"].sum(), 1)
    )
    print(
        "处理帧中带外部降功率指令（wtc_PowerRed）的占比: %.1f%%"
        % (100 * external_share)
    )

    print("\n" + "=" * 78)
    print("=== 主结果 A: 机舱风速污染 Δv (m/s，正=限功时读数偏高) ===")
    print("=" * 78)
    # 三个规格：与 ALTA2 逐字对应的未调风向版、加风向扇区分层的主规格，
    # 以及只留外部降功率指令帧的稳健性版。风向分层不是事后挑选：12 号脚本
    # 的 docstring 已把“可选按风向扇区分层”写入设计，而偏航设计检验在本风场
    # 显著，正是该选项预设的触发条件。
    external_only = aggregate[
        (aggregate["kind"] == "base") | (aggregate["ext_bin"] == "external")
    ]
    specifications = (
        ("pair x v_bin", STRATA_PLAIN, "all_capped", aggregate, arguments.n_boot_secondary),
        ("pair x v_bin x dir_bin", STRATA_DIR, "all_capped", aggregate, arguments.n_boot),
        (
            "pair x v_bin x dir_bin",
            STRATA_DIR,
            "external_command_only",
            external_only,
            arguments.n_boot_secondary,
        ),
    )
    channel_rows = []
    primary = None
    for strata_label, strata, definition, frame, draws in specifications:
        records = summarise(frame, strata, draws)
        is_primary = definition == "all_capped" and strata == STRATA_DIR
        print(
            "\n[分层=%s | 限功定义=%s | B=%d]%s"
            % (strata_label, definition, draws, "  <== 主规格" if is_primary else "")
        )
        for value, unit, label in CHANNELS:
            record = dict(records[value])
            record["strata"] = strata_label
            record["curtailment_definition"] = definition
            record["is_primary"] = is_primary
            channel_rows.append(record)
            print(
                "  %-16s DiD = %+8.3f %-5s  95%%CI [%+.3f, %+.3f]  n=%6d L=%4d%s"
                % (
                    label,
                    record["did"],
                    unit,
                    record["ci_low"],
                    record["ci_high"],
                    record["n_treated_frames"],
                    record["n_strata"],
                    "  <-- 显著" if record["significant"] else "  (不显著)",
                )
            )
        if is_primary:
            primary = records
    write_csv(OUT_DIR / "hot_did_channels.csv", channel_rows)

    dv_row = primary["dv"]
    rel_row = primary["dv_rel"]
    print(
        "\n主规格机舱风速 Δv 的日块 bootstrap 95%% CI: [%+.3f, %+.3f]"
        "  中位数 %+.3f m/s (B=%d)"
        % (
            dv_row["ci_low"],
            dv_row["ci_high"],
            dv_row["ci_median"],
            dv_row["bootstrap_draws"],
        )
    )
    print("  p(Δv>0) = %.4f" % dv_row["p_greater_than_zero"])

    print("\n--- 按参照机风速分层：加性偏差 vs 乘性偏差（主规格）---")
    print("%-12s %8s %10s %10s" % ("v_ref区间", "n_treat", "Δv(m/s)", "Δv/v"))
    band_rows = []
    bands = [band for band in aggregate["v_bin"].dropna().unique()]
    for band in sorted(bands, key=lambda item: item.left):
        subset = aggregate[aggregate["v_bin"] == band]
        if int(subset[subset["kind"] == "treat"]["n"].sum()) < 30:
            continue
        strata = ("pair", "dir_bin")
        point = did_from_design(prepare_design(collapse(subset, strata), strata))
        additive, n_treat, _ = point["dv"]
        relative = point["dv_rel"][0]
        band_rows.append(
            {
                "wind_band_ms": str(band),
                "band_low_ms": float(band.left),
                "band_high_ms": float(band.right),
                "n_treated_frames": n_treat,
                "delta_v_ms": additive,
                "relative_delta": relative,
            }
        )
        print("%-12s %8d %+10.3f %+10.3f" % (str(band), n_treat, additive, relative))
    write_csv(OUT_DIR / "hot_did_by_wind_band.csv", band_rows)

    print("\n" + "=" * 78)
    print("=== 主结果 B: 污染幅度随限功深度的变化（机制检验）===")
    print("=" * 78)
    print(
        "%-12s %8s %10s %10s %10s"
        % ("限功档位pu", "n_treat", "Δv(m/s)", "Δpitch", "Δrpm")
    )
    depth_rows = []
    for level in DEPTH_LABELS:
        subset = aggregate[
            (aggregate["kind"] == "base") | (aggregate["depth_bin"] == level)
        ]
        point = did_from_design(prepare_design(collapse(subset, STRATA_DIR), STRATA_DIR))
        wind_shift, n_treat, _ = point["dv"]
        depth_rows.append(
            {
                "depth_bin_pu": level,
                "n_treated_frames": n_treat,
                "delta_v_ms": wind_shift,
                "delta_pitch_deg": point["dpitch"][0],
                "delta_rpm": point["drpm"][0],
            }
        )
        print(
            "%-12s %8d %+10.3f %+10.2f %+10.1f"
            % (level, n_treat, wind_shift, point["dpitch"][0], point["drpm"][0])
        )
    write_csv(OUT_DIR / "hot_did_by_depth.csv", depth_rows)

    print("\n" + "=" * 78)
    print("=== 主结果 C: 污染的功率含义（经未限功功率曲线放大）===")
    print("=" * 78)
    centers, curve, slope, n_free_samples = empirical_power_curve(wide, free)
    write_csv(
        OUT_DIR / "hot_power_curve.csv",
        [
            {
                "wind_ms": float(center),
                "power_pu": float(power),
                "dpower_dwind_pu_per_ms": float(gradient),
                "dpower_dwind_kw_per_ms": float(gradient) * RATED,
            }
            for center, power, gradient in zip(centers, curve, slope)
        ],
    )
    print("经验功率曲线（未限功样本 n=%d）局部斜率 dP/dv:" % n_free_samples)
    for center, power, gradient in zip(centers, curve, slope):
        if 4 <= center <= 13:
            print(
                "   v=%4.1f m/s  P=%.3f pu  dP/dv=%.4f pu/(m/s)"
                % (center, power, gradient)
            )

    print("\n若把被污染的机舱风速直接送入 PAP 模型，等效功率偏差 = dP/dv x Δv：")
    print(
        "   （加性口径 Δv=%+.3f m/s；乘性口径 Δv=%+.1f%% x v）"
        % (dv_row["did"], 100 * rel_row["did"])
    )
    power_rows = []
    for target in (6.0, 7.0, 8.0, 9.0, 10.0, 11.0):
        index = int(np.argmin(np.abs(centers - target)))
        center, gradient = centers[index], slope[index]
        additive = gradient * dv_row["did"]
        multiplicative = gradient * rel_row["did"] * center
        power_rows.append(
            {
                "wind_ms": float(center),
                "dpower_dwind_pu_per_ms": float(gradient),
                "additive_bias_pu": float(additive),
                "additive_bias_kw": float(additive * RATED),
                "multiplicative_bias_pu": float(multiplicative),
                "multiplicative_bias_kw": float(multiplicative * RATED),
            }
        )
        print(
            "   v≈%4.1f m/s (dP/dv=%.4f): 加性 %+.4f pu (%+6.1f kW) | 乘性 %+.4f pu (%+6.1f kW)"
            % (
                center,
                gradient,
                additive,
                additive * RATED,
                multiplicative,
                multiplicative * RATED,
            )
        )
    write_csv(OUT_DIR / "hot_power_implication.csv", power_rows)

    print("\n" + "=" * 78)
    print("=== 主结果 D: 事件研究交叉验证（机组自身作参照）===")
    print("=" * 78)
    event_rows = []
    for direction, label, expectation in (
        ("onset", "限功投入", "应为正"),
        ("release", "限功解除", "应为负"),
    ):
        estimates = event_study(wide, capped, free, direction)
        if len(estimates) < 10:
            print("%-8s 样本不足 (n=%d)" % (label, len(estimates)))
            event_rows.append(
                {
                    "direction": direction,
                    "label": label,
                    "expected_sign": expectation,
                    "delta_v_ms": float("nan"),
                    "ci_low_ms": float("nan"),
                    "ci_high_ms": float("nan"),
                    "n_event_pairs": int(len(estimates)),
                }
            )
            continue
        draws = np.array(
            [
                RNG.choice(estimates, len(estimates), replace=True).mean()
                for _ in range(2000)
            ]
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        event_rows.append(
            {
                "direction": direction,
                "label": label,
                "expected_sign": expectation,
                "delta_v_ms": float(estimates.mean()),
                "ci_low_ms": float(low),
                "ci_high_ms": float(high),
                "n_event_pairs": int(len(estimates)),
            }
        )
        print(
            "%-8s Δv = %+.3f m/s  95%%CI [%+.3f, %+.3f]  (事件对 n=%d, %s)"
            % (label, estimates.mean(), low, high, len(estimates), expectation)
        )
    write_csv(OUT_DIR / "hot_event_study.csv", event_rows)

    onset = next(row for row in event_rows if row["direction"] == "onset")
    placebo = primary["damb"]
    design = primary["dyaw"]
    plain = next(
        row
        for row in channel_rows
        if row["channel"] == "dyaw" and row["strata"] == "pair x v_bin"
    )
    # “复现”的定义预先固定：CI 下界 > 0（与 ALTA2 同向且显著）。
    replicates = bool(dv_row["ci_low"] > 0)
    headline = {
        "farm": "Hill of Towie",
        "turbine_type": "SWT-2.3-VS-82 (2300 kW)",
        "rated_kw": RATED,
        "basis_minutes": 10,
        "months": MONTHS,
        "n_turbines": len(turbines),
        "primary_strata": "pair x v_bin x dir_bin",
        "primary_curtailment_definition": "all_capped",
        "delta_v_ms": dv_row["did"],
        "delta_v_ci_low_ms": dv_row["ci_low"],
        "delta_v_ci_high_ms": dv_row["ci_high"],
        "relative_delta": rel_row["did"],
        "relative_ci_low": rel_row["ci_low"],
        "relative_ci_high": rel_row["ci_high"],
        "n_treated_frames": dv_row["n_treated_frames"],
        "external_command_share_of_treated": external_share,
        "placebo_channel": "wtc_AmbieTmp_mean",
        "placebo_did": placebo["did"],
        "placebo_significant": placebo["significant"],
        "design_check_channel": "wtc_YawPos_mean",
        "design_check_did": design["did"],
        "design_check_significant": design["significant"],
        "design_check_did_before_direction_strata": plain["did"],
        "design_check_significant_before_direction_strata": plain["significant"],
        "event_study_delta_v_ms": onset["delta_v_ms"],
        "event_study_ci_low_ms": onset["ci_low_ms"],
        "event_study_ci_high_ms": onset["ci_high_ms"],
        "synchronous_curtailment_share": synchronous,
        "replicates_alta2_direction_and_significance": replicates,
        "power_curve": {
            "wind_ms": [float(value) for value in centers],
            "power_pu": [float(value) for value in curve],
            "dpower_dwind_pu_per_ms": [float(value) for value in slope],
            "n_free_samples": int(n_free_samples),
        },
        "provenance": "scripts/13_hot_contamination_did.py",
    }
    (OUT_DIR / "hot_did_headline.json").write_text(
        json.dumps(headline, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("=== 两风场并列（外部有效性）===")
    print("=" * 78)
    two_farm = [
        ALTA2_REFERENCE,
        {
            "farm": "Hill of Towie",
            "turbine_type": "SWT-2.3-VS-82 (2300 kW)",
            "n_reference_turbines": len(turbines),
            "basis_minutes": 10,
            "delta_v_ms": dv_row["did"],
            "delta_v_ci_low_ms": dv_row["ci_low"],
            "delta_v_ci_high_ms": dv_row["ci_high"],
            "relative_delta": rel_row["did"],
            "relative_ci_low": rel_row["ci_low"],
            "relative_ci_high": rel_row["ci_high"],
            "placebo_significant": placebo["significant"],
            "event_study_delta_v_ms": onset["delta_v_ms"],
            "provenance": "scripts/13_hot_contamination_did.py",
        },
    ]
    write_csv(OUT_DIR / "two_farm_contamination.csv", two_farm)
    print(
        "%-24s %6s %10s %22s %10s %10s"
        % ("风场", "参照机", "Δv(m/s)", "相对偏差 95%CI", "安慰剂显著", "事件研究Δv")
    )
    for row in two_farm:
        print(
            "%-24s %6d %+10.3f  %+6.1f%% [%+.1f%%, %+.1f%%] %10s %+10.3f"
            % (
                row["farm"],
                row["n_reference_turbines"],
                row["delta_v_ms"],
                100 * row["relative_delta"],
                100 * row["relative_ci_low"],
                100 * row["relative_ci_high"],
                "是" if row["placebo_significant"] else "否",
                row["event_study_delta_v_ms"],
            )
        )
    print(
        "\n安慰剂（环境温度）%s；设计检验（偏航）主规格%s，未调风向时%s。"
        % (
            "显著（识别失败）" if placebo["significant"] else "不显著",
            "显著（仍有残留不可比）" if design["significant"] else "不显著",
            "显著" if plain["significant"] else "不显著",
        )
    )
    if replicates:
        print(
            "HOT 与 ALTA2 同号同向且显著：两风场、两机型构成外部有效性。"
        )
    else:
        print(
            "HOT 未复现 ALTA2 的正向污染（CI 下界不大于 0）。不得将两风场概括为\n"
            "  “经两风场实测证明的系统性污染”；可辩护的表述是污染幅度与符号依赖于\n"
            "  限功策略与风速区间（见 hot_did_by_depth.csv 与 hot_did_by_wind_band.csv），\n"
            "  因而真值通道偏差不可先验地假定为零。"
        )
    print("\n已写入 %s" % OUT_DIR)


if __name__ == "__main__":
    main()
