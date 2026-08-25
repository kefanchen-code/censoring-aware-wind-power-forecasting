# -*- coding: utf-8 -*-
"""Release-event validation layer on the real Altahullion T11 record (C1).

Implements the auditable replacement protocol in
``protocols/release_event_protocol_v1_3.md``.  The missing legacy v1.2 source
cannot be authenticated from its hash alone, so v1.3 is explicitly marked as
reconstructed from the executable v1.2 code, the unchanged 122-event list,
and the archived v1.2 result summary.  It does not claim to reproduce the
missing document byte-for-byte.

Track semantics: there is no latent PAP truth on real data. Models are fitted
on observable labels only (U rows as points, R rows as right-censored at
PowerRef) and scored against the measured post-release power at +1/+5/+15 min
of the frozen 122 outcome-blind release events that fall in the test period.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch

from censored_wind_power.benchmark.core import BenchmarkContext
from censored_wind_power.benchmark.data import FitData, ObservableSplit, PredictionData
from censored_wind_power.benchmark.models import (
    B1Adapter,
    B4Adapter,
    B6Adapter,
    ClimatologyAdapter,
    PersistenceAdapter,
)
from censored_wind_power.benchmark.censored_qr import CLQRAdapter
from censored_wind_power.benchmark.realdata import (
    HORIZON_MINUTES,
    RELEASE_EVAL_HORIZONS_MIN,
    RATED_POWER_KW,
    WINDOW_MINUTES,
    assign_segments,
    build_real_windows,
    event_windows,
    load_real_frame,
    load_release_events,
    observed_after_release,
    usable_blocks,
)
from censored_wind_power.benchmark.scoring import (
    cluster_inference,
    exact_sign_flip_test,
)
from censored_wind_power.benchmark.protocol import set_determinism, write_json

PROTOCOL_PATH = PROJECT_DIR / "protocols" / "release_event_protocol_v1_3.md"
PROTOCOL_VERSION = "v1.3-reconstructed"
PROTOCOL_ID = "release-event-reconstructed-v1.3"
LEGACY_PROTOCOL_SHA256 = "8aa3bca2cce62a559cb55d803d18beb41ea26535473976e654c3cce40772d1bf"
FROZEN_EVENT_SHA256 = "e351c076d155dea95d94047b2c5a0bc365c7a680b9a859565695df36387d63fe"
FROZEN_EVENT_CANONICAL_SHA256 = (
    "a7f4c8e896f05453b323071659c2079ba1600dd6ca7083c37415ecc7170723de"
)
FROZEN_EVENT_COUNT = 122
EVENT_CSV = PROJECT_DIR / "results" / "altahullion_audit" / "qualified_release_events.csv"
EVENT_REFERENCE_CSV = (
    PROJECT_DIR / "reference_results" / "altahullion_audit" / "qualified_release_events.csv"
)
OUT_DIR = PROJECT_DIR / "results" / "release_event_validation_v1_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUANTILE_LEVELS = np.asarray([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
SEEDS = [42, 123, 256]
MODELS = {
    "B0_persistence": PersistenceAdapter,
    "B0_climatology": ClimatologyAdapter,
    "B1": B1Adapter,
    "B4": B4Adapter,
    "B6": B6Adapter,
    "CLQR": CLQRAdapter,
}
PRIMARY_FAMILIES = [
    ("censored_likelihood", "B4", "B6"),
    ("censored_quantile", "B4", "CLQR"),
    ("censored_likelihood_vs_free_only", "B1", "B6"),
]
REPORTING_REFERENCES = ("B0_persistence", "B0_climatology")
EXACT_CLUSTER_LIMIT = 16  # protocol: exact sign-flip below 17 clusters
HOLM_ALPHA = 0.05


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    """Hash text with LF newlines so CSV identity is operating-system neutral."""

    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def freeze_or_verify_protocol(
    protocol_path: Path = PROTOCOL_PATH,
    event_csv: Path = EVENT_CSV,
    output_dir: Path = OUT_DIR,
) -> dict:
    """Hash and validate the protocol and unchanged event list before scoring."""

    if not protocol_path.exists():
        raise FileNotFoundError("missing protocol document: %s" % protocol_path)
    if not event_csv.exists():
        raise FileNotFoundError("missing frozen event list: %s" % event_csv)
    event_count = int(len(pd.read_csv(event_csv)))
    event_sha256 = sha256_of(event_csv)
    event_canonical_sha256 = canonical_text_sha256(event_csv)
    if event_count != FROZEN_EVENT_COUNT:
        raise RuntimeError(
            "release-event count changed: expected %d, found %d"
            % (FROZEN_EVENT_COUNT, event_count)
        )
    if event_canonical_sha256 != FROZEN_EVENT_CANONICAL_SHA256:
        raise RuntimeError(
            "release-event list changed: expected canonical SHA-256 %s, found %s"
            % (FROZEN_EVENT_CANONICAL_SHA256, event_canonical_sha256)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "protocol_freeze.json"
    payload = {
        "protocol_document": "protocols/release_event_protocol_v1_3.md",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_of(protocol_path),
        "legacy_v1_2_protocol_sha256_unrecovered": LEGACY_PROTOCOL_SHA256,
        "reconstruction_status": (
            "replacement reconstructed from executable code and frozen artifacts; "
            "not a byte-for-byte recovery of v1.2"
        ),
        "event_list": "results/altahullion_audit/qualified_release_events.csv",
        "event_list_sha256": FROZEN_EVENT_SHA256,
        "event_list_sha256_scope": "legacy v1.2 CRLF bytes",
        "event_list_canonical_sha256": event_canonical_sha256,
        "event_list_file_sha256_at_freeze": event_sha256,
        "event_count_frozen": event_count,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        for key in (
            "protocol_version",
            "protocol_sha256",
            "event_list_sha256",
            "event_list_canonical_sha256",
            "event_count_frozen",
            "legacy_v1_2_protocol_sha256_unrecovered",
        ):
            if frozen.get(key) != payload[key]:
                raise RuntimeError(
                    "frozen release-event protocol changed (%s); bump the"
                    " protocol version instead of editing the frozen run" % key
                )
        return frozen
    write_json(freeze_path, payload)
    return payload


def real_context(output_dir: Path, device: str) -> BenchmarkContext:
    config = {
        "model": {
            "hidden_channels": 32,
            "layers": 4,
            "kernel_size": 3,
            "dropout": 0.10,
        },
        "training": {
            "min_epochs": 8,
            "max_epochs": 40,
            "patience": 6,
            "min_delta": 0.0001,
            "batch_size": 512,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "fail_on_nonconvergence": True,
        },
        "forecast": {
            "window_minutes": WINDOW_MINUTES,
            "horizon_minutes": HORIZON_MINUTES,
            "support_min_pu": 0.0,
            "support_max_pu": 1.06,
            "n_bins": 106,
        },
        "evaluation": {"quantiles": QUANTILE_LEVELS.tolist()},
    }
    edges = np.linspace(0.0, 1.06, 107)
    return BenchmarkContext(
        scenario="release_event_real",
        protocol_id=PROTOCOL_ID,
        output_dir=output_dir,
        device=device,
        config=config,
        quantile_levels=QUANTILE_LEVELS,
        bin_edges=edges,
        bin_centers=0.5 * (edges[:-1] + edges[1:]),
    )


def pinball_scores(quantiles: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Mean pinball loss over the 9-level grid, one value per window."""

    levels = torch.as_tensor(QUANTILE_LEVELS, dtype=torch.float32)
    prediction_tensor = torch.as_tensor(np.asarray(quantiles), dtype=torch.float32)
    observed_tensor = torch.as_tensor(np.asarray(observed), dtype=torch.float32)
    error = observed_tensor.unsqueeze(1) - prediction_tensor
    per_level = torch.maximum(
        levels.unsqueeze(0) * error, (levels.unsqueeze(0) - 1.0) * error
    )
    return per_level.mean(dim=1).numpy()


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (len(p_values) - rank)))
        adjusted[index] = running
    # Enforce monotonicity along the sorted order.
    cumulative = 0.0
    for index in order:
        cumulative = max(cumulative, adjusted[index])
        adjusted[index] = cumulative
    return adjusted.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="release-event validation layer")
    parser.add_argument("--smoke", action="store_true", help="one seed, two epochs")
    parser.add_argument(
        "--verify-protocol-only",
        action="store_true",
        help="verify the protocol and 122-event artifact without fitting models",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    event_csv = EVENT_CSV
    if arguments.verify_protocol_only and not event_csv.exists():
        event_csv = EVENT_REFERENCE_CSV
    freeze = freeze_or_verify_protocol(event_csv=event_csv)
    print("protocol frozen:", freeze["protocol_sha256"][:16])
    print("event list verified:", freeze["event_count_frozen"], "rows")
    if arguments.verify_protocol_only:
        return 0

    seeds = SEEDS[:1] if arguments.smoke else SEEDS
    frame = load_real_frame()
    blocks = usable_blocks(frame)
    assignment = assign_segments(blocks, 0.60, 0.15)
    windows, standardization = build_real_windows(blocks, assignment, HORIZON_MINUTES)
    split_sizes = {name: len(split) for name, split in windows.items()}
    print("window splits:", split_sizes)

    context = real_context(OUT_DIR / "fits", arguments.device)
    if arguments.smoke:
        context.config["training"]["min_epochs"] = 1
        context.config["training"]["max_epochs"] = 2
        context.config["training"]["fail_on_nonconvergence"] = False

    fit_data = FitData(
        train=ObservableSplit(
            features=windows["train"].features,
            observed=windows["train"].observed,
            cap=windows["train"].cap,
            censored=windows["train"].censored,
        ),
        validation=ObservableSplit(
            features=windows["validation"].features,
            observed=windows["validation"].observed,
            cap=windows["validation"].cap,
            censored=windows["validation"].censored,
        ),
        train_observed_rows=windows["train"].observed,
        feature_names=(
            "observed_power_pu",
            "wind_speed_train_standardized",
            "cap_pu",
            "censoring_indicator",
        ),
    )

    # --- frozen event list, test-period filter, feature stacks ---------------
    events = load_release_events()
    test_start = min(windows["test"].target_time)
    in_test = events["release_time"] >= test_start
    features, persistence, valid = event_windows(
        frame,
        events["release_time"],
        wind_standardization=standardization,
    )
    usable = in_test.to_numpy() & valid
    usable_events = events.loc[usable].reset_index(drop=True)
    usable_features = features[usable]
    usable_persistence = persistence[usable]
    print(
        "events: total=%d test_period=%d usable=%d"
        % (len(events), int(in_test.sum()), int(usable.sum()))
    )
    if usable.sum() == 0:
        raise RuntimeError("no usable release event falls in the test period")

    audit_spec = importlib.util.spec_from_file_location(
        "altahullion_audit", PROJECT_DIR / "scripts" / "10_altahullion_audit.py"
    )
    audit_module = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit_module)
    raw_frame = audit_module.load_frame()

    observations = {}
    for horizon in RELEASE_EVAL_HORIZONS_MIN:
        observations[horizon] = np.asarray(
            [
                observed_after_release(raw_frame, release_time, horizon)
                for release_time in usable_events["release_time"]
            ],
            dtype=np.float32,
        )

    prediction_data = PredictionData(
        features=usable_features,
        persistence=usable_persistence,
        feature_names=fit_data.feature_names,
    )

    # --- fit and score --------------------------------------------------------
    score_rows = []
    fitted_models = {}
    for model_name, adapter_class in MODELS.items():
        adapter = adapter_class()
        for seed in seeds:
            set_determinism(seed)
            fitted = adapter.fit(fit_data, context, seed if adapter.stochastic else None)
            forecast = adapter.predict(fitted, prediction_data, context)
            quantiles = forecast.quantiles
            for horizon in RELEASE_EVAL_HORIZONS_MIN:
                observed = observations[horizon]
                finite = np.isfinite(observed)
                scores = np.full(len(observed), np.nan, dtype=np.float32)
                scores[finite] = pinball_scores(quantiles[finite], observed[finite])
                exceedance = np.sum(
                    quantiles < observed[:, None], axis=1
                ).astype(np.float32)
                for event_index in range(len(usable_events)):
                    score_rows.append(
                        {
                            "event_id": int(usable_events.index[event_index]),
                            "release_time": usable_events["release_time"].iloc[event_index],
                            "date": usable_events["release_time"]
                            .iloc[event_index]
                            .date()
                            .isoformat(),
                            "model": model_name,
                            "seed": int(seed),
                            "horizon_min": int(horizon),
                            "score": float(scores[event_index]),
                            "observed_pu": float(observed[event_index]),
                            "exceedance_position": float(exceedance[event_index]),
                        }
                    )
            fitted_models[(model_name, seed)] = fitted.metadata
            print("scored %s seed=%d" % (model_name, seed))

    scores_frame = pd.DataFrame(score_rows)
    scores_frame.to_parquet(OUT_DIR / "event_scores.parquet", index=False)

    # --- paired inference per family and horizon ------------------------------
    seed_mean = (
        scores_frame.groupby(["event_id", "model", "horizon_min"])
        .agg(score=("score", "mean"), date=("date", "first"))
        .reset_index()
    )
    comparisons = []
    for horizon in RELEASE_EVAL_HORIZONS_MIN:
        horizon_frame = seed_mean.loc[seed_mean["horizon_min"] == horizon]
        pivoted = horizon_frame.pivot(index="event_id", columns="model", values="score")
        dates = horizon_frame.drop_duplicates("event_id").set_index("event_id")["date"]
        for family, reference, challenger in PRIMARY_FAMILIES:
            aligned = pivoted[[reference, challenger]].dropna()
            difference = (
                aligned[reference].to_numpy() - aligned[challenger].to_numpy()
            )
            event_dates = dates.loc[aligned.index].to_numpy()
            unique_dates = np.unique(event_dates)
            cluster_labels = np.searchsorted(unique_dates, event_dates)
            if len(unique_dates) <= EXACT_CLUSTER_LIMIT:
                result = exact_sign_flip_test(difference, cluster_labels)
                inference = {
                    "method": "exact_sign_flip",
                    "n_clusters": result["n_clusters"],
                    "p_two_sided": result["p_two_sided"],
                    "observed_difference": result["observed_difference"],
                }
            else:
                result = cluster_inference(
                    aligned[reference].to_numpy(),
                    aligned[challenger].to_numpy(),
                    cluster_labels,
                    seed=20260726,
                    bootstrap_draws=5000,
                    sign_flip_draws=100000,
                    exhaustive_max_clusters=0,
                )
                inference = {
                    "method": "cluster_bootstrap_sign_flip",
                    "n_clusters": result["n_clusters"],
                    "p_two_sided": result["sign_flip_p_two_sided"],
                    "observed_difference": result[
                        "difference_reference_minus_challenger"
                    ],
                    "ci_95": result["cluster_bootstrap_ci_95"],
                }
            comparisons.append(
                {
                    "family": family,
                    "reference": reference,
                    "challenger": challenger,
                    "horizon_min": int(horizon),
                    "n_events": int(len(aligned)),
                    **inference,
                }
            )

    for family in {family for family, _, _ in PRIMARY_FAMILIES}:
        family_rows = [row for row in comparisons if row["family"] == family]
        adjusted = holm_adjust([row["p_two_sided"] for row in family_rows])
        for row, adjusted_p in zip(family_rows, adjusted):
            row["p_holm_within_family"] = float(adjusted_p)

    # Reporting families against non-learned references (no Holm claim).
    for horizon in RELEASE_EVAL_HORIZONS_MIN:
        horizon_frame = seed_mean.loc[seed_mean["horizon_min"] == horizon]
        pivoted = horizon_frame.pivot(index="event_id", columns="model", values="score")
        for model_name in MODELS:
            for reference in REPORTING_REFERENCES:
                aligned = pivoted[[reference, model_name]].dropna()
                if model_name == reference or aligned.empty:
                    continue
                comparisons.append(
                    {
                        "family": "reporting_vs_%s" % reference,
                        "reference": reference,
                        "challenger": model_name,
                        "horizon_min": int(horizon),
                        "n_events": int(len(aligned)),
                        "mean_difference_reference_minus_challenger": float(
                            (aligned[reference] - aligned[model_name]).mean()
                        ),
                    }
                )

    # Sensitivity stratum (report-only; never enters screening).
    stratum = usable_events["post_exceeds_pre_by_10pct"].to_numpy()
    summary = {
        "protocol_freeze": freeze,
        "event_accounting": {
            "frozen_total": int(len(events)),
            "in_test_period": int(in_test.sum()),
            "usable_with_clean_history": int(usable.sum()),
            "in_test_but_invalid_history": int((in_test & ~valid).sum()),
            "invalid_history_all_events": int((~valid).sum()),
        },
        "split_window_counts": split_sizes,
        "seeds": seeds,
        "models": sorted(MODELS),
        "sensitivity_stratum_share_post_rise_gt_10pct": float(stratum.mean()),
        "comparisons": comparisons,
        "fit_metadata": {
            "%s_seed%d" % (name, seed): {
                key: metadata.get(key)
                for key in ("converged", "best_epoch", "epochs_run")
            }
            for (name, seed), metadata in fitted_models.items()
            if isinstance(metadata, dict)
        },
    }
    write_json(OUT_DIR / "release_event_summary.json", summary)

    render_figure(scores_frame, comparisons)
    print("output:", OUT_DIR)
    return 0


def render_figure(scores_frame: pd.DataFrame, comparisons: list) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    primary = [row for row in comparisons if "p_two_sided" in row]
    panel = axes[0]
    positions = []
    for offset, family in enumerate(sorted({row["family"] for row in primary})):
        family_rows = sorted(
            (row for row in primary if row["family"] == family),
            key=lambda row: row["horizon_min"],
        )
        for row in family_rows:
            y = offset * 4 + row["horizon_min"] / 5.0
            difference = row["observed_difference"]
            panel.errorbar(
                difference,
                y,
                fmt="o",
                ms=4,
                color="tab:blue" if row.get("p_holm_within_family", 1) < HOLM_ALPHA else "gray",
            )
            positions.append(y)
    panel.axvline(0.0, color="black", lw=0.6)
    panel.set_yticks([])
    panel.set_xlabel("score(reference) - score(challenger)")
    panel.set_title("paired score difference by horizon")

    panel = axes[1]
    seed_mean = (
        scores_frame.groupby(["model", "horizon_min"])["score"]
        .mean()
        .reset_index()
    )
    for horizon, marker in zip(RELEASE_EVAL_HORIZONS_MIN, "osd"):
        subset = seed_mean.loc[seed_mean["horizon_min"] == horizon]
        panel.scatter(
            range(len(subset)),
            subset["score"],
            marker=marker,
            s=18,
            label="+%d min" % horizon,
        )
    panel.set_xticks(range(len(MODELS)))
    panel.set_xticklabels(sorted(MODELS), rotation=45, ha="right")
    panel.set_ylabel("mean pinball score")
    panel.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(OUT_DIR / "Fig_release_events.png", dpi=300)
    figure.savefig(OUT_DIR / "Fig_release_events.svg")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
