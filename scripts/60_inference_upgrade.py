"""Exact sign-flip inference and TOST equivalence upgrade (plan item A1).

Reads the frozen per-window forecast artifacts of an existing benchmark run
and recomputes paired comparisons with:

1. an EXACT cluster-level sign-flip test enumerating all 2**K sign patterns
   (K=20 test segments -> 1,048,576 patterns; the frozen run used Monte Carlo
   because exhaustive_max_clusters was 16);
2. an exact permutation TOST on a pre-declared margin grid, reported together
   with the conventional t-TOST on cluster means and the conservative
   CI-containment verdict.

The frozen artifacts and comparisons.json are never modified; all outputs go
to results/inference_upgrade/<benchmark_name>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from censored_wind_power.benchmark.core import create_model  # noqa: E402
from censored_wind_power.benchmark.protocol import (  # noqa: E402
    import_extension_modules,
    load_config,
    sha256_file,
)
from censored_wind_power.benchmark.scoring import (  # noqa: E402
    exact_sign_flip_test,
    exact_tost,
)


TOST_MARGINS = (0.0025, 0.005, 0.01)
PRIMARY_MARGIN = 0.005


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_dir(
    output_dir: Path, scenario: str, model_name: str, seed: Any
) -> Path:
    run_name = "fixed" if seed is None else "seed_%d" % int(seed)
    return output_dir / "artifacts" / scenario / model_name / run_name


def _load_paired_difference(
    output_dir: Path,
    scenario: str,
    reference: str,
    challenger: str,
    metric: str,
    seeds: Sequence[int],
    stochastic_by_model: Dict[str, bool],
    protocol_id: str,
) -> Dict[str, Any]:
    """Mean per-window paired loss difference across seeds, plus segment ids."""

    def seed_average(model_name: str) -> Dict[str, Any]:
        expected_seeds: Sequence[Any] = seeds if stochastic_by_model[model_name] else [None]
        losses: List[np.ndarray] = []
        segment_id = None
        alignment_hash = None
        for seed in expected_seeds:
            directory = _artifact_dir(output_dir, scenario, model_name, seed)
            with (directory / "artifact.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("protocol_id") != protocol_id:
                raise RuntimeError("artifact protocol mismatch: %s" % directory)
            with np.load(directory / "forecast.npz", allow_pickle=False) as archive:
                loss = np.asarray(archive[metric], dtype=np.float64)
                current_segment = np.asarray(archive["segment_id"])
            if segment_id is None:
                segment_id = current_segment
            elif not np.array_equal(segment_id, current_segment):
                raise RuntimeError("segment alignment mismatch: %s" % directory)
            if alignment_hash is None:
                alignment_hash = metadata.get("alignment_hash")
            elif metadata.get("alignment_hash") != alignment_hash:
                raise RuntimeError("alignment hash mismatch: %s" % directory)
            losses.append(loss)
        return {"loss": np.mean(np.stack(losses), axis=0), "segment_id": segment_id}

    reference_pack = seed_average(reference)
    challenger_pack = seed_average(challenger)
    if not np.array_equal(reference_pack["segment_id"], challenger_pack["segment_id"]):
        raise RuntimeError("reference/challenger segment misalignment in %s" % scenario)
    difference = reference_pack["loss"] - challenger_pack["loss"]
    return {
        "difference": difference,
        "segment_id": reference_pack["segment_id"],
        "reference_mean_loss": float(reference_pack["loss"].mean()),
        "challenger_mean_loss": float(challenger_pack["loss"].mean()),
    }


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def main(config_path: Path, margins: Sequence[float]) -> Dict[str, Any]:
    config = load_config(config_path)
    import_extension_modules(config.get("model_modules", []))
    from censored_wind_power.benchmark.protocol import stable_json_hash, protocol_payload

    protocol_id = stable_json_hash(protocol_payload(config))
    source_output_dir = (PROJECT_ROOT / str(config["output_dir"])).resolve()
    out_dir = PROJECT_ROOT / "results" / "inference_upgrade" / source_output_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    stochastic_by_model = {
        name: bool(create_model(name).stochastic) for name in config["models"]
    }
    seeds = [int(seed) for seed in config["seeds"]]

    records: List[Dict[str, Any]] = []
    for specification in config.get("comparisons", []):
        metric = str(specification["metric"])
        reference = str(specification["reference"])
        challenger = str(specification["challenger"])
        family = str(
            specification.get(
                "family", "%s_vs_%s_%s" % (challenger, reference, metric)
            )
        )
        for scenario in config["scenarios"]:
            pack = _load_paired_difference(
                source_output_dir,
                scenario,
                reference,
                challenger,
                metric,
                seeds,
                stochastic_by_model,
                protocol_id,
            )
            sign_flip = exact_sign_flip_test(pack["difference"], pack["segment_id"])
            tost_rows = {
                "%.4f" % margin: exact_tost(
                    pack["difference"], pack["segment_id"], margin
                )
                for margin in margins
            }
            records.append(
                {
                    "family": family,
                    "scenario": scenario,
                    "metric": metric,
                    "reference": reference,
                    "challenger": challenger,
                    "direction": "positive means challenger has lower loss",
                    "reference_mean_loss": pack["reference_mean_loss"],
                    "challenger_mean_loss": pack["challenger_mean_loss"],
                    "difference_reference_minus_challenger": sign_flip[
                        "observed_difference"
                    ],
                    "exact_sign_flip": sign_flip,
                    "tost": tost_rows,
                }
            )
            print(
                "done %-45s %-20s diff=%+.6f p_exact=%.3e"
                % (
                    family,
                    scenario,
                    sign_flip["observed_difference"],
                    sign_flip["p_two_sided"],
                ),
                flush=True,
            )

    # Holm adjustment within family over scenarios, mirroring the frozen runner.
    families = sorted({record["family"] for record in records})
    for family in families:
        family_records = [r for r in records if r["family"] == family]
        adjusted = _holm_adjust(
            [r["exact_sign_flip"]["p_two_sided"] for r in family_records]
        )
        for record, value in zip(family_records, adjusted):
            record["holm_adjusted_exact_p_within_family"] = value

    payload = {
        "created_utc": _utc_now(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_output_dir": str(source_output_dir),
        "protocol_id": protocol_id,
        "method": {
            "sign_flip": (
                "exact enumeration of all 2^K cluster sign patterns; K test"
                " segments are the clusters; per-window losses are averaged"
                " across seeds before pairing"
            ),
            "tost": (
                "exact permutation TOST (shifted cluster sums sign-flipped over"
                " all 2^K patterns) plus t-TOST on cluster means and"
                " CI-containment verdict"
            ),
            "tost_margins": [float(margin) for margin in margins],
            "primary_margin": PRIMARY_MARGIN,
        },
        "records": records,
    }
    with (out_dir / "exact_signflip_and_tost.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    # Flat summaries for the manuscript tables.
    flip_fields = [
        "family",
        "scenario",
        "metric",
        "reference",
        "challenger",
        "difference_reference_minus_challenger",
        "n_clusters",
        "total_patterns",
        "p_two_sided",
        "quantile_0025",
        "quantile_975",
        "holm_adjusted_exact_p_within_family",
    ]
    with (out_dir / "exact_signflip_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(flip_fields)
        for record in records:
            flip = record["exact_sign_flip"]
            writer.writerow(
                [
                    record["family"],
                    record["scenario"],
                    record["metric"],
                    record["reference"],
                    record["challenger"],
                    record["difference_reference_minus_challenger"],
                    flip["n_clusters"],
                    flip["total_patterns"],
                    flip["p_two_sided"],
                    flip["quantile_0025"],
                    flip["quantile_975"],
                    record["holm_adjusted_exact_p_within_family"],
                ]
            )

    tost_fields = [
        "family",
        "scenario",
        "reference",
        "challenger",
        "margin",
        "observed_difference",
        "p_lower_boundary_permutation",
        "p_upper_boundary_permutation",
        "p_max_permutation",
        "equivalent_permutation",
        "p_max_t",
        "equivalent_t",
        "cluster_mean_ci_95",
        "equivalent_ci_containment",
    ]
    with (out_dir / "tost_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(tost_fields)
        for record in records:
            for margin_key, tost in record["tost"].items():
                writer.writerow(
                    [
                        record["family"],
                        record["scenario"],
                        record["reference"],
                        record["challenger"],
                        margin_key,
                        tost["observed_difference"],
                        tost["p_lower_boundary_permutation"],
                        tost["p_upper_boundary_permutation"],
                        tost["p_max_permutation"],
                        tost["equivalent_permutation"],
                        tost["p_max_t"],
                        tost["equivalent_t"],
                        tost["cluster_mean_ci_95"],
                        tost["equivalent_ci_containment"],
                    ]
                )

    print("wrote %s" % out_dir)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pap_benchmark_semisyn.json",
        help="benchmark configuration whose frozen artifacts are re-analysed",
    )
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=list(TOST_MARGINS),
        help="TOST equivalence margins",
    )
    main(parser.parse_args().config, parser.parse_args().margins)
