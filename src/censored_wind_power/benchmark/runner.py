"""Artifact-oriented runner for fair and extensible PAP comparisons."""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from . import models as _builtin_models  # noqa: F401  (registration side effect)
from .core import BenchmarkContext, BenchmarkError, ConfigurationError, create_model
from .data import ScenarioData, make_segment_assignment, prepare_scenario, segment_signature
from .protocol import (
    environment_info,
    hash_python_sources,
    import_extension_modules,
    load_config,
    protocol_payload,
    sha256_file,
    stable_json_hash,
    validate_config,
    write_json,
)
from .scoring import cluster_inference, score_forecast, stratified_rows, underestimation_rate


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_device(config: Mapping[str, Any]) -> str:
    requested = str(config.get("execution", {}).get("device", "auto")).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("execution.device=cuda but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ConfigurationError("execution.device must be auto, cpu, or cuda")
    return requested


def _grid(config: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast = config["forecast"]
    edges = np.linspace(
        float(forecast["support_min_pu"]),
        float(forecast["support_max_pu"]),
        int(forecast["n_bins"]) + 1,
        dtype=np.float64,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    quantiles = np.asarray(config["evaluation"]["quantiles"], dtype=np.float64)
    return edges, centers, quantiles


def _data_path(project_dir: Path, config: Mapping[str, Any], scenario: str) -> Path:
    filename = str(config["data"]["filename_pattern"]).format(scenario=scenario)
    path = project_dir / str(config["data"]["directory"]) / filename
    if not path.is_file():
        raise ConfigurationError("scenario input does not exist: %s" % path)
    return path.resolve()


def _validate_names(values: Iterable[str], label: str) -> None:
    invalid = [value for value in values if not SAFE_NAME.fullmatch(value)]
    if invalid:
        raise ConfigurationError("unsafe %s identifiers: %s" % (label, invalid))


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _artifact_dir(
    output_dir: Path,
    scenario: str,
    model_name: str,
    seed: Optional[int],
) -> Path:
    run_name = "fixed" if seed is None else "seed_%d" % seed
    return output_dir / "artifacts" / scenario / model_name / run_name


def _write_protocol_manifest(
    project_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    protocol_id: str,
) -> Dict[int, str]:
    """Create or verify the immutable split and input-data contract."""

    manifest_path = output_dir / "protocol_manifest.json"
    data_files = {
        scenario: {
            "path": str(_data_path(project_dir, config, scenario)),
            "sha256": sha256_file(_data_path(project_dir, config, scenario)),
        }
        for scenario in config["scenarios"]
    }
    first_scenario = str(config["scenarios"][0])
    first_frame = pd.read_parquet(_data_path(project_dir, config, first_scenario))
    current_signature = segment_signature(first_frame)

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("protocol_id") != protocol_id:
            raise BenchmarkError(
                "output directory contains a different protocol; choose a new output_dir"
            )
        existing_files = existing.get("data_files", {})
        for scenario, metadata in data_files.items():
            if (
                scenario in existing_files
                and existing_files[scenario].get("sha256") != metadata["sha256"]
            ):
                raise BenchmarkError("input data changed for scenario %s" % scenario)
        if existing.get("segment_signature") != current_signature:
            raise BenchmarkError("segment structure changed since split manifest was frozen")
        assignment = {
            int(key): str(value)
            for key, value in existing["segment_assignment"].items()
        }
        if set(data_files) != set(existing_files):
            existing["data_files"] = data_files
            write_json(manifest_path, existing)
        return assignment

    assignment = make_segment_assignment(
        first_frame,
        float(config["split"]["train_fraction"]),
        float(config["split"]["validation_fraction"]),
    )
    write_json(
        manifest_path,
        {
            "schema_version": 2,
            "created_utc": _utc_now(),
            "protocol_id": protocol_id,
            "protocol": protocol_payload(config),
            "data_files": data_files,
            "segment_signature": current_signature,
            "segment_assignment": {str(key): value for key, value in assignment.items()},
            "split_assignment_hash": stable_json_hash(assignment),
            "note": "Whole segments are frozen before any model fitting.",
        },
    )
    return assignment


def _prepare_selected_data(
    project_dir: Path,
    config: Mapping[str, Any],
    scenarios: Sequence[str],
    assignment: Mapping[int, str],
) -> Dict[str, ScenarioData]:
    prepared = {}
    reference_frame = pd.read_parquet(
        _data_path(project_dir, config, str(config["scenarios"][0])),
        columns=["segment_id"],
    )
    expected_signature = segment_signature(reference_frame)
    for scenario in scenarios:
        frame = pd.read_parquet(_data_path(project_dir, config, scenario))
        signature = segment_signature(frame)
        if signature != expected_signature:
            raise BenchmarkError(
                "scenario %s does not share the frozen segment structure" % scenario
            )
        prepared[scenario] = prepare_scenario(scenario, frame, assignment, config)
    return prepared


def _save_artifact(
    directory: Path,
    scenario_data: ScenarioData,
    model_name: str,
    model_version: str,
    model_config_hash: str,
    model_implementation_sha256: str,
    seed: Optional[int],
    protocol_id: str,
    oracle: bool,
    reconstruction: bool,
    fitted: Any,
    forecast: Any,
    score: Any,
    saved_fit_files: Mapping[str, str],
    timing: Mapping[str, float],
    runtime_environment: Mapping[str, Any],
) -> Dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    training_payload = {
        "metadata": fitted.metadata,
        "history": fitted.history,
    }
    write_json(directory / "training.json", training_payload)
    training_path = directory / "training.json"
    arrays: Dict[str, np.ndarray] = {
        "quantiles": forecast.quantiles.astype(np.float32),
        "target": scenario_data.test.truth.astype(np.float32),
        "segment_id": scenario_data.test.segment_id.astype(np.int64),
        "target_row": scenario_data.test.target_row.astype(np.int64),
        "target_censored": scenario_data.test.censored.astype(np.float32),
        "target_cap_pu": scenario_data.test.cap.astype(np.float32),
        "target_wind_ms": scenario_data.test.target_wind_ms.astype(np.float32),
        "history_censor_frac": scenario_data.test.history_censor_frac.astype(
            np.float32
        ),
        "wis": score.wis.astype(np.float64),
        "crps": score.crps.astype(np.float64),
        "pit": score.pit.astype(np.float64),
    }
    if forecast.pmf is not None:
        arrays["pmf"] = forecast.pmf.astype(np.float32)
    forecast_path = directory / "forecast.npz"
    _atomic_npz(forecast_path, **arrays)
    artifact = {
        "schema_version": 2,
        "created_utc": _utc_now(),
        "protocol_id": protocol_id,
        "scenario": scenario_data.scenario,
        "model": model_name,
        "model_version": model_version,
        "model_config_hash": model_config_hash,
        "model_implementation_sha256": model_implementation_sha256,
        "seed": seed,
        "oracle": bool(oracle),
        "reconstruction_inputs": bool(reconstruction),
        "n_test_windows": len(scenario_data.test),
        "alignment_hash": _array_hash(
            scenario_data.test.target_row,
            scenario_data.test.segment_id,
            scenario_data.test.truth,
        ),
        "forecast_sha256": sha256_file(forecast_path),
        "training_sha256": sha256_file(training_path),
        "forecast_metadata": forecast.metadata,
        "score_summary": score.summary,
        "training_file": "training.json",
        "fit_files": dict(saved_fit_files),
        "fit_file_sha256": {
            name: sha256_file(directory / relative_path)
            for name, relative_path in saved_fit_files.items()
        },
        "timing_seconds": dict(timing),
        "environment": dict(runtime_environment),
    }
    write_json(directory / "artifact.json", artifact)
    return artifact


def _existing_artifact_is_compatible(
    directory: Path,
    protocol_id: str,
    model_name: str,
    model_version: str,
    model_config_hash: str,
    model_implementation_sha256: str,
    seed: Optional[int],
) -> bool:
    metadata_path = directory / "artifact.json"
    forecast_path = directory / "forecast.npz"
    if not metadata_path.exists() or not forecast_path.exists():
        return False
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = (
        metadata.get("protocol_id") == protocol_id
        and metadata.get("model") == model_name
        and metadata.get("model_version") == model_version
        and metadata.get("model_config_hash") == model_config_hash
        and metadata.get("model_implementation_sha256")
        == model_implementation_sha256
        and metadata.get("seed") == seed
        and metadata.get("forecast_sha256") == sha256_file(forecast_path)
    )
    return bool(expected)


def _read_artifact(directory: Path, protocol_id: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    with (directory / "artifact.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("protocol_id") != protocol_id:
        raise BenchmarkError("artifact protocol mismatch: %s" % directory)
    forecast_path = directory / "forecast.npz"
    if metadata.get("forecast_sha256") != sha256_file(forecast_path):
        raise BenchmarkError("artifact checksum mismatch: %s" % forecast_path)
    training_path = directory / str(metadata.get("training_file", "training.json"))
    if metadata.get("training_sha256") != sha256_file(training_path):
        raise BenchmarkError("artifact checksum mismatch: %s" % training_path)
    for name, expected_hash in metadata.get("fit_file_sha256", {}).items():
        relative_path = metadata.get("fit_files", {}).get(name)
        if not relative_path or expected_hash != sha256_file(directory / relative_path):
            raise BenchmarkError("fitted-parameter checksum mismatch: %s" % directory)
    with np.load(forecast_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays


def _available_runs(
    output_dir: Path,
    scenario: str,
    model_name: str,
    seeds: Sequence[int],
    stochastic: bool,
    protocol_id: str,
    model_version: str,
    model_config_hash: str,
    model_implementation_sha256: str,
) -> List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    expected_seeds: Sequence[Optional[int]] = seeds if stochastic else [None]
    output = []
    for seed in expected_seeds:
        directory = _artifact_dir(output_dir, scenario, model_name, seed)
        if (directory / "artifact.json").exists():
            metadata, arrays = _read_artifact(directory, protocol_id)
            compatible = (
                metadata.get("model_version") == model_version
                and metadata.get("model_config_hash") == model_config_hash
                and metadata.get("model_implementation_sha256")
                == model_implementation_sha256
            )
            if not compatible:
                raise BenchmarkError(
                    "stale model artifact must be rerun: %s" % directory
                )
            output.append((metadata, arrays))
    return output


def _equivalence_verdict(ci: Sequence[float], margin: float) -> bool:
    """Conservative TOST-style call: the 95% CI must sit strictly inside
    the pre-declared equivalence interval (-margin, +margin)."""

    low, high = float(ci[0]), float(ci[1])
    return -margin < low and high < margin


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


def _experiment_id(
    protocol_id: str, output_dir: Path, config: Mapping[str, Any]
) -> Optional[str]:
    """Composite fingerprint: protocol + frozen input data + comparison plan.

    The protocol_id alone covers only the frozen protocol core, so two runs on
    different input data can share it.  The fingerprint deliberately excludes
    creation timestamps and absolute local paths so that an identical experiment
    receives the same identifier on every machine.
    """

    manifest_path = output_dir / "protocol_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    data_hashes = {
        str(scenario): str(metadata["sha256"])
        for scenario, metadata in manifest.get("data_files", {}).items()
    }
    identity_payload = {
        "protocol_id": protocol_id,
        "data_sha256": data_hashes,
        "segment_signature": manifest.get("segment_signature"),
        "segment_assignment": manifest.get("segment_assignment"),
        "comparisons": config.get("comparisons", []),
    }
    return stable_json_hash(identity_payload)


def aggregate_existing(
    project_dir: Path,
    config: Mapping[str, Any],
    require_complete: bool = True,
) -> Dict[str, Any]:
    """Compare compatible saved forecasts without refitting any model."""

    validate_config(config)
    import_extension_modules(config.get("model_modules", []))
    protocol_id = stable_json_hash(protocol_payload(config))
    output_dir = (project_dir / str(config["output_dir"])).resolve()
    quantile_levels = np.asarray(config["evaluation"]["quantiles"], dtype=np.float64)
    interval_alphas = [float(value) for value in config["evaluation"]["interval_alphas"]]
    rows: List[Dict[str, Any]] = []
    stratified_records: List[Dict[str, Any]] = []
    run_cache: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]] = {}
    missing = []
    nonconverged = []
    for scenario in config["scenarios"]:
        for model_name in config["models"]:
            adapter = create_model(model_name)
            adapter_source = inspect.getsourcefile(adapter.__class__)
            if adapter_source is None:
                raise BenchmarkError("cannot locate source for model %s" % model_name)
            runs = _available_runs(
                output_dir,
                scenario,
                model_name,
                [int(seed) for seed in config["seeds"]],
                adapter.stochastic,
                protocol_id,
                adapter.version,
                stable_json_hash(config.get("model_settings", {}).get(model_name, {})),
                sha256_file(Path(adapter_source)),
            )
            expected = len(config["seeds"]) if adapter.stochastic else 1
            if len(runs) != expected:
                missing.append(
                    {"scenario": scenario, "model": model_name, "found": len(runs), "expected": expected}
                )
            run_cache[(scenario, model_name)] = runs
            for metadata, arrays in runs:
                training_path = _artifact_dir(
                    output_dir, scenario, model_name, metadata.get("seed")
                ) / "training.json"
                with training_path.open("r", encoding="utf-8") as handle:
                    training = json.load(handle)
                convergence = training.get("metadata", {}).get("converged")
                if convergence is False:
                    nonconverged.append(
                        {"scenario": scenario, "model": model_name, "seed": metadata.get("seed")}
                    )
                summary = metadata["score_summary"]
                censored_underestimation_q50: Optional[float] = None
                censored_underestimation_q90: Optional[float] = None
                censored_windows: Optional[int] = None
                if "target_censored" in arrays:
                    censored_mask = arrays["target_censored"] >= 0.5
                    censored_windows = int(censored_mask.sum())
                    censored_underestimation_q50 = underestimation_rate(
                        arrays["quantiles"],
                        arrays["target"],
                        quantile_levels,
                        0.5,
                        censored_mask,
                    )
                    censored_underestimation_q90 = underestimation_rate(
                        arrays["quantiles"],
                        arrays["target"],
                        quantile_levels,
                        0.9,
                        censored_mask,
                    )
                    for stratified_row in stratified_rows(
                        arrays["quantiles"],
                        arrays["target"],
                        arrays["wis"],
                        arrays["crps"],
                        arrays["pit"],
                        quantile_levels,
                        interval_alphas,
                        arrays["target_censored"],
                        arrays["history_censor_frac"],
                        arrays["target_cap_pu"],
                        arrays["target_wind_ms"],
                    ):
                        stratified_records.append(
                            {
                                "scenario": scenario,
                                "model": model_name,
                                "seed": metadata.get("seed"),
                                **stratified_row,
                            }
                        )
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "seed": metadata.get("seed"),
                        "oracle": bool(metadata.get("oracle", False)),
                        "wis_mean": summary["wis_mean"],
                        "crps_mean": summary["crps_mean"],
                        "crps_kind": summary["crps_kind"],
                        "median_mae": summary["median_mae"],
                        "median_bias": summary["median_bias"],
                        "censored_windows": censored_windows,
                        "censored_underestimation_q50": censored_underestimation_q50,
                        "censored_underestimation_q90": censored_underestimation_q90,
                        "n_test_windows": int(len(arrays["target"])),
                    }
                )

    status = {
        "protocol_id": protocol_id,
        "experiment_id": _experiment_id(protocol_id, output_dir, config),
        "complete": not missing,
        "valid_for_final_comparison": not missing and not nonconverged,
        "missing_artifacts": missing,
        "nonconverged_runs": nonconverged,
    }
    write_json(output_dir / "benchmark_status.json", status)
    if require_complete and missing:
        raise BenchmarkError("benchmark is incomplete; see benchmark_status.json")
    if bool(config["training"].get("fail_on_nonconvergence", True)) and nonconverged:
        raise BenchmarkError(
            "final aggregation refused because learned runs did not converge; "
            "see benchmark_status.json"
        )

    if rows:
        with (output_dir / "metrics_by_run.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        frame = pd.DataFrame(rows)
        numeric = ["wis_mean", "crps_mean", "median_mae", "median_bias"]
        summary_rows = []
        for (scenario, model_name), group in frame.groupby(["scenario", "model"], sort=True):
            row: Dict[str, Any] = {
                "scenario": scenario,
                "model": model_name,
                "n_runs": int(len(group)),
                "crps_kind": ",".join(sorted(set(group["crps_kind"]))),
            }
            for metric in numeric:
                row[metric + "_mean"] = float(group[metric].mean())
                row[metric + "_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig"
        )

    if stratified_records:
        stratified_frame = pd.DataFrame(stratified_records)
        metric_columns = [
            column
            for column in stratified_frame.columns
            if column not in ("scenario", "model", "seed", "dimension", "stratum")
        ]
        aggregated = (
            stratified_frame.groupby(
                ["scenario", "model", "dimension", "stratum"], sort=True
            )[metric_columns]
            .mean()
            .reset_index()
        )
        run_counts = (
            stratified_frame.groupby(
                ["scenario", "model", "dimension", "stratum"], sort=True
            )
            .size()
            .rename("n_runs")
            .reset_index()
        )
        aggregated = aggregated.merge(
            run_counts, on=["scenario", "model", "dimension", "stratum"]
        )
        aggregated.to_csv(
            output_dir / "metrics_stratified.csv", index=False, encoding="utf-8-sig"
        )

    comparisons = []
    inference_config = config["inference"]
    for comparison_index, specification in enumerate(config.get("comparisons", [])):
        metric = str(specification["metric"])
        reference = str(specification["reference"])
        challenger = str(specification["challenger"])
        family = str(specification.get("family", "%s_vs_%s_%s" % (challenger, reference, metric)))
        for scenario in config["scenarios"]:
            reference_runs = run_cache.get((scenario, reference), [])
            challenger_runs = run_cache.get((scenario, challenger), [])
            if not reference_runs or not challenger_runs:
                continue
            reference_metadata, reference_arrays = reference_runs[0]
            challenger_metadata, challenger_arrays = challenger_runs[0]
            if reference_metadata["alignment_hash"] != challenger_metadata["alignment_hash"]:
                raise BenchmarkError("unaligned forecasts in %s" % scenario)
            if metric == "crps":
                reference_kind = reference_metadata["score_summary"]["crps_kind"]
                challenger_kind = challenger_metadata["score_summary"]["crps_kind"]
                if reference_kind != challenger_kind:
                    raise BenchmarkError(
                        "cannot compare unlike CRPS approximations: %s versus %s"
                        % (reference_kind, challenger_kind)
                    )
            reference_loss = np.mean(
                np.stack([arrays[metric] for _, arrays in reference_runs]), axis=0
            )
            challenger_loss = np.mean(
                np.stack([arrays[metric] for _, arrays in challenger_runs]), axis=0
            )
            inference_seed = int(inference_config["seed"]) ^ int(
                stable_json_hash([scenario, comparison_index])[:8], 16
            )
            result = cluster_inference(
                reference_loss,
                challenger_loss,
                reference_arrays["segment_id"],
                inference_seed,
                int(inference_config["cluster_bootstrap_draws"]),
                int(inference_config["sign_flip_draws"]),
                int(inference_config["exhaustive_max_clusters"]),
            )
            comparisons.append(
                {
                    "family": family,
                    "scenario": scenario,
                    "metric": metric,
                    "reference": reference,
                    "challenger": challenger,
                    "seed_aggregation": "per-window arithmetic mean before paired inference",
                    **result,
                }
            )
            if "equivalence_margin" in specification:
                margin = float(specification["equivalence_margin"])
                comparisons[-1]["equivalence_margin"] = margin
                comparisons[-1]["equivalent_within_margin"] = _equivalence_verdict(
                    result["cluster_bootstrap_ci_95"], margin
                )
    families = sorted({item["family"] for item in comparisons})
    for family in families:
        indices = [index for index, item in enumerate(comparisons) if item["family"] == family]
        adjusted = _holm_adjust(
            [comparisons[index]["sign_flip_p_two_sided"] for index in indices]
        )
        for index, value in zip(indices, adjusted):
            comparisons[index]["holm_adjusted_p_within_family"] = value
            comparisons[index]["holm_family_size"] = len(indices)
    write_json(
        output_dir / "comparisons.json",
        {
            "protocol_id": protocol_id,
            "experiment_id": status["experiment_id"],
            "primary_cross_model_score": "WIS",
            "comparisons": comparisons,
        },
    )
    return status


def run_benchmark(
    project_dir: Path,
    config_path: Path,
    models_override: Optional[Sequence[str]] = None,
    scenarios_override: Optional[Sequence[str]] = None,
    seeds_override: Optional[Sequence[int]] = None,
    smoke: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Fit selected models, save standalone forecasts, then aggregate artifacts."""

    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["scenarios"] = list(config["scenarios"][:1])
        config["seeds"] = list(config["seeds"][:1])
        config["training"]["max_epochs"] = 1
        config["training"]["min_epochs"] = 1
        config["training"]["patience"] = 1
        config["training"]["fail_on_nonconvergence"] = False
        config["inference"]["cluster_bootstrap_draws"] = 100
        config["inference"]["sign_flip_draws"] = 1000
        config["inference"]["exhaustive_max_clusters"] = 12
        validate_config(config)
        smoke_protocol = stable_json_hash(protocol_payload(config))[:12]
        config["output_dir"] = "%s_smoke_%s" % (
            config["output_dir"],
            smoke_protocol,
        )
    import_extension_modules(config.get("model_modules", []))
    selected_models = list(models_override or config["models"])
    selected_scenarios = list(scenarios_override or config["scenarios"])
    selected_seeds = [int(seed) for seed in (seeds_override or config["seeds"])]
    unknown_scenarios = sorted(set(selected_scenarios) - set(config["scenarios"]))
    if unknown_scenarios:
        raise ConfigurationError("scenarios are not declared in config: %s" % unknown_scenarios)
    unknown_seeds = sorted(set(selected_seeds) - {int(seed) for seed in config["seeds"]})
    if unknown_seeds:
        raise ConfigurationError("seeds are not declared in config: %s" % unknown_seeds)
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ConfigurationError("selected seeds must be non-empty and unique")
    _validate_names(selected_models, "model")
    _validate_names(selected_scenarios, "scenario")
    adapters = {name: create_model(name) for name in selected_models}
    protocol_id = stable_json_hash(protocol_payload(config))
    output_dir = (project_dir / str(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment = _write_protocol_manifest(
        project_dir, output_dir, config, protocol_id
    )
    prepared = _prepare_selected_data(
        project_dir, config, selected_scenarios, assignment
    )
    summary_path = output_dir / "data_summary.json"
    existing_summary: Dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            existing_summary = json.load(handle)
    existing_summary.update(
        {
            scenario: {
                "data_signature": data.data_signature,
                "feature_names": list(data.feature_names),
                **data.split_summary,
            }
            for scenario, data in prepared.items()
        }
    )
    write_json(
        summary_path,
        existing_summary,
    )
    edges, centers, quantiles = _grid(config)
    device = _resolve_device(config)
    runtime_environment = environment_info()
    completed = []
    reused = []
    for scenario in selected_scenarios:
        scenario_data = prepared[scenario]
        for model_name in selected_models:
            adapter = adapters[model_name]
            adapter_source = inspect.getsourcefile(adapter.__class__)
            if adapter_source is None:
                raise BenchmarkError("cannot locate source for model %s" % model_name)
            implementation_hash = sha256_file(Path(adapter_source))
            model_config_hash = stable_json_hash(
                config.get("model_settings", {}).get(model_name, {})
            )
            seeds: Sequence[Optional[int]] = (
                selected_seeds
                if adapter.stochastic
                else [None]
            )
            for seed in seeds:
                directory = _artifact_dir(output_dir, scenario, model_name, seed)
                if not force and _existing_artifact_is_compatible(
                    directory,
                    protocol_id,
                    model_name,
                    adapter.version,
                    model_config_hash,
                    implementation_hash,
                    seed,
                ):
                    reused.append(str(directory))
                    continue
                context = BenchmarkContext(
                    scenario=scenario,
                    protocol_id=protocol_id,
                    output_dir=directory,
                    device=device,
                    config=config,
                    quantile_levels=quantiles,
                    bin_edges=edges,
                    bin_centers=centers,
                )
                fit_started = time.perf_counter()
                # Latent-truth labels are dispatched only to declared oracle
                # adapters; reconstruction observables only to declared
                # reconstruction adapters; every other model keeps the
                # observable-only view.
                uses_latent_truth = bool(
                    getattr(adapter, "requires_latent_truth", False)
                )
                uses_reconstruction = bool(
                    getattr(adapter, "requires_reconstruction_inputs", False)
                )
                if uses_latent_truth and uses_reconstruction:
                    raise ConfigurationError(
                        "model %s declares both latent-truth and reconstruction"
                        " views; these label sources are mutually exclusive"
                        % model_name
                    )
                if uses_latent_truth:
                    fit_input = scenario_data.oracle_fit_view()
                elif uses_reconstruction:
                    fit_input = scenario_data.reconstruction_fit_view()
                else:
                    fit_input = scenario_data.fit_view()
                fitted = adapter.fit(fit_input, context, seed)
                fit_seconds = time.perf_counter() - fit_started
                directory.mkdir(parents=True, exist_ok=True)
                fit_files = adapter.save_fit(fitted, directory)
                predict_started = time.perf_counter()
                forecast = adapter.predict(
                    fitted, scenario_data.prediction_view(), context
                )
                predict_seconds = time.perf_counter() - predict_started
                forecast.validate(len(scenario_data.test), context)
                score = score_forecast(forecast, scenario_data.test.truth, context)
                _save_artifact(
                    directory,
                    scenario_data,
                    model_name,
                    adapter.version,
                    model_config_hash,
                    implementation_hash,
                    seed,
                    protocol_id,
                    uses_latent_truth,
                    uses_reconstruction,
                    fitted,
                    forecast,
                    score,
                    fit_files,
                    {"fit": fit_seconds, "predict": predict_seconds},
                    runtime_environment,
                )
                completed.append(str(directory))

    code_hashes = hash_python_sources(
        [project_dir / "src" / "censored_wind_power" / "benchmark", config_path]
    )
    run_manifest = {
        "schema_version": 2,
        "created_utc": _utc_now(),
        "protocol_id": protocol_id,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "effective_config": config,
        "selected_models": selected_models,
        "selected_scenarios": selected_scenarios,
        "selected_seeds": selected_seeds,
        "device": device,
        "environment": runtime_environment,
        "source_sha256": code_hashes,
        "completed_artifacts": completed,
        "reused_artifacts": reused,
        "force": force,
        "smoke": smoke,
    }
    write_json(output_dir / "last_run_manifest.json", run_manifest)

    # Same-architecture PMF models must start identically; all learned models
    # must share the encoder initialization for a given scenario and seed.
    if {"B1", "B4", "B6"}.issubset(config["models"]):
        for scenario in config["scenarios"]:
            for seed in config["seeds"]:
                payloads = []
                for model_name in ("B1", "B4", "B6"):
                    path = _artifact_dir(output_dir, scenario, model_name, int(seed)) / "training.json"
                    if path.exists():
                        with path.open("r", encoding="utf-8") as handle:
                            payloads.append(json.load(handle))
                if len(payloads) == 3:
                    hashes = [payload["metadata"].get("initial_state_hash") for payload in payloads]
                    if len(set(hashes)) != 1:
                        raise BenchmarkError(
                            "paired B1/B4/B6 initialization mismatch for %s seed %s"
                            % (scenario, seed)
                        )
    learned = [
        name
        for name in ("B1", "B4", "B6", "BQR", "B2_recon")
        if name in config["models"]
    ]
    if len(learned) > 1:
        for scenario in config["scenarios"]:
            for seed in config["seeds"]:
                encoder_hashes = []
                for model_name in learned:
                    path = _artifact_dir(output_dir, scenario, model_name, int(seed)) / "training.json"
                    if path.exists():
                        with path.open("r", encoding="utf-8") as handle:
                            encoder_hashes.append(
                                json.load(handle)["metadata"].get("initial_encoder_hash")
                            )
                if len(encoder_hashes) == len(learned) and len(set(encoder_hashes)) != 1:
                    raise BenchmarkError(
                        "learned-model encoder initialization mismatch for %s seed %s"
                        % (scenario, seed)
                    )

    status = aggregate_existing(
        project_dir,
        config,
        require_complete=(
            models_override is None
            and scenarios_override is None
            and seeds_override is None
        ),
    )
    return {
        "output_dir": str(output_dir),
        "protocol_id": protocol_id,
        "completed": len(completed),
        "reused": len(reused),
        "status": status,
    }
