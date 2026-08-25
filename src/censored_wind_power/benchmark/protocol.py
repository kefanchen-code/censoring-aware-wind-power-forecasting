"""Configuration validation, hashing, and run-manifest helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from .core import ConfigurationError


PROTOCOL_SECTIONS = (
    "schema_version",
    "data",
    "split",
    "forecast",
    "training",
    "model",
    "evaluation",
    "inference",
)


def load_config(path: Path) -> Dict[str, Any]:
    """Load and validate a benchmark JSON configuration."""

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail early on ambiguous or invalid benchmark settings."""

    missing = [key for key in PROTOCOL_SECTIONS if key not in config]
    for key in ("output_dir", "scenarios", "models", "seeds"):
        if key not in config:
            missing.append(key)
    if missing:
        raise ConfigurationError("missing config keys: %s" % sorted(set(missing)))
    if int(config["schema_version"]) != 2:
        raise ConfigurationError("only benchmark schema_version=2 is supported")
    if not config["scenarios"] or len(set(config["scenarios"])) != len(config["scenarios"]):
        raise ConfigurationError("scenarios must be non-empty and unique")
    if not config["models"] or len(set(config["models"])) != len(config["models"]):
        raise ConfigurationError("models must be non-empty and unique")
    if not isinstance(config.get("model_settings", {}), Mapping):
        raise ConfigurationError("model_settings must be a mapping")
    seeds = [int(seed) for seed in config["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ConfigurationError("seeds must be non-empty and unique")

    split = config["split"]
    train_fraction = float(split["train_fraction"])
    validation_fraction = float(split["validation_fraction"])
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ConfigurationError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ConfigurationError("train_fraction + validation_fraction must be < 1")
    if split.get("strategy") != "chronological_whole_segment":
        raise ConfigurationError("unsupported split strategy: %r" % split.get("strategy"))

    forecast = config["forecast"]
    if int(forecast["window_minutes"]) <= 0 or int(forecast["horizon_minutes"]) <= 0:
        raise ConfigurationError("window and horizon must be positive")
    if int(forecast["n_bins"]) < 2:
        raise ConfigurationError("n_bins must be at least two")
    if float(forecast["support_max_pu"]) <= float(forecast["support_min_pu"]):
        raise ConfigurationError("support_max_pu must exceed support_min_pu")

    data = config["data"]
    if float(data["rated_power_kw"]) <= 0:
        raise ConfigurationError("rated_power_kw must be positive")
    if int(data["sampling_interval_minutes"]) <= 0:
        raise ConfigurationError("sampling_interval_minutes must be positive")
    if not str(data["directory"]) or "{scenario}" not in str(data["filename_pattern"]):
        raise ConfigurationError(
            "data directory is required and filename_pattern must contain {scenario}"
        )

    training = config["training"]
    min_epochs = int(training["min_epochs"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    if not 1 <= min_epochs <= max_epochs:
        raise ConfigurationError("require 1 <= min_epochs <= max_epochs")
    if patience < 1:
        raise ConfigurationError("patience must be positive")
    if int(training["batch_size"]) < 1:
        raise ConfigurationError("batch_size must be positive")
    if float(training["learning_rate"]) <= 0:
        raise ConfigurationError("learning_rate must be positive")

    architecture = config["model"]
    if min(
        int(architecture["hidden_channels"]),
        int(architecture["layers"]),
        int(architecture["kernel_size"]),
    ) < 1:
        raise ConfigurationError("TCN dimensions must be positive")
    if not 0 <= float(architecture["dropout"]) < 1:
        raise ConfigurationError("dropout must lie in [0, 1)")

    quantiles = np.asarray(config["evaluation"]["quantiles"], dtype=float)
    if quantiles.ndim != 1 or len(quantiles) < 3:
        raise ConfigurationError("at least three quantiles are required")
    if np.any(quantiles <= 0) or np.any(quantiles >= 1):
        raise ConfigurationError("quantiles must lie strictly between zero and one")
    if np.any(np.diff(quantiles) <= 0) or not np.any(np.isclose(quantiles, 0.5)):
        raise ConfigurationError("quantiles must be strictly increasing and include 0.5")
    quantile_set = {round(float(value), 12) for value in quantiles}
    for alpha in config["evaluation"]["interval_alphas"]:
        lower = round(float(alpha) / 2.0, 12)
        upper = round(1.0 - float(alpha) / 2.0, 12)
        if lower not in quantile_set or upper not in quantile_set:
            raise ConfigurationError(
                "interval alpha %.6g lacks exact quantiles %.6g and %.6g"
                % (alpha, lower, upper)
            )

    inference = config["inference"]
    if int(inference["cluster_bootstrap_draws"]) < 1:
        raise ConfigurationError("cluster_bootstrap_draws must be positive")
    if int(inference["sign_flip_draws"]) < 1:
        raise ConfigurationError("sign_flip_draws must be positive")
    if int(inference["exhaustive_max_clusters"]) < 0:
        raise ConfigurationError("exhaustive_max_clusters must be non-negative")
    for comparison in config.get("comparisons", []):
        if comparison.get("metric") not in {"wis", "crps"}:
            raise ConfigurationError("comparison metric must be wis or crps")
        if not comparison.get("reference") or not comparison.get("challenger"):
            raise ConfigurationError("comparison requires reference and challenger")

    validate_contamination(config.get("contamination", {}))


def validate_contamination(contamination: Mapping[str, Any]) -> None:
    """Validate the optional input-contamination section.

    The section is optional so existing configurations stay valid; when present
    and enabled it must fully pin down the injected perturbation.
    """

    if not isinstance(contamination, Mapping):
        raise ConfigurationError("contamination must be a mapping")
    if not contamination:
        return
    if not contamination.get("enabled", False):
        return
    if contamination.get("channel") != "wind":
        raise ConfigurationError("only the wind channel can be contaminated")
    if contamination.get("trigger") != "is_censored":
        raise ConfigurationError("only the is_censored trigger is supported")
    mode = contamination.get("mode")
    if mode not in {"relative", "schedule"}:
        raise ConfigurationError("contamination mode must be relative or schedule")
    factor = float(contamination.get("intensity_factor", 1.0))
    if factor < 0:
        raise ConfigurationError("intensity_factor must be non-negative")
    if mode == "relative":
        if float(contamination["relative_delta"]) <= -1.0:
            raise ConfigurationError("relative_delta must exceed -1")
    else:
        bins = np.asarray(contamination["schedule_bins_ms"], dtype=float)
        deltas = np.asarray(contamination["schedule_relative_delta"], dtype=float)
        if bins.ndim != 1 or len(bins) < 2 or np.any(np.diff(bins) <= 0):
            raise ConfigurationError("schedule_bins_ms must be strictly increasing")
        if len(deltas) != len(bins) - 1:
            raise ConfigurationError(
                "schedule_relative_delta must have one entry per bin interval"
            )
        if np.any(deltas <= -1.0):
            raise ConfigurationError("schedule_relative_delta entries must exceed -1")
    if not str(contamination.get("measurement_provenance", "")):
        raise ConfigurationError(
            "contamination requires measurement_provenance so the injected"
            " magnitude stays traceable to its measurement"
        )


def protocol_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fields that define data and comparison compatibility.

    The model list and output directory are intentionally excluded so a new
    model can be added without invalidating existing compatible forecasts.

    ``contamination`` is folded in only when enabled: this keeps the payload of
    every pre-existing configuration byte-identical (their ``protocol_id`` stays
    valid, so nothing has to be re-run) while making two different injection
    magnitudes hash differently, which prevents a contaminated run from silently
    reusing clean forecasts.
    """

    payload = {key: config[key] for key in PROTOCOL_SECTIONS}
    contamination = config.get("contamination", {})
    if contamination.get("enabled", False):
        payload["contamination"] = dict(contamination)
    return payload


def stable_json_hash(value: Any) -> str:
    """Hash a JSON-serializable value using canonical key ordering."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_python_sources(paths: Iterable[Path]) -> Dict[str, str]:
    """Hash every Python source below the supplied paths."""

    output: Dict[str, str] = {}
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for file_path in files:
            output[str(file_path.resolve())] = sha256_file(file_path)
    return output


def import_extension_modules(module_names: Iterable[str]) -> None:
    """Import optional modules that register external model adapters."""

    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            raise ConfigurationError(
                "failed to import model extension %r: %s" % (module_name, error)
            ) from error


def set_determinism(seed: int) -> None:
    """Reset Python, NumPy, and PyTorch RNGs for an order-independent run."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def environment_info() -> Dict[str, Any]:
    """Return runtime versions required to reproduce a run."""

    import pandas as pd
    import scipy
    import sklearn
    import torch

    return {
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "determinism": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically write UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)
