"""Built-in reference models for the PAP benchmark."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .core import (
    BenchmarkContext,
    BenchmarkError,
    FitResult,
    Forecast,
    ModelAdapter,
    register_model,
)
from .data import FitData, ObservableSplit, PredictionData
from .protocol import set_determinism
from .scoring import pmf_to_quantiles


class _FeatureDataset(Dataset):
    """Torch view exposing only inputs and observable training labels."""

    def __init__(self, split: ObservableSplit) -> None:
        self.features = torch.from_numpy(split.features)
        self.observed = torch.from_numpy(split.observed)
        self.cap = torch.from_numpy(split.cap)
        self.censored = torch.from_numpy(split.censored)

    def __len__(self) -> int:
        return int(len(self.features))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.features[index],
            self.observed[index],
            self.cap[index],
            self.censored[index],
        )


class _PredictionDataset(Dataset):
    """Prediction view that makes accidental access to test truth impossible."""

    def __init__(
        self,
        data: PredictionData,
        feature_channels: Optional[Tuple[int, ...]] = None,
    ) -> None:
        features = data.features
        if feature_channels is not None:
            features = np.ascontiguousarray(
                features[:, list(feature_channels), :]
            )
        self.features = torch.from_numpy(features)

    def __len__(self) -> int:
        return int(len(self.features))

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]


class TemporalBlock(nn.Module):
    """Causal residual temporal-convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv1(inputs)[:, :, : inputs.shape[2]]
        output = self.dropout(F.relu(output))
        output = self.conv2(output)[:, :, : inputs.shape[2]]
        output = self.dropout(F.relu(output))
        return F.relu(output + self.residual(inputs))


class TCNEncoder(nn.Module):
    """Shared TCN encoder used by all learned reference models."""

    def __init__(
        self,
        n_features: int,
        hidden: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = n_features
        for layer in range(layers):
            blocks.append(
                TemporalBlock(
                    in_channels,
                    hidden,
                    kernel_size,
                    2 ** layer,
                    dropout,
                )
            )
            in_channels = hidden
        self.network = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)[:, :, -1]


class PMFTCN(nn.Module):
    """TCN with a finite-support probability-mass output."""

    def __init__(self, context: BenchmarkContext, n_features: int = 4) -> None:
        super().__init__()
        architecture = context.config["model"]
        hidden = int(architecture["hidden_channels"])
        dropout = float(architecture["dropout"])
        self.encoder = TCNEncoder(
            n_features=n_features,
            hidden=hidden,
            layers=int(architecture["layers"]),
            kernel_size=int(architecture["kernel_size"]),
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(context.bin_centers)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.head(self.encoder(inputs)), dim=-1)


class QuantileTCN(nn.Module):
    """TCN with direct quantile outputs."""

    def __init__(self, context: BenchmarkContext, n_features: int = 4) -> None:
        super().__init__()
        architecture = context.config["model"]
        hidden = int(architecture["hidden_channels"])
        dropout = float(architecture["dropout"])
        self.encoder = TCNEncoder(
            n_features=n_features,
            hidden=hidden,
            layers=int(architecture["layers"]),
            kernel_size=int(architecture["kernel_size"]),
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(context.quantile_levels)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


def point_nll(
    log_probabilities: torch.Tensor,
    targets: torch.Tensor,
    bin_edges: np.ndarray,
) -> torch.Tensor:
    """Per-example negative log-likelihood on the declared output grid."""

    interior_edges = torch.as_tensor(
        bin_edges[1:-1], dtype=targets.dtype, device=targets.device
    )
    indices = torch.bucketize(targets.contiguous(), interior_edges)
    return F.nll_loss(log_probabilities, indices, reduction="none")


def censored_nll(
    log_probabilities: torch.Tensor,
    observed: torch.Tensor,
    caps: torch.Tensor,
    censored: torch.Tensor,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
) -> torch.Tensor:
    """Use point likelihood for U and survival likelihood for right-censored R."""

    point_log_likelihood = -point_nll(log_probabilities, observed, bin_edges)
    centers = torch.as_tensor(
        bin_centers, dtype=caps.dtype, device=caps.device
    )
    survival_support = centers.unsqueeze(0).ge(caps.unsqueeze(1))
    survival = (
        log_probabilities.exp() * survival_support
    ).sum(dim=1).clamp_min(1e-7)
    log_likelihood = torch.where(
        censored.ge(0.5), survival.log(), point_log_likelihood
    )
    return -log_likelihood


def pinball_loss(
    predictions: torch.Tensor,
    observed: torch.Tensor,
    quantile_levels: np.ndarray,
) -> torch.Tensor:
    """Per-example mean pinball loss over the common quantile grid."""

    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    ).unsqueeze(0)
    error = observed.unsqueeze(1) - predictions
    return torch.maximum(levels * error, (levels - 1.0) * error).mean(dim=1)


def censored_pinball_loss(
    predictions: torch.Tensor,
    observed: torch.Tensor,
    caps: torch.Tensor,
    censored: torch.Tensor,
    quantile_levels: np.ndarray,
) -> torch.Tensor:
    """Censored pinball loss (Powell-type one-sided penalty).

    For a right-censored observation the latent target only satisfies
    ``A >= C`` (and the observable equals the cap, ``Y = C``), so:

    - predicted quantiles below the cap pay ``tau * (C - q)``, pushing the
      upper quantiles toward the censored mass;
    - quantiles at or above the cap pay nothing: any ``q >= C`` is consistent
      with the bound ``A >= C``, so penalising them (as the standard pinball
      against ``Y = C`` would) pins upper quantiles to the cap and makes the
      loss identical to the point-label pinball;
    - uncensored rows keep the standard pinball error against the observed
      value.
    """

    censored_rows = censored.ge(0.5)
    error = observed.unsqueeze(1) - predictions
    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    ).unsqueeze(0)
    standard = torch.maximum(levels * error, (levels - 1.0) * error)
    below_cap = predictions.lt(caps.unsqueeze(1))
    censored_penalty = levels * (caps.unsqueeze(1) - predictions)
    return torch.where(
        censored_rows.unsqueeze(1),
        torch.where(below_cap, censored_penalty, torch.zeros_like(standard)),
        standard,
    ).mean(dim=1)


def state_dict_hash(model: nn.Module) -> str:
    """Hash initialized weights to audit paired model starts."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _select_channels(
    split: ObservableSplit,
    feature_channels: Optional[Tuple[int, ...]],
) -> ObservableSplit:
    """Restrict the input channels an adapter may consume; labels unchanged."""

    if feature_channels is None:
        return split
    return ObservableSplit(
        features=np.ascontiguousarray(
            split.features[:, list(feature_channels), :]
        ),
        observed=split.observed,
        cap=split.cap,
        censored=split.censored,
    )


def _make_loader(
    split: ObservableSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _FeatureDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _loss_for_mode(
    mode: str,
    output: torch.Tensor,
    observed: torch.Tensor,
    cap: torch.Tensor,
    censored: torch.Tensor,
    context: BenchmarkContext,
) -> Tuple[Optional[torch.Tensor], int]:
    if mode == "B1":
        keep = censored.lt(0.5)
        weight = int(keep.sum().item())
        if weight == 0:
            return None, 0
        return point_nll(output[keep], observed[keep], context.bin_edges).mean(), weight
    if mode == "B4":
        return point_nll(output, observed, context.bin_edges).mean(), len(observed)
    if mode == "B6":
        return (
            censored_nll(
                output,
                observed,
                cap,
                censored,
                context.bin_edges,
                context.bin_centers,
            ).mean(),
            len(observed),
        )
    if mode == "BQR":
        return (
            pinball_loss(output, observed, context.quantile_levels).mean(),
            len(observed),
        )
    if mode == "CLQR":
        return (
            censored_pinball_loss(
                output, observed, cap, censored, context.quantile_levels
            ).mean(),
            len(observed),
        )
    raise BenchmarkError("unsupported training mode %r" % mode)


class _TorchTCNAdapter(ModelAdapter):
    """Shared deterministic training loop for learned TCN baselines."""

    mode = ""
    version = "2"
    stochastic = True
    # None means all four declared channels; a tuple restricts both training
    # and prediction inputs to the listed channel indices.
    feature_channels: Optional[Tuple[int, ...]] = None

    def _n_features(self) -> int:
        return 4 if self.feature_channels is None else len(self.feature_channels)

    def _build_model(self, context: BenchmarkContext) -> nn.Module:
        if self.mode in {"BQR", "CLQR"}:
            return QuantileTCN(context, n_features=self._n_features())
        return PMFTCN(context, n_features=self._n_features())

    def fit(
        self,
        data: FitData,
        context: BenchmarkContext,
        seed: Optional[int],
    ) -> FitResult:
        if seed is None:
            raise BenchmarkError("stochastic model %s requires a seed" % self.name)
        set_determinism(seed)
        device = torch.device(context.device)
        model = self._build_model(context).to(device)
        initial_hash = state_dict_hash(model)
        initial_encoder_hash = state_dict_hash(model.encoder)
        training = context.config["training"]
        batch_size = int(training["batch_size"])
        train_loader = _make_loader(
            _select_channels(data.train, self.feature_channels),
            batch_size,
            True,
            seed,
        )
        validation_loader = _make_loader(
            _select_channels(data.validation, self.feature_channels),
            batch_size,
            False,
            seed,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        max_epochs = int(training["max_epochs"])
        min_epochs = int(training["min_epochs"])
        patience = int(training["patience"])
        min_delta = float(training["min_delta"])
        best_loss = float("inf")
        best_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0
        stopped_by_patience = False
        history = []

        for epoch_index in range(max_epochs):
            model.train()
            train_sum = 0.0
            train_weight = 0
            for features, observed, cap, censored in train_loader:
                features = features.to(device, non_blocking=True)
                observed = observed.to(device, non_blocking=True)
                cap = cap.to(device, non_blocking=True)
                censored = censored.to(device, non_blocking=True)
                output = model(features)
                loss, weight = _loss_for_mode(
                    self.mode, output, observed, cap, censored, context
                )
                if loss is None:
                    continue
                if not bool(torch.isfinite(loss)):
                    raise BenchmarkError(
                        "%s produced a non-finite training loss" % self.name
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                optimizer.step()
                train_sum += float(loss.detach().cpu()) * weight
                train_weight += weight
            if train_weight == 0:
                raise BenchmarkError("%s has no usable training labels" % self.name)

            model.eval()
            validation_sum = 0.0
            validation_weight = 0
            with torch.no_grad():
                for features, observed, cap, censored in validation_loader:
                    features = features.to(device, non_blocking=True)
                    observed = observed.to(device, non_blocking=True)
                    cap = cap.to(device, non_blocking=True)
                    censored = censored.to(device, non_blocking=True)
                    output = model(features)
                    loss, weight = _loss_for_mode(
                        self.mode, output, observed, cap, censored, context
                    )
                    if loss is None:
                        continue
                    if not bool(torch.isfinite(loss)):
                        raise BenchmarkError(
                            "%s produced a non-finite validation loss" % self.name
                        )
                    validation_sum += float(loss.cpu()) * weight
                    validation_weight += weight
            if validation_weight == 0:
                raise BenchmarkError("%s has no usable validation labels" % self.name)

            epoch = epoch_index + 1
            validation_loss = validation_sum / validation_weight
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_sum / train_weight,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            if epoch >= min_epochs and bad_epochs >= patience:
                stopped_by_patience = True
                break

        if best_state is None:
            raise BenchmarkError("%s has no finite validation checkpoint" % self.name)
        model.load_state_dict(best_state)
        epochs_run = len(history)
        return FitResult(
            estimator=model,
            history=history,
            metadata={
                "mode": self.mode,
                "seed": int(seed),
                "feature_channels": (
                    "all"
                    if self.feature_channels is None
                    else list(self.feature_channels)
                ),
                "label_source": (
                    "latent_truth" if self.requires_latent_truth else "observable"
                ),
                "initial_state_hash": initial_hash,
                "initial_encoder_hash": initial_encoder_hash,
                "trainable_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                ),
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "stopped_by_patience": stopped_by_patience,
                "hit_max_epochs": epochs_run == max_epochs,
                "converged": stopped_by_patience,
                "checkpoint": "minimum native validation loss",
            },
        )

    def predict(
        self,
        fitted: FitResult,
        data: PredictionData,
        context: BenchmarkContext,
    ) -> Forecast:
        model = fitted.estimator
        model.eval()
        device = torch.device(context.device)
        loader = DataLoader(
            _PredictionDataset(data, self.feature_channels),
            batch_size=int(context.config["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        outputs = []
        with torch.no_grad():
            for features in loader:
                outputs.append(model(features.to(device, non_blocking=True)).cpu().numpy())
        raw = np.concatenate(outputs, axis=0)
        if self.mode in {"BQR", "CLQR"}:
            quantiles = np.sort(raw, axis=1)
            quantiles = np.clip(
                quantiles,
                float(context.config["forecast"]["support_min_pu"]),
                float(context.config["forecast"]["support_max_pu"]),
            ).astype(np.float32)
            return Forecast(
                quantiles=quantiles,
                metadata={"postprocessing": "monotone rearrangement and support clipping"},
            )
        probabilities = np.exp(raw).astype(np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return Forecast(
            quantiles=pmf_to_quantiles(
                probabilities, context.bin_centers, context.quantile_levels
            ),
            pmf=probabilities,
            metadata={"support": "common finite probability grid"},
        )

    def save_fit(self, fitted: FitResult, output_dir: Path) -> Dict[str, str]:
        """Persist the selected CPU checkpoint without optimizer state."""

        path = output_dir / "model_state.pt"
        temporary = output_dir / "model_state.pt.tmp"
        state = {
            name: value.detach().cpu()
            for name, value in fitted.estimator.state_dict().items()
        }
        torch.save(state, temporary)
        temporary.replace(path)
        return {"model_state": path.name}


@register_model
class PersistenceAdapter(ModelAdapter):
    """Last-observation persistence point forecast."""

    name = "B0_persistence"
    version = "2"
    stochastic = False

    def fit(
        self,
        data: FitData,
        context: BenchmarkContext,
        seed: Optional[int],
    ) -> FitResult:
        return FitResult(estimator=None, metadata={"fitted": False})

    def predict(
        self,
        fitted: FitResult,
        data: PredictionData,
        context: BenchmarkContext,
    ) -> Forecast:
        values = np.clip(
            data.persistence,
            float(context.config["forecast"]["support_min_pu"]),
            float(context.config["forecast"]["support_max_pu"]),
        )
        quantiles = np.repeat(
            values[:, None], len(context.quantile_levels), axis=1
        ).astype(np.float32)
        return Forecast(
            quantiles=quantiles,
            metadata={"distribution": "degenerate at last observed power"},
        )


@register_model
class ClimatologyAdapter(ModelAdapter):
    """Training-only empirical marginal distribution baseline."""

    name = "B0_climatology"
    version = "2"
    stochastic = False

    def fit(
        self,
        data: FitData,
        context: BenchmarkContext,
        seed: Optional[int],
    ) -> FitResult:
        counts, _ = np.histogram(data.train_observed_rows, bins=context.bin_edges)
        probabilities = counts.astype(np.float64) + 1e-6
        probabilities /= probabilities.sum()
        return FitResult(
            estimator=probabilities.astype(np.float32),
            metadata={
                "fitted": True,
                "source": "all observed-power rows in the training split only",
                "laplace_smoothing": 1e-6,
            },
        )

    def predict(
        self,
        fitted: FitResult,
        data: PredictionData,
        context: BenchmarkContext,
    ) -> Forecast:
        probabilities = np.repeat(
            np.asarray(fitted.estimator)[None, :], len(data), axis=0
        )
        return Forecast(
            quantiles=pmf_to_quantiles(
                probabilities, context.bin_centers, context.quantile_levels
            ),
            pmf=probabilities,
            metadata={"distribution": "training-only empirical climatology"},
        )

    def save_fit(self, fitted: FitResult, output_dir: Path) -> Dict[str, str]:
        """Persist fitted climatology probabilities."""

        path = output_dir / "fitted_parameters.npz"
        temporary = output_dir / "fitted_parameters.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, probabilities=fitted.estimator)
        temporary.replace(path)
        return {"fitted_parameters": path.name}


@register_model
class B1Adapter(_TorchTCNAdapter):
    """TCN trained only on uncensored targets."""

    name = "B1"
    mode = "B1"


@register_model
class B4Adapter(_TorchTCNAdapter):
    """Naive TCN treating curtailed observations as point targets."""

    name = "B4"
    mode = "B4"


@register_model
class B6Adapter(_TorchTCNAdapter):
    """TCN trained with right-censored likelihood."""

    name = "B6"
    mode = "B6"


@register_model
class BQRAdapter(_TorchTCNAdapter):
    """Direct multi-quantile TCN trained on observed targets."""

    name = "BQR"
    mode = "BQR"


@register_model
class B4PlainAdapter(_TorchTCNAdapter):
    """Traditional SCADA point-label TCN blind to cap and censor channels.

    Both training and prediction see only observed power and standardized
    wind, so the comparison B4 versus B4_plain isolates the value of the
    control-state inputs.
    """

    name = "B4_plain"
    mode = "B4"
    feature_channels = (0, 1)


@register_model
class OracleAdapter(_TorchTCNAdapter):
    """Same-architecture upper bound trained on complete latent truth.

    Training and validation labels are the latent PAP itself, so this model
    reports the gap between censoring-aware methods and full-truth training.
    It is excluded from the RQ1 confirmatory hypothesis family.
    """

    name = "ORACLE"
    mode = "B4"
    requires_latent_truth = True
