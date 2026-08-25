"""Parametric censored-Gaussian (Tobit) baseline on the shared TCN backbone.

This adapter intentionally lives outside ``models.py``: model identity is
bound to the adapter source file, so adding the baseline in a separate module
keeps every previously frozen artifact reusable. The training protocol is an
exact copy of the shared TCN driver; only the output head and the likelihood
change, keeping the distributional assumption as the single varied factor.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .core import (
    BenchmarkContext,
    BenchmarkError,
    FitResult,
    Forecast,
    register_model,
)
from .data import FitData, PredictionData
from .models import (
    TCNEncoder,
    _make_loader,
    _PredictionDataset,
    _TorchTCNAdapter,
    state_dict_hash,
)
from .protocol import set_determinism
from .scoring import pmf_to_quantiles

_HALF_LOG_TWO_PI = 0.5 * float(np.log(2.0 * np.pi))


def _sigma_floor(context: BenchmarkContext) -> float:
    """Declared scale floor; read from model settings for auditability."""

    settings = context.config.get("model_settings", {}).get("BT", {})
    return float(settings.get("sigma_floor", 1e-3))


class TobitTCN(nn.Module):
    """Shared TCN encoder with a heteroscedastic censored-Gaussian head."""

    def __init__(self, context: BenchmarkContext) -> None:
        super().__init__()
        architecture = context.config["model"]
        hidden = int(architecture["hidden_channels"])
        dropout = float(architecture["dropout"])
        self.sigma_floor = _sigma_floor(context)
        self.encoder = TCNEncoder(
            n_features=4,
            hidden=hidden,
            layers=int(architecture["layers"]),
            kernel_size=int(architecture["kernel_size"]),
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.head(self.encoder(inputs))
        location = output[:, 0]
        scale = F.softplus(output[:, 1]) + self.sigma_floor
        return location, scale


def tobit_nll(
    location: torch.Tensor,
    scale: torch.Tensor,
    observed: torch.Tensor,
    caps: torch.Tensor,
    censored: torch.Tensor,
) -> torch.Tensor:
    """Gaussian point likelihood for U rows, survival likelihood for R rows.

    The survival term uses ``log_ndtr`` so log(1 - Phi((c - mu) / sigma)) stays
    numerically stable without an explicit probability clamp.
    """

    standardized = (observed - location) / scale
    point_nll = _HALF_LOG_TWO_PI + scale.log() + 0.5 * standardized.pow(2)
    survival_log = torch.special.log_ndtr((location - caps) / scale)
    return torch.where(censored.ge(0.5), -survival_log, point_nll)


@register_model
class TobitAdapter(_TorchTCNAdapter):
    """Censored-Gaussian (Tobit) TCN trained with the shared protocol."""

    name = "BT"
    mode = "BT"
    version = "1"
    stochastic = True

    def _build_model(self, context: BenchmarkContext) -> nn.Module:
        return TobitTCN(context)

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
        train_loader = _make_loader(data.train, batch_size, True, seed)
        validation_loader = _make_loader(data.validation, batch_size, False, seed)
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
                location, scale = model(features)
                loss = tobit_nll(location, scale, observed, cap, censored).mean()
                weight = len(observed)
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
                    location, scale = model(features)
                    loss = tobit_nll(location, scale, observed, cap, censored).mean()
                    weight = len(observed)
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
                "sigma_floor": _sigma_floor(context),
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
            _PredictionDataset(data),
            batch_size=int(context.config["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        locations = []
        scales = []
        with torch.no_grad():
            for features in loader:
                location, scale = model(features.to(device, non_blocking=True))
                locations.append(location.cpu().numpy())
                scales.append(scale.cpu().numpy())
        location = np.concatenate(locations, axis=0).astype(np.float64)
        scale = np.concatenate(scales, axis=0).astype(np.float64)
        interior_edges = np.asarray(context.bin_edges, dtype=np.float64)[1:-1]
        standardized = (
            interior_edges[None, :] - location[:, None]
        ) / scale[:, None]
        cdf = torch.special.ndtr(torch.from_numpy(standardized)).numpy()
        probabilities = np.concatenate(
            [cdf[:, :1], np.diff(cdf, axis=1), 1.0 - cdf[:, -1:]], axis=1
        )
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities.astype(np.float32)
        return Forecast(
            quantiles=pmf_to_quantiles(
                probabilities, context.bin_centers, context.quantile_levels
            ),
            pmf=probabilities,
            metadata={
                "distribution": "heteroscedastic Gaussian discretised to the common grid",
                "tail_mass": "folded into the boundary support bins",
            },
        )
