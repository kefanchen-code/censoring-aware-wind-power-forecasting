"""Censored-aware quantile regression baseline (CL-QR).

CL-QR shares the QuantileTCN backbone, training driver, and prediction
postprocessing of BQR; the single varied factor is the objective. Instead of
the point-label pinball loss on observed targets, CL-QR minimises the censored
pinball loss (Powell-type one-sided penalty): on a right-censored row
(observable target = cap C, latent truth only known to exceed C) every
predicted quantile below the cap pays tau * (C - q), while quantiles at or
above the cap pay nothing. This keeps the model free of any latent truth
while letting the upper quantiles react to censoring mass, which is the
mechanism the point-label BQR lacks. Version 2 note: the original v1 penalty
also charged (1-tau) * (q - C) on censored rows with q >= C, which made the
loss exactly equal to the point-label pinball (observable Y = C on censored
rows) and rendered CL-QR identical to BQR; v2 zeroes that branch.

Information isolation: training consumes exactly the observable quadruple
(features, observed, cap, censored) handed to every learned baseline; no
reconstruction view and no latent truth. The fit audit records how many
training and validation windows were censored so the injected censoring mass
is traceable in the artifact manifest.
"""

from __future__ import annotations

from typing import Optional

from .core import BenchmarkContext, BenchmarkError, FitResult, register_model
from .data import FitData
from .models import _TorchTCNAdapter


@register_model
class CLQRAdapter(_TorchTCNAdapter):
    """Direct multi-quantile TCN trained with the censored pinball loss."""

    name = "CLQR"
    mode = "CLQR"
    version = "2"
    stochastic = True

    def fit(
        self,
        data: FitData,
        context: BenchmarkContext,
        seed: Optional[int],
    ) -> FitResult:
        if not isinstance(data, FitData):
            raise BenchmarkError(
                "%s requires the standard observable fit view" % self.name
            )
        audit = {
            "train_windows": int(len(data.train.observed)),
            "train_censored_windows": int(data.train.censored.sum()),
            "validation_windows": int(len(data.validation.observed)),
            "validation_censored_windows": int(data.validation.censored.sum()),
        }
        fitted = super().fit(data, context, seed)
        fitted.metadata.update(
            {
                "label_source": "observable",
                "objective": {
                    "loss": "censored_pinball",
                    "uncensored_rows": "standard pinball error against observed",
                    "censored_rows_below_cap": "tau * (cap - q) one-sided penalty",
                    "censored_rows_at_or_above_cap": "zero penalty (consistent with bound)",
                },
                "censoring_audit": audit,
            }
        )
        return fitted
