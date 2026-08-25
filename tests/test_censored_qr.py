"""Unit tests for the censored pinball loss and the CL-QR adapter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from censored_wind_power.benchmark.core import BenchmarkContext, list_models
from censored_wind_power.benchmark.data import FitData, ObservableSplit
from censored_wind_power.benchmark.models import (
    censored_pinball_loss,
    pinball_loss,
)
from censored_wind_power.benchmark.protocol import load_config
from censored_wind_power.benchmark.censored_qr import CLQRAdapter


LEVELS = np.asarray([0.1, 0.5, 0.9])


def _tiny_config() -> dict:
    config = load_config(PROJECT_DIR / "configs" / "pap_benchmark_v2.json")
    config["model"] = {
        "hidden_channels": 4,
        "layers": 1,
        "kernel_size": 2,
        "dropout": 0.0,
    }
    config["training"].update(
        {
            "min_epochs": 1,
            "max_epochs": 2,
            "patience": 1,
            "batch_size": 32,
            "fail_on_nonconvergence": False,
        }
    )
    config["forecast"] = {
        "window_minutes": 8,
        "horizon_minutes": 2,
        "support_min_pu": 0.0,
        "support_max_pu": 1.1,
        "n_bins": 22,
    }
    config["execution"] = {"device": "cpu"}
    return config


def _context(config: dict, output_dir: Path) -> BenchmarkContext:
    edges = np.linspace(
        config["forecast"]["support_min_pu"],
        config["forecast"]["support_max_pu"],
        config["forecast"]["n_bins"] + 1,
    )
    return BenchmarkContext(
        scenario="test",
        protocol_id="test-protocol",
        output_dir=output_dir,
        device="cpu",
        config=config,
        quantile_levels=np.asarray(config["evaluation"]["quantiles"]),
        bin_edges=edges,
        bin_centers=0.5 * (edges[:-1] + edges[1:]),
    )


class CensoredPinballLossTests(unittest.TestCase):
    def test_uncensored_rows_reproduce_plain_pinball(self) -> None:
        rng = np.random.default_rng(3)
        predictions = torch.as_tensor(rng.random((12, 3)), dtype=torch.float32)
        observed = torch.as_tensor(rng.random(12), dtype=torch.float32)
        caps = torch.full((12,), 0.8)
        censored = torch.zeros(12)
        np.testing.assert_allclose(
            censored_pinball_loss(predictions, observed, caps, censored, LEVELS)
            .detach()
            .numpy(),
            pinball_loss(predictions, observed, LEVELS).detach().numpy(),
            rtol=1e-6,
        )

    def test_censored_row_below_cap_pays_tau_times_gap(self) -> None:
        # One censored row with cap 0.8 and predictions (0.4, 0.8, 1.0).
        predictions = torch.as_tensor([[0.4, 0.8, 1.0]], dtype=torch.float32)
        observed = torch.as_tensor([0.8], dtype=torch.float32)
        caps = torch.as_tensor([0.8], dtype=torch.float32)
        censored = torch.ones(1)
        loss = censored_pinball_loss(
            predictions, observed, caps, censored, LEVELS
        ).item()
        expected = np.mean(
            [
                0.1 * (0.8 - 0.4),  # q below cap: tau * (C - q)
                0.0,  # q at cap: consistent with the bound, zero penalty
                0.0,  # q above cap: consistent with the bound, zero penalty
            ]
        )
        self.assertAlmostEqual(loss, expected, places=6)

    def test_censored_rows_above_cap_pay_nothing(self) -> None:
        # Every quantile above the cap is consistent with A >= C, so a fully
        # censored batch predicted above the cap must incur zero loss.
        predictions = torch.full((5, 3), 0.9, dtype=torch.float32)
        caps = torch.full((5,), 0.7)
        observed = caps.clone()
        censored = torch.ones(5)
        loss = censored_pinball_loss(predictions, observed, caps, censored, LEVELS)
        np.testing.assert_allclose(loss.detach().numpy(), 0.0, atol=1e-7)

    def test_censored_loss_diverges_from_point_label_pinball(self) -> None:
        # Regression guard for the v1 flaw: with observed == cap on censored
        # rows, charging the standard pinball against the cap made the
        # censored loss exactly equal to the point-label pinball (CL-QR
        # identical to BQR). The Powell-type penalty must differ whenever a
        # censored-row quantile sits above the cap.
        rng = np.random.default_rng(5)
        predictions = torch.as_tensor(rng.random((20, 3)), dtype=torch.float32)
        caps = torch.full((20,), 0.7)
        censored = torch.ones(20)
        censored_loss = censored_pinball_loss(
            predictions, caps, caps, censored, LEVELS
        ).detach().numpy()
        plain = pinball_loss(predictions, caps, LEVELS).detach().numpy()
        above_cap = predictions.gt(0.7).any(dim=1).numpy()
        below_or_at = ~above_cap
        if below_or_at.any():
            np.testing.assert_allclose(
                censored_loss[below_or_at], plain[below_or_at], rtol=1e-6
            )
        if above_cap.any():
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(
                    censored_loss[above_cap], plain[above_cap], rtol=1e-6
                )
            self.assertTrue((censored_loss[above_cap] <= plain[above_cap]).all())


class CLQRAdapterTests(unittest.TestCase):
    def test_adapter_is_registered_and_observable_only(self) -> None:
        self.assertIn("CLQR", list_models())
        adapter = CLQRAdapter()
        self.assertTrue(adapter.stochastic)
        self.assertFalse(adapter.requires_latent_truth)
        self.assertFalse(adapter.requires_reconstruction_inputs)

    def test_fit_reports_censoring_audit_and_valid_quantiles(self) -> None:
        config = _tiny_config()
        rng = np.random.default_rng(9)
        features = rng.random((64, 4, 8)).astype(np.float32)
        observed = rng.random(64).astype(np.float32)
        censored = np.zeros(64, dtype=np.float32)
        censored[::4] = 1.0
        observed[censored > 0.5] = 0.6
        split = ObservableSplit(
            features=features,
            observed=observed,
            cap=np.full(64, 0.6, dtype=np.float32),
            censored=censored,
        )
        data = FitData(
            train=split,
            validation=split,
            train_observed_rows=observed,
            feature_names=("p", "w", "c", "d"),
        )
        adapter = CLQRAdapter()
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(config, Path(temporary))
            fitted = adapter.fit(data, context, seed=42)
        self.assertEqual(
            fitted.metadata["censoring_audit"]["train_censored_windows"], 16
        )
        self.assertEqual(
            fitted.metadata["objective"]["loss"], "censored_pinball"
        )
        self.assertEqual(fitted.metadata["label_source"], "observable")


if __name__ == "__main__":
    unittest.main()
