"""Tests for the extensible PAP benchmark protocol."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from censored_wind_power.benchmark.core import (
    BenchmarkContext,
    BenchmarkError,
    ConfigurationError,
    Forecast,
)
from censored_wind_power.benchmark.data import (
    FitData,
    ObservableSplit,
    PredictionData,
    ReconstructionFitData,
    apply_input_contamination,
    apply_segment_assignment,
    make_segment_assignment,
    prepare_scenario,
)
from censored_wind_power.benchmark.models import (
    B4Adapter,
    B4PlainAdapter,
    B6Adapter,
    ClimatologyAdapter,
    OracleAdapter,
    PersistenceAdapter,
    censored_nll,
    state_dict_hash,
)
from censored_wind_power.benchmark.powercurve import (
    B2ReconAdapter,
    fit_power_curve,
    relabel_split,
)
from censored_wind_power.benchmark.protocol import (
    load_config,
    protocol_payload,
    set_determinism,
    stable_json_hash,
    validate_config,
)
from censored_wind_power.benchmark.runner import (
    _equivalence_verdict,
    aggregate_existing,
    run_benchmark,
)
from censored_wind_power.benchmark.scoring import (
    crps_from_pmf,
    stratified_rows,
    underestimation_rate,
    wis_components_per_sample,
    wis_per_sample,
)


def _base_config() -> dict:
    return load_config(PROJECT_DIR / "configs" / "pap_benchmark_v2.json")


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


class ProtocolUnitTests(unittest.TestCase):
    def test_new_model_list_does_not_change_data_protocol(self) -> None:
        config = _base_config()
        extended = copy.deepcopy(config)
        extended["models"].append("future_model")
        extended["model_modules"] = ["future.module"]
        extended["output_dir"] = "another/output"
        self.assertEqual(
            stable_json_hash(protocol_payload(config)),
            stable_json_hash(protocol_payload(extended)),
        )

    def test_whole_segments_do_not_cross_splits(self) -> None:
        frame = pd.DataFrame({"segment_id": np.repeat(np.arange(12), 20)})
        assignment = make_segment_assignment(frame, 0.6, 0.2)
        parts = apply_segment_assignment(frame, assignment)
        segment_sets = {name: set(part["segment_id"]) for name, part in parts.items()}
        self.assertTrue(segment_sets["train"].isdisjoint(segment_sets["validation"]))
        self.assertTrue(segment_sets["train"].isdisjoint(segment_sets["test"]))
        self.assertTrue(segment_sets["validation"].isdisjoint(segment_sets["test"]))

    def test_prediction_interface_contains_no_truth(self) -> None:
        prediction = PredictionData(
            features=np.zeros((3, 4, 5), dtype=np.float32),
            persistence=np.zeros(3, dtype=np.float32),
            feature_names=("a", "b", "c", "d"),
        )
        self.assertFalse(hasattr(prediction, "truth"))
        self.assertFalse(hasattr(prediction, "segment_id"))

    def test_censored_likelihood_includes_the_cap_bin(self) -> None:
        config = _base_config()
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(config, Path(temporary))
            probabilities = np.linspace(1.0, 2.0, len(context.bin_centers))
            probabilities /= probabilities.sum()
            log_probabilities = torch.log(
                torch.tensor(probabilities[None, :], dtype=torch.float32)
            )
            cap_index = 40
            cap = torch.tensor(
                [context.bin_centers[cap_index]], dtype=torch.float32
            )
            actual = censored_nll(
                log_probabilities,
                cap,
                cap,
                torch.ones(1),
                context.bin_edges,
                context.bin_centers,
            )
            expected = -np.log(probabilities[cap_index:].sum())
            self.assertAlmostEqual(float(actual), float(expected), places=5)

    def test_fast_discrete_crps_matches_definition(self) -> None:
        probabilities = np.asarray([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
        centers = np.asarray([0.0, 0.5, 1.0])
        targets = np.asarray([0.4, 0.8])
        distance = np.abs(centers[:, None] - centers[None, :])
        brute_force = np.sum(
            probabilities * np.abs(centers[None, :] - targets[:, None]), axis=1
        ) - 0.5 * np.einsum("bi,ij,bj->b", probabilities, distance, probabilities)
        np.testing.assert_allclose(
            crps_from_pmf(probabilities, targets, centers), brute_force, atol=1e-12
        )

    def test_wis_is_zero_for_a_perfect_degenerate_forecast(self) -> None:
        config = _base_config()
        levels = np.asarray(config["evaluation"]["quantiles"])
        targets = np.asarray([0.2, 0.7])
        quantiles = np.repeat(targets[:, None], len(levels), axis=1)
        actual = wis_per_sample(
            quantiles, targets, levels, config["evaluation"]["interval_alphas"]
        )
        np.testing.assert_allclose(actual, 0.0)

    def test_paired_pmf_models_start_from_identical_weights(self) -> None:
        config = _base_config()
        config["model"] = {
            "hidden_channels": 4,
            "layers": 1,
            "kernel_size": 2,
            "dropout": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(config, Path(temporary))
            set_determinism(42)
            b4 = B4Adapter()._build_model(context)
            set_determinism(42)
            b6 = B6Adapter()._build_model(context)
            self.assertEqual(state_dict_hash(b4), state_dict_hash(b6))

    def test_forecast_rejects_crossing_quantiles(self) -> None:
        config = _base_config()
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(config, Path(temporary))
            quantiles = np.zeros((2, len(context.quantile_levels)), dtype=np.float32)
            quantiles[0, 2] = -1.0
            with self.assertRaises(ValueError):
                Forecast(quantiles=quantiles).validate(2, context)


class RunnerIntegrationTests(unittest.TestCase):
    @staticmethod
    def _synthetic_frame() -> pd.DataFrame:
        segments = np.repeat(np.arange(12), 24)
        time = np.arange(len(segments))
        truth = 500.0 + 250.0 * np.sin(time / 18.0)
        cap = np.where((time % 17) < 6, 580.0, 1400.0)
        active = cap < 1400.0
        censored = active & (truth >= cap)
        frame = pd.DataFrame(
            {
                "A_true": truth,
                "Y_syn": np.minimum(truth, cap),
                "C_syn": cap,
                "wind": 8.0 + np.sin(time / 25.0),
                "is_censored": censored,
                "policy_active": active,
                "segment_id": segments,
            }
        )
        frame.index = pd.date_range("2025-01-01", periods=len(frame), freq="min")
        frame.index.name = "timestamp"
        return frame

    def test_new_adapter_can_be_added_without_refitting_existing_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            self._synthetic_frame().to_parquet(data_dir / "synthetic_S1.parquet")
            config = _base_config()
            config.update(
                {
                    "output_dir": str(root / "output"),
                    "model_modules": [],
                    "scenarios": ["S1"],
                    "models": ["B0_persistence", "B0_climatology"],
                    "seeds": [42],
                    "data": {
                        "directory": str(data_dir),
                        "filename_pattern": "synthetic_{scenario}.parquet",
                        "rated_power_kw": 1330.0,
                        "sampling_interval_minutes": 1,
                    },
                    "split": {
                        "strategy": "chronological_whole_segment",
                        "train_fraction": 0.5,
                        "validation_fraction": 0.25,
                    },
                    "forecast": {
                        "window_minutes": 4,
                        "horizon_minutes": 2,
                        "support_min_pu": 0.0,
                        "support_max_pu": 1.1,
                        "n_bins": 22,
                    },
                    "comparisons": [],
                    "execution": {"device": "cpu"},
                }
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            first = run_benchmark(root, config_path)
            protocol_id = first["protocol_id"]

            config["models"].append("example_residual_persistence")
            config["model_modules"] = [
                "censored_wind_power.benchmark.example_adapter"
            ]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            second = run_benchmark(
                root,
                config_path,
                models_override=["example_residual_persistence"],
            )
            self.assertEqual(protocol_id, second["protocol_id"])
            status = aggregate_existing(root, config, require_complete=True)
            self.assertTrue(status["valid_for_final_comparison"])
            self.assertEqual(second["completed"], 1)
            self.assertEqual(second["reused"], 0)

            forecast_path = (
                root
                / "output"
                / "artifacts"
                / "S1"
                / "example_residual_persistence"
                / "fixed"
                / "forecast.npz"
            )
            with forecast_path.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaises(BenchmarkError):
                aggregate_existing(root, config, require_complete=True)


class SemisynExtensionTests(unittest.TestCase):
    """B4_plain channel isolation, oracle gating, and stratified metrics."""

    @staticmethod
    def _tiny_config() -> dict:
        config = _base_config()
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

    @staticmethod
    def _fit_data(poison: bool) -> FitData:
        rng = np.random.default_rng(7)
        features = rng.random((96, 4, 8)).astype(np.float32)
        observed = rng.random(96).astype(np.float32)
        if poison:
            features[:, 2, :] = 1e9
            features[:, 3, :] = -1e9
        split = ObservableSplit(
            features=features,
            observed=observed,
            cap=np.ones(96, dtype=np.float32),
            censored=np.zeros(96, dtype=np.float32),
        )
        return FitData(
            train=split,
            validation=split,
            train_observed_rows=observed,
            feature_names=("p", "w", "c", "d"),
        )

    @staticmethod
    def _prediction_data(poison: bool) -> PredictionData:
        rng = np.random.default_rng(11)
        features = rng.random((16, 4, 8)).astype(np.float32)
        if poison:
            features[:, 2, :] = 1e9
            features[:, 3, :] = -1e9
        return PredictionData(
            features=features,
            persistence=features[:, 0, -1],
            feature_names=("p", "w", "c", "d"),
        )

    def test_b4_plain_is_blind_to_cap_and_censor_channels(self) -> None:
        config = self._tiny_config()
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(config, Path(temporary))
            adapter = B4PlainAdapter()
            clean_fit = adapter.fit(self._fit_data(False), context, 42)
            clean = adapter.predict(
                clean_fit, self._prediction_data(False), context
            )
            poisoned_fit = adapter.fit(self._fit_data(True), context, 42)
            poisoned = adapter.predict(
                poisoned_fit, self._prediction_data(True), context
            )
            np.testing.assert_allclose(clean.quantiles, poisoned.quantiles)

    def test_oracle_view_and_leakage_gating(self) -> None:
        config = self._tiny_config()
        config["forecast"]["window_minutes"] = 4
        config["forecast"]["horizon_minutes"] = 2
        frame = RunnerIntegrationTests._synthetic_frame()
        assignment = make_segment_assignment(frame, 0.5, 0.25)
        scenario = prepare_scenario("S1", frame, assignment, config)
        regular = scenario.fit_view()
        self.assertFalse(hasattr(regular.train, "truth"))
        self.assertFalse(hasattr(regular.train, "target_wind_ms"))
        self.assertFalse(hasattr(regular.train, "history_censor_frac"))
        self.assertTrue(np.any(regular.train.censored > 0))
        oracle = scenario.oracle_fit_view()
        np.testing.assert_allclose(oracle.train.observed, scenario.train.truth)
        np.testing.assert_allclose(
            oracle.validation.observed, scenario.validation.truth
        )
        self.assertEqual(float(np.abs(oracle.train.censored).sum()), 0.0)
        self.assertTrue(
            np.any(regular.train.observed < oracle.train.observed)
        )
        self.assertTrue(OracleAdapter.requires_latent_truth)
        self.assertFalse(getattr(B4Adapter, "requires_latent_truth"))
        self.assertFalse(getattr(B4PlainAdapter, "requires_latent_truth"))

    def test_underestimation_and_stratified_metrics(self) -> None:
        levels = np.asarray(
            [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        )
        quantiles = np.tile(np.linspace(0.1, 0.9, 9), (4, 1))
        targets = np.asarray([0.55, 0.40, 0.05, 0.50])
        censored = np.asarray([1.0, 1.0, 0.0, 0.0])
        censored_mask = censored >= 0.5
        self.assertAlmostEqual(
            underestimation_rate(quantiles, targets, levels, 0.5, censored_mask),
            0.5,
        )
        self.assertAlmostEqual(
            underestimation_rate(quantiles, targets, levels, 0.9, censored_mask),
            0.0,
        )
        self.assertIsNone(
            underestimation_rate(
                quantiles, targets, levels, 0.5, np.zeros(4, dtype=bool)
            )
        )
        rows = stratified_rows(
            quantiles,
            targets,
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            np.asarray([0.1, 0.2, 0.3, 0.4]),
            np.asarray([0.5, 0.6, 0.1, 0.4]),
            levels,
            [0.1, 0.2, 0.4],
            censored,
            np.asarray([0.0, 0.2, 0.5, 0.0]),
            np.asarray([0.5, 0.5, 0.7, 0.9]),
            np.asarray([5.0, 8.0, 11.0, 12.0]),
        )
        by_key = {(row["dimension"], row["stratum"]): row for row in rows}
        overall = by_key[("all", "all")]
        self.assertEqual(overall["n_windows"], 4)
        self.assertAlmostEqual(overall["coverage_90"], 0.75)
        censored_row = by_key[("target_censoring", "censored")]
        self.assertEqual(censored_row["n_windows"], 2)
        self.assertAlmostEqual(censored_row["wis_mean"], 1.5)
        self.assertAlmostEqual(censored_row["underestimation_q50"], 0.5)
        self.assertAlmostEqual(censored_row["underestimation_q90"], 0.0)
        self.assertAlmostEqual(censored_row["median_bias"], 0.025)
        self.assertEqual(
            by_key[("history_censor_frac", "0")]["n_windows"], 2
        )
        self.assertEqual(
            by_key[("history_censor_frac", "(0,0.3]")]["n_windows"], 1
        )
        self.assertEqual(
            by_key[("history_censor_frac", "(0.3,1]")]["n_windows"], 1
        )
        self.assertEqual(by_key[("target_cap_tertile", "low")]["n_windows"], 2)
        self.assertEqual(by_key[("target_cap_tertile", "mid")]["n_windows"], 1)
        self.assertEqual(by_key[("target_wind_band", "7-11")]["n_windows"], 2)

    def test_oracle_and_plain_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            RunnerIntegrationTests._synthetic_frame().to_parquet(
                data_dir / "synthetic_S1.parquet"
            )
            config = self._tiny_config()
            config.update(
                {
                    "output_dir": str(root / "output"),
                    "model_modules": [],
                    "scenarios": ["S1"],
                    "models": ["B4_plain", "ORACLE"],
                    "seeds": [42],
                    "data": {
                        "directory": str(data_dir),
                        "filename_pattern": "synthetic_{scenario}.parquet",
                        "rated_power_kw": 1330.0,
                        "sampling_interval_minutes": 1,
                    },
                    "split": {
                        "strategy": "chronological_whole_segment",
                        "train_fraction": 0.5,
                        "validation_fraction": 0.25,
                    },
                    "forecast": {
                        "window_minutes": 4,
                        "horizon_minutes": 2,
                        "support_min_pu": 0.0,
                        "support_max_pu": 1.1,
                        "n_bins": 22,
                    },
                    "comparisons": [],
                }
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = run_benchmark(root, config_path)
            self.assertEqual(result["completed"], 2)
            self.assertTrue((root / "output" / "metrics_stratified.csv").exists())
            artifact_path = (
                root
                / "output"
                / "artifacts"
                / "S1"
                / "ORACLE"
                / "seed_42"
                / "artifact.json"
            )
            with artifact_path.open("r", encoding="utf-8") as handle:
                oracle_artifact = json.load(handle)
            self.assertTrue(oracle_artifact["oracle"])
            training_path = artifact_path.parent / "training.json"
            with training_path.open("r", encoding="utf-8") as handle:
                oracle_training = json.load(handle)
            self.assertEqual(
                oracle_training["metadata"]["label_source"], "latent_truth"
            )
            plain_training_path = (
                root
                / "output"
                / "artifacts"
                / "S1"
                / "B4_plain"
                / "seed_42"
                / "training.json"
            )
            with plain_training_path.open("r", encoding="utf-8") as handle:
                plain_training = json.load(handle)
            self.assertEqual(
                plain_training["metadata"]["feature_channels"], [0, 1]
            )
            self.assertEqual(
                plain_training["metadata"]["label_source"], "observable"
            )


class WisDecompositionTests(unittest.TestCase):
    """Bracher three-way decomposition of the weighted interval score."""

    def test_components_sum_to_wis_and_carry_the_right_sign(self) -> None:
        levels = np.asarray([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
        alphas = [0.10, 0.20, 0.40, 0.60]
        rng = np.random.default_rng(3)
        quantiles = np.sort(rng.random((64, len(levels))), axis=1)
        targets = rng.random(64)
        components = wis_components_per_sample(quantiles, targets, levels, alphas)
        total = (
            components["dispersion"]
            + components["overprediction"]
            + components["underprediction"]
        )
        np.testing.assert_allclose(
            total, wis_per_sample(quantiles, targets, levels, alphas)
        )
        for values in components.values():
            self.assertTrue(np.all(values >= 0.0))

        far_above = wis_components_per_sample(
            quantiles, np.full(64, 10.0), levels, alphas
        )
        self.assertEqual(float(far_above["overprediction"].sum()), 0.0)
        self.assertGreater(float(far_above["underprediction"].sum()), 0.0)
        far_below = wis_components_per_sample(
            quantiles, np.full(64, -10.0), levels, alphas
        )
        self.assertEqual(float(far_below["underprediction"].sum()), 0.0)
        self.assertGreater(float(far_below["overprediction"].sum()), 0.0)


class ContaminationTests(unittest.TestCase):
    """Measured wind-channel contamination: injection, audit, and invariances."""

    @staticmethod
    def _relative_section(factor: float = 1.0) -> dict:
        return {
            "contamination": {
                "enabled": True,
                "channel": "wind",
                "trigger": "is_censored",
                "mode": "relative",
                "relative_delta": 0.167,
                "intensity_factor": factor,
                "measurement_provenance": "unit test",
            }
        }

    @staticmethod
    def _prepared_pair() -> tuple:
        config = SemisynExtensionTests._tiny_config()
        config["forecast"]["window_minutes"] = 4
        config["forecast"]["horizon_minutes"] = 2
        frame = RunnerIntegrationTests._synthetic_frame()
        assignment = make_segment_assignment(frame, 0.5, 0.25)
        contaminated_config = copy.deepcopy(config)
        contaminated_config.update(ContaminationTests._relative_section())
        return (
            config,
            prepare_scenario("S1", frame, assignment, config),
            prepare_scenario("S1", frame, assignment, contaminated_config),
        )

    def test_only_binding_frames_move_and_by_the_declared_amount(self) -> None:
        frame = RunnerIntegrationTests._synthetic_frame()
        contaminated, audit = apply_input_contamination(
            frame, self._relative_section()
        )
        mask = frame["is_censored"].to_numpy(dtype=bool)
        self.assertTrue(mask.any())
        self.assertFalse(mask.all())
        np.testing.assert_allclose(
            contaminated["wind_clean"].to_numpy(), frame["wind"].to_numpy()
        )
        np.testing.assert_allclose(
            contaminated.loc[~mask, "wind"].to_numpy(),
            frame.loc[~mask, "wind"].to_numpy(),
        )
        np.testing.assert_allclose(
            contaminated.loc[mask, "wind"].to_numpy(),
            frame.loc[mask, "wind"].to_numpy() * 1.167,
        )
        self.assertEqual(audit["contaminated_rows"], int(mask.sum()))
        self.assertAlmostEqual(audit["relative_delta_mean"], 0.167)

    def test_disabled_contamination_returns_the_frame_untouched(self) -> None:
        frame = RunnerIntegrationTests._synthetic_frame()
        for section in ({}, {"contamination": {"enabled": False}}):
            passthrough, audit = apply_input_contamination(frame, section)
            self.assertIsNone(audit)
            self.assertIs(passthrough, frame)
            self.assertNotIn("wind_clean", passthrough.columns)

    def test_intensity_factor_scales_the_measured_offset(self) -> None:
        frame = RunnerIntegrationTests._synthetic_frame()
        half, _ = apply_input_contamination(frame, self._relative_section(0.5))
        mask = frame["is_censored"].to_numpy(dtype=bool)
        np.testing.assert_allclose(
            half.loc[mask, "wind"].to_numpy(),
            frame.loc[mask, "wind"].to_numpy() * (1.0 + 0.5 * 0.167),
        )

    def test_schedule_bands_are_looked_up_with_clean_wind(self) -> None:
        frame = RunnerIntegrationTests._synthetic_frame()
        frame["wind"] = 6.9  # inside (6, 7]; a +30% shift would land in (8, 99]
        config = {
            "contamination": {
                "enabled": True,
                "channel": "wind",
                "trigger": "is_censored",
                "mode": "schedule",
                "intensity_factor": 1.0,
                "schedule_bins_ms": [0, 6, 7, 8, 99],
                "schedule_relative_delta": [0.4, 0.3, 0.2, 0.1],
                "measurement_provenance": "unit test",
            }
        }
        contaminated, audit = apply_input_contamination(frame, config)
        mask = frame["is_censored"].to_numpy(dtype=bool)
        np.testing.assert_allclose(
            contaminated.loc[mask, "wind"].to_numpy(), 6.9 * 1.3
        )
        self.assertAlmostEqual(audit["relative_delta_mean"], 0.3)

    def test_evaluation_strata_and_targets_survive_contamination(self) -> None:
        _, clean, dirty = self._prepared_pair()
        np.testing.assert_array_equal(
            clean.test.target_wind_ms, dirty.test.target_wind_ms
        )
        np.testing.assert_array_equal(clean.test.truth, dirty.test.truth)
        np.testing.assert_array_equal(
            clean.test.persistence, dirty.test.persistence
        )
        self.assertTrue(
            np.any(clean.test.features[:, 1, :] != dirty.test.features[:, 1, :])
        )
        clean_standardization = clean.split_summary[
            "wind_standardization_train_only"
        ]["mean"]
        dirty_standardization = dirty.split_summary[
            "wind_standardization_train_only"
        ]["mean"]
        self.assertGreater(dirty_standardization, clean_standardization)
        self.assertNotIn("contamination", clean.split_summary)
        audit = dirty.split_summary["contamination"]
        self.assertEqual(audit["mode"], "relative")
        self.assertGreater(audit["contaminated_rows"], 0)
        self.assertIn("contamination", dirty.split_summary["splits"]["test"])
        self.assertIn("wind_clean_mean_ms", dirty.split_summary["splits"]["test"])
        self.assertNotIn(
            "wind_clean_mean_ms", clean.split_summary["splits"]["test"]
        )

    def test_wind_free_baselines_are_identical_under_contamination(self) -> None:
        config, clean, dirty = self._prepared_pair()
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(config, Path(temporary))
            for adapter in (PersistenceAdapter(), ClimatologyAdapter()):
                clean_forecast = adapter.predict(
                    adapter.fit(clean.fit_view(), context, 42),
                    clean.prediction_view(),
                    context,
                )
                dirty_forecast = adapter.predict(
                    adapter.fit(dirty.fit_view(), context, 42),
                    dirty.prediction_view(),
                    context,
                )
                np.testing.assert_array_equal(
                    clean_forecast.quantiles, dirty_forecast.quantiles
                )

    def test_protocol_payload_ignores_absent_and_disabled_contamination(self) -> None:
        config = _base_config()
        baseline = stable_json_hash(protocol_payload(config))
        disabled = copy.deepcopy(config)
        disabled["contamination"] = {"enabled": False, "relative_delta": 0.167}
        self.assertNotIn("contamination", protocol_payload(disabled))
        self.assertEqual(baseline, stable_json_hash(protocol_payload(disabled)))
        measured = copy.deepcopy(config)
        measured.update(self._relative_section())
        stronger = copy.deepcopy(config)
        stronger.update(self._relative_section(1.5))
        self.assertEqual(
            len(
                {
                    baseline,
                    stable_json_hash(protocol_payload(measured)),
                    stable_json_hash(protocol_payload(stronger)),
                }
            ),
            3,
        )

    def test_shipped_contamination_configs_are_valid_and_distinct(self) -> None:
        configs = PROJECT_DIR / "configs"
        clean_hash = stable_json_hash(
            protocol_payload(load_config(configs / "pap_benchmark_semisyn.json"))
        )
        protocol_ids = set()
        for suffix in ("contam100", "contam050", "contam150", "contam_sched"):
            config = load_config(
                configs / ("pap_benchmark_semisyn_%s.json" % suffix)
            )
            self.assertTrue(config["contamination"]["enabled"])
            self.assertEqual(config["contamination"]["trigger"], "is_censored")
            protocol_ids.add(stable_json_hash(protocol_payload(config)))
        self.assertEqual(len(protocol_ids), 4)
        self.assertNotIn(clean_hash, protocol_ids)

    def test_invalid_contamination_sections_are_rejected(self) -> None:
        broken = [
            {"enabled": True, "channel": "power", "trigger": "is_censored",
             "mode": "relative", "relative_delta": 0.1,
             "measurement_provenance": "unit test"},
            {"enabled": True, "channel": "wind", "trigger": "policy_active",
             "mode": "relative", "relative_delta": 0.1,
             "measurement_provenance": "unit test"},
            {"enabled": True, "channel": "wind", "trigger": "is_censored",
             "mode": "relative", "relative_delta": 0.1},
            {"enabled": True, "channel": "wind", "trigger": "is_censored",
             "mode": "schedule", "schedule_bins_ms": [0, 6, 7],
             "schedule_relative_delta": [0.4],
             "measurement_provenance": "unit test"},
        ]
        for section in broken:
            config = _base_config()
            config["contamination"] = section
            with self.assertRaises(ConfigurationError):
                validate_config(config)


class SharedMaskAndEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _load_audit_module():
        import importlib.util

        path = PROJECT_DIR / "scripts" / "10_altahullion_audit.py"
        spec = importlib.util.spec_from_file_location("audit_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _tiny_base() -> pd.DataFrame:
        rng = np.random.default_rng(7)
        n = 300
        index = pd.date_range("2026-01-01", periods=n, freq="1min", name="timestamp")
        return pd.DataFrame(
            {
                "power": rng.uniform(100.0, 1300.0, n),
                "wind": rng.uniform(4.0, 15.0, n),
                "pitch": rng.uniform(0.0, 5.0, n),
                "gen_rpm": rng.uniform(900.0, 1500.0, n),
                "power_ref": np.full(n, 1330.0),
                "segment_id": np.repeat([0, 1], n // 2),
            },
            index=index,
        )

    def test_s1_s4_share_event_mask_and_s0_is_uncensored(self) -> None:
        module = self._load_audit_module()
        base = self._tiny_base()
        rated = module.P_RATED
        configs = {
            "S1_like": {
                "cap_type": "fixed",
                "cap_value": rated * 0.50,
                "policy_rate": 0.60,
                "duration": 20,
            },
            "S3_like": {
                "cap_type": "fixed",
                "cap_value": rated * 0.85,
                "policy_rate": 0.60,
                "duration": 20,
            },
            "S4_like": {
                "cap_type": "random_event",
                "cap_low": rated * 0.30,
                "cap_high": rated * 0.90,
                "policy_rate": 0.60,
                "duration": 20,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            module.OUT_DIR = Path(temporary)
            frames = {}
            for name, config in configs.items():
                _, frames[name] = module.construct_scenario(base, name, config)
            reference = frames["S1_like"]
            for name in ("S3_like", "S4_like"):
                np.testing.assert_array_equal(
                    reference["policy_active"].to_numpy(),
                    frames[name]["policy_active"].to_numpy(),
                )
                np.testing.assert_array_equal(
                    reference["event_id"].to_numpy(),
                    frames[name]["event_id"].to_numpy(),
                )
            active = reference["policy_active"].to_numpy()
            self.assertTrue(active.any())
            # Depth differs even though the mask is shared.
            self.assertFalse(
                np.array_equal(
                    reference.loc[active, "C_syn"].to_numpy(),
                    frames["S4_like"].loc[active, "C_syn"].to_numpy(),
                )
            )
            np.testing.assert_allclose(
                frames["S4_like"].loc[~active, "C_syn"].to_numpy(), rated * 1.05
            )

            _, s0 = module.construct_scenario(base, "S0_like", {"cap_type": "none"})
            self.assertFalse(s0["policy_active"].any())
            self.assertFalse(s0["is_censored"].any())
            np.testing.assert_array_equal(
                s0["Y_syn"].to_numpy(), s0["A_true"].to_numpy()
            )
            with self.assertRaises(ValueError):
                module.construct_scenario(base, "bad", {"cap_type": "bogus"})

    def test_equivalence_verdict(self) -> None:
        self.assertTrue(_equivalence_verdict([-0.004, 0.0049], 0.005))
        self.assertFalse(_equivalence_verdict([-0.006, 0.001], 0.005))
        self.assertFalse(_equivalence_verdict([0.001, 0.007], 0.005))
        self.assertFalse(_equivalence_verdict([-0.005, 0.004], 0.005))
        self.assertFalse(_equivalence_verdict([0.004, 0.005], 0.005))


class PowerCurveReconstructionTests(unittest.TestCase):
    """B2_recon baseline: power-curve fitting, relabeling, and view gating."""

    @staticmethod
    def _prepared_scenario(contaminated: bool = False):
        config = SemisynExtensionTests._tiny_config()
        config["forecast"]["window_minutes"] = 4
        config["forecast"]["horizon_minutes"] = 2
        config["model_settings"] = {
            "B2_recon": {"bin_width_ms": 0.5, "min_bin_count": 1}
        }
        if contaminated:
            config.update(ContaminationTests._relative_section())
        frame = RunnerIntegrationTests._synthetic_frame()
        assignment = make_segment_assignment(frame, 0.5, 0.25)
        return config, prepare_scenario("S1", frame, assignment, config)

    def test_power_curve_uses_clean_rows_only_and_interpolates(self) -> None:
        wind = np.asarray([5.0, 5.2, 9.0, 9.4, 12.1, 12.3])
        power = np.asarray([0.20, 0.24, 0.60, 0.64, 0.90, 0.94])
        censored = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        curve = fit_power_curve(wind, power, censored, 1.0, 1)
        # The censored 12.1 m/s row must not enter the 12-13 m/s bin mean.
        bin_12 = int(np.floor(12.0 / 1.0))
        self.assertAlmostEqual(float(curve.bin_power_mean[bin_12]), 0.94)
        # Flat extrapolation below the first knot and interpolation in range.
        self.assertAlmostEqual(float(curve.evaluate(np.asarray([0.0]))[0]), float(curve.knot_power_pu[0]))
        mid = curve.evaluate(np.asarray([7.0]))
        self.assertTrue(float(curve.knot_power_pu[0]) < float(mid[0]) < float(curve.knot_power_pu[-1]))
        with self.assertRaises(BenchmarkError):
            fit_power_curve(wind, power, np.ones_like(censored), 1.0, 1)
        with self.assertRaises(BenchmarkError):
            fit_power_curve(wind, power, censored, 1.0, 1000)

    def test_relabel_split_replaces_only_censored_windows(self) -> None:
        rng = np.random.default_rng(5)
        split = ObservableSplit(
            features=rng.random((8, 4, 4)).astype(np.float32),
            observed=rng.random(8).astype(np.float32),
            cap=np.full(8, 0.5, dtype=np.float32),
            censored=np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32),
        )
        wind = np.linspace(5.0, 12.0, 8)
        curve = fit_power_curve(
            np.asarray([5.0, 12.0, 12.2]),
            np.asarray([0.2, 0.9, 0.92]),
            np.zeros(3),
            1.0,
            1,
        )
        relabeled, replaced = relabel_split(split, wind, curve, 0.0, 1.06)
        self.assertEqual(replaced, 4)
        mask = split.censored >= 0.5
        np.testing.assert_allclose(
            relabeled.observed[~mask], split.observed[~mask]
        )
        np.testing.assert_allclose(
            relabeled.observed[mask],
            np.clip(curve.evaluate(wind[mask]), 0.0, 1.06).astype(np.float32),
        )
        np.testing.assert_array_equal(relabeled.features, split.features)
        with self.assertRaises(BenchmarkError):
            relabel_split(split, wind[:-1], curve, 0.0, 1.06)

    def test_reconstruction_view_is_declared_and_observable_only(self) -> None:
        _, scenario = self._prepared_scenario()
        regular = scenario.fit_view()
        self.assertNotIsInstance(regular, ReconstructionFitData)
        view = scenario.reconstruction_fit_view()
        self.assertIsInstance(view, ReconstructionFitData)
        self.assertNotIn("truth", view.__dataclass_fields__)
        self.assertNotIn("truth", view.train.__dataclass_fields__)
        self.assertEqual(
            len(view.train_wind_ms_rows), len(view.train_observed_rows)
        )
        self.assertEqual(
            len(view.train_censored_rows), len(view.train_observed_rows)
        )
        self.assertEqual(len(view.train_target_wind_ms), len(view.train))
        self.assertEqual(len(view.validation_target_wind_ms), len(view.validation))
        # Clean protocol: visible target wind equals the evaluation column.
        np.testing.assert_allclose(
            view.train_target_wind_ms, scenario.train.target_wind_ms
        )
        self.assertTrue(B2ReconAdapter.requires_reconstruction_inputs)
        self.assertFalse(B2ReconAdapter.requires_latent_truth)
        self.assertFalse(getattr(B4Adapter, "requires_reconstruction_inputs", False))
        self.assertFalse(getattr(B6Adapter, "requires_reconstruction_inputs", False))

    def test_visible_target_wind_tracks_the_contaminated_sensor(self) -> None:
        _, clean_scenario = self._prepared_scenario(contaminated=False)
        _, dirty_scenario = self._prepared_scenario(contaminated=True)
        clean_view = clean_scenario.reconstruction_fit_view()
        dirty_view = dirty_scenario.reconstruction_fit_view()
        self.assertEqual(
            len(dirty_view.train_target_wind_ms),
            len(dirty_scenario.train),
        )
        # Contaminated readings move on censored frames, so the visible
        # column must differ somewhere from the clean evaluation column.
        self.assertTrue(
            np.any(
                dirty_scenario.train.target_wind_visible_ms
                != dirty_scenario.train.target_wind_ms
            )
        )
        # The power curve itself is fit on uncensored rows, which are never
        # contaminated by construction.
        clean_mask = dirty_view.train_censored_rows < 0.5
        np.testing.assert_allclose(
            dirty_view.train_wind_ms_rows[clean_mask],
            clean_view.train_wind_ms_rows[clean_mask],
        )

    def test_adapter_refuses_the_plain_fit_view(self) -> None:
        config, scenario = self._prepared_scenario()
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(config, Path(temporary))
            adapter = B2ReconAdapter()
            with self.assertRaises(BenchmarkError):
                adapter.fit(scenario.fit_view(), context, 42)

    def test_adapter_fit_predict_and_saved_curve(self) -> None:
        config, scenario = self._prepared_scenario()
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(config, Path(temporary))
            adapter = B2ReconAdapter()
            fitted = adapter.fit(scenario.reconstruction_fit_view(), context, 42)
            self.assertEqual(
                fitted.metadata["label_source"],
                "observable_power_curve_reconstruction",
            )
            reconstruction = fitted.metadata["reconstruction"]
            self.assertGreater(reconstruction["clean_rows_used"], 0)
            self.assertGreaterEqual(
                reconstruction["train_censored_windows_replaced"], 0
            )
            forecast = adapter.predict(fitted, scenario.prediction_view(), context)
            forecast.validate(len(scenario.test), context)
            self.assertTrue(np.isfinite(forecast.quantiles).all())
            output_dir = Path(temporary) / "artifact"
            output_dir.mkdir()
            files = adapter.save_fit(fitted, output_dir)
            self.assertIn("model_state", files)
            self.assertIn("power_curve", files)
            self.assertTrue((output_dir / "power_curve.npz").is_file())
            with np.load(output_dir / "power_curve.npz") as archive:
                self.assertEqual(
                    len(archive["knot_wind_ms"]), len(archive["knot_power_pu"])
                )


if __name__ == "__main__":
    unittest.main()
