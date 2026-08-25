"""Tests for the restored Hill of Towie and release-protocol assets."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(
        name, PROJECT_DIR / "scripts" / filename
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class FastLogConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.converter = load_script("hot_converter", "20_hot_fastlog_to_1min.py")

    def test_historical_locf_contract_selftest(self) -> None:
        self.converter.selftest()

    def test_future_observation_is_not_used(self) -> None:
        source = pd.Series(
            [5.0, 9.0],
            index=pd.to_datetime(
                ["2026-01-01 00:00:00.500", "2026-01-01 00:00:01.500"]
            ),
        )
        sampled, _ = self.converter.causal_second_grid(
            source, pd.Timestamp("2026-01-01"), previous_value=2.0
        )
        self.assertEqual(sampled.iloc[0], 2.0)
        self.assertEqual(sampled.iloc[1], 5.0)
        self.assertEqual(sampled.iloc[2], 9.0)


class TruthChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.truth = load_script("hot_truth", "21_hot_truth_channel.py")

    def test_routing_rule(self) -> None:
        n = 200
        frame = pd.DataFrame(
            {
                "power_obs_kW": np.r_[np.full(80, 800.0), np.full(80, 600.0), np.full(40, 20.0)],
                "wind_ms": np.r_[
                    np.full(40, 7.1),
                    np.full(40, 7.6),
                    np.full(40, 7.1),
                    np.full(40, 7.6),
                    np.full(40, 8.0),
                ],
                "pref_kW": np.r_[np.full(80, 2300.0), np.full(80, 1500.0), np.full(40, 2300.0)],
                "power_red": np.zeros(n),
            },
            index=pd.date_range("2026-01-01", periods=n, freq="1min"),
        )
        output, audit = self.truth.build_truth_channel(frame)
        self.assertEqual(audit["pap_source_counts"]["actual"], 80)
        self.assertEqual(audit["pap_source_counts"]["power_curve"], 80)
        self.assertEqual(audit["pap_source_counts"]["none"], 40)
        self.assertEqual(int(output["curtailed"].sum()), 80)
        self.assertEqual(audit["neighbor_route_coverage"], 0)


class ReconstructedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_script(
            "release_protocol", "63_release_event_validation.py"
        )

    def test_reconstructed_protocol_verifies_legacy_event_artifact(self) -> None:
        event_path = (
            PROJECT_DIR
            / "reference_results"
            / "altahullion_audit"
            / "qualified_release_events.csv"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            first = self.protocol.freeze_or_verify_protocol(
                event_csv=event_path, output_dir=output_dir
            )
            second = self.protocol.freeze_or_verify_protocol(
                event_csv=event_path, output_dir=output_dir
            )
        self.assertEqual(first, second)
        self.assertEqual(first["event_count_frozen"], 122)
        self.assertEqual(
            first["event_list_sha256"], self.protocol.FROZEN_EVENT_SHA256
        )
        self.assertEqual(
            first["event_list_canonical_sha256"],
            self.protocol.FROZEN_EVENT_CANONICAL_SHA256,
        )
        self.assertEqual(
            first["legacy_v1_2_protocol_sha256_unrecovered"],
            self.protocol.LEGACY_PROTOCOL_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
