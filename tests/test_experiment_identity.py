"""Regression tests for machine-independent experiment fingerprints."""

import json
import tempfile
import unittest
from pathlib import Path

from censored_wind_power.benchmark.runner import _experiment_id


def _write_manifest(directory, *, created_utc, data_path, data_sha="abc123"):
    manifest = {
        "created_utc": created_utc,
        "data_files": {
            "S1": {"path": data_path, "sha256": data_sha},
        },
        "segment_signature": "segments-v1",
        "segment_assignment": {"1": "train", "2": "test"},
    }
    directory.mkdir()
    (directory / "protocol_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class ExperimentIdentityTests(unittest.TestCase):
    def test_ignores_timestamp_and_absolute_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            _write_manifest(
                first,
                created_utc="2026-01-01T00:00:00+00:00",
                data_path="C:/private/workstation/input.parquet",
            )
            _write_manifest(
                second,
                created_utc="2026-09-10T00:00:00+00:00",
                data_path="/srv/public/input.parquet",
            )
            config = {"comparisons": [{"reference": "B4", "challenger": "B6"}]}

            self.assertEqual(
                _experiment_id("protocol-v1", first, config),
                _experiment_id("protocol-v1", second, config),
            )

    def test_changes_with_scientific_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            _write_manifest(first, created_utc="same", data_path="same", data_sha="sha-a")
            _write_manifest(second, created_utc="same", data_path="same", data_sha="sha-b")
            config = {"comparisons": [{"reference": "B4", "challenger": "B6"}]}

            self.assertNotEqual(
                _experiment_id("protocol-v1", first, config),
                _experiment_id("protocol-v1", second, config),
            )


if __name__ == "__main__":
    unittest.main()
