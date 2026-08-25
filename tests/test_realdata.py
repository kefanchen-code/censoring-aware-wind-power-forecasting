"""Unit tests for the real-data release-event view (no truth anywhere)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from censored_wind_power.benchmark.realdata import (
    assign_segments,
    build_real_windows,
    event_windows,
    usable_blocks,
)


def _synthetic_real_frame(block_minutes: int = 100, n_blocks: int = 6) -> pd.DataFrame:
    """Natural-clock record: usable U/R blocks separated by unusable rows."""

    pieces = []
    cursor = pd.Timestamp("2026-01-01")
    for block in range(n_blocks):
        pieces.append(
            pd.date_range(cursor, periods=block_minutes, freq="1min")
        )
        cursor = pieces[-1][-1] + pd.Timedelta(minutes=1)
        if block < n_blocks - 1:
            pieces.append(pd.date_range(cursor, periods=5, freq="1min"))
            cursor = pieces[-1][-1] + pd.Timedelta(minutes=1)
    index = pieces[0]
    for piece in pieces[1:]:
        index = index.append(piece)
    n = len(index)
    rng = np.random.default_rng(4)
    usable = np.ones(n, dtype=bool)
    for block in range(n_blocks - 1):
        gap_start = (block + 1) * block_minutes + block * 5
        usable[gap_start : gap_start + 5] = False
    censored = np.zeros(n, dtype=np.float32)
    censored[20:60] = 1.0
    censored[block_minutes + 5 + 20 : block_minutes + 5 + 60] = 1.0
    labels = np.where(censored > 0.5, "R", "U")
    labels[~usable] = "X"
    observed = rng.random(n).astype(np.float32)
    observed[~usable] = np.nan
    frame = pd.DataFrame(
        {
            "observed_pu": observed,
            "cap_pu": np.where(censored > 0.5, 0.6, 1.05).astype(np.float32),
            "censored": censored,
            "wind_ms": (6.0 + rng.random(n)).astype(np.float32),
            "label": labels,
            "usable": usable,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


class RealWindowTests(unittest.TestCase):
    def test_assignment_is_chronological_and_complete(self) -> None:
        blocks = usable_blocks(_synthetic_real_frame())
        assignment = assign_segments(blocks, 0.70, 0.15)
        self.assertEqual(set(assignment), {0, 1, 2, 3, 4, 5})
        self.assertEqual(
            assignment, {0: "train", 1: "train", 2: "train", 3: "train",
                         4: "validation", 5: "test"}
        )

    def test_windows_never_cross_segment_or_split(self) -> None:
        blocks = usable_blocks(_synthetic_real_frame())
        assignment = assign_segments(blocks, 0.70, 0.15)
        windows, standardization = build_real_windows(
            blocks, assignment, horizon_minutes=2, window_minutes=10
        )
        self.assertIn("wind_mean_ms", standardization)
        for split_name in ("train", "validation", "test"):
            split = windows.get(split_name)
            if split is None:
                continue
            allowed = {
                segment
                for segment, label in assignment.items()
                if label == split_name
            }
            self.assertTrue(set(np.unique(split.segment_id)).issubset(allowed))
        # Gaps forbid any window straddling a dropped block, so each split
        # contains windows from exactly its own segments.
        self.assertTrue(set(np.unique(windows["train"].segment_id)) == {0, 1, 2, 3})
        self.assertTrue(set(np.unique(windows["validation"].segment_id)) == {4})
        self.assertTrue(set(np.unique(windows["test"].segment_id)) == {5})

    def test_standardization_uses_training_side_only(self) -> None:
        blocks = usable_blocks(_synthetic_real_frame())
        assignment = assign_segments(blocks, 0.70, 0.15)
        _, standardization = build_real_windows(
            blocks, assignment, horizon_minutes=2, window_minutes=10
        )
        train_wind = blocks.loc[
            blocks["segment_id"].isin(
                [s for s, label in assignment.items() if label == "train"]
            ),
            "wind_ms",
        ]
        self.assertAlmostEqual(
            standardization["wind_mean_ms"], float(train_wind.mean()), places=5
        )

    def test_event_windows_reject_gapped_history(self) -> None:
        frame = _synthetic_real_frame()
        good_time = frame.index[50]            # deep inside the first block
        after_gap_time = frame.index[108]      # history reaches over the gap
        features, persistence, valid = event_windows(
            frame,
            pd.Series([good_time, after_gap_time, pd.Timestamp("2030-01-01")]),
            window_minutes=10,
        )
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertFalse(valid[2])
        self.assertEqual(features.shape, (3, 4, 10))
        self.assertEqual(persistence[0], frame["observed_pu"].iloc[49])


if __name__ == "__main__":
    unittest.main()
