"""Tests for the exact sign-flip and TOST inference upgrade (plan A1)."""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from censored_wind_power.benchmark.scoring import (
    exact_sign_flip_test,
    exact_tost,
)


def _brute_force_two_sided_p(difference, segment_id):
    """Independent reference implementation via itertools enumeration."""

    difference = np.asarray(difference, dtype=np.float64)
    segment_id = np.asarray(segment_id)
    segments = np.unique(segment_id)
    cluster_sums = [difference[segment_id == value].sum() for value in segments]
    observed = difference.mean()
    total = 0
    exceedances = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(segments)):
        total += 1
        statistic = float(np.dot(signs, cluster_sums)) / len(difference)
        exceedances += int(abs(statistic) >= abs(observed) - 1e-15)
    return exceedances / total


class TestExactSignFlip(unittest.TestCase):
    def test_hand_computed_two_clusters(self):
        # Cluster sums 3.0 and 0.5 over 3 windows; patterns give statistics
        # +/-1.1667 and +/-0.8333, so the two-sided exact p is 2/4 = 0.5.
        difference = np.array([1.0, 2.0, 0.5])
        segment_id = np.array([0, 0, 1])
        result = exact_sign_flip_test(difference, segment_id)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["total_patterns"], 4)
        self.assertAlmostEqual(result["observed_difference"], 3.5 / 3.0)
        self.assertAlmostEqual(result["p_two_sided"], 0.5)

    def test_zero_difference_gives_p_one(self):
        difference = np.zeros(12)
        segment_id = np.repeat(np.arange(3), 4)
        result = exact_sign_flip_test(difference, segment_id)
        self.assertAlmostEqual(result["p_two_sided"], 1.0)

    def test_matches_brute_force(self):
        rng = np.random.default_rng(7)
        for trial in range(5):
            k = int(rng.integers(2, 6))
            sizes = rng.integers(1, 6, size=k)
            difference = rng.normal(size=int(sizes.sum()))
            segment_id = np.repeat(np.arange(k), sizes)
            result = exact_sign_flip_test(difference, segment_id)
            expected = _brute_force_two_sided_p(difference, segment_id)
            self.assertAlmostEqual(
                result["p_two_sided"],
                expected,
                places=12,
                msg="trial %d" % trial,
            )

    def test_rejects_strong_signal(self):
        # Every cluster contributes the same positive total: only the all-plus
        # and all-minus patterns are as extreme as observed.
        difference = np.repeat(np.array([2.0, 1.5, 3.0, 2.5]), 10)
        segment_id = np.repeat(np.arange(4), 10)
        result = exact_sign_flip_test(difference, segment_id)
        self.assertAlmostEqual(result["p_two_sided"], 2.0 / 16.0)

    def test_misaligned_inputs_raise(self):
        with self.assertRaises(Exception):
            exact_sign_flip_test(np.ones(5), np.ones(4))


class TestExactTost(unittest.TestCase):
    def test_zero_difference_is_equivalent(self):
        rng = np.random.default_rng(11)
        difference = rng.normal(scale=0.05, size=200)
        difference -= difference.mean()  # exactly zero mean effect
        segment_id = np.repeat(np.arange(20), 10)
        result = exact_tost(difference, segment_id, margin=0.5)
        self.assertTrue(result["equivalent_permutation"])
        self.assertTrue(result["equivalent_t"])
        self.assertTrue(result["equivalent_ci_containment"])

    def test_large_offset_is_not_equivalent(self):
        difference = np.full(200, 2.0) + np.repeat(
            np.linspace(-0.01, 0.01, 20), 10
        )
        segment_id = np.repeat(np.arange(20), 10)
        result = exact_tost(difference, segment_id, margin=1.0)
        self.assertFalse(result["equivalent_permutation"])
        self.assertFalse(result["equivalent_t"])
        # Lower boundary (delta <= -1) is decisively rejected.
        self.assertLess(result["p_lower_boundary_permutation"], 0.001)
        # Upper boundary (delta >= +1) is not rejected.
        self.assertGreater(result["p_upper_boundary_permutation"], 0.3)

    def test_small_offset_inside_margin_is_equivalent(self):
        rng = np.random.default_rng(13)
        cluster_effects = rng.normal(loc=0.05, scale=0.02, size=20)
        difference = np.repeat(cluster_effects, 10) + rng.normal(
            scale=0.01, size=200
        )
        segment_id = np.repeat(np.arange(20), 10)
        result = exact_tost(difference, segment_id, margin=0.5)
        self.assertTrue(result["equivalent_permutation"])
        self.assertTrue(result["equivalent_t"])

    def test_permutation_and_t_verdicts_agree_on_sign(self):
        # Negative offset well inside a wide margin: both boundaries rejected.
        difference = np.full(150, -0.1)
        segment_id = np.repeat(np.arange(15), 10)
        result = exact_tost(difference, segment_id, margin=0.5)
        self.assertTrue(result["equivalent_permutation"])
        self.assertTrue(result["equivalent_t"])

    def test_margin_must_be_positive(self):
        difference = np.ones(10)
        segment_id = np.arange(10) % 2
        with self.assertRaises(Exception):
            exact_tost(difference, segment_id, margin=0.0)


if __name__ == "__main__":
    unittest.main()
