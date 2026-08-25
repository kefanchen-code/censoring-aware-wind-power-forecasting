"""Model-independent scoring and segment-cluster comparisons."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from .core import BenchmarkContext, Forecast, ForecastValidationError


@dataclass(frozen=True)
class ScoreResult:
    """Per-window scores and JSON-ready aggregate metrics."""

    wis: np.ndarray
    crps: np.ndarray
    pit: np.ndarray
    summary: Mapping[str, Any]


def pmf_to_quantiles(
    probabilities: np.ndarray,
    centers: np.ndarray,
    quantile_levels: np.ndarray,
) -> np.ndarray:
    """Extract left-continuous quantiles from a discrete PMF."""

    cumulative = np.cumsum(probabilities, axis=1)
    output = np.empty((len(probabilities), len(quantile_levels)), dtype=np.float32)
    for column, level in enumerate(quantile_levels):
        indices = np.sum(cumulative < level, axis=1)
        output[:, column] = centers[np.clip(indices, 0, len(centers) - 1)]
    return output


def crps_from_pmf(
    probabilities: np.ndarray,
    targets: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Calculate exact CRPS for the configured finite discrete support."""

    term_one = np.sum(
        probabilities * np.abs(centers[None, :] - targets[:, None]), axis=1
    )
    weighted_support = probabilities * centers[None, :]
    cumulative_probability_before = np.cumsum(probabilities, axis=1) - probabilities
    cumulative_weighted_before = np.cumsum(weighted_support, axis=1) - weighted_support
    # This is 0.5 E|X-X'|, evaluated in O(n_bins) rather than O(n_bins^2).
    term_two = np.sum(
        probabilities
        * (
            centers[None, :] * cumulative_probability_before
            - cumulative_weighted_before
        ),
        axis=1,
    )
    return term_one - term_two


def crps_from_quantiles(
    quantiles: np.ndarray,
    targets: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """Approximate CRPS from a finite quantile grid.

    This score is labeled approximate and must not be silently mixed with PMF
    exact CRPS in a confirmatory comparison.
    """

    errors = targets[:, None] - quantiles
    pinball = np.maximum(
        levels[None, :] * errors, (levels[None, :] - 1.0) * errors
    )
    return 2.0 * np.trapz(pinball, levels, axis=1)


def wis_per_sample(
    quantiles: np.ndarray,
    targets: np.ndarray,
    levels: np.ndarray,
    interval_alphas: Sequence[float],
) -> np.ndarray:
    """Calculate WIS from exact, predeclared symmetric quantile pairs."""

    median_index = int(np.flatnonzero(np.isclose(levels, 0.5))[0])
    total = 0.5 * np.abs(targets - quantiles[:, median_index])
    for alpha in interval_alphas:
        lower_level = float(alpha) / 2.0
        upper_level = 1.0 - lower_level
        lower_index = int(np.flatnonzero(np.isclose(levels, lower_level))[0])
        upper_index = int(np.flatnonzero(np.isclose(levels, upper_level))[0])
        lower = quantiles[:, lower_index]
        upper = quantiles[:, upper_index]
        interval_score = (
            upper
            - lower
            + (2.0 / alpha) * (lower - targets) * (targets < lower)
            + (2.0 / alpha) * (targets - upper) * (targets > upper)
        )
        total += (alpha / 2.0) * interval_score
    return total / (len(interval_alphas) + 0.5)


def wis_components_per_sample(
    quantiles: np.ndarray,
    targets: np.ndarray,
    levels: np.ndarray,
    interval_alphas: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Decompose WIS into dispersion, overprediction, and underprediction.

    Follows Bracher et al. (2021). Because each interval penalty carries the
    weight ``(alpha / 2) * (2 / alpha) == 1``, the penalty terms reduce to plain
    positive parts, and the three components sum exactly to ``wis_per_sample``.

    Overprediction accumulates mass placed above the target (target below the
    lower quantile or the median); underprediction accumulates mass placed
    below it.
    """

    median_index = int(np.flatnonzero(np.isclose(levels, 0.5))[0])
    median = quantiles[:, median_index]
    dispersion = np.zeros(len(targets), dtype=np.float64)
    over = 0.5 * np.maximum(median - targets, 0.0)
    under = 0.5 * np.maximum(targets - median, 0.0)
    for alpha in interval_alphas:
        lower_level = float(alpha) / 2.0
        upper_level = 1.0 - lower_level
        lower_index = int(np.flatnonzero(np.isclose(levels, lower_level))[0])
        upper_index = int(np.flatnonzero(np.isclose(levels, upper_level))[0])
        lower = quantiles[:, lower_index]
        upper = quantiles[:, upper_index]
        dispersion += (float(alpha) / 2.0) * (upper - lower)
        over += np.maximum(lower - targets, 0.0)
        under += np.maximum(targets - upper, 0.0)
    weight = len(interval_alphas) + 0.5
    return {
        "dispersion": dispersion / weight,
        "overprediction": over / weight,
        "underprediction": under / weight,
    }


def _mid_pit_pmf(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(bin_edges[1:-1], targets, side="left")
    indices = np.clip(indices, 0, probabilities.shape[1] - 1)
    cumulative = np.cumsum(probabilities, axis=1)
    previous = np.zeros(len(targets), dtype=np.float64)
    positive = indices > 0
    previous[positive] = cumulative[
        np.flatnonzero(positive), indices[positive] - 1
    ]
    mass = probabilities[np.arange(len(targets)), indices]
    return previous + 0.5 * mass


def _pit_quantiles(
    quantiles: np.ndarray,
    targets: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(targets), dtype=np.float64)
    for index, target in enumerate(targets):
        values, unique_indices = np.unique(quantiles[index], return_index=True)
        unique_levels = levels[unique_indices]
        output[index] = np.interp(
            target, values, unique_levels, left=0.0, right=1.0
        )
    return output


def score_forecast(
    forecast: Forecast,
    targets: np.ndarray,
    context: BenchmarkContext,
) -> ScoreResult:
    """Score one forecast without exposing truth to the model adapter."""

    forecast.validate(len(targets), context)
    evaluation = context.config["evaluation"]
    alphas = [float(value) for value in evaluation["interval_alphas"]]
    wis = wis_per_sample(
        forecast.quantiles, targets, context.quantile_levels, alphas
    )
    if forecast.pmf is not None:
        crps = crps_from_pmf(forecast.pmf, targets, context.bin_centers)
        pit = _mid_pit_pmf(forecast.pmf, targets, context.bin_edges)
        crps_kind = "exact_discrete_pmf"
    else:
        crps = crps_from_quantiles(
            forecast.quantiles, targets, context.quantile_levels
        )
        pit = _pit_quantiles(
            forecast.quantiles, targets, context.quantile_levels
        )
        crps_kind = "finite_quantile_approximation"

    median_index = int(
        np.flatnonzero(np.isclose(context.quantile_levels, 0.5))[0]
    )
    coverage: Dict[str, float] = {}
    interval_width: Dict[str, float] = {}
    coverage_error: Dict[str, float] = {}
    for alpha in alphas:
        lower_index = int(
            np.flatnonzero(np.isclose(context.quantile_levels, alpha / 2.0))[0]
        )
        upper_index = int(
            np.flatnonzero(np.isclose(context.quantile_levels, 1.0 - alpha / 2.0))[0]
        )
        covered = (
            (targets >= forecast.quantiles[:, lower_index])
            & (targets <= forecast.quantiles[:, upper_index])
        )
        nominal = int(round(100 * (1.0 - alpha)))
        key = str(nominal)
        empirical_coverage = float(covered.mean())
        coverage[key] = empirical_coverage
        coverage_error[key] = empirical_coverage - (1.0 - alpha)
        interval_width[key] = float(
            np.mean(
                forecast.quantiles[:, upper_index]
                - forecast.quantiles[:, lower_index]
            )
        )
    summary = {
        "n_samples": int(len(targets)),
        "wis_mean": float(wis.mean()),
        "crps_mean": float(crps.mean()),
        "crps_kind": crps_kind,
        "median_mae": float(
            np.mean(np.abs(targets - forecast.quantiles[:, median_index]))
        ),
        "median_bias": float(
            np.mean(forecast.quantiles[:, median_index] - targets)
        ),
        "coverage": coverage,
        "coverage_error": coverage_error,
        "interval_width_mean": interval_width,
        "pit_mean": float(pit.mean()),
        "pit_std": float(pit.std()),
    }
    return ScoreResult(wis=wis, crps=crps, pit=pit, summary=summary)


def underestimation_rate(
    quantiles: np.ndarray,
    targets: np.ndarray,
    levels: np.ndarray,
    level: float,
    mask: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Share of selected windows whose declared quantile lies below truth."""

    index = int(np.flatnonzero(np.isclose(levels, level))[0])
    selected = (
        np.ones(len(targets), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if not selected.any():
        return None
    return float(
        np.mean(quantiles[selected, index] < targets[selected])
    )


def _stratum_metrics(
    quantiles: np.ndarray,
    targets: np.ndarray,
    wis: np.ndarray,
    crps: np.ndarray,
    pit: np.ndarray,
    levels: np.ndarray,
    interval_alphas: Sequence[float],
    mask: np.ndarray,
) -> Dict[str, Any]:
    median_index = int(np.flatnonzero(np.isclose(levels, 0.5))[0])
    selected_quantiles = quantiles[mask]
    selected_targets = targets[mask]
    row: Dict[str, Any] = {
        "n_windows": int(mask.sum()),
        "wis_mean": float(wis[mask].mean()),
        "crps_mean": float(crps[mask].mean()),
        "median_bias": float(
            np.mean(selected_quantiles[:, median_index] - selected_targets)
        ),
        "nmae": float(
            np.mean(np.abs(selected_quantiles[:, median_index] - selected_targets))
        ),
        "pit_mean": float(pit[mask].mean()),
        "pit_std": float(pit[mask].std()),
        "underestimation_q50": underestimation_rate(
            quantiles, targets, levels, 0.5, mask
        ),
        "underestimation_q90": underestimation_rate(
            quantiles, targets, levels, 0.9, mask
        ),
    }
    for alpha in interval_alphas:
        lower_index = int(
            np.flatnonzero(np.isclose(levels, float(alpha) / 2.0))[0]
        )
        upper_index = int(
            np.flatnonzero(np.isclose(levels, 1.0 - float(alpha) / 2.0))[0]
        )
        lower = selected_quantiles[:, lower_index]
        upper = selected_quantiles[:, upper_index]
        nominal = int(round(100 * (1.0 - float(alpha))))
        covered = (selected_targets >= lower) & (selected_targets <= upper)
        row["coverage_%d" % nominal] = float(covered.mean())
        row["width_%d" % nominal] = float(np.mean(upper - lower))
    return row


def stratified_rows(
    quantiles: np.ndarray,
    targets: np.ndarray,
    wis: np.ndarray,
    crps: np.ndarray,
    pit: np.ndarray,
    levels: np.ndarray,
    interval_alphas: Sequence[float],
    target_censored: np.ndarray,
    history_censor_frac: np.ndarray,
    target_cap_pu: np.ndarray,
    target_wind_ms: np.ndarray,
) -> List[Dict[str, Any]]:
    """Predeclared evaluation strata for one saved forecast run.

    Dimensions: overall, target censoring state, history-window censoring
    fraction {0, (0, 0.3], (0.3, 1]}, per-scenario cap tertiles over the test
    windows, and wind bands {<7, 7-11, >11 m/s}. Empty strata are skipped.
    """

    censored_mask = np.asarray(target_censored, dtype=float) >= 0.5
    frac = np.asarray(history_censor_frac, dtype=float)
    cap = np.asarray(target_cap_pu, dtype=float)
    wind = np.asarray(target_wind_ms, dtype=float)
    cap_low, cap_high = np.quantile(cap, [1.0 / 3.0, 2.0 / 3.0])
    strata: List[Tuple[str, str, np.ndarray]] = [
        ("all", "all", np.ones(len(targets), dtype=bool)),
        ("target_censoring", "uncensored", ~censored_mask),
        ("target_censoring", "censored", censored_mask),
        ("history_censor_frac", "0", frac <= 0.0),
        ("history_censor_frac", "(0,0.3]", (frac > 0.0) & (frac <= 0.3)),
        ("history_censor_frac", "(0.3,1]", frac > 0.3),
        ("target_cap_tertile", "low", cap <= cap_low),
        ("target_cap_tertile", "mid", (cap > cap_low) & (cap <= cap_high)),
        ("target_cap_tertile", "high", cap > cap_high),
        ("target_wind_band", "<7", wind < 7.0),
        ("target_wind_band", "7-11", (wind >= 7.0) & (wind <= 11.0)),
        ("target_wind_band", ">11", wind > 11.0),
    ]
    rows: List[Dict[str, Any]] = []
    for dimension, stratum, mask in strata:
        if not mask.any():
            continue
        rows.append(
            {
                "dimension": dimension,
                "stratum": stratum,
                **_stratum_metrics(
                    quantiles,
                    targets,
                    wis,
                    crps,
                    pit,
                    levels,
                    interval_alphas,
                    mask,
                ),
            }
        )
    return rows


def cluster_inference(
    reference_loss: np.ndarray,
    challenger_loss: np.ndarray,
    segment_id: np.ndarray,
    seed: int,
    bootstrap_draws: int,
    sign_flip_draws: int,
    exhaustive_max_clusters: int,
) -> Dict[str, Any]:
    """Compare aligned per-window losses using segment-level dependence."""

    if not (
        len(reference_loss) == len(challenger_loss) == len(segment_id)
    ):
        raise ForecastValidationError("pairwise losses and segment_id are not aligned")
    difference = np.asarray(reference_loss) - np.asarray(challenger_loss)
    segments = np.unique(segment_id)
    indices = [np.flatnonzero(segment_id == value) for value in segments]
    observed = float(difference.mean())
    rng = np.random.default_rng(seed)

    bootstrap = np.empty(bootstrap_draws, dtype=np.float64)
    for draw in range(bootstrap_draws):
        selected = rng.integers(0, len(segments), len(segments))
        sampled = np.concatenate([indices[index] for index in selected])
        bootstrap[draw] = difference[sampled].mean()
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])

    cluster_sums = np.asarray([difference[index].sum() for index in indices])
    if len(segments) <= exhaustive_max_clusters:
        exceedances = 0
        total_draws = 2 ** len(segments)
        for signs in itertools.product((-1.0, 1.0), repeat=len(segments)):
            statistic = np.dot(np.asarray(signs), cluster_sums) / len(difference)
            exceedances += int(abs(statistic) >= abs(observed) - 1e-15)
        p_value = float(exceedances / total_draws)
        sign_flip_type = "exhaustive_sensitivity"
    else:
        exceedances = 0
        remaining = sign_flip_draws
        while remaining:
            batch = min(10000, remaining)
            signs = rng.choice((-1.0, 1.0), size=(batch, len(segments)))
            statistics = signs.dot(cluster_sums) / len(difference)
            exceedances += int(np.sum(np.abs(statistics) >= abs(observed) - 1e-15))
            remaining -= batch
        total_draws = sign_flip_draws
        p_value = float((exceedances + 1) / (total_draws + 1))
        sign_flip_type = "monte_carlo_sensitivity"

    leave_one_out = []
    for index in indices:
        keep = np.ones(len(difference), dtype=bool)
        keep[index] = False
        leave_one_out.append(float(difference[keep].mean()))
    return {
        "estimand": "observation-weighted mean paired loss difference",
        "direction": "positive means challenger has lower loss",
        "difference_reference_minus_challenger": observed,
        "cluster_bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "n_clusters": int(len(segments)),
        "sign_flip_p_two_sided": p_value,
        "sign_flip_type": sign_flip_type,
        "sign_flip_draws": int(total_draws),
        "sign_flip_assumption": "independent sign-symmetric cluster contributions under the null",
        "leave_one_cluster_out_range": [
            float(min(leave_one_out)),
            float(max(leave_one_out)),
        ],
    }


def _all_sign_pattern_statistics(cluster_sums: np.ndarray, n_windows: int) -> np.ndarray:
    """Enumerate the full sign-flip null distribution of the test statistic.

    With K independent clusters there are 2**K equally likely sign patterns;
    for K<=22 the (2**K, K) sign matrix fits comfortably in memory and the
    matrix product completes in well under a second.
    """

    cluster_sums = np.asarray(cluster_sums, dtype=np.float64)
    k = len(cluster_sums)
    if k > 22:
        raise ForecastValidationError(
            "exhaustive enumeration is only supported for at most 22 clusters"
        )
    indices = np.arange(2 ** k, dtype=np.uint32)
    shifts = np.arange(k, dtype=np.uint32)
    signs = np.where(
        ((indices[:, None] >> shifts[None, :]) & np.uint32(1)) == 0,
        np.float64(1.0),
        np.float64(-1.0),
    )
    return signs @ cluster_sums / float(n_windows)


def exact_sign_flip_test(
    difference: np.ndarray,
    segment_id: np.ndarray,
) -> Dict[str, Any]:
    """Exact cluster-level sign-flip test over all 2**K sign patterns.

    ``difference`` holds aligned per-window paired losses (reference minus
    challenger).  Under the null of no systematic difference, each cluster's
    total contribution is sign-symmetric, so the exact two-sided p-value is the
    share of the 2**K sign-pattern statistics at least as extreme as observed.
    """

    difference = np.asarray(difference, dtype=np.float64)
    segment_id = np.asarray(segment_id)
    if len(difference) != len(segment_id):
        raise ForecastValidationError("difference and segment_id are not aligned")
    segments = np.unique(segment_id)
    cluster_sums = np.asarray(
        [difference[segment_id == value].sum() for value in segments]
    )
    observed = float(difference.mean())
    statistics = _all_sign_pattern_statistics(cluster_sums, len(difference))
    total = statistics.size
    exceedances = int(np.sum(np.abs(statistics) >= abs(observed) - 1e-15))
    # Count of strictly more extreme patterns (observed pattern excluded).
    strictly_more_extreme = max(int(np.sum(np.abs(statistics) > abs(observed) + 1e-15)), 0)
    return {
        "n_clusters": int(len(segments)),
        "total_patterns": int(total),
        "observed_difference": observed,
        "p_two_sided": float(exceedances / total),
        "p_two_sided_min": float((strictly_more_extreme + 1) / (total + 1)),
        "p_resolution": float(1.0 / total),
        "quantile_0025": float(np.quantile(statistics, 0.025)),
        "quantile_975": float(np.quantile(statistics, 0.975)),
    }


def exact_tost(
    difference: np.ndarray,
    segment_id: np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Exact permutation TOST for cluster-correlated paired losses.

    For the upper boundary H0: delta >= margin, the window-level differences
    shifted by ``-margin`` are sign-symmetric under the boundary null, so the
    one-sided p-value is the exact share of sign-pattern statistics of the
    shifted cluster sums that reach the observed shifted statistic; the lower
    boundary is symmetric.  Equivalence is declared when both one-sided tests
    reject at level ``alpha``.  The conventional t-TOST on cluster means and
    the conservative CI-containment verdict are reported alongside.
    """

    difference = np.asarray(difference, dtype=np.float64)
    segment_id = np.asarray(segment_id)
    if len(difference) != len(segment_id):
        raise ForecastValidationError("difference and segment_id are not aligned")
    margin = float(margin)
    if margin <= 0:
        raise ForecastValidationError("equivalence margin must be positive")
    segments = np.unique(segment_id)
    k = int(len(segments))
    observed = float(difference.mean())
    n_windows = int(len(difference))

    def one_sided(shift: float, upper: bool) -> float:
        shifted = difference - shift
        cluster_sums = np.asarray(
            [shifted[segment_id == value].sum() for value in segments]
        )
        statistics = _all_sign_pattern_statistics(cluster_sums, n_windows)
        observed_statistic = float(shifted.mean())
        if upper:
            exceedances = int(np.sum(statistics >= observed_statistic - 1e-15))
        else:
            exceedances = int(np.sum(statistics <= observed_statistic + 1e-15))
        return float(exceedances / statistics.size)

    # Lower boundary H0: delta <= -margin rejects on a LARGE shifted statistic
    # (upper tail); upper boundary H0: delta >= +margin rejects on a SMALL one
    # (lower tail).
    p_lower = one_sided(-margin, upper=True)
    p_upper = one_sided(+margin, upper=False)

    cluster_means = np.asarray(
        [difference[segment_id == value].mean() for value in segments]
    )
    sem = float(cluster_means.std(ddof=1) / np.sqrt(k))
    t_critical = float(scipy_stats.t.ppf(1.0 - alpha, df=k - 1))
    t_lower = (observed - (-margin)) / sem if sem > 0 else np.inf
    t_upper = (observed - margin) / sem if sem > 0 else -np.inf
    p_t_lower = float(scipy_stats.t.sf(t_lower, df=k - 1)) if sem > 0 else 0.0
    p_t_upper = float(scipy_stats.t.cdf(t_upper, df=k - 1)) if sem > 0 else 0.0
    ci_low = observed - t_critical * sem
    ci_high = observed + t_critical * sem
    return {
        "margin": margin,
        "alpha": float(alpha),
        "observed_difference": observed,
        "n_clusters": k,
        "p_lower_boundary_permutation": p_lower,
        "p_upper_boundary_permutation": p_upper,
        "p_max_permutation": max(p_lower, p_upper),
        "equivalent_permutation": bool(p_lower < alpha and p_upper < alpha),
        "p_lower_boundary_t": p_t_lower,
        "p_upper_boundary_t": p_t_upper,
        "p_max_t": max(p_t_lower, p_t_upper),
        "equivalent_t": bool(p_t_lower < alpha and p_t_upper < alpha),
        "cluster_mean_ci_95": [float(ci_low), float(ci_high)],
        "equivalent_ci_containment": bool(-margin < ci_low and ci_high < margin),
    }
