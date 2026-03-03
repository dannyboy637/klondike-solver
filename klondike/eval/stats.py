"""Statistical evaluation tools for Klondike solver experiments.

Wilson score CI, SPRT, paired comparison (McNemar's), control variate.
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class WinRateResult:
    wins: int
    total: int
    win_rate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    confidence: float


def _z_for_confidence(confidence: float) -> float:
    """Return z-score for common confidence levels."""
    z_map = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}
    return z_map.get(confidence, 1.9599640)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_ci(wins: int, total: int, confidence: float = 0.95) -> WinRateResult:
    """Compute win rate with Wilson score confidence interval.

    Wilson CI is preferred over Wald for proportions near 0 or 1,
    and for small sample sizes. It never produces negative bounds.
    """
    if total == 0:
        return WinRateResult(
            wins=0, total=0, win_rate=0.0,
            ci_lower=0.0, ci_upper=0.0, std_error=0.0, confidence=confidence,
        )

    z = _z_for_confidence(confidence)
    z2 = z * z
    n = total
    p_hat = wins / n

    # Wilson score interval
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))

    ci_lower = max(0.0, center - half_width)
    ci_upper = min(1.0, center + half_width)

    # Standard error (Wald, for reporting alongside Wilson CI)
    se = math.sqrt(p_hat * (1.0 - p_hat) / n)

    return WinRateResult(
        wins=wins, total=total, win_rate=p_hat,
        ci_lower=ci_lower, ci_upper=ci_upper,
        std_error=se, confidence=confidence,
    )


def sample_size_needed(margin: float, p: float = 0.5, confidence: float = 0.95) -> int:
    """How many games needed for a Wald CI half-width <= margin?

    Conservative estimate (Wald). Wilson would need slightly fewer.
    """
    z = _z_for_confidence(confidence)
    return math.ceil(z * z * p * (1.0 - p) / (margin * margin))


def sprt_test(
    wins: int,
    total: int,
    p0: float = 0.50,
    p1: float = 0.52,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> str:
    """Sequential Probability Ratio Test.

    Returns: "accept_h0" | "accept_h1" | "continue"

    H0: true rate = p0 (null, e.g. 50%)
    H1: true rate = p1 (alternative, e.g. 52%)

    Adaptively stops when evidence is sufficient, instead of
    requiring a fixed sample size.
    """
    if total == 0:
        return "continue"

    if p0 <= 0.0 or p0 >= 1.0 or p1 <= 0.0 or p1 >= 1.0:
        return "continue"

    losses = total - wins

    # Log-likelihood ratio
    log_lr = (
        wins * math.log(p1 / p0)
        + losses * math.log((1.0 - p1) / (1.0 - p0))
    )

    # Wald boundaries
    log_A = math.log((1.0 - beta) / alpha)   # Upper: accept H1
    log_B = math.log(beta / (1.0 - alpha))   # Lower: accept H0

    if log_lr >= log_A:
        return "accept_h1"
    elif log_lr <= log_B:
        return "accept_h0"
    else:
        return "continue"


def paired_comparison(
    results_a: list[bool],
    results_b: list[bool],
) -> dict:
    """Compare two solvers on the same deals using McNemar's test.

    results_a[i] and results_b[i] must be outcomes on the same deal i.
    Uses continuity correction (Edwards' correction).
    """
    assert len(results_a) == len(results_b), "Results must be same length (paired seeds)"
    n = len(results_a)

    both_win = sum(a and b for a, b in zip(results_a, results_b))
    a_only = sum(a and not b for a, b in zip(results_a, results_b))
    b_only = sum(not a and b for a, b in zip(results_a, results_b))
    both_lose = sum(not a and not b for a, b in zip(results_a, results_b))

    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
        chi2 = 0.0
    else:
        # McNemar's with continuity correction
        chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
        z = math.sqrt(chi2)
        p_value = 2.0 * (1.0 - _normal_cdf(z))

    return {
        "n_deals": n,
        "a_wins": sum(results_a),
        "b_wins": sum(results_b),
        "both_win": both_win,
        "a_only": a_only,
        "b_only": b_only,
        "both_lose": both_lose,
        "chi2": chi2,
        "p_value": p_value,
        "a_better": a_only > b_only,
        "diff": (sum(results_a) - sum(results_b)) / n if n > 0 else 0.0,
    }


def control_variate_estimate(
    target_results: np.ndarray,
    control_results: np.ndarray,
    control_known_rate: float,
) -> dict:
    """Variance reduction via control variate.

    Both solvers must be run on the same deals. control_known_rate
    is the known win rate of the control solver from a large prior run.
    """
    target_results = np.asarray(target_results, dtype=np.float64)
    control_results = np.asarray(control_results, dtype=np.float64)

    cov_matrix = np.cov(target_results, control_results)
    cov = cov_matrix[0, 1]
    var_control = np.var(control_results, ddof=1)
    beta = cov / var_control if var_control > 0 else 0.0

    raw_rate = np.mean(target_results)
    control_sim_rate = np.mean(control_results)
    cv_rate = raw_rate - beta * (control_sim_rate - control_known_rate)

    residuals = target_results - beta * (control_results - control_known_rate)
    n = len(target_results)
    se_cv = float(np.std(residuals, ddof=1) / np.sqrt(n))
    se_raw = float(np.std(target_results, ddof=1) / np.sqrt(n))

    return {
        "rate_cv": float(cv_rate),
        "rate_raw": float(raw_rate),
        "se_cv": se_cv,
        "se_raw": se_raw,
        "variance_reduction": (se_raw / se_cv) ** 2 if se_cv > 0 else 0.0,
        "beta": float(beta),
    }
