// Statistical functions: Wilson CI, SPRT, McNemar test.
// Wilson CI implemented in Phase R5 (needed by pipeline).
// SPRT and McNemar added in Phase R6.

/// Wilson score confidence interval for binomial proportions.
///
/// Returns `(win_rate, ci_lower, ci_upper)` at the given confidence level.
/// Common confidence levels: 0.90, 0.95, 0.99.
pub fn wilson_ci(wins: usize, total: usize, confidence: f64) -> (f64, f64, f64) {
    if total == 0 {
        return (0.0, 0.0, 0.0);
    }
    let n = total as f64;
    let p_hat = wins as f64 / n;

    // z-score for common confidence levels
    let z = if (confidence - 0.99).abs() < 0.001 {
        2.576
    } else if (confidence - 0.95).abs() < 0.001 {
        1.96
    } else if (confidence - 0.90).abs() < 0.001 {
        1.645
    } else {
        1.96 // Default to 95%
    };

    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denominator;
    let margin = z * ((p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (p_hat, lower, upper)
}

/// SPRT decision result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SprtResult {
    /// Accept null hypothesis (win rate <= p0).
    AcceptH0,
    /// Accept alternative hypothesis (win rate >= p1).
    AcceptH1,
    /// Not enough evidence yet; continue sampling.
    Continue,
}

/// Sequential Probability Ratio Test.
///
/// Tests H0: win_rate = p0 vs H1: win_rate = p1 with error rates alpha (Type I) and beta (Type II).
/// Returns `(result, log_likelihood_ratio)`.
///
/// - `wins`: number of wins observed
/// - `total`: total games played
/// - `p0`: null hypothesis win rate (e.g. 0.50)
/// - `p1`: alternative hypothesis win rate (e.g. 0.52)
/// - `alpha`: significance level (Type I error rate, e.g. 0.05)
/// - `beta`: power parameter (Type II error rate, e.g. 0.05)
pub fn sprt_test(
    wins: usize,
    total: usize,
    p0: f64,
    p1: f64,
    alpha: f64,
    beta: f64,
) -> (SprtResult, f64) {
    if total == 0 {
        return (SprtResult::Continue, 0.0);
    }
    // Guard against invalid probabilities (log of 0 or negative)
    if p0 <= 0.0 || p0 >= 1.0 || p1 <= 0.0 || p1 >= 1.0 {
        return (SprtResult::Continue, 0.0);
    }

    let losses = total - wins;
    let w = wins as f64;
    let l = losses as f64;

    // Log-likelihood ratio: sum of log(P(x_i | H1) / P(x_i | H0))
    let log_lr = w * (p1 / p0).ln() + l * ((1.0 - p1) / (1.0 - p0)).ln();

    // Wald's boundaries
    let log_a = ((1.0 - beta) / alpha).ln(); // Upper: accept H1
    let log_b = (beta / (1.0 - alpha)).ln(); // Lower: accept H0

    let result = if log_lr >= log_a {
        SprtResult::AcceptH1
    } else if log_lr <= log_b {
        SprtResult::AcceptH0
    } else {
        SprtResult::Continue
    };

    (result, log_lr)
}

/// McNemar's test result for paired comparison.
#[derive(Debug, Clone)]
pub struct McNemarResult {
    /// Chi-squared statistic with continuity correction.
    pub chi2: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Games won by A but lost by B.
    pub a_only: usize,
    /// Games won by B but lost by A.
    pub b_only: usize,
}

/// McNemar's test with continuity correction for paired binary outcomes.
///
/// Compares two solvers on the same set of games. The test focuses on
/// discordant pairs (games where the two solvers disagree).
///
/// - `results_a`: per-game win/loss for solver A
/// - `results_b`: per-game win/loss for solver B (same seeds, same order)
///
/// Panics if `results_a` and `results_b` have different lengths.
pub fn mcnemar_test(results_a: &[bool], results_b: &[bool]) -> McNemarResult {
    assert_eq!(
        results_a.len(),
        results_b.len(),
        "results_a and results_b must have the same length"
    );

    let mut a_only: usize = 0;
    let mut b_only: usize = 0;
    for (&a, &b) in results_a.iter().zip(results_b.iter()) {
        if a && !b {
            a_only += 1;
        } else if !a && b {
            b_only += 1;
        }
    }

    let discordant = a_only + b_only;
    if discordant == 0 {
        return McNemarResult {
            chi2: 0.0,
            p_value: 1.0,
            a_only,
            b_only,
        };
    }

    let diff = (a_only as f64 - b_only as f64).abs();
    // Continuity correction: subtract 1 before squaring
    let corrected = (diff - 1.0).max(0.0);
    let chi2 = corrected * corrected / discordant as f64;

    // Convert chi2 to p-value via normal approximation (chi2 with 1 df)
    let z = chi2.sqrt();
    let p_value = 2.0 * (1.0 - normal_cdf(z));

    McNemarResult {
        chi2,
        p_value,
        a_only,
        b_only,
    }
}

/// Standard normal CDF approximation (Abramowitz and Stegun).
///
/// Accurate to ~1e-5 for all z. Uses the rational approximation
/// from Handbook of Mathematical Functions, formula 26.2.17.
fn normal_cdf(z: f64) -> f64 {
    if z < -8.0 {
        return 0.0;
    }
    if z > 8.0 {
        return 1.0;
    }

    // Abramowitz and Stegun approximation 26.2.17
    let x = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp();
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    if z >= 0.0 {
        1.0 - p * poly
    } else {
        p * poly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Wilson CI tests ====================

    #[test]
    fn test_wilson_ci_basic() {
        let (rate, lower, upper) = wilson_ci(50, 100, 0.95);
        assert!((rate - 0.5).abs() < 1e-10);
        assert!(lower < 0.5);
        assert!(upper > 0.5);
        // Known approximate values for 50/100 at 95%
        assert!((lower - 0.402).abs() < 0.01);
        assert!((upper - 0.598).abs() < 0.01);
    }

    #[test]
    fn test_wilson_ci_zero() {
        let (rate, lower, upper) = wilson_ci(0, 0, 0.95);
        assert_eq!(rate, 0.0);
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 0.0);
    }

    #[test]
    fn test_wilson_ci_all_wins() {
        let (rate, lower, upper) = wilson_ci(100, 100, 0.95);
        assert!((rate - 1.0).abs() < 1e-10);
        assert!(lower > 0.95);
        assert!((upper - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wilson_ci_no_wins() {
        let (rate, lower, upper) = wilson_ci(0, 100, 0.95);
        assert!(rate.abs() < 1e-10);
        assert!(lower.abs() < 1e-6);
        assert!(upper < 0.05);
    }

    #[test]
    fn test_wilson_ci_large_sample() {
        // 5496/10000 = 54.96% (from Phase 2 benchmark)
        let (rate, lower, upper) = wilson_ci(5496, 10000, 0.95);
        assert!((rate - 0.5496).abs() < 1e-6);
        assert!(lower > 0.53);
        assert!(upper < 0.57);
    }

    // ==================== SPRT tests ====================

    #[test]
    fn test_sprt_obvious_h1() {
        // 600/1000 wins clearly exceeds p1=0.52
        let (result, log_lr) = sprt_test(600, 1000, 0.50, 0.52, 0.05, 0.05);
        assert_eq!(result, SprtResult::AcceptH1);
        assert!(log_lr > 0.0, "log_lr should be positive for many wins");
    }

    #[test]
    fn test_sprt_obvious_h0() {
        // 400/1000 wins clearly below p0=0.50
        let (result, log_lr) = sprt_test(400, 1000, 0.50, 0.52, 0.05, 0.05);
        assert_eq!(result, SprtResult::AcceptH0);
        assert!(log_lr < 0.0, "log_lr should be negative for few wins");
    }

    #[test]
    fn test_sprt_continue() {
        // 505/1000 is borderline — shouldn't be enough to decide
        let (result, _log_lr) = sprt_test(505, 1000, 0.50, 0.52, 0.05, 0.05);
        assert_eq!(result, SprtResult::Continue);
    }

    #[test]
    fn test_sprt_zero_total() {
        let (result, log_lr) = sprt_test(0, 0, 0.50, 0.52, 0.05, 0.05);
        assert_eq!(result, SprtResult::Continue);
        assert_eq!(log_lr, 0.0);
    }

    #[test]
    fn test_sprt_invalid_probs() {
        // p0=0.0 is invalid
        let (result, _) = sprt_test(50, 100, 0.0, 0.52, 0.05, 0.05);
        assert_eq!(result, SprtResult::Continue);
        // p1=1.0 is invalid
        let (result, _) = sprt_test(50, 100, 0.5, 1.0, 0.05, 0.05);
        assert_eq!(result, SprtResult::Continue);
    }

    #[test]
    fn test_sprt_log_lr_manual() {
        // 520/1000 wins, p0=0.50, p1=0.52
        let (_, log_lr) = sprt_test(520, 1000, 0.50, 0.52, 0.05, 0.05);
        // Manual: 520 * ln(0.52/0.50) + 480 * ln(0.48/0.50)
        let expected = 520.0 * (0.52_f64 / 0.50).ln() + 480.0 * (0.48_f64 / 0.50).ln();
        assert!(
            (log_lr - expected).abs() < 1e-10,
            "log_lr={} expected={}",
            log_lr,
            expected
        );
    }

    // ==================== McNemar tests ====================

    #[test]
    fn test_mcnemar_identical() {
        // Both solvers win the same games
        let a = vec![true, false, true, true, false];
        let b = vec![true, false, true, true, false];
        let result = mcnemar_test(&a, &b);
        assert_eq!(result.a_only, 0);
        assert_eq!(result.b_only, 0);
        assert_eq!(result.chi2, 0.0);
        assert_eq!(result.p_value, 1.0);
    }

    #[test]
    fn test_mcnemar_a_clearly_better() {
        // Build test data: 30 concordant wins, 20 concordant losses,
        // 40 A-only wins, 10 B-only wins
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..30 {
            a.push(true);
            b.push(true);
        }
        for _ in 0..20 {
            a.push(false);
            b.push(false);
        }
        for _ in 0..40 {
            a.push(true);
            b.push(false);
        }
        for _ in 0..10 {
            a.push(false);
            b.push(true);
        }

        let result = mcnemar_test(&a, &b);
        assert_eq!(result.a_only, 40);
        assert_eq!(result.b_only, 10);
        // chi2 = (|40-10| - 1)^2 / 50 = (29)^2/50 = 841/50 = 16.82
        assert!((result.chi2 - 16.82).abs() < 0.01, "chi2={}", result.chi2);
        // p-value should be very small (< 0.001)
        assert!(
            result.p_value < 0.001,
            "p_value={} should be very small",
            result.p_value
        );
    }

    #[test]
    fn test_mcnemar_known_value() {
        // Classic textbook: a_only=20, b_only=5
        // chi2 = (|20-5| - 1)^2 / (20+5) = 14^2 / 25 = 196/25 = 7.84
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..20 {
            a.push(true);
            b.push(false);
        }
        for _ in 0..5 {
            a.push(false);
            b.push(true);
        }
        for _ in 0..75 {
            a.push(true);
            b.push(true);
        }
        let result = mcnemar_test(&a, &b);
        assert_eq!(result.a_only, 20);
        assert_eq!(result.b_only, 5);
        assert!(
            (result.chi2 - 7.84).abs() < 0.01,
            "chi2={} expected 7.84",
            result.chi2
        );
        // With chi2=7.84, df=1, p ~= 0.005
        assert!(result.p_value < 0.01, "p_value={}", result.p_value);
        assert!(result.p_value > 0.001, "p_value={}", result.p_value);
    }

    #[test]
    fn test_mcnemar_symmetric() {
        // Equal discordant pairs → not significant
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..25 {
            a.push(true);
            b.push(false);
        }
        for _ in 0..25 {
            a.push(false);
            b.push(true);
        }
        for _ in 0..50 {
            a.push(true);
            b.push(true);
        }
        let result = mcnemar_test(&a, &b);
        assert_eq!(result.a_only, 25);
        assert_eq!(result.b_only, 25);
        // chi2 = (max(|25-25| - 1, 0))^2 / 50 = 0 (continuity correction floors at 0)
        assert!(
            result.chi2.abs() < 1e-10,
            "chi2={} expected 0.0",
            result.chi2
        );
        assert!(
            (result.p_value - 1.0).abs() < 1e-6,
            "p_value={} should be 1.0 for zero chi2",
            result.p_value
        );
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mcnemar_length_mismatch() {
        mcnemar_test(&[true, false], &[true]);
    }

    // ==================== normal_cdf tests ====================

    #[test]
    fn test_normal_cdf_known_values() {
        // z=0 → 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-4);
        // z=1.96 → 0.975
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        // z=-1.96 → 0.025
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-3);
        // z=2.576 → 0.995
        assert!((normal_cdf(2.576) - 0.995).abs() < 1e-3);
    }
}
