// Statistical functions: Wilson CI, SPRT, McNemar test.
// Wilson CI implemented in Phase R5 (needed by pipeline).
// SPRT and McNemar to be added in Phase R6.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
