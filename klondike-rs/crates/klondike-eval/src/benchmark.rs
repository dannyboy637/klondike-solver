// Parallel benchmark runner using rayon.
// Ported from klondike/eval/benchmark.py in Phase R6.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::stats::{sprt_test, wilson_ci, SprtResult};

/// Benchmark configuration.
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    /// Number of games to play.
    pub n_games: usize,
    /// Starting seed (game i uses seed_start + i).
    pub seed_start: u64,
    /// Print progress every N completed games.
    pub log_every: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            n_games: 10000,
            seed_start: 0,
            log_every: 100,
        }
    }
}

/// Benchmark result.
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Total wins.
    pub wins: usize,
    /// Total games played.
    pub total: usize,
    /// Win rate (wins / total).
    pub win_rate: f64,
    /// Wilson 95% CI lower bound.
    pub ci_lower: f64,
    /// Wilson 95% CI upper bound.
    pub ci_upper: f64,
    /// Per-game win/loss results (for McNemar paired comparison).
    pub game_results: Vec<bool>,
    /// Average wall-clock time per game in milliseconds.
    pub avg_time_ms: f64,
    /// Total wall-clock time in seconds.
    pub total_time_s: f64,
}

/// Run a benchmark: play N games in parallel using rayon.
///
/// `play_game` is a function that takes a seed and returns true if the game was won.
/// It must be `Sync` (safe to call from multiple threads) and `Send`.
///
/// Progress is printed to stderr every `config.log_every` games, including
/// Wilson CI and SPRT status.
pub fn run_benchmark(
    config: &BenchmarkConfig,
    play_game: &(dyn Fn(u64) -> bool + Sync),
) -> BenchmarkResult {
    let start = Instant::now();
    let completed = AtomicUsize::new(0);
    let wins_atomic = AtomicUsize::new(0);

    // Play all games in parallel, collect per-game results
    let game_results: Vec<bool> = (0..config.n_games)
        .into_par_iter()
        .map(|i| {
            let seed = config.seed_start + i as u64;
            let won = play_game(seed);

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if won {
                wins_atomic.fetch_add(1, Ordering::Relaxed);
            }

            // Progress logging (approximate — races are fine for display)
            if done.is_multiple_of(config.log_every) || done == config.n_games {
                let w = wins_atomic.load(Ordering::Relaxed);
                let (rate, ci_lo, ci_hi) = wilson_ci(w, done, 0.95);
                let (sprt, _) = sprt_test(w, done, 0.50, 0.52, 0.05, 0.05);
                let sprt_str = match sprt {
                    SprtResult::AcceptH0 => "H0",
                    SprtResult::AcceptH1 => "H1",
                    SprtResult::Continue => "..",
                };
                let elapsed = start.elapsed().as_secs_f64();
                let ms_per_game = if done > 0 {
                    elapsed * 1000.0 / done as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[{done:>6}/{total}] {w}/{done} = {rate:.2}% CI({ci_lo:.2}%, {ci_hi:.2}%) SPRT={sprt_str} {ms_per_game:.1}ms/game",
                    total = config.n_games,
                    rate = rate * 100.0,
                    ci_lo = ci_lo * 100.0,
                    ci_hi = ci_hi * 100.0,
                );
            }

            won
        })
        .collect();

    let total_time = start.elapsed().as_secs_f64();
    let wins = game_results.iter().filter(|&&w| w).count();
    let total = game_results.len();
    let (win_rate, ci_lower, ci_upper) = wilson_ci(wins, total, 0.95);
    let avg_time_ms = if total > 0 {
        total_time * 1000.0 / total as f64
    } else {
        0.0
    };

    BenchmarkResult {
        wins,
        total,
        win_rate,
        ci_lower,
        ci_upper,
        game_results,
        avg_time_ms,
        total_time_s: total_time,
    }
}

/// Run a benchmark sequentially (single-threaded).
///
/// Same as `run_benchmark` but does not use rayon. Useful when the game-playing
/// function captures non-Sync state (e.g. GPU neural net inference).
pub fn run_benchmark_sequential(
    config: &BenchmarkConfig,
    play_game: &dyn Fn(u64) -> bool,
) -> BenchmarkResult {
    let start = Instant::now();
    let mut wins: usize = 0;
    let mut game_results = Vec::with_capacity(config.n_games);

    for i in 0..config.n_games {
        let seed = config.seed_start + i as u64;
        let won = play_game(seed);
        if won {
            wins += 1;
        }
        game_results.push(won);

        let done = i + 1;
        if done.is_multiple_of(config.log_every) || done == config.n_games {
            let (rate, ci_lo, ci_hi) = wilson_ci(wins, done, 0.95);
            let (sprt, _) = sprt_test(wins, done, 0.50, 0.52, 0.05, 0.05);
            let sprt_str = match sprt {
                SprtResult::AcceptH0 => "H0",
                SprtResult::AcceptH1 => "H1",
                SprtResult::Continue => "..",
            };
            let elapsed = start.elapsed().as_secs_f64();
            let ms_per_game = if done > 0 {
                elapsed * 1000.0 / done as f64
            } else {
                0.0
            };
            eprintln!(
                "[{done:>6}/{total}] {wins}/{done} = {rate:.2}% CI({ci_lo:.2}%, {ci_hi:.2}%) SPRT={sprt_str} {ms_per_game:.1}ms/game",
                total = config.n_games,
                rate = rate * 100.0,
                ci_lo = ci_lo * 100.0,
                ci_hi = ci_hi * 100.0,
            );
        }
    }

    let total_time = start.elapsed().as_secs_f64();
    let total = game_results.len();
    let (win_rate, ci_lower, ci_upper) = wilson_ci(wins, total, 0.95);
    let avg_time_ms = if total > 0 {
        total_time * 1000.0 / total as f64
    } else {
        0.0
    };

    BenchmarkResult {
        wins,
        total,
        win_rate,
        ci_lower,
        ci_upper,
        game_results,
        avg_time_ms,
        total_time_s: total_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_basic() {
        // Simple game: win if seed is even
        let config = BenchmarkConfig {
            n_games: 100,
            seed_start: 0,
            log_every: 50,
        };
        let result = run_benchmark(&config, &|seed| seed % 2 == 0);

        assert_eq!(result.total, 100);
        assert_eq!(result.wins, 50);
        assert!((result.win_rate - 0.5).abs() < 1e-10);
        assert!(result.ci_lower < 0.5);
        assert!(result.ci_upper > 0.5);
        assert_eq!(result.game_results.len(), 100);
    }

    #[test]
    fn test_benchmark_all_wins() {
        let config = BenchmarkConfig {
            n_games: 20,
            seed_start: 0,
            log_every: 100, // no progress output
        };
        let result = run_benchmark(&config, &|_| true);
        assert_eq!(result.wins, 20);
        assert!((result.win_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_benchmark_seed_ordering() {
        // Verify that game_results[i] corresponds to seed_start + i
        let config = BenchmarkConfig {
            n_games: 10,
            seed_start: 100,
            log_every: 100,
        };
        // Win only for seed 103
        let result = run_benchmark(&config, &|seed| seed == 103);
        assert_eq!(result.wins, 1);
        assert!(result.game_results[3]); // seed 103 = index 3
    }
}
