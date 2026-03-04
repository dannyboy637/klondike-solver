// Parallel self-play game generation with rayon.
// Ported from klondike/alphazero/selfplay.py in Phase R4.

use std::collections::HashSet;
use std::time::Instant;

use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::env::KlondikeEnv;
use klondike_search::eval_fn::EvalFn;
use klondike_search::pimc::{pimc_search, PimcConfig};
use rayon::prelude::*;

use crate::buffer::TrainingSample;

/// Configuration for self-play game generation.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Number of determinizations per move (3-5 training, 50-100 eval).
    pub n_determinizations: u32,
    /// MCTS simulations per determinization.
    pub mcts_simulations: u32,
    /// PUCT exploration constant.
    pub c_puct: f32,
    /// Number of initial moves with temperature=1.0 for exploration.
    pub temperature_moves: u32,
    /// Temperature for moves after `temperature_moves` (near-greedy).
    pub temperature_final: f32,
    /// Dirichlet noise alpha for root exploration.
    pub dirichlet_alpha: f32,
    /// Fraction of Dirichlet noise mixed into root prior.
    pub dirichlet_frac: f32,
    /// Maximum moves per game (cycle/infinite-loop safety).
    pub max_moves: u32,
    /// Draw step (1 = draw-1, 3 = draw-3).
    pub draw_step: u8,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            n_determinizations: 5,
            mcts_simulations: 200,
            c_puct: 1.5,
            temperature_moves: 20,
            temperature_final: 0.05,
            dirichlet_alpha: 0.3,
            dirichlet_frac: 0.25,
            max_moves: 500,
            draw_step: 1,
        }
    }
}

/// Result of a single self-play game.
#[derive(Clone, Debug)]
pub struct GameData {
    /// Training samples from this game (obs, policy, legal_mask, value backfilled).
    pub samples: Vec<TrainingSample>,
    /// Whether the game was won.
    pub won: bool,
    /// Number of moves played.
    pub n_moves: usize,
    /// Seed used to create this game.
    pub seed: u64,
    /// Wall-clock time for the entire game in milliseconds.
    pub wall_time_ms: f64,
}

/// Play a single self-play game, collecting training samples.
///
/// Uses PC-PIMC search at each decision point. The temperature schedule
/// uses temperature=1.0 for the first `config.temperature_moves` moves,
/// then switches to `config.temperature_final` for the rest.
///
/// After the game ends, all sample values are backfilled with the outcome
/// (1.0 for win, 0.0 for loss).
pub fn play_game(config: &SelfPlayConfig, eval_fn: &dyn EvalFn, seed: u64) -> GameData {
    let start = Instant::now();
    let mut env = KlondikeEnv::new(seed, config.draw_step);

    let mut observations = Vec::new();
    let mut policies = Vec::new();
    let mut legal_masks = Vec::new();
    let mut move_count: u32 = 0;
    let mut seen_states = HashSet::new();

    while !env.is_done() {
        let legal = env.get_legal_actions();
        if legal.is_empty() {
            break;
        }

        // Cycle detection
        let state_hash = env.encode_state_hash();
        if seen_states.contains(&state_hash) {
            break;
        }
        seen_states.insert(state_hash);

        // Max moves safety
        if move_count >= config.max_moves {
            break;
        }

        // Temperature schedule
        let temperature = if move_count < config.temperature_moves {
            1.0
        } else {
            config.temperature_final
        };

        // Build PIMC config for this move
        let pimc_config = PimcConfig {
            n_determinizations: config.n_determinizations,
            mcts_simulations: config.mcts_simulations,
            c_puct: config.c_puct,
            temperature,
            dirichlet_alpha: config.dirichlet_alpha,
            dirichlet_frac: config.dirichlet_frac,
        };

        // Run PC-PIMC search (handles determinization loop internally)
        let result = pimc_search(&env, &pimc_config, eval_fn, seed, move_count);

        if result.action < 0 {
            break;
        }

        // Normalize visit counts to policy distribution
        let visit_sum: f32 = result.visit_counts.iter().sum();
        let policy = if visit_sum > 0.0 {
            let mut p = [0.0f32; ACTION_SPACE_SIZE];
            for (i, &vc) in result.visit_counts.iter().enumerate() {
                p[i] = vc / visit_sum;
            }
            p
        } else {
            // Fallback: uniform over legal moves
            let mut p = [0.0f32; ACTION_SPACE_SIZE];
            let inv = 1.0 / legal.len() as f32;
            for &a in &legal {
                p[a] = inv;
            }
            p
        };

        // Record training data
        observations.push(env.get_observation());
        policies.push(policy);
        legal_masks.push(env.get_legal_mask());

        // Use pimc_search's selected action (already applies temperature)
        let action = result.action as usize;

        match env.step(action) {
            Ok(_) => {}
            Err(_) => break,
        }
        move_count += 1;
    }

    // Backfill outcome
    let won = env.game().is_win();
    let outcome = if won { 1.0f32 } else { 0.0f32 };

    // Build training samples
    let samples: Vec<TrainingSample> = observations
        .into_iter()
        .zip(policies)
        .zip(legal_masks)
        .map(|((obs, pol), mask)| TrainingSample {
            observation: obs.to_vec(),
            policy: pol.to_vec(),
            value: outcome,
            legal_mask: mask.to_vec(),
        })
        .collect();

    GameData {
        samples,
        won,
        n_moves: move_count as usize,
        seed,
        wall_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

/// Play multiple self-play games in parallel using rayon.
///
/// Each game runs independently with its own seed. The `eval_fn` must be
/// `Sync` (stateless evaluators like `UniformEval` satisfy this automatically).
pub fn play_games_parallel(
    config: &SelfPlayConfig,
    eval_fn: &(dyn EvalFn + Sync),
    seeds: &[u64],
) -> Vec<GameData> {
    seeds
        .par_iter()
        .map(|&seed| play_game(config, eval_fn, seed))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use klondike_core::encoding::OBSERVATION_DIM;
    use klondike_search::eval_fn::UniformEval;

    #[test]
    fn test_play_single_game() {
        let config = SelfPlayConfig {
            n_determinizations: 2,
            mcts_simulations: 30,
            max_moves: 100,
            dirichlet_frac: 0.0, // No noise for speed
            ..SelfPlayConfig::default()
        };
        let eval = UniformEval;
        let game = play_game(&config, &eval, 42);

        // Game should have played some moves
        assert!(game.n_moves > 0, "game should play at least one move");
        assert_eq!(game.seed, 42);
        assert!(game.wall_time_ms > 0.0);

        // Samples should match move count
        assert_eq!(
            game.samples.len(),
            game.n_moves,
            "should have one sample per move"
        );

        // All values should be backfilled to the same outcome
        let expected_value = if game.won { 1.0f32 } else { 0.0f32 };
        for (i, sample) in game.samples.iter().enumerate() {
            assert_eq!(
                sample.value, expected_value,
                "sample {} value should be backfilled to {}",
                i, expected_value
            );

            // Observation should be 757-dim, all in [0, 1]
            assert_eq!(sample.observation.len(), OBSERVATION_DIM);
            for (j, &v) in sample.observation.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "sample {} obs[{}] = {} out of range",
                    i,
                    j,
                    v
                );
            }

            // Policy should sum to ~1.0
            let policy_sum: f32 = sample.policy.iter().sum();
            assert!(
                (policy_sum - 1.0).abs() < 0.01,
                "sample {} policy sum {} should be ~1.0",
                i,
                policy_sum
            );

            // Legal mask should have at least one legal action
            let n_legal: usize = sample.legal_mask.iter().filter(|&&v| v > 0.0).count();
            assert!(
                n_legal > 0,
                "sample {} should have at least one legal action",
                i
            );

            // Policy should only have nonzero values for legal actions
            for (a, &p) in sample.policy.iter().enumerate() {
                if sample.legal_mask[a] == 0.0 && p > 0.0 {
                    panic!("sample {} has policy {} for illegal action {}", i, p, a);
                }
            }
        }
    }

    #[test]
    fn test_play_games_parallel() {
        let config = SelfPlayConfig {
            n_determinizations: 2,
            mcts_simulations: 20,
            max_moves: 50,
            dirichlet_frac: 0.0,
            ..SelfPlayConfig::default()
        };
        let eval = UniformEval;
        let seeds: Vec<u64> = (0..20).collect();

        let games = play_games_parallel(&config, &eval, &seeds);

        assert_eq!(games.len(), 20, "should play all 20 games");

        let mut total_moves = 0;
        let mut wins = 0;
        for game in &games {
            assert!(
                game.n_moves > 0,
                "seed {} should play at least one move",
                game.seed
            );
            assert_eq!(
                game.samples.len(),
                game.n_moves,
                "seed {} sample count mismatch",
                game.seed
            );
            total_moves += game.n_moves;
            if game.won {
                wins += 1;
            }

            // Verify value backfill consistency
            let expected_value = if game.won { 1.0f32 } else { 0.0f32 };
            for sample in &game.samples {
                assert_eq!(
                    sample.value, expected_value,
                    "seed {} value backfill inconsistent",
                    game.seed
                );
            }
        }

        eprintln!(
            "20 parallel games: {wins}/20 wins, avg {:.1} moves/game",
            total_moves as f64 / 20.0
        );
    }

    #[test]
    fn test_cycle_detection_terminates() {
        // Even with very high max_moves, cycle detection should stop the game
        let config = SelfPlayConfig {
            n_determinizations: 1,
            mcts_simulations: 10,
            max_moves: 10000,
            dirichlet_frac: 0.0,
            ..SelfPlayConfig::default()
        };
        let eval = UniformEval;
        let game = play_game(&config, &eval, 42);

        // Game should terminate before max_moves due to either:
        // win, no legal moves, or cycle detection
        assert!(
            game.n_moves < 10000,
            "game should terminate before max_moves (got {})",
            game.n_moves
        );
    }

    #[test]
    fn test_temperature_schedule() {
        // With temperature_moves=0, all moves should use temperature_final (near-greedy).
        // With uniform eval the results won't be meaningfully different,
        // but we verify the game completes without error.
        let config = SelfPlayConfig {
            n_determinizations: 2,
            mcts_simulations: 20,
            temperature_moves: 0,
            temperature_final: 0.01,
            max_moves: 50,
            dirichlet_frac: 0.0,
            ..SelfPlayConfig::default()
        };
        let eval = UniformEval;
        let game = play_game(&config, &eval, 42);
        assert!(game.n_moves > 0);
    }

    #[test]
    fn test_deterministic_single_game() {
        let config = SelfPlayConfig {
            n_determinizations: 2,
            mcts_simulations: 30,
            max_moves: 50,
            dirichlet_frac: 0.0, // No noise for determinism
            ..SelfPlayConfig::default()
        };
        let eval = UniformEval;

        let g1 = play_game(&config, &eval, 42);
        let g2 = play_game(&config, &eval, 42);

        assert_eq!(
            g1.n_moves, g2.n_moves,
            "same seed should give same move count"
        );
        assert_eq!(g1.won, g2.won, "same seed should give same outcome");
        assert_eq!(g1.samples.len(), g2.samples.len());

        // Policies should be identical
        for (i, (s1, s2)) in g1.samples.iter().zip(g2.samples.iter()).enumerate() {
            assert_eq!(
                s1.policy, s2.policy,
                "sample {} policy should be deterministic",
                i
            );
        }
    }
}
