// PC-PIMC: Policy Combining Perfect Information Monte Carlo.
// Ported from klondike/alphazero/selfplay.py (the MCTS-based PIMC loop).
//
// At each decision point:
// 1. Sample D determinized worlds (shuffle hidden cards)
// 2. Run MCTS on each determinized world
// 3. Average visit distributions across worlds
// 4. Select action from averaged distribution
//
// The neural net sees ONLY the visible (real) state, forcing
// generalization across information sets.

use std::time::Instant;

use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::env::KlondikeEnv;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::eval_fn::EvalFn;
use crate::mcts::{mcts_search_with_arena, MctsArena, MctsConfig};
use crate::SearchResult;

/// PC-PIMC search configuration.
#[derive(Clone, Debug)]
pub struct PimcConfig {
    /// Number of determinizations per move (3-5 for training, 50-100 for eval).
    pub n_determinizations: u32,
    /// MCTS simulations per determinization.
    pub mcts_simulations: u32,
    /// PUCT exploration constant.
    pub c_puct: f32,
    /// Temperature for final action selection from averaged visits.
    pub temperature: f32,
    /// Dirichlet noise alpha (applied to first determinization only).
    pub dirichlet_alpha: f32,
    /// Dirichlet noise fraction (0.0 = no noise).
    pub dirichlet_frac: f32,
}

impl Default for PimcConfig {
    fn default() -> Self {
        Self {
            n_determinizations: 5,
            mcts_simulations: 200,
            c_puct: 1.5,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_frac: 0.25,
        }
    }
}

/// Run PC-PIMC search from the given environment state.
///
/// Determinizes the hidden cards D times, runs MCTS on each,
/// and averages the visit distributions. Returns the averaged policy
/// and selected action.
///
/// Seeding: each determinization uses `game_seed * 10000 + move_num * 100 + d`
/// for hidden card shuffling. Dirichlet noise is only applied to
/// the first determinization (d == 0).
pub fn pimc_search(
    env: &KlondikeEnv,
    config: &PimcConfig,
    eval_fn: &dyn EvalFn,
    game_seed: u64,
    move_num: u32,
) -> SearchResult {
    let start = Instant::now();

    // Check for no legal moves
    let legal_mask = env.get_legal_mask();
    let n_legal: usize = legal_mask.iter().filter(|&&v| v > 0.0).count();
    if n_legal == 0 {
        return SearchResult {
            action: -1,
            visit_counts: [0.0f32; ACTION_SPACE_SIZE],
            value: 0.0,
            determinizations_used: config.n_determinizations,
            simulations_per_det: config.mcts_simulations,
            wall_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let mut aggregated_visits = [0.0f64; ACTION_SPACE_SIZE];
    let mut total_value = 0.0f64;
    let mut arena = MctsArena::new();

    for d in 0..config.n_determinizations {
        // Clone and determinize hidden cards
        let mut det_env = env.clone();
        let shuffle_seed = game_seed * 10000 + move_num as u64 * 100 + d as u64;
        det_env.shuffle_hidden(shuffle_seed);

        // MCTS config: Dirichlet noise only on first determinization
        let mcts_config = MctsConfig {
            n_simulations: config.mcts_simulations,
            c_puct: config.c_puct,
            temperature: 1.0, // Irrelevant — we use visit_counts, not MCTS's action selection
            dirichlet_alpha: config.dirichlet_alpha,
            dirichlet_frac: if d == 0 { config.dirichlet_frac } else { 0.0 },
        };

        // Use a deterministic RNG per determinization (offset to avoid correlation with shuffle)
        let mcts_rng_seed = shuffle_seed.wrapping_add(500_000_000);
        let mut mcts_rng = SmallRng::seed_from_u64(mcts_rng_seed);

        let result =
            mcts_search_with_arena(&mut arena, &det_env, &mcts_config, eval_fn, &mut mcts_rng);

        // Accumulate raw visit counts
        for (agg, &vc) in aggregated_visits.iter_mut().zip(result.visit_counts.iter()) {
            *agg += vc as f64;
        }
        total_value += result.value as f64;
    }

    // Normalize to get policy (f32 visit_counts for SearchResult)
    let visit_sum: f64 = aggregated_visits.iter().sum();
    let mut visit_counts = [0.0f32; ACTION_SPACE_SIZE];
    if visit_sum > 0.0 {
        for (vc, &agg) in visit_counts.iter_mut().zip(aggregated_visits.iter()) {
            *vc = agg as f32;
        }
    }

    // Select action from averaged visits using temperature
    let mut action_rng_seed = game_seed * 10000 + move_num as u64 * 100;
    action_rng_seed = action_rng_seed.wrapping_add(999_000_000);
    let mut action_rng = SmallRng::seed_from_u64(action_rng_seed);
    let best_action =
        crate::mcts::select_action_temperature(&visit_counts, config.temperature, &mut action_rng);

    // Average value across determinizations
    let avg_value = if config.n_determinizations > 0 {
        (total_value / config.n_determinizations as f64) as f32
    } else {
        0.0
    };

    SearchResult {
        action: best_action,
        visit_counts,
        value: avg_value,
        determinizations_used: config.n_determinizations,
        simulations_per_det: config.mcts_simulations,
        wall_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_fn::UniformEval;

    #[test]
    fn test_pimc_basic_properties() {
        let env = KlondikeEnv::new(42, 1);
        let config = PimcConfig {
            n_determinizations: 3,
            mcts_simulations: 50,
            ..PimcConfig::default()
        };
        let eval = UniformEval;

        let result = pimc_search(&env, &config, &eval, 42, 0);

        // Action should be legal
        assert!(result.action >= 0);
        let legal = env.get_legal_actions();
        assert!(
            legal.contains(&(result.action as usize)),
            "action {} should be legal",
            result.action
        );

        // Total visit counts should equal n_det * n_sims
        let total: f32 = result.visit_counts.iter().sum();
        let expected = (config.n_determinizations * config.mcts_simulations) as f32;
        assert!(
            (total - expected).abs() < 1.0,
            "total visits {} should be ~{}",
            total,
            expected
        );

        // Only legal actions should have visits
        let mask = env.get_legal_mask();
        for (i, &vc) in result.visit_counts.iter().enumerate() {
            if mask[i] == 0.0 {
                assert_eq!(vc, 0.0, "illegal action {} should have zero visits", i);
            }
        }

        // Value should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&result.value),
            "value {} should be in [0, 1]",
            result.value
        );

        // Metadata
        assert_eq!(result.determinizations_used, config.n_determinizations);
        assert_eq!(result.simulations_per_det, config.mcts_simulations);
        assert!(result.wall_time_ms > 0.0);
    }

    #[test]
    fn test_pimc_deterministic() {
        let env = KlondikeEnv::new(42, 1);
        let config = PimcConfig {
            n_determinizations: 3,
            mcts_simulations: 30,
            ..PimcConfig::default()
        };
        let eval = UniformEval;

        let r1 = pimc_search(&env, &config, &eval, 42, 0);
        let r2 = pimc_search(&env, &config, &eval, 42, 0);

        assert_eq!(r1.action, r2.action, "same inputs should give same action");
        assert_eq!(
            r1.visit_counts, r2.visit_counts,
            "same inputs should give same visits"
        );
    }

    #[test]
    fn test_pimc_different_seeds() {
        // Use seed 100 which has multiple legal moves (avoids dominance pruning to 1 move)
        let env = KlondikeEnv::new(100, 1);
        let n_legal = env.get_legal_actions().len();

        // Only test with envs that have multiple legal moves
        if n_legal <= 1 {
            eprintln!(
                "seed 100 has {} legal moves, skipping different-seeds test",
                n_legal
            );
            return;
        }

        let config = PimcConfig {
            n_determinizations: 3,
            mcts_simulations: 50,
            ..PimcConfig::default()
        };
        let eval = UniformEval;

        let r1 = pimc_search(&env, &config, &eval, 42, 0);
        let r2 = pimc_search(&env, &config, &eval, 99, 0);

        // Different game seeds should produce different visit distributions
        // (not guaranteed but very likely when there are multiple legal moves)
        assert_ne!(
            r1.visit_counts, r2.visit_counts,
            "different seeds should usually produce different visits"
        );
    }

    #[test]
    fn test_pimc_play_100_games() {
        // Play 100 games with PIMC + UniformEval.
        // Verify: no panics, no infinite loops, games terminate.
        let config = PimcConfig {
            n_determinizations: 3,
            mcts_simulations: 30,
            temperature: 1.0,
            dirichlet_frac: 0.0, // No noise for speed
            ..PimcConfig::default()
        };
        let eval = UniformEval;

        let mut wins = 0u32;
        let mut total_moves = 0u32;

        for seed in 0..100u64 {
            let mut env = KlondikeEnv::new(seed, 1);
            let mut move_count = 0u32;
            let mut seen_states = std::collections::HashSet::new();

            while !env.is_done() && move_count < 500 {
                // Cycle detection
                let hash = env.encode_state_hash();
                if seen_states.contains(&hash) {
                    break;
                }
                seen_states.insert(hash);

                let result = pimc_search(&env, &config, &eval, seed, move_count);

                if result.action < 0 {
                    break;
                }

                match env.step(result.action as usize) {
                    Ok(_) => {}
                    Err(_) => break,
                }
                move_count += 1;
            }

            if env.game().is_win() {
                wins += 1;
            }
            total_moves += move_count;
        }

        // With uniform eval, win rate should be very low but not necessarily zero
        eprintln!(
            "100 games: {wins}/100 wins, avg {:.1} moves/game",
            total_moves as f64 / 100.0
        );

        // All 100 games should have completed (no infinite loops)
        // Total moves should be reasonable (not all maxed at 500)
        assert!(
            total_moves < 50000,
            "total moves {} seems too high, possible infinite loop",
            total_moves
        );
    }
}
