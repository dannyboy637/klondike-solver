// HOP (Hypothesis-Oriented Planning) game player.
// Thin wrapper around Lonelybot's pick_moves() for playing full games.
//
// This is Lonelybot's built-in solver — the 54.96% baseline on draw-1.
// Each "turn": determinize hidden cards → run pick_moves (UCB1 + traverse + HOP)
// on the determinized copy → play returned move sequence on the real game.

use std::collections::HashSet;

use lonelybot::engine::Solitaire;
use lonelybot::mcts_solver::pick_moves;
use lonelybot::shuffler::default_shuffle;
use lonelybot::tracking::DefaultTerminateSignal;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Configuration for the HOP game player.
#[derive(Clone, Debug)]
pub struct HopConfig {
    /// UCB1 bandit iterations per turn (n_times in pick_moves).
    pub n_times: usize,
    /// Max search nodes per HOP solve (node_limit).
    pub node_limit: usize,
    /// Max total moves per game (safety limit).
    pub max_moves: u32,
    /// Draw step (1 = draw-1, 3 = draw-3).
    pub draw_step: u8,
}

impl Default for HopConfig {
    fn default() -> Self {
        Self {
            n_times: 500,
            node_limit: 100,
            max_moves: 500,
            draw_step: 1,
        }
    }
}

/// Play a full game using Lonelybot's HOP solver.
///
/// Each turn:
/// 1. Clone game, shuffle_all hidden cards (PIMC determinization)
/// 2. Run pick_moves (UCB1 bandit + traverse + HOP) on the shuffled copy
/// 3. Play the returned move sequence on the REAL game
/// 4. Repeat until win, no moves, cycle, or max_moves
///
/// Returns (won, move_count).
pub fn play_hop_game(seed: u64, config: &HopConfig) -> (bool, u32) {
    let cards = default_shuffle(seed);
    let mut game = Solitaire::new(&cards, config.draw_step);

    // Cycle through the entire stock to reveal all deck cards.
    // In draw-1 this is free information — matches human play.
    while !game.get_deck().get_deck().is_empty() {
        game.get_deck_mut().deal_once();
    }

    let mut move_count: u32 = 0;
    let mut turn_count: u32 = 0;
    let mut seen_states = HashSet::new();
    let signal = DefaultTerminateSignal;

    while !game.is_win() && move_count < config.max_moves {
        // Cycle detection on game state
        let state_hash = game.encode();
        if seen_states.contains(&state_hash) {
            break;
        }
        seen_states.insert(state_hash);

        // Deterministic RNG per turn (matches Python: seed * 1000 + turn_count)
        let rng_seed = seed.wrapping_mul(1000).wrapping_add(turn_count as u64);
        let mut rng = SmallRng::seed_from_u64(rng_seed);

        // Clone and determinize
        let mut det_game = game.clone();
        det_game.get_hidden_mut().shuffle_all(&mut rng);

        // Run pick_moves on the determinized copy
        let best = pick_moves(
            &mut det_game,
            &mut rng,
            config.n_times,
            config.node_limit,
            &signal,
        );

        let Some(moves) = best else {
            break;
        };

        if moves.is_empty() {
            break;
        }

        // Play the move sequence on the REAL game
        for m in &moves {
            game.do_move(m);
            move_count += 1;
            if move_count >= config.max_moves {
                break;
            }
        }
        turn_count += 1;
    }

    (game.is_win(), move_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hop_game_completes() {
        let config = HopConfig {
            n_times: 100,
            node_limit: 50,
            max_moves: 500,
            draw_step: 1,
        };
        let (won, moves) = play_hop_game(42, &config);
        eprintln!("seed 42: won={}, moves={}", won, moves);
        assert!(moves <= 500);
    }

    #[test]
    fn test_hop_game_deterministic() {
        let config = HopConfig {
            n_times: 100,
            node_limit: 50,
            max_moves: 500,
            draw_step: 1,
        };
        let r1 = play_hop_game(42, &config);
        let r2 = play_hop_game(42, &config);
        assert_eq!(r1, r2, "same seed should give same result");
    }

    #[test]
    fn test_hop_20_games() {
        let config = HopConfig {
            n_times: 500,
            node_limit: 100,
            max_moves: 500,
            draw_step: 1,
        };
        let mut wins = 0;
        for seed in 0..20u64 {
            let (won, _) = play_hop_game(seed, &config);
            if won {
                wins += 1;
            }
        }
        eprintln!("HOP 20 games (nt=500, lim=100): {}/20 wins", wins);
        // With the strong config, we expect some wins (Python gets ~55%)
        // 20 games is too few to be precise, just verify it runs
    }
}
