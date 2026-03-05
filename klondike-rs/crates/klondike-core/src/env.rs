// KlondikeEnv: game environment wrapping Lonelybot's Solitaire.
// Ported from klondike/env.py in Phase R1.

use lonelybot::engine::{Move, MoveVec, Solitaire, UndoInfo};
use lonelybot::pruning::PruneInfo;
use lonelybot::shuffler::default_shuffle;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::actions::{
    action_to_move, move_to_action, move_to_tuple, tuple_to_move, ACTION_SPACE_SIZE,
};
use crate::encoding::{encode_state, OBSERVATION_DIM};

/// Gym-like environment wrapping Lonelybot's Solitaire engine.
///
/// Provides action-indexed step/observe/legal_actions interface
/// suitable for RL training and search.
#[derive(Clone)]
pub struct KlondikeEnv {
    game: Solitaire,
    draw_step: u8,
    seed: u64,
    undo_stack: Vec<(Move, UndoInfo)>,
}

impl KlondikeEnv {
    /// Create a new environment with the given seed and draw step.
    pub fn new(seed: u64, draw_step: u8) -> Self {
        let cards = default_shuffle(seed);
        let game = Solitaire::new(&cards, draw_step);
        Self {
            game,
            draw_step,
            seed,
            undo_stack: Vec::new(),
        }
    }

    /// Reset the game with a new seed, clearing the undo stack.
    pub fn reset(&mut self, seed: u64) {
        let cards = default_shuffle(seed);
        self.game = Solitaire::new(&cards, self.draw_step);
        self.seed = seed;
        self.undo_stack.clear();
    }

    /// Execute an action. Returns (reward, done).
    ///
    /// Reward is 1.0 on win, 0.0 otherwise.
    /// Returns an error if the action is illegal.
    pub fn step(&mut self, action: usize) -> Result<(f32, bool), anyhow::Error> {
        let (move_type, card_idx) = action_to_move(action);
        let m = tuple_to_move(move_type, card_idx)
            .ok_or_else(|| anyhow::anyhow!("invalid move type: {}", move_type))?;

        // Verify the move is legal (check against unpruned moves)
        let moves: MoveVec = self.game.list_moves::<false>(&[0u64; 5]);
        if !moves.contains(&m) {
            return Err(anyhow::anyhow!(
                "illegal move: action={}, move_type={}, card_idx={}",
                action,
                move_type,
                card_idx
            ));
        }

        let undo = self.game.do_move(&m);
        self.undo_stack.push((m, undo));

        let done = self.game.is_win();
        let reward = if done { 1.0 } else { 0.0 };
        Ok((reward, done))
    }

    /// Get list of legal action indices (with dominance pruning).
    pub fn get_legal_actions(&self) -> Vec<usize> {
        let prune = PruneInfo::default().prune_moves::<true>(&self.game);
        let moves: MoveVec = self.game.list_moves::<true>(&prune);
        let mut actions = Vec::with_capacity(moves.len());
        for m in &moves {
            let (mt, ci) = move_to_tuple(m);
            actions.push(move_to_action(mt, ci));
        }
        actions
    }

    /// Get legal action mask (1.0 for legal, 0.0 for illegal).
    /// Uses dominance pruning.
    pub fn get_legal_mask(&self) -> [f32; ACTION_SPACE_SIZE] {
        let mut mask = [0.0f32; ACTION_SPACE_SIZE];
        let prune = PruneInfo::default().prune_moves::<true>(&self.game);
        let moves: MoveVec = self.game.list_moves::<true>(&prune);
        for m in &moves {
            let (mt, ci) = move_to_tuple(m);
            let action = move_to_action(mt, ci);
            mask[action] = 1.0;
        }
        mask
    }

    /// Get the 757-dim observation vector for the current state.
    pub fn get_observation(&self) -> [f32; OBSERVATION_DIM] {
        let mut obs = [0.0f32; OBSERVATION_DIM];
        encode_state(&self.game, &mut obs);
        obs
    }

    /// Check if the game is over (won or no legal moves).
    pub fn is_done(&self) -> bool {
        if self.game.is_win() {
            return true;
        }
        let moves: MoveVec = self.game.list_moves::<false>(&[0u64; 5]);
        moves.is_empty()
    }

    /// Deep copy of the environment.
    pub fn clone_env(&self) -> Self {
        self.clone()
    }

    /// Shuffle hidden cards for PIMC determinization.
    /// Uses shuffle_all (not shuffle) to remove all oracle information.
    pub fn shuffle_hidden(&mut self, rng_seed: u64) {
        let mut rng = SmallRng::seed_from_u64(rng_seed);
        self.game.get_hidden_mut().shuffle_all(&mut rng);
    }

    /// Get encoded game state as u64 (for cycle detection, transposition tables).
    pub fn encode_state_hash(&self) -> u64 {
        self.game.encode()
    }

    /// Get the seed used to create this game.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the number of moves made so far.
    pub fn move_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Immutable reference to the underlying Solitaire game.
    pub fn game(&self) -> &Solitaire {
        &self.game
    }

    /// Mutable reference to the underlying Solitaire game.
    pub fn game_mut(&mut self) -> &mut Solitaire {
        &mut self.game
    }

    /// Cycle through the entire stock pile, dealing all cards to waste.
    ///
    /// In draw-1, this reveals all stock cards so they appear as LOC_WASTE
    /// (visible) in the encoding instead of LOC_UNKNOWN. This matches what
    /// a human player would do at the start of a game — flip through the
    /// entire deck to see what's available before making decisions.
    ///
    /// For the HOP solver this is a no-op in terms of search behavior
    /// (it already sees all stock cards), but for neural network encoding
    /// it provides much richer information.
    pub fn cycle_deck(&mut self) {
        let deck = self.game.get_deck_mut();
        // Deal until stock portion is empty (all cards in waste).
        // In draw-1 with 24 deck cards, this takes ~23 deal_once calls.
        while !deck.get_deck().is_empty() {
            deck.deal_once();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_reset() {
        let mut env = KlondikeEnv::new(42, 1);
        assert!(!env.is_done(), "new game should not be done");
        assert!(
            !env.get_legal_actions().is_empty(),
            "new game should have legal actions"
        );

        let mask_42 = env.game().get_visible_mask();

        env.reset(99);
        assert!(!env.is_done(), "reset game should not be done");
        assert!(
            !env.get_legal_actions().is_empty(),
            "reset game should have legal actions"
        );

        let mask_99 = env.game().get_visible_mask();
        assert_ne!(
            mask_42, mask_99,
            "different seeds should produce different visible cards"
        );
    }

    #[test]
    fn test_step_legal_move() {
        let mut env = KlondikeEnv::new(42, 1);
        let actions = env.get_legal_actions();
        assert!(!actions.is_empty(), "should have legal actions");

        let result = env.step(actions[0]);
        assert!(result.is_ok(), "stepping a legal action should succeed");
        assert_eq!(env.move_count(), 1, "move count should be 1 after one step");
    }

    #[test]
    fn test_step_illegal_move() {
        let mut env = KlondikeEnv::new(42, 1);
        // Action 259 = move_type 4 (Reveal), card_idx 51 (King of Spades)
        // Almost certainly not a legal move at the start
        let result = env.step(259);
        assert!(result.is_err(), "stepping an illegal action should fail");
    }

    #[test]
    fn test_clone_independence() {
        let mut env = KlondikeEnv::new(42, 1);
        let cloned = env.clone_env();

        let actions = env.get_legal_actions();
        env.step(actions[0]).expect("legal move should succeed");

        assert_ne!(
            env.encode_state_hash(),
            cloned.encode_state_hash(),
            "clone should be independent of original"
        );
        assert_eq!(
            cloned.move_count(),
            0,
            "clone should not have any moves after original steps"
        );
    }

    #[test]
    fn test_shuffle_hidden() {
        let env = KlondikeEnv::new(42, 1);
        let mut a = env.clone_env();
        let mut b = env.clone_env();

        // Visible state should be the same before shuffle
        let vis_before = a.game().get_visible_mask();

        a.shuffle_hidden(111);
        b.shuffle_hidden(222);

        // Visible cards should be unchanged (shuffle only affects hidden cards)
        assert_eq!(
            a.game().get_visible_mask(),
            vis_before,
            "shuffle should not change visible cards"
        );
        assert_eq!(
            b.game().get_visible_mask(),
            vis_before,
            "shuffle should not change visible cards"
        );

        // Legal actions should be the same (visible state unchanged)
        assert_eq!(
            a.get_legal_actions(),
            b.get_legal_actions(),
            "shuffling hidden cards should not change legal actions"
        );
    }

    #[test]
    fn test_legal_mask_consistency() {
        let env = KlondikeEnv::new(42, 1);
        let actions = env.get_legal_actions();
        let mask = env.get_legal_mask();

        // Count 1.0 entries in mask
        let mask_count = mask.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(
            mask_count,
            actions.len(),
            "mask should have exactly as many 1.0 entries as legal actions"
        );

        // Every action in the list should have mask == 1.0
        for &action in &actions {
            assert_eq!(
                mask[action], 1.0,
                "mask should be 1.0 for legal action {}",
                action
            );
        }

        // Every mask entry not in the list should be 0.0
        for (i, &v) in mask.iter().enumerate() {
            if !actions.contains(&i) {
                assert_eq!(v, 0.0, "mask should be 0.0 for non-legal action {}", i);
            }
        }
    }

    #[test]
    fn test_observation_shape() {
        let env = KlondikeEnv::new(42, 1);
        let obs = env.get_observation();

        assert_eq!(obs.len(), OBSERVATION_DIM, "observation should be 757-dim");

        for (i, &v) in obs.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "observation[{}] = {} should be in [0, 1]",
                i,
                v
            );
        }
    }
}
