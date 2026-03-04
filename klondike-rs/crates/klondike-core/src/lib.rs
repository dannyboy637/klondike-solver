//! Core game environment for Klondike Solitaire.
//!
//! Provides the game state wrapper ([`env::KlondikeEnv`]), 757-dim neural network
//! encoding ([`encoding::encode_state`]), and action space mapping ([`actions`]).
//! Wraps Lonelybot's Solitaire engine with zero FFI overhead.

pub mod actions;
pub mod encoding;
pub mod env;

#[cfg(test)]
mod tests {
    use lonelybot::engine::Solitaire;
    use lonelybot::shuffler::default_shuffle;

    #[test]
    fn test_create_game_and_make_move() {
        let cards = default_shuffle(42);
        let mut game = Solitaire::new(&cards, 1);

        // Game should not be won at start
        assert!(!game.is_win());

        // Should have legal moves
        let empty_prune = [0u64; 5];
        let moves = game.list_moves::<false>(&empty_prune);
        assert!(!moves.is_empty(), "new game should have legal moves");

        // Make first move — should not panic
        let first_move = moves[0];
        let undo = game.do_move(&first_move);

        // Undo should also work
        game.undo_move(&first_move, &undo);

        // State should be valid after undo
        let moves_after_undo = game.list_moves::<false>(&empty_prune);
        assert_eq!(
            moves.len(),
            moves_after_undo.len(),
            "move count should match after undo"
        );
    }

    #[test]
    fn test_different_seeds_different_games() {
        let cards_a = default_shuffle(42);
        let cards_b = default_shuffle(99);
        let game_a = Solitaire::new(&cards_a, 1);
        let game_b = Solitaire::new(&cards_b, 1);

        // Different seeds should produce different visible cards
        assert_ne!(
            game_a.get_visible_mask(),
            game_b.get_visible_mask(),
            "different seeds should produce different visible cards"
        );
    }

    #[test]
    fn test_clone_independence() {
        let cards = default_shuffle(42);
        let mut game = Solitaire::new(&cards, 1);
        let game_clone = game.clone();

        let empty_prune = [0u64; 5];
        let moves = game.list_moves::<false>(&empty_prune);
        game.do_move(&moves[0]);

        // Clone should be unaffected
        assert_ne!(
            game.encode(),
            game_clone.encode(),
            "clone should be independent"
        );
    }
}
