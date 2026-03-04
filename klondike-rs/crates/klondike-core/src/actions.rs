// Action space mapping for Klondike Solitaire.
// 5 move types x 52 cards = 260 actions.
// Ported from klondike/actions.py in Phase R1.

use lonelybot::card::{Card, N_CARDS};
use lonelybot::engine::Move;

/// Number of distinct move types in Klondike.
pub const N_MOVE_TYPES: usize = 5;

/// Total action space size: move_types * cards.
pub const ACTION_SPACE_SIZE: usize = N_MOVE_TYPES * N_CARDS as usize;

/// Move type names, indexed by type ID.
pub const MOVE_NAMES: [&str; N_MOVE_TYPES] =
    ["DeckStack", "PileStack", "DeckPile", "StackPile", "Reveal"];

/// Convert (move_type, card_idx) to flat action index.
/// flat index = move_type * 52 + card_idx
#[inline]
pub fn move_to_action(move_type: u8, card_idx: u8) -> usize {
    move_type as usize * N_CARDS as usize + card_idx as usize
}

/// Convert flat action index to (move_type, card_idx).
#[inline]
pub fn action_to_move(action: usize) -> (u8, u8) {
    let move_type = (action / N_CARDS as usize) as u8;
    let card_idx = (action % N_CARDS as usize) as u8;
    (move_type, card_idx)
}

/// Extract (move_type, card_value) from a Lonelybot Move enum.
#[inline]
pub fn move_to_tuple(m: &Move) -> (u8, u8) {
    match m {
        Move::DeckStack(c) => (0, c.value()),
        Move::PileStack(c) => (1, c.value()),
        Move::DeckPile(c) => (2, c.value()),
        Move::StackPile(c) => (3, c.value()),
        Move::Reveal(c) => (4, c.value()),
    }
}

/// Construct a Lonelybot Move from (move_type, card_value).
/// Returns None for invalid move_type.
pub fn tuple_to_move(move_type: u8, card_idx: u8) -> Option<Move> {
    let card = Card::from_value(card_idx);
    match move_type {
        0 => Some(Move::DeckStack(card)),
        1 => Some(Move::PileStack(card)),
        2 => Some(Move::DeckPile(card)),
        3 => Some(Move::StackPile(card)),
        4 => Some(Move::Reveal(card)),
        _ => None,
    }
}

/// Convenience: convert a Move directly to a flat action index.
#[inline]
pub fn move_to_action_from_move(m: &Move) -> usize {
    let (mt, ci) = move_to_tuple(m);
    move_to_action(mt, ci)
}

/// Convert card index to human-readable string (e.g. "AH", "KS").
pub fn card_idx_to_str(card_idx: u8) -> String {
    let rank = card_idx / 4;
    let suit = card_idx % 4;
    const RANK_NAMES: &[u8] = b"A23456789TJQK";
    const SUIT_NAMES: &[u8] = b"HDCS";
    let mut s = String::with_capacity(2);
    s.push(RANK_NAMES[rank as usize] as char);
    s.push(SUIT_NAMES[suit as usize] as char);
    s
}

/// Convert action index to human-readable string (e.g. "DeckStack(AH)").
pub fn action_to_str(action: usize) -> String {
    let (move_type, card_idx) = action_to_move(action);
    format!(
        "{}({})",
        MOVE_NAMES[move_type as usize],
        card_idx_to_str(card_idx)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_space_size() {
        assert_eq!(ACTION_SPACE_SIZE, 260);
        assert_eq!(N_MOVE_TYPES, 5);
    }

    #[test]
    fn test_round_trip_all() {
        for mt in 0..N_MOVE_TYPES as u8 {
            for ci in 0..N_CARDS {
                let action = move_to_action(mt, ci);
                let (mt2, ci2) = action_to_move(action);
                assert_eq!(mt, mt2, "move_type mismatch at ({}, {})", mt, ci);
                assert_eq!(ci, ci2, "card_idx mismatch at ({}, {})", mt, ci);
            }
        }
    }

    #[test]
    fn test_boundary_values() {
        // action 0 = (0, 0)
        assert_eq!(action_to_move(0), (0, 0));
        // action 259 = (4, 51)
        assert_eq!(action_to_move(259), (4, 51));
        // forward
        assert_eq!(move_to_action(0, 0), 0);
        assert_eq!(move_to_action(4, 51), 259);
    }

    #[test]
    fn test_tuple_to_move_round_trip() {
        // Test each move type with a known card
        let test_cards: [(u8, u8); 5] = [(0, 0), (1, 10), (2, 25), (3, 40), (4, 51)];
        for (mt, ci) in test_cards {
            let m = tuple_to_move(mt, ci).expect("valid move type");
            let (mt2, ci2) = move_to_tuple(&m);
            assert_eq!(mt, mt2, "move_type mismatch for type {}", mt);
            assert_eq!(ci, ci2, "card_idx mismatch for type {}", mt);
        }
    }

    #[test]
    fn test_tuple_to_move_invalid() {
        assert!(tuple_to_move(5, 0).is_none());
        assert!(tuple_to_move(255, 0).is_none());
    }

    #[test]
    fn test_move_to_action_from_move() {
        let m = Move::DeckStack(Card::from_value(0));
        assert_eq!(move_to_action_from_move(&m), 0);

        let m = Move::Reveal(Card::from_value(51));
        assert_eq!(move_to_action_from_move(&m), 259);

        let m = Move::DeckPile(Card::from_value(10));
        // type 2, card 10 => 2*52 + 10 = 114
        assert_eq!(move_to_action_from_move(&m), 114);
    }

    #[test]
    fn test_card_idx_to_str() {
        // card 0 = rank 0, suit 0 = Ace of Hearts
        assert_eq!(card_idx_to_str(0), "AH");
        // card 51 = rank 12, suit 3 = King of Spades
        assert_eq!(card_idx_to_str(51), "KS");
        // card 4 = rank 1, suit 0 = 2 of Hearts
        assert_eq!(card_idx_to_str(4), "2H");
    }

    #[test]
    fn test_action_to_str() {
        assert_eq!(action_to_str(0), "DeckStack(AH)");
        assert_eq!(action_to_str(259), "Reveal(KS)");
    }
}
