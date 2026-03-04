// 757-dim state encoding for the neural network.
// Ported from klondike/encoding.py in Phase R1.
//
// Encoding layout:
//   Card features:       52 * 13 = 676
//   Global features:              = 14
//   Uncertainty features:         = 67
//   Total:                        = 757

use lonelybot::card::{N_CARDS, N_SUITS};
use lonelybot::deck::N_PILES;
use lonelybot::engine::Solitaire;

/// Per-card feature count: 11 location one-hot + 1 is_visible + 1 pile_position.
pub const CARD_FEATURES: usize = 13;

/// Global features: 4 foundation + 7 pile_hidden + 1 deck_remaining + 1 waste_size + 1 deck_passes.
pub const GLOBAL_FEATURES: usize = 14;

/// Uncertainty features: 52 unseen_mask + 1 unseen_red_fraction + 7*2 per-pile.
pub const UNCERTAINTY_FEATURES: usize = 67;

/// Total observation dimension.
pub const OBSERVATION_DIM: usize =
    N_CARDS as usize * CARD_FEATURES + GLOBAL_FEATURES + UNCERTAINTY_FEATURES;

// Location one-hot indices (within the 11 location slots per card).
const LOC_FOUNDATION: usize = 0;
// LOC_PILE0..LOC_PILE6 = 1..7
const LOC_WASTE: usize = 8;
// LOC_STOCK: usize = 9;  // reserved for Phase 6
const LOC_UNKNOWN: usize = 10;
const LOC_IS_VISIBLE: usize = 11;
const LOC_PILE_POSITION: usize = 12;

/// Encode the visible game state into a pre-allocated f32 buffer.
///
/// The buffer must be exactly OBSERVATION_DIM (757) elements.
/// All values are written in [0.0, 1.0].
pub fn encode_state(game: &Solitaire, obs: &mut [f32; OBSERVATION_DIM]) {
    // Zero the buffer
    obs.fill(0.0);

    // Track which cards we've accounted for
    let mut seen = [false; N_CARDS as usize];

    // --- Card-centric features (52 * 13) ---

    // 1. Foundation cards
    let stack = game.get_stack();
    for suit in 0..N_SUITS {
        let foundation_rank = stack.get(suit);
        for rank in 0..foundation_rank {
            let card_idx = (rank * N_SUITS + suit) as usize;
            seen[card_idx] = true;
            encode_card_location(obs, card_idx, LOC_FOUNDATION, 0.0);
        }
    }

    // 2. Visible pile cards
    // Position is normalized over the FULL pile (hidden + visible), matching Python's get_piles().
    let visible_piles = game.get_visible_piles();
    let hidden = game.get_hidden();
    for pile_idx in 0..N_PILES {
        let pile = &visible_piles[pile_idx as usize];
        let n_hidden = hidden.len(pile_idx);
        let face_down = if n_hidden > 0 {
            (n_hidden - 1) as usize
        } else {
            0
        };
        let total_len = face_down + pile.len();
        for (vis_pos, card) in pile.iter().enumerate() {
            let card_idx = card.value() as usize;
            seen[card_idx] = true;
            let abs_pos = face_down + vis_pos;
            let position = if total_len > 1 {
                abs_pos as f32 / (total_len - 1) as f32
            } else {
                0.0
            };
            encode_card_location_pile(obs, card_idx, pile_idx as usize, position);
        }
    }

    // 3. Waste cards
    let waste = game.get_deck().get_waste();
    for card in waste {
        let card_idx = card.value() as usize;
        seen[card_idx] = true;
        encode_card_location(obs, card_idx, LOC_WASTE, 0.0);
    }

    // 4. All remaining cards -> "unknown"
    for (card_idx, &is_seen) in seen.iter().enumerate() {
        if !is_seen {
            encode_card_location(obs, card_idx, LOC_UNKNOWN, 0.0);
        }
    }

    // --- Global features (14) ---
    let mut idx = N_CARDS as usize * CARD_FEATURES;

    // Foundation ranks (4)
    for suit in 0..N_SUITS {
        obs[idx] = stack.get(suit) as f32 / 13.0;
        idx += 1;
    }

    // Pile hidden counts (7)
    for pile_idx in 0..N_PILES {
        let n = hidden.len(pile_idx);
        let n_face_down = if n > 0 { n - 1 } else { 0 };
        obs[idx] = n_face_down as f32 / 6.0;
        idx += 1;
    }

    // Deck remaining (stock cards, not waste)
    let stock_len = game.get_deck().get_deck().len();
    obs[idx] = stock_len as f32 / 24.0;
    idx += 1;

    // Waste size
    obs[idx] = waste.len() as f32 / 24.0;
    idx += 1;

    // Deck passes (placeholder for Phase 6)
    obs[idx] = 0.0;
    idx += 1;

    // --- Uncertainty features (67) ---

    // 52 unseen card mask
    let mut n_unseen: u32 = 0;
    let mut n_unseen_red: u32 = 0;
    let mut n_unseen_useful: u32 = 0; // rank <= 6
    for (card_idx, &is_seen) in seen.iter().enumerate() {
        if !is_seen {
            obs[idx] = 1.0;
            n_unseen += 1;
            // Suit convention: 0=hearts(red), 1=diamonds(red), 2=clubs(black), 3=spades(black)
            // card_idx = rank * 4 + suit, so suit = card_idx % 4
            if card_idx % 4 < 2 {
                n_unseen_red += 1;
            }
            // rank = card_idx / 4
            if card_idx / 4 <= 6 {
                n_unseen_useful += 1;
            }
        }
        idx += 1;
    }

    // Unseen red fraction (1)
    if n_unseen > 0 {
        obs[idx] = n_unseen_red as f32 / n_unseen as f32;
    }
    idx += 1;

    // Per-pile uncertainty (7 piles * 2 features = 14)
    for pile_idx in 0..N_PILES {
        let n = hidden.len(pile_idx);
        let n_hidden = if n > 0 { n - 1 } else { 0 };
        if n_hidden == 0 || n_unseen == 0 {
            // hidden_share
            obs[idx] = 0.0;
            idx += 1;
            // useful_rank_fraction
            obs[idx] = 0.0;
            idx += 1;
        } else {
            // hidden_share: this pile's share of hidden cards
            obs[idx] = n_hidden as f32 / n_unseen.max(1) as f32;
            idx += 1;
            // useful_rank_fraction: fraction of unseen cards with rank <= 6
            obs[idx] = n_unseen_useful as f32 / n_unseen as f32;
            idx += 1;
        }
    }

    debug_assert_eq!(idx, OBSERVATION_DIM, "encoding index mismatch");
}

/// Set location one-hot, is_visible, and pile_position for a card.
/// Used for foundation, waste, and unknown locations.
#[inline]
fn encode_card_location(
    obs: &mut [f32; OBSERVATION_DIM],
    card_idx: usize,
    loc: usize,
    _position: f32,
) {
    let base = card_idx * CARD_FEATURES;
    obs[base + loc] = 1.0;
    // is_visible: 1.0 unless unknown or stock
    obs[base + LOC_IS_VISIBLE] = if loc == LOC_UNKNOWN || loc == 9 {
        0.0
    } else {
        1.0
    };
    // pile_position stays 0.0 (already zeroed)
}

/// Set location one-hot for a pile card with position.
#[inline]
fn encode_card_location_pile(
    obs: &mut [f32; OBSERVATION_DIM],
    card_idx: usize,
    pile_idx: usize,
    position: f32,
) {
    let base = card_idx * CARD_FEATURES;
    // Location one-hot: pile indices are 1..7
    obs[base + 1 + pile_idx] = 1.0;
    obs[base + LOC_IS_VISIBLE] = 1.0;
    obs[base + LOC_PILE_POSITION] = position;
}

#[cfg(test)]
mod tests {
    use super::*;
    use lonelybot::shuffler::default_shuffle;

    #[test]
    fn test_observation_dim() {
        assert_eq!(
            OBSERVATION_DIM,
            N_CARDS as usize * CARD_FEATURES + GLOBAL_FEATURES + UNCERTAINTY_FEATURES
        );
        assert_eq!(OBSERVATION_DIM, 757);
    }

    #[test]
    fn test_encode_state_basic() {
        let cards = default_shuffle(42);
        let game = Solitaire::new(&cards, 1);
        let mut obs = [0.0f32; OBSERVATION_DIM];
        encode_state(&game, &mut obs);

        // All values should be in [0, 1]
        for (i, &v) in obs.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "obs[{}] = {} out of range [0, 1]",
                i,
                v
            );
            assert!(!v.is_nan(), "obs[{}] is NaN", i);
            assert!(!v.is_infinite(), "obs[{}] is infinite", i);
        }
    }

    #[test]
    fn test_encode_state_card_accounting() {
        let cards = default_shuffle(42);
        let game = Solitaire::new(&cards, 1);
        let mut obs = [0.0f32; OBSERVATION_DIM];
        encode_state(&game, &mut obs);

        // Each of the 52 cards should have exactly one location one-hot set
        for card_idx in 0..N_CARDS as usize {
            let base = card_idx * CARD_FEATURES;
            let mut location_sum = 0.0f32;
            for loc in 0..11 {
                location_sum += obs[base + loc];
            }
            assert!(
                (location_sum - 1.0).abs() < 1e-6,
                "card {} has location_sum = {} (expected 1.0)",
                card_idx,
                location_sum
            );
        }
    }

    #[test]
    fn test_encode_state_foundation_cards() {
        // Start a game and make some foundation moves, verify encoding
        let cards = default_shuffle(42);
        let game = Solitaire::new(&cards, 1);
        let mut obs = [0.0f32; OBSERVATION_DIM];
        encode_state(&game, &mut obs);

        // At game start, foundation should be empty (all ranks 0)
        let stack = game.get_stack();
        for suit in 0..N_SUITS {
            assert_eq!(stack.get(suit), 0, "foundation should be empty at start");
        }

        // Global foundation features should all be 0.0
        let global_start = N_CARDS as usize * CARD_FEATURES;
        for suit in 0..4usize {
            assert_eq!(
                obs[global_start + suit],
                0.0,
                "foundation rank for suit {} should be 0.0",
                suit
            );
        }
    }

    #[test]
    fn test_encode_state_multiple_seeds() {
        // Different seeds should produce different encodings
        let cards_a = default_shuffle(42);
        let cards_b = default_shuffle(99);
        let game_a = Solitaire::new(&cards_a, 1);
        let game_b = Solitaire::new(&cards_b, 1);

        let mut obs_a = [0.0f32; OBSERVATION_DIM];
        let mut obs_b = [0.0f32; OBSERVATION_DIM];
        encode_state(&game_a, &mut obs_a);
        encode_state(&game_b, &mut obs_b);

        // They should differ somewhere
        let differs = obs_a
            .iter()
            .zip(obs_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "different seeds should produce different encodings"
        );
    }

    #[test]
    fn test_encode_state_visible_consistency() {
        let cards = default_shuffle(42);
        let game = Solitaire::new(&cards, 1);
        let mut obs = [0.0f32; OBSERVATION_DIM];
        encode_state(&game, &mut obs);

        // Cards in foundation, piles, or waste should be visible
        // Cards in unknown should not be visible
        for card_idx in 0..N_CARDS as usize {
            let base = card_idx * CARD_FEATURES;
            let is_unknown = obs[base + LOC_UNKNOWN] > 0.5;
            let is_visible = obs[base + LOC_IS_VISIBLE] > 0.5;

            if is_unknown {
                assert!(
                    !is_visible,
                    "card {} is unknown but marked visible",
                    card_idx
                );
            } else {
                assert!(
                    is_visible,
                    "card {} is not unknown but marked invisible",
                    card_idx
                );
            }
        }
    }

    #[test]
    fn test_encode_state_unseen_mask() {
        let cards = default_shuffle(42);
        let game = Solitaire::new(&cards, 1);
        let mut obs = [0.0f32; OBSERVATION_DIM];
        encode_state(&game, &mut obs);

        // Unseen mask should match the unknown location bits
        let unseen_start = N_CARDS as usize * CARD_FEATURES + GLOBAL_FEATURES;
        for card_idx in 0..N_CARDS as usize {
            let base = card_idx * CARD_FEATURES;
            let is_unknown = obs[base + LOC_UNKNOWN] > 0.5;
            let is_unseen = obs[unseen_start + card_idx] > 0.5;
            assert_eq!(
                is_unknown, is_unseen,
                "card {} unknown={} but unseen={}",
                card_idx, is_unknown, is_unseen
            );
        }
    }
}
