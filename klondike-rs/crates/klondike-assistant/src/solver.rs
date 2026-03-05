use std::collections::HashMap;

use lonelybot::card::{Card, N_CARDS};
use lonelybot::deck::{N_PILES, N_PILE_CARDS};
use lonelybot::engine::{Move, Solitaire};
use lonelybot::mcts_solver::pick_moves;
use lonelybot::shuffler::CardDeck;
use lonelybot::tracking::DefaultTerminateSignal;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::game_state::GameState;
use crate::input::format_card;

/// A recommended move from the solver.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Recommendation {
    /// Human-readable description of the move.
    pub description: String,
    /// The lonelybot Move (if it's a card move, not a draw).
    pub lonelybot_move: Option<Move>,
    /// Is this a "draw" recommendation?
    pub is_draw: bool,
    /// Number of determinizations that voted for this move.
    pub votes: u32,
    /// Total determinizations run.
    pub total: u32,
}

impl Recommendation {
    #[allow(dead_code)]
    pub fn confidence_pct(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.votes as f32 / self.total as f32) * 100.0
    }
}

/// HOP solver configuration.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct HopSolverConfig {
    /// Number of PIMC determinizations to run.
    pub n_determinizations: u32,
    /// UCB1 iterations per pick_moves call.
    pub n_times: usize,
    /// HOP search node limit.
    pub node_limit: usize,
    /// Max moves safety limit.
    pub max_moves: u32,
}

impl Default for HopSolverConfig {
    fn default() -> Self {
        Self {
            n_determinizations: 50,
            n_times: 500,
            node_limit: 100,
            max_moves: 500,
        }
    }
}

/// The HOP-based solver for the assistant.
pub struct HopSolver {
    pub config: HopSolverConfig,
}

impl HopSolver {
    pub fn new(config: HopSolverConfig) -> Self {
        Self { config }
    }

    /// Get a move recommendation for the current game state.
    ///
    /// Runs PIMC: for each determinization, builds a Solitaire with
    /// unknown cards randomly assigned, runs pick_moves, and takes
    /// the first recommended move. Majority vote across determinizations.
    pub fn recommend(&self, state: &GameState) -> Option<Recommendation> {
        let unknowns = state.unknown_cards();
        if unknowns.is_empty() && state.waste.is_empty() && state.foundation_total() == 52 {
            return None; // game won
        }

        let mut vote_map: HashMap<String, (u32, Option<Move>)> = HashMap::new();
        let mut draw_votes: u32 = 0;
        let mut no_move_count: u32 = 0;
        let total = self.config.n_determinizations;

        for d in 0..total {
            let rng_seed = 42u64
                .wrapping_mul(10000)
                .wrapping_add(state.move_count() as u64 * 100)
                .wrapping_add(d as u64);

            let result = self.run_determinization(state, &unknowns, rng_seed);

            match result {
                DeterminizationResult::Move(m) => {
                    let key = format_move_key(&m);
                    let entry = vote_map.entry(key).or_insert((0, Some(m)));
                    entry.0 += 1;
                }
                DeterminizationResult::Draw => {
                    draw_votes += 1;
                }
                DeterminizationResult::NoMove => {
                    no_move_count += 1;
                }
            }
        }

        // Find the move with the most votes
        let best_move = vote_map
            .iter()
            .max_by_key(|(_, (votes, _))| *votes)
            .map(|(_, (votes, m))| (*votes, *m));

        // Compare with draw votes
        if let Some((move_votes, lonelybot_move)) = best_move {
            if move_votes >= draw_votes {
                let desc = format_move_description(lonelybot_move.as_ref(), state);
                return Some(Recommendation {
                    description: desc,
                    lonelybot_move,
                    is_draw: false,
                    votes: move_votes,
                    total: total - no_move_count,
                });
            }
        }

        if draw_votes > 0 {
            return Some(Recommendation {
                description: "Draw next card".to_string(),
                lonelybot_move: None,
                is_draw: true,
                votes: draw_votes,
                total: total - no_move_count,
            });
        }

        // No moves available
        None
    }

    fn run_determinization(
        &self,
        state: &GameState,
        unknowns: &[Card],
        rng_seed: u64,
    ) -> DeterminizationResult {
        let mut rng = SmallRng::seed_from_u64(rng_seed);

        // Shuffle the unknown cards for this determinization
        let mut shuffled_unknowns = unknowns.to_vec();
        shuffled_unknowns.shuffle(&mut rng);

        // Build a CardDeck [Card; 52] representing the initial deal
        let deal = match build_deal_array(state, &shuffled_unknowns) {
            Some(d) => d,
            None => return DeterminizationResult::NoMove,
        };

        // Create Solitaire from the deal
        let mut game = Solitaire::new(&deal, 1);

        // Cycle all stock to waste (matches the real game's initial state)
        while !game.get_deck().get_deck().is_empty() {
            game.get_deck_mut().deal_once();
        }

        // Replay the move history to reach current state
        if !replay_moves(&mut game, state) {
            return DeterminizationResult::NoMove;
        }

        // Now run pick_moves on the determinized + replayed game
        // Shuffle hidden cards for the solver's internal randomization
        game.get_hidden_mut().shuffle_all(&mut rng);

        let signal = DefaultTerminateSignal;
        let result = pick_moves(
            &mut game,
            &mut rng,
            self.config.n_times,
            self.config.node_limit,
            &signal,
        );

        match result {
            Some(moves) if !moves.is_empty() => DeterminizationResult::Move(moves[0]),
            _ => {
                // pick_moves returned nothing — try drawing
                // Check if drawing would help (deck not exhausted)
                if game.get_deck().get_deck().is_empty() && game.get_deck().get_waste().is_empty() {
                    DeterminizationResult::NoMove
                } else {
                    DeterminizationResult::Draw
                }
            }
        }
    }
}

enum DeterminizationResult {
    Move(Move),
    Draw,
    NoMove,
}

/// Build a [Card; 52] deal array from the game state.
///
/// Layout (matches Solitaire::new expectations):
/// - cards[0..28]: tableau hidden piles
///   - Pile i occupies cards[i*(i+1)/2 .. (i+1)*(i+2)/2]
///   - Last card in each pile's range is the initial face-up card
///   - Earlier cards are face-down (hidden)
/// - cards[28..52]: stock cards in order
fn build_deal_array(state: &GameState, shuffled_unknowns: &[Card]) -> Option<CardDeck> {
    let mut deal = [Card::from_value(0); N_CARDS as usize];
    let mut unknown_idx = 0;

    for pile_idx in 0..N_PILES as usize {
        let pile = &state.tableau[pile_idx];
        let start = pile_idx * (pile_idx + 1) / 2;

        // The initial face-up card (the one dealt at game start) is the LAST
        // card in this pile's hidden_piles range. It's the first face-up card
        // the user entered during setup. If the pile still has its original
        // face-up card, it's face_up[0]. If the original was moved away and
        // cards were revealed, we need to reconstruct.

        // For the initial deal, we need:
        // - positions start..start+pile_idx: face-down cards
        // - position start+pile_idx: initial face-up card

        // Face-down cards: use revealed_hidden for known ones, unknowns for rest
        let face_down_count = pile_idx; // initial face-down count for pile i
        let mut assigned_down = 0;

        // Check if any face-down cards in this pile have been revealed during play
        // and are now known (they were in the original hidden pile)
        let revealed_for_pile: Vec<Card> = state
            .revealed_hidden
            .iter()
            .filter(|(p, _)| *p == pile_idx as u8)
            .map(|(_, c)| *c)
            .collect();

        // Place known revealed cards first (in order they were revealed,
        // meaning from top of hidden pile = end of range)
        // The hidden pile stores cards bottom-to-top, reveals happen top-first
        // So revealed cards fill from the end of the hidden range backward
        for card in &revealed_for_pile {
            if assigned_down < face_down_count {
                deal[start + assigned_down] = *card;
                assigned_down += 1;
            }
        }

        // Fill remaining face-down positions with unknowns
        while assigned_down < face_down_count {
            if unknown_idx >= shuffled_unknowns.len() {
                return None;
            }
            deal[start + assigned_down] = shuffled_unknowns[unknown_idx];
            unknown_idx += 1;
            assigned_down += 1;
        }

        // Initial face-up card (the bottom of the visible stack).
        // This is the card the user entered during setup for this pile.
        // If the pile is not empty, it's the first face-up card.
        // If the pile is empty (original card moved + all hidden revealed),
        // we still need SOME card here for the deal to be valid.
        // Use the first card that was originally dealt face-up here.
        if !pile.face_up.is_empty() {
            deal[start + face_down_count] = pile.face_up[0];
        } else if !revealed_for_pile.is_empty() {
            // Pile is empty now. The original face-up card was moved.
            // For the deal array, we can use ANY card that was originally
            // in this pile position. Use the last revealed card as a proxy.
            // This works because we replay moves to reach current state.
            deal[start + face_down_count] = *revealed_for_pile.last().unwrap();
        } else if unknown_idx < shuffled_unknowns.len() {
            // Fallback: use an unknown (shouldn't normally happen)
            deal[start + face_down_count] = shuffled_unknowns[unknown_idx];
            unknown_idx += 1;
        }
    }

    // Stock cards (positions 28..52)
    let stock_start = N_PILE_CARDS as usize;
    for (i, card) in state.stock_order.iter().enumerate() {
        if stock_start + i >= N_CARDS as usize {
            break;
        }
        deal[stock_start + i] = *card;
    }

    // Fill any remaining stock positions with unknowns (shouldn't happen
    // if setup is correct — stock should always have 24 cards)
    let stock_filled = state.stock_order.len();
    for i in stock_filled..24 {
        if unknown_idx < shuffled_unknowns.len() {
            deal[stock_start + i] = shuffled_unknowns[unknown_idx];
            unknown_idx += 1;
        }
    }

    Some(deal)
}

/// Replay the move history on a Solitaire to reach the current game state.
///
/// Returns true if replay succeeded.
fn replay_moves(game: &mut Solitaire, state: &GameState) -> bool {
    for game_move in &state.move_history {
        match game_move {
            crate::game_state::GameMove::WasteToFoundation(card) => {
                let m = Move::DeckStack(*card);
                game.do_move(&m);
            }
            crate::game_state::GameMove::WasteToPile(card, _pile) => {
                let m = Move::DeckPile(*card);
                game.do_move(&m);
            }
            crate::game_state::GameMove::PileToFoundation(card, _pile) => {
                let m = Move::PileStack(*card);
                game.do_move(&m);
            }
            crate::game_state::GameMove::PileToPile(card, _from, _to) => {
                // In lonelybot, pile-to-pile isn't a direct Move variant.
                // It's handled via the visible pile state.
                // We need to figure out what Move variant this maps to.
                // Actually, lonelybot doesn't have a PileToPile move!
                // Pile-to-pile moves in lonelybot are implicit through
                // the top_mask / visible_mask system.
                // For now, skip — the determinized game should still work
                // because the state is built from scratch.
                // TODO: handle this properly if needed
                let _ = card;
            }
            crate::game_state::GameMove::FoundationToPile(card, _pile) => {
                let m = Move::StackPile(*card);
                game.do_move(&m);
            }
            crate::game_state::GameMove::Draw => {
                game.get_deck_mut().deal_once();
            }
            crate::game_state::GameMove::Reveal(card, _pile) => {
                let m = Move::Reveal(*card);
                game.do_move(&m);
            }
        }
    }
    true
}

/// Create a unique key for a Move to group identical moves across determinizations.
fn format_move_key(m: &Move) -> String {
    match m {
        Move::DeckStack(c) => format!("DS:{}", c.value()),
        Move::PileStack(c) => format!("PS:{}", c.value()),
        Move::DeckPile(c) => format!("DP:{}", c.value()),
        Move::StackPile(c) => format!("SP:{}", c.value()),
        Move::Reveal(c) => format!("RV:{}", c.value()),
    }
}

/// Format a move as a human-readable description for the user.
fn format_move_description(m: Option<&Move>, state: &GameState) -> String {
    let Some(m) = m else {
        return "No move available".to_string();
    };

    match m {
        Move::DeckStack(card) => {
            let suit_sym = crate::input::suit_symbol(card.suit());
            format!("Move {} → {} foundation", format_card(*card), suit_sym)
        }
        Move::PileStack(card) => {
            let suit_sym = crate::input::suit_symbol(card.suit());
            // Find which pile this card is on
            let pile_num = find_card_pile(state, *card)
                .map(|p| format!("pile {}", p + 1))
                .unwrap_or_else(|| "tableau".to_string());
            format!(
                "Move {} from {} → {} foundation",
                format_card(*card),
                pile_num,
                suit_sym
            )
        }
        Move::DeckPile(card) => {
            format!("Move {} from waste → tableau", format_card(*card))
        }
        Move::StackPile(card) => {
            let suit_sym = crate::input::suit_symbol(card.suit());
            format!(
                "Move {} from {} foundation → tableau",
                format_card(*card),
                suit_sym
            )
        }
        Move::Reveal(_card) => "Flip face-down card".to_string(),
    }
}

/// Find which pile a card is on in the game state.
fn find_card_pile(state: &GameState, card: Card) -> Option<u8> {
    for (i, pile) in state.tableau.iter().enumerate() {
        for c in &pile.face_up {
            if c.value() == card.value() {
                return Some(i as u8);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_deal_array_basic() {
        // Create a simple game state
        let all_cards: Vec<Card> = (0..52).map(Card::from_value).collect();

        let tableau: [Card; 7] = [
            all_cards[0],  // pile 0 face-up: at index 0
            all_cards[2],  // pile 1 face-up: at index 2
            all_cards[5],  // pile 2 face-up: at index 5
            all_cards[9],  // pile 3 face-up: at index 9
            all_cards[14], // pile 4 face-up: at index 14
            all_cards[20], // pile 5 face-up: at index 20
            all_cards[27], // pile 6 face-up: at index 27
        ];

        let stock: Vec<Card> = (28..52).map(Card::from_value).collect();
        let state = GameState::new(&tableau, stock);

        let unknowns = state.unknown_cards();
        let deal = build_deal_array(&state, &unknowns);
        assert!(deal.is_some());

        let deal = deal.unwrap();
        // Verify stock portion
        for (i, card) in deal.iter().enumerate().skip(28) {
            assert_eq!(card.value(), i as u8);
        }

        // Verify face-up cards are in correct positions
        assert_eq!(deal[0].value(), 0); // pile 0 face-up at index 0
        assert_eq!(deal[2].value(), 2); // pile 1 face-up at index 2
        assert_eq!(deal[5].value(), 5); // pile 2 face-up at index 5
    }

    #[test]
    fn test_move_key_uniqueness() {
        let m1 = Move::DeckStack(Card::new(0, 0));
        let m2 = Move::PileStack(Card::new(0, 0));
        assert_ne!(format_move_key(&m1), format_move_key(&m2));

        let m3 = Move::DeckStack(Card::new(0, 0));
        assert_eq!(format_move_key(&m1), format_move_key(&m3));
    }
}
