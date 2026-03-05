use std::collections::HashMap;
use std::io::Write;

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

    /// Check if a move is valid from the player's actual visible game state.
    /// Filters out moves that involve cards not actually accessible.
    fn is_move_valid_for_player(m: &Move, state: &GameState) -> bool {
        match m {
            Move::DeckStack(card) | Move::DeckPile(card) => {
                // Card must be the current waste top card
                state
                    .current_waste_card()
                    .is_some_and(|w| w.value() == card.value())
            }
            Move::PileStack(card) => {
                // Card must be face-up on top of a tableau pile AND foundation must be ready
                // foundations[suit] stores count: 0=empty, 1=A placed, etc.
                // For PileStack(card), foundation must have rank-1 already (i.e., count == rank)
                let suit = card.suit() as usize;
                let rank = card.rank();
                let foundation_ready = state.foundations[suit] == rank;
                foundation_ready
                    && state
                        .tableau
                        .iter()
                        .any(|pile| pile.top().is_some_and(|top| top.value() == card.value()))
            }
            Move::StackPile(card) => {
                // Card must be on top of a foundation
                let suit = card.suit() as usize;
                let rank = card.rank();
                state.foundations[suit] > 0 && state.foundations[suit] - 1 == rank
            }
            Move::Reveal(card) => {
                // Card must be face-up on a tableau pile (pile-to-pile move)
                state
                    .tableau
                    .iter()
                    .any(|pile| pile.face_up.iter().any(|c| c.value() == card.value()))
            }
        }
    }

    /// Get a move recommendation for the current game state.
    ///
    /// Runs PIMC: for each determinization, builds a Solitaire with
    /// unknown cards randomly assigned, runs pick_moves, and takes
    /// the first recommended move. Majority vote across determinizations.
    pub fn recommend(&self, state: &GameState, call_id: u32) -> Option<Recommendation> {
        let unknowns = state.unknown_cards();
        if unknowns.is_empty() && state.waste.is_empty() && state.foundation_total() == 52 {
            return None; // game won
        }

        let mut vote_map: HashMap<String, (u32, Option<Move>)> = HashMap::new();
        let mut draw_votes: u32 = 0;
        let mut no_move_count: u32 = 0;
        let mut invalid_count: u32 = 0;
        let total = self.config.n_determinizations;

        // Open debug log (append mode)
        let mut log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/assistant_debug.log")
            .ok();

        if let Some(ref mut f) = log {
            let _ = writeln!(
                f,
                "\n=== recommend() call_id={} move_count={} ===",
                call_id,
                state.move_count()
            );
            let _ = writeln!(
                f,
                "Waste top: {:?}",
                state.current_waste_card().map(format_card)
            );
            for (i, pile) in state.tableau.iter().enumerate() {
                let tops: Vec<String> = pile.face_up.iter().map(|c| format_card(*c)).collect();
                let _ = writeln!(
                    f,
                    "  P{}: {} down, face_up=[{}]",
                    i + 1,
                    pile.face_down,
                    tops.join(", ")
                );
            }
        }

        for d in 0..total {
            let rng_seed = (call_id as u64)
                .wrapping_mul(10000)
                .wrapping_add(state.move_count() as u64 * 100)
                .wrapping_add(d as u64);

            let result = self.run_determinization(state, &unknowns, rng_seed);

            match result {
                DeterminizationResult::Move(m) => {
                    // Validate: is this move actually playable from the real game state?
                    if Self::is_move_valid_for_player(&m, state) {
                        let key = format_move_key(&m);
                        let entry = vote_map.entry(key).or_insert((0, Some(m)));
                        entry.0 += 1;
                    } else {
                        if let Some(ref mut f) = log {
                            let _ = writeln!(
                                f,
                                "  det {}: INVALID move {:?} (card not visible)",
                                d,
                                format_move_key(&m)
                            );
                        }
                        invalid_count += 1;
                    }
                }
                DeterminizationResult::Draw => {
                    draw_votes += 1;
                }
                DeterminizationResult::NoMove => {
                    no_move_count += 1;
                }
            }
        }

        let effective_total = total - no_move_count;

        if let Some(ref mut f) = log {
            let _ = writeln!(
                f,
                "Votes: {:?}",
                vote_map
                    .iter()
                    .map(|(k, (v, _))| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
            );
            let _ = writeln!(
                f,
                "Draw={} NoMove={} Invalid={}",
                draw_votes, no_move_count, invalid_count
            );
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
                if let Some(ref mut f) = log {
                    let _ = writeln!(
                        f,
                        ">>> Recommend: {} [{}/{}]",
                        desc, move_votes, effective_total
                    );
                }
                return Some(Recommendation {
                    description: desc,
                    lonelybot_move,
                    is_draw: false,
                    votes: move_votes,
                    total: effective_total,
                });
            }
        }

        if draw_votes > 0 {
            if let Some(ref mut f) = log {
                let _ = writeln!(
                    f,
                    ">>> Recommend: Draw [{}/{}]",
                    draw_votes, effective_total
                );
            }
            return Some(Recommendation {
                description: "Draw next card".to_string(),
                lonelybot_move: None,
                is_draw: true,
                votes: draw_votes,
                total: effective_total,
            });
        }

        if let Some(ref mut f) = log {
            let _ = writeln!(f, ">>> No recommendation (all NoMove/Invalid)");
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
        // P0.5 fix: Do NOT call shuffle_all — it corrupts known face-up anchors
        // stored in Hidden. The unknown cards were already randomly assigned when
        // building the deal array (shuffled_unknowns). pick_moves handles its
        // own internal randomization.

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
        let start = pile_idx * (pile_idx + 1) / 2;

        // For the initial deal, we need:
        // - positions start..start+pile_idx: face-down cards (bottom-to-top in hidden pile)
        // - position start+pile_idx: initial face-up card (anchor)

        let face_down_count = pile_idx; // initial face-down count for pile i

        // Collect revealed cards for this pile (in reveal order = top-first)
        let revealed_for_pile: Vec<Card> = state
            .revealed_hidden
            .iter()
            .filter(|(p, _)| *p == pile_idx as u8)
            .map(|(_, c)| *c)
            .collect();

        // Hidden pile stores bottom-to-top. Reveals happen top-first.
        // So first-revealed card was at the TOP of hidden pile = highest index.
        // Place revealed cards in reverse: first-revealed goes to highest index.
        let n_revealed = revealed_for_pile.len().min(face_down_count);
        for (i, card) in revealed_for_pile.iter().take(n_revealed).enumerate() {
            // First revealed (i=0) goes to start + face_down_count - 1 (top of hidden)
            // Second revealed (i=1) goes to start + face_down_count - 2, etc.
            deal[start + face_down_count - 1 - i] = *card;
        }

        // Fill remaining face-down positions (bottom of hidden pile) with unknowns
        let remaining_unknown_slots = face_down_count - n_revealed;
        for i in 0..remaining_unknown_slots {
            if unknown_idx >= shuffled_unknowns.len() {
                return None;
            }
            deal[start + i] = shuffled_unknowns[unknown_idx];
            unknown_idx += 1;
        }

        // Anchor slot: always use initial_face_up (P0.4 fix)
        // This is correct even if the card has since been moved away,
        // because replay_moves will recreate the actual current state.
        deal[start + face_down_count] = state.initial_face_up[pile_idx];
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
                // In Lonelybot, pile-to-pile is Move::Reveal(card) — it moves
                // the card (and sub-stack) from one pile to another, revealing
                // the hidden card below on the source pile.
                let m = Move::Reveal(*card);
                game.do_move(&m);
            }
            crate::game_state::GameMove::FoundationToPile(card, _pile) => {
                let m = Move::StackPile(*card);
                game.do_move(&m);
            }
            crate::game_state::GameMove::Draw => {
                // Lonelybot's deal_once has a two-step reset: when all cards are
                // in waste (offset == len), the first deal_once resets offset to 0
                // (no current waste card), and a second deal_once advances to the
                // first card. The assistant's draw model skips the empty reset state
                // and goes directly to the next card. So after deal_once, if we
                // landed on the reset state, call deal_once again.
                game.get_deck_mut().deal_once();
                let deck = game.get_deck();
                if deck.get_offset() == 0 && !deck.get_deck().is_empty() {
                    game.get_deck_mut().deal_once();
                }
            }
            crate::game_state::GameMove::Reveal(_card, _pile) => {
                // P0.2 fix: Skip Reveal in replay. Card reveals happen automatically
                // in Lonelybot when the covering card is moved away (via PileToPile
                // which maps to Move::Reveal). The GameMove::Reveal is the app's
                // explicit tracking of the user flipping a face-down card — Lonelybot
                // handles this internally.
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
        Move::Reveal(card) => {
            let pile_num = find_card_pile(state, *card)
                .map(|p| format!("pile {}", p + 1))
                .unwrap_or_else(|| "tableau".to_string());
            format!(
                "Move {} from {} → another pile (reveal hidden)",
                format_card(*card),
                pile_num
            )
        }
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
