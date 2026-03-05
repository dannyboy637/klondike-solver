use lonelybot::card::{Card, N_CARDS, N_SUITS};
use lonelybot::deck::N_PILES;

use crate::input::format_card;

#[allow(dead_code)]
/// A single tableau pile as seen by the player.
#[derive(Clone, Debug)]
pub struct Pile {
    /// Number of face-down cards (unknown identity).
    pub face_down: u8,
    /// Known face-up cards, bottom to top.
    pub face_up: Vec<Card>,
}

/// A move in the game, from the player's perspective.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum GameMove {
    /// Move card from waste to foundation.
    WasteToFoundation(Card),
    /// Move card from waste to a tableau pile.
    WasteToPile(Card, u8),
    /// Move card (top of sub-stack) from one pile to another.
    PileToPile(Card, u8, u8),
    /// Move card from tableau to foundation.
    PileToFoundation(Card, u8),
    /// Move card from foundation back to tableau.
    FoundationToPile(Card, u8),
    /// Draw next card from stock (advance waste position).
    Draw,
    /// Reveal a face-down card on a pile (automatic after clearing face-up cards).
    Reveal(Card, u8),
}

/// Partial-information game state tracked by the assistant.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct GameState {
    /// Foundation top rank per suit (0=empty, 1..13=A..K).
    pub foundations: [u8; N_SUITS as usize],
    /// Tableau piles.
    pub tableau: [Pile; N_PILES as usize],
    /// Full known stock order (from initial cycle). Indices go left-to-right
    /// as the user sees them when cycling: first element is the first card drawn.
    pub stock_order: Vec<Card>,
    /// Cards currently in the waste pile (subset of stock_order).
    pub waste: Vec<Card>,
    /// Move history for undo.
    pub move_history: Vec<GameMove>,
    /// Set of all known card locations (value → true if we know where it is).
    pub known_cards: [bool; N_CARDS as usize],
    /// Cards that were originally face-down but have been revealed during play.
    /// Maps (pile_index, original_position_from_bottom) → Card.
    pub revealed_hidden: Vec<(u8, Card)>,
    /// The initial face-up cards dealt to each tableau pile at game start.
    /// Used by the solver to reconstruct the deal array even after the card has moved.
    pub initial_face_up: [Card; N_PILES as usize],
    /// Current waste index. Points to the current top-of-waste card.
    /// After full cycling, starts at waste.len()-1 (last card = stock[23]).
    /// Draw advances: wraps to 0 on overflow, matching Lonelybot's deal_once.
    pub waste_idx: usize,
}

impl Pile {
    pub fn new(face_down: u8, face_up_card: Card) -> Self {
        Self {
            face_down,
            face_up: vec![face_up_card],
        }
    }

    #[allow(dead_code)]
    pub fn empty() -> Self {
        Self {
            face_down: 0,
            face_up: Vec::new(),
        }
    }

    pub fn top(&self) -> Option<&Card> {
        self.face_up.last()
    }

    pub fn is_empty(&self) -> bool {
        self.face_down == 0 && self.face_up.is_empty()
    }

    /// Total visible + hidden count.
    pub fn total_cards(&self) -> usize {
        self.face_down as usize + self.face_up.len()
    }
}

impl GameState {
    /// Create a new game state from the 7 initial face-up tableau cards
    /// and the 24 stock cards (in draw order).
    pub fn new(tableau_cards: &[Card; N_PILES as usize], stock_cards: Vec<Card>) -> Self {
        let mut known_cards = [false; N_CARDS as usize];

        // Mark tableau face-up cards as known
        let tableau = std::array::from_fn(|i| {
            let card = tableau_cards[i];
            known_cards[card.value() as usize] = true;
            Pile::new(i as u8, card) // pile i has i face-down cards
        });

        // Mark stock cards as known
        for card in &stock_cards {
            known_cards[card.value() as usize] = true;
        }

        // Initially waste is empty, stock has all 24 cards
        // After the user cycles through the stock in the real game,
        // all 24 cards are in the waste (draw-1).
        // We model this as: waste contains all stock cards (cycled),
        // with waste_idx pointing to current accessible card.
        let waste = stock_cards.clone();
        let waste_idx = if waste.is_empty() { 0 } else { waste.len() - 1 };

        Self {
            foundations: [0; N_SUITS as usize],
            tableau,
            stock_order: stock_cards,
            waste,
            move_history: Vec::new(),
            known_cards,
            revealed_hidden: Vec::new(),
            initial_face_up: *tableau_cards,
            waste_idx,
        }
    }

    /// Get the current top waste card (the one the player can interact with).
    pub fn current_waste_card(&self) -> Option<Card> {
        if self.waste.is_empty() {
            return None;
        }
        Some(self.waste[self.waste_idx])
    }

    /// Draw: advance to the next stock card.
    /// In draw-1 after full cycling, this matches Lonelybot's deal_once:
    /// offset cycles 0, 1, 2, ..., len-1, then wraps back to 0.
    pub fn draw(&mut self) {
        if self.waste.is_empty() {
            return;
        }

        // Advance waste_idx, wrapping around (matches Lonelybot deal_once cycling)
        self.waste_idx = (self.waste_idx + 1) % self.waste.len();

        self.move_history.push(GameMove::Draw);
    }

    /// Remove the current waste top card, adjusting waste_idx.
    /// After removal, waste_idx points to the card that was "below" the removed one
    /// (the previous card in cycling order). If waste becomes empty, waste_idx = 0.
    fn remove_waste_card(&mut self, card: Card) {
        if let Some(pos) = self.waste.iter().position(|c| c.value() == card.value()) {
            self.waste.remove(pos);
            if self.waste.is_empty() {
                self.waste_idx = 0;
            } else if pos <= self.waste_idx {
                // Card was at or before current index — shift back, wrapping
                if self.waste_idx == 0 {
                    self.waste_idx = self.waste.len() - 1;
                } else {
                    self.waste_idx -= 1;
                }
            }
            // If pos > waste_idx, no adjustment needed
        }
    }

    /// Apply a move: waste card to foundation.
    pub fn waste_to_foundation(&mut self, card: Card) {
        let suit = card.suit() as usize;
        self.foundations[suit] = card.rank() + 1;
        self.remove_waste_card(card);
        self.move_history.push(GameMove::WasteToFoundation(card));
    }

    /// Apply a move: waste card to tableau pile.
    pub fn waste_to_pile(&mut self, card: Card, pile_idx: u8) {
        self.remove_waste_card(card);
        self.tableau[pile_idx as usize].face_up.push(card);
        self.move_history
            .push(GameMove::WasteToPile(card, pile_idx));
    }

    /// Apply a move: tableau card to foundation.
    pub fn pile_to_foundation(&mut self, card: Card, from_pile: u8) {
        let pile = &mut self.tableau[from_pile as usize];
        // Remove top card
        pile.face_up.pop();
        let suit = card.suit() as usize;
        self.foundations[suit] = card.rank() + 1;
        self.move_history
            .push(GameMove::PileToFoundation(card, from_pile));
    }

    /// Apply a move: move sub-stack from one pile to another.
    /// `card` is the bottom card of the sub-stack being moved.
    pub fn pile_to_pile(&mut self, card: Card, from_pile: u8, to_pile: u8) {
        let from = &self.tableau[from_pile as usize];
        let pos = from.face_up.iter().position(|c| c.value() == card.value());
        if let Some(pos) = pos {
            let moved: Vec<Card> = self.tableau[from_pile as usize].face_up[pos..].to_vec();
            self.tableau[from_pile as usize].face_up.truncate(pos);
            self.tableau[to_pile as usize].face_up.extend(moved);
        }
        self.move_history
            .push(GameMove::PileToPile(card, from_pile, to_pile));
    }

    /// Apply a move: foundation card back to tableau.
    pub fn foundation_to_pile(&mut self, card: Card, to_pile: u8) {
        let suit = card.suit() as usize;
        self.foundations[suit] = self.foundations[suit].saturating_sub(1);
        self.tableau[to_pile as usize].face_up.push(card);
        self.move_history
            .push(GameMove::FoundationToPile(card, to_pile));
    }

    /// Reveal a hidden card on a pile. Called when a face-down card is flipped.
    pub fn reveal_card(&mut self, card: Card, pile_idx: u8) {
        let pile = &mut self.tableau[pile_idx as usize];
        if pile.face_up.is_empty() && pile.face_down > 0 {
            pile.face_down -= 1;
            pile.face_up.push(card);
            self.known_cards[card.value() as usize] = true;
            self.revealed_hidden.push((pile_idx, card));
            self.move_history.push(GameMove::Reveal(card, pile_idx));
        }
    }

    /// Check if a pile needs a reveal (face-up is empty but face-down > 0).
    pub fn pile_needs_reveal(&self, pile_idx: u8) -> bool {
        let pile = &self.tableau[pile_idx as usize];
        pile.face_up.is_empty() && pile.face_down > 0
    }

    /// Find which pile (if any) needs a card reveal after the last move.
    pub fn find_pile_needing_reveal(&self) -> Option<u8> {
        (0..N_PILES).find(|&i| self.pile_needs_reveal(i))
    }

    /// Get the list of unknown cards (not yet seen).
    pub fn unknown_cards(&self) -> Vec<Card> {
        let mut unknowns = Vec::new();
        for val in 0..N_CARDS {
            if !self.known_cards[val as usize] {
                unknowns.push(Card::from_value(val));
            }
        }
        unknowns
    }

    /// Number of unknown cards.
    pub fn unknown_count(&self) -> usize {
        self.known_cards.iter().filter(|&&k| !k).count()
    }

    /// Check if the game is won (all foundations at King).
    pub fn is_won(&self) -> bool {
        self.foundations.iter().all(|&r| r == 13)
    }

    /// Check if card is already known (placed somewhere).
    pub fn is_known(&self, card: Card) -> bool {
        self.known_cards[card.value() as usize]
    }

    /// Total cards on foundations.
    pub fn foundation_total(&self) -> u8 {
        self.foundations.iter().sum()
    }

    /// Total moves played.
    pub fn move_count(&self) -> usize {
        self.move_history.len()
    }

    /// Format a GameMove as human-readable text.
    #[allow(dead_code)]
    pub fn format_move(m: &GameMove) -> String {
        match m {
            GameMove::WasteToFoundation(c) => {
                format!(
                    "Move {} → {} foundation",
                    format_card(*c),
                    crate::input::suit_symbol(c.suit())
                )
            }
            GameMove::WasteToPile(c, p) => {
                format!("Move {} from waste → pile {}", format_card(*c), p + 1)
            }
            GameMove::PileToPile(c, from, to) => {
                format!(
                    "Move {} from pile {} → pile {}",
                    format_card(*c),
                    from + 1,
                    to + 1
                )
            }
            GameMove::PileToFoundation(c, _from) => {
                format!(
                    "Move {} → {} foundation",
                    format_card(*c),
                    crate::input::suit_symbol(c.suit())
                )
            }
            GameMove::FoundationToPile(c, p) => {
                format!("Move {} from foundation → pile {}", format_card(*c), p + 1)
            }
            GameMove::Draw => "Draw next card".to_string(),
            GameMove::Reveal(c, p) => {
                format!("Reveal {} on pile {}", format_card(*c), p + 1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_card(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    // Build non-overlapping test cards: 7 tableau + 24 stock from first 31 card values
    fn test_cards() -> ([Card; 7], Vec<Card>) {
        let tableau: [Card; 7] = std::array::from_fn(|i| Card::from_value(i as u8));
        let stock: Vec<Card> = (7..31).map(Card::from_value).collect();
        (tableau, stock)
    }

    #[test]
    fn test_new_game_state() {
        let (tableau, stock) = test_cards();
        let state = GameState::new(&tableau, stock);

        // 7 face-up + 24 stock = 31 known, 52 - 31 = 21 unknown
        assert_eq!(state.unknown_count(), 21);
        assert_eq!(state.tableau[0].face_down, 0);
        assert_eq!(state.tableau[1].face_down, 1);
        assert_eq!(state.tableau[6].face_down, 6);
        assert!(!state.is_won());
    }

    #[test]
    fn test_reveal_reduces_unknown() {
        let (tableau, stock) = test_cards();
        let mut state = GameState::new(&tableau, stock);

        let initial_unknown = state.unknown_count();
        // Value 40 is unknown (values 0-30 are known from tableau+stock)
        let new_card = Card::from_value(40);
        assert!(!state.is_known(new_card));

        // Simulate clearing pile 1's face-up card (e.g., moved to foundation)
        // so that reveal_card can work (requires face_up empty + face_down > 0)
        state.tableau[1].face_up.clear();
        assert!(state.pile_needs_reveal(1)); // pile 1 has 1 face-down

        state.reveal_card(new_card, 1);
        assert_eq!(state.unknown_count(), initial_unknown - 1);
        assert!(state.is_known(new_card));
    }

    #[test]
    fn test_waste_to_foundation() {
        let tableau = std::array::from_fn(|i| make_card(i as u8 + 1, 0));
        let stock = vec![make_card(0, 1)]; // AD
        let mut state = GameState::new(&tableau, stock);

        let card = make_card(0, 1); // AD
        state.waste_to_foundation(card);
        assert_eq!(state.foundations[1], 1); // Diamonds has A
        assert!(state.waste.is_empty());
    }

    #[test]
    fn test_pile_to_pile() {
        let tableau = std::array::from_fn(|i| make_card(i as u8 + 1, i as u8 % 4));
        let stock: Vec<Card> = vec![make_card(0, 0)];
        let mut state = GameState::new(&tableau, stock);

        // Pile 0 has 2H (rank=1, suit=0), pile 1 has 3D (rank=2, suit=1)
        // Move 2H from pile 0 to pile 1 (red 2 on black 3 — valid in real game)
        let card = make_card(1, 0); // 2H
        state.pile_to_pile(card, 0, 1);

        assert!(state.tableau[0].face_up.is_empty());
        assert_eq!(state.tableau[1].face_up.len(), 2);
    }
}
