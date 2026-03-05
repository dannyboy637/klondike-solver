use lonelybot::card::{Card, N_CARDS};
use lonelybot::deck::N_PILES;
use lonelybot::engine::Move;

use crate::game_state::GameState;
use crate::input::{format_card, parse_card, parse_cards};
use crate::solver::{HopSolver, HopSolverConfig, Recommendation};

/// Application phase / state machine.
#[derive(Clone)]
pub enum AppPhase {
    /// Waiting for the 7 face-up tableau cards.
    Setup {},
    /// Waiting for the 24 stock cards.
    StockCycle {
        tableau_cards: [Card; N_PILES as usize],
    },
    /// Active gameplay.
    Playing {
        state: Box<GameState>,
        recommendation: Option<Recommendation>,
        solving: bool,
        awaiting_reveal: Option<u8>, // pile index waiting for card reveal
    },
    /// Game finished.
    GameOver { won: bool, move_count: usize },
}

/// Main application state.
pub struct App {
    pub phase: AppPhase,
    pub input_buffer: String,
    pub error_message: Option<String>,
    pub should_quit: bool,
    pub session_wins: u32,
    pub session_losses: u32,
    solver: HopSolver,
}

impl App {
    pub fn new() -> Self {
        let config = HopSolverConfig::default();
        Self {
            phase: AppPhase::Setup {},
            input_buffer: String::new(),
            error_message: None,
            should_quit: false,
            session_wins: 0,
            session_losses: 0,
            solver: HopSolver::new(config),
        }
    }

    /// Handle a character input.
    pub fn on_char(&mut self, c: char) {
        self.error_message = None;
        self.input_buffer.push(c);
    }

    /// Handle backspace.
    pub fn on_backspace(&mut self) {
        self.error_message = None;
        self.input_buffer.pop();
    }

    /// Handle Enter key.
    pub fn on_enter(&mut self) {
        self.error_message = None;
        let input = self.input_buffer.trim().to_string();

        match &self.phase {
            AppPhase::Setup {} => self.handle_setup_input(&input),
            AppPhase::StockCycle { .. } => self.handle_stock_cycle_input(&input),
            AppPhase::Playing { .. } => self.handle_playing_input(&input),
            AppPhase::GameOver { .. } => self.handle_game_over_input(&input),
        }
    }

    /// Handle special keys during playing phase.
    pub fn on_key(&mut self, key: char) {
        match &self.phase {
            AppPhase::Playing {
                awaiting_reveal, ..
            } => {
                if awaiting_reveal.is_some() {
                    // In reveal mode, only accept card input via the input buffer
                    return;
                }
                match key {
                    'q' => self.should_quit = true,
                    'u' => self.handle_undo(),
                    'r' => self.compute_recommendation(),
                    _ => {}
                }
            }
            AppPhase::GameOver { .. } => match key {
                'q' => self.should_quit = true,
                'n' => self.start_new_game(),
                _ => {}
            },
            _ => {
                if key == 'q' {
                    self.should_quit = true;
                }
            }
        }
    }

    fn handle_setup_input(&mut self, input: &str) {
        if input.is_empty() {
            return;
        }

        match parse_cards(input) {
            Ok(cards) => {
                if cards.len() != N_PILES as usize {
                    self.error_message = Some(format!("Need exactly 7 cards, got {}", cards.len()));
                    return;
                }
                let tableau: [Card; N_PILES as usize] = cards.try_into().unwrap();
                self.phase = AppPhase::StockCycle {
                    tableau_cards: tableau,
                };
                self.input_buffer.clear();
            }
            Err(e) => {
                self.error_message = Some(e);
            }
        }
    }

    fn handle_stock_cycle_input(&mut self, input: &str) {
        if input.is_empty() {
            return;
        }

        let tableau_cards = match &self.phase {
            AppPhase::StockCycle { tableau_cards } => *tableau_cards,
            _ => return,
        };

        match parse_cards(input) {
            Ok(stock_cards) => {
                if stock_cards.len() != 24 {
                    self.error_message = Some(format!(
                        "Need exactly 24 stock cards, got {}",
                        stock_cards.len()
                    ));
                    return;
                }

                // Check for duplicates with tableau cards
                let mut seen = [false; N_CARDS as usize];
                for card in &tableau_cards {
                    seen[card.value() as usize] = true;
                }
                for card in &stock_cards {
                    let val = card.value() as usize;
                    if seen[val] {
                        self.error_message =
                            Some(format!("Duplicate card: {}", format_card(*card)));
                        return;
                    }
                    seen[val] = true;
                }

                let state = GameState::new(&tableau_cards, stock_cards);
                self.phase = AppPhase::Playing {
                    state: Box::new(state),
                    recommendation: None,
                    solving: false,
                    awaiting_reveal: None,
                };
                self.input_buffer.clear();
                // Auto-compute first recommendation
                self.compute_recommendation();
            }
            Err(e) => {
                self.error_message = Some(e);
            }
        }
    }

    fn handle_playing_input(&mut self, input: &str) {
        // Check if we're waiting for a card reveal
        let awaiting = match &self.phase {
            AppPhase::Playing {
                awaiting_reveal, ..
            } => *awaiting_reveal,
            _ => return,
        };

        if let Some(pile_idx) = awaiting {
            // Parse the revealed card
            if input.is_empty() {
                return;
            }
            match parse_card(input) {
                Ok(card) => {
                    if let AppPhase::Playing { state, .. } = &self.phase {
                        if state.is_known(card) {
                            self.error_message =
                                Some(format!("{} is already known/placed", format_card(card)));
                            return;
                        }
                    }

                    if let AppPhase::Playing {
                        state,
                        awaiting_reveal,
                        ..
                    } = &mut self.phase
                    {
                        state.reveal_card(card, pile_idx);
                        *awaiting_reveal = None;
                    }
                    self.input_buffer.clear();
                    // Check if another pile needs reveal (shouldn't happen but be safe)
                    self.check_for_reveals();
                    self.compute_recommendation();
                }
                Err(e) => {
                    self.error_message = Some(e);
                }
            }
            return;
        }

        // Normal playing input
        if input.is_empty() || input == "y" {
            // Confirm recommended move
            self.apply_recommendation();
            self.input_buffer.clear();
            return;
        }

        // Check if it's a card code (2 chars) — could be a reveal
        if input.len() == 2 {
            if let Ok(card) = parse_card(input) {
                // User is manually inputting a revealed card
                if let AppPhase::Playing { state, .. } = &self.phase {
                    if let Some(pile_idx) = state.find_pile_needing_reveal() {
                        if state.is_known(card) {
                            self.error_message =
                                Some(format!("{} is already known/placed", format_card(card)));
                            return;
                        }
                        if let AppPhase::Playing { state, .. } = &mut self.phase {
                            state.reveal_card(card, pile_idx);
                        }
                        self.input_buffer.clear();
                        self.compute_recommendation();
                        return;
                    }
                }
            }
        }

        self.input_buffer.clear();
    }

    fn handle_game_over_input(&mut self, input: &str) {
        match input {
            "q" => self.should_quit = true,
            "n" => self.start_new_game(),
            _ => {}
        }
        self.input_buffer.clear();
    }

    /// Apply the current recommendation to the game state.
    fn apply_recommendation(&mut self) {
        let (rec_move, is_draw) = match &self.phase {
            AppPhase::Playing { recommendation, .. } => {
                if let Some(rec) = recommendation {
                    (rec.lonelybot_move, rec.is_draw)
                } else {
                    self.error_message =
                        Some("No recommendation yet. Press 'r' to compute.".to_string());
                    return;
                }
            }
            _ => return,
        };

        if is_draw {
            if let AppPhase::Playing {
                state,
                recommendation,
                ..
            } = &mut self.phase
            {
                state.draw();
                *recommendation = None;
            }
            self.compute_recommendation();
            return;
        }

        if let Some(m) = rec_move {
            self.apply_lonelybot_move(m);
        }
    }

    /// Apply a lonelybot Move to our game state.
    fn apply_lonelybot_move(&mut self, m: Move) {
        let state = match &mut self.phase {
            AppPhase::Playing { state, .. } => state,
            _ => return,
        };

        match m {
            Move::DeckStack(card) => {
                state.waste_to_foundation(card);
            }
            Move::PileStack(card) => {
                if let Some(pile_idx) = find_card_pile(state, card) {
                    state.pile_to_foundation(card, pile_idx);
                }
            }
            Move::DeckPile(card) => {
                // Find a valid target pile for this card
                if let Some(target) = find_target_pile(state, card) {
                    state.waste_to_pile(card, target);
                }
            }
            Move::StackPile(card) => {
                if let Some(target) = find_target_pile(state, card) {
                    state.foundation_to_pile(card, target);
                }
            }
            Move::Reveal(_card) => {
                // Reveal is handled by awaiting user input for the actual card
            }
        }

        // Clear recommendation
        if let AppPhase::Playing { recommendation, .. } = &mut self.phase {
            *recommendation = None;
        }

        // Check for game won
        if let AppPhase::Playing { state, .. } = &self.phase {
            if state.is_won() {
                let move_count = state.move_count();
                self.phase = AppPhase::GameOver {
                    won: true,
                    move_count,
                };
                self.session_wins += 1;
                return;
            }
        }

        // Check if a pile needs reveal
        self.check_for_reveals();
        if matches!(
            &self.phase,
            AppPhase::Playing {
                awaiting_reveal: Some(_),
                ..
            }
        ) {
            return; // Wait for reveal input
        }

        // Auto-compute next recommendation
        self.compute_recommendation();
    }

    /// Check if any pile needs a face-down card revealed.
    fn check_for_reveals(&mut self) {
        if let AppPhase::Playing {
            state,
            awaiting_reveal,
            ..
        } = &mut self.phase
        {
            if let Some(pile_idx) = state.find_pile_needing_reveal() {
                *awaiting_reveal = Some(pile_idx);
            }
        }
    }

    /// Compute a recommendation from the solver (blocking).
    fn compute_recommendation(&mut self) {
        let state = match &self.phase {
            AppPhase::Playing { state, .. } => (**state).clone(),
            _ => return,
        };

        // Mark as solving
        if let AppPhase::Playing { solving, .. } = &mut self.phase {
            *solving = true;
        }

        let rec = self.solver.recommend(&state);

        if let AppPhase::Playing {
            recommendation,
            solving,
            ..
        } = &mut self.phase
        {
            *recommendation = rec;
            *solving = false;
        }

        // If no recommendation and no reveal pending, game might be stuck
        if let AppPhase::Playing {
            recommendation,
            awaiting_reveal,
            state,
            ..
        } = &self.phase
        {
            if recommendation.is_none() && awaiting_reveal.is_none() && !state.is_won() {
                let move_count = state.move_count();
                self.phase = AppPhase::GameOver {
                    won: false,
                    move_count,
                };
                self.session_losses += 1;
            }
        }
    }

    fn handle_undo(&mut self) {
        self.error_message = Some("Undo not yet implemented".to_string());
    }

    fn start_new_game(&mut self) {
        self.phase = AppPhase::Setup {};
        self.input_buffer.clear();
        self.error_message = None;
    }
}

/// Find which pile a card is on.
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

/// Find a valid target pile for placing a card.
/// Returns the pile index where the card can legally go.
fn find_target_pile(state: &GameState, card: Card) -> Option<u8> {
    let rank = card.rank();
    let is_red = card.suit() < 2;

    // King goes to an empty pile
    if rank == 12 {
        for (i, pile) in state.tableau.iter().enumerate() {
            if pile.is_empty() {
                return Some(i as u8);
            }
        }
        return None;
    }

    // Non-king: find a pile whose top card is rank+1 and opposite color
    for (i, pile) in state.tableau.iter().enumerate() {
        if let Some(top) = pile.top() {
            let top_red = top.suit() < 2;
            if top.rank() == rank + 1 && top_red != is_red {
                return Some(i as u8);
            }
        }
    }
    None
}
