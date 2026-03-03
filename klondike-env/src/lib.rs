use pyo3::prelude::*;
use pyo3::types::PyDict;

use rand::rngs::SmallRng;
use rand::SeedableRng;

use lonelybot::card::{Card, N_CARDS, N_RANKS, N_SUITS};
use lonelybot::deck::N_PILES;
use lonelybot::engine::{Move, MoveVec, Solitaire, UndoInfo};
use lonelybot::hop_solver::hop_solve_game;
use lonelybot::mcts_solver::pick_moves;
use lonelybot::pruning::PruneInfo;
use lonelybot::shuffler::default_shuffle;
use lonelybot::tracking::DefaultTerminateSignal;

/// Convert a Move enum to (type_id, card_value)
fn move_to_tuple(m: &Move) -> (u8, u8) {
    match m {
        Move::DeckStack(c) => (0, c.value()),
        Move::PileStack(c) => (1, c.value()),
        Move::DeckPile(c) => (2, c.value()),
        Move::StackPile(c) => (3, c.value()),
        Move::Reveal(c) => (4, c.value()),
    }
}

/// Convert (type_id, card_value) back to a Move
fn tuple_to_move(move_type: u8, card_idx: u8) -> Option<Move> {
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

#[pyclass]
struct KlondikeGame {
    game: Solitaire,
    draw_step: u8,
    seed: u64,
    undo_stack: Vec<(Move, UndoInfo)>,
}

#[pymethods]
impl KlondikeGame {
    #[new]
    #[pyo3(signature = (seed, draw_step=1))]
    fn new(seed: u64, draw_step: u8) -> Self {
        let cards = default_shuffle(seed);
        let game = Solitaire::new(&cards, draw_step);
        Self {
            game,
            draw_step,
            seed,
            undo_stack: Vec::new(),
        }
    }

    /// Reset the game with a new seed
    fn reset(&mut self, seed: u64) {
        let cards = default_shuffle(seed);
        self.game = Solitaire::new(&cards, self.draw_step);
        self.seed = seed;
        self.undo_stack.clear();
    }

    /// Get legal moves as list of (move_type, card_value) tuples.
    /// move_type: 0=DeckStack, 1=PileStack, 2=DeckPile, 3=StackPile, 4=Reveal
    /// card_value: rank * 4 + suit (rank 0-12, suit 0-3)
    fn get_legal_moves(&self) -> Vec<(u8, u8)> {
        let prune = PruneInfo::default().prune_moves::<true>(&self.game);
        let moves: MoveVec = self.game.list_moves::<true>(&prune);
        moves.iter().map(|m| move_to_tuple(m)).collect()
    }

    /// Get legal moves WITHOUT dominance pruning (all legal moves).
    fn get_all_legal_moves(&self) -> Vec<(u8, u8)> {
        let moves: MoveVec = self.game.list_moves::<false>(&Default::default());
        moves.iter().map(|m| move_to_tuple(m)).collect()
    }

    /// Execute a move. Returns true if successful, false if move is not legal.
    fn do_move(&mut self, move_type: u8, card_idx: u8) -> bool {
        let Some(m) = tuple_to_move(move_type, card_idx) else {
            return false;
        };

        // Verify the move is legal (check against unpruned moves)
        let moves: MoveVec = self.game.list_moves::<false>(&Default::default());
        if !moves.contains(&m) {
            return false;
        }

        let undo = self.game.do_move(&m);
        self.undo_stack.push((m, undo));
        true
    }

    /// Undo the last move. Returns true if successful, false if no moves to undo.
    fn undo_move(&mut self) -> bool {
        if let Some((m, undo)) = self.undo_stack.pop() {
            self.game.undo_move(&m, &undo);
            true
        } else {
            false
        }
    }

    /// Check if the game is won (all cards on foundations).
    fn is_win(&self) -> bool {
        self.game.is_win()
    }

    /// Check if the game is a guaranteed win (all cards face up, deck nearly empty).
    fn is_sure_win(&self) -> bool {
        self.game.is_sure_win()
    }

    /// Get foundation state as [rank_for_suit_0, ..., rank_for_suit_3].
    /// Each value is the number of cards stacked for that suit (0 = empty, 13 = complete).
    fn get_foundation(&self) -> Vec<u8> {
        let stack = self.game.get_stack();
        (0..N_SUITS).map(|s| stack.get(s)).collect()
    }

    /// Get visible pile cards for each of the 7 piles.
    /// Returns list of 7 piles, each pile is list of (rank, suit, is_visible).
    /// Hidden cards are returned as (255, 255, false) placeholders.
    fn get_piles(&self) -> Vec<Vec<(u8, u8, bool)>> {
        let hidden = self.game.get_hidden();
        let visible_piles = self.game.get_visible_piles();

        let mut result = Vec::with_capacity(N_PILES as usize);
        for pos in 0..N_PILES {
            let mut pile = Vec::new();

            // Add hidden card placeholders (n_hidden - 1, since the topmost hidden
            // card becomes the first visible card when revealed)
            let n_hidden = hidden.len(pos);
            let n_hidden_below = if n_hidden > 0 { n_hidden - 1 } else { 0 };
            for _ in 0..n_hidden_below {
                pile.push((255u8, 255u8, false));
            }

            // Add visible cards
            for card in &visible_piles[pos as usize] {
                pile.push((card.rank(), card.suit(), true));
            }

            result.push(pile);
        }
        result
    }

    /// Get the number of hidden (face-down) cards under each pile.
    /// Returns [count_pile_0, ..., count_pile_6].
    /// Note: hidden.len(pos) includes the topmost hidden card (which is the card
    /// that would be revealed next). The actual face-down count is n_hidden - 1
    /// for piles with visible cards, or n_hidden for empty piles.
    fn get_hidden_counts(&self) -> Vec<u8> {
        let hidden = self.game.get_hidden();
        (0..N_PILES).map(|pos| {
            let n = hidden.len(pos);
            if n > 0 { n - 1 } else { 0 }
        }).collect()
    }

    /// Get deck info: waste cards, stock size, draw_step, offset.
    fn get_deck_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let deck = self.game.get_deck();
        let dict = PyDict::new(py);

        // Waste cards (face-up discard pile)
        let waste: Vec<(u8, u8)> = deck
            .get_waste()
            .iter()
            .map(|c| (c.rank(), c.suit()))
            .collect();
        dict.set_item("waste", waste)?;

        // Stock cards remaining (face-down draw pile)
        let stock: Vec<(u8, u8)> = deck
            .get_deck()
            .iter()
            .map(|c| (c.rank(), c.suit()))
            .collect();
        dict.set_item("stock_count", stock.len())?;

        dict.set_item("draw_step", deck.draw_step())?;
        dict.set_item("offset", deck.get_offset())?;
        dict.set_item("total_cards", deck.len())?;

        Ok(dict)
    }

    /// Get visible card indices from the visible_mask bitmask.
    /// Returns list of card values (rank * 4 + suit) for all face-up cards on tableau.
    fn get_visible_cards(&self) -> Vec<u8> {
        let mut mask = self.game.get_visible_mask();
        let mut cards = Vec::new();
        while mask != 0 {
            let card = Card::from_mask(&mask);
            cards.push(card.value());
            // Clear the lowest set bit
            mask &= mask.wrapping_sub(1);
        }
        cards
    }

    /// Clone the game state (including undo stack).
    fn clone_game(&self) -> KlondikeGame {
        KlondikeGame {
            game: self.game.clone(),
            draw_step: self.draw_step,
            seed: self.seed,
            undo_stack: self.undo_stack.clone(),
        }
    }

    /// Shuffle hidden cards for PIMC determinization.
    /// Uses shuffle_all (NOT shuffle) — removes all oracle information.
    fn shuffle_hidden(&mut self, rng_seed: u64) {
        let mut rng = SmallRng::seed_from_u64(rng_seed);
        self.game.get_hidden_mut().shuffle_all(&mut rng);
    }

    /// Run HOP solver for each legal move.
    /// n_determinizations: number of random worlds to sample per move
    /// node_limit: max search nodes per determinized world
    /// Returns dict mapping (move_type, card_value) -> {"wins": int, "skips": int, "played": int}
    fn hop_solve<'py>(
        &self,
        py: Python<'py>,
        n_determinizations: usize,
        node_limit: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let prune_info = PruneInfo::default();
        let prune = prune_info.prune_moves::<true>(&self.game);
        let moves: MoveVec = self.game.list_moves::<true>(&prune);
        let signal = DefaultTerminateSignal;

        let dict = PyDict::new(py);
        for m in &moves {
            let mut rng = SmallRng::seed_from_u64(self.seed.wrapping_mul(31).wrapping_add(
                match m {
                    Move::DeckStack(c) | Move::PileStack(c) | Move::DeckPile(c)
                    | Move::StackPile(c) | Move::Reveal(c) => c.value() as u64,
                },
            ));
            let (wins, skips, played) = hop_solve_game(
                &self.game,
                m,
                &mut rng,
                n_determinizations,
                node_limit,
                &signal,
                &prune_info,
            );
            let key = move_to_tuple(m);
            let result = PyDict::new(py);
            result.set_item("wins", wins)?;
            result.set_item("skips", skips)?;
            result.set_item("played", played)?;
            dict.set_item(key, result)?;
        }
        Ok(dict)
    }

    /// Play one "HOP turn" matching the binary's do_hop() approach:
    /// 1. Clone game, shuffle_all hidden cards (determinization)
    /// 2. Run pick_moves (UCB1 bandit + traverse + HOP) on the shuffled copy
    /// 3. Play the returned move sequence on the REAL game
    /// Returns list of (move_type, card_value) played, or empty list if no moves found.
    fn play_hop_turn(
        &mut self,
        rng_seed: u64,
        n_times: usize,
        node_limit: usize,
    ) -> Vec<(u8, u8)> {
        let mut rng = SmallRng::seed_from_u64(rng_seed);
        let signal = DefaultTerminateSignal;

        // Clone and shuffle for determinization (matching do_hop)
        let mut gg = self.game.clone();
        gg.get_hidden_mut().shuffle_all(&mut rng);

        // pick_moves: UCB1 + traverse + hop_solve_game
        let best = pick_moves(&mut gg, &mut rng, n_times, node_limit, &signal);

        let Some(moves) = best else {
            return Vec::new();
        };

        // Play the move sequence on the REAL game
        let mut played = Vec::new();
        for m in &moves {
            let undo = self.game.do_move(m);
            self.undo_stack.push((*m, undo));
            played.push(move_to_tuple(m));
        }
        played
    }

    /// Get encoded game state as u64 (for transposition tables, cycle detection).
    fn encode(&self) -> u64 {
        self.game.encode()
    }

    /// Get complete game state as a dict.
    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("foundation", self.get_foundation())?;
        dict.set_item("piles", self.get_piles())?;
        dict.set_item("hidden_counts", self.get_hidden_counts())?;
        dict.set_item("deck_info", self.get_deck_info(py)?)?;
        dict.set_item("visible_cards", self.get_visible_cards())?;
        dict.set_item("is_win", self.is_win())?;
        dict.set_item("is_sure_win", self.is_sure_win())?;
        dict.set_item("seed", self.seed)?;
        dict.set_item("n_moves", self.undo_stack.len())?;
        Ok(dict)
    }

    /// Get the number of moves made so far.
    fn move_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Get the seed used to create this game.
    fn get_seed(&self) -> u64 {
        self.seed
    }

    /// Check if the game state is internally consistent.
    fn is_valid(&self) -> bool {
        self.game.is_valid()
    }

    /// Get total face-down cards remaining.
    fn total_hidden_cards(&self) -> u8 {
        self.game.get_hidden().total_down_cards()
    }

    fn __repr__(&self) -> String {
        let foundation = self.get_foundation();
        let moves = self.undo_stack.len();
        format!(
            "KlondikeGame(seed={}, draw_step={}, moves={}, foundation={:?}, win={})",
            self.seed, self.draw_step, moves, foundation, self.is_win()
        )
    }
}

#[pymodule]
fn klondike_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KlondikeGame>()?;
    m.add("N_CARDS", N_CARDS)?;
    m.add("N_SUITS", N_SUITS)?;
    m.add("N_RANKS", N_RANKS)?;
    m.add("N_PILES", N_PILES)?;
    m.add("N_MOVE_TYPES", 5u8)?;
    m.add("ACTION_SPACE_SIZE", 260u16)?;
    Ok(())
}
