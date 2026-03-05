//! Dump the deal for a given seed and play a HOP game with move logging.
//!
//! Usage: cargo run -p klondike-assistant --bin dump_deal --release -- [seed]

use std::collections::HashSet;

use lonelybot::card::{Card, NUMBERS, SYMBOLS};
use lonelybot::deck::N_PILES;
use lonelybot::engine::{Move, Solitaire};
use lonelybot::mcts_solver::pick_moves;
use lonelybot::shuffler::default_shuffle;
use lonelybot::tracking::DefaultTerminateSignal;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn fmt(card: Card) -> String {
    let r = card.rank() as usize;
    let s = card.suit() as usize;
    format!("{}{}", NUMBERS[r], SYMBOLS[s])
}

fn fmt_code(card: Card) -> String {
    let rank_char = match card.rank() {
        0 => 'a',
        1 => '2',
        2 => '3',
        3 => '4',
        4 => '5',
        5 => '6',
        6 => '7',
        7 => '8',
        8 => '9',
        9 => 't',
        10 => 'j',
        11 => 'q',
        12 => 'k',
        _ => '?',
    };
    let suit_char = match card.suit() {
        0 => 'h',
        1 => 'd',
        2 => 'c',
        3 => 's',
        _ => '?',
    };
    format!("{}{}", rank_char, suit_char)
}

fn fmt_move(m: &Move) -> String {
    match m {
        Move::DeckStack(c) => format!("Waste→Foundation: {}", fmt(*c)),
        Move::PileStack(c) => format!("Pile→Foundation: {}", fmt(*c)),
        Move::DeckPile(c) => format!("Waste→Pile: {}", fmt(*c)),
        Move::StackPile(c) => format!("Foundation→Pile: {}", fmt(*c)),
        Move::Reveal(c) => format!("Pile→Pile (reveal): move {} to another pile", fmt(*c)),
    }
}

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let cards = default_shuffle(seed);

    println!("=== Seed {} Deal ===\n", seed);

    // Tableau face-up cards
    let mut face_up_codes = Vec::new();
    for pile in 0..N_PILES as usize {
        let start = pile * (pile + 1) / 2;
        let face_up_idx = start + pile;
        let card = cards[face_up_idx];
        face_up_codes.push(fmt_code(card));

        let hidden: Vec<String> = (start..start + pile).map(|i| fmt(cards[i])).collect();
        if hidden.is_empty() {
            println!("  P{}: {}", pile + 1, fmt(card));
        } else {
            println!(
                "  P{}: {} [hidden: {}]",
                pile + 1,
                fmt(card),
                hidden.join(" ")
            );
        }
    }

    println!("\nTUI step 1: {}", face_up_codes.join(" "));

    // Stock
    let mut stock_codes = Vec::new();
    for card in cards.iter().skip(28) {
        stock_codes.push(fmt_code(*card));
    }
    println!("TUI step 2: {}", stock_codes.join(" "));

    // Hidden cards reference
    println!("\nHidden cards (for reveal inputs):");
    for pile in 1..N_PILES as usize {
        let start = pile * (pile + 1) / 2;
        // Hidden are stored bottom-to-top, reveals happen top-first
        let hidden: Vec<String> = (start..start + pile)
            .rev()
            .map(|i| format!("{} ({})", fmt(cards[i]), fmt_code(cards[i])))
            .collect();
        println!("  P{}: {}", pile + 1, hidden.join(", "));
    }

    // Play HOP game with verbose logging
    println!("\n=== HOP Game (nt=500, lim=100) ===\n");

    let mut game = Solitaire::new(&cards, 1);

    // Cycle deck
    while !game.get_deck().get_deck().is_empty() {
        game.get_deck_mut().deal_once();
    }

    let mut move_count: u32 = 0;
    let mut turn_count: u32 = 0;
    let mut seen_states = HashSet::new();
    let signal = DefaultTerminateSignal;

    while !game.is_win() && move_count < 500 {
        let state_hash = game.encode();
        if seen_states.contains(&state_hash) {
            println!("  [CYCLE DETECTED at move {}]", move_count);
            break;
        }
        seen_states.insert(state_hash);

        let rng_seed = seed.wrapping_mul(1000).wrapping_add(turn_count as u64);
        let mut rng = SmallRng::seed_from_u64(rng_seed);

        let mut det_game = game.clone();
        det_game.get_hidden_mut().shuffle_all(&mut rng);

        let best = pick_moves(&mut det_game, &mut rng, 500, 100, &signal);

        let Some(moves) = best else {
            println!("  [NO MOVES at turn {}]", turn_count);
            break;
        };

        if moves.is_empty() {
            println!("  [EMPTY MOVES at turn {}]", turn_count);
            break;
        }

        println!("Turn {}: ({} moves)", turn_count + 1, moves.len());
        for m in &moves {
            move_count += 1;
            println!("  #{:3}: {}", move_count, fmt_move(m));
            game.do_move(m);
            if move_count >= 500 {
                break;
            }
        }
        turn_count += 1;
    }

    let result = if game.is_win() { "WON" } else { "LOST" };
    println!(
        "\nResult: {} ({} moves, {} turns)",
        result, move_count, turn_count
    );
}
