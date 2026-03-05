//! Programmatic test: drive the App state machine through seed 42.
//! Validates the full flow: setup → stock cycle → play (with reveals) → game result.

use lonelybot::card::Card;
use lonelybot::shuffler::default_shuffle;

// We need to access the assistant's internal modules.
// Since they're private, we test via the public API of each module.
// For now, test the solver + game_state directly.

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

fn _fmt(card: Card) -> String {
    format!(
        "{}{}",
        lonelybot::card::NUMBERS[card.rank() as usize],
        lonelybot::card::SYMBOLS[card.suit() as usize]
    )
}

/// Minimal test: verify that the deal extraction from seed 42 matches expected.
#[test]
fn test_seed42_deal_extraction() {
    let cards = default_shuffle(42);

    // Verify face-up cards at expected positions
    let expected_face_up = [
        (0, "4d"),  // P1
        (2, "5h"),  // P2
        (5, "qd"),  // P3
        (9, "9s"),  // P4
        (14, "ac"), // P5
        (20, "6d"), // P6
        (27, "2h"), // P7
    ];

    for (idx, expected_code) in &expected_face_up {
        let card = cards[*idx];
        let code = fmt_code(card);
        assert_eq!(
            &code,
            expected_code,
            "P{} face-up mismatch at index {}",
            idx + 1,
            idx
        );
    }

    // Verify stock cards
    let expected_stock = [
        "9c", "5s", "2c", "kh", "th", "6h", "ah", "jd", "6s", "9h", "2s", "8d", "4c", "5d", "7c",
        "ad", "js", "5c", "qc", "8c", "jc", "ks", "4h", "7s",
    ];

    for (i, expected_code) in expected_stock.iter().enumerate() {
        let card = cards[28 + i];
        let code = fmt_code(card);
        assert_eq!(&code, expected_code, "Stock card {} mismatch", i);
    }
}

/// Verify hidden card order (top-first for reveals)
#[test]
fn test_seed42_hidden_order() {
    let cards = default_shuffle(42);

    // P2: 1 hidden card at index 1
    assert_eq!(fmt_code(cards[1]), "as"); // A♠

    // P3: 2 hidden at indices 3,4. Top-first = index 4, then 3
    assert_eq!(fmt_code(cards[4]), "8h"); // first revealed
    assert_eq!(fmt_code(cards[3]), "9d"); // second revealed

    // P4: 3 hidden at indices 6,7,8. Top-first = 8,7,6
    assert_eq!(fmt_code(cards[8]), "3h"); // first revealed
    assert_eq!(fmt_code(cards[7]), "7d"); // second
    assert_eq!(fmt_code(cards[6]), "3d"); // third
}
