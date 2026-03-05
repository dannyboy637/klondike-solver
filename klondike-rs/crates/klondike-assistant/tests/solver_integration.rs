/// Integration test: verify the solver produces recommendations for a real game.
use lonelybot::card::Card;
use lonelybot::deck::{N_PILES, N_PILE_CARDS};
use lonelybot::shuffler::default_shuffle;

const RANKS: [char; 13] = [
    'a', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k',
];
const SUITS: [char; 4] = ['h', 'd', 'c', 's'];

fn card_code(card: Card) -> String {
    format!(
        "{}{}",
        RANKS[card.rank() as usize],
        SUITS[card.suit() as usize]
    )
}

// We need access to the assistant's internals.
// Since assistant is a binary crate, we test through its public API indirectly
// by reimplementing the key logic here.

/// Extract the 7 initial face-up tableau cards from a standard deal.
fn extract_tableau_faceup(cards: &[Card; 52]) -> [Card; 7] {
    std::array::from_fn(|i| {
        let idx = i * (i + 1) / 2 + i; // face-up position for pile i
        cards[idx]
    })
}

/// Extract the 24 stock cards from a standard deal.
fn extract_stock(cards: &[Card; 52]) -> Vec<Card> {
    cards[N_PILE_CARDS as usize..].to_vec()
}

#[test]
fn test_deal_extraction_seed42() {
    let cards = default_shuffle(42);

    let tableau = extract_tableau_faceup(&cards);
    let stock = extract_stock(&cards);

    // Verify we got 7 + 24 = 31 cards, all unique
    let mut seen = [false; 52];
    for card in &tableau {
        let v = card.value() as usize;
        assert!(!seen[v], "duplicate in tableau: {}", v);
        seen[v] = true;
    }
    for card in &stock {
        let v = card.value() as usize;
        assert!(!seen[v], "duplicate in stock: {}", v);
        seen[v] = true;
    }

    assert_eq!(seen.iter().filter(|&&s| s).count(), 31);
    assert_eq!(stock.len(), 24);

    // Print for debugging
    eprintln!("Seed 42 tableau face-up:");
    for (i, card) in tableau.iter().enumerate() {
        eprintln!(
            "  Pile {}: rank={}, suit={} (value={})",
            i,
            card.rank(),
            card.suit(),
            card.value()
        );
    }
}

#[test]
fn test_print_deal_for_assistant() {
    // Prints deal in assistant card-code format for manual testing.
    // Run with: cargo test -p klondike-assistant -- --nocapture test_print_deal
    let seed = 42;
    let cards = default_shuffle(seed);
    let tableau = extract_tableau_faceup(&cards);
    let stock = extract_stock(&cards);

    let tableau_codes: Vec<String> = tableau.iter().map(|c| card_code(*c)).collect();
    let stock_codes: Vec<String> = stock.iter().map(|c| card_code(*c)).collect();

    eprintln!("\n=== Seed {} deal (copy-paste into assistant) ===", seed);
    eprintln!("Tableau: {}", tableau_codes.join(" "));
    eprintln!("Stock:   {}", stock_codes.join(" "));
    eprintln!("===\n");
}

#[test]
fn test_deal_matches_solitaire_new() {
    // Verify our extraction matches what Solitaire::new produces
    use lonelybot::engine::Solitaire;

    let cards = default_shuffle(42);
    let game = Solitaire::new(&cards, 1);

    let tableau = extract_tableau_faceup(&cards);

    // The visible piles should start with our extracted face-up cards
    let piles = game.get_visible_piles();
    for i in 0..N_PILES as usize {
        assert_eq!(
            piles[i].len(),
            1,
            "pile {} should have 1 visible card at start",
            i
        );
        assert_eq!(
            piles[i][0].value(),
            tableau[i].value(),
            "pile {} face-up card mismatch",
            i
        );
    }
}
