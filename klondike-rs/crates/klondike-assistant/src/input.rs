use lonelybot::card::{Card, NUMBERS, N_CARDS, SYMBOLS};

/// Parse a single 2-character card code like "ah" → Ace of Hearts.
///
/// Rank: a 2 3 4 5 6 7 8 9 t j q k
/// Suit: h d c s (Hearts=0, Diamonds=1, Clubs=2, Spades=3)
pub fn parse_card(s: &str) -> Result<Card, String> {
    let s = s.trim().to_lowercase();
    if s.len() != 2 {
        return Err(format!("Card code must be 2 characters, got '{}'", s));
    }

    let bytes = s.as_bytes();

    let rank = match bytes[0] {
        b'a' => 0,
        b'2' => 1,
        b'3' => 2,
        b'4' => 3,
        b'5' => 4,
        b'6' => 5,
        b'7' => 6,
        b'8' => 7,
        b'9' => 8,
        b't' => 9,
        b'j' => 10,
        b'q' => 11,
        b'k' => 12,
        c => {
            return Err(format!(
                "Invalid rank '{}'. Use: a 2 3 4 5 6 7 8 9 t j q k",
                c as char
            ))
        }
    };

    let suit = match bytes[1] {
        b'h' => 0, // Hearts
        b'd' => 1, // Diamonds
        b'c' => 2, // Clubs
        b's' => 3, // Spades
        c => return Err(format!("Invalid suit '{}'. Use: h d c s", c as char)),
    };

    Ok(Card::new(rank, suit))
}

/// Parse a space-separated sequence of card codes.
pub fn parse_cards(s: &str) -> Result<Vec<Card>, String> {
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if tokens.is_empty() {
        return Err("No cards provided".to_string());
    }

    let mut cards = Vec::with_capacity(tokens.len());
    let mut seen = [false; N_CARDS as usize];

    for (i, token) in tokens.iter().enumerate() {
        let card = parse_card(token).map_err(|e| format!("Card {}: {}", i + 1, e))?;
        let val = card.value() as usize;
        if seen[val] {
            return Err(format!("Duplicate card: {} ({})", token, format_card(card)));
        }
        seen[val] = true;
        cards.push(card);
    }

    Ok(cards)
}

/// Format a Card as a human-readable string like "A♥" or "T♠".
pub fn format_card(card: Card) -> String {
    let rank = card.rank() as usize;
    let suit = card.suit() as usize;
    format!("{}{}", NUMBERS[rank], SYMBOLS[suit])
}

/// Format a Card as a 2-character code like "ah" or "ts".
#[allow(dead_code)]
pub fn format_card_code(card: Card) -> String {
    let rank_char = match card.rank() {
        0 => 'A',
        1 => '2',
        2 => '3',
        3 => '4',
        4 => '5',
        5 => '6',
        6 => '7',
        7 => '8',
        8 => '9',
        9 => 'T',
        10 => 'J',
        11 => 'Q',
        12 => 'K',
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

/// Suit color: true = red (hearts, diamonds), false = black (clubs, spades)
pub fn is_red(card: Card) -> bool {
    card.suit() < 2
}

/// Suit symbol
pub fn suit_symbol(suit: u8) -> &'static str {
    SYMBOLS[suit as usize]
}

/// Suit name
#[allow(dead_code)]
pub fn suit_name(suit: u8) -> &'static str {
    match suit {
        0 => "Hearts",
        1 => "Diamonds",
        2 => "Clubs",
        3 => "Spades",
        _ => "?",
    }
}

/// Format a suit index as its symbol, useful for foundations
pub fn foundation_label(suit: u8, rank: u8) -> String {
    if rank == 0 {
        format!("{}:--", SYMBOLS[suit as usize])
    } else {
        let rank_idx = (rank - 1) as usize;
        format!("{}:{}", SYMBOLS[suit as usize], NUMBERS[rank_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_card() {
        // Ace of Hearts: rank=0, suit=0 → value=0
        let c = parse_card("ah").unwrap();
        assert_eq!(c.rank(), 0);
        assert_eq!(c.suit(), 0);
        assert_eq!(c.value(), 0);

        // 10 of Spades: rank=9, suit=3 → value=39
        let c = parse_card("ts").unwrap();
        assert_eq!(c.rank(), 9);
        assert_eq!(c.suit(), 3);
        assert_eq!(c.value(), 39);

        // Queen of Diamonds: rank=11, suit=1 → value=45
        let c = parse_card("qd").unwrap();
        assert_eq!(c.rank(), 11);
        assert_eq!(c.suit(), 1);
        assert_eq!(c.value(), 45);

        // King of Clubs: rank=12, suit=2 → value=50
        let c = parse_card("kc").unwrap();
        assert_eq!(c.rank(), 12);
        assert_eq!(c.suit(), 2);
        assert_eq!(c.value(), 50);

        // Case insensitive
        let c = parse_card("AH").unwrap();
        assert_eq!(c.value(), 0);
    }

    #[test]
    fn test_parse_card_errors() {
        assert!(parse_card("").is_err());
        assert!(parse_card("a").is_err());
        assert!(parse_card("ahh").is_err());
        assert!(parse_card("xh").is_err());
        assert!(parse_card("ax").is_err());
    }

    #[test]
    fn test_parse_cards() {
        let cards = parse_cards("ah ts qd").unwrap();
        assert_eq!(cards.len(), 3);
        assert_eq!(cards[0].value(), 0); // AH
        assert_eq!(cards[1].value(), 39); // TS
        assert_eq!(cards[2].value(), 45); // QD
    }

    #[test]
    fn test_parse_cards_duplicate() {
        assert!(parse_cards("ah ts ah").is_err());
    }

    #[test]
    fn test_format_card() {
        assert_eq!(format_card(Card::new(0, 0)), "A♥");
        assert_eq!(format_card(Card::new(9, 3)), "10♠");
        assert_eq!(format_card(Card::new(11, 1)), "Q♦");
        assert_eq!(format_card(Card::new(12, 2)), "K♣");
    }

    #[test]
    fn test_all_52_cards_roundtrip() {
        for val in 0..N_CARDS {
            let card = Card::from_value(val);
            let code = format_card_code(card);
            let parsed = parse_card(&code).unwrap();
            assert_eq!(parsed.value(), val, "roundtrip failed for value {}", val);
        }
    }

    #[test]
    fn test_suit_colors() {
        assert!(is_red(Card::new(0, 0))); // Hearts = red
        assert!(is_red(Card::new(0, 1))); // Diamonds = red
        assert!(!is_red(Card::new(0, 2))); // Clubs = black
        assert!(!is_red(Card::new(0, 3))); // Spades = black
    }

    #[test]
    fn test_foundation_label() {
        assert_eq!(foundation_label(0, 0), "♥:--");
        assert_eq!(foundation_label(0, 1), "♥:A");
        assert_eq!(foundation_label(3, 13), "♠:K");
    }
}
