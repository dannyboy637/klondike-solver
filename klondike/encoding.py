import numpy as np

# Encoding dimensions breakdown:
# Per card (52 cards):
#   - location one-hot: foundation/pile_0-6/waste/stock/unknown (11 slots) = 11
#     NOTE: stock and unknown are SEPARATE slots. A card known to be in the stock
#     (will eventually be drawn) is informationally different from a card whose
#     location is truly unknown (face-down under a tableau pile).
#   - is_visible: 1
#   - pile_position: 1 (normalized 0-1, 0 if not in pile)
# Subtotal per card: 13
# Total card features: 52 * 13 = 676
#
# Global features:
#   - foundation_ranks: 4 (normalized 0-1)
#   - pile_hidden_counts: 7 (normalized 0-1)
#   - deck_remaining: 1 (normalized)
#   - waste_size: 1
#   - deck_passes: 1 (if trackable)
# Total global: 14
#
# Uncertainty features:
#   - unseen_card_mask: 52 (binary -- which cards haven't been seen)
#   - unseen_red_fraction: 1
#   - per_pile (7 * 2 = 14): hidden share + useful-rank fraction
# Total uncertainty: 67
#
# TOTAL: 676 + 14 + 67 = 757 (baseline)

CARD_FEATURES = 13  # 11 location one-hot + 1 visible + 1 position
N_CARDS = 52
GLOBAL_FEATURES = 14
UNCERTAINTY_FEATURES = 67

OBSERVATION_DIM = N_CARDS * CARD_FEATURES + GLOBAL_FEATURES + UNCERTAINTY_FEATURES
# = 676 + 14 + 67 = 757 (baseline)


def encode_state(game) -> np.ndarray:
    """
    Encode visible game state as a float32 vector.

    Args:
        game: KlondikeGame instance (Rust PyO3 object)

    Returns:
        np.ndarray of shape (OBSERVATION_DIM,), dtype float32
    """
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

    # --- Card-centric features (52 * 13) ---
    foundation = game.get_foundation()  # [rank * 4 suits]
    piles = game.get_piles()            # [[(rank, suit, visible), ...] * 7]
    hidden_counts = game.get_hidden_counts()  # [count * 7]
    deck_info = game.get_deck_info()
    visible_cards = set(game.get_visible_cards())

    # Track which cards are accounted for
    seen_cards = set()

    # Foundation cards
    for suit in range(4):
        for rank in range(foundation[suit]):
            card_idx = rank * 4 + suit
            seen_cards.add(card_idx)
            _encode_card_location(obs, card_idx, loc="foundation")

    # Pile cards (visible)
    for pile_idx in range(7):
        pile = piles[pile_idx]
        for pos, (rank, suit, is_visible) in enumerate(pile):
            card_idx = rank * 4 + suit
            if is_visible:
                seen_cards.add(card_idx)
                _encode_card_location(obs, card_idx, loc="pile",
                                     pile_idx=pile_idx,
                                     position=pos / max(len(pile) - 1, 1))

    # Waste cards (visible discard pile)
    waste_cards = deck_info.get("waste", [])
    for rank, suit in waste_cards:
        card_idx = rank * 4 + suit
        seen_cards.add(card_idx)
        _encode_card_location(obs, card_idx, loc="waste")

    # All other cards are "unknown" (includes stock + hidden tableau cards)
    # NOTE: stock/unknown separation is deferred to Phase 6.
    # For baseline, all unaccounted cards are marked "unknown".
    for card_idx in range(52):
        if card_idx not in seen_cards:
            _encode_card_location(obs, card_idx, loc="unknown")

    idx = N_CARDS * CARD_FEATURES

    # --- Global features (14) ---
    for suit in range(4):
        obs[idx] = foundation[suit] / 13.0
        idx += 1
    for pile_idx in range(7):
        obs[idx] = hidden_counts[pile_idx] / 6.0
        idx += 1
    obs[idx] = deck_info.get("stock_count", 0) / 24.0
    idx += 1
    obs[idx] = len(waste_cards) / 24.0
    idx += 1
    obs[idx] = 0.0  # deck_passes: not tracked yet, placeholder for Phase 6
    idx += 1

    # --- Uncertainty features (67) ---
    # 52 dims: unseen card mask
    unseen = set(range(52)) - seen_cards
    for card_idx in range(52):
        obs[idx] = 1.0 if card_idx in unseen else 0.0
        idx += 1

    # Global unseen summary (1 dim)
    n_unseen = len(unseen)
    unseen_red_frac = 0.0
    if n_unseen > 0:
        # Suit convention: 0=hearts(red), 1=diamonds(red), 2=clubs(black), 3=spades(black)
        # Card idx convention: rank * 4 + suit
        unseen_red_frac = sum(1 for c in unseen if c % 4 < 2) / n_unseen
    obs[idx] = unseen_red_frac
    idx += 1

    # Per-pile uncertainty hints (7 piles * 2 features = 14 dims)
    for pile_idx in range(7):
        n_hidden = hidden_counts[pile_idx]
        if n_hidden == 0 or n_unseen == 0:
            obs[idx] = 0.0      # this pile's share of hidden cards
            idx += 1
            obs[idx] = 0.0      # useful-rank fraction among unseen
            idx += 1
        else:
            obs[idx] = n_hidden / max(n_unseen, 1)  # this pile's share of hidden cards
            idx += 1
            # Fraction of unseen cards with rank <= 6 (more placeable on tableau)
            obs[idx] = sum(1 for c in unseen if c // 4 <= 6) / n_unseen
            idx += 1
    # Total uncertainty: 52 (unseen mask) + 1 (global) + 14 (per-pile) = 67

    return obs


def _encode_card_location(obs, card_idx, loc, pile_idx=0, position=0.0):
    """Encode a single card's location features."""
    base = card_idx * CARD_FEATURES
    # Location one-hot: [foundation, pile0-6, waste, stock, unknown] = 11 dims
    # stock and unknown are SEPARATE: a card in the stock will be drawn eventually,
    # while an unknown card is face-down under a tableau pile -- different info.
    loc_map = {
        "foundation": 0,
        "pile": 1 + pile_idx,  # 1-7
        "waste": 8,
        "stock": 9,
        "unknown": 10,
    }
    loc_idx = loc_map.get(loc, 10)
    obs[base + loc_idx] = 1.0
    # Visibility
    obs[base + 11] = 1.0 if loc not in ("unknown", "stock") else 0.0
    # Position in pile (normalized)
    obs[base + 12] = position if loc == "pile" else 0.0
