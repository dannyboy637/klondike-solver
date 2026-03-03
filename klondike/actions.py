N_CARDS = 52
N_MOVE_TYPES = 5
ACTION_SPACE_SIZE = N_MOVE_TYPES * N_CARDS  # 260

# Move type mapping (matches Rust enum order):
# 0 = DeckStack  (deck card -> foundation)
# 1 = PileStack  (pile card -> foundation)
# 2 = DeckPile   (deck card -> tableau pile)
# 3 = StackPile  (foundation card -> tableau pile)
# 4 = Reveal     (move cards to reveal hidden card in pile)

MOVE_NAMES = ["DeckStack", "PileStack", "DeckPile", "StackPile", "Reveal"]


def move_to_action(move_type: int, card_idx: int) -> int:
    """Convert (move_type, card_idx) to flat action index."""
    return move_type * N_CARDS + card_idx


def action_to_move(action: int) -> tuple[int, int]:
    """Convert flat action index to (move_type, card_idx)."""
    return divmod(action, N_CARDS)


def card_idx_to_str(card_idx: int) -> str:
    """Convert card index to human-readable string."""
    rank = card_idx // 4
    suit = card_idx % 4
    rank_names = "A23456789TJQK"
    suit_names = "HDCS"
    return f"{rank_names[rank]}{suit_names[suit]}"


def action_to_str(action: int) -> str:
    """Convert action index to human-readable string."""
    move_type, card_idx = action_to_move(action)
    return f"{MOVE_NAMES[move_type]}({card_idx_to_str(card_idx)})"
