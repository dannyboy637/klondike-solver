# Phase 1: PyO3 Bridge + Python Game Interface

**Gate**: Gate 1 preparation
**Parallelizable**: YES — Rust (Task A) and Python (Task B) can be developed in parallel
**Estimated sessions**: 2-3
**Dependencies**: Phase 0 complete (Lonelybot builds, baselines recorded)

---

## Overview

Build the bridge between Lonelybot (Rust) and our Python training pipeline. This is the **biggest engineering risk** in the project — if this is slow or unreliable, everything downstream suffers.

**Deliverables:**
1. `klondike-env/` — Rust crate with PyO3 bindings exposing Lonelybot to Python
2. `klondike/env.py` — Python wrapper providing a clean game interface
3. `klondike/encoding.py` — State encoding (757-dim observation vector, baseline)
4. `klondike/actions.py` — Action space enumeration and mapping
5. Tests proving round-trip correctness and benchmarks showing acceptable latency

---

## Team Agent Strategy

### TASK A (Rust Agent): PyO3 Crate
- Creates `klondike-env/` crate
- Implements all `#[pyclass]` and `#[pymethods]`
- Handles Card, Move, Game state, shuffling, solving
- Benchmark: verify <1μs per function call overhead

### TASK B (Python Agent): Game Interface + Encoding
- Creates `klondike/` package skeleton
- Implements `env.py`, `encoding.py`, `actions.py`
- Uses stub/mock interface matching Task A's API
- Once Task A delivers the compiled module, integration test

### SHARED INTERFACE (Both agents must agree on this):

```python
# The Python-visible API from the Rust crate (klondike_env module)
class KlondikeGame:
    def __init__(self, seed: int, draw_step: int = 1) -> None: ...
    def reset(self, seed: int) -> None: ...
    def get_legal_moves(self) -> list[tuple[int, int]]: ...  # [(move_type, card_idx), ...]
    def do_move(self, move_type: int, card_idx: int) -> bool: ...
    def undo_move(self) -> bool: ...
    def is_win(self) -> bool: ...
    def is_sure_win(self) -> bool: ...
    def get_state(self) -> dict: ...  # Full visible state for encoding
    def get_foundation(self) -> list[int]: ...  # [rank_per_suit × 4]
    def get_piles(self) -> list[list[tuple[int, int, bool]]]: ...  # [(rank, suit, visible), ...]
    def get_hidden_counts(self) -> list[int]: ...  # [count × 7 piles]
    def get_deck_info(self) -> dict: ...  # waste cards, remaining, position
    def get_visible_cards(self) -> list[int]: ...  # card indices that are face-up
    def clone(self) -> "KlondikeGame": ...
    def shuffle_hidden(self, rng_seed: int) -> None: ...  # PIMC determinization
    def hop_solve(self, n_determinizations: int, node_limit: int) -> dict: ...
    def encode(self) -> int: ...  # u64 state hash for transposition

# Constants
N_CARDS: int = 52
N_SUITS: int = 4
N_RANKS: int = 13
N_PILES: int = 7
N_MOVE_TYPES: int = 5  # DeckStack, PileStack, DeckPile, StackPile, Reveal
ACTION_SPACE_SIZE: int = 260  # 5 × 52
```

---

## TASK A: Rust PyO3 Crate (`klondike-env/`)

### Directory Structure

```
klondike-env/
├── Cargo.toml
├── pyproject.toml
└── src/
    └── lib.rs
```

### Cargo.toml

```toml
[package]
name = "klondike-env"
version = "0.1.0"
edition = "2021"

[lib]
name = "klondike_env"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
lonelybot = { path = "../lonelybot" }
rand = { version = "0.9.0", features = ["small_rng"] }

[profile.release]
lto = true
codegen-units = 1
```

### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "klondike-env"
requires-python = ">=3.10"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### Key Rust Types to Wrap

These are the Lonelybot types that need PyO3 exposure:

```rust
// From lonelybot — Card is u8 internally
// Card::new(rank, suit) where rank 0-12, suit 0-3
// Card::rank() -> u8, Card::suit() -> u8

// Move enum — 5 variants, each containing a Card
// Move::DeckStack(Card)   = type 0
// Move::PileStack(Card)   = type 1
// Move::DeckPile(Card)    = type 2
// Move::StackPile(Card)   = type 3
// Move::Reveal(Card)      = type 4

// Solitaire — main game state
// SolitaireEngine<FullPruner> — move execution wrapper
// Stack — foundation (4 suit stacks, u16 internally)
// Deck — draw pile (24 cards max, with draw_step and position)
// Hidden — face-down tableau cards

// Key functions:
// shuffler::default_shuffle(seed: u64) -> [Card; 52]
// Solitaire::new(&cards, draw_step) -> Solitaire
// SolitaireEngine::new(solitaire) -> SolitaireEngine
// engine.list_moves_dom() -> ArrayVec<Move, 24>
// engine.do_move(m) -> bool
// engine.undo_move() -> bool
// engine.state() -> &Solitaire
// state.is_win() -> bool
// state.is_sure_win() -> bool
// state.get_stack() -> &Stack
// state.get_deck() -> &Deck
// state.get_hidden() -> &Hidden
// state.compute_visible_piles() -> [PileVec; 7]
// state.hidden_shuffle(&mut self, rng) — PIMC determinization!
// state.encode() -> u64
// hop_solver::hop_solve_game(...) -> HopResult { wins, skips, played }
```

### Implementation Guide (lib.rs)

The `#[pyclass]` struct wraps `SolitaireEngine<FullPruner>`:

```rust
use pyo3::prelude::*;
use lonelybot::{
    card::{Card, N_CARDS, N_RANKS, N_SUITS},
    deck::{N_PILES, Deck},
    engine::SolitaireEngine,
    moves::Move,
    pruning::FullPruner,
    shuffler::default_shuffle,
    state::Solitaire,
};
use std::num::NonZeroU8;

/// Move type constants matching the Move enum order
const MOVE_DECK_STACK: u8 = 0;
const MOVE_PILE_STACK: u8 = 1;
const MOVE_DECK_PILE: u8 = 2;
const MOVE_STACK_PILE: u8 = 3;
const MOVE_REVEAL: u8 = 4;

#[pyclass]
struct KlondikeGame {
    engine: SolitaireEngine<FullPruner>,
    draw_step: NonZeroU8,
    seed: u64,
}
```

**Key method implementations:**

1. **`__new__` / `reset`**: Shuffle cards, create Solitaire, wrap in engine
2. **`get_legal_moves`**: Call `engine.list_moves_dom()`, convert each Move to `(type_id, card_index)` where `card_index = rank * 4 + suit`
3. **`do_move`**: Convert `(type_id, card_index)` back to Move, call `engine.do_move()`
4. **`undo_move`**: Call `engine.undo_move()`
5. **`get_foundation`**: Read `state.get_stack()`, return 4 rank values
6. **`get_piles`**: Call `state.compute_visible_piles()`, return nested lists
7. **`get_hidden_counts`**: Read `state.get_hidden()`, return 7 counts
8. **`get_deck_info`**: Read `state.get_deck()`, return waste/stock info
9. **`get_visible_cards`**: Compute from visible_mask or pile/deck/stack data
10. **`clone`**: Clone the engine for PIMC branching
11. **`shuffle_hidden`**: Call `state.hidden_shuffle(rng)` — crucial for PIMC
12. **`hop_solve`**: Call `hop_solve_game()` for search-only baseline

**Card index convention**: `card_idx = rank * 4 + suit` (rank 0-12, suit 0-3, total 0-51).
This matches Lonelybot's internal `Card(u8)` but may need adjustment based on Card's actual encoding. Check `Card::new(rank, suit)` internals.

### Move Encoding

```
Action space: 260 entries
Index = move_type * 52 + card_idx
  move_type: 0=DeckStack, 1=PileStack, 2=DeckPile, 3=StackPile, 4=Reveal
  card_idx: rank * 4 + suit (0-51)

Legal move mask: 260-dim binary vector (1 = legal, 0 = illegal)
At most 24 moves are legal in any position (N_MOVES_MAX = 24)
```

### Build & Install

```bash
cd klondike-env
pip install maturin
maturin develop --release
```

After building, `import klondike_env` works in Python.

### Performance Target

- Function call overhead: <1μs (PyO3 baseline is ~60-70ns)
- `get_legal_moves()`: <10μs
- `do_move()`: <5μs
- `get_state()` (full observation): <50μs
- `shuffle_hidden()`: <10μs
- `clone()`: <50μs

---

## TASK B: Python Game Interface (`klondike/`)

### Directory Structure

```
klondike/
├── __init__.py
├── env.py          # High-level game environment
├── encoding.py     # State → 757-dim vector (baseline)
└── actions.py      # Action space mapping
```

### `klondike/__init__.py`

```python
from klondike.env import KlondikeEnv
from klondike.encoding import encode_state, OBSERVATION_DIM
from klondike.actions import ACTION_SPACE_SIZE, action_to_move, move_to_action
```

### `klondike/env.py` — Game Environment

Wraps the Rust `KlondikeGame` with a gym-like interface:

```python
import numpy as np
# from klondike_env import KlondikeGame  # The Rust module

class KlondikeEnv:
    """Gym-like wrapper around Lonelybot via PyO3."""

    def __init__(self, draw_step: int = 1):
        self.draw_step = draw_step
        self._game = None

    def reset(self, seed: int) -> np.ndarray:
        """Reset to a new deal. Returns observation vector."""
        self._game = KlondikeGame(seed=seed, draw_step=self.draw_step)
        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take action. Returns (obs, reward, done, info)."""
        move_type, card_idx = action_to_move(action)
        success = self._game.do_move(move_type, card_idx)
        if not success:
            raise ValueError(f"Illegal move: action={action}")

        done = self._game.is_win()
        reward = 1.0 if done else 0.0
        obs = self._get_obs()
        info = {"is_sure_win": self._game.is_sure_win()}
        return obs, reward, done, info

    def get_legal_actions(self) -> np.ndarray:
        """Returns binary mask of legal actions (size ACTION_SPACE_SIZE)."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for move_type, card_idx in self._game.get_legal_moves():
            action = move_to_action(move_type, card_idx)
            mask[action] = 1.0
        return mask

    def get_legal_action_list(self) -> list[int]:
        """Returns list of legal action indices."""
        return [move_to_action(mt, ci) for mt, ci in self._game.get_legal_moves()]

    def clone(self) -> "KlondikeEnv":
        """Deep copy for PIMC branching."""
        new_env = KlondikeEnv(draw_step=self.draw_step)
        new_env._game = self._game.clone()
        return new_env

    def shuffle_hidden(self, rng_seed: int) -> None:
        """Randomly rearrange hidden cards (PIMC determinization)."""
        self._game.shuffle_hidden(rng_seed)

    def is_done(self) -> bool:
        """Check if game is won or no legal moves remain."""
        return self._game.is_win() or len(self._game.get_legal_moves()) == 0

    def _get_obs(self) -> np.ndarray:
        """Extract observation vector from current state."""
        from klondike.encoding import encode_state
        return encode_state(self._game)
```

### `klondike/encoding.py` — State Encoding (757 dimensions baseline)

The observation vector represents what a player can see (NOT hidden cards).

**Dimension consistency note**: Earlier docs reference "~857 dims" — that includes
future expansion features (action history, stock-pass context, cycle detection) that
are Phase 6 experiments. The BASELINE implementation uses **757 dims**. All phases
(1-4) must use this number. If/when features are added in Phase 6, bump the version
and retrain from scratch — do not mix checkpoint dimensions.

```python
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
# Total card features: 52 × 13 = 676
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
#   - unseen_card_mask: 52 (binary — which cards haven't been seen)
#   - per_pile_red_fraction: 7 (of unseen cards that could be in each pile)
#   - per_pile_avg_rank: 7 (of unseen cards)
#   - cards_in_stock_estimate: 1
# Total uncertainty: 67
#
# TOTAL: 676 + 14 + 67 = 757 (baseline)
#
# FUTURE EXPANSION (Phase 6 experiments, NOT in baseline):
#   - stock-pass context: which pass through the stock (1 feature)
#   - action history: last 5 actions as (move_type, card_idx) (5 × 7 = 35 dims)
#   - move-cycle features: has current state been visited (1 bit)
#   - deck position: exact position in deck cycling (1 feature)
#   These additions would bring total to ~795-857 dims.

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
    idx = 0

    # --- Card-centric features (52 × 12) ---
    foundation = game.get_foundation()  # [rank × 4 suits]
    piles = game.get_piles()            # [[( rank, suit, visible ), ...] × 7]
    hidden_counts = game.get_hidden_counts()  # [count × 7]
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

    # Waste/stock visible cards
    waste_cards = deck_info.get("waste_cards", [])
    for rank, suit in waste_cards:
        card_idx = rank * 4 + suit
        seen_cards.add(card_idx)
        _encode_card_location(obs, card_idx, loc="waste")

    # All other cards are "unknown"
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
    obs[idx] = deck_info.get("remaining", 0) / 24.0
    idx += 1
    obs[idx] = len(waste_cards) / 24.0
    idx += 1
    obs[idx] = deck_info.get("passes", 0) / 10.0  # normalize passes
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
        # Card idx convention: rank * 4 + suit (verify in Phase 0)
        unseen_red_frac = sum(1 for c in unseen if c % 4 < 2) / n_unseen
    obs[idx] = unseen_red_frac
    idx += 1

    # Per-pile uncertainty hints (7 piles × 2 features = 14 dims)
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
    # Total uncertainty: 52 (unseen mask) + 1 (global) + 14 (per-pile) = 67 ✓

    return obs


def _encode_card_location(obs, card_idx, loc, pile_idx=0, position=0.0):
    """Encode a single card's location features."""
    base = card_idx * CARD_FEATURES
    # Location one-hot: [foundation, pile0-6, waste, stock, unknown] = 11 dims
    # stock and unknown are SEPARATE: a card in the stock will be drawn eventually,
    # while an unknown card is face-down under a tableau pile — different info.
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
```

### `klondike/actions.py` — Action Space

```python
N_CARDS = 52
N_MOVE_TYPES = 5
ACTION_SPACE_SIZE = N_MOVE_TYPES * N_CARDS  # 260

# Move type mapping (matches Rust enum order):
# 0 = DeckStack  (deck card → foundation)
# 1 = PileStack  (pile card → foundation)
# 2 = DeckPile   (deck card → tableau pile)
# 3 = StackPile  (foundation card → tableau pile)
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
```

---

## Integration & Testing

### Test 1: Round-trip correctness
```python
# Create game, get legal moves, play one, verify state changes
env = KlondikeEnv(draw_step=1)
obs = env.reset(seed=42)
assert obs.shape == (OBSERVATION_DIM,)

legal = env.get_legal_action_list()
assert len(legal) > 0

obs2, reward, done, info = env.step(legal[0])
assert obs2.shape == (OBSERVATION_DIM,)
```

### Test 2: Clone + shuffle for PIMC
```python
env = KlondikeEnv(draw_step=1)
env.reset(seed=42)

# Clone and determinize
clone = env.clone()
clone.shuffle_hidden(rng_seed=123)

# Legal moves may differ (different hidden cards)
original_moves = env.get_legal_action_list()
clone_moves = clone.get_legal_action_list()
# Visible state should be same, but future play diverges
```

### Test 3: Play full game
```python
import random

env = KlondikeEnv(draw_step=1)
env.reset(seed=0)
moves = 0
while not env.is_done():
    legal = env.get_legal_action_list()
    if not legal:
        break
    env.step(random.choice(legal))
    moves += 1
    if moves > 500:  # Safety limit (real games rarely exceed 200)
        break
```

### Test 4: Performance benchmark
```python
import time

env = KlondikeEnv(draw_step=1)
N = 10000

start = time.perf_counter()
for seed in range(N):
    env.reset(seed=seed)
    while not env.is_done():
        legal = env.get_legal_action_list()
        if not legal:
            break
        env.step(legal[0])  # always play first legal move
elapsed = time.perf_counter() - start
print(f"{N} random games in {elapsed:.2f}s ({N/elapsed:.0f} games/sec)")
# Target: >1000 games/sec for random rollouts
```

---

## Acceptance Criteria

- [ ] `maturin develop --release` builds successfully
- [ ] `import klondike_env` works in Python
- [ ] All 4 tests above pass
- [ ] Performance: >1000 random-play games/sec
- [ ] Clone + shuffle_hidden works correctly (PIMC foundation)
- [ ] Legal move mask is consistent with Lonelybot's move generation
- [ ] State encoding produces consistent vectors (same state → same encoding)

---

## Decision Gate

No go/no-go here — this phase is infrastructure. If PyO3 builds and tests pass, proceed to Phase 2.

**If PyO3 fails to build**: Fall back to subprocess communication (10-100x slower but workable for prototyping). File an issue on Lonelybot if needed.

---

## Handoff to Phase 2

Phase 2 needs:
1. Working `klondike_env` Python module (installed via maturin)
2. Working `klondike/env.py` with `reset()`, `step()`, `get_legal_actions()`, `clone()`, `shuffle_hidden()`
3. Working `klondike/encoding.py` with `encode_state()` and `OBSERVATION_DIM`
4. Working `klondike/actions.py` with `ACTION_SPACE_SIZE`, conversion functions
5. Performance benchmarks recorded

---

## Lonelybot API Quick Reference

### Card
- `Card::new(rank: u8, suit: u8)` — rank 0-12 (A=0, K=12), suit 0-3
- `Card::rank() -> u8`, `Card::suit() -> u8`
- Index convention for Python: `card_idx = rank * 4 + suit`

### Move (5 types)
- `DeckStack(Card)` — type 0: deck/waste card → foundation
- `PileStack(Card)` — type 1: pile top → foundation
- `DeckPile(Card)` — type 2: deck/waste card → tableau pile
- `StackPile(Card)` — type 3: foundation top → tableau pile
- `Reveal(Card)` — type 4: clear pile top to reveal hidden card
- Max legal moves per position: 24

### SolitaireEngine
- `new(solitaire)` — wrap game state
- `list_moves_dom() -> ArrayVec<Move, 24>` — legal moves with dominance pruning
- `do_move(m) -> bool` — execute move
- `undo_move() -> bool` — undo last move
- `state() -> &Solitaire` — access underlying state

### Solitaire
- `new(cards, draw_step)` — create from shuffled deck
- `is_win() -> bool`, `is_sure_win() -> bool`
- `gen_moves::<true>() -> MoveMask` — with dominance pruning
- `get_stack() -> &Stack`, `get_deck() -> &Deck`, `get_hidden() -> &Hidden`
- `compute_visible_piles() -> [PileVec; 7]`
- `hidden_shuffle(&mut self, rng)` — PIMC determinization
- `encode() -> u64` — state hash for transposition table

### Shuffler
- `default_shuffle(seed: u64) -> [Card; 52]`
- Other shufflers: `solvitaire_shuffle`, `ks_shuffle`, `greenfelt_shuffle`

### HOP Solver
- `hop_solve_game(game, move, rng, n_times, limit, sign, pruner) -> HopResult`
- `HopResult { wins, skips, played }`
