# Phase 5: Deployment Pipeline (Optional)

**Gate**: Gate 4 — Deployment (only if Phase 4 achieves >50% win rate)
**Parallelizable**: YES — CV pipeline (Task A) and real-time advisor (Task B) are independent
**Estimated sessions**: 2-3
**Dependencies**: Phase 4 complete (trained model with >50% win rate confirmed)

---

## Overview

Build the pipeline to use the trained solver during actual online play. Screenshot of game board → computer vision → game state → model → move recommendation → display to player.

**Important caveats (from Gate 0.5 research)**:
- This phase should NOT be started until Gate 0.5 questions are answered:
  - Decision latency allowed on 1xbet
  - Hidden timers or forced completion windows
  - Rule fidelity confirmed
  - Account risk assessed
- Solver win rate ≠ deployable EV (latency, input errors, human execution drift)

**Deliverables:**
1. Screenshot → card recognition → game state extraction
2. Real-time move advisor with latency display
3. Rule fidelity verification against 1xbet
4. Session logging and bankroll tracking

---

## Team Agent Strategy

### TASK A (CV Agent): Screenshot → Game State
- Card detection from screenshots
- Template matching or small CNN for card recognition
- Board layout parsing (which cards are where)
- Handle various resolutions and themes

### TASK B (Advisor Agent): Real-Time Move Recommendation
- Game state → model inference → best move
- Display overlay or terminal output
- Latency tracking (must be fast enough for live play)
- Game session logger

---

## TASK A: Computer Vision Pipeline

### Approach Options

1. **Template Matching (Simplest)**
   - Capture reference images of each card
   - Use OpenCV template matching to locate cards
   - Works well if card appearance is consistent
   - Fragile to resolution/theme changes

2. **YOLO / Object Detection (More Robust)**
   - Train a small YOLO model on labeled card screenshots
   - Detects cards regardless of position
   - Needs training data (annotated screenshots)
   - More robust but more setup

3. **Manual Input (Fallback)**
   - Player types/clicks card positions manually
   - No CV needed — just a structured input form
   - Slow but guaranteed accurate

**Recommended**: Start with template matching (fast to build), upgrade to YOLO if needed.

### `klondike/deploy/cv.py` — Card Recognition

```python
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectedCard:
    rank: int       # 0-12
    suit: int       # 0-3
    x: int          # Pixel position
    y: int          # Pixel position
    confidence: float


@dataclass
class BoardState:
    foundation: list[list[DetectedCard]]    # 4 foundation piles
    tableau: list[list[DetectedCard]]       # 7 tableau piles
    waste: list[DetectedCard]               # Waste pile visible cards
    stock_remaining: int                     # Cards remaining in stock (count only)


def capture_screenshot() -> np.ndarray:
    """Capture screen region containing the game board."""
    # Platform-specific: pyautogui, mss, or similar
    import mss
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = np.array(sct.grab(monitor))
    return screenshot


def detect_board(screenshot: np.ndarray) -> BoardState:
    """
    Detect card positions and identities from screenshot.

    Steps:
    1. Locate game board boundaries
    2. Identify card regions (foundation, tableau, waste, stock)
    3. For each card region, identify rank and suit
    4. Return structured board state
    """
    # Implementation depends on game UI layout
    # This is a placeholder — actual implementation requires:
    # - Reference card templates (one per card face)
    # - Board region detection
    # - Per-card template matching
    raise NotImplementedError("Implement based on actual 1xbet UI")


def board_to_game_state(board: BoardState) -> dict:
    """Convert detected board to KlondikeEnv-compatible state."""
    # Map detected cards to the encoding expected by the model
    # Must handle: partially visible cards, hidden face-down cards
    pass
```

### Reference Images

Create `data/card_templates/` directory with:
- One image per card face (52 images)
- Card back image
- Empty pile image
- Screenshots of various game states for testing

---

## TASK B: Real-Time Advisor

### `klondike/deploy/advisor.py`

```python
import torch
import time
from klondike.nn.model import KlondikeNet
from klondike.encoding import encode_state
from klondike.actions import action_to_str, ACTION_SPACE_SIZE


class MoveAdvisor:
    """
    Real-time move recommendation engine.

    Given the current visible game state, recommends the best move
    using the trained neural network.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = KlondikeNet()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        self.model = torch.compile(self.model, mode='reduce-overhead')

    def recommend(self, game_state) -> dict:
        """
        Get move recommendation from game state.

        Args:
            game_state: KlondikeGame instance or compatible state dict

        Returns:
            dict with:
                "best_move": str (human-readable)
                "confidence": float (model's win probability)
                "top_moves": list of (move_str, probability)
                "latency_ms": float
        """
        start = time.perf_counter()

        obs = encode_state(game_state)
        # Need legal mask — either from game engine or from CV detection
        legal_mask = self._get_legal_mask(game_state)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)
            output = self.model(obs_t, mask_t)

            policy = torch.exp(output["policy"]).squeeze(0).cpu().numpy()
            value = output["value"].squeeze().item()

        # Top moves
        legal_actions = [a for a in range(ACTION_SPACE_SIZE) if legal_mask[a] > 0]
        sorted_actions = sorted(legal_actions, key=lambda a: policy[a], reverse=True)

        top_moves = [(action_to_str(a), float(policy[a]))
                     for a in sorted_actions[:5]]

        latency = (time.perf_counter() - start) * 1000

        return {
            "best_move": action_to_str(sorted_actions[0]) if sorted_actions else "None",
            "confidence": value,
            "top_moves": top_moves,
            "latency_ms": latency,
        }

    def _get_legal_mask(self, game_state):
        """Get legal action mask from game state."""
        # If using KlondikeGame (Rust): game_state.get_legal_moves()
        # If using CV-detected state: derive from board configuration
        # This may need a separate legal move generator
        pass


class GameSession:
    """Track a full game session for logging and analysis."""

    def __init__(self, advisor: MoveAdvisor):
        self.advisor = advisor
        self.moves = []
        self.start_time = time.time()

    def log_move(self, state, recommendation, actual_move, notes=""):
        self.moves.append({
            "timestamp": time.time(),
            "recommendation": recommendation,
            "actual_move": actual_move,
            "followed_advice": recommendation["best_move"] == actual_move,
            "notes": notes,
        })

    def summary(self) -> dict:
        """Session summary statistics."""
        n_moves = len(self.moves)
        followed = sum(1 for m in self.moves if m["followed_advice"])
        avg_latency = np.mean([m["recommendation"]["latency_ms"]
                              for m in self.moves]) if self.moves else 0

        return {
            "total_moves": n_moves,
            "advice_followed": followed,
            "follow_rate": followed / n_moves if n_moves > 0 else 0,
            "avg_latency_ms": avg_latency,
            "session_duration_s": time.time() - self.start_time,
        }
```

### `klondike/deploy/manual_input.py` — Manual State Entry (Fallback)

```python
"""
For when CV doesn't work or isn't set up yet.
Player manually enters the board state via terminal.
"""

def input_board_state() -> dict:
    """Interactive board state input."""
    print("Enter board state:")
    print("Cards: Rank (A,2-9,T,J,Q,K) + Suit (H,D,C,S)")
    print("Face-down cards: use 'X' for unknown")
    print()

    # Foundation
    foundation = []
    for suit_name in ["Hearts", "Diamonds", "Clubs", "Spades"]:
        top = input(f"  Foundation {suit_name} top card (or empty): ").strip()
        foundation.append(_parse_card(top) if top else None)

    # Tableau
    tableau = []
    for i in range(7):
        pile_str = input(f"  Pile {i+1} (space-separated, X for face-down): ").strip()
        cards = [_parse_card(c) for c in pile_str.split()] if pile_str else []
        tableau.append(cards)

    # Waste
    waste_str = input("  Waste top card (or empty): ").strip()
    waste = [_parse_card(waste_str)] if waste_str else []

    # Stock
    stock_count = int(input("  Stock remaining count: ").strip() or "0")

    return {
        "foundation": foundation,
        "tableau": tableau,
        "waste": waste,
        "stock_remaining": stock_count,
    }

def _parse_card(s: str) -> tuple[int, int] | None:
    if not s or s == "X":
        return None
    rank_map = {"A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
                "7": 6, "8": 7, "9": 8, "T": 9, "J": 10, "Q": 11, "K": 12}
    suit_map = {"H": 0, "D": 1, "C": 2, "S": 3}
    rank = rank_map.get(s[0].upper())
    suit = suit_map.get(s[-1].upper())
    if rank is None or suit is None:
        return None
    return (rank, suit)
```

---

## Rule Fidelity Verification

Before relying on the solver for real games, verify that Lonelybot's rules match 1xbet's:

### Checklist
- [ ] Draw-1: confirm stock cycles one card at a time
- [ ] Unlimited passes: confirm no limit on stock cycling
- [ ] Kings-only to empty columns: confirm empty tableau only accepts kings
- [ ] No undo: confirm no undo button available
- [ ] Foundation rules: confirm cards can be moved OFF foundation (StackPile)
- [ ] Multiple card moves: confirm moving sequences between tableau piles
- [ ] Waste pile: confirm top waste card is the only accessible one
- [ ] Win condition: confirm all 52 cards on foundation = win
- [ ] Any hidden timers: check if there's a time limit per game or per move
- [ ] Disconnection: what happens if connection drops mid-game?

### Testing Method
Play 10 games manually, noting any rule difference. For each game, also input the state into Lonelybot and verify legal moves match what 1xbet allows.

---

## Acceptance Criteria

- [ ] Can extract game state from screenshot OR manual input
- [ ] Move advisor produces recommendations in <100ms (CPU) or <10ms (GPU)
- [ ] Recommendations match what the model would play
- [ ] Game session logger tracks all moves and advice-follow rate
- [ ] Rule fidelity verified against 1xbet (at least 10 games)
- [ ] Latency is acceptable for live play

---

## Deployment Considerations

### Account Risk
- Play at normal speed (don't make moves instantly — looks suspicious)
- Add random delays between moves (3-10 seconds, human-like)
- Don't win every session — variance is natural
- Track net profit and stop if flagged

### Bankroll Management
- Kelly criterion: bet fraction = edge / odds = (win_rate - 0.5) / 1.0
- At 52% win rate: Kelly says bet 4% of bankroll per game
- At 51% win rate: Kelly says bet 2% of bankroll per game
- Use fractional Kelly (half-Kelly) for safety

### Expected Profit (if 52% win rate confirmed)
- Edge: 4% per game (win $2 at 52%, lose $1 at 48%, net = $0.04/game)
- At $10/game: $0.40/game profit
- At 60 games/hour: $24/hour gross
- Variance: huge. Need 1000+ games to be statistically confident you're ahead.

---

## Compute & Storage

| Component | Estimate |
|-----------|----------|
| Card templates (52 images) | ~5 MB |
| OpenCV + dependencies | ~200 MB |
| Trained model (CPU inference) | ~10 MB |
| Session logs | ~1 MB/session |
| Total new storage | ~250 MB |
