from dataclasses import dataclass
import numpy as np

from klondike.actions import ACTION_SPACE_SIZE


@dataclass
class SearchResult:
    """Shared return type for all search functions."""
    action: int                 # Best action index (0-259), or -1 if no legal moves
    visit_counts: np.ndarray    # Shape (ACTION_SPACE_SIZE,) — win rates or visit distribution
    value: float                # Estimated win probability (0-1)
    determinizations_used: int
    simulations_per_det: int
    wall_time_ms: float


from klondike.search.pimc import (
    HopConfig,
    FullHopConfig,
    PIMCConfig,
    hop_search,
    pimc_search,
    play_full_hop_game,
)
from klondike.search.mcts import (
    MCTSConfig,
    MCTSNode,
    EvalFn,
    default_eval_fn,
    mcts_search,
)
