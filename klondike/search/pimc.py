import time
from dataclasses import dataclass

import numpy as np

from klondike.actions import ACTION_SPACE_SIZE, move_to_action
from klondike.env import KlondikeEnv
from klondike.search import SearchResult


@dataclass
class HopConfig:
    """Config for single-call HOP search (Lonelybot's per-move solver).

    NOTE: This uses hop_solve() which evaluates moves individually.
    For full game-playing, use FullHopConfig + play_full_hop_game().
    """
    n_determinizations: int = 50    # Determinizations inside Rust solver
    node_limit: int = 500           # Max search nodes per determinization


@dataclass
class FullHopConfig:
    """Config for full HOP game-playing (matching lonecli binary behavior).

    Uses pick_moves() which does UCB1 + traverse + HOP at decision points.
    This is the correct algorithm — matches the binary's ~49-55% win rate.
    """
    n_times: int = 500              # Total evaluations per decision (UCB1 budget)
    node_limit: int = 500           # Max search nodes per determinization


@dataclass
class PIMCConfig:
    """Config for outer PIMC loop with Python-level determinization."""
    n_determinizations: int = 10    # Outer determinizations (Python shuffle)
    n_hop_iterations: int = 20      # Inner determinizations per HOP call
    node_limit: int = 200           # Max search nodes per inner determinization


def _empty_result() -> SearchResult:
    return SearchResult(
        action=-1,
        visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.float32),
        value=0.0,
        determinizations_used=0,
        simulations_per_det=0,
        wall_time_ms=0.0,
    )


def hop_search(
    env: KlondikeEnv,
    config: HopConfig,
    game_seed: int = 0,
    move_num: int = 0,
) -> SearchResult:
    """
    Direct wrapper around Lonelybot's HOP solver.

    Calls hop_solve() once on the current state. The Rust solver internally
    does determinization (with oracle-free shuffle_all), so this is already
    an imperfect-info search. No Python-level shuffling needed.

    Returns SearchResult with win rates per action in visit_counts.
    """
    start = time.perf_counter()

    # hop_solve returns dict: (move_type, card_value) -> {"wins", "skips", "played"}
    hop_result = env._game.hop_solve(config.n_determinizations, config.node_limit)

    if not hop_result:
        return _empty_result()

    visit_counts = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    best_action = -1
    best_rate = -1.0

    for (move_type, card_idx), stats in hop_result.items():
        played = stats["played"]
        if played == 0:
            continue
        action = move_to_action(move_type, card_idx)
        win_rate = stats["wins"] / played
        visit_counts[action] = win_rate

        if win_rate > best_rate:
            best_rate = win_rate
            best_action = action

    elapsed_ms = (time.perf_counter() - start) * 1000

    return SearchResult(
        action=best_action,
        visit_counts=visit_counts,
        value=best_rate if best_rate >= 0 else 0.0,
        determinizations_used=config.n_determinizations,
        simulations_per_det=config.node_limit,
        wall_time_ms=elapsed_ms,
    )


def pimc_search(
    env: KlondikeEnv,
    config: PIMCConfig,
    game_seed: int = 0,
    move_num: int = 0,
) -> SearchResult:
    """
    PIMC search: outer Python determinization loop + inner HOP solver.

    Algorithm:
    1. For each outer determinization d:
       a. Clone the environment
       b. Shuffle hidden cards with deterministic seed
       c. Run HOP solver (which does its own inner determinizations)
       d. Accumulate per-move win stats
    2. Average win rates across outer determinizations
    3. Return best action

    Seeding: deterministic game_seed * 10000 + move_num * 100 + d
    """
    start = time.perf_counter()

    # Accumulate win counts per action across outer determinizations
    action_wins = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
    action_played = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)

    for d in range(config.n_determinizations):
        det_env = env.clone()
        rng_seed = game_seed * 10000 + move_num * 100 + d
        det_env.shuffle_hidden(rng_seed)

        # HOP solver evaluates ALL legal moves at once
        hop_result = det_env._game.hop_solve(config.n_hop_iterations, config.node_limit)

        for (move_type, card_idx), stats in hop_result.items():
            played = stats["played"]
            if played == 0:
                continue
            action = move_to_action(move_type, card_idx)
            action_wins[action] += stats["wins"]
            action_played[action] += played

    # Compute average win rate per action
    visit_counts = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    best_action = -1
    best_rate = -1.0

    for action in range(ACTION_SPACE_SIZE):
        if action_played[action] > 0:
            rate = action_wins[action] / action_played[action]
            visit_counts[action] = rate
            if rate > best_rate:
                best_rate = rate
                best_action = action

    elapsed_ms = (time.perf_counter() - start) * 1000

    return SearchResult(
        action=best_action,
        visit_counts=visit_counts,
        value=best_rate if best_rate >= 0 else 0.0,
        determinizations_used=config.n_determinizations,
        simulations_per_det=config.n_hop_iterations,
        wall_time_ms=elapsed_ms,
    )


def play_full_hop_game(
    seed: int,
    config: FullHopConfig,
    draw_step: int = 1,
    max_moves: int = 500,
) -> tuple[bool, int, float]:
    """Play a complete game using the full HOP algorithm (pick_moves).

    This matches the binary's do_hop() approach:
    1. Clone game, shuffle_all hidden cards (determinization)
    2. Run pick_moves (UCB1 bandit + traverse + HOP) on shuffled copy
    3. Play returned move sequence on REAL game
    4. Repeat until win/loss

    Returns: (won, n_moves, wall_time_ms)
    """
    from klondike_env import KlondikeGame

    start = time.perf_counter()
    game = KlondikeGame(seed=seed, draw_step=draw_step)
    move_count = 0
    rng_base = seed * 1000

    while not game.is_win():
        moves = game.play_hop_turn(
            rng_base + move_count,
            config.n_times,
            config.node_limit,
        )
        if not moves:
            break
        move_count += len(moves)
        if move_count > max_moves:
            break

    elapsed_ms = (time.perf_counter() - start) * 1000
    return game.is_win(), move_count, elapsed_ms
