"""Benchmark runner for Klondike solver experiments.

Plays N games with a given search function, logs results to JSONL,
reports win rate with CI and SPRT status during progress.
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from klondike.env import KlondikeEnv
from klondike.eval.stats import compute_ci, sprt_test


@dataclass
class BenchmarkConfig:
    n_games: int = 10000
    seed_start: int = 0
    draw_step: int = 1
    log_every: int = 100
    output_dir: str = "logs"
    max_moves: int = 500


@dataclass
class GameRecord:
    seed: int
    won: bool
    n_moves: int
    wall_time_ms: float
    search_time_ms: float


def run_benchmark(
    search_fn,
    config: BenchmarkConfig,
    name: str = "unnamed",
) -> dict:
    """Run a benchmark: play N games using the given search function.

    search_fn signature: (env: KlondikeEnv, game_seed: int, move_num: int) -> SearchResult
    SearchResult must have at least: .action (int)

    Returns dict with stats and path to JSONL log file.
    Cycle detection via state hash; max_moves safety limit.
    """
    records = []
    wins = 0

    output_path = Path(config.output_dir) / f"{name}_{config.seed_start}_{config.n_games}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = KlondikeEnv(draw_step=config.draw_step)

    for i in range(config.n_games):
        seed = config.seed_start + i
        game_start = time.perf_counter()
        search_time_total = 0.0

        env.reset(seed=seed)
        n_moves = 0
        seen_states = set()

        while not env.is_done():
            legal = env.get_legal_action_list()
            if not legal:
                break

            # Cycle detection
            state_hash = env._game.encode()
            if state_hash in seen_states:
                break
            seen_states.add(state_hash)

            search_start = time.perf_counter()
            result = search_fn(env, seed, n_moves)
            search_time_total += time.perf_counter() - search_start

            if result.action < 0:
                break

            env.step(result.action)
            n_moves += 1

            if n_moves >= config.max_moves:
                break

        won = env._game.is_win()
        if won:
            wins += 1

        elapsed_ms = (time.perf_counter() - game_start) * 1000
        record = GameRecord(seed, won, n_moves, elapsed_ms, search_time_total * 1000)
        records.append(record)

        with open(output_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        if (i + 1) % config.log_every == 0:
            ci = compute_ci(wins, i + 1)
            sprt = sprt_test(wins, i + 1)
            print(
                f"[{name}] {i+1}/{config.n_games}: "
                f"win_rate={ci.win_rate:.4f} "
                f"CI=({ci.ci_lower:.4f}, {ci.ci_upper:.4f}) "
                f"SPRT={sprt}"
            )

    ci = compute_ci(wins, config.n_games)
    sprt_final = sprt_test(wins, config.n_games)

    return {
        "name": name,
        "n_games": config.n_games,
        "wins": wins,
        "win_rate": ci.win_rate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
        "std_error": ci.std_error,
        "sprt": sprt_final,
        "log_file": str(output_path),
        "records": records,
    }
