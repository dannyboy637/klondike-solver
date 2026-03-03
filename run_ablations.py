#!/usr/bin/env python3
"""Phase 2 Search-Only Ablation Experiments.

Runs HOP, PIMC, and MCTS variants on draw-1 Klondike.
Uses 200 games per config with paired seeds (start=0) for direct comparison.
"""

import sys
import time
import json
from pathlib import Path

from klondike.env import KlondikeEnv
from klondike.search import (
    hop_search, pimc_search, mcts_search,
    SearchResult, HopConfig, PIMCConfig, MCTSConfig,
)
from klondike.eval.benchmark import run_benchmark, BenchmarkConfig
from klondike.eval.stats import compute_ci

N_GAMES = 200
SEED_START = 0
LOG_EVERY = 50

results_table = []


def run_experiment(name, search_fn, n_games=N_GAMES):
    """Run a single experiment and record results."""
    config = BenchmarkConfig(
        n_games=n_games,
        seed_start=SEED_START,
        draw_step=1,
        log_every=LOG_EVERY,
        output_dir="logs/ablations",
        max_moves=500,
    )
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    start = time.perf_counter()
    result = run_benchmark(search_fn, config, name=name)
    elapsed = time.perf_counter() - start

    entry = {
        "name": name,
        "wins": result["wins"],
        "n_games": result["n_games"],
        "win_rate": result["win_rate"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "sprt": result["sprt"],
        "wall_time_s": elapsed,
        "sec_per_game": elapsed / n_games,
    }
    results_table.append(entry)

    print(f"\nResult: {result['wins']}/{result['n_games']} = {result['win_rate']:.4f}")
    print(f"CI: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
    print(f"SPRT: {result['sprt']}")
    print(f"Time: {elapsed:.1f}s ({elapsed/n_games:.2f}s/game)")
    return result


# === Experiment 1: HOP baseline ===
# This should match Phase 0's draw-1 full-shuffle result (~49%)

def hop_search_fn(env, game_seed, move_num):
    return hop_search(env, HopConfig(n_determinizations=50, node_limit=500), game_seed, move_num)

run_experiment("hop_baseline_50det_500lim", hop_search_fn)


# === Experiment 2: Vary determinization count ===
for n_det in [1, 3, 5, 10, 20, 50]:
    def make_hop_fn(nd):
        def fn(env, game_seed, move_num):
            return hop_search(env, HopConfig(n_determinizations=nd, node_limit=500), game_seed, move_num)
        return fn
    run_experiment(f"hop_det{n_det}", make_hop_fn(n_det))


# === Experiment 3: Vary search budget (node_limit) ===
for limit in [100, 500, 1000, 3000, 5000]:
    def make_hop_fn(lim):
        def fn(env, game_seed, move_num):
            return hop_search(env, HopConfig(n_determinizations=10, node_limit=lim), game_seed, move_num)
        return fn
    run_experiment(f"hop_lim{limit}", make_hop_fn(limit))


# === Experiment 4: PIMC (outer determinization + inner HOP) ===
for n_outer in [3, 5, 10]:
    def make_pimc_fn(no):
        def fn(env, game_seed, move_num):
            return pimc_search(env, PIMCConfig(n_determinizations=no, n_hop_iterations=10, node_limit=200), game_seed, move_num)
        return fn
    run_experiment(f"pimc_outer{n_outer}", make_pimc_fn(n_outer))


# === Experiment 5: MCTS with uniform policy ===
for n_sims in [100, 200, 400]:
    def make_mcts_fn(ns):
        def fn(env, game_seed, move_num):
            return mcts_search(env, MCTSConfig(n_simulations=ns, temperature=0.0))
        return fn
    run_experiment(f"mcts_uniform_{n_sims}", make_mcts_fn(n_sims))


# === Print summary table ===
print("\n" + "="*100)
print("ABLATION RESULTS SUMMARY")
print("="*100)
print(f"{'Name':<30} {'Wins':>5} {'N':>5} {'WinRate':>8} {'CI_Low':>8} {'CI_High':>8} {'SPRT':>12} {'s/game':>8}")
print("-"*100)

best_rate = 0.0
best_name = ""

for r in results_table:
    print(f"{r['name']:<30} {r['wins']:>5} {r['n_games']:>5} {r['win_rate']:>8.4f} {r['ci_lower']:>8.4f} {r['ci_upper']:>8.4f} {r['sprt']:>12} {r['sec_per_game']:>8.2f}")
    if r['win_rate'] > best_rate:
        best_rate = r['win_rate']
        best_name = r['name']

print("-"*100)
print(f"Best: {best_name} at {best_rate:.4f}")

# Save results
output_file = Path("logs/ablations/summary.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(results_table, f, indent=2)
print(f"\nResults saved to {output_file}")

# === Gate 1 Decision ===
print("\n" + "="*60)
print("GATE 1 DECISION")
print("="*60)

best_entry = max(results_table, key=lambda r: r['win_rate'])
ci_low = best_entry['ci_lower']
ci_high = best_entry['ci_upper']
wr = best_entry['win_rate']

if ci_low > 0.50:
    print(f"CI lower bound {ci_low:.4f} > 50%")
    print("SEARCH ALONE IS SUFFICIENT! Neural net is bonus optimization.")
elif ci_high >= 0.50 and ci_low < 0.50:
    print(f"CI ({ci_low:.4f}, {ci_high:.4f}) includes 50%")
    print("PROMISING. Neural net guidance likely pushes it over. PROCEED.")
elif ci_high >= 0.48:
    print(f"CI upper bound {ci_high:.4f} between 48-50%")
    print("Neural net needs to add 0-2%. Achievable. PROCEED CAUTIOUSLY.")
else:
    print(f"CI upper bound {ci_high:.4f} < 48%")
    print("MAJOR GAP. Reassess feasibility — may need different search approach.")
