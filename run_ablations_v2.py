#!/usr/bin/env python3
"""Phase 2 Search-Only Ablation Experiments — v2 (with correct pick_moves algorithm).

Uses play_full_hop_game which matches the binary's do_hop() approach.
Varies n_times and node_limit to find optimal search parameters.
"""

import time
import json
from pathlib import Path
from klondike.search.pimc import play_full_hop_game, FullHopConfig
from klondike.eval.stats import compute_ci

N_GAMES = 200
SEED_START = 0

results_table = []


def run_full_hop_experiment(name, config, n_games=N_GAMES):
    """Run a full-game HOP experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name} (n_times={config.n_times}, limit={config.node_limit})")
    print(f"{'='*60}")

    wins = 0
    total_time = 0.0
    start = time.perf_counter()

    for i in range(n_games):
        seed = SEED_START + i
        won, n_moves, game_ms = play_full_hop_game(seed, config, draw_step=1, max_moves=500)
        if won:
            wins += 1
        total_time += game_ms

        if (i + 1) % 50 == 0:
            ci = compute_ci(wins, i + 1)
            elapsed = time.perf_counter() - start
            print(f"  {i+1}/{n_games}: {wins}/{i+1} = {ci.win_rate:.3f} "
                  f"CI({ci.ci_lower:.3f}, {ci.ci_upper:.3f}) [{elapsed:.1f}s]")

    elapsed = time.perf_counter() - start
    ci = compute_ci(wins, n_games)

    entry = {
        "name": name,
        "n_times": config.n_times,
        "node_limit": config.node_limit,
        "wins": wins,
        "n_games": n_games,
        "win_rate": ci.win_rate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
        "wall_time_s": elapsed,
        "sec_per_game": elapsed / n_games,
    }
    results_table.append(entry)
    print(f"  RESULT: {wins}/{n_games} = {ci.win_rate:.3f} CI({ci.ci_lower:.3f}, {ci.ci_upper:.3f})")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_games:.2f}s/game)")


# === Experiment 1: Vary n_times (search budget) with fixed node_limit=500 ===
for nt in [100, 200, 500, 1000, 2000, 3000]:
    run_full_hop_experiment(
        f"hop_nt{nt}_lim500",
        FullHopConfig(n_times=nt, node_limit=500),
    )

# === Experiment 2: Vary node_limit with fixed n_times=500 ===
for lim in [100, 200, 500, 1000, 2000]:
    run_full_hop_experiment(
        f"hop_nt500_lim{lim}",
        FullHopConfig(n_times=500, node_limit=lim),
    )

# === Experiment 3: Heavy config (matching binary's heavy params) ===
run_full_hop_experiment(
    "hop_heavy_nt3000_lim1000",
    FullHopConfig(n_times=3000, node_limit=1000),
)


# === Print summary table ===
print("\n" + "="*110)
print("ABLATION RESULTS SUMMARY (Full HOP with pick_moves)")
print("="*110)
print(f"{'Name':<30} {'n_times':>7} {'limit':>6} {'Wins':>5} {'N':>5} "
      f"{'WinRate':>8} {'CI_Low':>8} {'CI_High':>8} {'s/game':>8}")
print("-"*110)

best_rate = 0.0
best_name = ""

for r in results_table:
    print(f"{r['name']:<30} {r['n_times']:>7} {r['node_limit']:>6} "
          f"{r['wins']:>5} {r['n_games']:>5} {r['win_rate']:>8.3f} "
          f"{r['ci_lower']:>8.3f} {r['ci_upper']:>8.3f} {r['sec_per_game']:>8.2f}")
    if r['win_rate'] > best_rate:
        best_rate = r['win_rate']
        best_name = r['name']

print("-"*110)
print(f"Best: {best_name} at {best_rate:.3f}")

# Save results
output_file = Path("logs/ablations_v2/summary.json")
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
name = best_entry['name']

print(f"Best config: {name}")
print(f"Win rate: {wr:.3f}, CI: ({ci_low:.3f}, {ci_high:.3f})")

if ci_low > 0.50:
    print("=> CI lower bound > 50%")
    print("=> SEARCH ALONE IS SUFFICIENT! Neural net is bonus optimization.")
    print("=> GATE 1: PASS")
elif ci_high >= 0.50 and ci_low < 0.50:
    print("=> CI includes 50%")
    print("=> PROMISING. Neural net guidance likely pushes it over. PROCEED.")
    print("=> GATE 1: CONDITIONAL PASS")
elif ci_high >= 0.48:
    print("=> CI upper bound between 48-50%")
    print("=> Neural net needs 0-2%. Achievable. PROCEED CAUTIOUSLY.")
    print("=> GATE 1: MARGINAL")
else:
    print(f"=> CI upper bound {ci_high:.3f} < 48%")
    print("=> MAJOR GAP. Reassess feasibility.")
    print("=> GATE 1: FAIL")
