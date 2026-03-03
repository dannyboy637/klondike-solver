#!/usr/bin/env python3
"""Run HOP benchmark for the OLD version of lonecli (draw-3 hardcoded, stdout output).

Usage:
    python3 run_benchmark_old.py <start_seed> <num_games> [timeout_secs] [workers]
"""
import subprocess
import sys
import time
import math
from multiprocessing import Pool

LONECLI = "/Users/Daniel/claude/solitaire/lonecli_v022_patched"


def run_hop_game(args):
    """Run a single HOP game. Returns (seed, result, elapsed)."""
    seed, timeout_secs = args
    # Old version: no draw_step argument, hardcoded to draw-3
    cmd = [LONECLI, "hop", "default", str(seed)]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_secs
        )
        elapsed = time.time() - t0
        # Old version outputs to stdout
        output = result.stdout.strip()
        if "Solved" in output:
            return (seed, "win", elapsed)
        elif "Lost" in output:
            return (seed, "loss", elapsed)
        # Also check stderr just in case
        err = result.stderr.strip()
        if "Solved" in err:
            return (seed, "win", elapsed)
        elif "Lost" in err:
            return (seed, "loss", elapsed)
        return (seed, "unknown", elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return (seed, "timeout", elapsed)


def wilson_ci(n, k, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - spread), p, min(1, center + spread)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    start_seed = int(sys.argv[1])
    num_games = int(sys.argv[2])
    timeout_secs = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    wins = 0
    losses = 0
    timeouts = 0
    unknowns = 0
    total = 0
    start_time = time.time()

    print(f"Draw-3 HOP (OLD version) benchmark: seeds {start_seed}-{start_seed + num_games - 1}")
    print(f"Config: timeout={timeout_secs}s, workers={workers}")
    print(f"{'done':>6} {'wins':>6} {'losses':>6} {'tmout':>6} {'unk':>5} {'rate':>7} {'CI_lo':>7} {'CI_hi':>7} {'elapsed':>8}")
    print("-" * 75)

    tasks = [(start_seed + i, timeout_secs) for i in range(num_games)]

    with Pool(processes=workers) as pool:
        try:
            for seed, result, elapsed in pool.imap_unordered(run_hop_game, tasks, chunksize=1):
                total += 1
                if result == "win":
                    wins += 1
                elif result == "loss":
                    losses += 1
                elif result == "timeout":
                    timeouts += 1
                else:
                    unknowns += 1

                if total % 10 == 0 or result == "win" or total <= 5:
                    wall = time.time() - start_time
                    lo, p, hi = wilson_ci(total, wins)
                    line = (f"{total:>6} {wins:>6} {losses:>6} {timeouts:>6} {unknowns:>5} "
                            f"{p:>7.4f} {lo:>7.4f} {hi:>7.4f} {wall:>7.0f}s")
                    print(line, flush=True)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            print(f"\nInterrupted at {total} games.")

    elapsed_total = time.time() - start_time
    lo, p, hi = wilson_ci(total, wins)

    print(f"\n{'=' * 75}")
    print(f"FINAL RESULTS — Draw-3 HOP (OLD version @ 3e88e66)")
    print(f"  Seeds: {start_seed} to {start_seed + num_games - 1}")
    print(f"  Games: {total} (wins={wins}, losses={losses}, timeouts={timeouts}, unknown={unknowns})")
    print(f"  Win rate: {wins}/{total} = {p:.4f}")
    print(f"  95% Wilson CI: ({lo:.4f}, {hi:.4f})")
    if timeouts > 0:
        print(f"  Timeouts (cycling): {timeouts}/{total} = {timeouts / total:.4f}")
        non_timeout = wins + losses
        if non_timeout > 0:
            lo2, p2, hi2 = wilson_ci(non_timeout, wins)
            print(f"  Win rate (completed only): {wins}/{non_timeout} = {p2:.4f}, CI: ({lo2:.4f}, {hi2:.4f})")
    print(f"  Wall time: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
