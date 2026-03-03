#!/usr/bin/env python3
"""Run HOP benchmark with per-game timeout and parallel workers.

Usage:
    python3 run_benchmark.py <draw_step> <start_seed> <num_games> [timeout_secs] [workers] [output_file]

Examples:
    python3 run_benchmark.py 3 0 500 120 8           # Draw-3 HOP, 8 workers, 120s timeout
    python3 run_benchmark.py 1 0 10000 120 8 d1.txt   # Draw-1 HOP, save to file
"""
import subprocess
import sys
import time
import math
import signal
from multiprocessing import Pool, Value
import ctypes

LONECLI = "/Users/Daniel/claude/solitaire/lonecli_fast"


def run_hop_game(args):
    """Run a single HOP game. Returns (seed, result, elapsed)."""
    seed, draw_step, timeout_secs = args
    cmd = [LONECLI, "hop", "default", str(seed), str(draw_step)]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_secs
        )
        elapsed = time.time() - t0
        # Verbose output goes to stderr, not stdout
        output = result.stderr.strip()
        if "Solved" in output:
            return (seed, "win", elapsed)
        elif "Lost" in output:
            return (seed, "loss", elapsed)
        else:
            # Fallback: check stdout too
            out = result.stdout.strip()
            if "Solved" in out:
                return (seed, "win", elapsed)
            elif "Lost" in out:
                return (seed, "loss", elapsed)
            return (seed, "unknown", elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return (seed, "timeout", elapsed)


def wilson_ci(n, k, z=1.96):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - spread), p, min(1, center + spread)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    draw_step = int(sys.argv[1])
    start_seed = int(sys.argv[2])
    num_games = int(sys.argv[3])
    timeout_secs = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    output_file = sys.argv[6] if len(sys.argv) > 6 else None

    f = open(output_file, "w") if output_file else None

    wins = 0
    losses = 0
    timeouts = 0
    unknowns = 0
    total = 0
    start_time = time.time()

    print(f"Draw-{draw_step} HOP benchmark: seeds {start_seed}-{start_seed + num_games - 1}")
    print(f"Config: timeout={timeout_secs}s, workers={workers}")
    print(f"{'done':>6} {'wins':>6} {'losses':>6} {'tmout':>6} {'rate':>7} {'CI_lo':>7} {'CI_hi':>7} {'elapsed':>8}")
    print("-" * 70)

    # Build task list
    tasks = [(start_seed + i, draw_step, timeout_secs) for i in range(num_games)]

    # Process results as they come in using imap_unordered for better throughput
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

                # Print progress every 10 games or on wins
                if total % 10 == 0 or result == "win" or total <= 5:
                    wall = time.time() - start_time
                    lo, p, hi = wilson_ci(total, wins)
                    line = (f"{total:>6} {wins:>6} {losses:>6} {timeouts:>6} "
                            f"{p:>7.4f} {lo:>7.4f} {hi:>7.4f} {wall:>7.0f}s")
                    print(line, flush=True)

                if f and (total % 50 == 0 or result == "win"):
                    wall = time.time() - start_time
                    lo, p, hi = wilson_ci(total, wins)
                    f.write(f"{total:>6} {wins:>6} {losses:>6} {timeouts:>6} "
                            f"{p:>7.4f} {lo:>7.4f} {hi:>7.4f} {wall:>7.0f}s\n")
                    f.flush()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            print(f"\nInterrupted at {total} games.")

    elapsed_total = time.time() - start_time
    lo, p, hi = wilson_ci(total, wins)

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS — Draw-{draw_step} HOP")
    print(f"  Seeds: {start_seed} to {start_seed + num_games - 1}")
    print(f"  Games: {total} (wins={wins}, losses={losses}, timeouts={timeouts}, unknown={unknowns})")
    print(f"  Win rate: {wins}/{total} = {p:.4f}")
    print(f"  95% Wilson CI: ({lo:.4f}, {hi:.4f})")
    if timeouts > 0:
        print(f"  Timeouts (treated as losses): {timeouts}/{total} = {timeouts / total:.4f}")
        non_timeout = wins + losses
        if non_timeout > 0:
            lo2, p2, hi2 = wilson_ci(non_timeout, wins)
            print(f"  Win rate (completed only): {wins}/{non_timeout} = {p2:.4f}, CI: ({lo2:.4f}, {hi2:.4f})")
    print(f"  Wall time: {elapsed_total:.1f}s ({elapsed_total / max(total, 1):.1f}s/game, "
          f"{elapsed_total / max(total, 1) * workers:.1f}s/game effective)")

    if f:
        f.write(f"\nFINAL: draw-{draw_step} {wins}/{total}={p:.4f} CI=({lo:.4f},{hi:.4f}) "
                f"timeouts={timeouts} wall={elapsed_total:.0f}s\n")
        f.close()
        print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
