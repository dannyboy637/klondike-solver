#!/usr/bin/env python3
"""Monitor AlphaZero training progress. Run anytime to check status."""
import json
import sys
from pathlib import Path

def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/alphazero"
    log_path = Path(ckpt_dir) / "training_log.json"

    if not log_path.exists():
        print(f"No training log found at {log_path}")
        return

    with open(log_path) as f:
        log = json.load(f)

    if not log:
        print("Training log is empty.")
        return

    print(f"{'Iter':>4} {'SP Win%':>8} {'P Loss':>8} {'V Loss':>8} {'Buffer':>8} {'s/game':>7} {'Eval':>10}")
    print("-" * 65)

    for entry in log:
        i = entry["iteration"]
        sp = entry["selfplay_win_rate"] * 100
        pl = entry["policy_loss"]
        vl = entry["value_loss"]
        buf = entry["buffer_size"]
        spd = entry.get("avg_game_time", 0)
        eval_str = ""
        if "eval_win_rate" in entry:
            ew = entry["eval_win_rate"] * 100
            lo = entry["eval_ci_lower"] * 100
            hi = entry["eval_ci_upper"] * 100
            eval_str = f"{ew:5.1f}% ({lo:.0f}-{hi:.0f})"

        print(f"{i:4d} {sp:7.1f}% {pl:8.4f} {vl:8.4f} {buf:8d} {spd:6.1f}s {eval_str}")

    # Summary
    last = log[-1]
    total_time = sum(e.get("iter_time", 0) for e in log)
    total_games = sum(1 for _ in log) * 50  # approximate
    evals = [e for e in log if "eval_win_rate" in e]

    print(f"\n--- Summary ---")
    print(f"Iterations: {len(log)}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Buffer: {last['buffer_size']:,} positions")
    print(f"Latest policy loss: {last['policy_loss']:.4f}")
    print(f"Latest value loss: {last['value_loss']:.4f}")

    if evals:
        best = max(evals, key=lambda e: e["eval_win_rate"])
        print(f"Best eval: {best['eval_win_rate']*100:.1f}% at iter {best['iteration']}")

    if len(log) >= 2:
        pl_trend = log[-1]["policy_loss"] - log[0]["policy_loss"]
        print(f"Policy loss trend: {pl_trend:+.4f} ({'improving' if pl_trend < 0 else 'worsening'})")


if __name__ == "__main__":
    main()
