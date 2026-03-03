# Phase 0: Setup & Reproduce Baselines

**Gate**: Gate 0 — Validate Setup & Reproduce Known Results
**Parallelizable**: No (sequential steps)
**Estimated sessions**: 1-2
**Compute time**: ~2-6 hours (mostly waiting for MCTS runs)

---

## Overview

This phase has NO coding. It's all about building Lonelybot, running it, and getting the critical draw-1 number that determines the project's viability.

**Deliverables:**
1. Lonelybot built and running on PC
2. Draw-3 Thoughtful solvability reproduced (~81.95%)
3. Draw-3 HOP/MCTS reproduced (~47.58%)
4. **Draw-1 MCTS baseline with tight CI** (THE critical number)

---

## Prerequisites

- Rust toolchain installed (`rustup` — https://rustup.rs/)
- Lonelybot cloned to `solitaire/lonelybot/`
- PC with Ryzen 5700X available for long runs

---

## Step-by-Step

### Step 1: Build Lonelybot

```bash
cd solitaire/lonelybot
cargo build --release
```

Verify the binary exists:
```bash
ls -la target/release/lonecli
# or on Windows: target/release/lonecli.exe
```

### Step 2: Explore CLI

**Important**: Run `--help` first and verify all subcommand names before proceeding.
Clap uses kebab-case by default (e.g., `HopLoop` → `hop-loop`). If any commands below
fail, check `--help` output for the correct spelling.

```bash
# Print help — verify all subcommand names here
cargo run --release -p lonecli -- --help

# Print a game state (seed 0, draw-3)
cargo run --release -p lonecli -- print default 0

# Solve a single game (seed 42, draw-3)
cargo run --release -p lonecli -- solve default 42 3

# Play interactive game (seed 100, draw-1)
cargo run --release -p lonecli -- play default 100 1
```

### Step 3: Reproduce Draw-3 Thoughtful Solvability

The exact solver (DFS with full pruning) solves Thoughtful Solitaire — all cards visible.

```bash
# Solve 1000 games sequentially (draw-3, seeds 0-999)
# Use the `rate` command which loops through seeds
cargo run --release -p lonecli -- rate default 0 3
```

**Expected result**: ~81.95% solve rate (matches Solvitaire's 81.945% ± 0.084%)

**Note**: The exact solver can take minutes on hard instances. If `rate` runs too slowly, try smaller batches or use `solve` on individual seeds to verify a handful match known results.

Alternatively, run the solver in parallel across multiple terminal windows:
```bash
# Terminal 1: seeds 0-249
cargo run --release -p lonecli -- rate default 0 3
# Terminal 2: seeds 1000-1249
cargo run --release -p lonecli -- rate default 1000 3
# etc.
```

**Record**: exact solve rate, number of games, 95% CI.

### Step 4: Reproduce Draw-3 HOP/MCTS

The HOP solver uses determinization + search to handle imperfect information.

```bash
# Run HOP on a single game
cargo run --release -p lonecli -- hop default 0 3

# Run HOP loop over sequential seeds
cargo run --release -p lonecli -- hop-loop default 0 3
```

**Expected result**: ~47.58% win rate (Lonelybot's published draw-3 HOP result)

Run at least 1000 games to get a reasonable CI. With 1000 games at p≈0.48:
- 95% CI: ±3.1% (wide but sufficient to confirm we're in the right ballpark)

For tighter CI, run 10,000+ games:
- 95% CI: ±1.0%

**Record**: win rate, number of games, 95% CI.

### Step 5: THE CRITICAL RUN — Draw-1 MCTS

This is the number that determines the project's viability.

```bash
# Run HOP loop with draw-1
cargo run --release -p lonecli -- hop-loop default 0 1
```

**Run at least 10,000 games** for a CI of ±1.0%.
Ideally 50,000 games for CI of ±0.45%.

**Parallelize across terminal windows** (different seed ranges):
```bash
# Terminal 1
cargo run --release -p lonecli -- hop-loop default 0 1
# Terminal 2
cargo run --release -p lonecli -- hop-loop default 10000 1
# Terminal 3
cargo run --release -p lonecli -- hop-loop default 20000 1
# etc. — use all 16 threads
```

**Timing estimate**: At ~2-4 sec/game (MCTS), 50K games across 8 parallel processes ≈ 3-7 hours.

**Record**: win rate, number of games, 95% CI. This is the most important number in the entire project.

### Step 6: Compute Confidence Intervals

```python
import math

def compute_ci(wins, total, z=1.96):
    p = wins / total
    se = math.sqrt(p * (1 - p) / total)
    return p, p - z * se, p + z * se

# Example: 5,100 wins out of 10,000 games
p, lo, hi = compute_ci(5100, 10000)
print(f"Win rate: {p:.4f}, 95% CI: ({lo:.4f}, {hi:.4f})")
```

---

## Acceptance Criteria

- [ ] Lonelybot builds successfully with `cargo build --release`
- [ ] Draw-3 Thoughtful solve rate is within expected range (~80-83%)
- [ ] Draw-3 HOP rate is within expected range (~45-50%)
- [ ] Draw-1 HOP baseline measured with ≥10K games and CI recorded
- [ ] All results recorded in `session_log.md`

---

## Decision Gate

| Draw-1 HOP Result | Action |
|-------------------|--------|
| **≥ 49.5%** | Full speed ahead. Neural guidance plausibly crosses 50%. |
| **48.0% - 49.5%** | Cautious proceed. Neural net needs to add 1-2%. Achievable but thin. |
| **45.0% - 48.0%** | Reassess. Need significant improvement from neural guidance. Still possible but odds drop. |
| **< 45.0%** | Major reassessment. Draw-1 may not be as advantageous as expected. Consider different search approaches before neural nets. |

**Important**: Even a low number does NOT kill the project. It means the specific HOP config underperforms — neural guidance, ISMCTS, or different search parameters could still reach 50%.

---

## Handoff to Phase 1

Phase 1 needs:
1. Confirmed that Lonelybot builds and runs correctly
2. The draw-1 baseline number and CI (informs how much improvement is needed)
3. Understanding of Lonelybot's CLI and behavior
4. Notes on any build issues, platform quirks, or unexpected behavior

Update `current_context.md` with the draw-1 result and `session_log.md` with full details.

---

## CLI Reference (from Lonelybot)

```
COMMANDS:
  exact      Turn seed to exact permutation number
  print      Print game state in Solvitaire JSON format
  bench      Benchmark move generation
  solve      Solve a single game (exact DFS solver)
  randsolve  Solve from intermediate state
  graph      Generate full game graph to CSV
  play       Interactive game loop
  random     Random rollout test (10000 games)
  rate       Statistical solver loop
  hop        Single HOP solve (3000 iters, 1000 node limit)
  hop-loop   Loop HOP solver over sequential seeds

SEED TYPES: default, solvitaire, klondikesolve, greenfelt, exact, microsoft

USAGE: lonecli <SEED_TYPE> <SEED> <DRAW_STEP>
  DRAW_STEP: 1 for draw-1, 3 for draw-3
```
