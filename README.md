# Klondike Solver

An AlphaZero-style solver for Klondike Solitaire under imperfect information. The goal: exceed 50% win rate on draw-1, unlimited passes, no undo -- turning Klondike into a +EV game at 2x payout.

**Status**: Active research project. Calibrated confidence of reaching >50%: ~30-45%. This is genuinely unexplored territory -- no one has applied modern ML to Klondike as of 2026.

## Why This Matters

All existing Klondike solvers use classical search (constraint programming, DFS, heuristic rule lists). The best published results:

| Method | Draw Mode | Win Rate | Source |
|--------|-----------|----------|--------|
| Thoughtful Solitaire (theoretical ceiling) | draw-1 | ~82-91% | Solvitaire, 1M deals |
| Turnbull heuristic (closed source, **unverified**) | draw-1 | ~53% | Self-reported |
| Lonelybot HOP/MCTS | draw-3 | 47.58% +/- 0.80% | Published, reproducible |
| Best human (no undo) | draw-1 | ~43% | Jupiter Scientific, 442 games |

At 2x payout, every percentage point above 50% is pure edge. Even 51% gives a 2% edge, better than most card counting scenarios.

## Approach

The architecture combines [AlphaZe**](https://github.com/BluemlJ/AlphaZe) (Bluml, Czech & Kersting, Frontiers in AI, 2023) with the [Lonelybot](https://github.com/vuonghy2442/lonelybot) Rust engine:

- **PC-PIMC** (Policy Combining Perfect Information Monte Carlo): sample determinized worlds from hidden card distributions, run MCTS on each, average the resulting policies
- **Neural network** sees only visible cards (not determinized state), forcing generalization across information sets
- **TrueSight Learning**: Phase 1 trains on Thoughtful Solitaire (all cards visible), Phase 2 switches to PIMC with hidden information
- **MLP with residual blocks** (~2M parameters): dual policy/value heads, 757-dim hybrid state encoding with uncertainty summary

PIMC is well-suited to solitaire because there is no adversary exploiting information asymmetry -- hidden cards are placed by the shuffle, not strategically hidden by an opponent.

## Features

### TUI Assistant
An interactive terminal-based Klondike game built with [ratatui](https://ratatui.rs/), with integrated solver suggestions:

```
cargo run -p klondike-assistant --release
```

### Benchmarking Scripts
Python scripts for running solver experiments with statistical rigor:

- `run_benchmark.py` -- parallel HOP benchmark with per-game timeout and Wilson confidence intervals
- `run_ablations.py` -- Phase 2 search-only ablation experiments (HOP, PIMC, MCTS variants)
- `run_hop.py` -- wrapper for Lonelybot's hop-loop with pseudo-tty for line-buffered output

### Research Pipeline
A gated, phased approach from baselines through full AlphaZero self-play:

0. **Setup & Baselines** -- reproduce known results, establish draw-1 baseline
1. **PyO3 Bridge** -- Rust-Python bindings via maturin for the Lonelybot engine
2. **Search Framework** -- PIMC search + MCTS + statistical evaluation
3. **Supervised Learning** -- train policy net on search-generated data
4. **AlphaZero Self-Play** -- full PC-PIMC self-play with neural guidance
5. **Deployment** (optional) -- CV pipeline for real-time play assistance

## Architecture

```
klondike-solver/
├── klondike-rs/              # Pure Rust rebuild (5-crate workspace)
│   └── crates/
│       ├── klondike-core       # Environment, encoding (757-dim), action space
│       ├── klondike-search     # MCTS, PIMC, HOP search algorithms
│       ├── klondike-nn         # Neural net via tch-rs (PyTorch C++ backend)
│       ├── klondike-eval       # Statistical evaluation, benchmarking
│       ├── klondike-train      # Self-play, replay buffer, training pipeline
│       └── klondike-assistant  # TUI game (ratatui + crossterm)
├── klondike/                 # Python package (original implementation)
│   ├── env.py                  # Game environment wrapper
│   ├── encoding.py             # State encoding
│   ├── actions.py              # Action space enumeration
│   ├── search/                 # PIMC, MCTS (Python)
│   ├── eval/                   # Stats, benchmarking
│   ├── nn/                     # MLP-ResNet model + trainer
│   └── alphazero/              # Self-play, replay buffer, pipeline
├── klondike-env/             # Rust PyO3 crate (Python-Rust bridge)
├── lonelybot/                # Lonelybot engine (git submodule, unmodified)
├── scripts/                  # Training & evaluation scripts
├── specs/                    # Phased implementation specs
├── run_benchmark.py          # Parallel HOP benchmarking
├── run_ablations.py          # Search ablation experiments
└── run_hop.py                # Lonelybot hop-loop wrapper
```

### Two Implementations

- **Python** (`klondike/`, `klondike-env/`, `scripts/`): the original implementation, kept as fallback
- **Rust** (`klondike-rs/`): pure Rust rebuild targeting 50-100x speedup via rayon parallelism, arena-allocated MCTS, and batched GPU inference through tch-rs. Direct Lonelybot dependency with no FFI overhead.

## Tech Stack

- **Rust**: game engine (Lonelybot), search algorithms, neural net inference (tch-rs), TUI (ratatui)
- **Python + PyTorch**: original training pipeline, benchmarking scripts
- **PyO3 + maturin**: Rust-Python bridge for the original Python pipeline
- **Lonelybot**: MIT-licensed Klondike engine with HOP, MCTS, and Thoughtful solvers

### Hardware Targets

- **Training**: AMD Ryzen 7 5700X (16 threads) + NVIDIA RTX 4060 (8GB VRAM)
- **Development**: Apple M4 MacBook Pro

## Setup

### Prerequisites

- Rust toolchain (rustup)
- Python 3.10+
- PyTorch (for tch-rs / neural net training)

### Clone

```bash
git clone --recurse-submodules https://github.com/dannyboy637/klondike-solver.git
cd klondike-solver
```

### Build the Rust workspace

```bash
cd klondike-rs
cargo build --release
```

For crates that depend on tch-rs (klondike-nn, klondike-train), set the LibTorch environment:

```bash
LIBTORCH_USE_PYTORCH=1 cargo build --release
```

### Build the PyO3 bridge (for Python pipeline)

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy torch scipy
cd klondike-env
maturin develop --release
```

### Run the TUI assistant

```bash
cargo run -p klondike-assistant --release
```

### Run benchmarks

```bash
# Draw-1 HOP benchmark, 500 games, 8 workers, 120s timeout per game
python run_benchmark.py 1 0 500 120 8
```

## Research Docs

| File | Description |
|------|-------------|
| `CLAUDE.md` | Project context, architecture decisions, session findings |
| `alphazero-klondike-research.md` | Deep dive on PIMC theory, AlphaZe** architecture, state encoding |
| `klondike-master.md` | Compiled research (analysis + discussion + project plan) |
| `current_context.md` | Living research doc with latest findings and open questions |
| `quant-techniques-reference.md` | Variance reduction, Brier score, particle filter reference |
| `specs/` | Phased implementation specs with decision gates |

## Key Findings So Far

- **Draw-1 HOP baseline (true imperfect info)**: 49.0%, CI (45.9%, 52.1%) -- just below breakeven, 50% inside CI
- **Oracle leak discovery**: Lonelybot's `hidden_shuffle()` preserves the topmost hidden card per pile, worth ~19 percentage points. Fixed with `shuffle_all()`.
- **HEAD regression**: Lonelybot at HEAD regresses to ~4% on draw-3 HOP (vs published 47.58%). Pinned to commit `3e88e66` (v0.2.2 era).

## License

Research project. Lonelybot submodule is MIT licensed.
