# Klondike AlphaZero Solver — Master Spec Plan

## Project Goal
Build the first AlphaZero-style solver for Klondike Solitaire under imperfect information.
Target: >50% win rate on draw-1, unlimited passes, no undo. At 2x payout, every % above 50% is pure edge.

**Calibrated confidence: ~30-45%.** This is research, not a guaranteed outcome.

---

## Storage Budget (~10-15GB on PC)

| Component | Size |
|-----------|------|
| Rust toolchain (rustup) | ~1.5 GB |
| Lonelybot repo + build artifacts | ~500 MB |
| Python 3.10+ base | ~100 MB |
| PyTorch + CUDA toolkit | ~5 GB |
| Other Python deps (numpy, scipy, maturin) | ~500 MB |
| PyO3 crate build artifacts | ~200 MB |
| Training data (100K games) | ~500 MB - 1 GB |
| Model checkpoints (50 saves) | ~500 MB |
| Replay buffer (saved) | ~1-2 GB |
| Logs + eval results | ~500 MB |
| **Total** | **~10-15 GB** |

**MacBook (<20GB free)**: Spec writing and code review ONLY. All building/training on PC.

---

## Phase Summary

| Phase | Gate | Title | Parallelizable | Est. Coding Time |
|-------|------|-------|----------------|------------------|
| 0 | Gate 0 | Setup & Reproduce Baselines | No | 1-2 sessions |
| 1 | Gate 1 prep | PyO3 Bridge + Python Interface | Yes (Rust + Python) | 2-3 sessions |
| 2 | Gate 1 | Search & Evaluation Framework | Yes (Search + Stats) | 2-3 sessions |
| 3 | Gate 2 | Supervised Learning | Partially (Data + NN) | 2-3 sessions |
| 4 | Gate 3 | AlphaZero Self-Play | Sequential | 3-4 sessions |
| 5 | Gate 4 | Deployment (Optional) | Yes (CV + Advisor) | 2-3 sessions |
| 6 | Post-Gate 3 | Iteration & Optimization | Yes (independent experiments) | Ongoing (weeks-months) |

---

## Dependency Graph

```
Phase 0: Setup & Baselines
    |
    v
Phase 1: PyO3 Bridge                    FOUNDATION CODE
    |                                    (all infra built
    v                                     in Phases 0-4)
Phase 2: Search Framework
    |
    v
Phase 3: Supervised Learning
    |
    v
Phase 4: AlphaZero Self-Play ---------> Phase 6: Iteration & Optimization
    |                                    (tune, experiment, scale — uses
    v                                     all code from Phases 1-4)
Phase 5: Deployment (Optional)
```

**Phases 0-4 = build the complete codebase.** Phase 5 = deployment. Phase 6 = run experiments and optimize using the code already built.

---

## Decision Gates (Go/No-Go)

### After Phase 0
- **Proceed if**: Draw-1 HOP baseline >= 49.5% with tight CI
- **Reassess if**: Draw-1 HOP baseline <= 48.5%
- **Stop if**: Cannot reproduce known draw-3 results (build/methodology issues)

### After Phase 2
- **Proceed if**: Best search-only draw-1 rate has CI upper bound >= 49%
- **Reassess if**: CI upper bound < 49% after full computation budget

### After Phase 3
- **Proceed if**: Supervised net matches search quality at 100x speed
- **Reassess if**: Supervised net significantly worse than search

### After Phase 4
- **Proceed if**: AlphaZero paired CI shows > 50.2% win rate
- **Stop if**: CI upper bound < 50.2% after full compute budget

---

## Project Directory Structure (End State)

```
solitaire/
├── specs/                    # These spec documents
├── lonelybot/                # Cloned repo (unmodified)
├── klondike-env/             # Rust PyO3 crate (new)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs
├── klondike/                 # Python package (new)
│   ├── __init__.py
│   ├── env.py                # Game environment wrapper
│   ├── encoding.py           # State encoding (~857 dims)
│   ├── actions.py            # Action space enumeration
│   ├── search/
│   │   ├── __init__.py
│   │   ├── pimc.py           # PIMC determinization search
│   │   └── mcts.py           # Python MCTS implementation
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── stats.py          # CI, SPRT, variance reduction
│   │   └── benchmark.py      # Benchmark runner
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── model.py          # MLP-ResNet architecture
│   │   └── trainer.py        # Training loop
│   └── alphazero/
│       ├── __init__.py
│       ├── selfplay.py       # Self-play game generation
│       ├── pipeline.py       # Async training pipeline
│       └── buffer.py         # Replay buffer
├── scripts/
│   ├── validate_baselines.sh # Phase 0 validation
│   ├── generate_data.py      # Phase 3 data generation
│   └── train.py              # Main training entry point
├── tests/
│   ├── test_env.py           # Environment tests
│   ├── test_encoding.py      # Encoding tests
│   └── test_search.py        # Search tests
├── data/                     # Training data (gitignored)
├── checkpoints/              # Model checkpoints (gitignored)
├── logs/                     # Training logs (gitignored)
└── [research docs *.md]
```

---

## Hardware Target

**Build & Training Machine (PC):**
- CPU: AMD Ryzen 7 5700X (8 cores / 16 threads)
- RAM: 32 GB
- GPU: NVIDIA RTX 4060 (8 GB VRAM)
- Storage: 200+ GB free
- OS: Likely Windows or Linux (specs are OS-agnostic)

**Dev Machine (MacBook Pro M4):**
- Specs and code review only (<20GB free)

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Theoretical solvability (Thoughtful, draw-3) | 81.95% |
| Lonelybot HOP/MCTS (draw-3) | 47.58% ± 0.80% |
| Lonelybot MCTS (draw-1) | **UNKNOWN — Phase 0 deliverable** |
| Turnbull heuristic (draw-1, UNVERIFIED) | ~53% |
| Breakeven at 2x payout | 50.0% |
| State encoding dimensions | ~857 |
| Action space size | 260 (5 move types × 52 cards, masked) |
| Neural net parameters | ~2M |
| MCTS sims/move (training) | 100-400 |
| PIMC determinizations (training) | 3-5 |
| PIMC determinizations (eval) | 50-100 |
| Target training games | 50K-500K (100K middle) |

---

## How to Use These Specs

1. **Read the phase spec** for the phase you're about to implement
2. Each spec is **self-contained** — all context, interfaces, and types are inline
3. **Team agents**: phases marked "Parallelizable" include TASK A / TASK B splits
4. **Don't skip gates** — each phase has acceptance criteria and a decision gate
5. **Update session_log.md** after each phase with results and decisions
6. **Update current_context.md** with any new findings that change the project direction

---

## Spec Files

| File | Description |
|------|-------------|
| `00-overview.md` | This file — master plan |
| `phase-0-setup-baselines.md` | Clone, build, reproduce, get draw-1 number |
| `phase-1-pyo3-bridge.md` | Rust PyO3 crate + Python game interface |
| `phase-2-search-evaluation.md` | PIMC search + statistical evaluation |
| `phase-3-supervised-learning.md` | Data generation + neural net training |
| `phase-4-alphazero-selfplay.md` | Full AlphaZero self-play pipeline |
| `phase-5-deployment.md` | CV pipeline + real-time advisor |
| `phase-6-iteration-optimization.md` | Hyperparameter tuning, architecture experiments, scaling, when to stop |
