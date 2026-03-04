# Plan: Rebuild AlphaZero Pipeline in Pure Rust

## Context

The Python AlphaZero pipeline (Phase 4) works but is too slow for heavy experimentation. A single heavy-config self-play game takes ~92s in Python. With 100 iterations x 200 games, that's 500+ hours on MacBook. The GIL prevents true parallelism.

**Goal**: Rebuild the entire pipeline in pure Rust (no Python at all) for 50-100x speedup. The key wins are: true parallel self-play (rayon), batched GPU inference (tch-rs), zero FFI overhead (Lonelybot is already Rust), and arena-allocated MCTS trees.

**Constraint**: Keep the existing Python project as-is. The Rust rebuild lives in `klondike-rs/`. If it fails, we fall back to Python.

---

## Architecture Overview

| Python Module | Rust Crate | Key Change |
|---|---|---|
| `klondike/env.py` | `klondike-core` | Direct Lonelybot calls, no FFI |
| `klondike/encoding.py` | `klondike-core` | Write into `[f32; 757]`, zero-copy to tensor |
| `klondike/actions.py` | `klondike-core` | Trivial arithmetic, zero-cost |
| `klondike/search/mcts.py` | `klondike-search` | Arena-allocated `Vec<Node>` with indices |
| `klondike/search/pimc.py` | `klondike-search` | Parallel determinizations via rayon |
| `klondike/nn/model.py` | `klondike-nn` | `tch::nn` with VarStore, batched inference |
| `klondike/alphazero/buffer.py` | `klondike-train` | FIFO ring buffer, serde save/load |
| `klondike/alphazero/selfplay.py` | `klondike-train` | Rayon parallel + batched GPU eval |
| `klondike/alphazero/pipeline.py` | `klondike-train` | Full training loop, checkpoint management |
| `klondike/eval/stats.py` | `klondike-eval` | Pure math, no deps |
| `scripts/train_alphazero.py` | `klondike-train` bin | clap CLI |

**Biggest speedup**: Batched leaf evaluation. Python calls `eval_fn(env)` per leaf (Python->PyO3->Rust->Python->GPU). Rust collects N leaves across M parallel MCTS trees, does ONE GPU forward pass.

---

## Directory Structure

```
solitaire/
  lonelybot/              # Existing submodule (unchanged)
  klondike/               # Existing Python package (unchanged)
  klondike-env/           # Existing PyO3 bridge (unchanged)
  klondike-rs/            # Pure Rust rebuild
    Cargo.toml            # Workspace root
    CLAUDE.md             # Rust-specific conventions
    crates/
      klondike-core/      # Game env, encoding, action mapping
      klondike-search/    # MCTS, PIMC
      klondike-nn/        # Neural network (tch-rs)
      klondike-eval/      # Statistics, benchmarking
      klondike-train/     # Self-play, buffer, pipeline, CLI binaries
```

## Build Environment

```bash
# Required env vars for tch-rs to find PyTorch:
export LIBTORCH_USE_PYTORCH=1
export VIRTUAL_ENV=/path/to/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export DYLD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.14/site-packages/torch/lib

# Then:
cd klondike-rs
cargo build --release
cargo test --all
cargo clippy --all-targets -- -D warnings
```

---

## Phased Implementation (7 phases)

### Phase R0: Skeleton + Build System -- COMPLETE
- Workspace compiles, Lonelybot links, tch-rs MPS test passes
- tch 0.23 (not 0.17) required for PyTorch 2.10

### Phase R1: Core Environment
Port `env.py`, `encoding.py`, `actions.py`. Verify encoding matches Python bit-for-bit.

Key details:
- Move type mapping: 0=DeckStack, 1=PileStack, 2=DeckPile, 3=StackPile, 4=Reveal
- Card(u8) = rank*4+suit. ACTION_SPACE_SIZE=260, OBSERVATION_DIM=757
- Encoding writes into &mut [f32; 757] (pre-allocated buffer)
- hidden.len(pos) includes topmost hidden card -- face-down count = max(0, hidden.len(pos) - 1)
- CRITICAL TEST: encoding parity vs Python within 1e-6

### Phase R2: MCTS + PIMC Search
Arena-based MCTS (Vec<MctsNode> with u32 indices) + PC-PIMC wrapper.

MctsNode: visit_count(u32), value_sum(f32), prior(f32), children_start(u32), children_count(u16), action(u16)

Known bugs NOT to re-introduce:
- Root visit_count incremented in outer loop, NOT in backpropagation
- Root value = sum(child.value_sum) / sum(child.visit_count)
- Temperature uses LOG-SPACE computation to avoid overflow
- PIMC seeding: game_seed * 10000 + move_num * 100 + d

### Phase R3: Neural Network (tch-rs)
Port KlondikeNet to tch-rs. Load Python weights, verify identical forward pass.

Critical: VarStore parameter naming must match Python state_dict keys exactly.
BatchedInference engine: collects N positions, ONE GPU forward pass.

### Phase R4: Self-Play + Buffer
Parallel self-play with rayon. Start with UniformEval (no GPU).
FIFO replay buffer with serde save/load. 200K capacity ~1GB RAM.

### Phase R5: Training Loop + Neural Self-Play
Full pipeline: selfplay -> buffer -> train -> eval -> checkpoint.
Training via nn::AdamW, cross-entropy + MSE loss.
Neural self-play with batched inference.

### Phase R6: Evaluation Tools
Wilson CI, SPRT, McNemar. Parallel benchmark runner.
Weight converter (Python -> Rust).

### Phase R7: Polish + Performance
Profile with cargo flamegraph. Pre-allocate staging tensors.
Target: <5s/game at heavy config (vs 92s Python).

---

## Critical Success Factors

1. **Encoding parity test (Phase R1)** -- if the 757-dim encoding doesn't match Python bit-for-bit, nothing else matters
2. **VarStore naming (Phase R3)** -- must match Python state_dict keys exactly for weight transfer
3. **tch-rs MPS stability (Phase R0)** -- VERIFIED: MPS working with tch 0.23 + PyTorch 2.10
4. **Arena MCTS correctness (Phase R2)** -- fixed-seed deterministic comparison with Python output
5. **Batched inference design (Phase R3-R5)** -- this is where the real speedup lives

## Per-Phase Prompts

See the full plan document or `specs/rust-rebuild-plan.md` for copy-paste prompts for each phase.
Each phase gets its own fresh Claude Code session to protect context window.
