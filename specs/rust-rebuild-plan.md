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

## Team Agent Strategy

**When to use teams**: Phases R1, R3, R4, and R6 have independent sub-tasks that benefit from parallel agents. Phases R2, R5, R7 are tightly coupled and best done single-agent.

**How it works**: Claude Code's TeamCreate tool spawns parallel agents that work in isolated worktrees. The lead agent creates tasks, assigns them to teammates, and merges results. Each teammate gets a focused prompt with just their piece.

**General rules for team phases**:
- The lead agent reads the phase prompt below, creates the team, and splits tasks
- Each teammate works in an isolated worktree (`isolation: "worktree"`)
- Teammates should run `cargo check` after changes (they can't run full `cargo test --all` since tch-rs tests need DYLD_LIBRARY_PATH)
- The lead agent merges worktree branches, resolves any conflicts, and runs the full test + clippy verification
- If a phase says "Team: optional", it's fine to do it single-agent if the work is small enough

---

## Phased Implementation (7 phases)

### Phase R0: Skeleton + Build System -- COMPLETE
- Workspace compiles, Lonelybot links, tch-rs MPS test passes
- tch 0.23 (not 0.17) required for PyTorch 2.10

### Phase R1: Core Environment
**Team: recommended (2 agents)**

Port `env.py`, `encoding.py`, `actions.py`. Verify encoding matches Python bit-for-bit.

Key details:
- Move type mapping: 0=DeckStack, 1=PileStack, 2=DeckPile, 3=StackPile, 4=Reveal
- Card(u8) = rank*4+suit. ACTION_SPACE_SIZE=260, OBSERVATION_DIM=757
- Encoding writes into &mut [f32; 757] (pre-allocated buffer)
- hidden.len(pos) includes topmost hidden card -- face-down count = max(0, hidden.len(pos) - 1)
- CRITICAL TEST: encoding parity vs Python within 1e-6

### Phase R2: MCTS + PIMC Search
**Team: not recommended (pimc depends on mcts, tight coupling)**

Arena-based MCTS (Vec<MctsNode> with u32 indices) + PC-PIMC wrapper.

MctsNode: visit_count(u32), value_sum(f32), prior(f32), children_start(u32), children_count(u16), action(u16)

Known bugs NOT to re-introduce:
- Root visit_count incremented in outer loop, NOT in backpropagation
- Root value = sum(child.value_sum) / sum(child.visit_count)
- Temperature uses LOG-SPACE computation to avoid overflow
- PIMC seeding: game_seed * 10000 + move_num * 100 + d

### Phase R3: Neural Network (tch-rs)
**Team: recommended (2 agents)**

Port KlondikeNet to tch-rs. Load Python weights, verify identical forward pass.

Critical: VarStore parameter naming must match Python state_dict keys exactly.
BatchedInference engine: collects N positions, ONE GPU forward pass.

### Phase R4: Self-Play + Buffer
**Team: recommended (2 agents)**

Parallel self-play with rayon. Start with UniformEval (no GPU).
FIFO replay buffer with serde save/load. 200K capacity ~1GB RAM.

### Phase R5: Training Loop + Neural Self-Play
**Team: not recommended (integrates all crates, sequential by nature)**

Full pipeline: selfplay -> buffer -> train -> eval -> checkpoint.
Training via nn::AdamW, cross-entropy + MSE loss.
Neural self-play with batched inference.

### Phase R6: Evaluation Tools
**Team: recommended (2 agents)**

Wilson CI, SPRT, McNemar. Parallel benchmark runner.
Weight converter (Python -> Rust).

### Phase R7: Polish + Performance
**Team: not recommended (profiling is iterative)**

Profile with cargo flamegraph. Pre-allocate staging tensors.
Target: <5s/game at heavy config (vs 92s Python).

---

## Critical Success Factors

1. **Encoding parity test (Phase R1)** -- if the 757-dim encoding doesn't match Python bit-for-bit, nothing else matters
2. **VarStore naming (Phase R3)** -- must match Python state_dict keys exactly for weight transfer
3. **tch-rs MPS stability (Phase R0)** -- VERIFIED: MPS working with tch 0.23 + PyTorch 2.10
4. **Arena MCTS correctness (Phase R2)** -- fixed-seed deterministic comparison with Python output
5. **Batched inference design (Phase R3-R5)** -- this is where the real speedup lives

---

## Per-Phase Prompts

Each phase gets its own fresh Claude Code session to protect context window.
Always start from: `cd /Users/Daniel/claude/solitaire && claude`

Phases marked **[TEAM]** include a team split section. Use `TeamCreate` to spawn parallel agents.
Phases marked **[SOLO]** are best done single-agent.

---

### Phase R1: Core Environment [TEAM]

```
Read specs/rust-rebuild-plan.md and klondike-rs/CLAUDE.md.

We're in Phase R1 of the Rust rebuild. The workspace skeleton exists at klondike-rs/ (created in Phase R0). All 5 crates compile.

PHASE R1: Port env.py, encoding.py, actions.py from Python to Rust in klondike-core.

Use TeamCreate to parallelize this work with 2 agents:

TEAM SPLIT:
- Agent "encoding" (general-purpose, worktree): Port actions.rs + encoding.rs
- Agent "env" (general-purpose, worktree): Port env.rs
- Lead: write encoding parity test after both agents finish, run final cargo test + clippy

AGENT "encoding" TASKS:
1. Read klondike/actions.py (~80 lines) and klondike-env/src/lib.rs (for Lonelybot API reference)
2. Port to klondike-rs/crates/klondike-core/src/actions.rs:
   - Constants: N_CARDS=52, ACTION_SPACE_SIZE=260
   - move_to_action(move_type, card_idx) -> usize
   - action_to_move(action) -> (u8, u8)
   - move_to_tuple(&Move) -> (u8, u8) -- match on Move enum variants
   - tuple_to_move(type, card) -> Option<Move> -- construct Move from indices
   - Move type mapping: 0=DeckStack, 1=PileStack, 2=DeckPile, 3=StackPile, 4=Reveal
   - Unit tests for round-trip conversion
3. Read klondike/encoding.py (~100 lines) and klondike-env/src/lib.rs:164-170 (hidden card logic)
4. Port to klondike-rs/crates/klondike-core/src/encoding.rs:
   - Constants: OBSERVATION_DIM=757, CARD_FEATURES=13
   - encode_state(game: &Solitaire, obs: &mut [f32; 757]) -- writes into pre-allocated buffer
   - 676 card features: 52 cards x 13 (11 location one-hot + is_visible + pile_position)
   - 14 global features: foundation_ranks, pile_hidden_counts, deck_remaining, waste_size, deck_passes
   - 67 uncertainty features: 52 unseen_card_mask + unseen_red_fraction + 14 per-pile uncertainty
   - CRITICAL: hidden.len(pos) includes topmost hidden card. Face-down count = max(0, hidden.len(pos) - 1)
   - Unit tests for dimension checks
5. Run cargo check -p klondike-core

AGENT "env" TASKS:
1. Read klondike/env.py (~120 lines) and klondike-env/src/lib.rs (for Lonelybot API)
2. Port to klondike-rs/crates/klondike-core/src/env.rs:
   - KlondikeEnv struct: game (Solitaire), draw_step (u8), seed (u64), undo_stack (Vec)
   - new(seed, draw_step) -- uses default_shuffle + Solitaire::new
   - reset(seed) -- reinitialize game
   - step(action) -> Result<(f32, bool)> -- decode action, do_move, check win
   - get_legal_actions() -> Vec<usize> -- list_moves + move_to_action
   - get_legal_mask() -> [f32; 260] -- 1.0 for legal, 0.0 for illegal
   - get_observation() -> [f32; 757] -- calls encode_state
   - clone() via Clone derive
   - shuffle_hidden(rng_seed: u64) -- uses hidden.shuffle_all() (PIMC-safe)
   - encode_state_hash() -> u64 -- via game.encode() for cycle detection
   - Implements Clone trait
3. Run cargo check -p klondike-core

LEAD TASKS (after both agents complete):
1. Merge both worktree branches
2. Write encoding parity test:
   - Create game seed 42 in Python (via existing PyO3: `python -c "from klondike.env import KlondikeEnv; ..."`) and dump 757 floats to a JSON file
   - Create same game in Rust, encode, compare all 757 floats within 1e-6
   - Make 10 moves in both, verify encoding matches after each move
3. Run cargo test --all, cargo clippy --all-targets -- -D warnings
4. Commit

STOP after Phase R1. Do NOT continue to Phase R2.
```

---

### Phase R2: MCTS + PIMC Search [SOLO]

```
Read specs/rust-rebuild-plan.md and klondike-rs/CLAUDE.md.

Phase R2 of the Rust rebuild. klondike-core is complete with env, encoding, actions (from Phase R1).

PHASE R2: Port mcts.py and pimc.py to Rust in klondike-search. Do this single-agent (pimc depends on mcts, tight coupling).

Reference files to read first:
- klondike/search/mcts.py (~280 lines) -- MCTS with UCB1
- klondike/search/pimc.py (~130 lines) -- PC-PIMC wrapper

Port into:
- klondike-rs/crates/klondike-search/src/mcts.rs -- Arena-based MCTS
- klondike-rs/crates/klondike-search/src/pimc.rs -- PC-PIMC
- klondike-rs/crates/klondike-search/src/eval_fn.rs -- EvalFn trait + UniformEval

Key architecture change: Use arena allocation (Vec<MctsNode> with u32 indices) instead of Python's dict of heap-allocated nodes. Benefits: cache locality, no per-node allocation, clear() reuses memory.

MctsNode fields: visit_count(u32), value_sum(f32), prior(f32), children_start(u32), children_count(u16), action(u16)

EvalFn trait: fn evaluate(&self, env: &KlondikeEnv) -> ([f32; 260], f32)

BUGS NOT TO RE-INTRODUCE (from project spec caveats):
- Root visit_count incremented in outer loop, NOT in backpropagation
- Root value = sum(child.value_sum) / sum(child.visit_count), NOT root.value_sum/root.visit_count
- Temperature uses LOG-SPACE computation to avoid float64 overflow (see lines 123-134 of mcts.py)

PIMC details:
- Deterministic seeding: game_seed * 10000 + move_num * 100 + d
- Dirichlet noise only on first determinization (d==0)

Tests:
1. Fixed-seed MCTS with UniformEval (seed 42, 100 sims) -- verify visit distribution is reasonable
2. Play 100 games with PIMC+UniformEval -- no panics, no infinite loops (max_moves=500)
3. Compare MCTS visit counts against Python on same seed if feasible

Run cargo test, cargo clippy. Commit.

STOP after Phase R2. Do NOT continue to Phase R3.
```

---

### Phase R3: Neural Network (tch-rs) [TEAM]

```
Read specs/rust-rebuild-plan.md and klondike-rs/CLAUDE.md.

Phase R3 of the Rust rebuild. klondike-core and klondike-search are complete.

PHASE R3: Port model.py to tch-rs in klondike-nn. Build the batched inference engine.

Use TeamCreate to parallelize with 2 agents:

TEAM SPLIT:
- Agent "model" (general-purpose, worktree): Port KlondikeNet to tch-rs
- Agent "inference" (general-purpose, worktree): Build BatchedInference engine
- Lead: merge, write weight-loading parity test, run full verification

AGENT "model" TASKS:
1. Read klondike/nn/model.py (~350 lines)
2. Port to klondike-rs/crates/klondike-nn/src/model.rs:
   - ResidualBlock: Linear(h,h) -> ReLU -> Linear(h,h) + skip -> ReLU. Fixup init: zero fc2 weights/bias
   - KlondikeNet: input_proj(757->512) -> ReLU -> 4 ResidualBlocks -> policy_head(512->260, log_softmax) + value_head(512->256->1, sigmoid)
   - Aux heads (foundation, facedown): optional, dormant
   - NetConfig struct: input_dim=757, hidden_dim=512, n_blocks=4, action_dim=260
   - CRITICAL: VarStore parameter naming must match Python state_dict keys exactly
     Python names: input_proj.weight, blocks.0.fc1.weight, blocks.0.fc2.weight, blocks.1.fc1.weight, ...
     Use vs / format!("blocks.{i}") / "fc1" in tch-rs
   - Print vs.variables() keys in a test and compare with Python model.state_dict().keys()
3. Test: create model, verify ~2.75M parameters, forward pass with random input
4. Run cargo check -p klondike-nn

AGENT "inference" TASKS:
1. Read the model.rs interface (or the Python model.py for shape reference)
2. Create klondike-rs/crates/klondike-nn/src/inference.rs:
   - BatchedInference struct: holds model reference + device
   - evaluate_batch(observations: &[[f32; 757]], legal_masks: &[[f32; 260]]) -> Vec<([f32; 260], f32)>
   - Collects N positions into staging tensors
   - One GPU forward pass
   - Applies legal mask to log_policy (set illegal to -inf before softmax)
   - Extracts (policy, value) results back to CPU
   - auto_device() function: CUDA > MPS > CPU
3. Test: batch of 10 random observations -> no panics, output shapes correct
4. Run cargo check -p klondike-nn

LEAD TASKS (after both agents complete):
1. Merge both worktree branches
2. Write weight-loading parity test:
   - Load existing Python checkpoint (checkpoints/supervised_v2/best_supervised.pt or checkpoints/alphazero/best_model.pt if exists)
   - Run same 10 input vectors through both Python and Rust
   - Compare log_policy and value outputs within 1e-5
3. Benchmark: 1000 batch-of-64 forward passes -- should be under 50ms each
4. Run cargo test --all, cargo clippy. Commit.

STOP after Phase R3. Do NOT continue to Phase R4.
```

---

### Phase R4: Self-Play + Buffer [TEAM]

```
Read specs/rust-rebuild-plan.md and klondike-rs/CLAUDE.md.

Phase R4 of the Rust rebuild. Core, search, and NN crates are complete.

PHASE R4: Port buffer.py and selfplay.py to Rust in klondike-train.

Use TeamCreate to parallelize with 2 agents:

TEAM SPLIT:
- Agent "buffer" (general-purpose, worktree): Port ReplayBuffer
- Agent "selfplay" (general-purpose, worktree): Port parallel self-play
- Lead: merge, run integration test (100 games -> buffer -> save/load), cargo test + clippy

AGENT "buffer" TASKS:
1. Read klondike/alphazero/buffer.py (~120 lines)
2. Port to klondike-rs/crates/klondike-train/src/buffer.rs:
   - TrainingSample struct: observation [f32; 757], policy [f32; 260], value f32, legal_mask [f32; 260]
   - ReplayBuffer: pre-allocated Vecs, FIFO via index % capacity
   - add(&mut self, sample: TrainingSample)
   - add_game(&mut self, samples: Vec<TrainingSample>)
   - sample_batch(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<&TrainingSample>
   - save(&self, path: &Path) / load(path: &Path) -> Self via serde
   - len(), capacity(), is_empty()
   - 200K capacity x (757+260+1+260) x 4 bytes = ~1.02 GB
3. Unit tests: add samples, verify FIFO eviction, save/load round-trip
4. Run cargo check -p klondike-train

AGENT "selfplay" TASKS:
1. Read klondike/alphazero/selfplay.py (~180 lines)
2. Port to klondike-rs/crates/klondike-train/src/selfplay.rs:
   - SelfPlayConfig: n_determinizations=5, mcts_simulations=200, c_puct=1.5, temperature_moves=20, temperature_final=0.05, max_moves=500
   - GameData struct: samples (Vec<TrainingSample>), won (bool), n_moves (usize)
   - play_game(config, eval_fn, seed) -> GameData
     - Create KlondikeEnv, play with PIMC search, collect (obs, policy, legal_mask) at each step
     - Backfill value: 1.0 for win, 0.0 for loss (applied to all positions after game ends)
     - Temperature: 1.0 for first temperature_moves, then temperature_final
     - Cycle detection: track state hashes, break on repeat
   - play_games_parallel(config, eval_fn, seeds) -> Vec<GameData>
     - Uses rayon::par_iter() over seeds
     - Start with UniformEval first (no GPU needed, validates parallelism)
3. Test: 20 games with UniformEval+PIMC in parallel -- no panics
4. Run cargo check -p klondike-train

LEAD TASKS (after both agents complete):
1. Merge both worktree branches
2. Integration test: run 100 parallel uniform-eval games, add to buffer, save/load, verify
3. Benchmark: 100 uniform-eval games in under 30 seconds (rayon on 8 cores)
4. Run cargo test --all, cargo clippy. Commit.

STOP after Phase R4. Do NOT continue to Phase R5.
```

---

### Phase R5: Training Loop + Neural Self-Play [SOLO]

```
Read specs/rust-rebuild-plan.md and klondike-rs/CLAUDE.md.

Phase R5 of the Rust rebuild. All component crates are complete. This is the integration phase. Do this single-agent (integrates everything, sequential by nature).

PHASE R5: Port pipeline.py and build the full training loop with neural-guided self-play.

Reference files:
- klondike/alphazero/pipeline.py (~250 lines) -- Training pipeline
- scripts/train_alphazero.py (~100 lines) -- CLI entry point

Port into:
- klondike-rs/crates/klondike-train/src/pipeline.rs -- Main training loop
- klondike-rs/crates/klondike-train/src/bin/train.rs -- CLI binary (clap)

Pipeline loop: selfplay -> buffer -> train -> eval -> checkpoint
- Training: nn::AdamW, cross-entropy policy loss + MSE value loss
- Gradient clipping: manual grad().clamp_(-1, 1) per parameter
- Checkpoint: VarStore::save/load (PyTorch-compatible .pt format)
- Training log: JSON file (same format as Python for compatibility with monitor_training.py)
- Eval seeds start at 5,000,000 (disjoint from Phase 6 test sets)

Neural self-play options (pick simpler first):
A) Sequential: each game fully completes before next, but uses batched inference within each game's MCTS
B) Channel-based: worker threads send leaves to GPU thread via channels (more complex, higher throughput)

Start with option A. Upgrade to B if throughput is insufficient.

CLI (clap derive): --iterations, --games-per-iter, --det, --sims, --hidden-dim, --n-blocks, --lr, --batch-size, --buffer-capacity, --eval-every, --eval-games, --checkpoint-dir, --warm-start

Tests:
1. 5 iterations x 10 games x 3d x 50s -> under 5 minutes on MacBook
2. Training log JSON readable by Python monitor_training.py
3. Load Python checkpoint, greedy eval 100 games -> compare win rate
4. Single heavy game (5d x 200s) -> under 5s

Run cargo test, cargo clippy. Commit.

STOP after Phase R5. Do NOT continue to Phase R6.
```

---

### Phase R6: Evaluation Tools [TEAM]

```
Read specs/rust-rebuild-plan.md and klondike-rs/CLAUDE.md.

Phase R6 of the Rust rebuild. The full pipeline works from Phase R5.

PHASE R6: Port stats.py, benchmark tools, and weight converter.

Use TeamCreate to parallelize with 2 agents:

TEAM SPLIT:
- Agent "stats" (general-purpose, worktree): Port stats.rs + benchmark.rs + benchmark CLI
- Agent "tools" (general-purpose, worktree): Port monitor CLI + convert_weights CLI
- Lead: merge, run final validation suite, cargo test + clippy

AGENT "stats" TASKS:
1. Read klondike/eval/stats.py (~150 lines) and klondike/eval/benchmark.py (~100 lines)
2. Port to klondike-rs/crates/klondike-eval/src/stats.rs:
   - wilson_ci(wins, total, confidence) -> (f64, f64, f64) -- (rate, lower, upper)
   - sprt_test(wins, total, p0, p1, alpha, beta) -> SprtResult (accept/reject/continue + log_lr)
   - mcnemar_test(results_a: &[bool], results_b: &[bool]) -> (f64, f64) -- (chi2, p_value)
   - Pure math, no external dependencies
3. Port to klondike-rs/crates/klondike-eval/src/benchmark.rs:
   - BenchmarkConfig: n_games, seeds_start, eval_fn config, max_moves
   - run_benchmark(config, eval_fn) -> BenchmarkResult
   - Uses rayon::par_iter() for parallel games
   - Prints progress bar, Wilson CI at end
4. Port to klondike-rs/crates/klondike-train/src/bin/benchmark.rs:
   - CLI: --games, --seed-start, --det, --sims, --checkpoint, --threads
5. Unit tests for Wilson CI (compare known values), SPRT, McNemar
6. Run cargo check

AGENT "tools" TASKS:
1. Port to klondike-rs/crates/klondike-train/src/bin/monitor.rs:
   - Read training_log.json, print formatted table of iterations
   - Columns: iter, games, self_play_win%, policy_loss, value_loss, eval_win%, time
   - Read scripts/monitor_training.py (~65 lines) for format reference
2. Port to klondike-rs/crates/klondike-train/src/bin/convert_weights.rs:
   - CLI: --input (Python .pt path) --output (Rust .pt path) --hidden-dim --n-blocks
   - Create KlondikeNet with matching architecture
   - VarStore::load(python_checkpoint) -> VarStore::save(rust_format)
   - Print parameter names and shapes for verification
3. Run cargo check

LEAD TASKS (after both agents complete):
1. Merge both worktree branches
2. FINAL VALIDATION:
   - Run 5000-game benchmark with Rust HOP solver -> should produce ~54.96% matching Python
   - Load Python-trained weights via convert_weights -> run greedy eval -> compare win rate
3. Run cargo test --all, cargo clippy. Commit.

STOP after Phase R6. Do NOT continue to Phase R7.
```

---

### Phase R7: Polish + Performance [SOLO]

```
Read specs/rust-rebuild-plan.md and klondike-rs/CLAUDE.md.

Phase R7 of the Rust rebuild. Everything works end-to-end. Do this single-agent (profiling is iterative).

PHASE R7: Profile, optimize, and polish.

1. Profile with cargo flamegraph -- identify hotspots in self-play and training
2. Pre-allocate staging tensors in BatchedInference (reuse instead of create each call)
3. SIMD-friendly encoding (13 contiguous floats per card, verify auto-vectorization)
4. Buffer: consider Box<[f32; N]> for fixed-size arrays
5. Verify Rust checkpoints loadable in Python (bidirectional compatibility)
6. Rustdoc comments on all public APIs

Performance targets:
- Self-play: ~1-2s/game at heavy config (5d x 200s) with 8 threads
- Training iteration (50 games + train): ~2-3 minutes
- Full 100 heavy iterations: ~3-5 hours

7. Final benchmark suite: publish throughput numbers
8. Run cargo clippy, cargo fmt, cargo test.
9. Commit and push.
```

---

## Team Agent Reference

### Creating a Team (lead agent does this)
```
Use TeamCreate with team_name like "r1-core" or "r3-nn".
Then use TaskCreate for each agent's work.
Spawn agents with Task tool using team_name parameter and isolation: "worktree".
```

### Team Phase Checklist
1. Lead reads phase prompt, creates team
2. Lead creates tasks via TaskCreate
3. Lead spawns teammate agents with focused prompts + worktree isolation
4. Teammates work independently, mark tasks complete
5. Lead merges worktree branches
6. Lead runs full cargo test + clippy verification
7. Lead commits

### When Teams Aren't Worth It
- The phase has < 2 truly independent work streams
- Files have circular dependencies (one agent's output is another's input)
- The total work is small enough that team overhead exceeds time saved
- Debugging/profiling (inherently iterative and sequential)
