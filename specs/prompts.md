# Claude Code Prompts — Copy-Paste Per Phase

Each prompt below is self-contained. Paste it into a fresh Claude Code session to start that phase. Claude will read the spec file and begin implementation.

---

## Phase 0: Setup & Baselines

```
Read the spec file at specs/phase-0-setup-baselines.md and execute it step by step.

We're starting Gate 0 of the Klondike AlphaZero project. Lonelybot is already cloned at lonelybot/. Your job:

1. Build Lonelybot with cargo build --release
2. Run --help FIRST to verify CLI subcommand names (Clap uses kebab-case: hop-loop, not hoploop)
3. Explore the CLI — run a few test commands (print, solve, hop-loop) to verify it works
4. Run draw-3 Thoughtful solver on ~1000 games to reproduce ~81.95% solvability
5. Run draw-3 HOP to reproduce ~47.58% win rate
6. Run draw-1 HOP on 10K+ games (parallelize across terminals if possible) — this is THE critical number
7. Compute 95% confidence intervals for all results
8. Record everything in session_log.md and update current_context.md with the draw-1 result

IMPORTANT: CLI commands use kebab-case (hop-loop not hoploop). Always verify with --help first.

Check the decision gate table in the spec after getting the draw-1 number.
```

---

## Phase 1: PyO3 Bridge + Python Interface

### If using team agents (recommended):

```
Read specs/phase-1-pyo3-bridge.md thoroughly. This phase has two parallel tasks.

Create a team with two agents:
- TASK A (Rust): Build the klondike-env/ PyO3 crate that wraps Lonelybot for Python
- TASK B (Python): Build the klondike/ Python package with env.py, encoding.py, actions.py

The shared interface is defined in the spec under "SHARED INTERFACE" — both agents must match it.

Task A creates klondike-env/ with Cargo.toml, pyproject.toml, and src/lib.rs. It wraps SolitaireEngine<FullPruner> with PyO3 #[pyclass] and exposes: reset, get_legal_moves, do_move, undo_move, is_win, get_foundation, get_piles, get_hidden_counts, get_deck_info, get_visible_cards, clone, shuffle_hidden, hop_solve, encode.

Task B creates klondike/ with env.py (gym-like KlondikeEnv wrapper), encoding.py (757-dim state encoding), and actions.py (260-action space mapping). Task B can use stubs/mocks for the Rust module initially.

After both tasks complete, integrate: maturin develop --release, then run all 4 integration tests from the spec. Benchmark should show >1000 random games/sec.

IMPORTANT: Encoding is 757-dim baseline (52 cards × 13 features). Stock and unknown are SEPARATE one-hot slots (index 9 and 10 respectively) — do not conflate them.

Update session_log.md with results.
```

### If working solo:

```
Read specs/phase-1-pyo3-bridge.md thoroughly.

Build the PyO3 bridge between Lonelybot (Rust) and Python in two steps:

Step 1 — Rust crate (klondike-env/):
Create the PyO3 crate that wraps Lonelybot's SolitaireEngine. Follow the Cargo.toml, pyproject.toml, and lib.rs specs exactly. Key methods: reset, get_legal_moves, do_move, undo_move, is_win, get_foundation, get_piles, get_hidden_counts, get_deck_info, get_visible_cards, clone, shuffle_hidden.

Step 2 — Python package (klondike/):
Create env.py (KlondikeEnv with gym-like interface), encoding.py (757-dim observation vector), actions.py (260-action space). Follow the code templates in the spec.

Build with maturin develop --release, run all 4 tests from the spec. Target: >1000 random games/sec.

IMPORTANT: Verify Card encoding — check if Card(u8) uses rank*4+suit or suit*13+rank internally. Get this right or everything downstream breaks.

Encoding is 757-dim baseline: 52 cards × 13 features (11 location one-hot with stock/unknown separated + visible bit + position). Stock and unknown must be SEPARATE one-hot slots.
```

---

## Phase 2: Search & Evaluation Framework

### If using team agents:

```
Read specs/phase-2-search-evaluation.md thoroughly. This phase has two parallel tasks.

Create a team with two agents:
- TASK A (Search): Build klondike/search/pimc.py and klondike/search/mcts.py
- TASK B (Stats): Build klondike/eval/stats.py and klondike/eval/benchmark.py

The shared interface is defined in the spec: SearchResult dataclass and SearchFn callable type.

Task A implements:
1. PIMC search wrapping Lonelybot's HOP solver via PyO3
2. Python MCTS framework with pluggable eval_fn (default_eval_fn for now, neural net later)
3. Both return SearchResult with action, visit_counts, value

Task B implements:
1. compute_ci, sample_size_needed, sprt_test, paired_comparison
2. control_variate_estimate for variance reduction
3. BenchmarkConfig + run_benchmark that plays N games with any search function and logs results to JSONL

After both tasks complete, run the ablation experiments from the spec:
- Vary determinization count (1, 3, 5, 10, 20, 50)
- Vary search budget (100, 500, 1000, 3000, 5000 iterations)
- MCTS with uniform policy baseline

IMPORTANT implementation notes:
- MCTS root visit count: root.visit_count must be incremented after each _simulate() call (backprop only updates children)
- PIMC seeding: use deterministic seeds (game_seed * 10000 + move_num * 100 + d), NOT wall-clock time
- max_moves is 500 (not 2000) — unlimited passes means cycle detection matters more than move limits

Record best search-only win rate on draw-1 with CI. Check decision gate.
Update session_log.md and current_context.md.
```

### If working solo:

```
Read specs/phase-2-search-evaluation.md thoroughly.

Build the search and evaluation framework:

1. klondike/search/pimc.py — PIMC search using Lonelybot's HOP solver via PyO3. Sample N determinized worlds (clone + shuffle_hidden), run HOP on each, average win rates per action.

2. klondike/search/mcts.py — Python MCTS with pluggable eval_fn. This will accept a neural net in Phase 4 — for now use default_eval_fn (uniform policy, 0.5 value). Implement MCTSNode, UCB selection, expansion, backprop.

3. klondike/eval/stats.py — compute_ci, sprt_test, paired_comparison, control_variate_estimate. Follow the code in the spec.

4. klondike/eval/benchmark.py — run_benchmark that plays N games with any SearchFn, logs to JSONL, prints progress with running CI.

Then run at least one ablation: vary determinization count on draw-1 (1, 5, 10, 50). Record best search-only win rate with CI.

IMPORTANT implementation notes:
- MCTS root visit count: root.visit_count must be incremented after each _simulate() call (backprop only updates children)
- PIMC seeding: use deterministic seeds (game_seed * 10000 + move_num * 100 + d), NOT wall-clock time
- max_moves is 500 — with unlimited passes, add cycle detection (state hash set) to catch infinite loops

Update docs with results.
```

---

## Phase 3: Supervised Learning

### If using team agents:

```
Read specs/phase-3-supervised-learning.md thoroughly. This phase has two tasks (partially parallel).

Create a team:
- TASK A (Data): Generate training data by running the search solver on 10K-50K games
- TASK B (NN): Build KlondikeNet architecture and training loop

Task A: Create scripts/generate_data.py following the spec. Use the best PIMC config from Phase 2. For each game, record (observation, search_policy, legal_mask) at every move, then label all positions with game outcome. Save as numpy memmap arrays in data/supervised/ (NOT append+concatenate — use memmap to avoid RAM blowup). Also save game_offsets.npy to track game boundaries. Start with 10K games (can scale later).

Task B: Create klondike/nn/model.py (MLP-ResNet with 4 residual blocks × 512 hidden, dual policy+value heads, Fixup init, auxiliary targets, ~2.75M params) and klondike/nn/trainer.py (KlondikeDataset, train_supervised with AdamW + cosine LR). Task B can start immediately using synthetic random data to verify the architecture works, then switch to real data when Task A delivers.

After data is ready and model is built: train for 50 epochs. IMPORTANT: Use game-level train/val split (not position-level random_split — same-game positions must stay together to prevent data leakage). Evaluate policy accuracy and Brier score on validation set. Then benchmark: greedy policy vs search baseline using paired seeds.

Check decision gate. Update docs.
```

### If working solo:

```
Read specs/phase-3-supervised-learning.md thoroughly.

Two-step phase:

Step 1 — Generate training data:
Create scripts/generate_data.py. Run PIMC search on 10K+ games (use best config from Phase 2). Record (observation, policy, legal_mask, value) for every position. Save as memmap .npy files in data/supervised/ (NOT append+concatenate — use np.lib.format.open_memmap to avoid RAM blowup). Also save game_offsets.npy to track game boundaries for train/val splitting.

Step 2 — Build and train neural net:
Create klondike/nn/model.py — KlondikeNet: MLP-ResNet (input → 512-dim projection → 4 residual blocks → policy head (260 actions) + value head (scalar)). Use Fixup init, no batch norm. ~2.75M params.

Create klondike/nn/trainer.py — supervised training with AdamW, cosine LR, policy cross-entropy + value MSE loss.

IMPORTANT: Use game-level train/val split (group positions by game, not random_split). Same-game positions in both train and val = data leakage. Use game_offsets.npy to split at game boundaries.

Train 50 epochs. Then evaluate:
1. Validation policy top-1 accuracy (does net pick same move as search?)
2. Brier score on value predictions
3. Greedy policy win rate on draw-1 (play games using only the net, no search)
4. Compare greedy vs search baseline with paired seeds

PyTorch MPS backend works fine on M4 for this model size. Use device = "mps" if on Mac, "cuda" on PC.

Check decision gate. Update docs.
```

---

## Phase 4: AlphaZero Self-Play

```
Read specs/phase-4-alphazero-selfplay.md thoroughly.

This is the main training phase. Build the full AlphaZero pipeline:

1. klondike/alphazero/buffer.py — ReplayBuffer with fixed capacity (200K positions), add_game, sample_batch, save/load.

2. klondike/alphazero/selfplay.py — self_play_game using PC-PIMC: at each move, clone+shuffle_hidden for D determinizations, run MCTS with neural eval on each, average visit distributions, select action. Record (obs, policy, mask) per move, label with outcome.

3. klondike/alphazero/pipeline.py — run_pipeline: the main loop. Each iteration: generate 200 self-play games → add to buffer → train 20 mini-batches → evaluate every 5 iters on 1000 fixed seeds (starting at seed 5,000,000 — disjoint from Phase 6 test sets). Save checkpoints. Resume from checkpoint if interrupted.

4. scripts/train.py — entry point.

Initialize model from supervised weights (Phase 3's best_supervised.pt) for warm start.

TrueSight curriculum: optionally start with Phase 1 (Thoughtful Solitaire — set n_determinizations=1, use hidden_clear instead of hidden_shuffle) for 10-20 iterations, then switch to Phase 2 (PIMC with hidden cards).

Enable FP16 mixed precision with torch.amp for 2x speedup.

IMPORTANT implementation notes:
- Eval seeds start at 5,000,000 (disjoint from Phase 6: test_set_A=1M, test_set_B=2M, test_set_C=3M)
- Add cycle detection: track state hashes per game, break on repeat (treat as loss)
- PIMC seeding must be deterministic (game_seed * 10000 + move_num * 100 + d)
- max_moves = 500 per game
- Runtime estimate: ~20-60s/game with PIMC+MCTS, total ~170-200 hours for 100 iterations × 200 games on 4060+5700X. Plan accordingly.
- Gate criterion: 1000 eval games needs ~53% point estimate for CI lower bound >50%. Consider SPRT or 10K+ games for thin margins.

Start a training run. Monitor self-play win rate and eval win rate. Check decision gate after sufficient iterations.

NOTE: This phase is compute-heavy. If on MacBook, do a short run (10-20 iterations) to verify the pipeline works, then transfer to PC for the full run.

Update docs with training curves and eval results.
```

---

## Phase 5: Deployment (Optional)

```
Read specs/phase-5-deployment.md thoroughly. Only start this if Phase 4 achieved >50% win rate.

Two parallel tasks:

TASK A — Card recognition:
Start simple with manual input (klondike/deploy/manual_input.py) so the advisor is usable immediately. Then optionally build CV with OpenCV template matching.

TASK B — Move advisor:
Create klondike/deploy/advisor.py — MoveAdvisor class that loads trained model, takes game state, returns top move recommendations with confidence and latency. Add GameSession logger for tracking advice-follow rate.

Also: run the rule fidelity checklist from the spec against 1xbet. Document any differences.

Test end-to-end: manually input a game state → get move recommendation → verify it makes sense.
```

---

## Phase 6: Iteration & Optimization

```
Read specs/phase-6-iteration-optimization.md thoroughly.

This phase is experiment-driven, not a single build task. Pick experiments based on current results:

IF win rate is close to 50% (48-51%):
→ Start with Tier 1: one-at-a-time hyperparameter sweep (~20 configs total, NOT grid search). Try different MCTS sims (100-800), determinizations (3-10), learning rates (3e-4 to 3e-3). Train each config for 20 iterations, eval on test_set_C (seeds 3M-3.005M).

IF win rate is plateaued below 48%:
→ Try Tier 2: encoding improvements (add stock-pass, action history), architecture variants (deeper/wider net, transformer), ISMCTS vs HOP comparison.

IF win rate is above 50% but want more:
→ Try Tier 3: batched NN inference in MCTS (5-10x throughput), multiprocess self-play, playout cap randomization.

IF nothing works after weeks:
→ Consider Tier 4 pivots: MuZero, direct RL, ensemble.

CRITICAL: Maintain evaluation discipline. Use fixed test sets. Don't tune on your final eval set. Record every experiment in the tracking spreadsheet format from the spec.

Update session_log.md and current_context.md after each significant experiment.
```
