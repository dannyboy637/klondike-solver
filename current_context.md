# Current Context — Living Research Document

Last updated: 2026-03-04 (Phase 2 complete — Gate 1 PASS, search alone > 50%)

---

## Core Hypothesis
A neural net + PIMC solver can exceed 50% win rate on Klondike draw-1 unlimited passes under imperfect information, making the game +EV at 2x payout.

**Calibrated confidence: ~30-45%.** Plausible but far from certain. All architecture choices below are leading candidates pending empirical validation, not commitments.

## What We Know (Verified)

### Solvability
- Draw-3 Thoughtful solvability: 81.945% ± 0.084% (Solvitaire, 1M deals — gold standard)
- Draw-3 Thoughtful confirmed: 81.95% ± 0.03% (Lonelybot, 5.9M games)
- Draw-1 Thoughtful solvability: ~82-91% (lower bound from draw-3, upper bound is community estimate — not rigorously established)
- ~9-18% of deals are mathematically impossible regardless of method

### Imperfect Information Win Rates
- Lonelybot HOP/MCTS on draw-3: 47.58% ± 0.80% (15K games) — verified, reproducible
- Lonelybot MCTS solves ~58% of solvable draw-3 deals (476/820)
- Turnbull heuristic on draw-1: 52.9-53.4% (10M games) — **UNVERIFIED, closed source, treat as hypothesis**
- Best documented human on draw-1: ~43% (Jupiter Scientific, 442 games)
- Average human on draw-1: ~33% (Solitaired.com, 2.9M games)

### EV Math
- At 2x payout: EV = (win_rate × 2) - 1
- Breakeven: 50% win rate
- At 33%: lose $0.34 per dollar (34% house edge)
- At 43%: lose $0.14 per dollar (14% house edge)
- At 53%: profit $0.06 per dollar (6% player edge)
- Even 51% = 2% edge, better than most card counting scenarios

## What We Don't Know (Key Unknowns)

### ~~Critical Unknown #1~~ RESOLVED: Lonelybot MCTS on Draw-1 (with Oracle Leak Correction)
- **With oracle leak (topmost card preserved): 68.9%, CI (66.5%, 71.2%), n=1500**
- **True imperfect info (full shuffle): 49.0%, CI (45.9%, 52.1%), n=1000**
- Oracle leak in `hidden_shuffle()` preserves topmost hidden card per pile — worth ~19pp
- Implemented `shuffle_all()` to remove all oracle information
- Draw-1 full-shuffle baseline: 49.0% — just below breakeven, 50% inside CI
- Draw-3 full-shuffle: 31.0%, CI (25.0%, 37.7%), n=200 (vs 49.4% with oracle)
- **Raw HOP solver is borderline under true imperfect info — neural guidance is needed to cross 50%**

### RESOLVED: HOP Solver Regression at HEAD
- Published draw-3 HOP: 47.58% (commit 3e88e66, v0.2.2 era, 2024-05-14)
- Measured draw-3 HOP at HEAD: **~4% on 50 games** — confirmed regression
- **Reproduced at old commit**: 247/500 = 49.4%, CI (45.0%, 53.8%) — matches published 47.58%
- Resolution: use old commit (3e88e66) patched with draw_step support and MAX_MOVES
- Saved as `lonecli_v022_patched` binary — this is the working solver
- HEAD regression is NOT blocking; we don't need HEAD for anything

### Critical Unknown #2: Can a Neural Net Improve on Raw MCTS?
- AlphaZe** shows neural net guidance improves PIMC search in Stratego
- But NN guidance often provides compute-efficiency gains more than guaranteed asymptotic win-rate improvement
- Klondike's state space is much smaller than Stratego — diminishing returns possible
- The net might learn patterns MCTS misses (e.g., long-range card sequencing)
- Or MCTS might already be near-optimal for Klondike's complexity
- **Status: no one has tried**

### Critical Unknown #3: Is 1xbet's Shuffle Fair?
- Curacao license = no mandated independent RNG testing
- Proprietary 1xGames, not third-party audited
- Could be verified statistically with enough logged games
- **Status: untested, but mathematically irrelevant until unknowns #1 and #2 are resolved**

### Critical Unknown #4: Rule Fidelity
- Any mismatch between Lonelybot's rules and 1xbet's actual implementation invalidates results
- Must verify empirically: unlimited passes really unlimited? Hidden timers? Cards movable off foundations?
- **Status: unverified**

## Assumptions Being Tested
1. Draw-1 MCTS win rate is meaningfully higher than draw-3 (>50%)
2. Neural net + PIMC can match or exceed raw MCTS performance
3. The improvement from neural guidance justifies the training investment
4. A solver-assisted human can capture the theoretical edge in real-time play
5. Solver win rate translates to deployable casino EV (latency, input errors, human execution drift)
6. Lonelybot's rules match 1xbet's actual implementation

## Architecture — Leading Candidate (from alphazero-klondike-research.md)

### AlphaZe** PC-PIMC
- Standard AlphaZero can't handle imperfect info (assumes full observability)
- AlphaZe** replaces MCTS with PC-PIMC: sample n determinized worlds → run MCTS on each → average policies
- Neural net sees ONLY visible game state, NOT the determinized hidden cards — forces generalization
- **MuZero deferred**: too complex for phase 1, revisit if AlphaZe** plateaus

### Why PIMC works well for Klondike (but isn't perfect)
- Strategy fusion concern is **mitigated** by non-adversarial setting — not fully eliminated
- Klondike is non-adversarial — hidden cards are random, not strategically placed by opponent
- High disambiguation — cards are progressively revealed during play
- **Optimistic bias from HOP is real and documented** — PIMC overestimates win probability because determinization makes future card flips correlated
- Known mitigations: ISMCTS, RecPIMC, EPIMC — should be tested at Gate 1 before committing to neural approach
- Klondike may be a strong testbed for AlphaZe** methods, but "ideal" is an unproven claim

### Determinization Counts
- Training: **3-5 determinizations** per move (speed priority)
- Evaluation/play: **50-100 determinizations** per move (precision priority)
- Bridge AI (GIB) uses 50, Skat saturates at 100-200, AlphaZe** works with just 3-12 + neural net

### TrueSight Learning (Curriculum)
- Phase 1: Train on Thoughtful Solitaire (all cards visible) — net learns basic Klondike strategy
- Phase 2: Switch to PIMC with hidden information — net learns to handle uncertainty
- Mirrors how humans learn card games: cards face-up first

### State Encoding — 757 dims (baseline), ~857 with Phase 6 expansions
- Binary card-centric encoding (676 values): 52 cards × 13 features (11 location one-hot + visible + position)
  - stock and unknown are SEPARATE one-hot slots (card in stock ≠ card face-down under tableau)
- Uncertainty summary (67 values): unseen card set mask, per-pile belief hints (red/black fraction, rank distribution)
- Global features (14 values): foundation ranks, hidden counts, deck info
- 3 KB per state, batch 4096 = ~12 MB — trivial for 8GB VRAM
- **Phase 6 expansion to ~857**: stock-pass context, action history (last 5 moves), move-cycle features, deck position

### Neural Net Architecture — Leading Candidate
- MLP with 4 residual blocks × 512 hidden units (NOT CNN — card games lack spatial locality)
- ~2M parameters, ~8 MB FP32
- Dual heads: policy (move probabilities) + value (win probability)
- Auxiliary targets: foundation count, face-down cards remaining (KataGo insight — sharper gradients)
- No batch normalization — Fixup initialization (KataGo: 1.6-1.8x speedup)
- **Alternatives to evaluate if MLP plateaus**: transformer/set-network (relational card-card interactions), LSTM history encoder (implicit belief tracking)

### Integration: Lonelybot via PyO3 — COMPLETE (Phase 1)
- PyO3 bindings built and tested — `klondike-env/` crate with maturin
- **17,935 games/sec** raw Rust through Python (well above 1000 target)
- **32,146 encoding calls/sec** for 757-dim observation vectors
- Full undo support via internal stack (Lonelybot's do_move returns UndoInfo, not bool)
- Clone + shuffle_hidden for PIMC determinization working
- HOP solver accessible from Python (per-move evaluation)
- **Engineering risk resolved**: PyO3 integration is fast and reliable

### Timeline Estimates (Revised Post-Critique)
- **First credible result: 6-12 weeks**
- **Robust reproducible claim with ablations: 3-6 months**
- Most likely blowups: Rust/Python integration, batched in-tree inference, reproducibility under multithreading, hyperparameter churn, thin-margin evaluation statistics

### Evaluation Methodology
- **Paired seeds**: run different solvers on identical deals for direct comparison
- **SPRT / sequential testing**: for thin margins around 50%
- **Multiple-comparison control**: avoid false positives from repeated tuning on fixed test sets
- **Evaluation leakage**: don't tune hyperparameters on the same test set used for final claims

### Key Optimization Priorities (ordered by impact)
1. PyO3 bindings with GIL release
2. FP16 mixed precision (2x speedup, 50% VRAM savings)
3. Batched NN inference during MCTS (batch 16-64 leaf evaluations)
4. Async pipeline: overlap CPU self-play with GPU training
5. torch.compile(mode='max-autotune') for 20-40% inference speedup
6. KataGo playout cap randomization (more games per compute unit)

### Framework Decision
- **Custom PyTorch** based on alpha-zero-general (suragnair, 4.4k stars)
- Enhanced fork: kevaday/alphazero-general (Cython MCTS — 30x faster than pure Python)
- Reference LightZero (OpenDILab, NeurIPS 2023) for C++ MCTS patterns
- Stable Baselines3 doesn't support MCTS; RLlib's AlphaZero contrib is buggy

### Useful Open-Source References
- **alpha-zero-general** (suragnair): clean Game/NeuralNet abstractions
- **kevaday/alphazero-general**: Cython MCTS, 30x faster
- **LightZero** (OpenDILab): C++ MCTS, unified AlphaZero/MuZero
- **CrazyAra** / AlphaZe code: github.com/BluemlJ/AlphaZe
- **BetaZero** (Moss et al, Stanford 2024): AlphaZero for POMDPs with particle belief — Julia-only
- **tmoer/alphazero_singleplayer**: single-player adaptations (Q-value initialization)

## MuZero — Deferred (Not Ruled Out)
- Could theoretically squeeze extra 1-2% via learned latent world model
- Captures uncertainty about hidden cards without explicit determinization
- Stochastic MuZero achieved SOTA on 2048 (closest analog to Klondike)
- BUT: needs 3 learned functions (vs 1), 5-10x more training games, months instead of weeks
- Must learn card game rules from scratch — wasted compute when we have a perfect simulator
- **Decision: start with AlphaZe**, only try MuZero if AlphaZe** plateaus below target**
- Middle ground: BetaZero (Stanford 2024) — AlphaZero for POMDPs with particle belief, Julia-only

## Potential Failure Modes
- Lonelybot MCTS comes back low on draw-1 → search-only solver may not cross 50% → but neural guidance could still help
- Optimistic bias in HOP — PIMC overestimates win probability (known, must be addressed, not "resolved")
- Strategy fusion is mitigated but NOT eliminated by non-adversarial setting — information-set inconsistency still possible
- Neural net overfits to training distribution, doesn't generalize to unusual deals
- ~~PyO3 integration latency in MCTS inner loop~~ **RESOLVED**: 17,935 games/sec, not a bottleneck
- The edge is so thin (1-3%) that variance makes it impractical even if real
- Evaluation leakage: repeated tuning on fixed test sets with small deltas around 50% creates false positives
- Rule fidelity: Lonelybot rules may not match 1xbet exactly
- Solver win rate ≠ deployable casino EV (latency, input errors, human execution drift, ban risk, withdrawal risk)
- 1xbet detects solver-assisted play and bans account
- 1xbet's RNG is not fair, nullifying theoretical edge

## Research Questions Informed by Research (Leading Answers, Not Final)
- State representation → 757-dim baseline encoding (stock/unknown separated; ~857 with Phase 6 expansions)
- Belief state encoding → Uncertainty summary or LSTM history encoder
- PIMC determinizations → 3-5 training, 50-100 eval
- Game engine → Rust (Lonelybot) via PyO3 bindings
- Architecture → MLP-ResNet (but transformer/set-network is realistic alternative)
- MuZero vs AlphaZe** → AlphaZe** first, MuZero if plateau

## Spec Review — All Defects Fixed (Session 3)

External agent review identified 12 findings (P0-P2). All verified against source code and fixed:

**Critical fixes (P0):**
- MCTS root visit count: `root.visit_count` never incremented (backprop only updates children). UCB exploration at root was always 0. Fixed in Phase 2 spec.
- Position-level train/val split leaked same-game data. Replaced with game-level splitting using `game_offsets.npy`. Fixed in Phase 3 spec.
- Append+concatenate data generation caused O(N²) RAM blowup. Replaced with memmap-based writing. Fixed in Phase 3 spec.
- Runtime estimate was 5s/game (gut feel); bottom-up math gives 20-60s/game. Total training time revised from 30-40h to 170-200h. Fixed in Phase 4 spec.

**Important fixes (P1):**
- CLI commands: `hoploop` → `hop-loop` (Clap kebab-case). Fixed in Phase 0 spec.
- Eval seed range overlapped Phase 6 test_set_A. Changed to 5M+ (disjoint). Fixed in Phase 4 spec.
- PIMC seeding used wall-clock `time.time()` — breaks reproducibility. Changed to deterministic. Fixed in Phase 2 spec.
- Gate criterion: 1000 games needs ~53% point estimate for CI lower bound >50%. Added math explanation. Fixed in Phase 4 spec.
- Encoding dimension: was 705/857 inconsistently. Settled on 757 baseline. Fixed across all specs.

**Minor fixes (P2):**
- max_moves reduced from 2000→500 across all phases (with cycle detection added)
- Hyperparameter sweep: "grid search" → "one-at-a-time (~20 configs)" in Phase 6
- Missing imports added across Phase 2-4 specs
- Added cycle detection (state hash set) to self-play and evaluation in Phase 4

### Final Cleanup (Session 3 continued)
6 additional findings verified and fixed:
- SearchFn signature updated to `(env, game_seed, move_num)` — all call sites in Phase 2 benchmark/ablations and Phase 3 data gen now pass seeds correctly
- Uncertainty features (15 dims) implemented with actual computation — no longer TODO zeros
- Final-save RAM spike eliminated — streaming copy to new memmap instead of `np.array(memmap_slice)`
- Aux loss deferred to Phase 6 — Phase 3 trains policy+value only (aux heads exist but are dormant)
- Phase 3 runtime estimate made conditional on HOP approach (fast path vs per-move)
- MCTS root value computed from children instead of always-zero `root.value`
- Phase 5 acceptance criteria split into core (required) vs deferred (CV scaffolds)

### Evaluation Seed Protocol (Three Disjoint Sets)
- **Phase 4 model selection**: seeds 5,000,000+ (1000 games per eval)
- **Phase 6 test_set_C (tuning)**: seeds 3,000,000 - 3,004,999 (5K games)
- **Phase 6 test_set_A (final claims)**: seeds 1,000,000 - 1,009,999 (10K games)
- **Phase 6 test_set_B (confirmation)**: seeds 2,000,000 - 2,009,999 (10K games)
- **Rule**: NEVER tune on test_set_A/B. Final claims on test_set_A only.

## Implementation Specs — Complete

Full spec plan in `specs/` directory (7 files). Each phase is self-contained for a fresh Claude Code session.

| Phase | Gate | What Gets Built | Status |
|-------|------|-----------------|--------|
| 0 | Gate 0 | Build Lonelybot, reproduce baselines, get draw-1 number | **COMPLETE** ✅ |
| 1 | Gate 1 prep | PyO3 Rust crate + Python game interface + encoding + actions | **COMPLETE** ✅ |
| 2 | Gate 1 | PIMC search + Python MCTS + statistical evaluation framework | **COMPLETE** ✅ — GATE 1 PASS |
| 3 | Gate 2 | Training data + MLP-ResNet (~2.7M params) + supervised training | |
| 4 | Gate 3 | PC-PIMC self-play + replay buffer + AlphaZero training pipeline | |
| 5 | Gate 4 | CV pipeline + move advisor (optional, only if >50%) | |
| 6 | Post-3 | Hyperparameter tuning, architecture experiments, scaling | |

**All foundation code is built by end of Phase 4.** Phase 6 is experiments on that foundation.

### Resolved Questions (from spec work)
- Action space: 5 move types × 52 cards = 260, with legal masking (N_MOVES_MAX = 21 per position)
- Q-value init: V(s) not 0 — critical for single-player MCTS
- Replay buffer: 100K-500K positions (200K default)
- Temperature: τ=1.0 first 15-20 moves, then τ→0
- Observation encoding: 757 dims baseline (expandable to ~857 with stock-pass, history, cycle features in Phase 6)
- Lonelybot has built-in PIMC support via `shuffle_all(rng)` (safe) and `shuffle(rng)` (oracle leak)
- **Actual API**: everything on `Solitaire` directly (NO SolitaireEngine), do_move returns UndoInfo(u8)
- **PyO3 bindings built**: `klondike-env/` crate, installed via `maturin develop --release`

### Still Open (for Phase 6 experiments)
- ISMCTS vs HOP: which gives less optimistic bias for Klondike?
- Encoding additions: do stock-pass/history features meaningfully improve performance?
- Auxiliary targets: how much do foundation count / face-down predictions help?
- Architecture: MLP-ResNet vs transformer/set-network

## Session 4-5 Findings (Phase 0 — COMPLETE)

### Gate 0 Results — ALL PASSED ✅

| Step | Result | Status |
|------|--------|--------|
| Draw-3 Thoughtful | 82.13% (8963 games) — matches published 81.95% | ✅ |
| Draw-3 HOP (old commit, oracle) | 49.4% (500 games) — matches published 47.58% | ✅ |
| Draw-1 HOP (oracle) | 68.9% (1500 games), CI (66.5%, 71.2%) | ✅ |
| **Draw-1 HOP (full shuffle)** | **49.0% (1000 games), CI (45.9%, 52.1%)** | Borderline |
| Draw-3 HOP (full shuffle) | 31.0% (200 games), CI (25.0%, 37.7%) | ✅ |

### Key Numbers (Updated with Oracle Leak Correction)
- **Draw-1 HOP with oracle: 68.9%** — inflated by topmost-card oracle leak
- **Draw-1 HOP full-shuffle (true imperfect info): 49.0%, CI (45.9%, 52.1%), n=1000**
- **Draw-3 HOP full-shuffle: 31.0%, CI (25.0%, 37.7%), n=200**
- Oracle leak worth ~19pp on both draw modes
- Draw-1 full-shuffle is borderline — 50% inside CI but point estimate below breakeven

### Tools and Binaries
- `lonecli_v022` — unpatched old binary (draw-3 only, commit 3e88e66)
- `lonecli_v022_patched` — patched binary (draw-1 support, MAX_MOVES=5000, eprintln) — has oracle leak
- `lonecli_fullshuffle` — heavy params (3000/1000), full shuffle, decode fix, no oracle leak
- `lonecli_fast` — fast params (500/500), full shuffle, decode fix, no oracle leak (11× faster, -0.8pp)
- `run_benchmark.py` — parallel benchmark script (currently points to `lonecli_fast`)
- `run_benchmark_old.py` — benchmark script for old binary format

### Fast vs Heavy Params
| Config | N_TIMES | LIMIT | Win Rate | Speed |
|--------|---------|-------|----------|-------|
| Heavy | 3000 | 1000 | ~49.0% | ~5.3s/game |
| Fast | 500 | 500 | ~48.0% | ~0.5s/game |

Use fast for large-N benchmarks (1M games), heavy for precise ceiling estimates.

### Decode Bug Fix
`hidden.rs:decode()` was missing `update_map()` and `compute_first_layer_mask()` calls. Fixed. No win rate impact on current workload but prevents latent state corruption in MCTS encode/decode cycles.

### Implications for Project Direction (Revised after Oracle Leak Discovery)
- Raw HOP solver is borderline (49.0%) under true imperfect info — neural guidance IS needed to cross 50%
- The project is genuinely a research challenge again, not a solved problem
- Value of neural guidance: crossing the 50% threshold, not just optimization
- Gate 1 search-only ablations (more determinizations, ISMCTS) are critical path
- The ~49% baseline is encouraging — only ~1-2pp improvement needed from neural nets
- Oracle leak explains why published Lonelybot results seemed surprisingly high for imperfect-info play

## Phase 1 Results — PyO3 Bridge COMPLETE ✅

### Deliverables
- **`klondike-env/`** — Rust PyO3 crate (pyo3 0.28, lonelybot path dep, rand 0.8.5)
  - KlondikeGame pyclass: reset, get_legal_moves, do_move, undo_move, clone_game, shuffle_hidden, hop_solve, encode
  - Internal undo stack (Lonelybot's do_move returns UndoInfo, Python gets simple bool API)
  - Build: `. .venv/bin/activate && cd klondike-env && maturin develop --release`
- **`klondike/`** — Python package
  - `env.py`: Gym-like KlondikeEnv (reset/step/get_legal_actions/clone/shuffle_hidden/is_done)
  - `encoding.py`: 757-dim state encoding (52×13 card + 14 global + 67 uncertainty)
  - `actions.py`: 260-action space (5 types × 52 cards), bidirectional conversion

### Performance
| Metric | Result | Target |
|--------|--------|--------|
| Raw games/sec | 17,935 | >1,000 ✅ |
| Encoding calls/sec | 32,146 | — |
| Full env (encode every step) | 199 games/sec | — |

### Spec Corrections Discovered
- NO SolitaireEngine — all methods on `Solitaire` directly
- `do_move` returns `UndoInfo(u8)`, not bool; `undo_move` needs `(&Move, &UndoInfo)`
- `list_moves` takes `&[u64; 5]` prune mask (use `PruneInfo::default().prune_moves::<true>(&game)`)
- N_MOVES_MAX = 21 (not 24), rand 0.8.5 (not 0.9), PyO3 0.28 (not 0.22)
- Hidden.len(pos) includes topmost hidden card (the face-up card); get_hidden_counts applies n-1

### Environment
- `.venv/` — Python 3.14, maturin 1.12.6, numpy, klondike-env 0.1.0
- Lonelybot submodule at commit 3e88e66 (v0.2.1-15)

## Phase 2 Results — Search & Eval Framework COMPLETE, Gate 1 PASS ✅

### Critical Discovery: hop_solve vs pick_moves
- Python `hop_solve()` = per-move DFS evaluation → **~18% win rate** (WRONG algorithm)
- Binary's `pick_moves()` = UCB1 bandit over decision states with traverse → **~55-60%** (CORRECT)
- Fix: exposed `pick_moves` via `play_hop_turn()` in PyO3

### Oracle Leak Fix in Inner Solver
- Changed `hop_solver.rs:76` from `shuffle()` to `shuffle_all()`
- Inner solver now fully oracle-free (matching outer loop)
- **Improved win rate** from Phase 0's ~49% to ~55-60%

### Gate 1 Ablation Results

| Config | Win Rate | CI (95%) | s/game |
|--------|----------|----------|--------|
| nt=100, lim=500 | 48.0% | (41.2%, 54.9%) | 0.10 |
| nt=500, lim=500 | 55.0% | (48.1%, 61.7%) | 0.42 |
| nt=1000, lim=500 | 56.5% | (49.6%, 63.2%) | 0.77 |
| nt=3000, lim=500 | 56.5% | (49.6%, 63.2%) | 2.09 |
| **nt=500, lim=100** | **59.5%** | **(52.6%, 66.1%)** | **0.17** |
| nt=500, lim=1000 | 59.5% | (52.6%, 66.1%) | 0.66 |
| heavy (3000/1000) | 58.0% | (51.1%, 64.6%) | 3.50 |

**Best: nt=500, lim=100 → 59.5%, CI lower 52.6% > 50%**
- Search alone exceeds breakeven
- Neural net is optimization, not necessity
- Best efficiency at 0.17s/game — fast enough for real-time play
- Diminishing returns from n_times > 500 and node_limit > 100

### 5000-Game Definitive Benchmark (nt=500, lim=100)
- **2748/5000 = 54.96%, Wilson 95% CI (53.58%, 56.33%)**
- CI lower bound **53.6%** — 3.6pp above breakeven, statistically significant
- Longest losing streak: 11 games (seeds 4322-4332), longest win streak: 18
- EV at 2x payout: +$0.10 per dollar wagered (10% player edge)
- ~0.23s/game average across 5000 games

### Gate 1 Decision: **PASS — Search alone sufficient for >50% win rate**

### Deliverables
- `klondike/search/pimc.py` — HopConfig, FullHopConfig, PIMCConfig, hop_search, pimc_search, play_full_hop_game
- `klondike/search/mcts.py` — MCTSConfig, MCTSNode, mcts_search with pluggable eval_fn
- `klondike/eval/stats.py` — Wilson CI, SPRT, McNemar's, control variate
- `klondike/eval/benchmark.py` — BenchmarkConfig, run_benchmark with cycle detection
- `klondike-env/src/lib.rs` — play_hop_turn (pick_moves via PyO3)

## Additional Research to Investigate
- POMCP (Silver & Veness, 2010) and POMCPOW variants
- SP-MCTS literature (single-player puzzle adaptations)
- SPRT / sequential testing for thin-margin evaluation
- Belief-state modeling baselines (particle-belief search) as non-neural control arm
