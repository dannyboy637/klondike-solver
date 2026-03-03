# Session Log

---

## Session 1 — 2026-03-03: Research Compilation & Project Scoping

### What happened
- Compiled three source files (klondike-analysis.md, discussion-notes.md, klondike research.rtf) into a single master document (`klondike-master.md`)
- Deep discussion on solver approaches: heuristic vs MCTS vs hybrid
  - Heuristic: fixed priority rules, fast (~1ms/game), limited by rule quality. Collins gets ~7.6% draw-3, Turnbull claims ~53% draw-1 (closed source)
  - MCTS/HOP (Lonelybot): simulates thousands of futures, handles hidden cards via determinization (sampling possible arrangements). ~2-4 sec/game. 47.58% on draw-3.
  - Key insight: Turnbull's 53% heuristic vs Lonelybot's ~50-55% MCTS are surprisingly close — good heuristics get you most of the way
- Discussed training a heuristic from MCTS data (AlphaZero distillation approach) — use MCTS as teacher, train fast policy net on its decisions
- Hardware feasibility analysis:
  - M4 MacBook Pro: good for dev and quick runs
  - Ryzen 5700X + 32GB + RTX 4060: 16 threads for parallel MCTS, GPU for neural net training
  - Lonelybot is CPU-only, single-threaded per game, trivially parallelizable
  - 50K MCTS games on Ryzen: ~2-4 hours. 1M Thoughtful games: ~2-3 min. Compute is NOT the bottleneck.
- Clarified that Lonelybot's 47.58% (and Turnbull's 53%) are measured against ALL deals, including the ~18% that are mathematically impossible. Against solvable-only deals, MCTS solves ~58%.
- Researched academic landscape (2024-2026):
  - Constraint Models for Klondike (CP 2025) — Dang et al, St Andrews. Still classical search.
  - Blake & Gent Solvitaire updated (arXiv v5, Aug 2024, JAIR Feb 2026). Invited talk at CP 2024 titled "Solving Patience with Good Old Fashioned AI."
  - NO modern ML approaches exist for Klondike. No AlphaZero, no deep RL, no neural net + MCTS, no MuZero, no transformers. The field is wide open.
- Discovered AlphaZe** paper (Frontiers in AI, 2023): AlphaZero adapted for imperfect info games using PIMC. Tested on Stratego. Directly applicable to Klondike.
- Devil's advocate analysis — why this might not work:
  - Turnbull's 53% is unverified/closed source
  - Real MCTS draw-1 number doesn't exist yet — could be <50%
  - Even at 53%, edge is razor-thin ($0.06/dollar), variance is brutal
  - 1xbet platform risk: account freezing, unverifiable RNG, Curacao license
  - The casino is pricing for 33% players, not solver-assisted play
- Generated a research prompt for Claude web to deep-dive: PIMC implementation details, AlphaZe** architecture, MuZero for imperfect info, state representation, self-play for single-player games, practical framework selection

### Key decisions
- Pursuing AlphaZe** (PIMC + neural net) approach — first application to Klondike
- First priority: run Lonelybot MCTS on draw-1 to get the critical number
- If >50%: proceed with neural net pipeline. If <50%: project becomes academic/educational only.

### Open threads for next session
- Clone and build Lonelybot
- Run draw-1 MCTS (the first domino)
- Review Claude web research results on PIMC / AlphaZe** / state representation
- Decide: wrap Lonelybot vs build custom Python engine
- Decide: neural net architecture and training framework

### Dead ends / things ruled out
- GPU acceleration for Lonelybot — not applicable, pure CPU
- Full state space lookup tables — too large (billions of states)
- Exhaustive Solvitaire-style proving — not needed, sampling is sufficient for our purposes

---

## Session 1 (continued) — 2026-03-03: Deep Research Review

### What happened
- Reviewed "How to Simulate Like a Quant Desk" (gemchanger, Feb 2026) for applicable techniques
  - Extracted: MC confidence intervals, Brier score, variance reduction (antithetic + stratified + control variates), particle filter for belief state
  - Variance reduction can cut MCTS compute budget by 3-5x — 10K games with tricks ≈ 50K crude
  - Particle filter maps to Klondike belief state tracking (updating hidden card probabilities as cards are revealed)
  - Saved as `quant-techniques-reference.md`

- Reviewed deep research document from Claude web (`alphazero-klondike-research.md`) — this is the implementation blueprint
  - **Major findings that changed our understanding:**

  1. **PIMC's strategy fusion failure mode doesn't apply to Klondike** — it's a non-adversarial game. Hidden cards are random, not strategically placed. This eliminates the biggest theoretical risk we identified.

  2. **Concrete determinization counts**: 3-5 during training, 50-100 during eval. AlphaZe** works with just 3 when neural net guided. Way fewer than expected.

  3. **TrueSight Learning curriculum**: Train Phase 1 on Thoughtful Solitaire (perfect info, Lonelybot wins ~82%), Phase 2 switch to PIMC with hidden cards. Get a strong baseline "for free."

  4. **State encoding decided**: Proposal D, ~857 dimensions. Binary card locations + uncertainty summary. 3.4 KB per state. Trivial VRAM footprint.

  5. **MLP not CNN**: Card games lack spatial locality. MLP with 4 residual blocks × 512 hidden, ~2M params, ~8 MB.

  6. **PyO3 is the integration path**: ~60-70ns call overhead vs 1-10ms subprocess. Lonelybot has a `python/` dir suggesting some groundwork exists.

  7. **MuZero definitively ruled out**: Too complex, needs 3 learned functions, LightZero benchmarks say use AlphaZero when simulator available, original needed 16 TPUv3 + 1000 TPUs.

  8. **Training timeline**: 2-4 weeks training, 5-8 weeks total with engineering.

  9. **Framework**: Custom PyTorch based on alpha-zero-general. Stable Baselines3 doesn't support MCTS. RLlib buggy.

  10. **KataGo optimizations**: Playout cap randomization, auxiliary targets (foundation count, face-down count), Fixup init (drop batch norm), stochastic weight averaging.

### Key decisions made
- Architecture: AlphaZe** PC-PIMC — fully decided, no more deliberation needed
- State encoding: Proposal D (~857 dims)
- Network: MLP-ResNet, ~2M params
- Integration: Lonelybot via PyO3
- Framework: Custom PyTorch + alpha-zero-general
- MuZero: ruled out
- Strategy fusion concern: eliminated (non-adversarial game)

### Open threads for next session
- Clone and build Lonelybot
- Run draw-1 MCTS (still the first domino — everything else waits on this number)
- Start exploring PyO3 bindings to Lonelybot
- Review alpha-zero-general and kevaday/alphazero-general repos for codebase structure

### Risks identified
- **Biggest engineering risk**: Rust-Python integration latency in MCTS inner loop. Must batch NN queries and release GIL during Rust computation.
- Training timeline of "2-4 weeks" is optimistic — expect 5-8 weeks with debugging and iteration
- Turnbull's 53% still unverified — our draw-1 MCTS number could come back <50%

### Final discussion: MuZero reconsideration
- User asked if MuZero might yield higher win rate than AlphaZe**
- Answer: possibly 1-2% higher in theory (learned world model captures hidden card uncertainty better than explicit determinization)
- But 5-10x more training, months instead of weeks, must learn card game rules from scratch
- **Decision: deferred, not ruled out.** Start AlphaZe**, only try MuZero if it plateaus below target.
- BetaZero (Stanford 2024) noted as middle ground — AlphaZero for POMDPs with particle belief, Julia-only

### External critique review
Received deep critical review from another agent. Key valid points incorporated:

**Accepted (docs updated):**
1. "Architecture selected, evidence later" — we had premature lock-in. Changed all "DECIDED" to "leading candidate." Architecture choices are hypotheses until validated by experiment.
2. Strategy fusion marked as "mitigated, not eliminated" instead of ~~resolved~~. Optimistic bias is a real concern needing active mitigation.
3. Added gated approach: Gate 0 (reproduce known results) → Gate 1 (search-only ablations before any NN) → Gate 2 (cheap supervised distillation) → Gate 3 (full self-play). No skipping gates.
4. "If Lonelybot draw-1 <50% then -EV for everyone" was too strong. Corrected: it means that specific solver is below breakeven, not that 50% is unreachable.
5. Added predefined stop criteria at each gate.
6. Timeline revised: first credible result 6-12 weeks, robust claim 3-6 months (was 2-4 weeks).
7. Added calibrated confidence: ~30-45% chance plan yields >50%.
8. Added missing risks: rule fidelity, evaluation leakage, deployable EV ≠ solver win rate, source quality.
9. Added missing research: POMCP, SP-MCTS, SPRT/sequential testing, belief-state baselines.
10. State encoding gaps flagged: stock-pass context, action history, move-cycle features.
11. MLP alternative noted: transformer/set-network for relational card interactions.
12. Added evaluation methodology: paired seeds, SPRT, multiple-comparison control.

**Pushed back on:**
1. "Source quality risk" — core numbers (Solvitaire, Lonelybot draw-3) are from published papers with reproducible code. Weak sources (Turnbull, draw-1 extrapolation) were already flagged.
2. "Casino deployment as core workstream" — disagree on ordering. Must answer "can ANY solver beat 50%?" before deployment matters. CV/latency/detection is genuinely downstream.
3. POMCP/DESPOT suggestions — relevant but designed for general POMDPs. Klondike's specific structure makes PIMC a more natural fit. Worth reading, unlikely to outperform domain-specific approaches.

### Status at end of session
- Architecture choices softened to "leading candidates" pending validation
- Gated approach with stop criteria added
- Falsification discipline added to Claude guidelines
- Ready to start implementation in next session
- **First action next session: Gate 0 — clone Lonelybot, reproduce draw-3 results, THEN run draw-1 MCTS**

---

## Session 2 — 2026-03-03: Comprehensive Spec Plan

### What happened
- Cloned Lonelybot to `solitaire/lonelybot/`
- Before coding, user requested comprehensive spec plan broken into phases (150k tokens per phase, designed for Claude Code sessions with team agents)
- Launched two research agents in parallel:
  1. **Lonelybot codebase exploration** — full directory structure, all public Rust APIs (Card, Move, Solitaire, SolitaireEngine, Deck, Stack, Hidden, solvers, shufflers), CLI interface, existing Python dir (graph.py only, no PyO3)
  2. **AlphaZero research doc extraction** — all implementation-specific numbers, dimensions, formulas, code patterns
- Created 6 spec files in `specs/` directory:

| File | Description |
|------|-------------|
| `00-overview.md` | Master plan — phase summary, dependency graph, decision gates, directory structure, storage budget |
| `phase-0-setup-baselines.md` | Clone/build Lonelybot, reproduce draw-3, get draw-1 MCTS baseline |
| `phase-1-pyo3-bridge.md` | Rust PyO3 crate + Python game interface + state encoding + action space |
| `phase-2-search-evaluation.md` | PIMC search + Python MCTS + statistical evaluation (CI, SPRT, variance reduction) |
| `phase-3-supervised-learning.md` | Training data generation + MLP-ResNet architecture + supervised training loop |
| `phase-4-alphazero-selfplay.md` | PC-PIMC self-play + replay buffer + training pipeline + TrueSight curriculum |
| `phase-5-deployment.md` | CV pipeline + move advisor + rule fidelity + bankroll management |

### Key findings from Lonelybot exploration
- **No existing PyO3 bindings** — `python/graph.py` is a CNF analysis tool only
- **SolitaireEngine<FullPruner>** is the right wrapper — has public `do_move()`, `list_moves_dom()`, `undo_move()`
- **`hidden_shuffle(&mut self, rng)`** is built-in PIMC support — randomly rearranges hidden cards
- Card encoding: `Card(u8)` with `rank * 4 + suit` convention (to verify)
- `N_MOVES_MAX = 24` — max legal moves per position
- Action space: 5 move types × 52 cards = 260 (with legal masking)
- No STD support — library can run in WASM
- Observation encoding landed at ~705 dims baseline (expandable to ~857 with extras)

### Key spec design decisions
- Each phase spec is **self-contained** — includes all context, interfaces, types inline
- Phases marked for **team agent parallelism** where possible (Phase 1: Rust+Python, Phase 2: Search+Stats)
- **Shared interfaces** defined at top of each parallel phase
- Storage estimate: ~10-15 GB on PC, MacBook for specs only
- All phases have explicit **acceptance criteria** and **decision gates**

### Storage estimate
- Rust toolchain: ~1.5 GB
- Lonelybot + build: ~500 MB
- Python + PyTorch + CUDA: ~5-6 GB
- Training data: ~2-5 GB
- Checkpoints + buffer: ~1-2 GB
- Total: ~10-15 GB (well within 200GB+ on PC)

- Added Phase 6 spec (`phase-6-iteration-optimization.md`) — covers post-pipeline iteration:
  - Tier 1: hyperparameter sweeps, eval with more determinizations, extended training
  - Tier 2: encoding improvements, architecture variants, ISMCTS vs HOP, auxiliary target ablation
  - Tier 3: batched NN inference in MCTS, multiprocess self-play, SWA, playout cap randomization
  - Tier 4: pivots if plateaued (MuZero, direct RL, ensembles, opening book)
  - Evaluation discipline: fixed test sets, tracking spreadsheet, stop criteria
- Updated `00-overview.md` with Phase 6 in dependency graph and file table
- Clarified for user: Phases 0-4 build ALL foundation code; Phase 6 is experiments using that code, not new infrastructure

### Key decisions
- 7 total phases (0-6), with Phase 5 optional and Phase 6 ongoing
- All coding/infrastructure is complete by end of Phase 4
- Phase 6 is structured experimentation, not new code — hyperparams, architectures, scaling

- Created `specs/prompts.md` — copy-paste prompts for each phase (solo + team agent variants)
- Created `specs/machine-transfer-plan.md` — full git sync workflow between MacBook and PC
- Decided MacBook can handle Phases 0-3 (M4 is fine for 2.7M param model on MPS), PC needed for Phase 4+
- User has ~30GB free on MacBook — enough for Phases 0-3 (~6-7 GB)
- Set up GitHub:
  - Initialized git repo in `solitaire/`
  - Lonelybot added as git submodule (not committed source)
  - `.gitignore` covers build artifacts, training data, checkpoints, venvs
  - Private repo created: `https://github.com/dannyboy637/klondike-solver`
  - Initial commit pushed to `main`

### Key decisions
- 7 total phases (0-6), with Phase 5 optional and Phase 6 ongoing
- All coding/infrastructure is complete by end of Phase 4
- Phase 6 is structured experimentation, not new code — hyperparams, architectures, scaling
- MacBook for Phases 0-3 (traveling), PC for Phase 4+ (heavy compute)
- Git + GitHub private repo for machine sync, large files (data/checkpoints) stay local

### Open threads for next session
- **Execute Phase 0**: Build Lonelybot, reproduce baselines, get draw-1 number
- Copy prompt from `specs/prompts.md` → Phase 0 section
- Verify Card encoding convention (`Card(u8)` internals vs our `rank * 4 + suit`)
- On PC later: `git clone --recurse-submodules https://github.com/dannyboy637/klondike-solver.git`

---

## Session 3 — 2026-03-03: Spec Review & Defect Fixes

### What happened
- Performed comprehensive self-review of all spec files (phases 0-6, overview, prompts, machine transfer plan)
- Received and analyzed external agent review with 12 findings (P0-P2 severity)
- Verified every finding against actual Lonelybot source code (`main.rs`, CLI, solver APIs)
- All 12 findings confirmed at least partially valid (0 false positives)
- Identified additional issues the external agent missed
- Applied all fixes across 10 files (265 insertions, 113 deletions)
- Updated prompts, current_context, session_log, and CLAUDE.md to reflect fixes

### External agent findings — verification results

| # | Finding | External Severity | Our Assessment | Action |
|---|---------|------------------|----------------|--------|
| 1 | CLI hoploop vs hop-loop | P0 | P1 (confirmed: Clap kebab-case) | Fixed all 5 occurrences in Phase 0 |
| 2 | MCTS root visit_count never incremented | P0 | P0 (confirmed: backprop only updates children) | Fixed in Phase 2 — added root.visit_count += 1 |
| 3 | Position-level train/val split | P0 | P0 (confirmed: data leakage) | Replaced with game_level_split() in Phase 3 |
| 4 | Append+concatenate RAM blowup | P1 | P0 for large N (confirmed: O(N²)) | Replaced with memmap-based writing in Phase 3 |
| 5 | Eval seeds overlap Phase 6 test sets | P0 | P1 (confirmed: 1M overlaps test_set_A) | Changed Phase 4 eval seeds to 5M+ |
| 6 | Wall-clock PIMC seeding | P1 | P1 (confirmed: breaks reproducibility) | Changed to deterministic seeding in Phase 2 |
| 7 | Runtime 5s/game was gut feel | P1 | P1 (confirmed: bottom-up gives 20-60s) | Updated Phase 4 from 30-40h to 170-200h total |
| 8 | 705 vs 857-dim inconsistency | P1 | P1 (confirmed: multiple files disagree) | Settled on 757 baseline across all files |
| 9 | Gate criterion math wrong | P1 | P1 (confirmed: 1000 games needs ~53% for CI>50%) | Added math explanation + SPRT recommendation |
| 10 | Grid search 1296 configs | P2 | P2 (confirmed: would take ~4.9 years) | Changed to one-at-a-time sweep (~20 configs) |
| 11 | max_moves 2000 too generous | P2 | P2 (agreed: unlimited passes need cycle detection) | Reduced to 500 + added cycle detection |
| 12 | Missing imports in specs | P2 | P2 (confirmed) | Added time, Callable, dataclass, F, etc. |

### Additional issues found (beyond external review)
- Stock and unknown sharing one-hot index 9 — separated into indices 9 and 10 (757 dims)
- Nested-loop HOP performance warning missing — added to Phase 2
- No cycle detection in Phase 4 self-play/eval — added state hash set
- Missing `game_offsets.npy` in data format spec — added to Phase 3

### Files modified
- `specs/phase-0-setup-baselines.md` — CLI naming fixes, --help verification note
- `specs/phase-1-pyo3-bridge.md` — encoding 757 dims, stock/unknown separated, max_moves 500
- `specs/phase-2-search-evaluation.md` — MCTS root fix, deterministic PIMC seeding, performance warning, max_moves 500
- `specs/phase-3-supervised-learning.md` — game-level split, memmap data gen, game_offsets.npy, max_moves 500
- `specs/phase-4-alphazero-selfplay.md` — eval seeds 5M+, cycle detection, runtime 170-200h, gate math, max_moves 500
- `specs/phase-6-iteration-optimization.md` — one-at-a-time sweep, encoding 757
- `specs/00-overview.md` — encoding 757 baseline
- `specs/prompts.md` — all phase prompts updated with key caveats
- `CLAUDE.md` — encoding 757
- `current_context.md` — encoding 757, spec review summary, eval seed protocol

### Memory files updated
- Created `memory/spec-review-lessons.md` — patterns to avoid in future specs
- Updated `memory/MEMORY.md` — defect fixes summary

### Key lessons captured
1. **MCTS invariants**: root.visit_count must be updated explicitly (backprop skips root)
2. **Bottom-up runtime estimates**: multiply per-move cost × moves/game × games — never use gut feel
3. **RAM estimation**: check peak memory for data pipelines, not just final output size
4. **Three disjoint seed sets**: model selection, tuning, and final claims must never overlap
5. **Game-level splitting**: any dataset where multiple rows come from one episode needs episode-level splits
6. **Single source of truth**: dimension constants should be defined once and referenced everywhere
7. **CLI verification**: always run --help before hard-coding command names in specs
8. **Hyperparameter sweep budgets**: compute total configs × training time before proposing sweep strategies

### Key decisions
- Encoding: 757-dim baseline (stock/unknown separated), ~857 for Phase 6 expansion
- All dimension references unified across all files
- Evaluation protocol: 4 disjoint seed ranges established

### Status at end of Session 3 (first pass)
- All 14 spec defects fixed and verified
- All documentation updated to reflect fixes

### Session 3 (continued) — Final Cleanup

Second external review found 6 remaining issues (P1-P3). All verified valid and fixed:

| # | Finding | Severity | Fix |
|---|---------|----------|-----|
| 1 | PIMC seeds not wired through call sites (benchmark, ablations, data gen all used defaults 0,0) | P1 | Updated `SearchFn` to `(env, game_seed, move_num)`. Fixed benchmark loop, all ablation lambdas, and Phase 3 `generate_game_data` |
| 2 | Uncertainty features were TODO zeros (15 of 67 dims) | P1 | Implemented: 1 global (unseen red fraction) + 14 per-pile (hidden share, useful-rank fraction) |
| 3 | Final-save `np.array(memmap[:N])` spiked RAM | P1 | Replaced with streaming copy to new correctly-sized memmap + atomic rename |
| 4 | Aux loss documented but not in training loop | P2 | Deferred aux loss to Phase 6. Removed from Phase 3 config/docstring. Aux heads remain for compatibility |
| 5 | Phase 3 runtime "2-4 sec/game" conflicts with Phase 2 "minutes/game" warning | P2 | Made conditional: fast path (2-10s), per-move HOP path (minutes) |
| 6 | MCTS `root.value` always ~0 (backprop never updates root) | P3 | Compute from children: `sum(c.value_sum) / sum(c.visit_count)` |

Also fixed Phase 5 acceptance criteria: split into core (manual input, advisor, rule fidelity) vs deferred (CV scaffolds).

### Files modified (final cleanup)
- `specs/phase-2-search-evaluation.md` — SearchFn signature, benchmark call site, ablation lambdas, MCTS root value
- `specs/phase-1-pyo3-bridge.md` — uncertainty features implemented
- `specs/phase-3-supervised-learning.md` — PIMC seed passthrough, memmap truncation, aux loss deferred, runtime estimate
- `specs/phase-5-deployment.md` — scaffold clarity, acceptance criteria split
- `CLAUDE.md` — known caveats updated
- `current_context.md` — final cleanup summary added
- `session_log.md` — this entry

### Final status
- All spec defects from both review rounds are fixed
- All 4 meta-files (prompts, CLAUDE.md, current_context, session_log) are current
- **Specs are ready for Phase 0 execution**
- **Next action: Start fresh session, paste Phase 0 prompt from `specs/prompts.md`, execute on PC**

---

## Session 4 — 2026-03-03: Phase 0 Execution (Partial — MacBook)

### What happened

#### Setup & Build
- Installed Rust toolchain on MacBook (wasn't installed)
- Built Lonelybot: `cargo build --release -p lonecli` — success
- Verified CLI help — confirmed kebab-case: `hop-loop`, `rand-solve`, `rate`, etc.
- Tested `print`, `solve`, `hop` commands successfully

#### Draw-3 Thoughtful Solvability (PASSED)
- Ran `rate` command (exact DFS solver) on ~8963 games
- **Result: 7361/8963 = 82.13%, Wilson 95% CI: (81.32%, 82.91%)**
- Matches published 81.95% ± 0.03% — **Gate 0 step 1 PASSED**

#### HOP Solver — Infinite Loop Bug
- Discovered that `do_hop()` in `lonecli/src/main.rs` has NO cycle detection or move limit
- The `while !game.state().is_win()` loop runs forever when games cycle
- With unlimited passes, `pick_moves()` always returns `Some(moves)` (draw-from-stock is always legal)
- Confirmed empirically: seed 10000 draw-1 cycles `PS 6♥, SP 6♥` hundreds of times without terminating
- This affected BOTH draw-1 and draw-3 benchmarks
- Background processes running `hop-loop` would hang on their first or second game

#### Fix Applied: MAX_MOVES limit
- Added `const MAX_MOVES: usize = 5000;` to `do_hop()` — games hitting this limit counted as losses
- Also changed verbose output from `println!` to `eprintln!` (Rust buffers stdout to 8KB when not tty; stderr is unbuffered)
- Rebuilds confirmed working: cycling games now terminate in ~7-8s instead of hanging forever

#### Critical Discovery: HOP Draw-3 Win Rate Much Lower Than Published
- Published result: **47.58% ± 0.80% (15K games)**
- Our measurement at HEAD (latest code): **2/50 = 4.0%, CI: (1.1%, 13.5%)**
- This is NOT a timeout/move-limit issue — 46 of 50 games ended as "Lost" (MCTS returned None), only 2 timed out
- Games complete quickly (~2.9s average) — they're just losing

#### Root Cause Investigation: Version Mismatch
- The 47.58% result was published at commit 3e88e66 (2024-05-14)
- Current HEAD is 63 commits later, including:
  - **`3ef9413` "breaking change to rng"** (2025-02-26): switched from `StdRng` to `SmallRng`, changed rand crate version
  - **`5ba05b4` "major change from top_mask to locked_mask"** (2024-10-17): major pruner change
  - **`7db65e6` "more_dominances"** (2025-05-03): merged new pruner dominances
  - Multiple other pruner and cleanup changes
- The old version (`3e88e66`) also:
  - Had `do_hop` **hardcoded to draw-3** (no draw_step parameter!)
  - Used `StdRng` instead of `SmallRng`
  - Did NOT have `ucb1` parameter in `pick_moves` call
  - Had `SolitaireEngine` vs direct `Solitaire` type (API changed)
- **Conclusion: the current HEAD code is a different solver than what produced 47.58%**

### Files created
- `run_benchmark.py` — parallel benchmark script with per-game timeout, multiprocessing, Wilson CI tracking
  - Reads stderr (not stdout) for result detection
  - Supports configurable draw_step, seed range, timeout, worker count

### Local modifications to Lonelybot (uncommitted, in stash)
- `lonecli/src/main.rs`:
  - Added `MAX_MOVES = 5000` to `do_hop()`
  - Changed `println!` → `eprintln!` for verbose output (fixes stdout buffering in pipes/background)
  - Changed `println!` → `eprintln!` for `hop_loop` stats output

### Key findings
1. **Draw-3 Thoughtful: 82.13% — MATCHES published 81.95%** ✅
2. **Draw-3 HOP at HEAD: ~4% — DOES NOT MATCH published 47.58%** ❌
3. Root cause: 63 commits of pruner/RNG changes since the published results
4. The old `do_hop` was hardcoded draw-3 only — no draw-1 support existed
5. The HOP solver has no cycle detection — games with unlimited passes loop infinitely

### What still needs to be done (Phase 0 continuation)
1. **Reproduce draw-3 HOP at old commit (3e88e66)** — build v0.2.2 tag, run benchmark, confirm ~47.58%
   - This tells us if our benchmark methodology is correct
   - Need to handle that old version has no move limit (use subprocess timeout)
   - Old version uses stdout, not stderr, for output
2. **Test draw-3 HOP at HEAD with larger sample** — maybe 50 games is too few (unlikely given 4% vs 47%)
3. **Investigate if current HEAD is a regression** — the pruner changes may have broken HOP
4. **Get draw-1 HOP baseline** — THE critical number. Need to decide which version to use:
   - Old version (3e88e66): no draw-1 support, would need to add draw_step param
   - Current HEAD: has draw-1 support but HOP appears broken for draw-3
   - May need to find an intermediate commit that supports draw-1 AND has working HOP
5. **Update session_log.md, current_context.md, CLAUDE.md** — done (this entry)

### Recommended approach for next session
1. **Start by running draw-3 HOP at both old (3e88e66) and HEAD versions** with 200+ games each
   - Old version: use subprocess timeout (120s) since no move limit; check stdout for "Solved"/"Lost"
   - HEAD: already built with MAX_MOVES, check stderr
2. If old version reproduces ~47.58%, the current HEAD is a regression
3. Then decide: add draw-1 support to old version, or bisect to find which commit broke HOP
4. The `run_benchmark.py` script needs the output detection adjusted per version (stdout vs stderr)

### Status at end of Session 4
- Gate 0 step 1 (Thoughtful draw-3): **PASSED** ✅
- Gate 0 step 2 (HOP draw-3 reproduction): **FAILED** — likely version mismatch, needs investigation
- Gate 0 step 3 (draw-1 HOP): **NOT YET ATTEMPTED**
- Lonelybot is currently checked out at HEAD (main) with stashed local changes (MAX_MOVES, eprintln)

---

## Session 5 — 2026-03-03: Gate 0 PASSED — Draw-1 HOP = 68.9%

### What happened

#### Reproduced Draw-3 HOP at Old Commit (3e88e66)
- Checked out commit 3e88e66, built, saved binary as `lonecli_v022`
- Ran 500 games draw-3: **247/500 = 49.4%, Wilson CI (45.0%, 53.8%)**
- Published 47.58% falls within our CI — **benchmark methodology VALIDATED**
- Zero timeouts on draw-3 at old commit
- Confirms HEAD is a regression (HEAD ~4% vs old ~49%)

#### Patched Old Version for Draw-1 Support
- Modified `do_hop()` at commit 3e88e66:
  - Added `draw_step: u8` parameter (was hardcoded to 3)
  - Added `MAX_MOVES = 5000` to prevent infinite loops
  - Changed `println!` → `eprintln!` to fix stdout buffering in pipes
- Added `draw_step` to Hop and HopLoop CLI commands (default: 3)
- Saved patched binary as `lonecli_v022_patched`
- Verified draw-3 behavior unchanged (28/50 = 56%, consistent with 49.4%)

#### THE Critical Measurement: Draw-1 HOP
- **First batch (seeds 0-499)**: 335/500 = 67.0%, CI (62.8%, 71.0%)
- **Second batch (seeds 500-1499)**: 698/1000 = 69.8%, CI (66.9%, 72.6%)
- **Combined (seeds 0-1499)**: **1033/1500 = 68.9%, Wilson CI (66.5%, 71.2%)**
- Zero timeouts across all 1500 games
- ~0.5s/game average (3.7-4.0s/game effective with 8 workers on M4 MacBook)

### Key Results Table

| Metric | Draw-3 (old) | Draw-1 (patched) |
|--------|-------------|-----------------|
| Win rate | 49.4% | **68.9%** |
| 95% CI | (45.0%, 53.8%) | **(66.5%, 71.2%)** |
| Sample size | 500 | 1500 |
| Timeouts | 0 | 0 |
| EV at 2x payout | ~-$0.01/$ | **+$0.38/$** |

### Gate 0 Assessment: ALL STEPS PASSED ✅

1. Draw-3 Thoughtful: 82.13% — matches published ✅
2. Draw-3 HOP: 49.4% — reproduces 47.58% (within CI) ✅
3. Draw-1 HOP: **68.9%** — massively exceeds 49.5% threshold ✅
4. **Conditional assessment: STRONG PROCEED** — baseline is 18.9% above breakeven

### Implications
- The ~50-55% extrapolation for draw-1 was **wildly conservative** — actual is ~69%
- Even without ANY neural net guidance, raw HOP already has massive edge at 2x payout
- The project goal of ">50%" is already achieved by the raw solver — neural nets could push even higher
- This changes the project calculus: the value of neural guidance is now about speed (real-time play) and possibly squeezing a few more %, not about crossing the 50% threshold
- **Gate 0.5 (operational viability)** becomes MORE important now — the edge exists, the question is can it be captured

### Files created/modified
- `lonecli_v022` — unpatched old binary (draw-3 only)
- `lonecli_v022_patched` — patched old binary (draw-1 support, MAX_MOVES, eprintln)
- `run_benchmark.py` — updated LONECLI path to patched binary
- `run_benchmark_old.py` — updated LONECLI path to patched binary
- `lonelybot/lonecli/src/main.rs` — patched at commit 3e88e66 (detached HEAD, local only)

### Lonelybot State
- Currently at detached HEAD (commit 3e88e66) with local modifications
- Stash contains HEAD (main branch) modifications from Session 4
- Saved binaries in project root: `lonecli_v022`, `lonecli_v022_patched`

### Next Steps
1. **Gate 0.5: Operational Viability** — Can the edge actually be captured on 1xbet?
   - What is the allowed decision latency per move?
   - Do Lonelybot's rules match 1xbet's implementation?
   - Account risk analysis
2. **Gate 1: Search-only ablations** — Can HOP parameters be tuned higher?
   - More MCTS iterations (currently 3000)
   - More determinizations per move
   - Different pruning strategies
3. **Proceed to Phase 1** (PyO3 bindings) — the baseline justifies the investment
4. Consider: is the raw solver already sufficient? Do we even need neural nets?

---

## Session 5 Continuation — Oracle Leak Discovery & Full-Shuffle Benchmarks

### What happened

#### Oracle Leak Identified in hidden_shuffle()
- Lonelybot's `hidden_shuffle()` in `hidden.rs` uses `split_last()` pattern
- This preserves the topmost hidden card in each tableau pile during shuffling
- Effect: the solver knows which card will be revealed next when uncovering a pile
- This is oracle information — real imperfect-info play doesn't have this knowledge

#### Fix: shuffle_all() Method
- Implemented `shuffle_all()` in `hidden.rs` — two-step shuffle:
  1. Shuffle deeper hidden cards as before
  2. Additionally shuffle topmost hidden cards among themselves
- This removes all oracle information from the determinization
- Architecture: `hop_solver.rs:76` uses `shuffle()` (not `shuffle_all()`) for inner determinizations; `main.rs do_hop()` uses `shuffle_all()` for outer loop
- Built as separate binary: `lonecli_fullshuffle`

#### Benchmark Results — Full Shuffle (True Imperfect Information)

| Variant | Oracle (preserve topmost) | Full Shuffle (true imperfect info) | Delta |
|---------|--------------------------|-----------------------------------|-------|
| Draw-3 | 49.4% CI(45.0, 53.8) n=500 | 31.0% CI(25.0, 37.7) n=200 | -18.4pp |
| Draw-1 | 68.9% CI(66.5, 71.2) n=1500 | 49.0% CI(45.9, 52.1) n=1000 | -19.9pp |

### Key Findings
1. **Oracle leak worth ~19 percentage points** — consistent across both draw modes
2. **Draw-1 full-shuffle HOP: 49.0%** — just below breakeven, but 50% is inside the 95% CI
3. **Draw-3 full-shuffle: 31.0%** — dramatically below published 47.58%, confirming oracle explained most of the gap
4. The previous 68.9% draw-1 number was inflated by oracle knowledge, not representative of true imperfect-info play

### Gate 0 Assessment (Revised)
- Previous assessment was overly optimistic — the 68.9% included oracle information
- True draw-1 imperfect-info baseline: **49.0% +/- 3.1pp**
- This is borderline — exactly at the 49.5% conditional threshold from CLAUDE.md
- Neural guidance is now critical for crossing 50%, not just an optimization
- The project calculus reverts: this IS a research challenge, not a solved problem

### Code Changes
- `lonelybot/solitaire/src/hidden.rs` — added `shuffle_all()` method
- `lonelybot/solvers/src/hop_solver.rs:76` — uses `shuffle()` for inner determinizations
- `lonelybot/lonecli/src/main.rs` — `do_hop()` outer loop uses `shuffle_all()`
- Binary: `lonecli_fullshuffle`

### Implications
- The raw solver does NOT already exceed breakeven under true imperfect info
- Neural guidance is genuinely needed to push past 50%
- The ~49% baseline is encouraging — only need ~1-2pp improvement from neural nets
- Gate 1 search-only ablations (more determinizations, ISMCTS) become critical path
- The oracle leak explains why published Lonelybot results (~47.58% draw-3) seemed surprisingly high for imperfect-info play

---

## Session 5b — Codebase Audit, Decode Fix, Fast Params

### Codebase Audit
Performed full audit of Lonelybot for information leaks beyond the oracle leak. Findings:

**CRITICAL (fixed)**:
- `hidden.rs:decode()` did not call `update_map()` or `compute_first_layer_mask()` after restoring `n_hidden` counts. This left `pile_map` and `first_layer_mask` stale after encode/decode cycles in MCTS. Fixed by adding both calls to `decode()`.
- Impact: `first_layer_mask` is used in `mcts_solver.rs:41` and `pruning.rs:57` — stale values could cause wrong move pruning/filtering.
- Verified fix doesn't change win rate on same seeds (490/1000 = 49.0% both before and after) — the bug likely only triggers in edge cases not hit in current HOP flow.

**HIGH (documented, safe for now)**:
- `get_visible_piles()` in engine.rs uses `hidden.peek(pos)` — exposes topmost hidden card identity. Only used in tui.rs (display) and convert.rs (move output), NOT in solvers. Must not be used in neural net state encoding.
- `to_piles()` in hidden.rs uses same `split_last()` oracle pattern — safe if not in learning pipeline.

**LOW**:
- `clear()` preserves topmost cards (asymmetric with `shuffle_all()`) — could reintroduce oracle if misused
- `mask()` returns cards below topmost only — naming could be clearer

### Fast Parameters Test
Tested reduced HOP params (N_TIMES=500, LIMIT=500) vs original (3000/1000):

| Config | Win Rate | CI | Speed |
|--------|----------|-----|-------|
| Heavy (3000/1000) | 49.0% (seeds 0-399, n=400) | (43.9%, 53.6%) | ~5.3s/game |
| Fast (500/500) | 48.0% (seeds 0-499, n=500) | (43.7%, 52.4%) | ~0.5s/game |

- **0.8pp degradation for 11× speedup** — excellent tradeoff for large-N benchmarks
- Binary: `lonecli_fast` (500/500 params, full shuffle, decode fix)

### 1M Game Benchmark Plan
To get definitive below/above 50% answer with tight CI (±0.1pp):

**Three-machine parallel run:**
| Machine | Threads | Games | Seeds | Est. Time |
|---------|---------|-------|-------|-----------|
| M4 MacBook | 8 | 250K | 0-249999 | ~4.3 hrs |
| M1 MacBook | 8 | 250K | 250000-499999 | ~5.5 hrs |
| PC Ryzen 5700X | 16 | 500K | 500000-999999 | ~4.3 hrs |

Commands:
```bash
# Build on each machine first (same source, different architectures)
cd lonelybot && cargo build --release -p lonecli

# M4 MacBook
python3 run_benchmark.py 1 0 250000 180 8 results_m4.txt

# M1 MacBook
python3 run_benchmark.py 1 250000 250000 180 8 results_m1.txt

# PC Ryzen 5700X
python3 run_benchmark.py 1 500000 500000 180 16 results_pc.txt
```

Combine by raw counts: `wins_total = sum(wins_i)`, one Wilson CI on `(wins_total, n_total)`.

### Binaries
- `lonecli_fullshuffle` — heavy params (3000/1000), full shuffle, decode fix
- `lonecli_fast` — fast params (500/500), full shuffle, decode fix
- `run_benchmark.py` — currently points to `lonecli_fast` (change LONECLI to switch)

### Code Changes (all in lonelybot/ at commit 3e88e66)
- `src/hidden.rs`: Added `shuffle_all()`, fixed `decode()` (added `update_map()` + `compute_first_layer_mask()`)
- `src/hop_solver.rs:76`: Uses `shuffle()` for inner determinizations
- `lonecli/src/main.rs`: `do_hop()` uses `shuffle_all()`, configurable N_TIMES/LIMIT constants
