# Klondike Solitaire — AlphaZero-Style Imperfect Info Solver

## Project Goal
Build the first AlphaZero-style solver for Klondike Solitaire under imperfect information. Target: >50% win rate on draw-1, unlimited passes, no undo. At 2x payout, every percentage point above 50% is pure edge.

**Calibrated confidence that this plan yields a statistically defensible >50%: ~30-45%.** This is a research exploration, not a guaranteed outcome. All architecture choices are leading candidates, not final commitments — they must be validated empirically at each gate.

**Conditional confidence** (updated after Gate 0):
- If draw-1 HOP baseline ≥49.5% with tight CI → crossing 50% with learned guidance is plausible
- If draw-1 HOP baseline ≤48.5% → odds drop sharply unless search approach changes materially

## Why This Matters
Nobody has applied modern ML to Klondike. All existing research is classical search (constraint programming, DFS, heuristic rule lists). The research gaps are documented in `klondike-master.md` and `/memory/research-gaps.md`. This is genuinely unexplored territory as of March 2026.

## Key Numbers (memorize these)
- Theoretical solvability (Thoughtful, draw-1): ~82-91%
- Best human (no undo): ~43%
- Best heuristic solver (Turnbull, closed source, UNVERIFIED): ~52.9-53.4%
- Best MCTS (Lonelybot, draw-3): 47.58% ± 0.80%
- **Lonelybot MCTS on draw-1: UNKNOWN — this is the first thing to determine**
- Breakeven at 2x payout: 50%

**Source quality warning**: Turnbull's 53% is closed source and unverified. The draw-1 "~50-55%" is an extrapolation, not a measurement. Core numbers (Solvitaire, Lonelybot draw-3) are from published papers with reproducible code.

## Architecture: AlphaZe** PC-PIMC (Leading Candidate — Not Final)
Detailed research: `alphazero-klondike-research.md`

- Replace standard MCTS with **PC-PIMC** (Policy Combining Perfect Information Monte Carlo)
- At each decision: sample n determinized worlds → run MCTS on each → average policies → act
- Neural net sees ONLY visible cards (not determinized state) — forces generalization across info sets
- **Determinization counts**: 3-5 during training (speed), 50-100 during evaluation (precision)
- **TrueSight Learning**: Phase 1 train on Thoughtful Solitaire (all visible), Phase 2 switch to PIMC
- PIMC's strategy fusion concern is **mitigated** (not eliminated) by non-adversarial setting. Optimistic bias from HOP is real and requires attention (ISMCTS, RecPIMC, EPIMC are known mitigations).
- Reference: AlphaZe** (Blüml, Czech & Kersting, Frontiers in AI, 2023)

### Neural Net Architecture (Leading Candidate)
- **757-dim hybrid state encoding** (baseline): binary card locations (11 one-hot per card) + uncertainty summary
  - **Phase 6 expansion to ~857**: stock-pass context, action history, move-cycle features, richer per-position uncertainty
- **MLP with residual blocks** (NOT CNN — card games lack spatial locality)
  - **Alternative to evaluate**: transformer/set-network for relational card-card interactions, LSTM history encoder
- ~2M parameters, ~8 MB FP32, ~100 MB VRAM at batch 1024 (1.2% of 4060)
- Dual heads: policy (move probabilities) + value (win probability)
- No batch normalization — use Fixup initialization (KataGo optimization)
- Auxiliary targets: predict foundation count, face-down cards remaining (sharper gradients)

### Training Estimates
- 50K-500K training games needed (100K realistic middle target)
- **First credible result: 6-12 weeks. Robust reproducible claim: 3-6 months.**
- 100-900 games/hour at 400 MCTS sims/move on RTX 4060 + Ryzen 5700X
- Most likely timeline blowups: Rust/Python integration, batched in-tree inference, hyperparameter tuning, statistically valid evaluation at thin margins

## Technical Stack
- **Lonelybot** (Rust, MIT): game engine, wrapped via **PyO3** bindings (~60-70ns call overhead)
- **Python + PyTorch**: neural net training, custom based on alpha-zero-general
- **PyO3 + maturin**: Rust-Python bridge (NOT subprocess — 1000x faster)
- **Hardware**: Ryzen 5700X (16 threads) for MCTS self-play, RTX 4060 (8GB) for training, M4 MacBook for dev

## Project Phases — Gated Approach
Each gate must pass before proceeding. Predefined stop criteria at each stage.

### Gate 0: Validate Setup & Reproduce Known Results
1. Clone Lonelybot, build, verify on draw-3 Thoughtful (should match 81.95%)
2. Run draw-3 HOP/MCTS, reproduce 47.58% ± 0.80%
3. **Only then** run draw-1 MCTS with 50K-100K games using paired seeds
4. Record draw-1 baseline with tight CI
5. **Conditional assessment**: if baseline ≥49.5% → proceed with confidence; if ≤48.5% → reassess approach before continuing

### Gate 0.5: Operational Viability Check (Lightweight — Before Deep RL Investment)
- What is the allowed decision latency per move on 1xbet? (test empirically)
- Are there hidden timers, forced completion windows, disconnection penalties?
- Do Lonelybot's rules actually match 1xbet's implementation? (spot-check a few games)
- What are realistic account-risk assumptions? (max win before flag, withdrawal history)
- This is NOT full deployment engineering — it's preventing investment in an edge that cannot be captured

### Gate 1: Search-Only Ablations (Before Any Neural Nets)
5. Test HOP variants on draw-1: vary determinization counts, iteration budgets
6. Try ISMCTS (information-set MCTS) as alternative to HOP — may reduce optimistic bias
7. Establish the best search-only win rate as the baseline to beat
8. **Stop criterion**: if best search-only CI upper bound < 49% after full budget, reassess scope

### Gate 2: Cheap Supervised Learning (Before Full Self-Play)
9. Build PyO3 bindings to Lonelybot
10. Generate training data from search-only solver (state, best move pairs)
11. Train supervised policy net — does it match search quality at 100x speed?
12. This validates the state encoding and training pipeline cheaply

### Gate 3: Full AlphaZe** Self-Play
13. TrueSight Phase 1: train on Thoughtful Solitaire (perfect info)
14. TrueSight Phase 2: switch to PIMC with hidden information
15. Evaluate against search-only baseline with paired seeds and proper CI
16. **Stop criterion**: if paired CI upper bound < 50.2% after X compute, stop or pivot

### Gate 4 (Optional): Deployment
17. Build CV pipeline for real-time online play assistance
18. Rule fidelity verification against actual 1xbet implementation
19. Latency, input error, and execution drift analysis

## MuZero — Deferred (Not Ruled Out)
- Could theoretically squeeze extra 1-2% via learned latent world model
- Captures hidden card uncertainty without explicit determinization
- BUT: 3 learned functions, 5-10x more training, months instead of weeks on consumer hardware
- LightZero benchmarks: "use AlphaZero when you have a simulator" — we have Lonelybot
- **Decision: start with AlphaZe**, only revisit MuZero if AlphaZe** plateaus below target**
- Middle ground: BetaZero (Stanford 2024) — AlphaZero for POMDPs with particle belief (Julia)

## 1xbet Rules (the exact target)
- Draw-1, unlimited passes, no undo, no hints
- Standard 52-card deck, 7 tableau columns, kings-only to empty columns
- 2x payout (bet $1, get $2 on win, lose $1 on loss)
- Curacao license, proprietary RNG, unverifiable fairness
- **Rule fidelity risk**: any mismatch between Lonelybot's rules and 1xbet's actual implementation invalidates results. Must verify empirically.

## File Structure

### Implementation Specs (start here for coding)
- `specs/prompts.md` — **copy-paste prompts for each phase** (solo + team agent variants)
- `specs/machine-transfer-plan.md` — git sync workflow between MacBook and PC
- `specs/00-overview.md` — master plan, dependency graph, decision gates, directory structure
- `specs/phase-0-setup-baselines.md` — build Lonelybot, reproduce baselines, get draw-1 number
- `specs/phase-1-pyo3-bridge.md` — Rust PyO3 crate + Python interface + encoding + actions
- `specs/phase-2-search-evaluation.md` — PIMC search + MCTS + statistical evaluation
- `specs/phase-3-supervised-learning.md` — data generation + MLP-ResNet + supervised training
- `specs/phase-4-alphazero-selfplay.md` — PC-PIMC self-play + replay buffer + training pipeline
- `specs/phase-5-deployment.md` — CV pipeline + move advisor (optional)
- `specs/phase-6-iteration-optimization.md` — hyperparameter tuning, experiments, scaling, stop criteria

### Research & Reference
- `alphazero-klondike-research.md` — deep research: PIMC theory, AlphaZe** architecture, state encoding
- `klondike-master.md` — compiled research document (analysis + discussion notes + project plan)
- `quant-techniques-reference.md` — variance reduction, Brier score, particle filter code
- `current_context.md` — living research doc (evolving understanding, hypotheses, key findings)
- `session_log.md` — detailed session log with reasoning and decisions

### Source (original)
- `klondike-analysis.md` — original EV analysis
- `discussion-notes.md` — notes from initial research review
- `klondike research.rtf` — original project plan with phases

### Code (built during implementation)
- `lonelybot/` — Lonelybot repo (git submodule, unmodified)
- `klondike-env/` — Rust PyO3 crate (Phase 1)
- `klondike/` — Python package (Phases 1-4)
- `klondike-rs/` — **Pure Rust rebuild** (see below)

## Rust Rebuild (klondike-rs/)
The Python pipeline works but is too slow for heavy experimentation. `klondike-rs/` is a pure Rust rebuild targeting 50-100x speedup via rayon parallelism, arena-allocated MCTS, and batched GPU inference (tch-rs). See `specs/rust-rebuild-plan.md` for the full plan and `klondike-rs/CLAUDE.md` for Rust conventions.

- **Python** (`klondike/`, `klondike-env/`, `scripts/`): original implementation, kept as-is for fallback
- **Rust** (`klondike-rs/`): 5-crate workspace (core, search, nn, eval, train). Direct Lonelybot dependency, no FFI.
- **tch-rs setup**: requires `LIBTORCH_USE_PYTORCH=1` and `DYLD_LIBRARY_PATH` pointing to PyTorch's lib dir
- Build: `cd klondike-rs && LIBTORCH_USE_PYTORCH=1 VIRTUAL_ENV=../.venv PATH=../.venv/bin:$PATH cargo build`
- Test: same env vars + `DYLD_LIBRARY_PATH=../.venv/lib/python3.14/site-packages/torch/lib cargo test --all`
- **Team agents**: Phases R1, R3, R4, R6 use parallel Claude Code team agents (2 agents per phase). Prompts in `specs/rust-rebuild-plan.md` include team split instructions. Phases R2, R5, R7 are single-agent.

## Repository
- **GitHub**: `https://github.com/dannyboy637/klondike-solver` (private)
- **Lonelybot**: git submodule — clone with `--recurse-submodules`
- **Machine workflow**: MacBook (Phases 0-3, MPS) ↔ PC (Phase 4+, CUDA). See `specs/machine-transfer-plan.md`
- **PyTorch device**: use `"mps"` on Mac, `"cuda"` on PC (auto-detect in code)

## Evaluation Seed Protocol (Three Disjoint Sets — Never Mix)
- **Phase 4 model selection**: seeds 5,000,000+ (1000 games per eval checkpoint)
- **Phase 6 test_set_C (tuning)**: seeds 3,000,000 - 3,004,999 (5K games)
- **Phase 6 test_set_A (final claims)**: seeds 1,000,000 - 1,009,999 (10K games)
- **Phase 6 test_set_B (confirmation)**: seeds 2,000,000 - 2,009,999 (10K games)
- **Rule**: NEVER tune hyperparameters on test_set_A/B. Final claims on test_set_A only, confirmed on test_set_B.

## Session 4 Critical Finding: HOP Regression at HEAD
- **Draw-3 Thoughtful: 82.13% — CONFIRMED** ✅
- **Draw-3 HOP at HEAD: ~4% — DOES NOT MATCH published 47.58%** ❌
- Root cause: 63 commits since published results (RNG change, pruner redesign, dominance rules)
- Published results were at commit `3e88e66` (2024-05-14, v0.2.2 era)
- Old `do_hop()` was **hardcoded draw-3** — no draw_step parameter existed
- HOP has **no cycle detection** — infinite loops on unlimited passes. Local fix: `MAX_MOVES = 5000`
- Local modifications in `lonecli/src/main.rs` (uncommitted on main): MAX_MOVES, eprintln for buffering
- Benchmark script: `run_benchmark.py` (parallel workers, subprocess timeout, Wilson CI)
- **NEXT**: reproduce 47.58% at old commit, then figure out draw-1 support

## Known Spec Caveats (from Session 3 review + final cleanup)
- MCTS: root.visit_count must be incremented after each `_simulate()` call — backprop only updates children
- MCTS: root value must be computed from children (`sum(c.value_sum) / sum(c.visit_count)`), not `root.value`
- PIMC seeding: always deterministic (`game_seed * 10000 + move_num * 100 + d`), never wall-clock
- SearchFn signature: `Callable[[KlondikeEnv, int, int], SearchResult]` — all call sites must pass `(env, game_seed, move_num)`
- Data generation: use memmap throughout, including final truncation (streaming copy to new memmap, not `np.array()`)
- Train/val split: game-level, not position-level (prevents same-game data leakage)
- Aux loss (foundation count, face-down prediction): deferred to Phase 6. Model has aux heads but Phase 3 trains policy+value only.
- Cycle detection: track state hashes in self-play/eval, break on repeat (unlimited passes can loop)
- Runtime: ~20-60s/game with PIMC+MCTS, ~170-200 hours total for 100 iterations × 200 games. Phase 3 data gen runtime depends on HOP approach (fast path: 2-10s/game, per-move HOP: minutes/game).
- CLI: Lonelybot uses Clap kebab-case (`hop-loop`, not `hoploop`) — always verify with `--help`

## Guidelines for Claude
- Always check `current_context.md` before starting work — it has the latest state of understanding
- Check `session_log.md` for what's been tried and what failed
- **Falsification discipline**: seek evidence against the plan, not just for it. Every claim needs a source, sample size, and CI.
- **No premature commitment**: architecture choices are hypotheses until validated by experiment. Use language like "leading candidate" not "decided."
- Be skeptical of win rate claims without verified source and sample size
- The 53% Turnbull number is unverified — treat it as a hypothesis, not a fact
- **Lonelybot draw-1 <50% does NOT mean "definitively -EV for everyone"** — it means that specific solver config is below breakeven. Neural guidance could still push past 50%.
- When evaluating at thin margins around 50%, use paired seeds, SPRT/sequential testing, and watch for evaluation leakage from repeated tuning on fixed test sets
- Respect the gates. Don't skip ahead to Gate 3 before passing Gates 0-2.
- For neural net work, the 4060 (8GB VRAM) is the training GPU — size networks accordingly
- Keep code simple. This is a research/exploration project, not production software.

## Additional Research to Investigate
- POMCP (Silver & Veness, 2010) and POMCPOW variants for online POMDP planning
- SP-MCTS literature (single-player MCTS / puzzle adaptations) for rollout/value init design
- Paired statistical testing methodology (SPRT, sequential testing) for thin-margin evaluation
- Belief-state modeling baselines (particle-belief search) as non-neural control arm
