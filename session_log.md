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

### Open threads for next session
- **Execute Phase 0**: Build Lonelybot on PC, reproduce baselines, get draw-1 number
- Verify Card encoding convention (`Card(u8)` internals vs our `rank * 4 + suit`)
- Transfer project to PC for building (MacBook <20GB, specs only)
