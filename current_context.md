# Current Context — Living Research Document

Last updated: 2026-03-03 (post-spec plan)

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

### Critical Unknown #1: Lonelybot MCTS on Draw-1
- The single most important number for the entire project
- Draw-3 is 47.58%. Draw-1 should be higher (each stock card individually accessible)
- Extrapolation suggests ~50-55% but this is a **GUESS, not a measurement**
- Nobody has published this number
- **Important**: even if this comes back <50%, it does NOT mean ">50% is impossible for everyone." It means that specific solver/config is below breakeven. Neural guidance could still push past 50%.
- **Status: not yet tested**

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

### State Encoding — Proposal D (~857 dims) — Leading Candidate
- Binary card-centric encoding (~700 values): for each of 52 cards, encode location + visibility
- Uncertainty summary (~157 values): unseen card set mask, per-pile belief hints (red/black fraction, rank distribution)
- 3.4 KB per state, batch 4096 = ~14 MB — trivial for 8GB VRAM
- **Known gaps to address**: stock-pass context (which pass through stock?), action history (recent moves), move-cycle features (detecting loops), richer per-position uncertainty

### Neural Net Architecture — Leading Candidate
- MLP with 4 residual blocks × 512 hidden units (NOT CNN — card games lack spatial locality)
- ~2M parameters, ~8 MB FP32
- Dual heads: policy (move probabilities) + value (win probability)
- Auxiliary targets: foundation count, face-down cards remaining (KataGo insight — sharper gradients)
- No batch normalization — Fixup initialization (KataGo: 1.6-1.8x speedup)
- **Alternatives to evaluate if MLP plateaus**: transformer/set-network (relational card-card interactions), LSTM history encoder (implicit belief tracking)

### Integration: Lonelybot via PyO3
- PyO3 bindings with maturin — ~60-70ns call overhead (vs 1-10ms subprocess)
- Release GIL during Rust computation for true multi-threading
- Lonelybot already has a `python/` directory suggesting some interop work started
- Key modifications: observation vector extraction, action space definition, PIMC sampling
- **Biggest engineering risk**: Rust-Python integration latency in MCTS inner loop

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
- PyO3 integration latency in MCTS inner loop — biggest engineering risk
- The edge is so thin (1-3%) that variance makes it impractical even if real
- Evaluation leakage: repeated tuning on fixed test sets with small deltas around 50% creates false positives
- Rule fidelity: Lonelybot rules may not match 1xbet exactly
- Solver win rate ≠ deployable casino EV (latency, input errors, human execution drift, ban risk, withdrawal risk)
- 1xbet detects solver-assisted play and bans account
- 1xbet's RNG is not fair, nullifying theoretical edge

## Research Questions Informed by Research (Leading Answers, Not Final)
- State representation → Proposal D ~857-dim hybrid encoding (but missing stock-pass, action history, cycle features)
- Belief state encoding → Uncertainty summary or LSTM history encoder
- PIMC determinizations → 3-5 training, 50-100 eval
- Game engine → Rust (Lonelybot) via PyO3 bindings
- Architecture → MLP-ResNet (but transformer/set-network is realistic alternative)
- MuZero vs AlphaZe** → AlphaZe** first, MuZero if plateau

## Implementation Specs — Complete

Full spec plan in `specs/` directory (7 files). Each phase is self-contained for a fresh Claude Code session.

| Phase | Gate | What Gets Built |
|-------|------|-----------------|
| 0 | Gate 0 | Build Lonelybot, reproduce baselines, get draw-1 number |
| 1 | Gate 1 prep | PyO3 Rust crate + Python game interface + encoding + actions |
| 2 | Gate 1 | PIMC search + Python MCTS + statistical evaluation framework |
| 3 | Gate 2 | Training data + MLP-ResNet (~2.7M params) + supervised training |
| 4 | Gate 3 | PC-PIMC self-play + replay buffer + AlphaZero training pipeline |
| 5 | Gate 4 | CV pipeline + move advisor (optional, only if >50%) |
| 6 | Post-3 | Hyperparameter tuning, architecture experiments, scaling |

**All foundation code is built by end of Phase 4.** Phase 6 is experiments on that foundation.

### Resolved Questions (from spec work)
- Action space: 5 move types × 52 cards = 260, with legal masking (N_MOVES_MAX = 24 per position)
- Q-value init: V(s) not 0 — critical for single-player MCTS
- Replay buffer: 100K-500K positions (200K default)
- Temperature: τ=1.0 first 15-20 moves, then τ→0
- Observation encoding: ~705 dims baseline (expandable to ~857 with stock-pass, history, cycle features)
- Lonelybot has built-in PIMC support via `hidden_shuffle(rng)` — no modifications needed
- SolitaireEngine<FullPruner> is the right API to wrap (public do_move, list_moves_dom, undo_move)
- No existing PyO3 bindings — must build from scratch in `klondike-env/` crate

### Still Open (for Phase 6 experiments)
- ISMCTS vs HOP: which gives less optimistic bias for Klondike?
- Encoding additions: do stock-pass/history features meaningfully improve performance?
- Auxiliary targets: how much do foundation count / face-down predictions help?
- Architecture: MLP-ResNet vs transformer/set-network

## Additional Research to Investigate
- POMCP (Silver & Veness, 2010) and POMCPOW variants
- SP-MCTS literature (single-player puzzle adaptations)
- SPRT / sequential testing for thin-margin evaluation
- Belief-state modeling baselines (particle-belief search) as non-neural control arm
