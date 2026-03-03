# Klondike Solitaire 1xbet EV Analysis — Master Document

## Objective

Empirically determine the win rate of Klondike Solitaire under 1xbet's exact ruleset (draw-1, unlimited passes, no undo) to calculate whether the 2x payout creates a +EV opportunity. The breakeven win rate is 50%. Academic research suggests theoretical solvability is ~82–91% but practical imperfect-information play drops to ~33–53% depending on player/solver sophistication.

---

## 1xbet Ruleset (Exact Rules)

- Standard 52-card deck
- 7 tableau columns (1 to 7 cards each), top card face up, rest face down
- 4 foundation piles, built up by suit from Ace to King
- Draw 1 card at a time from stock pile
- Unlimited passes through the stock (can cycle as many times as needed)
- Tableau building: descending order, alternating colors
- Can move single cards or valid sequences between columns
- Only Kings can fill empty tableau columns
- No undo functionality
- No hints
- Win condition: all 52 cards moved to foundations
- Payout: 2x on win (bet $1, get $2 back), lose $1 on loss

---

## Core Finding

**A skilled player without undo wins roughly 33–43% of Klondike draw-1 games with unlimited passes — well below the 50% threshold needed for a 2x payout to break even.** The gap between theoretical solvability (~82–91%) and what humans actually achieve (~33–43%) is enormous, and that gap is precisely what makes this game profitable for any casino offering it. The "no undo" rule is the single most important mechanism converting a potentially beatable game into a reliable house earner. Even the world's best Klondike AI, playing without perfect information, barely cracks 52–53% — a razor-thin edge that no human can replicate consistently.

---

## Theoretical Solvability

### Thoughtful Klondike (Perfect Information — All Cards Visible)

This gives the upper bound on solvability, since no information is hidden.

| Variant | Solvability | Source | Sample Size |
|---|---|---|---|
| Draw-3, unlimited passes | 81.945% ± 0.084% | Solvitaire (Blake & Gent, JAIR 2026) | ~1M deals, 2.4yr CPU |
| Draw-3, unlimited passes | 81.95% ± 0.03% | Lonelybot (vuonghy2442, 2024) | 5.9M games |
| Draw-1, unlimited passes | ~82–91% | Community estimate / extrapolation | Not rigorously established |

- The commonly cited "~80% winnable" figure originates from these Thoughtful Klondike studies
- Bjarnason, Tadepalli & Fern at Oregon State first established "no less than 82%" in 2007
- The 82% floor comes from draw-3 (draw-1 is strictly easier since every stock card is individually accessible)
- The ~90.5% ceiling is a community estimate, not a rigorous academic result
- Total possible Klondike deals: ~8 x 10^67 — impossible to enumerate, must sample
- If a deal is solvable and you have perfect information, a solver finds the winning path 100% of the time — it's a finite search problem
- The ~9–18% of unsolvable deals are mathematically impossible regardless of computing power
- Persi Diaconis of Stanford famously called the question of Klondike's win rate "one of the embarrassments of applied mathematics" (NIPS 2005)

### Why the Draw-1 Solvability Range Is So Wide

The 82–91% range is not a statistical noise issue — it exists because no large-scale rigorous study has been done for draw-1 specifically. Draw-3 is pinned down tightly thanks to Solvitaire and Lonelybot. Draw-1 solvability is strictly higher than draw-3, but the precise number has not been established with comparable rigor.

---

## Practical Win Rates (No Undo)

### Summary Table

| Variant | Player Type | Win Rate | Source |
|---|---|---|---|
| Draw-1, unlimited, no undo | Average app player | 33% | Solitaired.com (2.9M games) |
| Draw-1, unlimited, no undo | Experienced self-reported | 29–36% | Forum/community reports |
| Draw-1, unlimited, no undo | Best documented skilled human | ~43% | Jupiter Scientific (442 games) |
| Draw-1, unlimited, no undo | Heuristic computer solver | 52.9–53.4% | Bill Turnbull (10M games) |
| Draw-1, unlimited, no undo | MCTS AI (estimated) | ~50–55% | Extrapolated from Lonelybot draw-3 |
| Draw-1, unlimited, with undo | Aggressive human | ~91% | Community reports |
| Draw-3, unlimited, no undo | Average app player | 11.1% | Solitaired.com (1.4M games) |
| Draw-3, unlimited, no undo | Heuristic solver | 17.8% | Bill Turnbull (10M games) |
| Draw-3, unlimited, no undo | MCTS AI (Lonelybot) | 47.58% ± 0.80% | Lonelybot (15K games) |
| Draw-3, unlimited, Thoughtful | Perfect play | 81.95% | Lonelybot (5.9M games) |
| Draw-1, unlimited, Thoughtful | Perfect play | ~85–91% | **Needs to be determined** |

**No documented human has won more than ~43% of draw-1 unlimited-pass games without undo.**

### The ~53% AI Win Rate (Imperfect Information)

- Does **not** require a supercomputer
- **Turnbull's heuristic solver** (52.9–53.4%): hand-crafted rule priority list, runs on a laptop, 10M games tested
- **Lonelybot MCTS** (~47.6% draw-3, est. ~52–53% draw-1): Monte Carlo Tree Search, heavier but still desktop-grade
- Neither uses pre-computed lookup tables or databases of solved deals — each game is played fresh
- The AI advantage over humans isn't raw compute — it's consistency, perfect memory, and never making emotional/fatigue errors

### The Impact of Undo

The undo button transforms Klondike from a game of imperfect information into something approaching Thoughtful Solitaire:

| Variant | Without Undo | With Undo | Multiplier |
|---|---|---|---|
| Draw-3, unlimited passes | 10–12% | 52.35% (5,551 games) | 4–5x |
| Draw-1, unlimited passes | ~33% | ~53–56% | ~1.6x |

The mechanism: Klondike presents critical decision points where imperfect information forces guessing — which King to place in an empty column, which of two same-rank cards to move, foundation vs. tableau placement, when to cycle the stock. Without undo, each wrong guess can doom an otherwise winnable game. With undo, you try both options and keep the better outcome. **The "no undo" restriction is the mathematical foundation of the house edge.**

---

## EV Mathematics

### Core Formula

```
EV per $1 bet = (win_rate x 2) - 1
Breakeven requires win_rate > 0.50 (50%)
```

### EV by Player Type

| Scenario | Win Rate | EV per $1 Bet | House Edge |
|---|---|---|---|
| Average player (draw-1, unlimited, no undo) | 33% | -$0.34 | 34% |
| Skilled human (best documented) | 43% | -$0.14 | 14% |
| Hypothetical perfect human matching best AI | 53% | +$0.06 | -6% (player edge) |
| With unlimited undo | 53–56% | +$0.06 to +$0.12 | -6% to -12% |

### Comparison to Other Casino Games

| Game | House Edge |
|---|---|
| Blackjack (optimal play) | 0.5% |
| Baccarat | 1.06% |
| American roulette | 5.26% |
| Slots (typical) | 2–15% |
| **Klondike at 2x (skilled human)** | **14%** |
| **Klondike at 2x (average player)** | **34%** |

The game's effective RTP is 66–86% depending on skill, compared to the industry slot average of 93–97%.

### Hourly Rate Analysis

```
hourly_ev = ev_per_bet x bet_size x games_per_hour
Average game duration: 5–15 minutes (4–12 games per hour)
```

| Win Rate | Bet Size | Games/Hr | Hourly EV |
|---|---|---|---|
| 53% | $100 | 6 | +$36/hr |
| 43% | $100 | 6 | -$84/hr |
| 33% | $100 | 6 | -$204/hr |

### Variance & Bankroll

- Binary outcome — model as Bernoulli trials
- Standard deviation per game ≈ 0.499 at ~50% rate — very high variance relative to tiny edge
- Kelly criterion at 53%: bet 6% of bankroll (f* = (bp - q) / b where b=1, p=0.53, q=0.47)

---

## Why the Casino Doesn't Need to Rig the Shuffle

From the house's perspective, Klondike solitaire at 2x is remarkably profitable to offer. The mathematics of imperfect information deliver a built-in edge of 14–34% depending on player skill, vastly exceeding the margins on blackjack, baccarat, or roulette. The game also creates a powerful **illusion of control** — players believe skill can overcome the odds, and the theoretical solvability of most deals reinforces this belief.

As the Wizard of Odds (Michael Shackleford) has noted: "It isn't often that I say this when it comes to gambling, but my answer is 'I don't know'" when asked about optimal solitaire strategy, confirming that **no known optimal strategy exists**.

---

## Could a Lookup Table + MCTS Do Better?

- **In theory, yes** — but the full state space is too large to store
- Klondike state space (52 cards across tableau/stock/waste/foundations with hidden/revealed states) is in the billions+
- Lookup tables work for small state spaces (e.g., chess endgame tablebases with ≤7 pieces)
- Potential middle-ground approaches:
  - **Partial lookup tables** for late-game positions where most cards are revealed
  - **Neural network evaluator** trained on millions of games (AlphaZero-style) combined with MCTS
  - Nobody has published results combining these for Klondike specifically
- The bottleneck is lack of research investment — it's a niche problem

---

## 1xbet Platform Concerns

### Regulatory Red Flags

- Operates under a **Curacao eGaming license** — widely regarded as one of the weakest regulatory jurisdictions, with minimal enforcement and no requirement for independent RNG testing
- Founders face international arrest warrants from Russia
- Subsidiary 1xCorp NV declared bankrupt by Dutch Supreme Court (January 2023)
- UK gambling license revoked following a Sunday Times investigation

### Proprietary Games

- Solitaire falls under "1xGames" — proprietary games developed exclusively for the platform
- Not supplied by established third-party providers (Evolution, Betsoft) that undergo independent lab testing
- No mandated independent RNG testing like UK Gambling Commission or Malta Gaming Authority require

### User Complaints

- Trustpilot reviews document extensive complaints about blocked withdrawals, frozen accounts, and perceived game manipulation
- "Simultaneous play on two or more tables is prohibited" — they actively monitor play patterns
- Documented history of freezing accounts of winning players

### Whether the Shuffle Is Fair

Unverifiable from the outside. But crucially, **it doesn't need to be rigged** — standard rules deliver 14–34% house edge naturally. Additional mechanisms that further reduce win rates:
- Limited passes through stock (drops draw-1 from ~33% to ~4.6% with single pass)
- Preventing cards from being moved off foundations
- Time pressure forcing suboptimal play
- Connection instability counting as a loss
- Hidden timer or forced completion deadline

---

## Project Plan

### Phase 1: Fork & Configure Lonelybot

Lonelybot is the current state-of-the-art Klondike solver, written in Rust (MIT license).

**Repository**: https://github.com/vuonghy2442/lonelybot

**Setup:**
```bash
git clone https://github.com/vuonghy2442/lonelybot.git
cd lonelybot
cargo build --release
cd lonecli && cargo run --release -- help
```

**Configuration for 1xbet rules:**
- Draw count: 1
- Passes: unlimited
- King-only empty column fills
- No undo (inherent in imperfect-info mode)

**Two modes to run:**
1. **Thoughtful Mode** — perfect information, gives upper bound on solvability
2. **Random/HOP/MCTS Mode** — imperfect information, realistic AI ceiling

**Seed types available:** default (Rust RNG), legacy, solvitaire, klondike-solver, greenfelt, exact (256-bit permutation)

### Phase 2: Large-Scale Simulation

**Run 2A — Thoughtful Mode (Upper Bound):**
- 100,000+ games, draw-1 unlimited passes
- Expected: ~85–91% solvability
- Establishes the theoretical ceiling

**Run 2B — HOP/MCTS Imperfect Info (THE Critical Number):**
- 10,000–50,000 games, draw-1 unlimited passes
- Expected: ~50–55% (extrapolated from 47.58% on draw-3)
- At ~3 sec/game, 50K games ≈ 42 hours on single core
- If this is >50%, game is theoretically beatable by superhuman AI
- If this is <50%, game is definitively -EV for everyone

**Run 2C — Sensitivity Analysis:**
- Vary pass limits (1, 2, 3, unlimited)
- Test draw-3 as baseline comparison against published results
- Cross-validate with Solvitaire seeds

**Statistical precision:**

| Sample Size | 95% CI at ~50% Win Rate |
|---|---|
| 10,000 games | ± 1.0% |
| 50,000 games | ± 0.45% |
| 100,000 games | ± 0.31% |

**Data collection:** Log every game (seed, outcome, move count, time) as CSV.

### Phase 3: Human-Like Heuristic Player

Lonelybot's MCTS does deep lookahead no human can replicate. A "strong human" baseline is needed.

**Option A — Adapt existing solvers:**
- Daniel Collins' SolitaireSolver (Java, GPL): https://github.com/danielrcollins1/SolitaireSolver
- Reverse-engineer Bill Turnbull's heuristics (52.9% on draw-1 unlimited)

**Option B — Build custom Python engine:**
```
klondike/
├── engine.py          # Core game state, legal move generation
├── strategies/
│   ├── base.py        # Strategy interface
│   ├── greedy.py      # Always play to foundation when possible
│   ├── heuristic.py   # Weighted heuristic rules
│   └── mcts.py        # Simple MCTS for comparison
├── simulator.py       # Run N games, collect stats
├── analyzer.py        # EV calculations, confidence intervals
└── main.py            # CLI entry point
```

**Heuristic priority rules:**
1. Always play Aces and Twos to foundation immediately
2. Prefer moves that reveal face-down cards
3. Prefer moves that free up tableau columns
4. Safe-to-play rule: only send to foundation if both opposite-color cards one rank lower are already there
5. Prefer moving cards from larger piles
6. Among equal moves, prefer increasing face-up card count
7. Draw from stock only when no beneficial tableau moves exist
8. Prefer Kings that expose face-down cards for empty columns
9. Between two Kings, prefer the one with more face-down cards behind it
10. Detect stuck states (cycling stock with no new moves) and give up

**Target:** 100,000+ games, expected 35–45% win rate.

### Phase 4: EV Dashboard

**Inputs:** empirical win rate, bet size, games per hour, bankroll size

**Outputs:**
- EV per game and per hour
- RTP equivalent
- Probability of ruin at various bankroll levels
- Kelly bet sizing (if +EV)
- Comparison vs. slot games (96% RTP baseline)
- Required win rate to justify playing

### Phase 5: Risk Assessment & Additional Investigations

- **Shuffle fairness testing:** Play large number of games, record dealt sequences, run statistical tests (chi-squared, runs test) for non-random patterns
- **Hidden rules:** Test empirically if unlimited passes are real, check for hidden timers, verify disconnection handling
- **Account risk:** 1xbet's documented history of freezing winning accounts
- **Bot detection:** Play patterns, decision speed, win rate anomalies — risk of ban and fund seizure
- **RL/ML approach (longer-term):** Train reinforcement learning agent on Klondike engine (Stable Baselines3, RLlib)

---

## Decision Framework

| MCTS Draw-1 Result | EV Status | Verdict |
|---|---|---|
| < 45% | Negative | Do not play — worse than slots |
| 45–50% | Slightly negative | Marginal — only with extreme volume and discipline |
| 50–52% | Breakeven to slim positive | Theoretically exploitable but platform risk dominates |
| > 52% | Positive | Potentially exploitable IF shuffle is fair AND withdrawals work |

**The single most important number to get is Lonelybot's HOP/MCTS win rate on draw-1 unlimited passes.** If that's below 50%, nothing else matters — no human will beat the AI. If it's above 50%, the question becomes whether a human (potentially solver-assisted) can replicate it.

---

## Key Resources

### Solvers & Engines

| Project | Language | License | URL | Best For |
|---|---|---|---|---|
| Lonelybot | Rust | MIT | https://github.com/vuonghy2442/lonelybot | SOTA solver, Thoughtful + MCTS |
| Solvitaire | C++ | GPL | University of St Andrews | Gold standard Thoughtful solving |
| MinimalKlondike | C++ | MIT | https://github.com/ShootMe/MinimalKlondike | Minimal-length solutions |
| Klondike-Solver | C++ | MIT | https://github.com/ShootMe/Klondike-Solver | Fast solving, configurable draw |
| SolitaireSolver (Collins) | Java | GPL | https://github.com/danielrcollins1/SolitaireSolver | Heuristic play, large-scale sim |
| SolitaireSolver (Mayhew) | Python | — | https://github.com/mayhewsw/SolitaireSolver | Screen-reading + OpenCV |

### Academic Papers

- Blake & Gent, "The Winnability of Klondike Solitaire and Many Other Patience Games" (arXiv:1906.12314v5, JAIR 2026)
- Bjarnason, Fern, Tadepalli, "Searching Solitaire in Real Time" (Oregon State, ICGA 2007)
- Yan, Diaconis, Rusmevichientong, Van Roy, "Solitaire: Man Versus Machine" (Stanford, NIPS 2005)
- Voima, "Klondike Solitaire Solvability" (Bachelor's thesis, 2021)

### Community Data

- Bill Turnbull's Solitaire Tester: 100M game simulation results
- Solitaired.com: aggregate stats across millions of games
- Wizard of Odds: https://wizardofodds.com/games/solitaire/
- MadMath blog: http://www.madmath.com/2009/05/winning-solitaire.html

---

## Execution Priority

1. Clone Lonelybot, build, run draw-1 Thoughtful mode (100K games) — upper bound
2. Run Lonelybot HOP/MCTS on draw-1 (10K–50K games) — **the critical number**
3. If MCTS > 50%: investigate further, build heuristic player, assess platform risk
4. If MCTS < 50%: game is confirmed -EV even for perfect AI play — stop here unless building for learning
5. Build EV dashboard regardless (useful reference)

---

## Bottom Line

The gap between knowing a deal is solvable and actually solving it is the entire story. Approximately 82–91% of Klondike draw-1 deals have a winning path, but finding that path without seeing hidden cards or using undo drops the practical win rate to 33–43% for humans. At a 2x payout, you need to win more than half your games to break even, and no documented human player has achieved this without undo functionality.

The expected value is unambiguously negative for all realistic player skill levels. A skilled player loses roughly $0.14 per dollar wagered over the long run — and that assumes a perfectly fair, random shuffle with no time pressure, no connectivity issues, and no platform-side manipulation. For an average player, the loss rate approaches $0.34 per dollar. This makes Klondike solitaire at 2x one of the worst-EV gambling propositions commonly available, masquerading as a skill game.
