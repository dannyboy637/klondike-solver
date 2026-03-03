# Klondike Solitaire — Discussion Notes

Notes from initial discussion reviewing the klondike-analysis.md document.

---

## Perfect Information Solving

- If a deal is solvable and you have perfect information (all cards visible), a solver will find the winning path **100% of the time**
- It's a finite search problem — the solution exists in the game tree, you just need to find it
- The ~9-18% of unsolvable deals are mathematically impossible regardless of computing power

## Why the Draw-1 Solvability Range (82-91%) Is So Wide

- **Not a statistical noise issue** — it's because no large-scale rigorous study has been done for draw-1 specifically
- Draw-3 is pinned down tightly: **81.945% ± 0.084%** (Solvitaire, ~1M deals, 2.4 years CPU time)
- The 82% floor comes from draw-3 (draw-1 is strictly easier since every stock card is individually accessible)
- The ~90.5% ceiling is a community estimate, not a rigorous result
- Total possible Klondike deals: ~8 x 10^67 — impossible to enumerate them all, must sample
- Proving a single deal is *unsolvable* requires exhaustively searching the entire game tree

## The ~53% AI Win Rate (Imperfect Information)

- Does **not** require a supercomputer
- **Turnbull's heuristic solver** (52.9-53.4%): hand-crafted rule priority list, runs on a laptop, 10M games tested
- **Lonelybot MCTS** (~47.6% draw-3, est. ~52-53% draw-1): Monte Carlo Tree Search, heavier but still desktop-grade
- Neither uses pre-computed lookup tables or databases of solved deals — each game is played fresh
- The AI advantage over humans isn't raw compute — it's consistency, perfect memory, and never making emotional/fatigue errors

## Could a Lookup Table + MCTS Do Better?

- **In theory, yes** — but the full state space is too large to store
- Klondike state space (52 cards across tableau/stock/waste/foundations with hidden/revealed states) is in the billions+
- Lookup tables work for small state spaces (e.g., chess endgame tablebases with <=7 pieces)
- Potential middle-ground approaches:
  - **Partial lookup tables** for late-game positions where most cards are revealed
  - **Neural network evaluator** trained on millions of games (AlphaZero-style) combined with MCTS
  - Nobody has published results combining these for Klondike specifically
- The bottleneck is lack of research investment — it's a niche problem

## Potential Project Directions

- Build a Klondike simulation engine to deal and test millions of hands
- Implement a heuristic solver and measure win rates
- Implement MCTS-based solver for imperfect information play
- Compare strategies head-to-head
- Explore hybrid approaches (MCTS + neural net evaluator, partial endgame tables)
