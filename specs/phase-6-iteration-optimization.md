# Phase 6: Iteration, Optimization & Hardening

**Gate**: Post-Gate 3 — You have a trained model. Now make it better.
**Parallelizable**: YES — many experiments can run independently
**Estimated sessions**: Ongoing (this is where most calendar time goes)
**Dependencies**: Phase 4 complete (at least one full AlphaZero training run done)

---

## Why This Phase Exists

Phases 0-4 get you a **first working model**. But the first model is almost never the final model. This phase is the systematic improvement loop — the difference between "it kind of works" and "it reliably beats 50%."

This is NOT a single coding session. It's a structured checklist of experiments and optimizations to try, roughly ordered by expected impact.

---

## What's Already Done vs What's Left

| Activity | Covered In | Status After Phase 4 |
|----------|-----------|---------------------|
| Build Lonelybot, get baselines | Phase 0 | Done |
| PyO3 bindings + state encoding | Phase 1 | Done |
| PIMC search + evaluation stats | Phase 2 | Done |
| Supervised pre-training | Phase 3 | Done |
| AlphaZero self-play (first run) | Phase 4 | Done |
| CV + deployment | Phase 5 | Done (if applicable) |
| **Hyperparameter tuning** | **Phase 6** | Starts here |
| **Architecture experiments** | **Phase 6** | Starts here |
| **Search improvements** | **Phase 6** | Starts here |
| **Encoding improvements** | **Phase 6** | Starts here |
| **Scaling up training** | **Phase 6** | Starts here |
| **Robustness testing** | **Phase 6** | Starts here |

---

## Tier 1: Quick Wins (Days)

These are the first things to try after Phase 4. Each is a focused experiment that can be done in 1-2 sessions.

### 1A. Hyperparameter Sweep

Run a small grid search over the most impactful parameters:

```
Parameters to sweep:
- MCTS simulations/move: [100, 200, 400, 800]
- PIMC determinizations (training): [3, 5, 10]
- Learning rate: [3e-4, 1e-3, 3e-3]
- Replay buffer size: [50K, 100K, 200K, 500K]
- Games per iteration: [100, 200, 500]
- Temperature schedule: [15, 20, 30 moves at τ=1]
```

**Method**: For each config, train for 20 iterations (not full 100), evaluate on 1000 fixed seeds. Pick the best, then do a full run.

**Expected impact**: 1-3% win rate improvement from better hyperparams alone.

### 1B. Evaluation with More Determinizations

Phase 4's eval uses greedy policy (no search). Try evaluating with PIMC:

```
Evaluation modes to compare:
1. Greedy policy (no search) — fastest
2. MCTS (200 sims, no PIMC) — medium
3. PC-PIMC (5 det × 100 sims) — slow but strongest for training
4. PC-PIMC (50 det × 200 sims) — slowest, best estimate of true strength
```

**Key insight**: The model's *playing strength with search* may be much higher than its *greedy strength*. If greedy is 48% but PIMC-guided is 52%, the model IS good enough — you just need search at deployment time.

### 1C. Extended Training

If Phase 4 ran 100 iterations, try 200-500. Check if win rate is still climbing.

Plot: win rate vs training iteration. If the curve is still going up, you just need more training. If it's plateaued, move to Tier 2.

---

## Tier 2: Architecture & Encoding Experiments (1-2 Weeks Each)

### 2A. State Encoding Improvements

The baseline encoding (~705 dims) has known gaps. Try adding:

```
Additions to test (one at a time, measure impact):
1. Stock-pass context: which pass through the stock are we on? (1 feature)
2. Action history: last 5 actions as one-hot (5 × 260 = 1300 dims — may be too many)
   Alternative: last 5 actions as (move_type, card_idx) pairs (5 × 7 = 35 dims)
3. Move-cycle features: has the current state been visited before? (1 bit)
4. Deck position: exact position in deck cycling (1 feature, normalized)
5. Per-pile: top card rank/suit as features (7 × 2 = 14 dims)
6. Cards-playable-from-waste count (1 feature)
```

**Method**: For each addition, retrain from supervised checkpoint (Phase 3), compare eval results. Only keep additions that measurably improve win rate.

### 2B. Network Architecture Variants

```
Architectures to try:
1. Deeper: 6 or 8 residual blocks (from 4) — more capacity
2. Wider: 768 or 1024 hidden (from 512) — more capacity
3. Smaller: 2 blocks × 256 hidden — faster, maybe enough for Klondike
4. Transformer: 52 card embeddings + 4 attention layers
   - Each card = 16-dim embedding with position/location features
   - Self-attention captures card-card relationships
   - May learn "this card blocks that card" patterns better than MLP
5. Set network: treat hand as unordered set of card features
```

**Method**: For each, train supervised (Phase 3 is fast) and compare policy accuracy. Only retrain with AlphaZero for the top 1-2 candidates.

### 2C. ISMCTS vs HOP

HOP (Hindsight Optimization) has known optimistic bias. ISMCTS (Information Set MCTS) builds a single tree over information sets instead.

```
Implementation:
- ISMCTS: don't sample-then-search.
  Instead, at each node, sample a determinization ONLY when needed.
  One tree, many implicit worlds.
- Advantage: avoids strategy fusion, less optimistic bias
- Disadvantage: more complex implementation, slower per simulation

Test: Compare HOP vs ISMCTS at same computation budget.
```

**Expected impact**: 0-2% — depends on how much optimistic bias matters for Klondike.

### 2D. Auxiliary Targets Ablation

Phase 4 uses auxiliary predictions (foundation count, face-down count). Measure their impact:

```
Train three models:
1. No auxiliary targets (policy + value only)
2. With auxiliary targets (current)
3. With MORE auxiliary targets:
   - Predict which pile will next reveal a card
   - Predict number of moves until next foundation placement
   - Predict if current deal is solvable
```

---

## Tier 3: Advanced Techniques (Weeks)

### 3A. Batched Neural Net Inference in MCTS

Currently, MCTS evaluates one leaf at a time (1 GPU call per leaf). Batching multiple pending leaf evaluations into a single GPU call can speed up self-play 5-10x.

```
Implementation approach:
1. Modified MCTS that yields pending leaf evaluations
2. Collect N pending leaves across M parallel games
3. Batch evaluate on GPU in one forward pass
4. Distribute results back to the correct MCTS trees
5. This is the same pattern used by AlphaGo/KataGo
```

**Impact**: 5-10x self-play throughput → can run 5-10x more training games.

### 3B. Multiprocess Self-Play

Run multiple self-play workers on separate CPU cores, sharing one GPU for inference.

```
Architecture:
- 8 self-play workers (one per CPU core)
- 1 GPU inference server (receives batched leaf eval requests)
- 1 training process (samples from buffer, updates model)
- Workers and GPU communicate via shared memory or queues
```

**Impact**: Linear scaling with CPU cores. 8 cores = 8x throughput.

### 3C. Stochastic Weight Averaging (SWA)

```python
# Save model weights every 250K training positions
# Average the last 4 snapshots
# Use averaged model for evaluation and next round of self-play
# KataGo reports consistent improvement from this
```

### 3D. Playout Cap Randomization (KataGo)

```
Most moves: fast search (100 simulations)
Occasional moves (20% probability): full search (400 simulations)
Only full-search moves contribute to policy training targets
Value head trains on ALL moves

Why: resolves tension between policy targets (need precise search)
and value targets (need many positions from varied play)
```

---

## Tier 4: Alternative Approaches (If Plateaued)

If Tiers 1-3 don't get past 50%, consider these bigger pivots:

### 4A. MuZero

The deferred option from Session 1. Learns a latent world model instead of explicit determinization.

**When to try**: Only if AlphaZe** plateaus below 49% after Tiers 1-3.
**Cost**: 2-4 additional weeks of implementation + training.

### 4B. Direct RL (No MCTS)

Skip MCTS entirely. Train a policy network with pure RL (PPO, SAC).

**Pros**: Much simpler, faster per game, no search overhead.
**Cons**: Likely weaker — MCTS provides significant strength.
**When to try**: If MCTS is too slow for practical deployment.

### 4C. Ensemble Models

Train multiple models with different seeds/architectures. At decision time, average their policies.

**Expected impact**: 0.5-1% from ensemble diversity.

### 4D. Opening Book

Pre-compute optimal first 5-10 moves for common opening patterns.
Many deals share similar early structures — cache the search result.

---

## Ongoing: Evaluation Discipline

Throughout Phase 6, maintain rigorous evaluation:

### Fixed Test Sets
```
- test_set_A: seeds 1,000,000 - 1,009,999 (10K games, primary eval)
- test_set_B: seeds 2,000,000 - 2,009,999 (10K games, confirmation)
- test_set_C: seeds 3,000,000 - 3,004,999 (5K games, tuning)

Rules:
- NEVER tune hyperparameters on test_set_A or test_set_B
- Use test_set_C for hyperparameter tuning
- Final claims use test_set_A only
- Confirm on test_set_B
```

### Tracking Spreadsheet

For every experiment, record:

| Experiment | Config Change | Test Set | N Games | Win Rate | CI | vs Baseline |
|------------|--------------|----------|---------|----------|-----|-------------|
| baseline_v1 | Phase 4 default | C | 5000 | 47.2% | ±1.4% | — |
| lr_3e-4 | Lower LR | C | 5000 | 48.1% | ±1.4% | +0.9% |
| 8_blocks | Deeper net | C | 5000 | 47.5% | ±1.4% | +0.3% |

### When to Stop

**Success**: CI lower bound > 50% on test_set_A, confirmed on test_set_B.

**Failure criteria** (stop investing):
- 3+ months of Tier 2-3 experiments with no improvement trend
- Best result CI upper bound < 49% after full compute budget
- Theoretical analysis shows the gap can't be closed with available compute

**Pivot criteria** (change approach):
- Plateau for 2+ weeks at same win rate despite varied experiments
- Evidence that optimistic bias accounts for >2% of measured win rate

---

## Timeline Estimate

| Tier | Calendar Time | Compute Cost |
|------|--------------|--------------|
| Tier 1 (quick wins) | 1-2 weeks | Low (reuse existing infra) |
| Tier 2 (architecture) | 2-4 weeks | Medium (multiple training runs) |
| Tier 3 (scaling) | 2-4 weeks | High (engineering + long runs) |
| Tier 4 (pivots) | 2-4 weeks each | High (new implementations) |
| **Total realistic** | **2-3 months** | **~500-2000 GPU-hours on 4060** |

This aligns with the project timeline estimate: 6-12 weeks first credible result, 3-6 months robust claim.

---

## Decision Framework

```
After each experiment:
  1. Did win rate improve? (compare CI, use paired seeds)
  2. Is improvement statistically significant? (SPRT or CI non-overlap)
  3. Is the improvement worth the compute cost?
  4. Update the tracking spreadsheet
  5. Pick next experiment from highest-tier unexplored option
```
