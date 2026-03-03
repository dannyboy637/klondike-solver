# Quant Simulation Techniques — Applicable to Klondike Project

Extracted from "How to Simulate Like a Quant Desk" (gemchanger, Feb 2026).
Source: `/Users/Daniel/Documents/Obsidian/4-Knowledge/How to Simulate Like a Quant Desk.md`

---

## 1. Monte Carlo Confidence Intervals

Our win rate lives near p=0.50, which is the **worst case for MC convergence** (variance is maximized).

### Formula
```
Variance: Var(p_hat) = p(1-p) / N
Standard Error: SE = sqrt(p(1-p) / N)
95% CI: p_hat ± 1.96 * SE
```

### Sample Size Requirements (at p=0.50)

| Precision (±) | Confidence | Required N |
|---|---|---|
| ±1.0% | 95% | 9,604 |
| ±0.5% | 95% | 38,416 |
| ±0.45% | 95% | ~47,500 |
| ±0.31% | 95% | ~100,000 |

### Runnable Code
```python
import numpy as np

def compute_ci(wins, total_games, confidence=0.95):
    """Compute win rate confidence interval from simulation results."""
    p_hat = wins / total_games
    se = np.sqrt(p_hat * (1 - p_hat) / total_games)
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    return {
        'win_rate': p_hat,
        'std_error': se,
        'ci_lower': p_hat - z * se,
        'ci_upper': p_hat + z * se,
        'n_games': total_games
    }

# Example: 5,300 wins out of 10,000 MCTS games
result = compute_ci(5300, 10000)
print(f"Win rate: {result['win_rate']:.4f}")
print(f"95% CI: ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})")
# → Win rate: 0.5300, 95% CI: (0.5202, 0.5398)
```

**Key insight**: At 50K games with p≈0.50, CI is ±0.45%. This is tight enough to determine whether the win rate is above or below 50% — the critical threshold.

---

## 2. Brier Score — Model Calibration Metric

Use this to evaluate the neural net's predicted win probabilities against actual outcomes.

### Formula
```
BS = (1/N) * sum( (predicted_i - outcome_i)^2 )
```

### Benchmarks
- Below 0.20: good
- Below 0.10: excellent
- Best election forecasters: 0.06–0.12
- Always-uncertain baseline (predict 0.50): BS = 0.25

### Runnable Code
```python
import numpy as np

def brier_score(predictions, outcomes):
    """
    Evaluate model calibration.

    predictions: array of predicted win probabilities (0.0 to 1.0)
    outcomes: array of actual outcomes (0 = loss, 1 = win)
    """
    return np.mean((np.array(predictions) - np.array(outcomes))**2)

# Compare models
# Model A: neural net predicts P(win) for each game state
# Model B: always predicts 0.50 (no information baseline)
# Model C: heuristic-based estimate

model_nn_preds = [0.72, 0.31, 0.85, 0.15, 0.60]
model_baseline = [0.50, 0.50, 0.50, 0.50, 0.50]
actual =         [1,    0,    1,    0,    1   ]

print(f"Neural net Brier:  {brier_score(model_nn_preds, actual):.4f}")  # lower = better
print(f"Baseline Brier:    {brier_score(model_baseline, actual):.4f}")  # 0.25
```

### Application to Klondike
- After training the neural net, for each game state predict P(win from here)
- Compare against actual game outcome
- Track Brier score across training iterations — should decrease
- Compare Brier score of: raw heuristic vs MCTS-guided vs neural net

---

## 3. Variance Reduction Techniques

These three techniques **stack multiplicatively** and can reduce required sample sizes by 3-5x.

### 3A. Antithetic Variates (Free Symmetry)

For each random deal, also test its "mirror" arrangement. When the outcome function is monotone (which win/loss is), variance reduction is guaranteed.

```
Var_AV = sigma^2 * (1 + rho) / (2n)
where rho = Corr(h(U), h(1-U))
```

Typical reduction: **50–75%**. Zero extra computational cost.

```python
import numpy as np

def antithetic_win_rate(play_game_fn, n_pairs=25000):
    """
    Run n_pairs of games using antithetic variates.
    play_game_fn(seed, antithetic=False) -> 0 or 1

    For each random seed, play the game normally AND with
    the "mirrored" random choices. Average both results.
    """
    results = []
    for seed in range(n_pairs):
        # Original game
        win_normal = play_game_fn(seed, antithetic=False)
        # Antithetic game (mirror the random decisions)
        win_mirror = play_game_fn(seed, antithetic=True)
        # Average the pair
        results.append((win_normal + win_mirror) / 2.0)

    p_hat = np.mean(results)
    se = np.std(results) / np.sqrt(len(results))
    return p_hat, se
```

**Klondike application**: When sampling hidden card arrangements for PIMC, for each sampled arrangement also test the "complementary" arrangement. If cards A,B,C are placed in positions 1,2,3 in the original sample, place them in positions 3,2,1 in the antithetic sample.

### 3B. Stratified Sampling (Divide and Conquer)

Partition deals by some difficulty measure, sample within each stratum.

```python
import numpy as np

def stratified_win_rate(play_game_fn, n_total=50000, n_strata=10):
    """
    Stratified sampling: partition games by difficulty,
    sample within each stratum for tighter CIs.

    Difficulty proxy: number of initially accessible aces,
    or average depth of aces in the tableau.
    """
    n_per_stratum = n_total // n_strata
    stratum_means = []

    for stratum in range(n_strata):
        # Generate deals that fall into this difficulty bucket
        wins = 0
        for _ in range(n_per_stratum):
            # Deal a game, check which stratum it falls in
            # Only play games in the current stratum
            wins += play_game_fn(stratum_filter=stratum)
        stratum_means.append(wins / n_per_stratum)

    # Each stratum weighted equally (uniform allocation)
    p_hat = np.mean(stratum_means)
    se = np.std(stratum_means) / np.sqrt(n_strata)
    return p_hat, se
```

**Klondike application**: Stratify deals by:
- Number of aces in the initial tableau (0, 1, 2, 3, 4)
- Number of face-up cards that can immediately move
- Stock composition (how many useful cards are near the top)

### 3C. Control Variates (Exploit What You Know)

Use the heuristic solver's win rate as a known baseline. Measure only the *difference* between MCTS and heuristic — much lower variance than measuring each independently.

```python
import numpy as np

def control_variate_comparison(mcts_results, heuristic_results, heuristic_known_rate):
    """
    Compare MCTS vs heuristic using control variates.

    mcts_results: array of 0/1 per game (same deals as heuristic)
    heuristic_results: array of 0/1 per game (same deals as mcts)
    heuristic_known_rate: known overall win rate of heuristic (from large prior run)

    CRITICAL: Both solvers must play the SAME deals for this to work.
    """
    # Estimate beta (optimal control coefficient)
    cov = np.cov(mcts_results, heuristic_results)[0, 1]
    var_h = np.var(heuristic_results)
    beta = cov / var_h if var_h > 0 else 0

    # Control variate estimator
    p_mcts_raw = np.mean(mcts_results)
    p_heuristic_sim = np.mean(heuristic_results)

    p_mcts_cv = p_mcts_raw - beta * (p_heuristic_sim - heuristic_known_rate)

    # Adjusted standard error
    residuals = mcts_results - beta * (heuristic_results - heuristic_known_rate)
    se_cv = np.std(residuals) / np.sqrt(len(mcts_results))

    return {
        'mcts_rate_cv': p_mcts_cv,
        'mcts_rate_raw': p_mcts_raw,
        'se_cv': se_cv,
        'se_raw': np.std(mcts_results) / np.sqrt(len(mcts_results)),
        'variance_reduction': (np.std(mcts_results) / np.std(residuals))**2
    }
```

**Klondike application**:
1. Run heuristic solver on 1M games (fast, ~minutes) to get precise baseline rate
2. Run MCTS on 10K of the SAME deals
3. Also run heuristic on those same 10K deals
4. Use control variate to get a much tighter estimate of the MCTS rate
5. Typical variance reduction: 2-10x depending on correlation between solvers

### Stacking All Three
Antithetic variates inside each stratum, with control variate correction:
- Typical combined reduction: **100-500x** over crude MC
- 10K games with all three ≈ precision of 500K-5M crude games
- This is the difference between "run overnight" and "run in 20 minutes"

---

## 4. Particle Filter — Belief State Tracking (Future Use)

Relevant when building the neural net's state representation for imperfect information.

### The Connection to Klondike
In Klondike, hidden cards create a **filtering problem**:
- **Hidden state**: the actual arrangement of face-down cards
- **Observations**: each card you flip reveals new information
- **Belief update**: after seeing a card, update probabilities of where remaining unknown cards are

A particle filter maintains N "particles" — each one a hypothesis about the hidden card arrangement — and reweights them as you observe new cards.

### Algorithm
```
1. INITIALIZE: Sample N possible hidden card arrangements
               Weight each equally: w_i = 1/N

2. FOR each new card revealed:
   a. REWEIGHT: Increase weight of arrangements consistent with observation
                Zero out arrangements that are impossible given what we see
   b. NORMALIZE: w_i = w_i / sum(all weights)
   c. RESAMPLE if effective sample size drops below N/2
```

### Runnable Code (Adapted for Card Games)
```python
import numpy as np
from scipy.special import expit, logit

class CardBeliefTracker:
    """
    Particle filter for tracking belief state over hidden cards.

    Each particle is a possible arrangement of the unknown cards.
    As cards are revealed, impossible arrangements are eliminated
    and the belief concentrates on consistent arrangements.
    """
    def __init__(self, n_particles=5000):
        self.n_particles = n_particles
        self.particles = []  # list of possible hidden card arrangements
        self.weights = np.ones(n_particles) / n_particles

    def initialize(self, known_cards, unknown_positions):
        """
        Generate initial particles: random permutations of unknown
        cards across unknown positions.
        """
        unknown_cards = [c for c in range(52) if c not in known_cards]
        for _ in range(self.n_particles):
            perm = np.random.permutation(unknown_cards)
            self.particles.append(dict(zip(unknown_positions, perm)))

    def observe(self, position, card):
        """
        A card at 'position' was revealed to be 'card'.
        Reweight particles: zero out any that had a different card there.
        """
        for i, particle in enumerate(self.particles):
            if position in particle and particle[position] != card:
                self.weights[i] = 0.0

        # Normalize
        total = self.weights.sum()
        if total > 0:
            self.weights /= total

        # Resample if needed
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.n_particles / 2:
            self._resample()

    def sample_world(self):
        """Sample one possible world (hidden card arrangement) from belief."""
        idx = np.random.choice(self.n_particles, p=self.weights)
        return self.particles[idx]

    def _resample(self):
        """Systematic resampling."""
        cumsum = np.cumsum(self.weights)
        u = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        indices = np.searchsorted(cumsum, u)
        self.particles = [self.particles[i].copy() for i in indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
```

**When to use this**: When building the PIMC solver. Instead of sampling hidden card arrangements uniformly (what Lonelybot does), use the particle filter to sample arrangements that are **consistent with everything you've observed so far**. This is more principled and should yield better move decisions as the game progresses and more information is revealed.

---

## Summary: What to Use Where

| Technique | Phase | Purpose |
|---|---|---|
| MC confidence intervals | Phase 2 (simulations) | Validate sample sizes, report results rigorously |
| Brier score | Phase 3-4 (model training) | Evaluate neural net calibration |
| Antithetic variates | Phase 2-3 (simulations) | 50-75% variance reduction for free |
| Stratified sampling | Phase 2-3 (simulations) | Tighter CIs by partitioning by deal difficulty |
| Control variates | Phase 2-3 (comparisons) | Precise MCTS vs heuristic comparison with less compute |
| Particle filter | Phase 3+ (belief state) | Principled hidden card tracking for PIMC |
