# Phase 2: Search & Evaluation Framework

**Gate**: Gate 1 — Search-Only Ablations
**Parallelizable**: YES — Search (Task A) and Evaluation (Task B) can be developed in parallel
**Estimated sessions**: 2-3
**Dependencies**: Phase 1 complete (PyO3 bindings + Python interface working)

---

## Overview

Build the search framework (PIMC with determinization) and statistical evaluation tools. This phase answers: **"What's the best win rate we can achieve with search alone, no neural nets?"**

This is a crucial baseline — if search alone crosses 50% on draw-1, the project is already viable without ML. If it doesn't, we know exactly how much the neural net needs to add.

**Deliverables:**
1. `klondike/search/pimc.py` — PIMC search with configurable determinizations
2. `klondike/search/mcts.py` — Python MCTS (for later neural net integration)
3. `klondike/eval/stats.py` — Confidence intervals, SPRT, variance reduction
4. `klondike/eval/benchmark.py` — Benchmark runner with paired seeds
5. Search-only ablation results: HOP vs PIMC variants, different budgets
6. Best search-only win rate on draw-1 with proper CI

---

## Team Agent Strategy

### TASK A (Search Agent): PIMC + MCTS
- Implements PIMC search using Lonelybot's HOP solver via PyO3
- Implements Python MCTS framework (will accept neural net plugin in Phase 4)
- Tests different determinization counts and search budgets
- Runs ablation experiments

### TASK B (Evaluation Agent): Statistical Framework
- Implements CI computation, SPRT, variance reduction
- Builds benchmark runner with paired seeds
- Implements result logging and comparison tools
- Can mock search results for testing

### SHARED INTERFACE:

```python
# Search functions return this:
@dataclass
class SearchResult:
    action: int                 # Best action index (0-259)
    visit_counts: np.ndarray    # Visit distribution over actions (size 260)
    value: float                # Estimated win probability (0-1)
    determinizations_used: int
    simulations_per_det: int
    wall_time_ms: float

# Benchmark functions expect this:
SearchFn = Callable[[KlondikeEnv], SearchResult]
```

---

## TASK A: PIMC Search

### `klondike/search/pimc.py`

PIMC (Perfect Information Monte Carlo) with determinization:

```python
import numpy as np
from dataclasses import dataclass
from klondike.env import KlondikeEnv
from klondike.actions import ACTION_SPACE_SIZE

@dataclass
class PIMCConfig:
    n_determinizations: int = 10    # Number of sampled hidden arrangements
    node_limit: int = 1000          # Max nodes per determinized search
    n_hop_iterations: int = 3000    # HOP iterations per determinization

@dataclass
class SearchResult:
    action: int
    visit_counts: np.ndarray    # Shape (ACTION_SPACE_SIZE,)
    value: float
    determinizations_used: int
    simulations_per_det: int
    wall_time_ms: float


def pimc_search(env: KlondikeEnv, config: PIMCConfig) -> SearchResult:
    """
    PIMC search: sample determinized worlds, solve each, aggregate.

    Algorithm:
    1. Get legal moves from current position
    2. For each determinization d = 1..n:
       a. Clone the environment
       b. Shuffle hidden cards (determinize)
       c. For each legal move, run HOP to estimate win rate
       d. Record per-move win rates
    3. Average win rates across determinizations
    4. Return best action and full distribution

    This uses Lonelybot's built-in HOP solver via PyO3.
    """
    legal_actions = env.get_legal_action_list()
    if not legal_actions:
        return SearchResult(
            action=-1,
            visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.float32),
            value=0.0,
            determinizations_used=0,
            simulations_per_det=0,
            wall_time_ms=0.0,
        )

    start_time = time.perf_counter()

    # Accumulate win rates per action across determinizations
    action_wins = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
    action_plays = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)

    for d in range(config.n_determinizations):
        # Clone + determinize
        det_env = env.clone()
        det_env.shuffle_hidden(rng_seed=d * 1000 + int(time.time() * 1000) % 10000)

        # For each legal move, estimate win rate via HOP
        for action in legal_actions:
            move_type, card_idx = action_to_move(action)
            # Use Lonelybot's HOP solver
            hop_result = det_env._game.hop_solve(
                move_type=move_type,
                card_idx=card_idx,
                n_times=config.n_hop_iterations,
                node_limit=config.node_limit,
            )
            if hop_result["played"] > 0:
                action_wins[action] += hop_result["wins"]
                action_plays[action] += hop_result["played"]

    # Compute average win rate per action
    visit_counts = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for action in legal_actions:
        if action_plays[action] > 0:
            visit_counts[action] = action_wins[action] / action_plays[action]

    best_action = legal_actions[np.argmax([visit_counts[a] for a in legal_actions])]
    value = visit_counts[best_action]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SearchResult(
        action=best_action,
        visit_counts=visit_counts,
        value=value,
        determinizations_used=config.n_determinizations,
        simulations_per_det=config.n_hop_iterations,
        wall_time_ms=elapsed_ms,
    )
```

**Note**: The above wraps Lonelybot's HOP solver. If the PyO3 `hop_solve` call is per-move (evaluating one move at a time), we call it for each legal move. If Lonelybot's MCTS solver (`pick_moves`) gives the best move directly, we can use that instead for a simpler interface.

### Alternative: Use Lonelybot's MCTS directly

```python
def lonelybot_mcts_search(env: KlondikeEnv, n_sims: int = 3000, limit: int = 1000) -> SearchResult:
    """
    Use Lonelybot's built-in MCTS via PyO3.
    Simpler than PIMC — let Lonelybot handle the search.
    """
    # This calls pick_moves() which does MCTS internally
    result = env._game.mcts_solve(n_sims=n_sims, limit=limit)
    # Convert result to SearchResult format
    ...
```

### `klondike/search/mcts.py` — Python MCTS (for Phase 4)

A pure-Python MCTS that will accept a neural net evaluation function. This is needed for Phase 4 (AlphaZero) but the skeleton should be built now.

```python
import numpy as np
import math
from dataclasses import dataclass, field
from klondike.env import KlondikeEnv
from klondike.actions import ACTION_SPACE_SIZE

@dataclass
class MCTSConfig:
    n_simulations: int = 400        # Simulations per move
    c_puct: float = 1.5             # Exploration constant
    temperature: float = 1.0        # Action selection temperature
    dirichlet_alpha: float = 0.3    # Root noise for exploration
    dirichlet_frac: float = 0.25    # Fraction of noise at root


class MCTSNode:
    """Single node in the MCTS tree."""

    def __init__(self, prior: float = 0.0):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: dict[int, "MCTSNode"] = {}  # action -> child node

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """Upper Confidence Bound for Trees."""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration


# Evaluation function type — neural net plugs in here
EvalFn = Callable[[KlondikeEnv], tuple[np.ndarray, float]]
# Returns (policy: ndarray[ACTION_SPACE_SIZE], value: float)


def default_eval_fn(env: KlondikeEnv) -> tuple[np.ndarray, float]:
    """Default evaluation: uniform policy, 0.5 value (no neural net)."""
    legal = env.get_legal_actions()
    policy = legal / (legal.sum() + 1e-8)  # Uniform over legal moves
    return policy, 0.5


def mcts_search(
    env: KlondikeEnv,
    config: MCTSConfig,
    eval_fn: EvalFn = default_eval_fn,
) -> SearchResult:
    """
    Run MCTS with a pluggable evaluation function.

    For Phase 2: eval_fn = default_eval_fn (random rollouts)
    For Phase 4: eval_fn = neural_net_eval (learned policy + value)
    """
    root = MCTSNode()
    legal_mask = env.get_legal_actions()

    # Initialize root with eval
    policy, value = eval_fn(env)
    policy = policy * legal_mask
    policy = policy / (policy.sum() + 1e-8)

    # Add Dirichlet noise at root
    noise = np.random.dirichlet([config.dirichlet_alpha] * ACTION_SPACE_SIZE)
    policy = (1 - config.dirichlet_frac) * policy + config.dirichlet_frac * noise
    policy = policy * legal_mask
    policy = policy / (policy.sum() + 1e-8)

    # Create children for legal moves
    for action in range(ACTION_SPACE_SIZE):
        if legal_mask[action] > 0:
            root.children[action] = MCTSNode(prior=policy[action])

    # Run simulations
    for _ in range(config.n_simulations):
        _simulate(env, root, config, eval_fn)

    # Extract visit distribution
    visit_counts = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for action, child in root.children.items():
        visit_counts[action] = child.visit_count

    # Select action based on temperature
    if config.temperature < 0.01:
        best_action = max(root.children, key=lambda a: root.children[a].visit_count)
    else:
        visit_probs = visit_counts ** (1 / config.temperature)
        visit_probs = visit_probs / (visit_probs.sum() + 1e-8)
        best_action = np.random.choice(ACTION_SPACE_SIZE, p=visit_probs)

    return SearchResult(
        action=best_action,
        visit_counts=visit_counts,
        value=root.value,
        determinizations_used=0,
        simulations_per_det=config.n_simulations,
        wall_time_ms=0.0,  # caller should time
    )


def _simulate(env: KlondikeEnv, root: MCTSNode, config: MCTSConfig, eval_fn: EvalFn):
    """
    Single MCTS simulation: select → expand → evaluate → backpropagate.

    NOTE: This clones the env at the start and plays moves down the tree.
    When it reaches a leaf, it evaluates with eval_fn and backprops.
    """
    sim_env = env.clone()
    path = []  # [(node, action)]
    node = root

    # Selection: walk down tree picking best UCB
    while node.children and not sim_env.is_done():
        action = max(
            node.children,
            key=lambda a: node.children[a].ucb_score(node.visit_count, config.c_puct)
        )
        path.append((node, action))
        node = node.children[action]
        move_type, card_idx = action_to_move(action)
        sim_env._game.do_move(move_type, card_idx)

        # If node hasn't been expanded yet, expand it
        if node.visit_count == 0:
            break

    # Expansion + Evaluation
    if sim_env.is_done():
        value = 1.0 if sim_env._game.is_win() else 0.0
    else:
        legal_mask = sim_env.get_legal_actions()
        if legal_mask.sum() == 0:
            value = 0.0  # No legal moves = loss
        else:
            policy, value = eval_fn(sim_env)
            policy = policy * legal_mask
            policy = policy / (policy.sum() + 1e-8)
            for action in range(ACTION_SPACE_SIZE):
                if legal_mask[action] > 0 and action not in node.children:
                    node.children[action] = MCTSNode(prior=policy[action])

    # Backpropagation
    for parent_node, action in reversed(path):
        child = parent_node.children[action]
        child.visit_count += 1
        child.value_sum += value
```

---

## TASK B: Statistical Evaluation Framework

### `klondike/eval/stats.py`

```python
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class WinRateResult:
    wins: int
    total: int
    win_rate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    confidence: float


def compute_ci(wins: int, total: int, confidence: float = 0.95) -> WinRateResult:
    """Compute win rate with confidence interval."""
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    p = wins / total if total > 0 else 0.0
    se = math.sqrt(p * (1 - p) / total) if total > 0 else 0.0
    return WinRateResult(
        wins=wins, total=total, win_rate=p,
        ci_lower=max(0, p - z * se), ci_upper=min(1, p + z * se),
        std_error=se, confidence=confidence,
    )


def sample_size_needed(margin: float, p: float = 0.5, confidence: float = 0.95) -> int:
    """How many games needed for a given CI margin?"""
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)
    return math.ceil((z ** 2) * p * (1 - p) / (margin ** 2))


def sprt_test(
    wins: int,
    total: int,
    p0: float = 0.50,     # Null hypothesis: win rate = 50%
    p1: float = 0.52,     # Alternative: win rate = 52%
    alpha: float = 0.05,  # False positive rate
    beta: float = 0.05,   # False negative rate
) -> str:
    """
    Sequential Probability Ratio Test.

    Returns: "accept_h0" | "accept_h1" | "continue"

    Use this for thin-margin testing around 50%. Instead of running a
    fixed N games, SPRT adaptively stops when there's enough evidence.
    """
    if total == 0:
        return "continue"

    # Log-likelihood ratio
    losses = total - wins
    if p0 == 0 or p0 == 1 or p1 == 0 or p1 == 1:
        return "continue"

    log_lr = (
        wins * math.log(p1 / p0) +
        losses * math.log((1 - p1) / (1 - p0))
    )

    # Wald boundaries
    log_A = math.log((1 - beta) / alpha)    # Upper boundary (accept H1)
    log_B = math.log(beta / (1 - alpha))    # Lower boundary (accept H0)

    if log_lr >= log_A:
        return "accept_h1"
    elif log_lr <= log_B:
        return "accept_h0"
    else:
        return "continue"


def paired_comparison(
    results_a: list[bool],
    results_b: list[bool],
) -> dict:
    """
    Compare two solvers on the same deals (paired seeds).

    Uses McNemar's test for paired binary data.
    results_a[i] and results_b[i] are the outcomes on the same deal i.
    """
    assert len(results_a) == len(results_b)
    n = len(results_a)

    # Contingency table
    both_win = sum(a and b for a, b in zip(results_a, results_b))
    a_only = sum(a and not b for a, b in zip(results_a, results_b))
    b_only = sum(not a and b for a, b in zip(results_a, results_b))
    both_lose = sum(not a and not b for a, b in zip(results_a, results_b))

    # McNemar's statistic
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
        chi2 = 0.0
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
        # Approximate p-value from chi2 distribution (1 df)
        # For quick computation, use normal approximation
        z = math.sqrt(chi2)
        p_value = 2 * (1 - _normal_cdf(z))

    return {
        "n_deals": n,
        "a_wins": sum(results_a),
        "b_wins": sum(results_b),
        "both_win": both_win,
        "a_only": a_only,
        "b_only": b_only,
        "both_lose": both_lose,
        "chi2": chi2,
        "p_value": p_value,
        "a_better": a_only > b_only,
        "diff": (sum(results_a) - sum(results_b)) / n,
    }


def _normal_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# --- Variance Reduction ---

def control_variate_estimate(
    target_results: np.ndarray,      # 0/1 array for target solver
    control_results: np.ndarray,     # 0/1 array for control solver (same deals)
    control_known_rate: float,       # Known win rate of control from large prior run
) -> dict:
    """
    Use control variate to get tighter estimate of target solver's win rate.
    Both solvers must be run on the same deals.
    """
    cov = np.cov(target_results, control_results)[0, 1]
    var_control = np.var(control_results)
    beta = cov / var_control if var_control > 0 else 0

    raw_rate = np.mean(target_results)
    control_sim_rate = np.mean(control_results)
    cv_rate = raw_rate - beta * (control_sim_rate - control_known_rate)

    residuals = target_results - beta * (control_results - control_known_rate)
    se_cv = np.std(residuals) / np.sqrt(len(target_results))
    se_raw = np.std(target_results) / np.sqrt(len(target_results))

    return {
        "rate_cv": float(cv_rate),
        "rate_raw": float(raw_rate),
        "se_cv": float(se_cv),
        "se_raw": float(se_raw),
        "variance_reduction": float((se_raw / se_cv) ** 2) if se_cv > 0 else 0,
        "beta": float(beta),
    }
```

### `klondike/eval/benchmark.py`

```python
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from klondike.env import KlondikeEnv
from klondike.eval.stats import compute_ci, sprt_test

@dataclass
class BenchmarkConfig:
    n_games: int = 10000
    seed_start: int = 0
    draw_step: int = 1
    log_every: int = 100
    output_dir: str = "logs"

@dataclass
class GameRecord:
    seed: int
    won: bool
    n_moves: int
    wall_time_ms: float
    search_time_ms: float


def run_benchmark(
    search_fn,            # Callable[[KlondikeEnv], SearchResult]
    config: BenchmarkConfig,
    name: str = "unnamed",
) -> dict:
    """
    Run a benchmark: play N games using the given search function.
    Returns statistics and saves detailed logs.
    """
    records = []
    wins = 0

    output_path = Path(config.output_dir) / f"{name}_{config.seed_start}_{config.n_games}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = KlondikeEnv(draw_step=config.draw_step)

    for i in range(config.n_games):
        seed = config.seed_start + i
        game_start = time.perf_counter()
        search_time_total = 0.0

        obs = env.reset(seed=seed)
        n_moves = 0

        while not env.is_done():
            legal = env.get_legal_action_list()
            if not legal:
                break

            search_start = time.perf_counter()
            result = search_fn(env)
            search_time_total += time.perf_counter() - search_start

            if result.action < 0:
                break

            obs, reward, done, info = env.step(result.action)
            n_moves += 1

            if n_moves > 2000:  # Safety: prevent infinite loops
                break

        won = env._game.is_win()
        if won:
            wins += 1

        elapsed = (time.perf_counter() - game_start) * 1000
        record = GameRecord(seed, won, n_moves, elapsed, search_time_total * 1000)
        records.append(record)

        # Log to file
        with open(output_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        # Progress
        if (i + 1) % config.log_every == 0:
            ci = compute_ci(wins, i + 1)
            print(f"[{name}] {i+1}/{config.n_games}: "
                  f"win_rate={ci.win_rate:.4f} "
                  f"CI=({ci.ci_lower:.4f}, {ci.ci_upper:.4f})")

    # Final results
    ci = compute_ci(wins, config.n_games)
    return {
        "name": name,
        "n_games": config.n_games,
        "wins": wins,
        "win_rate": ci.win_rate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
        "std_error": ci.std_error,
        "log_file": str(output_path),
    }
```

---

## Search-Only Ablation Experiments

Once the framework is built, run these experiments on draw-1:

### Experiment 1: Lonelybot HOP baseline
```python
# Just use Lonelybot's built-in HOP solver with default settings
# This should match Phase 0's draw-1 result
config = BenchmarkConfig(n_games=10000, draw_step=1)
result = run_benchmark(lonelybot_hop_search, config, name="hop_default")
```

### Experiment 2: Vary determinization count
```python
for n_det in [1, 3, 5, 10, 20, 50]:
    pimc_config = PIMCConfig(n_determinizations=n_det)
    search_fn = lambda env: pimc_search(env, pimc_config)
    result = run_benchmark(search_fn, config, name=f"pimc_det{n_det}")
```

### Experiment 3: Vary search budget
```python
for n_iter in [100, 500, 1000, 3000, 5000]:
    pimc_config = PIMCConfig(n_hop_iterations=n_iter)
    search_fn = lambda env: pimc_search(env, pimc_config)
    result = run_benchmark(search_fn, config, name=f"pimc_iter{n_iter}")
```

### Experiment 4: MCTS with uniform policy (no neural net)
```python
for n_sims in [100, 200, 400, 800]:
    mcts_config = MCTSConfig(n_simulations=n_sims, temperature=0.0)
    search_fn = lambda env: mcts_search(env, mcts_config, eval_fn=default_eval_fn)
    result = run_benchmark(search_fn, config, name=f"mcts_uniform_{n_sims}")
```

Record ALL results in a comparison table.

---

## Acceptance Criteria

- [ ] PIMC search runs correctly on draw-1 games
- [ ] Python MCTS skeleton works with default (uniform) eval function
- [ ] Statistical tools compute correct CIs (verify with known values)
- [ ] SPRT correctly identifies when to stop testing
- [ ] Paired comparison works on matched seed results
- [ ] Benchmark runner logs results to JSONL files
- [ ] At least one ablation experiment completed with results recorded
- [ ] Best search-only win rate on draw-1 recorded with CI

---

## Decision Gate

| Best Search-Only Result | Action |
|------------------------|--------|
| **CI lower bound > 50%** | Search alone is sufficient! Neural net is bonus optimization. |
| **CI includes 50%** | Promising. Neural net guidance likely pushes it over. Proceed. |
| **CI upper bound 48-50%** | Neural net needs to add 0-2%. Achievable. Proceed cautiously. |
| **CI upper bound < 48%** | Major gap to close. Reassess feasibility — may need different search approach. |

---

## Handoff to Phase 3

Phase 3 needs:
1. Working search framework (`pimc_search`, `mcts_search`)
2. Working evaluation framework (`compute_ci`, `sprt_test`, `run_benchmark`)
3. Best search-only win rate and CI on draw-1
4. MCTS skeleton ready to accept neural net `eval_fn`
5. Ablation results showing optimal search parameters
6. Benchmark logs for comparison with future neural net results
