# Phase 4: AlphaZero Self-Play (AlphaZe** PC-PIMC)

**Gate**: Gate 3 — Full AlphaZero Self-Play
**Parallelizable**: Sequential within phase, but self-play and training overlap via async pipeline
**Estimated sessions**: 3-4
**Dependencies**: Phase 3 complete (supervised net trained, MCTS with neural eval working)

---

## Overview

This is the main event. Implement the full AlphaZero self-play pipeline adapted for imperfect information via PC-PIMC (Policy Combining Perfect Information Monte Carlo).

**Key difference from standard AlphaZero**: At each decision, we sample multiple determinized worlds (hidden card arrangements), run MCTS on each, and average the resulting policies. The neural net sees only the visible game state — NOT the sampled hidden cards.

**Deliverables:**
1. `klondike/alphazero/selfplay.py` — Self-play game generation with PC-PIMC
2. `klondike/alphazero/buffer.py` — Replay buffer for training data
3. `klondike/alphazero/pipeline.py` — Training pipeline (self-play → data → train cycle)
4. Trained AlphaZero model
5. Win rate evaluation with proper CI on draw-1

---

## Architecture Recap

```
PC-PIMC Algorithm (at each move during self-play):

1. Observe visible state s (what player can see)
2. For d = 1..D (D = 3-5 during training):
   a. Clone game state
   b. Randomly reassign hidden cards (determinize)
   c. Run MCTS guided by neural net on determinized state
   d. Extract visit distribution π_d from MCTS root
3. Average: π = (1/D) × Σ π_d
4. Select action: a ~ π (with temperature)
5. Record training sample: (encode(s), π, legal_mask)
6. Execute action a in actual game
7. After game ends: label all positions with outcome z ∈ {0, 1}

Training:
- Loss = CrossEntropy(model_policy, π) + MSE(model_value, z)
- Replay buffer: store last 100K-500K positions
- Mini-batch SGD on randomly sampled positions
```

---

## Implementation

### `klondike/alphazero/buffer.py` — Replay Buffer

```python
import numpy as np
from collections import deque
from dataclasses import dataclass
from klondike.encoding import OBSERVATION_DIM
from klondike.actions import ACTION_SPACE_SIZE


@dataclass
class TrainingSample:
    observation: np.ndarray     # (OBSERVATION_DIM,)
    policy: np.ndarray          # (ACTION_SPACE_SIZE,) — MCTS visit distribution
    value: float                # Game outcome: 1.0 win, 0.0 loss
    legal_mask: np.ndarray      # (ACTION_SPACE_SIZE,) — binary


class ReplayBuffer:
    """
    Fixed-size replay buffer for AlphaZero training data.

    Stores recent self-play positions. Old positions are evicted FIFO.
    No gating needed (single-player game — no catastrophic forgetting risk).
    """

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.observations = np.zeros((capacity, OBSERVATION_DIM), dtype=np.float32)
        self.policies = np.zeros((capacity, ACTION_SPACE_SIZE), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.legal_masks = np.zeros((capacity, ACTION_SPACE_SIZE), dtype=np.float32)
        self.size = 0
        self.index = 0

    def add(self, sample: TrainingSample):
        """Add a single training sample."""
        idx = self.index % self.capacity
        self.observations[idx] = sample.observation
        self.policies[idx] = sample.policy
        self.values[idx] = sample.value
        self.legal_masks[idx] = sample.legal_mask
        self.index += 1
        self.size = min(self.size + 1, self.capacity)

    def add_game(self, observations, policies, legal_masks, outcome: float):
        """Add all positions from a completed game."""
        for obs, pol, mask in zip(observations, policies, legal_masks):
            self.add(TrainingSample(obs, pol, outcome, mask))

    def sample_batch(self, batch_size: int) -> dict:
        """Sample a random mini-batch for training."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "policies": self.policies[indices],
            "values": self.values[indices],
            "legal_masks": self.legal_masks[indices],
        }

    def save(self, path: str):
        np.savez_compressed(path,
            observations=self.observations[:self.size],
            policies=self.policies[:self.size],
            values=self.values[:self.size],
            legal_masks=self.legal_masks[:self.size],
        )

    def load(self, path: str):
        data = np.load(path)
        n = len(data["observations"])
        self.observations[:n] = data["observations"]
        self.policies[:n] = data["policies"]
        self.values[:n] = data["values"]
        self.legal_masks[:n] = data["legal_masks"]
        self.size = n
        self.index = n
```

### `klondike/alphazero/selfplay.py` — Self-Play with PC-PIMC

```python
import numpy as np
import time
from dataclasses import dataclass
from klondike.env import KlondikeEnv
from klondike.encoding import encode_state, OBSERVATION_DIM
from klondike.actions import ACTION_SPACE_SIZE, action_to_move
from klondike.search.mcts import MCTSConfig, mcts_search


@dataclass
class SelfPlayConfig:
    n_determinizations: int = 5         # Per move during self-play
    mcts_simulations: int = 200         # MCTS sims per determinization
    c_puct: float = 1.5
    temperature_moves: int = 20         # Play with τ=1 for first N moves
    temperature_final: float = 0.05     # Then τ→0 (near-greedy)
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    max_moves: int = 2000               # Safety limit
    draw_step: int = 1


def self_play_game(
    model,
    device,
    seed: int,
    config: SelfPlayConfig,
) -> dict:
    """
    Play one self-play game using PC-PIMC.

    Returns game data: observations, policies, legal_masks, outcome, metadata.
    """
    from klondike.nn.model import KlondikeNet  # circular import guard
    model.eval()

    env = KlondikeEnv(draw_step=config.draw_step)
    env.reset(seed=seed)

    observations = []
    policies = []
    legal_masks = []

    move_count = 0
    start_time = time.perf_counter()

    while not env.is_done():
        legal = env.get_legal_action_list()
        if not legal:
            break

        # Temperature schedule
        if move_count < config.temperature_moves:
            temperature = 1.0
        else:
            temperature = config.temperature_final

        # --- PC-PIMC: average over determinized worlds ---
        aggregated_visits = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)

        eval_fn = _make_eval_fn(model, device)

        for d in range(config.n_determinizations):
            # Clone + determinize hidden cards
            det_env = env.clone()
            det_env.shuffle_hidden(rng_seed=seed * 1000 + move_count * 100 + d)

            # Run MCTS on determinized state
            mcts_config = MCTSConfig(
                n_simulations=config.mcts_simulations,
                c_puct=config.c_puct,
                temperature=temperature,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_frac=config.dirichlet_frac if move_count == 0 else 0.0,
            )
            result = mcts_search(det_env, mcts_config, eval_fn=eval_fn)
            aggregated_visits += result.visit_counts

        # Average across determinizations
        aggregated_visits /= config.n_determinizations

        # Normalize to policy distribution
        total_visits = aggregated_visits.sum()
        if total_visits > 0:
            policy = aggregated_visits / total_visits
        else:
            # Fallback: uniform over legal
            policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            for a in legal:
                policy[a] = 1.0 / len(legal)

        # Record training data
        obs = encode_state(env._game)
        mask = env.get_legal_actions()
        observations.append(obs)
        policies.append(policy.astype(np.float32))
        legal_masks.append(mask)

        # Select action from averaged policy
        if temperature < 0.1:
            action = legal[np.argmax([policy[a] for a in legal])]
        else:
            probs = np.array([policy[a] for a in legal])
            probs = probs / (probs.sum() + 1e-8)
            action = legal[np.random.choice(len(legal), p=probs)]

        env.step(action)
        move_count += 1

        if move_count > config.max_moves:
            break

    won = env._game.is_win()
    outcome = 1.0 if won else 0.0
    elapsed = time.perf_counter() - start_time

    return {
        "observations": np.array(observations, dtype=np.float32),
        "policies": np.array(policies, dtype=np.float32),
        "legal_masks": np.array(legal_masks, dtype=np.float32),
        "outcome": outcome,
        "won": won,
        "n_moves": move_count,
        "seed": seed,
        "wall_time": elapsed,
    }


def _make_eval_fn(model, device):
    """Create MCTS evaluation function from neural net."""
    import torch
    from klondike.encoding import encode_state

    def eval_fn(env):
        obs = encode_state(env._game)
        legal_mask = env.get_legal_actions()

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device)
            output = model(obs_t, mask_t)
            policy = torch.exp(output["policy"]).squeeze(0).cpu().numpy()
            value = output["value"].squeeze().item()

        return policy, value

    return eval_fn
```

### `klondike/alphazero/pipeline.py` — Training Pipeline

```python
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from klondike.nn.model import KlondikeNet
from klondike.alphazero.buffer import ReplayBuffer
from klondike.alphazero.selfplay import self_play_game, SelfPlayConfig
from klondike.eval.stats import compute_ci
from klondike.eval.benchmark import run_benchmark


@dataclass
class PipelineConfig:
    # Self-play
    games_per_iteration: int = 200
    selfplay_config: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    # Training
    batch_size: int = 1024
    batches_per_iteration: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Buffer
    buffer_capacity: int = 200_000
    min_buffer_size: int = 5000     # Don't train until this many positions

    # Evaluation
    eval_every: int = 5             # Evaluate every N iterations
    eval_games: int = 1000          # Games per evaluation
    eval_seeds_start: int = 1_000_000  # Fixed eval seeds (separate from training)

    # Checkpointing
    checkpoint_dir: str = "checkpoints/alphazero"
    total_iterations: int = 100

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_pipeline(config: PipelineConfig):
    """
    Main AlphaZero training pipeline.

    Loop:
    1. Self-play: generate games with current model
    2. Add positions to replay buffer
    3. Train: sample mini-batches from buffer, update model
    4. Evaluate: periodically test against fixed seed set
    5. Checkpoint: save model and buffer
    """
    device = torch.device(config.device)

    # Initialize or load model
    model = KlondikeNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    buffer = ReplayBuffer(capacity=config.buffer_capacity)
    checkpoint_path = Path(config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Try to resume from checkpoint
    latest_ckpt = checkpoint_path / "latest.pt"
    start_iteration = 0
    if latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"] + 1
        buffer_path = checkpoint_path / "buffer.npz"
        if buffer_path.exists():
            buffer.load(str(buffer_path))
        print(f"Resumed from iteration {start_iteration}")

    # TrueSight Phase 1 option: initialize from supervised weights
    supervised_path = Path("checkpoints/supervised/best_supervised.pt")
    if supervised_path.exists() and start_iteration == 0:
        model.load_state_dict(torch.load(supervised_path, map_location=device))
        print("Initialized from supervised pre-training (TrueSight Phase 1)")

    best_eval_rate = 0.0

    for iteration in range(start_iteration, config.total_iterations):
        iter_start = time.perf_counter()

        # --- 1. Self-Play ---
        model.eval()
        games_played = 0
        games_won = 0
        total_positions = 0

        for g in range(config.games_per_iteration):
            seed = iteration * config.games_per_iteration + g
            game_data = self_play_game(
                model, device, seed, config.selfplay_config
            )

            buffer.add_game(
                game_data["observations"],
                game_data["policies"],
                game_data["legal_masks"],
                game_data["outcome"],
            )

            games_played += 1
            games_won += int(game_data["won"])
            total_positions += game_data["n_moves"]

        selfplay_rate = games_won / games_played if games_played > 0 else 0

        # --- 2. Train ---
        if buffer.size >= config.min_buffer_size:
            model.train()
            total_policy_loss = 0
            total_value_loss = 0

            for _ in range(config.batches_per_iteration):
                batch = buffer.sample_batch(config.batch_size)

                obs = torch.from_numpy(batch["observations"]).to(device)
                target_policy = torch.from_numpy(batch["policies"]).to(device)
                target_value = torch.from_numpy(batch["values"]).to(device)
                legal_mask = torch.from_numpy(batch["legal_masks"]).to(device)

                output = model(obs, legal_mask)

                # Policy loss: KL divergence with MCTS visit distribution
                log_policy = output["policy"]
                policy_loss = -torch.sum(target_policy * log_policy, dim=-1).mean()

                # Value loss: MSE with game outcome
                value_loss = F.mse_loss(output["value"].squeeze(), target_value)

                loss = (config.policy_loss_weight * policy_loss +
                       config.value_loss_weight * value_loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

            avg_p_loss = total_policy_loss / config.batches_per_iteration
            avg_v_loss = total_value_loss / config.batches_per_iteration
        else:
            avg_p_loss = avg_v_loss = 0.0

        iter_time = time.perf_counter() - iter_start

        print(f"Iter {iteration}: "
              f"selfplay={selfplay_rate:.3f} ({games_won}/{games_played}), "
              f"positions={total_positions}, buffer={buffer.size}, "
              f"p_loss={avg_p_loss:.4f}, v_loss={avg_v_loss:.4f}, "
              f"time={iter_time:.1f}s")

        # --- 3. Evaluate ---
        if (iteration + 1) % config.eval_every == 0:
            eval_result = _evaluate(
                model, device, config.eval_games,
                config.eval_seeds_start, config.selfplay_config.draw_step,
            )
            print(f"  EVAL: win_rate={eval_result['win_rate']:.4f}, "
                  f"CI=({eval_result['ci_lower']:.4f}, {eval_result['ci_upper']:.4f})")

            if eval_result["win_rate"] > best_eval_rate:
                best_eval_rate = eval_result["win_rate"]
                torch.save(model.state_dict(), checkpoint_path / "best.pt")
                print(f"  New best: {best_eval_rate:.4f}")

        # --- 4. Checkpoint ---
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        }, latest_ckpt)
        buffer.save(str(checkpoint_path / "buffer.npz"))


def _evaluate(model, device, n_games, seed_start, draw_step):
    """Evaluate model with greedy policy on fixed seeds."""
    model.eval()
    wins = 0

    for i in range(n_games):
        seed = seed_start + i
        env = KlondikeEnv(draw_step=draw_step)
        env.reset(seed=seed)
        eval_fn = _make_eval_fn(model, device)

        move_count = 0
        while not env.is_done():
            legal = env.get_legal_action_list()
            if not legal:
                break
            policy, value = eval_fn(env)
            action = max(legal, key=lambda a: policy[a])
            env.step(action)
            move_count += 1
            if move_count > 2000:
                break

        if env._game.is_win():
            wins += 1

    ci = compute_ci(wins, n_games)
    return {
        "wins": wins,
        "total": n_games,
        "win_rate": ci.win_rate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
    }
```

### `scripts/train.py` — Main Entry Point

```python
"""
Main training script for AlphaZero Klondike solver.

Usage:
  python scripts/train.py                    # Default settings
  python scripts/train.py --iterations 200   # More iterations
  python scripts/train.py --resume           # Resume from checkpoint
"""
from klondike.alphazero.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    games_per_iteration=200,
    batch_size=1024,
    batches_per_iteration=20,
    eval_every=5,
    eval_games=1000,
    total_iterations=100,
)

run_pipeline(config)
```

---

## TrueSight Curriculum (Two-Phase Training)

### Phase 1: Thoughtful Solitaire (Perfect Information)
- All cards visible → no hidden cards, no PIMC needed
- Standard AlphaZero MCTS (1 determinization = true state)
- Network learns basic Klondike strategy
- Expected: should approach ~82% win rate (matches exact solver)
- Duration: 10-20 iterations (quick convergence on easy problem)

### Phase 2: Switch to PIMC
- Enable hidden cards, switch to PC-PIMC (n_determinizations=5)
- Network must generalize from perfect info to uncertainty
- Initialize from Phase 1 weights (warm start)
- Expected: win rate drops initially, then climbs past search-only baseline
- Duration: 50-100+ iterations

**Implementation**: Set `n_determinizations=1` and use `hidden_clear()` instead of `hidden_shuffle()` for Phase 1. Switch to normal PIMC for Phase 2.

---

## Performance Optimizations (Apply in Order)

1. **FP16 Mixed Precision** — Enable `torch.amp` for 2x training speedup:
   ```python
   scaler = torch.amp.GradScaler()
   with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
       output = model(obs, legal_mask)
       loss = compute_loss(output, targets)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Batched Neural Net Inference in MCTS** — Instead of evaluating one leaf at a time, collect pending leaves and evaluate in one GPU batch. This requires modifying the MCTS to yield pending evaluations.

3. **torch.compile** — Compile the model for 20-40% inference speedup:
   ```python
   model = torch.compile(model, mode='max-autotune')
   ```

4. **Multiprocess Self-Play** — Run N self-play workers on CPU threads, share one GPU for inference:
   ```python
   # Each worker: run game, collect leaf positions
   # Central GPU: batch evaluate all pending leaves
   # Workers: receive evaluations, continue MCTS
   ```

---

## Acceptance Criteria

- [ ] Self-play generates games without crashing
- [ ] Replay buffer correctly stores and samples positions
- [ ] Training loss decreases over iterations
- [ ] Self-play win rate increases over iterations
- [ ] TrueSight Phase 1 reaches >70% on Thoughtful Solitaire
- [ ] TrueSight Phase 2 maintains >40% after switching to PIMC
- [ ] Eval win rate on draw-1 measured with proper CI
- [ ] Best model saved and loadable

---

## Decision Gate

| AlphaZero Draw-1 Result | Action |
|-------------------------|--------|
| **CI lower bound > 50%** | Success! The solver beats the house. Proceed to deployment if desired. |
| **Win rate 49-51%, CI includes 50%** | Promising but inconclusive. Run more games for tighter CI. Tune hyperparameters. |
| **Win rate 47-49%** | Better than search-only but not enough. Try more training, different architectures, ISMCTS. |
| **Win rate ≤ search baseline** | Neural net isn't helping. Debug value head, check for data issues. |

---

## Handoff to Phase 5

Phase 5 (deployment) needs:
1. Trained model weights (best.pt)
2. Confirmed win rate > 50% with statistical evidence
3. `_make_eval_fn` for real-time move recommendations
4. Understanding of model latency (ms per decision)
5. Action mapping to human-readable moves

---

## Compute Estimates

| Component | Estimate |
|-----------|----------|
| Self-play (200 games/iter, ~5s/game) | ~17 min/iter |
| Training (20 batches × 1024) | ~30 sec/iter |
| Evaluation (1000 games, greedy) | ~5 min / every 5 iters |
| **Full 100 iterations** | **~30-40 hours** |
| Buffer storage (200K positions) | ~600 MB |
| Model checkpoints (10 saves) | ~100 MB |
| Total new storage | ~1-2 GB |

**Parallelization opportunity**: Self-play is embarrassingly parallel. With 8 CPU cores doing self-play while GPU trains, throughput can increase 2-4x.
