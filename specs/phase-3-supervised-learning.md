# Phase 3: Supervised Learning

**Gate**: Gate 2 — Cheap Supervised Distillation Before Full Self-Play
**Parallelizable**: Partially — Data generation (Task A) then NN training (Task B)
**Estimated sessions**: 2-3
**Dependencies**: Phase 2 complete (search framework working, baseline results recorded)

---

## Overview

Train a neural network by imitating the search solver's decisions (supervised learning / distillation). This validates the entire pipeline cheaply before committing to full AlphaZero self-play.

**Key insight**: If a supervised net can match search quality at 100x the speed, we know the state encoding works, the training pipeline works, and the neural net architecture is expressive enough. If it can't, we fix problems here instead of in the more complex self-play loop.

**Deliverables:**
1. Training data: 50K-100K game states with search policy + outcome labels
2. `klondike/nn/model.py` — MLP-ResNet architecture (~2M params)
3. `klondike/nn/trainer.py` — Training loop with validation
4. Trained supervised policy net
5. Comparison: supervised net vs search baseline (speed and quality)

---

## Team Agent Strategy

### TASK A (Data Agent): Generate Training Data
- Runs search solver on 50K-100K deals
- Records (observation, search_policy, game_outcome) triples
- Saves as efficient binary format (numpy arrays)
- Can start immediately once Phase 2 is done

### TASK B (NN Agent): Build and Train Network
- Implements MLP-ResNet architecture
- Implements training loop
- Can start with architecture + training code while data generates
- Trains once data is ready

### SHARED INTERFACE:

```python
# Training data format (numpy arrays on disk):
# observations.npy: shape (N, OBSERVATION_DIM), dtype float32
# policies.npy: shape (N, ACTION_SPACE_SIZE), dtype float32
#   - Search visit distribution (normalized), masked to legal moves
# values.npy: shape (N,), dtype float32
#   - Game outcome: 1.0 = win, 0.0 = loss
# legal_masks.npy: shape (N, ACTION_SPACE_SIZE), dtype float32
#   - Binary mask of legal moves at each state

# All arrays aligned by index: row i is the same game state
```

---

## TASK A: Training Data Generation

### `scripts/generate_data.py`

```python
"""
Generate supervised training data by running search solver on many deals.

For each deal:
1. Play the game using the search solver (PIMC or HOP)
2. At each move, record:
   - The observation vector (857-dim)
   - The search policy (visit distribution over actions)
   - The legal action mask
3. After the game ends, label all positions with the outcome (win/loss)

Output: numpy arrays saved to data/ directory
"""
import numpy as np
from pathlib import Path
from klondike.env import KlondikeEnv
from klondike.encoding import OBSERVATION_DIM, encode_state
from klondike.actions import ACTION_SPACE_SIZE
from klondike.search.pimc import pimc_search, PIMCConfig

# Use best search config from Phase 2 ablations
SEARCH_CONFIG = PIMCConfig(
    n_determinizations=10,
    n_hop_iterations=3000,
    node_limit=1000,
)

def generate_game_data(seed: int, draw_step: int = 1) -> dict:
    """Play one game with search solver, recording all states."""
    env = KlondikeEnv(draw_step=draw_step)
    env.reset(seed=seed)

    observations = []
    policies = []
    legal_masks = []
    n_moves = 0

    while not env.is_done():
        legal = env.get_legal_action_list()
        if not legal:
            break

        # Get observation
        obs = encode_state(env._game)

        # Get search policy
        result = pimc_search(env, SEARCH_CONFIG)

        # Normalize visit counts to probability distribution
        policy = result.visit_counts.copy()
        total = policy.sum()
        if total > 0:
            policy = policy / total

        # Legal mask
        mask = env.get_legal_actions()

        observations.append(obs)
        policies.append(policy)
        legal_masks.append(mask)

        # Play the best move
        env.step(result.action)
        n_moves += 1

        if n_moves > 2000:
            break

    won = env._game.is_win()
    value = 1.0 if won else 0.0

    return {
        "observations": np.array(observations, dtype=np.float32),
        "policies": np.array(policies, dtype=np.float32),
        "legal_masks": np.array(legal_masks, dtype=np.float32),
        "values": np.full(len(observations), value, dtype=np.float32),
        "won": won,
        "n_moves": n_moves,
        "seed": seed,
    }


def generate_dataset(
    n_games: int = 50000,
    seed_start: int = 0,
    output_dir: str = "data/supervised",
    draw_step: int = 1,
):
    """Generate full dataset by playing many games."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_policies = []
    all_masks = []
    all_values = []
    wins = 0

    for i in range(n_games):
        seed = seed_start + i
        data = generate_game_data(seed, draw_step)

        all_obs.append(data["observations"])
        all_policies.append(data["policies"])
        all_masks.append(data["legal_masks"])
        all_values.append(data["values"])

        if data["won"]:
            wins += 1

        if (i + 1) % 100 == 0:
            total_positions = sum(len(o) for o in all_obs)
            print(f"Game {i+1}/{n_games}: "
                  f"win_rate={wins/(i+1):.3f}, "
                  f"total_positions={total_positions}")

        # Save checkpoint every 5000 games
        if (i + 1) % 5000 == 0:
            _save_arrays(all_obs, all_policies, all_masks, all_values, output_path)

    _save_arrays(all_obs, all_policies, all_masks, all_values, output_path)
    print(f"Done: {n_games} games, {wins} wins, "
          f"{sum(len(o) for o in all_obs)} total positions")


def _save_arrays(obs_list, pol_list, mask_list, val_list, output_path):
    """Concatenate and save arrays."""
    np.save(output_path / "observations.npy",
            np.concatenate(obs_list, axis=0))
    np.save(output_path / "policies.npy",
            np.concatenate(pol_list, axis=0))
    np.save(output_path / "legal_masks.npy",
            np.concatenate(mask_list, axis=0))
    np.save(output_path / "values.npy",
            np.concatenate(val_list, axis=0))
```

**Data size estimate**: 50K games × ~50 moves/game × ~4KB/position = ~10GB raw.
With compression or chunked loading: ~2-5GB on disk.

**Compute time**: 50K games × ~2-4 sec/game (PIMC search) = 28-56 hours.
Parallelized across 8 cores: ~3.5-7 hours.

---

## TASK B: Neural Network Architecture & Training

### `klondike/nn/model.py` — MLP-ResNet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from klondike.encoding import OBSERVATION_DIM
from klondike.actions import ACTION_SPACE_SIZE


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and skip connection."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Fixup initialization (KataGo: drop batch norm, 1.6-1.8x speedup)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.relu(x + residual)


class KlondikeNet(nn.Module):
    """
    MLP-ResNet for Klondike Solitaire.

    Architecture:
    - Input: observation vector (~857 dims, but use OBSERVATION_DIM)
    - Input projection: Linear → ReLU
    - N residual blocks (default 4 × 512 hidden)
    - Dual heads:
      - Policy head: Linear → softmax (over ACTION_SPACE_SIZE=260)
      - Value head: Linear → Linear → tanh → rescale to [0, 1]
    - Optional auxiliary heads: foundation count, face-down count

    No batch normalization — uses Fixup initialization instead.
    ~2M parameters at default settings.
    """

    def __init__(
        self,
        input_dim: int = OBSERVATION_DIM,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        action_dim: int = ACTION_SPACE_SIZE,
        use_auxiliary: bool = True,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual tower
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])

        # Policy head
        self.policy_fc = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value_fc1 = nn.Linear(hidden_dim, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Auxiliary heads (KataGo insight — sharper gradients)
        self.use_auxiliary = use_auxiliary
        if use_auxiliary:
            self.aux_foundation = nn.Linear(hidden_dim, 1)  # Predict foundation count
            self.aux_facedown = nn.Linear(hidden_dim, 1)    # Predict face-down count

    def forward(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            obs: (batch, input_dim) observation vectors
            legal_mask: (batch, action_dim) binary mask of legal actions

        Returns dict with:
            "policy": (batch, action_dim) log-probabilities (masked)
            "value": (batch, 1) win probability [0, 1]
            "aux_foundation": (batch, 1) predicted foundation count [optional]
            "aux_facedown": (batch, 1) predicted face-down count [optional]
        """
        # Shared trunk
        x = F.relu(self.input_proj(obs))
        for block in self.blocks:
            x = block(x)

        # Policy head
        logits = self.policy_fc(x)
        if legal_mask is not None:
            # Mask illegal moves to -inf before log_softmax
            logits = logits + (1 - legal_mask) * (-1e9)
        log_policy = F.log_softmax(logits, dim=-1)

        # Value head
        v = F.relu(self.value_fc1(x))
        v = torch.sigmoid(self.value_fc2(v))  # [0, 1] = win probability

        result = {"policy": log_policy, "value": v}

        # Auxiliary heads
        if self.use_auxiliary:
            result["aux_foundation"] = self.aux_foundation(x)
            result["aux_facedown"] = self.aux_facedown(x)

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**Parameter count estimate**:
- Input projection: 705 × 512 + 512 = 361,472
- 4 residual blocks: 4 × (512×512 + 512 + 512×512 + 512) = 4 × 525,312 = 2,101,248
- Policy head: 512 × 260 + 260 = 133,380
- Value head: 512 × 256 + 256 + 256 × 1 + 1 = 131,329
- Auxiliary: 512 × 2 + 2 = 1,026
- **Total: ~2.7M parameters**

### `klondike/nn/trainer.py` — Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from klondike.nn.model import KlondikeNet


class KlondikeDataset(Dataset):
    """Dataset of (observation, policy_target, value_target, legal_mask) tuples."""

    def __init__(self, data_dir: str):
        path = Path(data_dir)
        self.observations = np.load(path / "observations.npy", mmap_mode="r")
        self.policies = np.load(path / "policies.npy", mmap_mode="r")
        self.values = np.load(path / "values.npy", mmap_mode="r")
        self.legal_masks = np.load(path / "legal_masks.npy", mmap_mode="r")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.observations[idx].copy()),
            torch.from_numpy(self.policies[idx].copy()),
            torch.tensor(self.values[idx], dtype=torch.float32),
            torch.from_numpy(self.legal_masks[idx].copy()),
        )


class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 50
    val_split: float = 0.1
    checkpoint_dir: str = "checkpoints/supervised"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    aux_loss_weight: float = 0.5


def train_supervised(
    model: KlondikeNet,
    data_dir: str,
    config: TrainerConfig,
):
    """
    Train model on supervised data from search solver.

    Loss = policy_weight * CrossEntropy(policy, target)
         + value_weight * MSE(value, target)
         + aux_weight * MSE(aux_predictions, aux_targets)
    """
    device = torch.device(config.device)
    model = model.to(device)

    dataset = KlondikeDataset(data_dir)

    # Train/val split
    n_val = int(len(dataset) * config.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    checkpoint_path = Path(config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for obs, target_policy, target_value, legal_mask in train_loader:
            obs = obs.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)
            legal_mask = legal_mask.to(device)

            output = model(obs, legal_mask)

            # Policy loss: cross-entropy with search distribution
            # target_policy is a probability distribution, not one-hot
            # Use KL divergence or cross-entropy
            log_policy = output["policy"]
            policy_loss = -torch.sum(target_policy * log_policy, dim=-1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(output["value"].squeeze(), target_value)

            loss = (config.policy_loss_weight * policy_loss +
                    config.value_loss_weight * value_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_policy_acc = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for obs, target_policy, target_value, legal_mask in val_loader:
                obs = obs.to(device)
                target_policy = target_policy.to(device)
                target_value = target_value.to(device)
                legal_mask = legal_mask.to(device)

                output = model(obs, legal_mask)

                log_policy = output["policy"]
                policy_loss = -torch.sum(target_policy * log_policy, dim=-1).mean()
                value_loss = F.mse_loss(output["value"].squeeze(), target_value)
                loss = policy_loss + value_loss

                # Top-1 accuracy: does the net pick the same best move as search?
                pred_action = output["policy"].argmax(dim=-1)
                target_action = target_policy.argmax(dim=-1)
                acc = (pred_action == target_action).float().mean()

                val_loss += loss.item()
                val_policy_acc += acc.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        avg_val_acc = val_policy_acc / max(n_val_batches, 1)

        print(f"Epoch {epoch+1}/{config.epochs}: "
              f"train_loss={train_loss/len(train_loader):.4f}, "
              f"val_loss={avg_val_loss:.4f}, "
              f"policy_acc={avg_val_acc:.3f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),
                      checkpoint_path / "best_supervised.pt")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                      checkpoint_path / f"epoch_{epoch+1}.pt")

    return model
```

---

## Evaluation: Supervised Net vs Search

After training, evaluate the supervised net's play strength:

```python
def neural_eval_fn(model, device):
    """Create an eval function for MCTS from a trained model."""
    model.eval()

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


# Option 1: Pure policy (no search) — fast but weaker
def greedy_policy_search(model, device):
    """Use model policy directly without MCTS (fastest)."""
    eval_fn = neural_eval_fn(model, device)

    def search_fn(env):
        policy, value = eval_fn(env)
        legal = env.get_legal_action_list()
        best = max(legal, key=lambda a: policy[a])
        return SearchResult(action=best, visit_counts=policy,
                          value=value, determinizations_used=0,
                          simulations_per_det=0, wall_time_ms=0)
    return search_fn


# Option 2: MCTS guided by neural net — slower but stronger
def neural_mcts_search(model, device, n_sims=200):
    """Use model to guide MCTS (Phase 4 preview)."""
    eval_fn = neural_eval_fn(model, device)
    config = MCTSConfig(n_simulations=n_sims, temperature=0.0)

    def search_fn(env):
        return mcts_search(env, config, eval_fn=eval_fn)
    return search_fn
```

### Benchmark comparisons to run:

1. **Search-only** (best config from Phase 2) — quality baseline
2. **Greedy policy** (no search) — speed baseline
3. **Neural MCTS** (model-guided, 200 sims) — quality + speed
4. **Neural PIMC** (model-guided determinization) — optional

Compare: win rate, CI, time per game.

---

## Acceptance Criteria

- [ ] Training data generated (≥10K games minimum, 50K target)
- [ ] KlondikeNet builds, trains, and produces sensible outputs
- [ ] Validation policy accuracy > 50% (net picks same move as search majority of time)
- [ ] Brier score on value predictions < 0.25 (better than always-predict-0.5)
- [ ] Greedy policy plays complete games without crashing
- [ ] Greedy policy win rate measured with CI
- [ ] Neural MCTS win rate measured with CI
- [ ] All results compared with search baseline using paired seeds

---

## Decision Gate

| Supervised Net Result | Action |
|----------------------|--------|
| **Greedy > 90% of search win rate** | Excellent. State encoding and architecture work. Proceed to Phase 4. |
| **Greedy = 70-90% of search** | Acceptable. Architecture is learning, may need tuning. Proceed. |
| **Greedy < 70% of search** | Problem. Debug encoding, architecture, or training. Don't proceed until fixed. |
| **Neural MCTS > search baseline** | The holy grail at Gate 2. Neural guidance already helps! |

---

## Handoff to Phase 4

Phase 4 needs:
1. Working `KlondikeNet` model class
2. Working training loop (`train_supervised`)
3. Trained model weights (best_supervised.pt)
4. `neural_eval_fn` — plug model into MCTS as evaluation function
5. Benchmark results: supervised net vs search baseline
6. Data pipeline proven to work (observations, policies, values)
7. Any encoding fixes discovered during training

---

## Compute & Storage Estimates

| Component | Estimate |
|-----------|----------|
| Data generation (50K games, PIMC search) | 3-7 hours (8-core parallel) |
| Raw training data on disk | 2-5 GB |
| Training (50 epochs, RTX 4060) | ~1-2 hours |
| Model checkpoint | ~10 MB |
| Total new storage | ~3-6 GB |
