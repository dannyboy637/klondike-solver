import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from klondike.nn.model import KlondikeNet


class KlondikeDataset(Dataset):
    """Dataset of (observation, policy_target, value_target, legal_mask) tuples.

    Loads memmap arrays from disk — constant RAM regardless of dataset size.
    """

    def __init__(self, data_dir: str):
        path = Path(data_dir)
        self.observations = np.load(path / "observations.npy", mmap_mode="r")
        self.policies = np.load(path / "policies.npy", mmap_mode="r")
        self.values = np.load(path / "values.npy", mmap_mode="r")
        self.legal_masks = np.load(path / "legal_masks.npy", mmap_mode="r")
        self.game_offsets = np.load(path / "game_offsets.npy")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.observations[idx].copy()),
            torch.from_numpy(self.policies[idx].copy()),
            torch.tensor(self.values[idx], dtype=torch.float32),
            torch.from_numpy(self.legal_masks[idx].copy()),
        )


def game_level_split(dataset: KlondikeDataset, val_fraction: float = 0.1, rng_seed: int = 42):
    """
    Split dataset at the GAME level, not position level.

    Positions from the same game share near-identical features and the same
    value label. Position-level random_split leaks information between train
    and val, producing overoptimistic validation metrics.
    """
    rng = np.random.RandomState(rng_seed)
    n_games = len(dataset.game_offsets) - 1
    game_indices = rng.permutation(n_games)
    n_val_games = max(1, int(n_games * val_fraction))

    val_games = set(game_indices[:n_val_games].tolist())

    train_positions = []
    val_positions = []
    for g in range(n_games):
        start = int(dataset.game_offsets[g])
        end = int(dataset.game_offsets[g + 1])
        positions = list(range(start, end))
        if g in val_games:
            val_positions.extend(positions)
        else:
            train_positions.extend(positions)

    return Subset(dataset, train_positions), Subset(dataset, val_positions)


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 50
    val_split: float = 0.1
    checkpoint_dir: str = "checkpoints/supervised"
    device: str = ""  # Auto-detect if empty
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    label_smoothing: float = 0.0  # Smooth one-hot policy targets toward uniform over legal moves
    patience: int = 0  # Early stopping patience (0 = disabled)


def _get_device(config: TrainerConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_supervised(
    model: KlondikeNet,
    data_dir: str,
    config: TrainerConfig,
) -> KlondikeNet:
    """
    Train model on supervised data from search solver.

    Loss = policy_weight * CrossEntropy(policy, target)
         + value_weight * MSE(value, target)

    Auxiliary targets (foundation count, face-down count) are deferred to Phase 6.
    """
    device = _get_device(config)
    print(f"Training on device: {device}")
    model = model.to(device)

    dataset = KlondikeDataset(data_dir)
    print(f"Dataset: {len(dataset)} positions from {len(dataset.game_offsets)-1} games")

    train_ds, val_ds = game_level_split(dataset, config.val_split)
    print(f"Train: {len(train_ds)} positions, Val: {len(val_ds)} positions")

    # num_workers=0 on MPS (multiprocessing overhead dominates).
    # On CUDA with large datasets, use 4 workers.
    n_workers = 4 if device.type == "cuda" and len(train_ds) > 50000 else 0
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, num_workers=n_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=n_workers, pin_memory=pin,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    checkpoint_path = Path(config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()

        # Training
        model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_batches = 0

        for obs, target_policy, target_value, legal_mask in train_loader:
            obs = obs.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)
            legal_mask = legal_mask.to(device)

            output = model(obs, legal_mask)

            # Apply label smoothing: blend one-hot target toward uniform over legal moves
            if config.label_smoothing > 0:
                n_legal = legal_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                uniform_legal = legal_mask / n_legal
                target_policy = (
                    (1 - config.label_smoothing) * target_policy
                    + config.label_smoothing * uniform_legal
                )

            # Policy loss: cross-entropy with (possibly smoothed) distribution
            log_policy = output["policy"]
            policy_loss = -torch.sum(target_policy * log_policy, dim=-1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(output["value"].squeeze(-1), target_value)

            loss = (config.policy_loss_weight * policy_loss +
                    config.value_loss_weight * value_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_policy_loss += policy_loss.item()
            train_value_loss += value_loss.item()
            train_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_policy_acc = 0.0
        val_value_mae = 0.0
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
                value_loss = F.mse_loss(output["value"].squeeze(-1), target_value)

                # Top-1 accuracy: does the net pick the same best move as search?
                pred_action = output["policy"].argmax(dim=-1)
                target_action = target_policy.argmax(dim=-1)
                acc = (pred_action == target_action).float().mean()

                # Value MAE (easier to interpret than MSE)
                mae = (output["value"].squeeze(-1) - target_value).abs().mean()

                val_policy_loss += policy_loss.item()
                val_value_loss += value_loss.item()
                val_policy_acc += acc.item()
                val_value_mae += mae.item()
                n_val_batches += 1

        avg_train_ploss = train_policy_loss / max(train_batches, 1)
        avg_train_vloss = train_value_loss / max(train_batches, 1)
        avg_val_ploss = val_policy_loss / max(n_val_batches, 1)
        avg_val_vloss = val_value_loss / max(n_val_batches, 1)
        avg_val_acc = val_policy_acc / max(n_val_batches, 1)
        avg_val_mae = val_value_mae / max(n_val_batches, 1)
        avg_val_loss = avg_val_ploss + avg_val_vloss
        elapsed = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch+1:3d}/{config.epochs}: "
            f"train_p={avg_train_ploss:.4f} train_v={avg_train_vloss:.4f} | "
            f"val_p={avg_val_ploss:.4f} val_v={avg_val_vloss:.4f} "
            f"acc={avg_val_acc:.3f} mae={avg_val_mae:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path / "best_supervised.pt")
            print(f"  -> New best val_loss={avg_val_loss:.4f}")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), checkpoint_path / f"epoch_{epoch+1}.pt")

        # Early stopping
        if config.patience > 0 and patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={config.patience})")
            break

    # Brier score on validation set (calibration metric)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for obs, _, target_value, legal_mask in val_loader:
            obs = obs.to(device)
            legal_mask = legal_mask.to(device)
            output = model(obs, legal_mask)
            all_preds.append(output["value"].squeeze(-1).cpu())
            all_targets.append(target_value)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    brier = ((preds - targets) ** 2).mean().item()
    print(f"\nFinal Brier score (val): {brier:.4f} (< 0.25 = better than always-predict-0.5)")

    return model
