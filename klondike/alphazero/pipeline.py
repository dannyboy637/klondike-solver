"""AlphaZero training pipeline for Klondike Solitaire.

Main loop: self-play -> replay buffer -> train -> evaluate -> checkpoint.
Supports warm start from supervised weights (TrueSight Phase 1),
FP16 mixed precision, and resumption from checkpoint.
"""

import time
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from klondike.nn.model import KlondikeNet
from klondike.env import KlondikeEnv
from klondike.encoding import encode_state
from klondike.alphazero.buffer import ReplayBuffer
from klondike.alphazero.selfplay import self_play_game, SelfPlayConfig, _make_eval_fn
from klondike.eval.stats import compute_ci


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
    min_buffer_size: int = 5000  # Don't train until this many positions

    # Evaluation — MODEL SELECTION set (disjoint from Phase 6 final test sets)
    # Phase 6 test sets: A=1M-1.01M, B=2M-2.01M, C=3M-3.005M
    eval_every: int = 5
    eval_games: int = 1000
    eval_seeds_start: int = 5_000_000

    # Checkpointing
    checkpoint_dir: str = "checkpoints/alphazero"
    total_iterations: int = 100

    # Model architecture (for fresh model creation)
    hidden_dim: int = 512
    n_blocks: int = 4
    dropout: float = 0.0

    # Hardware
    device: str = ""  # Auto-detect
    use_amp: bool = True  # FP16 mixed precision

    # Warm start
    supervised_weights: str = "checkpoints/supervised_v2/best_supervised.pt"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_pipeline(config: PipelineConfig):
    """Main AlphaZero training pipeline.

    Loop:
    1. Self-play: generate games with current model via PC-PIMC
    2. Add positions to replay buffer
    3. Train: sample mini-batches from buffer, update model
    4. Evaluate: periodically test greedy policy on fixed seeds
    5. Checkpoint: save model, optimizer, buffer
    """
    device_str = config.device or _auto_device()
    device = torch.device(device_str)
    print(f"Device: {device}")

    # AMP setup (MPS doesn't support GradScaler)
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_dtype = torch.float16 if use_amp else torch.float32

    # Initialize model
    model = KlondikeNet(
        hidden_dim=config.hidden_dim,
        n_blocks=config.n_blocks,
        dropout=config.dropout,
    ).to(device)
    print(f"Model: {model.count_parameters():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    buffer = ReplayBuffer(capacity=config.buffer_capacity)
    checkpoint_path = Path(config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Try to resume from checkpoint
    latest_ckpt = checkpoint_path / "latest.pt"
    start_iteration = 0
    best_eval_rate = 0.0
    training_log = []

    if latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"] + 1
        best_eval_rate = ckpt.get("best_eval_rate", 0.0)
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        buffer_path = checkpoint_path / "buffer.npz"
        if buffer_path.exists():
            buffer.load(str(buffer_path))
        log_path = checkpoint_path / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                training_log = json.load(f)
        print(f"Resumed from iteration {start_iteration}, buffer={buffer.size}, best_eval={best_eval_rate:.4f}")

    # Warm start from supervised weights (only if starting fresh)
    elif config.supervised_weights and Path(config.supervised_weights).exists():
        supervised_state = torch.load(config.supervised_weights, map_location=device, weights_only=True)
        model.load_state_dict(supervised_state)
        print(f"Initialized from supervised pre-training: {config.supervised_weights}")

    for iteration in range(start_iteration, config.total_iterations):
        iter_start = time.perf_counter()
        iter_log = {"iteration": iteration}

        # --- 1. Self-Play ---
        model.eval()
        games_won = 0
        total_positions = 0
        total_game_time = 0.0

        for g in range(config.games_per_iteration):
            seed = iteration * config.games_per_iteration + g
            game_data = self_play_game(model, device, seed, config.selfplay_config)

            if game_data["n_moves"] > 0:
                buffer.add_game(
                    game_data["observations"],
                    game_data["policies"],
                    game_data["legal_masks"],
                    game_data["outcome"],
                )

            games_won += int(game_data["won"])
            total_positions += game_data["n_moves"]
            total_game_time += game_data["wall_time"]

        selfplay_rate = games_won / config.games_per_iteration
        avg_game_time = total_game_time / config.games_per_iteration
        iter_log["selfplay_win_rate"] = selfplay_rate
        iter_log["selfplay_positions"] = total_positions
        iter_log["avg_game_time"] = avg_game_time

        # --- 2. Train ---
        avg_p_loss = 0.0
        avg_v_loss = 0.0

        if buffer.size >= config.min_buffer_size:
            model.train()
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for _ in range(config.batches_per_iteration):
                batch = buffer.sample_batch(config.batch_size)

                obs = torch.from_numpy(batch["observations"]).to(device)
                target_policy = torch.from_numpy(batch["policies"]).to(device)
                target_value = torch.from_numpy(batch["values"]).to(device)
                legal_mask = torch.from_numpy(batch["legal_masks"]).to(device)

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        output = model(obs, legal_mask)
                        policy_loss = -torch.sum(target_policy * output["policy"], dim=-1).mean()
                        value_loss = F.mse_loss(output["value"].squeeze(), target_value)
                        loss = config.policy_loss_weight * policy_loss + config.value_loss_weight * value_loss

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(obs, legal_mask)
                    policy_loss = -torch.sum(target_policy * output["policy"], dim=-1).mean()
                    value_loss = F.mse_loss(output["value"].squeeze(), target_value)
                    loss = config.policy_loss_weight * policy_loss + config.value_loss_weight * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

            avg_p_loss = total_policy_loss / config.batches_per_iteration
            avg_v_loss = total_value_loss / config.batches_per_iteration

        iter_log["policy_loss"] = avg_p_loss
        iter_log["value_loss"] = avg_v_loss
        iter_log["buffer_size"] = buffer.size

        iter_time = time.perf_counter() - iter_start
        iter_log["iter_time"] = iter_time

        print(
            f"Iter {iteration:3d}: "
            f"selfplay={selfplay_rate:.3f} ({games_won}/{config.games_per_iteration}), "
            f"pos={total_positions}, buf={buffer.size}, "
            f"p_loss={avg_p_loss:.4f}, v_loss={avg_v_loss:.4f}, "
            f"avg_game={avg_game_time:.1f}s, iter={iter_time:.0f}s"
        )

        # --- 3. Evaluate ---
        if (iteration + 1) % config.eval_every == 0:
            eval_result = _evaluate_greedy(
                model, device, config.eval_games,
                config.eval_seeds_start, config.selfplay_config.draw_step,
            )
            ci = eval_result
            print(
                f"  EVAL: {ci['wins']}/{ci['total']} = {ci['win_rate']:.4f}, "
                f"CI=({ci['ci_lower']:.4f}, {ci['ci_upper']:.4f})"
            )
            iter_log["eval_win_rate"] = ci["win_rate"]
            iter_log["eval_ci_lower"] = ci["ci_lower"]
            iter_log["eval_ci_upper"] = ci["ci_upper"]

            if ci["win_rate"] > best_eval_rate:
                best_eval_rate = ci["win_rate"]
                torch.save(model.state_dict(), checkpoint_path / "best.pt")
                print(f"  New best: {best_eval_rate:.4f}")

        # --- 4. Checkpoint ---
        ckpt_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "best_eval_rate": best_eval_rate,
        }
        if scaler:
            ckpt_data["scaler"] = scaler.state_dict()
        torch.save(ckpt_data, latest_ckpt)
        buffer.save(str(checkpoint_path / "buffer.npz"))

        training_log.append(iter_log)
        with open(checkpoint_path / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    print(f"\nTraining complete. Best eval rate: {best_eval_rate:.4f}")
    print(f"Checkpoints in: {checkpoint_path}")


def _evaluate_greedy(model, device, n_games, seed_start, draw_step):
    """Evaluate model with greedy policy on fixed seeds.

    Uses argmax action selection (no search) for speed.
    This is the fast eval for tracking trends, not for final claims.
    """
    model.eval()
    eval_fn = _make_eval_fn(model, device)
    wins = 0

    for i in range(n_games):
        seed = seed_start + i
        env = KlondikeEnv(draw_step=draw_step)
        env.reset(seed=seed)

        move_count = 0
        seen_states = set()

        while not env.is_done():
            legal = env.get_legal_action_list()
            if not legal:
                break

            state_hash = env._game.encode()
            if state_hash in seen_states:
                break
            seen_states.add(state_hash)

            policy, _ = eval_fn(env)
            action = max(legal, key=lambda a: policy[a])
            env.step(action)
            move_count += 1
            if move_count > 500:
                break

        if env._game.is_win():
            wins += 1

        if (i + 1) % 100 == 0:
            ci = compute_ci(wins, i + 1)
            print(f"  eval {i+1}/{n_games}: {wins}/{i+1} = {ci.win_rate:.4f} CI({ci.ci_lower:.4f}, {ci.ci_upper:.4f})")

    ci = compute_ci(wins, n_games)
    return {
        "wins": wins,
        "total": n_games,
        "win_rate": ci.win_rate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
    }
