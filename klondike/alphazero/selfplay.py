"""Self-play game generation with PC-PIMC for AlphaZero training.

PC-PIMC (Policy Combining Perfect Information Monte Carlo):
At each move: sample D determinized worlds -> run MCTS on each -> average
visit distributions -> select action. Neural net sees ONLY visible state.
"""

import numpy as np
import time
import torch
from dataclasses import dataclass

from klondike.env import KlondikeEnv
from klondike.encoding import encode_state, OBSERVATION_DIM
from klondike.actions import ACTION_SPACE_SIZE, action_to_move
from klondike.search.mcts import MCTSConfig, MCTSNode, mcts_search


@dataclass
class SelfPlayConfig:
    n_determinizations: int = 5     # Per move during self-play (3-5 for speed)
    mcts_simulations: int = 200     # MCTS sims per determinization
    c_puct: float = 1.5
    temperature_moves: int = 20     # tau=1 for first N moves (exploration)
    temperature_final: float = 0.05 # Then tau -> near-greedy
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    max_moves: int = 500
    draw_step: int = 1


def self_play_game(
    model,
    device,
    seed: int,
    config: SelfPlayConfig,
) -> dict:
    """Play one self-play game using PC-PIMC.

    Returns dict with observations, policies, legal_masks, outcome, metadata.
    """
    model.eval()

    env = KlondikeEnv(draw_step=config.draw_step)
    env.reset(seed=seed)

    observations = []
    policies = []
    legal_masks = []

    move_count = 0
    start_time = time.perf_counter()
    seen_states = set()  # Cycle detection

    eval_fn = _make_eval_fn(model, device)

    while not env.is_done():
        legal = env.get_legal_action_list()
        if not legal:
            break

        # Cycle detection
        state_hash = env._game.encode()
        if state_hash in seen_states:
            break  # Treat as loss
        seen_states.add(state_hash)

        if move_count >= config.max_moves:
            break

        # Temperature schedule
        temperature = 1.0 if move_count < config.temperature_moves else config.temperature_final

        # --- PC-PIMC: average MCTS policies over determinized worlds ---
        aggregated_visits = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)

        for d in range(config.n_determinizations):
            # Clone + determinize hidden cards (deterministic seeding)
            det_env = env.clone()
            det_env.shuffle_hidden(rng_seed=seed * 10000 + move_count * 100 + d)

            # Run MCTS on determinized state
            mcts_config = MCTSConfig(
                n_simulations=config.mcts_simulations,
                c_puct=config.c_puct,
                temperature=temperature,
                dirichlet_alpha=config.dirichlet_alpha,
                # Only add noise on first determinization to avoid washing out signal
                dirichlet_frac=config.dirichlet_frac if d == 0 else 0.0,
            )
            result = mcts_search(det_env, mcts_config, eval_fn=eval_fn)
            aggregated_visits += result.visit_counts

        # Average across determinizations -> policy target
        total_visits = aggregated_visits.sum()
        if total_visits > 0:
            policy = (aggregated_visits / total_visits).astype(np.float32)
        else:
            # Fallback: uniform over legal
            policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            for a in legal:
                policy[a] = 1.0 / len(legal)

        # Record training data (encode visible state, NOT determinized)
        obs = encode_state(env._game)
        mask = env.get_legal_actions()
        observations.append(obs)
        policies.append(policy)
        legal_masks.append(mask)

        # Select action from averaged policy
        if temperature < 0.1:
            # Near-greedy: pick highest-policy legal action
            action = max(legal, key=lambda a: policy[a])
        else:
            probs = np.array([policy[a] for a in legal], dtype=np.float64)
            probs = probs / (probs.sum() + 1e-8)
            action = legal[np.random.choice(len(legal), p=probs)]

        env.step(action)
        move_count += 1

    won = env._game.is_win()
    outcome = 1.0 if won else 0.0
    elapsed = time.perf_counter() - start_time

    return {
        "observations": np.array(observations, dtype=np.float32) if observations else np.zeros((0, OBSERVATION_DIM), dtype=np.float32),
        "policies": np.array(policies, dtype=np.float32) if policies else np.zeros((0, ACTION_SPACE_SIZE), dtype=np.float32),
        "legal_masks": np.array(legal_masks, dtype=np.float32) if legal_masks else np.zeros((0, ACTION_SPACE_SIZE), dtype=np.float32),
        "outcome": outcome,
        "won": won,
        "n_moves": move_count,
        "seed": seed,
        "wall_time": elapsed,
    }


def _make_eval_fn(model, device):
    """Create MCTS evaluation function from neural net.

    Returns (policy_probs, value) for a given env state.
    Uses torch.no_grad and optional autocast for speed.
    """

    @torch.no_grad()
    def eval_fn(env):
        obs = encode_state(env._game)
        legal_mask = env.get_legal_actions()

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device)

        output = model(obs_t, mask_t)
        # model outputs log-probs; convert to probs
        policy = torch.exp(output["policy"]).squeeze(0).cpu().numpy()
        value = output["value"].squeeze().item()

        return policy, value

    return eval_fn
