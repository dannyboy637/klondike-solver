import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from klondike.actions import ACTION_SPACE_SIZE, action_to_move
from klondike.env import KlondikeEnv
from klondike.search import SearchResult


@dataclass
class MCTSConfig:
    n_simulations: int = 400
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25


class MCTSNode:
    """Single node in the MCTS tree."""

    __slots__ = ("visit_count", "value_sum", "prior", "children")

    def __init__(self, prior: float = 0.0):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: dict[int, "MCTSNode"] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration


# Neural net plugs in here: (env) -> (policy, value)
EvalFn = Callable[[KlondikeEnv], tuple[np.ndarray, float]]


def default_eval_fn(env: KlondikeEnv) -> tuple[np.ndarray, float]:
    """Uniform policy over legal moves, value 0.5."""
    legal = env.get_legal_actions()
    n_legal = legal.sum()
    if n_legal == 0:
        return np.zeros(ACTION_SPACE_SIZE, dtype=np.float32), 0.0
    policy = legal / n_legal
    return policy, 0.5


def mcts_search(
    env: KlondikeEnv,
    config: MCTSConfig,
    eval_fn: EvalFn = default_eval_fn,
) -> SearchResult:
    """
    MCTS with pluggable evaluation function.

    Phase 2: eval_fn = default_eval_fn (uniform policy, 0.5 value)
    Phase 4: eval_fn = neural_net_eval (learned policy + value)
    """
    start = time.perf_counter()

    legal_mask = env.get_legal_actions()
    if legal_mask.sum() == 0:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SearchResult(
            action=-1,
            visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.float32),
            value=0.0,
            determinizations_used=0,
            simulations_per_det=config.n_simulations,
            wall_time_ms=elapsed_ms,
        )

    root = MCTSNode()

    # Initialize root with eval
    policy, _ = eval_fn(env)
    policy = policy * legal_mask
    total = policy.sum()
    if total > 0:
        policy = policy / total
    else:
        policy = legal_mask / legal_mask.sum()

    # Add Dirichlet noise at root for exploration
    noise = np.random.dirichlet([config.dirichlet_alpha] * ACTION_SPACE_SIZE)
    policy = (1 - config.dirichlet_frac) * policy + config.dirichlet_frac * noise
    policy = policy * legal_mask
    total = policy.sum()
    if total > 0:
        policy = policy / total

    # Create children for legal moves
    for action in range(ACTION_SPACE_SIZE):
        if legal_mask[action] > 0:
            root.children[action] = MCTSNode(prior=float(policy[action]))

    # Run simulations
    for _ in range(config.n_simulations):
        _simulate(env, root, config, eval_fn)
        # CRITICAL: root.visit_count must be incremented here, not in backprop.
        # Backprop only updates children in the path.
        root.visit_count += 1

    # Extract visit distribution
    visit_counts = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for action, child in root.children.items():
        visit_counts[action] = child.visit_count

    # Select action based on temperature (log-space to avoid overflow)
    if config.temperature < 0.01:
        best_action = max(root.children, key=lambda a: root.children[a].visit_count)
    else:
        # Use log-space: log(n^(1/t)) = log(n)/t, then softmax
        log_visits = np.where(visit_counts > 0, np.log(np.maximum(visit_counts, 1e-30)), -1e9)
        log_probs = log_visits / config.temperature
        log_probs -= log_probs.max()  # Numerical stability
        visit_probs = np.exp(log_probs)
        visit_probs *= (visit_counts > 0).astype(np.float32)  # Zero out non-visited
        total = visit_probs.sum()
        if total > 0:
            visit_probs = visit_probs / total
            best_action = int(np.random.choice(ACTION_SPACE_SIZE, p=visit_probs))
        else:
            best_action = max(root.children, key=lambda a: root.children[a].visit_count)

    # CRITICAL: root value from children, not root.value_sum
    total_child_visits = sum(c.visit_count for c in root.children.values())
    root_value = (
        sum(c.value_sum for c in root.children.values()) / max(total_child_visits, 1)
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return SearchResult(
        action=best_action,
        visit_counts=visit_counts,
        value=root_value,
        determinizations_used=0,
        simulations_per_det=config.n_simulations,
        wall_time_ms=elapsed_ms,
    )


def _simulate(env: KlondikeEnv, root: MCTSNode, config: MCTSConfig, eval_fn: EvalFn):
    """
    Single MCTS simulation: select -> expand -> evaluate -> backpropagate.

    Clones the env and plays moves down the tree. At a leaf, evaluates with
    eval_fn and backprops the value.
    """
    sim_env = env.clone()
    path: list[tuple[MCTSNode, int]] = []
    node = root

    # Selection: walk down tree picking best UCB child
    while node.children and not sim_env.is_done():
        action = max(
            node.children,
            key=lambda a: node.children[a].ucb_score(node.visit_count, config.c_puct),
        )
        path.append((node, action))
        node = node.children[action]

        move_type, card_idx = action_to_move(action)
        sim_env._game.do_move(move_type, card_idx)

        # If node hasn't been visited, it's a leaf — stop and expand
        if node.visit_count == 0:
            break

    # Expansion + Evaluation
    if sim_env.is_done():
        value = 1.0 if sim_env._game.is_win() else 0.0
    else:
        legal_mask = sim_env.get_legal_actions()
        if legal_mask.sum() == 0:
            value = 0.0
        else:
            policy, value = eval_fn(sim_env)
            policy = policy * legal_mask
            total = policy.sum()
            if total > 0:
                policy = policy / total
            else:
                policy = legal_mask / legal_mask.sum()
            # Expand: add children for legal moves not yet in tree
            for action in range(ACTION_SPACE_SIZE):
                if legal_mask[action] > 0 and action not in node.children:
                    node.children[action] = MCTSNode(prior=float(policy[action]))

    # Backpropagation: update visit counts and values along the path.
    # root.visit_count is incremented by the caller (mcts_search), not here.
    for parent_node, action in reversed(path):
        child = parent_node.children[action]
        child.visit_count += 1
        child.value_sum += value
