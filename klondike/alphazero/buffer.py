"""Replay buffer for AlphaZero training data.

Fixed-size FIFO buffer. Stores (observation, MCTS policy, game outcome, legal_mask)
tuples from self-play games. Old positions evicted when capacity reached.
"""

import numpy as np
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
    """Fixed-size replay buffer for AlphaZero training.

    Stores recent self-play positions. Old positions evicted FIFO.
    No gating needed (single-player — no catastrophic forgetting risk).
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
        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return {
            "observations": self.observations[indices],
            "policies": self.policies[indices],
            "values": self.values[indices],
            "legal_masks": self.legal_masks[indices],
        }

    def save(self, path: str):
        np.savez_compressed(
            path,
            observations=self.observations[:self.size],
            policies=self.policies[:self.size],
            values=self.values[:self.size],
            legal_masks=self.legal_masks[:self.size],
            index=np.array([self.index]),
        )

    def load(self, path: str):
        data = np.load(path)
        n = len(data["observations"])
        self.observations[:n] = data["observations"]
        self.policies[:n] = data["policies"]
        self.values[:n] = data["values"]
        self.legal_masks[:n] = data["legal_masks"]
        self.size = n
        if "index" in data:
            self.index = int(data["index"][0])
        else:
            self.index = n

    def __len__(self):
        return self.size
