import numpy as np
from klondike_env import KlondikeGame  # The Rust module
from klondike.encoding import encode_state
from klondike.actions import ACTION_SPACE_SIZE, action_to_move, move_to_action


class KlondikeEnv:
    """Gym-like wrapper around Lonelybot via PyO3."""

    def __init__(self, draw_step: int = 1):
        self.draw_step = draw_step
        self._game = None

    def reset(self, seed: int) -> np.ndarray:
        """Reset to a new deal. Returns observation vector."""
        self._game = KlondikeGame(seed=seed, draw_step=self.draw_step)
        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take action. Returns (obs, reward, done, info)."""
        move_type, card_idx = action_to_move(action)
        success = self._game.do_move(move_type, card_idx)
        if not success:
            raise ValueError(f"Illegal move: action={action}")

        done = self._game.is_win()
        reward = 1.0 if done else 0.0
        obs = self._get_obs()
        info = {"is_sure_win": self._game.is_sure_win()}
        return obs, reward, done, info

    def get_legal_actions(self) -> np.ndarray:
        """Returns binary mask of legal actions (size ACTION_SPACE_SIZE)."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for move_type, card_idx in self._game.get_legal_moves():
            action = move_to_action(move_type, card_idx)
            mask[action] = 1.0
        return mask

    def get_legal_action_list(self) -> list[int]:
        """Returns list of legal action indices."""
        return [move_to_action(mt, ci) for mt, ci in self._game.get_legal_moves()]

    def clone(self) -> "KlondikeEnv":
        """Deep copy for PIMC branching."""
        new_env = KlondikeEnv(draw_step=self.draw_step)
        new_env._game = self._game.clone_game()
        return new_env

    def shuffle_hidden(self, rng_seed: int) -> None:
        """Randomly rearrange hidden cards (PIMC determinization)."""
        self._game.shuffle_hidden(rng_seed)

    def is_done(self) -> bool:
        """Check if game is won or no legal moves remain."""
        return self._game.is_win() or len(self._game.get_legal_moves()) == 0

    def _get_obs(self) -> np.ndarray:
        """Extract observation vector from current state."""
        return encode_state(self._game)
