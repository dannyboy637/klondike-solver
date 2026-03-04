#!/usr/bin/env python3
"""
Generate supervised training data by running the strong HOP solver on many deals.

Uses play_hop_turn (pick_moves algorithm = UCB1 + traverse + HOP) which achieves
~55% win rate on draw-1. This is the correct strong algorithm (not hop_solve which
only gets ~18% due to myopic per-move evaluation).

For each deal:
1. Play the game using play_hop_turn (compound move sequences)
2. Clone before each turn, replay to record intermediate states
3. At each state, record: observation (757-dim), one-hot policy of move played, legal mask
4. After game ends, label all positions with outcome (win=1.0, loss=0.0)

Output: memmap numpy arrays in data/supervised/
"""
import argparse
import time
from pathlib import Path

import numpy as np

from klondike.encoding import OBSERVATION_DIM, encode_state
from klondike.actions import ACTION_SPACE_SIZE, move_to_action


def generate_game_data(seed: int, n_times: int = 500, node_limit: int = 100,
                       draw_step: int = 1, max_moves: int = 500,
                       soft_targets: bool = True,
                       hop_n_det: int = 50, hop_limit: int = 200) -> dict:
    """Play one game with strong solver, recording all states.

    Uses play_hop_turn for strong trajectories (~55% win rate).
    Records (observation, policy, legal_mask) at each individual move
    by cloning before each turn and replaying the compound move sequence.

    Policy targets:
    - soft_targets=True: hop_solve win rate distribution at each state (distributional)
    - soft_targets=False: one-hot of move played by strong solver (behavior cloning)
    """
    from klondike_env import KlondikeGame

    game = KlondikeGame(seed=seed, draw_step=draw_step)

    observations = []
    policies = []
    legal_masks = []

    move_count = 0
    rng_base = seed * 1000
    seen_states = set()

    while not game.is_win():
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        # Cycle detection at turn boundaries
        state_hash = game.encode()
        if state_hash in seen_states:
            break
        seen_states.add(state_hash)

        # Clone before play_hop_turn (which modifies game in place)
        clone = game.clone_game()

        # Run strong solver — returns compound move sequence
        moves = game.play_hop_turn(
            rng_base + move_count,
            n_times,
            node_limit,
        )
        if not moves:
            break

        # Replay on clone to record intermediate states
        for move_type, card_idx in moves:
            obs = encode_state(clone)

            # Legal mask from clone's current state
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            for mt, ci in clone.get_legal_moves():
                mask[move_to_action(mt, ci)] = 1.0

            if soft_targets:
                # Soft targets: hop_solve win rate distribution
                hop_result = clone.hop_solve(hop_n_det, hop_limit)
                policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                for (mt_h, ci_h), stats in hop_result.items():
                    played = stats["played"]
                    if played > 0:
                        policy[move_to_action(mt_h, ci_h)] = stats["wins"] / played
                # Normalize to probability distribution
                total = policy.sum()
                if total > 0:
                    policy = policy / total
                else:
                    # Fallback: one-hot of the move played
                    action = move_to_action(move_type, card_idx)
                    policy[action] = 1.0
            else:
                # Hard targets: one-hot of move played by strong solver
                policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                action = move_to_action(move_type, card_idx)
                policy[action] = 1.0

            observations.append(obs)
            policies.append(policy)
            legal_masks.append(mask)

            # Advance clone
            success = clone.do_move(move_type, card_idx)
            if not success:
                print(f"WARNING: Move ({move_type}, {card_idx}) illegal on clone at seed {seed}")
                break

        move_count += len(moves)
        if move_count > max_moves:
            break

    won = game.is_win()
    n_positions = len(observations)

    if n_positions == 0:
        return {
            "observations": np.empty((0, OBSERVATION_DIM), dtype=np.float32),
            "policies": np.empty((0, ACTION_SPACE_SIZE), dtype=np.float32),
            "legal_masks": np.empty((0, ACTION_SPACE_SIZE), dtype=np.float32),
            "won": won,
            "n_moves": move_count,
            "n_positions": 0,
            "seed": seed,
        }

    return {
        "observations": np.array(observations, dtype=np.float32),
        "policies": np.array(policies, dtype=np.float32),
        "legal_masks": np.array(legal_masks, dtype=np.float32),
        "won": won,
        "n_moves": move_count,
        "n_positions": n_positions,
        "seed": seed,
    }


def generate_dataset(
    n_games: int = 10000,
    seed_start: int = 0,
    output_dir: str = "data/supervised",
    draw_step: int = 1,
    n_times: int = 500,
    node_limit: int = 100,
    max_positions: int = 2_000_000,
    soft_targets: bool = True,
    hop_n_det: int = 50,
    hop_limit: int = 200,
):
    """
    Generate full dataset using memmap arrays for constant RAM usage.

    Writes directly to memory-mapped files on disk. RAM stays constant
    regardless of dataset size.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Pre-allocate memory-mapped arrays
    obs_mm = np.lib.format.open_memmap(
        str(output_path / "observations.npy"), mode="w+",
        dtype=np.float32, shape=(max_positions, OBSERVATION_DIM))
    pol_mm = np.lib.format.open_memmap(
        str(output_path / "policies.npy"), mode="w+",
        dtype=np.float32, shape=(max_positions, ACTION_SPACE_SIZE))
    mask_mm = np.lib.format.open_memmap(
        str(output_path / "legal_masks.npy"), mode="w+",
        dtype=np.float32, shape=(max_positions, ACTION_SPACE_SIZE))
    val_mm = np.lib.format.open_memmap(
        str(output_path / "values.npy"), mode="w+",
        dtype=np.float32, shape=(max_positions,))

    game_offsets = [0]
    pos_idx = 0
    wins = 0
    total_time = 0.0
    games_done = 0

    for i in range(n_games):
        seed = seed_start + i
        t0 = time.perf_counter()
        data = generate_game_data(seed, n_times, node_limit, draw_step,
                                  soft_targets=soft_targets,
                                  hop_n_det=hop_n_det, hop_limit=hop_limit)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        n_pos = data["n_positions"]
        if n_pos == 0:
            game_offsets.append(pos_idx)
            games_done += 1
            continue

        if pos_idx + n_pos > max_positions:
            print(f"WARNING: Reached max_positions={max_positions} at game {i}. Stopping.")
            break

        obs_mm[pos_idx:pos_idx + n_pos] = data["observations"]
        pol_mm[pos_idx:pos_idx + n_pos] = data["policies"]
        mask_mm[pos_idx:pos_idx + n_pos] = data["legal_masks"]
        # Value = game outcome for ALL positions in this game
        val_mm[pos_idx:pos_idx + n_pos] = 1.0 if data["won"] else 0.0

        pos_idx += n_pos
        game_offsets.append(pos_idx)

        if data["won"]:
            wins += 1
        games_done += 1

        if (i + 1) % 100 == 0:
            avg_time = total_time / (i + 1)
            eta_min = avg_time * (n_games - i - 1) / 60
            print(
                f"Game {i+1}/{n_games}: "
                f"win_rate={wins/(i+1):.3f}, "
                f"positions={pos_idx}, "
                f"avg_pos/game={pos_idx/(i+1):.1f}, "
                f"{avg_time:.2f}s/game, "
                f"ETA {eta_min:.1f}min"
            )

    # Truncate to actual size via streaming copy
    for name_arr, src_mm, shape in [
        ("observations.npy", obs_mm, (pos_idx, OBSERVATION_DIM)),
        ("policies.npy", pol_mm, (pos_idx, ACTION_SPACE_SIZE)),
        ("legal_masks.npy", mask_mm, (pos_idx, ACTION_SPACE_SIZE)),
        ("values.npy", val_mm, (pos_idx,)),
    ]:
        dst = np.lib.format.open_memmap(
            str(output_path / f"_{name_arr}"), mode="w+",
            dtype=np.float32, shape=shape)
        # Stream copy in chunks to avoid loading full array
        chunk = 100_000
        for start in range(0, pos_idx, chunk):
            end = min(start + chunk, pos_idx)
            dst[start:end] = src_mm[start:end]
        del dst
        (output_path / f"_{name_arr}").rename(output_path / name_arr)

    np.save(str(output_path / "game_offsets.npy"), np.array(game_offsets, dtype=np.int64))

    del obs_mm, pol_mm, mask_mm, val_mm

    print(f"\nDone: {games_done} games, {wins} wins ({wins/max(games_done,1)*100:.1f}%), "
          f"{pos_idx} total positions")
    print(f"Avg positions/game: {pos_idx/max(games_done,1):.1f}")
    print(f"Total time: {total_time/60:.1f} min ({total_time/max(games_done,1):.2f}s/game)")
    print(f"Saved to {output_path}")
    print(f"game_offsets.npy: {len(game_offsets)} entries ({len(game_offsets)-1} games)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate supervised training data")
    parser.add_argument("--n-games", type=int, default=10000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data/supervised")
    parser.add_argument("--draw-step", type=int, default=1)
    parser.add_argument("--n-times", type=int, default=500,
                        help="UCB1 budget per decision (default: 500, Phase 2 best)")
    parser.add_argument("--node-limit", type=int, default=100,
                        help="Max search nodes per determinization (default: 100)")
    parser.add_argument("--max-positions", type=int, default=2_000_000)
    parser.add_argument("--no-soft-targets", action="store_true",
                        help="Use one-hot targets instead of hop_solve soft targets")
    parser.add_argument("--hop-n-det", type=int, default=50,
                        help="Determinizations for hop_solve soft targets")
    parser.add_argument("--hop-limit", type=int, default=200,
                        help="Node limit for hop_solve soft targets")
    args = parser.parse_args()

    generate_dataset(
        n_games=args.n_games,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        draw_step=args.draw_step,
        n_times=args.n_times,
        node_limit=args.node_limit,
        max_positions=args.max_positions,
        soft_targets=not args.no_soft_targets,
        hop_n_det=args.hop_n_det,
        hop_limit=args.hop_limit,
    )
