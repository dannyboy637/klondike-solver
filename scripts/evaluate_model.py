#!/usr/bin/env python3
"""
Evaluate trained model: greedy policy play, Neural MCTS, vs search baseline.

Modes:
- greedy: model argmax policy (no search) — fast, tests policy quality
- neural-mcts: MCTS guided by neural net policy+value — tests as MCTS guide
- search: HOP search baseline for paired comparison
"""
import argparse
import time

import numpy as np
import torch

from klondike_env import KlondikeGame
from klondike.encoding import OBSERVATION_DIM, encode_state
from klondike.actions import ACTION_SPACE_SIZE, move_to_action, action_to_move
from klondike.nn.model import KlondikeNet
from klondike.env import KlondikeEnv
from klondike.eval.stats import compute_ci
from klondike.search.pimc import play_full_hop_game, FullHopConfig
from klondike.search.mcts import MCTSConfig, mcts_search, EvalFn


def make_neural_eval_fn(model: KlondikeNet, device: torch.device) -> EvalFn:
    """Create an MCTS evaluation function from a trained neural net.

    Returns a closure that maps KlondikeEnv -> (policy, value).
    The policy is exponentiated from log-probs and masked to legal moves.
    """
    model.eval()

    def neural_eval(env: KlondikeEnv) -> tuple[np.ndarray, float]:
        obs = env._get_obs()
        legal_mask = env.get_legal_actions()

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device)
            output = model(obs_t, mask_t)

            # Convert log-probs to probs
            policy = torch.exp(output["policy"]).squeeze(0).cpu().numpy()
            value = output["value"].squeeze().item()

        # Mask and renormalize
        policy = policy * legal_mask
        total = policy.sum()
        if total > 0:
            policy = policy / total
        else:
            n_legal = legal_mask.sum()
            if n_legal > 0:
                policy = legal_mask / n_legal

        return policy, value

    return neural_eval


def play_neural_mcts_game(
    model: KlondikeNet,
    device: torch.device,
    seed: int,
    mcts_config: MCTSConfig,
    draw_step: int = 1,
    max_moves: int = 500,
) -> tuple[bool, int, float]:
    """Play a complete game using MCTS guided by the neural net.

    Returns: (won, n_moves, wall_time_ms)
    """
    env = KlondikeEnv(draw_step=draw_step)
    env.reset(seed)
    eval_fn = make_neural_eval_fn(model, device)

    move_count = 0
    seen_states = set()
    start = time.perf_counter()

    while not env.is_done():
        # Cycle detection
        state_hash = env._game.encode()
        if state_hash in seen_states:
            break
        seen_states.add(state_hash)

        result = mcts_search(env, mcts_config, eval_fn)
        if result.action < 0:
            break

        mt, ci = action_to_move(result.action)
        env._game.do_move(mt, ci)

        move_count += 1
        if move_count >= max_moves:
            break

    elapsed_ms = (time.perf_counter() - start) * 1000
    return env._game.is_win(), move_count, elapsed_ms


def evaluate_neural_mcts(
    model: KlondikeNet,
    device: torch.device,
    n_games: int = 100,
    seed_start: int = 0,
    n_simulations: int = 100,
    draw_step: int = 1,
) -> dict:
    """Evaluate Neural MCTS over N games."""
    mcts_config = MCTSConfig(
        n_simulations=n_simulations,
        temperature=0.0,  # Deterministic selection
        c_puct=1.5,
    )

    wins = 0
    total_moves = 0
    total_time = 0.0

    for i in range(n_games):
        seed = seed_start + i
        won, n_moves, wall_ms = play_neural_mcts_game(
            model, device, seed, mcts_config, draw_step
        )
        if won:
            wins += 1
        total_moves += n_moves
        total_time += wall_ms

        if (i + 1) % 10 == 0:
            ci = compute_ci(wins, i + 1)
            print(
                f"  [{i+1}/{n_games}] "
                f"win_rate={ci.win_rate:.4f} "
                f"CI=({ci.ci_lower:.4f}, {ci.ci_upper:.4f}) "
                f"avg_moves={total_moves/(i+1):.1f} "
                f"{total_time/(i+1)/1000:.1f}s/game"
            )

    ci = compute_ci(wins, n_games)
    return {
        "wins": wins,
        "n_games": n_games,
        "win_rate": ci.win_rate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
        "avg_moves": total_moves / n_games,
        "avg_time_ms": total_time / n_games,
    }


def play_greedy_game(
    model: KlondikeNet,
    device: torch.device,
    seed: int,
    draw_step: int = 1,
    max_moves: int = 500,
) -> tuple[bool, int, float]:
    """Play a complete game using the model's greedy policy (no search).

    Returns: (won, n_moves, wall_time_ms)
    """
    game = KlondikeGame(seed=seed, draw_step=draw_step)
    move_count = 0
    seen_states = set()
    start = time.perf_counter()

    model.eval()
    with torch.no_grad():
        while not game.is_win():
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            # Cycle detection
            state_hash = game.encode()
            if state_hash in seen_states:
                break
            seen_states.add(state_hash)

            # Encode state
            obs = encode_state(game)
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            for mt, ci in legal_moves:
                mask[move_to_action(mt, ci)] = 1.0

            # Model prediction
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            output = model(obs_t, mask_t)
            action = output["policy"].argmax(dim=-1).item()

            # Execute move
            mt, ci = action_to_move(action)
            success = game.do_move(mt, ci)
            if not success:
                # Shouldn't happen with legal masking, but safety fallback
                break

            move_count += 1
            if move_count >= max_moves:
                break

    elapsed_ms = (time.perf_counter() - start) * 1000
    return game.is_win(), move_count, elapsed_ms


def evaluate_greedy(
    model: KlondikeNet,
    device: torch.device,
    n_games: int = 1000,
    seed_start: int = 0,
    draw_step: int = 1,
) -> dict:
    """Evaluate greedy policy over N games."""
    wins = 0
    total_moves = 0
    total_time = 0.0

    for i in range(n_games):
        seed = seed_start + i
        won, n_moves, wall_ms = play_greedy_game(model, device, seed, draw_step)
        if won:
            wins += 1
        total_moves += n_moves
        total_time += wall_ms

        if (i + 1) % 100 == 0:
            ci = compute_ci(wins, i + 1)
            print(
                f"  [{i+1}/{n_games}] "
                f"win_rate={ci.win_rate:.4f} "
                f"CI=({ci.ci_lower:.4f}, {ci.ci_upper:.4f}) "
                f"avg_moves={total_moves/(i+1):.1f} "
                f"{total_time/(i+1):.1f}ms/game"
            )

    ci = compute_ci(wins, n_games)
    return {
        "wins": wins,
        "n_games": n_games,
        "win_rate": ci.win_rate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
        "avg_moves": total_moves / n_games,
        "avg_time_ms": total_time / n_games,
    }


def evaluate_search_baseline(
    n_games: int = 1000,
    seed_start: int = 0,
    n_times: int = 500,
    node_limit: int = 100,
    draw_step: int = 1,
) -> dict:
    """Run search baseline on same seeds for paired comparison."""
    config = FullHopConfig(n_times=n_times, node_limit=node_limit)
    wins = 0
    total_time = 0.0

    for i in range(n_games):
        seed = seed_start + i
        won, _, wall_ms = play_full_hop_game(seed, config, draw_step)
        if won:
            wins += 1
        total_time += wall_ms

        if (i + 1) % 100 == 0:
            ci = compute_ci(wins, i + 1)
            print(
                f"  [{i+1}/{n_games}] "
                f"win_rate={ci.win_rate:.4f} "
                f"CI=({ci.ci_lower:.4f}, {ci.ci_upper:.4f}) "
                f"{total_time/(i+1):.1f}ms/game"
            )

    ci = compute_ci(wins, n_games)
    return {
        "wins": wins,
        "n_games": n_games,
        "win_rate": ci.win_rate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
        "avg_time_ms": total_time / n_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/supervised/best_supervised.pt")
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=100000,
                        help="Use different seeds than training data (default: 100000)")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip search baseline (if already have numbers)")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--neural-mcts", action="store_true",
                        help="Run Neural MCTS evaluation (model as MCTS guide)")
    parser.add_argument("--mcts-sims", type=int, default=100,
                        help="MCTS simulations per move for neural-mcts mode")
    parser.add_argument("--mcts-games", type=int, default=100,
                        help="Number of games for neural-mcts evaluation")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    model = KlondikeNet(hidden_dim=args.hidden_dim, n_blocks=args.n_blocks)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model: {model.count_parameters():,} params from {args.checkpoint}")

    # Neural MCTS evaluation (if requested)
    if args.neural_mcts:
        print(f"\n--- Neural MCTS ({args.mcts_games} games, {args.mcts_sims} sims/move, "
              f"seeds {args.seed_start}+) ---")
        nmcts = evaluate_neural_mcts(
            model, device, args.mcts_games, args.seed_start, args.mcts_sims
        )
        print(f"\nNeural MCTS: {nmcts['wins']}/{nmcts['n_games']} = {nmcts['win_rate']:.4f} "
              f"CI=({nmcts['ci_lower']:.4f}, {nmcts['ci_upper']:.4f}) "
              f"{nmcts['avg_time_ms']/1000:.1f}s/game")
        return

    # Greedy policy evaluation
    print(f"\n--- Greedy Policy ({args.n_games} games, seeds {args.seed_start}+) ---")
    greedy = evaluate_greedy(model, device, args.n_games, args.seed_start)
    print(f"\nGreedy: {greedy['wins']}/{greedy['n_games']} = {greedy['win_rate']:.4f} "
          f"CI=({greedy['ci_lower']:.4f}, {greedy['ci_upper']:.4f}) "
          f"{greedy['avg_time_ms']:.1f}ms/game")

    # Search baseline (paired seeds)
    if not args.skip_search:
        print(f"\n--- Search Baseline (nt=500, lim=100, {args.n_games} games, same seeds) ---")
        search = evaluate_search_baseline(args.n_games, args.seed_start)
        print(f"\nSearch: {search['wins']}/{search['n_games']} = {search['win_rate']:.4f} "
              f"CI=({search['ci_lower']:.4f}, {search['ci_upper']:.4f}) "
              f"{search['avg_time_ms']:.1f}ms/game")

        # Comparison
        ratio = greedy['win_rate'] / max(search['win_rate'], 0.001)
        speed_ratio = search['avg_time_ms'] / max(greedy['avg_time_ms'], 0.001)
        print(f"\n--- Comparison ---")
        print(f"Greedy/Search win rate ratio: {ratio:.3f} ({ratio*100:.1f}%)")
        print(f"Speed advantage: {speed_ratio:.0f}x faster")

        # Paired comparison (McNemar's)
        both_win = 0
        greedy_only = 0
        search_only = 0
        both_lose = 0

        # Re-run paired for McNemar's (expensive but precise)
        print("\nRunning paired seed comparison...")
        search_config = FullHopConfig(n_times=500, node_limit=100)
        for i in range(args.n_games):
            seed = args.seed_start + i
            g_won, _, _ = play_greedy_game(model, device, seed)
            s_won, _, _ = play_full_hop_game(seed, search_config)

            if g_won and s_won:
                both_win += 1
            elif g_won and not s_won:
                greedy_only += 1
            elif not g_won and s_won:
                search_only += 1
            else:
                both_lose += 1

        print(f"Both win: {both_win}, Greedy only: {greedy_only}, "
              f"Search only: {search_only}, Both lose: {both_lose}")

        # McNemar's test
        n_discordant = greedy_only + search_only
        if n_discordant > 0:
            chi2 = (greedy_only - search_only) ** 2 / n_discordant
            print(f"McNemar's chi2: {chi2:.2f} (>3.84 = significant at p<0.05)")


if __name__ == "__main__":
    main()
