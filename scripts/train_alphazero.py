#!/usr/bin/env python3
"""AlphaZero self-play training for Klondike Solitaire.

Usage:
  python scripts/train_alphazero.py                          # Default settings
  python scripts/train_alphazero.py --iterations 200         # More iterations
  python scripts/train_alphazero.py --games-per-iter 50      # Quick test run
  python scripts/train_alphazero.py --det 3 --sims 100       # Faster self-play
  python scripts/train_alphazero.py --small-model             # 590K param model
"""

import argparse
from klondike.alphazero.pipeline import PipelineConfig, run_pipeline
from klondike.alphazero.selfplay import SelfPlayConfig


def main():
    parser = argparse.ArgumentParser(description="AlphaZero training for Klondike")

    # Pipeline
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=1000)

    # Self-play
    parser.add_argument("--det", type=int, default=5, help="Determinizations per move")
    parser.add_argument("--sims", type=int, default=200, help="MCTS sims per determinization")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temp-moves", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=500)

    # Training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--batches-per-iter", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--min-buffer", type=int, default=5000, help="Min positions before training starts")

    # Model
    parser.add_argument("--small-model", action="store_true", help="Use 590K param model (256 hidden, 2 blocks)")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Hardware
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-amp", action="store_true", help="Disable FP16 mixed precision")

    # Warm start
    parser.add_argument("--supervised-weights", type=str, default="checkpoints/supervised_v2/best_supervised.pt")
    parser.add_argument("--no-warm-start", action="store_true", help="Start from random weights")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/alphazero")

    args = parser.parse_args()

    # Small model preset
    if args.small_model:
        args.hidden_dim = 256
        args.n_blocks = 2

    selfplay_config = SelfPlayConfig(
        n_determinizations=args.det,
        mcts_simulations=args.sims,
        c_puct=args.c_puct,
        temperature_moves=args.temp_moves,
        max_moves=args.max_moves,
    )

    config = PipelineConfig(
        games_per_iteration=args.games_per_iter,
        selfplay_config=selfplay_config,
        batch_size=args.batch_size,
        batches_per_iteration=args.batches_per_iter,
        lr=args.lr,
        weight_decay=args.weight_decay,
        buffer_capacity=args.buffer_capacity,
        min_buffer_size=args.min_buffer,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        checkpoint_dir=args.checkpoint_dir,
        total_iterations=args.iterations,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        device=args.device,
        use_amp=not args.no_amp,
        supervised_weights="" if args.no_warm_start else args.supervised_weights,
    )

    print("=" * 60)
    print("AlphaZero Klondike Training")
    print("=" * 60)
    print(f"Iterations: {config.total_iterations}")
    print(f"Games/iter: {config.games_per_iteration}")
    print(f"Determinizations: {selfplay_config.n_determinizations}")
    print(f"MCTS sims: {selfplay_config.mcts_simulations}")
    print(f"Model: {args.hidden_dim}h x {args.n_blocks}b")
    print(f"Buffer: {config.buffer_capacity:,}")
    print(f"AMP: {config.use_amp}")
    print(f"Warm start: {config.supervised_weights or 'None'}")
    print(f"Checkpoint: {config.checkpoint_dir}")
    print("=" * 60)

    run_pipeline(config)


if __name__ == "__main__":
    main()
