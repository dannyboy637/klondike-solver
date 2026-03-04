#!/usr/bin/env python3
"""Train supervised policy+value net on search solver data."""
import argparse

import torch

from klondike.nn.model import KlondikeNet
from klondike.nn.trainer import TrainerConfig, train_supervised


def main():
    parser = argparse.ArgumentParser(description="Train supervised KlondikeNet")
    parser.add_argument("--data-dir", type=str, default="data/supervised")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/supervised")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=off)")
    args = parser.parse_args()

    model = KlondikeNet(hidden_dim=args.hidden_dim, n_blocks=args.n_blocks, dropout=args.dropout)
    print(f"Model: {model.count_parameters():,} parameters")

    config = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
    )

    model = train_supervised(model, args.data_dir, config)

    # Save final model
    torch.save(model.state_dict(), f"{args.checkpoint_dir}/final.pt")
    print(f"Final model saved to {args.checkpoint_dir}/final.pt")


if __name__ == "__main__":
    main()
