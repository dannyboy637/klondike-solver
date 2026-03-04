#!/usr/bin/env python3
"""Dump neural net parity data for Rust verification.

Loads the supervised_v2 checkpoint, runs 10 deterministic inputs through the model,
and saves the inputs + outputs to JSON for comparison with the Rust implementation.
"""

import json
import sys
import torch
import numpy as np

sys.path.insert(0, ".")
from klondike.nn.model import KlondikeNet

CHECKPOINT = "checkpoints/supervised_v2/best_supervised.pt"
OUTPUT = "klondike-rs/crates/klondike-nn/tests/nn_parity_data.json"
N_SAMPLES = 10
SEED = 12345


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model
    model = KlondikeNet(use_auxiliary=True, dropout=0.0)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    # Handle both raw state_dict and wrapped checkpoint formats
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Print state_dict keys for reference
    print("Python state_dict keys:")
    for k, v in model.state_dict().items():
        print(f"  {k}: {list(v.shape)}")
    print(f"\nTotal parameters: {model.count_parameters()}")

    # Generate deterministic test inputs
    rng = np.random.RandomState(SEED)
    samples = []
    for i in range(N_SAMPLES):
        # Random observation in [0, 1]
        obs = rng.rand(757).astype(np.float32)
        # Random legal mask (at least 1 legal action)
        mask = np.zeros(260, dtype=np.float32)
        n_legal = rng.randint(1, 20)
        legal_actions = rng.choice(260, size=n_legal, replace=False)
        mask[legal_actions] = 1.0

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        with torch.no_grad():
            result = model(obs_t, mask_t)

        log_policy = result["policy"].squeeze(0).numpy()
        value = result["value"].squeeze().item()

        samples.append({
            "observation": obs.tolist(),
            "legal_mask": mask.tolist(),
            "log_policy": log_policy.tolist(),
            "value": value,
        })

    data = {
        "checkpoint": CHECKPOINT,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "param_count": model.count_parameters(),
        "state_dict_keys": list(model.state_dict().keys()),
        "samples": samples,
    }

    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nWrote {N_SAMPLES} samples to {OUTPUT}")
    print(f"Value range: [{min(s['value'] for s in samples):.6f}, {max(s['value'] for s in samples):.6f}]")


if __name__ == "__main__":
    main()
