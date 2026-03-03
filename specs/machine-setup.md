# Machine Setup — Lonelybot Patches for 1M Game Benchmark

## Overview
The lonelybot submodule needs 3 local patches applied before building. These patches are NOT committed to the submodule — they must be applied manually on each machine.

All patches are against **commit 3e88e66** of the lonelybot submodule.

## Step 0: Clone and checkout

```bash
git clone --recurse-submodules https://github.com/dannyboy637/klondike-solver.git
cd klondike-solver

# Checkout the correct lonelybot commit
cd lonelybot
git checkout 3e88e66
cd ..
```

If already cloned, just make sure the submodule is at 3e88e66:
```bash
cd lonelybot && git checkout 3e88e66 && cd ..
```

## Patch 1: `lonelybot/src/hidden.rs`

### 1a. Fix `decode()` — add `update_map()` and `compute_first_layer_mask()`

Find this function (~line 166):
```rust
    pub fn decode(&mut self, mut hidden_encode: u16) {
        for i in 0..N_PILES {
            let n_options = u16::from(i) + 2;
            self.n_hidden[i as usize] = (hidden_encode % n_options) as u8;
            hidden_encode /= n_options;
        }
    }
```

Replace with:
```rust
    pub fn decode(&mut self, mut hidden_encode: u16) {
        for i in 0..N_PILES {
            let n_options = u16::from(i) + 2;
            self.n_hidden[i as usize] = (hidden_encode % n_options) as u8;
            hidden_encode /= n_options;
        }
        self.update_map();
        self.first_layer_mask = self.compute_first_layer_mask();
    }
```

### 1b. Add `shuffle_all()` method

Add this method at the end of `impl Hidden` (after the `shuffle()` method, before the closing `}`):

```rust
    /// Shuffle ALL hidden cards including the topmost (last) card per pile.
    /// Unlike `shuffle()`, this does not preserve any oracle information.
    /// Works by: shuffling deeper cards normally, then also shuffling the
    /// topmost cards among themselves so the solver doesn't know which card
    /// will be revealed next.
    pub fn shuffle_all<R: RngCore>(&mut self, rng: &mut R) {
        // First: shuffle the deeper cards (standard behavior)
        self.shuffle(rng);

        // Second: collect all topmost hidden cards, shuffle, redistribute
        let mut tops = ArrayVec::<Card, { N_PILES as usize }>::new();
        for pos in 0..N_PILES {
            if let Some((&last, _)) = self.get(pos).split_last() {
                tops.push(last);
            }
        }
        tops.shuffle(rng);

        let mut idx = 0;
        for pos in 0..N_PILES {
            let pile = self.get_mut(pos);
            if !pile.is_empty() {
                let len = pile.len();
                pile[len - 1] = tops[idx];
                idx += 1;
            }
        }
        self.update_map();
        self.first_layer_mask = self.compute_first_layer_mask();
    }
```

## Patch 2: `lonelybot/src/hop_solver.rs`

Find line 76 (inside the `for _ in 0..n_times` loop in `hop_solve_game`):
```rust
        gg.get_hidden_mut().shuffle(rng);
```

**No change needed** — this must stay as `shuffle()` (not `shuffle_all()`). The inner determinizations correctly preserve topmost cards; only the outer loop in `do_hop()` uses `shuffle_all()`.

## Patch 3: `lonelybot/lonecli/src/main.rs`

### 3a. Replace the `do_hop` function

Find the `do_hop` function (~line 162) and replace it entirely:

```rust
fn do_hop(seed: &Seed, draw_step: u8, verbose: bool) -> bool {
    const N_TIMES: usize = 500;
    const LIMIT: usize = 500;
    const MAX_MOVES: usize = 5000;

    let mut game = Solitaire::new(&shuffle(seed), draw_step);
    let mut rng = StdRng::seed_from_u64(seed.seed().as_u64());
    let mut move_count: usize = 0;

    while !game.is_win() {
        let mut gg = game.clone();
        gg.get_hidden_mut().shuffle_all(&mut rng);
        let best = pick_moves(
            &mut gg,
            &mut rng,
            N_TIMES,
            LIMIT,
            &DefaultTerminateSignal {},
        );
        let Some(best) = best else {
            if verbose {
                eprintln!("Lost");
            }
            return false;
        };
        if verbose {
            for m in &best {
                eprint!("{m}, ");
            }
            eprintln!();
        }
        for m in best {
            game.do_move(&m);
            move_count += 1;
            if move_count >= MAX_MOVES {
                if verbose {
                    eprintln!("Lost (max moves)");
                }
                return false;
            }
        }
    }
    if verbose {
        eprintln!("Solved");
    }
    true
}
```

Key changes from original:
- `N_TIMES: 3000 → 500` (fast params; use 3000 for heavy)
- `LIMIT: 1000 → 500` (fast params; use 1000 for heavy)
- `MAX_MOVES: 5000` added (prevents infinite loops on unlimited passes)
- `draw_step: u8` parameter added (original was hardcoded draw-3)
- `shuffle_all(&mut rng)` instead of `clear()` (removes oracle leak)
- `eprintln!` instead of `println!` (prevents stdout buffering when piped)

### 3b. Add `draw_step` to CLI Hop and HopLoop commands

Find the `Hop` and `HopLoop` variants in the `Commands` enum (~line 460):
```rust
    Hop {
        #[command(flatten)]
        seed: StringSeed,
    },
    HopLoop {
        #[command(flatten)]
        seed: StringSeed,
    },
```

Replace with:
```rust
    Hop {
        #[command(flatten)]
        seed: StringSeed,
        #[arg(default_value_t = 3)]
        draw_step: u8,
    },
    HopLoop {
        #[command(flatten)]
        seed: StringSeed,
        #[arg(default_value_t = 3)]
        draw_step: u8,
    },
```

### 3c. Update `main()` match arms for Hop and HopLoop

Find the match arms (~line 494):
```rust
        Commands::Hop { seed } => {
            do_hop(&seed.into(), true);
        }
        Commands::HopLoop { seed } => {
```

Replace both arms:
```rust
        Commands::Hop { seed, draw_step } => {
            do_hop(&seed.into(), *draw_step, true);
        }
        Commands::HopLoop { seed, draw_step } => {
```

And update the `do_hop` call inside HopLoop similarly (add `*draw_step` and `false` for verbose).

## Build

```bash
cd lonelybot
cargo build --release -p lonecli
```

The binary will be at `lonelybot/target/release/lonecli`.

## Verify

```bash
# Should solve and print "Solved" to stderr
./target/release/lonecli hop default 42 1 2>&1 | tail -1
# Expected: Solved
```

## Run the benchmark

Update `run_benchmark.py` line 19 to point to the correct binary path:
```python
LONECLI = "/path/to/klondike-solver/lonelybot/target/release/lonecli"
```

Then run:
```bash
# M4 MacBook (this machine)
python3 run_benchmark.py 1 0 250000 180 8 results_m4.txt

# M1 MacBook (home)
python3 run_benchmark.py 1 250000 250000 180 8 results_m1.txt

# PC Ryzen 5700X
python3 run_benchmark.py 1 500000 500000 180 16 results_pc.txt
```

## Combine results

Sum raw wins/losses/timeouts from all three output files, compute one Wilson CI on the totals.

## Parameter Reference

| Config | N_TIMES | LIMIT | Win Rate | Speed | Use For |
|--------|---------|-------|----------|-------|---------|
| Fast | 500 | 500 | ~48.0% | ~0.5s/game | 1M game benchmark |
| Heavy | 3000 | 1000 | ~49.0% | ~5.3s/game | Precise ceiling estimate |

To switch between fast/heavy, change `N_TIMES` and `LIMIT` in `do_hop()` and rebuild.
