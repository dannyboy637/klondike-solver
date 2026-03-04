// AlphaZero training CLI for Klondike Solitaire.
// Ported from scripts/train_alphazero.py in Phase R5.

use anyhow::Result;
use clap::Parser;
use klondike_nn::model::NetConfig;
use klondike_train::pipeline::{run_pipeline, PipelineConfig};
use klondike_train::selfplay::SelfPlayConfig;

#[derive(Parser)]
#[command(
    name = "train",
    about = "AlphaZero training for Klondike Solitaire (Rust)"
)]
struct Args {
    /// Total training iterations.
    #[arg(long, default_value_t = 100)]
    iterations: usize,

    /// Self-play games per iteration.
    #[arg(long, default_value_t = 200)]
    games_per_iter: usize,

    /// Determinizations per move (PIMC).
    #[arg(long, default_value_t = 5)]
    det: u32,

    /// MCTS simulations per determinization.
    #[arg(long, default_value_t = 200)]
    sims: u32,

    /// Hidden dimension of neural network.
    #[arg(long, default_value_t = 512)]
    hidden_dim: i64,

    /// Number of residual blocks.
    #[arg(long, default_value_t = 4)]
    n_blocks: i64,

    /// Learning rate.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// Training batch size.
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,

    /// Replay buffer capacity.
    #[arg(long, default_value_t = 200_000)]
    buffer_capacity: usize,

    /// Evaluate every N iterations.
    #[arg(long, default_value_t = 5)]
    eval_every: usize,

    /// Number of games per evaluation.
    #[arg(long, default_value_t = 1000)]
    eval_games: usize,

    /// Checkpoint directory.
    #[arg(long, default_value = "checkpoints/alphazero_rs")]
    checkpoint_dir: String,

    /// Path to warm-start weights (Rust VarStore .pt format).
    #[arg(long, default_value = "")]
    warm_start: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let selfplay = SelfPlayConfig {
        n_determinizations: args.det,
        mcts_simulations: args.sims,
        ..SelfPlayConfig::default()
    };

    let net_config = NetConfig {
        hidden_dim: args.hidden_dim,
        n_blocks: args.n_blocks,
        ..Default::default()
    };

    let config = PipelineConfig {
        total_iterations: args.iterations,
        games_per_iteration: args.games_per_iter,
        selfplay,
        batch_size: args.batch_size,
        lr: args.lr,
        buffer_capacity: args.buffer_capacity,
        eval_every: args.eval_every,
        eval_games: args.eval_games,
        checkpoint_dir: args.checkpoint_dir,
        net_config,
        warm_start: args.warm_start,
        ..Default::default()
    };

    println!("{}", "=".repeat(60));
    println!("AlphaZero Klondike Training (Rust)");
    println!("{}", "=".repeat(60));
    println!("Iterations:       {}", config.total_iterations);
    println!("Games/iter:       {}", config.games_per_iteration);
    println!("Determinizations: {}", config.selfplay.n_determinizations);
    println!("MCTS sims:        {}", config.selfplay.mcts_simulations);
    println!(
        "Model:            {}h x {}b",
        config.net_config.hidden_dim, config.net_config.n_blocks
    );
    println!("Batch size:       {}", config.batch_size);
    println!("Buffer capacity:  {}", config.buffer_capacity);
    println!(
        "Warm start:       {}",
        if config.warm_start.is_empty() {
            "None"
        } else {
            &config.warm_start
        }
    );
    println!("Checkpoint dir:   {}", config.checkpoint_dir);
    println!("{}", "=".repeat(60));

    run_pipeline(config)
}
