// Benchmark CLI for running N-game evaluations.
// Supports Lonelybot HOP, PIMC+UniformEval, and neural greedy modes.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::env::KlondikeEnv;
use klondike_eval::benchmark::{run_benchmark, run_benchmark_sequential, BenchmarkConfig};
use klondike_eval::stats::{sprt_test, SprtResult};
use klondike_nn::inference::BatchedInference;
use klondike_nn::model::NetConfig;
use klondike_search::eval_fn::{EvalFn, UniformEval};
use klondike_search::hop::{play_hop_game, HopConfig};
use klondike_search::pimc::{pimc_search, PimcConfig};

#[derive(Parser)]
#[command(
    name = "benchmark",
    about = "Benchmark a Klondike solver over N games with Wilson CI and SPRT."
)]
struct Args {
    /// Number of games to play.
    #[arg(long, default_value_t = 1000)]
    games: usize,

    /// Starting seed.
    #[arg(long, default_value_t = 0)]
    seed_start: u64,

    /// Draw step (1 = draw-1, 3 = draw-3).
    #[arg(long, default_value_t = 1)]
    draw_step: u8,

    /// PIMC determinizations per move.
    #[arg(long, default_value_t = 5)]
    det: u32,

    /// MCTS simulations per determinization.
    #[arg(long, default_value_t = 200)]
    sims: u32,

    /// Maximum moves per game.
    #[arg(long, default_value_t = 500)]
    max_moves: u32,

    /// Print progress every N games.
    #[arg(long, default_value_t = 100)]
    log_every: usize,

    /// Number of rayon threads (0 = auto).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Path to neural network checkpoint (safetensors format).
    /// If not provided, uses UniformEval with PIMC search (HOP baseline).
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Hidden dimension (for loading checkpoint).
    #[arg(long, default_value_t = 512)]
    hidden_dim: i64,

    /// Number of residual blocks (for loading checkpoint).
    #[arg(long, default_value_t = 4)]
    n_blocks: i64,

    /// Use Lonelybot's built-in HOP solver (the 54.96% baseline).
    /// Overrides --det/--sims (uses --n-times/--node-limit instead).
    #[arg(long)]
    hop: bool,

    /// HOP: UCB1 bandit iterations per turn (only with --hop).
    #[arg(long, default_value_t = 500)]
    n_times: usize,

    /// HOP: max search nodes per HOP solve (only with --hop).
    #[arg(long, default_value_t = 100)]
    node_limit: usize,

    /// SPRT null hypothesis (p0).
    #[arg(long, default_value_t = 0.50)]
    sprt_p0: f64,

    /// SPRT alternative hypothesis (p1).
    #[arg(long, default_value_t = 0.52)]
    sprt_p1: f64,
}

/// Play a single game using PIMC search with the given eval function.
/// Returns true if the game was won.
fn play_pimc_game(
    seed: u64,
    draw_step: u8,
    pimc_config: &PimcConfig,
    eval_fn: &dyn EvalFn,
    max_moves: u32,
) -> bool {
    let mut env = KlondikeEnv::new(seed, draw_step);
    let mut move_num: u32 = 0;
    let mut seen_states = HashSet::new();

    while !env.is_done() && move_num < max_moves {
        // Cycle detection
        let hash = env.encode_state_hash();
        if seen_states.contains(&hash) {
            break;
        }
        seen_states.insert(hash);

        let result = pimc_search(&env, pimc_config, eval_fn, seed, move_num);
        if result.action < 0 {
            break;
        }

        match env.step(result.action as usize) {
            Ok(_) => {}
            Err(_) => break,
        }
        move_num += 1;
    }

    env.game().is_win()
}

/// Play a single game using greedy neural policy (argmax over legal actions).
/// Returns true if the game was won.
fn play_greedy_game(
    seed: u64,
    draw_step: u8,
    infer: &mut BatchedInference,
    max_moves: u32,
) -> bool {
    let mut env = KlondikeEnv::new(seed, draw_step);
    let mut move_count: u32 = 0;
    let mut seen_states = HashSet::new();

    while !env.is_done() && move_count < max_moves {
        let hash = env.encode_state_hash();
        if seen_states.contains(&hash) {
            break;
        }
        seen_states.insert(hash);

        let obs = env.get_observation();
        let mask = env.get_legal_mask();
        let (policy, _value) = infer.evaluate_single(&obs, &mask);

        // Argmax over legal actions
        let mut best_action: i32 = -1;
        let mut best_prob: f32 = -1.0;
        for i in 0..ACTION_SPACE_SIZE {
            if mask[i] > 0.0 && policy[i] > best_prob {
                best_prob = policy[i];
                best_action = i as i32;
            }
        }

        if best_action < 0 {
            break;
        }

        match env.step(best_action as usize) {
            Ok(_) => {}
            Err(_) => break,
        }
        move_count += 1;
    }

    env.game().is_win()
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Configure rayon thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok(); // Ignore error if already initialized
    }

    let config = BenchmarkConfig {
        n_games: args.games,
        seed_start: args.seed_start,
        log_every: args.log_every,
    };

    println!("{}", "=".repeat(60));
    println!("Klondike Benchmark");
    println!("{}", "=".repeat(60));
    println!("Games:            {}", args.games);
    println!("Seed start:       {}", args.seed_start);
    println!("Draw step:        {}", args.draw_step);
    println!("Max moves:        {}", args.max_moves);

    let result = if let Some(ref checkpoint_path) = args.checkpoint {
        // Neural greedy mode
        let net_config = NetConfig {
            hidden_dim: args.hidden_dim,
            n_blocks: args.n_blocks,
            ..Default::default()
        };
        println!("Mode:             Neural greedy");
        println!("Checkpoint:       {}", checkpoint_path.display());
        println!(
            "Model:            {}h x {}b",
            args.hidden_dim, args.n_blocks
        );
        println!("{}", "=".repeat(60));

        let mut infer = BatchedInference::load(&net_config, checkpoint_path)?;
        let draw_step = args.draw_step;
        let max_moves = args.max_moves;

        // Neural inference uses GPU — run sequentially (BatchedInference needs &mut self)
        run_benchmark_sequential(&config, &mut |seed| {
            play_greedy_game(seed, draw_step, &mut infer, max_moves)
        })
    } else if args.hop {
        // Lonelybot HOP solver (the 54.96% baseline)
        let hop_config = HopConfig {
            n_times: args.n_times,
            node_limit: args.node_limit,
            max_moves: args.max_moves,
            draw_step: args.draw_step,
        };
        println!("Mode:             Lonelybot HOP");
        println!("n_times:          {}", args.n_times);
        println!("node_limit:       {}", args.node_limit);
        println!("{}", "=".repeat(60));

        run_benchmark(&config, &|seed| {
            let (won, _) = play_hop_game(seed, &hop_config);
            won
        })
    } else {
        // PIMC + UniformEval (random baseline)
        let pimc_config = PimcConfig {
            n_determinizations: args.det,
            mcts_simulations: args.sims,
            temperature: 0.05,   // Near-greedy for evaluation
            dirichlet_frac: 0.0, // No noise for evaluation
            ..PimcConfig::default()
        };
        println!("Mode:             PIMC + UniformEval");
        println!("Determinizations: {}", args.det);
        println!("MCTS sims:        {}", args.sims);
        println!("{}", "=".repeat(60));

        let draw_step = args.draw_step;
        let max_moves = args.max_moves;

        run_benchmark(&config, &|seed| {
            let eval = UniformEval;
            play_pimc_game(seed, draw_step, &pimc_config, &eval, max_moves)
        })
    };

    // Final report
    println!();
    println!("{}", "=".repeat(60));
    println!("RESULTS");
    println!("{}", "=".repeat(60));
    println!(
        "Win rate:    {}/{} = {:.2}%",
        result.wins,
        result.total,
        result.win_rate * 100.0
    );
    println!(
        "Wilson 95%:  ({:.2}%, {:.2}%)",
        result.ci_lower * 100.0,
        result.ci_upper * 100.0
    );

    let (sprt, log_lr) = sprt_test(
        result.wins,
        result.total,
        args.sprt_p0,
        args.sprt_p1,
        0.05,
        0.05,
    );
    let sprt_str = match sprt {
        SprtResult::AcceptH0 => format!("Accept H0 (win_rate <= {:.0}%)", args.sprt_p0 * 100.0),
        SprtResult::AcceptH1 => format!("Accept H1 (win_rate >= {:.0}%)", args.sprt_p1 * 100.0),
        SprtResult::Continue => "Inconclusive (need more games)".to_string(),
    };
    println!("SPRT:        {} (log_lr={:.3})", sprt_str, log_lr);
    println!(
        "Time:        {:.1}ms/game, {:.1}s total",
        result.avg_time_ms, result.total_time_s
    );
    println!("{}", "=".repeat(60));

    Ok(())
}
