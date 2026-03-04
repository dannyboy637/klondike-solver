// Final benchmark suite for klondike-rs.
// Measures throughput at various configs: micro-ops, self-play, HOP, and parallel scaling.

use std::time::Instant;

use klondike_core::encoding::{encode_state, OBSERVATION_DIM};
use klondike_core::env::KlondikeEnv;
use klondike_search::eval_fn::UniformEval;
use klondike_search::hop::{play_hop_game, HopConfig};
use klondike_search::mcts::{mcts_search, MctsConfig};
use klondike_train::selfplay::{play_game, play_games_parallel, SelfPlayConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    println!("================================================================");
    println!("  klondike-rs Final Benchmark Suite");
    println!("================================================================\n");

    bench_micro_ops();
    bench_mcts();
    bench_selfplay_configs();
    bench_parallel_selfplay();
    bench_hop();

    println!("\n================================================================");
    println!("  Benchmark Complete");
    println!("================================================================");
}

fn bench_micro_ops() {
    println!("--- Micro-benchmarks ---");
    let env = KlondikeEnv::new(42, 1);

    // Env clone
    let n = 100_000;
    let start = Instant::now();
    for _ in 0..n {
        let _ = std::hint::black_box(env.clone());
    }
    let elapsed = start.elapsed();
    println!(
        "  env.clone():       {:.2}us  ({n} calls, {:.1}s)",
        elapsed.as_secs_f64() * 1e6 / n as f64,
        elapsed.as_secs_f64()
    );

    // Encoding
    let mut obs = [0.0f32; OBSERVATION_DIM];
    let start = Instant::now();
    for _ in 0..n {
        encode_state(env.game(), &mut obs);
        std::hint::black_box(&obs);
    }
    let elapsed = start.elapsed();
    println!(
        "  encode_state():    {:.2}us  ({n} calls, {:.1}s)",
        elapsed.as_secs_f64() * 1e6 / n as f64,
        elapsed.as_secs_f64()
    );

    // Legal mask
    let start = Instant::now();
    for _ in 0..n {
        let _ = std::hint::black_box(env.get_legal_mask());
    }
    let elapsed = start.elapsed();
    println!(
        "  get_legal_mask():  {:.2}us  ({n} calls, {:.1}s)",
        elapsed.as_secs_f64() * 1e6 / n as f64,
        elapsed.as_secs_f64()
    );

    // Legal actions
    let start = Instant::now();
    for _ in 0..n {
        let _ = std::hint::black_box(env.get_legal_actions());
    }
    let elapsed = start.elapsed();
    println!(
        "  get_legal_actions(): {:.2}us ({n} calls, {:.1}s)",
        elapsed.as_secs_f64() * 1e6 / n as f64,
        elapsed.as_secs_f64()
    );

    // Observation
    let start = Instant::now();
    for _ in 0..n {
        let _ = std::hint::black_box(env.get_observation());
    }
    let elapsed = start.elapsed();
    println!(
        "  get_observation(): {:.2}us  ({n} calls, {:.1}s)",
        elapsed.as_secs_f64() * 1e6 / n as f64,
        elapsed.as_secs_f64()
    );
    println!();
}

fn bench_mcts() {
    println!("--- MCTS Search (UniformEval) ---");
    let env = KlondikeEnv::new(42, 1);
    let eval = UniformEval;

    for sims in [50, 100, 200, 400] {
        let config = MctsConfig {
            n_simulations: sims,
            dirichlet_frac: 0.0,
            ..MctsConfig::default()
        };
        let n = 200;
        let start = Instant::now();
        for i in 0..n {
            let mut rng = SmallRng::seed_from_u64(i);
            let _ = std::hint::black_box(mcts_search(&env, &config, &eval, &mut rng));
        }
        let elapsed = start.elapsed();
        println!(
            "  {} sims: {:.3}ms/search  ({n} calls, {:.2}s)",
            sims,
            elapsed.as_secs_f64() * 1000.0 / n as f64,
            elapsed.as_secs_f64()
        );
    }
    println!();
}

fn bench_selfplay_configs() {
    println!("--- Self-play Configs (UniformEval, sequential) ---");
    let eval = UniformEval;

    let configs = [
        ("2d x 30s", 2, 30, 20),
        ("3d x 50s", 3, 50, 20),
        ("5d x 200s", 5, 200, 5),
    ];

    for (label, det, sims, n_games) in configs {
        let config = SelfPlayConfig {
            n_determinizations: det,
            mcts_simulations: sims,
            max_moves: 500,
            dirichlet_frac: 0.0,
            ..SelfPlayConfig::default()
        };
        let start = Instant::now();
        let mut total_moves = 0;
        let mut wins = 0;
        for seed in 0..n_games as u64 {
            let game = play_game(&config, &eval, seed);
            total_moves += game.n_moves;
            if game.won {
                wins += 1;
            }
        }
        let elapsed = start.elapsed();
        let avg_moves = total_moves as f64 / n_games as f64;
        println!(
            "  {label}: {:.2}ms/game, {:.3}ms/move, avg {avg_moves:.0} moves/game, {wins}/{n_games} wins  ({:.2}s total)",
            elapsed.as_secs_f64() * 1000.0 / n_games as f64,
            elapsed.as_secs_f64() * 1000.0 / total_moves.max(1) as f64,
            elapsed.as_secs_f64()
        );
    }
    println!();
}

fn bench_parallel_selfplay() {
    println!("--- Parallel Self-play (UniformEval, rayon) ---");
    let eval = UniformEval;
    let config = SelfPlayConfig {
        n_determinizations: 3,
        mcts_simulations: 50,
        max_moves: 500,
        dirichlet_frac: 0.0,
        ..SelfPlayConfig::default()
    };

    let n_games = 100;
    let seeds: Vec<u64> = (0..n_games).collect();
    let start = Instant::now();
    let games = play_games_parallel(&config, &eval, &seeds);
    let elapsed = start.elapsed();
    let wins: usize = games.iter().filter(|g| g.won).count();
    let total_moves: usize = games.iter().map(|g| g.n_moves).sum();
    let games_per_sec = n_games as f64 / elapsed.as_secs_f64();
    println!(
        "  {n_games} games (3d x 50s): {:.0} games/sec, {wins}/{n_games} wins, {:.0} moves total ({:.2}s)",
        games_per_sec,
        total_moves,
        elapsed.as_secs_f64()
    );
    println!();
}

fn bench_hop() {
    println!("--- HOP Solver (Lonelybot, nt=500, lim=100) ---");
    let config = HopConfig {
        n_times: 500,
        node_limit: 100,
        max_moves: 500,
        draw_step: 1,
    };
    let n_games = 20;
    let start = Instant::now();
    let mut wins = 0;
    for seed in 0..n_games as u64 {
        let (won, _) = play_hop_game(seed, &config);
        if won {
            wins += 1;
        }
    }
    let elapsed = start.elapsed();
    println!(
        "  {n_games} games: {:.0}ms/game, {wins}/{n_games} wins ({:.2}s total)",
        elapsed.as_secs_f64() * 1000.0 / n_games as f64,
        elapsed.as_secs_f64()
    );
}
