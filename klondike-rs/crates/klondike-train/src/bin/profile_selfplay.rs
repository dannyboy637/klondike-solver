// Final benchmark suite for klondike-rs.
// Measures throughput at various configs: micro-ops, self-play, HOP, and parallel scaling.

use std::time::Instant;

use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::encoding::{encode_state, OBSERVATION_DIM};
use klondike_core::env::KlondikeEnv;
use klondike_nn::inference::auto_device;
use klondike_nn::model::{KlondikeNet, NetConfig};
use klondike_search::eval_fn::{EvalFn, UniformEval};
use klondike_search::hop::{play_hop_game, HopConfig};
use klondike_search::mcts::{mcts_search, MctsConfig};
use klondike_train::inference_server::{create_inference_channel, InferenceServer};
use klondike_train::selfplay::{play_game, play_games_parallel, SelfPlayConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use tch::{nn, Device, Tensor};

/// Direct batch-1 neural eval (old approach — each thread calls GPU independently).
struct DirectNeuralEval<'a> {
    net: &'a KlondikeNet,
    device: Device,
}

unsafe impl Sync for DirectNeuralEval<'_> {}

impl EvalFn for DirectNeuralEval<'_> {
    fn evaluate(&self, env: &KlondikeEnv) -> ([f32; ACTION_SPACE_SIZE], f32) {
        let obs = env.get_observation();
        let mask = env.get_legal_mask();
        let obs_t = Tensor::from_slice(&obs)
            .reshape([1, OBSERVATION_DIM as i64])
            .to_device(self.device);
        let mask_t = Tensor::from_slice(&mask)
            .reshape([1, ACTION_SPACE_SIZE as i64])
            .to_device(self.device);
        let (log_policy, value) = tch::no_grad(|| self.net.forward(&obs_t, Some(&mask_t)));
        let policy_probs = log_policy.exp();
        let policy_cpu = policy_probs.to_device(Device::Cpu).squeeze_dim(0);
        let policy_data: Vec<f32> = Vec::try_from(&policy_cpu).expect("extract");
        let mut result = [0.0f32; ACTION_SPACE_SIZE];
        result.copy_from_slice(&policy_data);
        let val = value.double_value(&[0, 0]) as f32;
        (result, val)
    }
}

fn main() {
    println!("================================================================");
    println!("  klondike-rs Final Benchmark Suite");
    println!("================================================================\n");

    bench_micro_ops();
    bench_mcts();
    bench_selfplay_configs();
    bench_parallel_selfplay();
    bench_hop();
    bench_neural_batching();

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

fn bench_neural_batching() {
    let device = auto_device();
    println!("\n--- Neural Batching: Parallelism Scaling (device: {:?}) ---", device);

    let vs = nn::VarStore::new(device);
    let net_config = NetConfig::default();
    let net = KlondikeNet::new(&vs.root(), &net_config);
    let param_count = KlondikeNet::count_parameters(&vs);
    println!("  Model: {} parameters (5d x 200s config)", param_count);

    let selfplay_config = SelfPlayConfig {
        n_determinizations: 5,
        mcts_simulations: 200,
        max_moves: 500,
        dirichlet_frac: 0.0,
        ..SelfPlayConfig::default()
    };

    // Previous result: Direct batch-1 = 17.48s/game (20 games)
    println!("  Reference: Direct batch-1 = 17.48s/game (from prior run)\n");

    for n_games in [16, 32, 64, 128] {
        let seeds: Vec<u64> = (100..100 + n_games as u64).collect();
        let max_batch = if n_games <= 64 { 64 } else { 128 };

        print!("  [{:3} games, batch<={}] running... ", n_games, max_batch);

        let start = Instant::now();
        let games = std::thread::scope(|s| {
            let (client, request_rx) = create_inference_channel(max_batch);
            let mut server = InferenceServer::new(&net, device, request_rx, max_batch);

            let server_handle = s.spawn(move || {
                server.run();
            });

            let games = play_games_parallel(&selfplay_config, &client, &seeds);

            drop(client);
            server_handle.join().expect("server panicked");
            games
        });
        let elapsed = start.elapsed();

        let wins: usize = games.iter().filter(|g| g.won).count();
        let total_moves: usize = games.iter().map(|g| g.n_moves).sum();
        let wall_s = elapsed.as_secs_f64();
        let per_game = wall_s / n_games as f64;
        let per_move_ms = wall_s * 1000.0 / total_moves.max(1) as f64;
        let throughput = n_games as f64 / wall_s;
        let speedup = 17.48 / per_game;

        println!(
            "{:.1}s wall | {:.2}s/game | {:.1}ms/move | {:.1} games/sec | {:.1}x vs batch-1 | {}/{} wins | {} moves",
            wall_s, per_game, per_move_ms, throughput, speedup, wins, n_games, total_moves,
        );
    }
}
