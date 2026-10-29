// Full AlphaZero training loop: selfplay -> buffer -> train -> eval -> checkpoint.
// Ported from klondike/alphazero/pipeline.py in Phase R5.
//
// Neural self-play Option A: sequential games, single-evaluation per MCTS leaf.
// Each game completes fully before the next. Simpler, sufficient for initial runs.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::encoding::OBSERVATION_DIM;
use klondike_core::env::KlondikeEnv;
use klondike_eval::stats::wilson_ci;
use klondike_nn::inference::auto_device;
use klondike_nn::model::{KlondikeNet, NetConfig};
use klondike_search::eval_fn::EvalFn;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::buffer::ReplayBuffer;
use crate::inference_server::{create_inference_channel, InferenceServer};
use crate::selfplay::{play_games_parallel, SelfPlayConfig};

/// Pipeline configuration.
pub struct PipelineConfig {
    // Self-play
    pub games_per_iteration: usize,
    pub selfplay: SelfPlayConfig,

    // Training
    pub batch_size: usize,
    pub batches_per_iteration: usize,
    pub lr: f64,
    pub weight_decay: f64,

    // Buffer
    pub buffer_capacity: usize,
    pub min_buffer_size: usize,

    // Evaluation — MODEL SELECTION set (disjoint from Phase 6 final test sets)
    pub eval_every: usize,
    pub eval_games: usize,
    pub eval_seeds_start: u64,

    // Checkpoint
    pub checkpoint_dir: String,
    pub total_iterations: usize,

    // Model architecture
    pub net_config: NetConfig,

    // Warm start (path to weights file saved by Rust VarStore::save)
    pub warm_start: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            games_per_iteration: 200,
            selfplay: SelfPlayConfig::default(),
            batch_size: 1024,
            batches_per_iteration: 20,
            lr: 1e-3,
            weight_decay: 1e-4,
            buffer_capacity: 200_000,
            min_buffer_size: 5000,
            eval_every: 5,
            eval_games: 1000,
            eval_seeds_start: 5_000_000,
            checkpoint_dir: "checkpoints/alphazero_rs".into(),
            total_iterations: 100,
            net_config: NetConfig::default(),
            warm_start: String::new(),
        }
    }
}

/// Training log entry — JSON-compatible with Python monitor_training.py.
#[derive(Serialize, Deserialize)]
struct IterationLog {
    iteration: usize,
    selfplay_win_rate: f64,
    selfplay_positions: usize,
    avg_game_time: f64,
    policy_loss: f64,
    value_loss: f64,
    buffer_size: usize,
    iter_time: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eval_win_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eval_ci_lower: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eval_ci_upper: Option<f64>,
}

/// Checkpoint metadata (model weights saved separately via VarStore).
#[derive(Serialize, Deserialize)]
struct CheckpointMeta {
    iteration: usize,
    best_eval_rate: f64,
}

/// Neural network evaluation function for MCTS leaf evaluation.
/// Wraps a KlondikeNet reference for single-position inference.
struct NeuralEvalFn<'a> {
    net: &'a KlondikeNet,
    device: Device,
}

// SAFETY: PyTorch model inference is thread-safe for read-only access.
// During self-play, weights are frozen (no_grad), and each thread creates
// its own input/output tensors. The raw pointer in tch::Tensor wraps a
// reference-counted C++ at::Tensor that supports concurrent reads.
unsafe impl Sync for NeuralEvalFn<'_> {}

impl EvalFn for NeuralEvalFn<'_> {
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
        let policy_data: Vec<f32> =
            Vec::try_from(&policy_cpu).expect("policy tensor extraction failed");

        let mut result_policy = [0.0f32; ACTION_SPACE_SIZE];
        result_policy.copy_from_slice(&policy_data);

        let val: f64 = value.double_value(&[0, 0]);

        (result_policy, val as f32)
    }
}

/// Run the full AlphaZero training pipeline.
///
/// Loop:
/// 1. Self-play: generate games with current model via PC-PIMC
/// 2. Add positions to replay buffer
/// 3. Train: sample mini-batches from buffer, update model
/// 4. Evaluate: periodically test greedy policy on fixed seeds
/// 5. Checkpoint: save model, optimizer state is lost (acceptable for research)
pub fn run_pipeline(config: PipelineConfig) -> anyhow::Result<()> {
    let device = auto_device();
    println!("Device: {:?}", device);

    // Initialize model
    let mut vs = nn::VarStore::new(device);
    let net = KlondikeNet::new(&vs.root(), &config.net_config);
    let param_count = KlondikeNet::count_parameters(&vs);
    println!("Model: {} parameters", param_count);

    let mut opt = nn::AdamW {
        wd: config.weight_decay,
        ..Default::default()
    }
    .build(&vs, config.lr)?;

    let mut buffer = ReplayBuffer::new(config.buffer_capacity);
    let checkpoint_path = Path::new(&config.checkpoint_dir);
    fs::create_dir_all(checkpoint_path)?;

    let mut start_iteration = 0;
    let mut best_eval_rate = 0.0f64;
    let mut training_log: Vec<IterationLog> = Vec::new();
    let mut rng = StdRng::seed_from_u64(42);

    // Try to resume from checkpoint
    let meta_path = checkpoint_path.join("metadata.json");
    let model_path = checkpoint_path.join("model_latest.safetensors");

    if model_path.exists() && meta_path.exists() {
        vs.load(&model_path)?;
        let meta: CheckpointMeta = serde_json::from_str(&fs::read_to_string(&meta_path)?)?;
        start_iteration = meta.iteration + 1;
        best_eval_rate = meta.best_eval_rate;

        let buffer_bin = checkpoint_path.join("buffer.bin");
        let buffer_json = checkpoint_path.join("buffer.json");
        if buffer_bin.exists() {
            buffer = ReplayBuffer::load(&buffer_bin)?;
        } else if buffer_json.exists() {
            buffer = ReplayBuffer::load(&buffer_json)?;
        }

        let log_path = checkpoint_path.join("training_log.json");
        if log_path.exists() {
            training_log = serde_json::from_str(&fs::read_to_string(&log_path)?)?;
        }

        println!(
            "Resumed from iteration {}, buffer={}, best_eval={:.4}",
            start_iteration,
            buffer.len(),
            best_eval_rate
        );
    } else {
        // No model checkpoint — check for buffer-only warm start
        let buffer_bin = checkpoint_path.join("buffer.bin");
        let buffer_json = checkpoint_path.join("buffer.json");
        if buffer_bin.exists() {
            buffer = ReplayBuffer::load(&buffer_bin)?;
        } else if buffer_json.exists() {
            buffer = ReplayBuffer::load(&buffer_json)?;
            println!(
                "Loaded existing buffer ({} positions), starting with fresh weights",
                buffer.len()
            );
        }

        if !config.warm_start.is_empty() {
            let ws_path = Path::new(&config.warm_start);
            if ws_path.exists() {
                vs.load(ws_path)?;
                println!("Initialized from warm start: {}", config.warm_start);
            }
        }
    }

    for iteration in start_iteration..config.total_iterations {
        let iter_start = Instant::now();

        // --- 1. Self-Play (parallel via rayon, batched GPU inference) ---
        let seeds: Vec<u64> = (0..config.games_per_iteration)
            .map(|g| (iteration * config.games_per_iteration + g) as u64)
            .collect();
        let all_games = std::thread::scope(|s| {
            let (client, request_rx) = create_inference_channel(64);

            // Construct server here so its unsafe Send impl covers &net
            let mut server = InferenceServer::new(&net, device, request_rx, 64);

            // GPU inference server thread — batches requests from rayon workers
            let server_handle = s.spawn(move || {
                server.run();
            });

            let games = play_games_parallel(&config.selfplay, &client, &seeds);

            // Drop client → closes channel → server exits
            drop(client);
            server_handle.join().expect("inference server thread panicked");
            games
        });

        let mut games_won = 0usize;
        let mut total_positions = 0usize;
        let mut total_game_time_s = 0.0f64;

        for game_data in all_games {
            if game_data.n_moves > 0 {
                buffer.add_game(game_data.samples);
            }

            if game_data.won {
                games_won += 1;
            }
            total_positions += game_data.n_moves;
            total_game_time_s += game_data.wall_time_ms / 1000.0;
        }

        let selfplay_rate = games_won as f64 / config.games_per_iteration as f64;
        let avg_game_time = total_game_time_s / config.games_per_iteration as f64;

        // --- 2. Train ---
        let (avg_p_loss, avg_v_loss) = if buffer.len() >= config.min_buffer_size {
            train_batches(
                &net,
                &mut opt,
                &buffer,
                config.batch_size,
                config.batches_per_iteration,
                device,
                &mut rng,
            )
        } else {
            (0.0, 0.0)
        };

        let iter_time = iter_start.elapsed().as_secs_f64();

        let mut iter_log = IterationLog {
            iteration,
            selfplay_win_rate: selfplay_rate,
            selfplay_positions: total_positions,
            avg_game_time,
            policy_loss: avg_p_loss,
            value_loss: avg_v_loss,
            buffer_size: buffer.len(),
            iter_time,
            eval_win_rate: None,
            eval_ci_lower: None,
            eval_ci_upper: None,
        };

        println!(
            "Iter {:3}: selfplay={:.3} ({}/{}), pos={}, buf={}, \
             p_loss={:.4}, v_loss={:.4}, avg_game={:.1}s, iter={:.0}s",
            iteration,
            selfplay_rate,
            games_won,
            config.games_per_iteration,
            total_positions,
            buffer.len(),
            avg_p_loss,
            avg_v_loss,
            avg_game_time,
            iter_time
        );

        // --- 3. Evaluate ---
        if (iteration + 1) % config.eval_every == 0 {
            let (wins, total) = evaluate_greedy(
                &net,
                device,
                config.eval_games,
                config.eval_seeds_start,
                config.selfplay.draw_step,
            );
            let (win_rate, ci_lower, ci_upper) = wilson_ci(wins, total, 0.95);

            println!(
                "  EVAL: {}/{} = {:.4}, CI=({:.4}, {:.4})",
                wins, total, win_rate, ci_lower, ci_upper
            );

            iter_log.eval_win_rate = Some(win_rate);
            iter_log.eval_ci_lower = Some(ci_lower);
            iter_log.eval_ci_upper = Some(ci_upper);

            if win_rate > best_eval_rate {
                best_eval_rate = win_rate;
                vs.save(checkpoint_path.join("model_best.safetensors"))?;
                println!("  New best: {:.4}", best_eval_rate);
            }
        }

        // --- 4. Checkpoint ---
        vs.save(checkpoint_path.join("model_latest.safetensors"))?;
        let meta = CheckpointMeta {
            iteration,
            best_eval_rate,
        };
        fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
        buffer.save(&checkpoint_path.join("buffer.bin"))?;

        training_log.push(iter_log);
        fs::write(
            checkpoint_path.join("training_log.json"),
            serde_json::to_string_pretty(&training_log)?,
        )?;
    }

    println!("\nTraining complete. Best eval rate: {:.4}", best_eval_rate);
    println!("Checkpoints in: {}", config.checkpoint_dir);

    Ok(())
}

/// Run training batches from the replay buffer.
///
/// Returns (avg_policy_loss, avg_value_loss).
fn train_batches(
    net: &KlondikeNet,
    opt: &mut nn::Optimizer,
    buffer: &ReplayBuffer,
    batch_size: usize,
    n_batches: usize,
    device: Device,
    rng: &mut impl Rng,
) -> (f64, f64) {
    if n_batches == 0 {
        return (0.0, 0.0);
    }

    let mut total_policy_loss = 0.0f64;
    let mut total_value_loss = 0.0f64;

    for _ in 0..n_batches {
        let batch = buffer.sample_batch(batch_size, rng);
        let n = batch.len() as i64;
        if n == 0 {
            continue;
        }

        // Build tensors from batch
        let obs_flat: Vec<f32> = batch
            .iter()
            .flat_map(|s| s.observation.iter().copied())
            .collect();
        let policy_flat: Vec<f32> = batch
            .iter()
            .flat_map(|s| s.policy.iter().copied())
            .collect();
        let value_flat: Vec<f32> = batch.iter().map(|s| s.value).collect();
        let mask_flat: Vec<f32> = batch
            .iter()
            .flat_map(|s| s.legal_mask.iter().copied())
            .collect();

        let obs_t = Tensor::from_slice(&obs_flat)
            .reshape([n, OBSERVATION_DIM as i64])
            .to_device(device);
        let target_policy = Tensor::from_slice(&policy_flat)
            .reshape([n, ACTION_SPACE_SIZE as i64])
            .to_device(device);
        let target_value = Tensor::from_slice(&value_flat).to_device(device);
        let mask_t = Tensor::from_slice(&mask_flat)
            .reshape([n, ACTION_SPACE_SIZE as i64])
            .to_device(device);

        // Forward pass (with gradients for training)
        let (log_policy, value) = net.forward(&obs_t, Some(&mask_t));

        // Policy loss: cross-entropy = -sum(target * log_policy, dim=-1).mean()
        let per_sample_ce =
            (target_policy * &log_policy).sum_dim_intlist(&[-1i64][..], false, Kind::Float);
        let policy_loss = -per_sample_ce.mean(Kind::Float);

        // Value loss: MSE
        let value_pred = value.squeeze_dim(1);
        let diff = &value_pred - &target_value;
        let value_loss = (&diff * &diff).mean(Kind::Float);

        // Extract scalar values before backward frees the graph
        let p_val = policy_loss.double_value(&[]);
        let v_val = value_loss.double_value(&[]);

        // Combined loss
        let loss = &policy_loss + &value_loss;

        // Backward + per-element gradient clip to [-1, 1] + optimizer step
        opt.backward_step_clip(&loss, 1.0);

        total_policy_loss += p_val;
        total_value_loss += v_val;
    }

    (
        total_policy_loss / n_batches as f64,
        total_value_loss / n_batches as f64,
    )
}

/// Evaluate model with greedy policy on fixed seeds.
///
/// Uses argmax action selection (no search) for speed.
/// This is the fast eval for tracking trends, not for final claims.
fn evaluate_greedy(
    net: &KlondikeNet,
    device: Device,
    n_games: usize,
    seed_start: u64,
    draw_step: u8,
) -> (usize, usize) {
    let eval_fn = NeuralEvalFn { net, device };
    let mut wins = 0usize;

    for i in 0..n_games {
        let seed = seed_start + i as u64;
        let mut env = KlondikeEnv::new(seed, draw_step);
        let mut move_count = 0usize;
        let mut seen_states = HashSet::new();

        while !env.is_done() {
            let legal = env.get_legal_actions();
            if legal.is_empty() {
                break;
            }

            let state_hash = env.encode_state_hash();
            if seen_states.contains(&state_hash) {
                break;
            }
            seen_states.insert(state_hash);

            let (policy, _) = eval_fn.evaluate(&env);
            let action = *legal
                .iter()
                .max_by(|&&a, &&b| {
                    policy[a]
                        .partial_cmp(&policy[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("legal should be non-empty");

            match env.step(action) {
                Ok(_) => {}
                Err(_) => break,
            }

            move_count += 1;
            if move_count > 500 {
                break;
            }
        }

        if env.game().is_win() {
            wins += 1;
        }

        if n_games >= 100 && (i + 1) % 100 == 0 {
            let (win_rate, ci_lower, ci_upper) = wilson_ci(wins, i + 1, 0.95);
            println!(
                "  eval {}/{}: {}/{} = {:.4} CI({:.4}, {:.4})",
                i + 1,
                n_games,
                wins,
                i + 1,
                win_rate,
                ci_lower,
                ci_upper
            );
        }
    }

    (wins, n_games)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.games_per_iteration, 200);
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.total_iterations, 100);
        assert_eq!(config.eval_seeds_start, 5_000_000);
        assert_eq!(config.buffer_capacity, 200_000);
    }

    #[test]
    fn test_neural_eval_fn() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig {
            hidden_dim: 64,
            n_blocks: 1,
            ..Default::default()
        };
        let net = KlondikeNet::new(&vs.root(), &config);
        let eval_fn = NeuralEvalFn {
            net: &net,
            device: Device::Cpu,
        };

        let env = KlondikeEnv::new(42, 1);
        let (policy, value) = eval_fn.evaluate(&env);

        // Policy should sum to ~1.0
        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "policy sum {} should be ~1.0",
            sum
        );

        // Value should be in [0, 1]
        assert!((0.0..=1.0).contains(&value), "value {} out of range", value);
    }

    #[test]
    fn test_pipeline_smoke() {
        let dir = std::env::temp_dir().join("klondike_pipeline_smoke");
        let _ = fs::remove_dir_all(&dir);

        let ckpt_dir = dir.to_string_lossy().to_string();
        let config = PipelineConfig {
            total_iterations: 2,
            games_per_iteration: 3,
            selfplay: SelfPlayConfig {
                n_determinizations: 2,
                mcts_simulations: 10,
                max_moves: 50,
                dirichlet_frac: 0.0,
                ..SelfPlayConfig::default()
            },
            batch_size: 32,
            batches_per_iteration: 2,
            min_buffer_size: 10,
            eval_every: 2,
            eval_games: 5,
            net_config: NetConfig {
                hidden_dim: 64,
                n_blocks: 1,
                ..Default::default()
            },
            checkpoint_dir: ckpt_dir.clone(),
            ..Default::default()
        };

        run_pipeline(config).expect("pipeline should complete");

        // Verify training log exists and is valid JSON
        let log_path = Path::new(&ckpt_dir).join("training_log.json");
        let log_str = fs::read_to_string(&log_path).expect("training log should exist");
        let log: Vec<IterationLog> = serde_json::from_str(&log_str).expect("valid JSON");
        assert_eq!(log.len(), 2, "should have 2 iteration entries");

        // Iteration 1 should have eval results (eval_every=2)
        assert!(
            log[1].eval_win_rate.is_some(),
            "iteration 1 should have eval"
        );

        // Verify model checkpoint exists
        assert!(Path::new(&ckpt_dir).join("model_latest.safetensors").exists());

        // Verify metadata
        let meta_str = fs::read_to_string(Path::new(&ckpt_dir).join("metadata.json"))
            .expect("metadata should exist");
        let meta: CheckpointMeta = serde_json::from_str(&meta_str).expect("valid metadata JSON");
        assert_eq!(meta.iteration, 1);

        // Cleanup
        let _ = fs::remove_dir_all(&ckpt_dir);
    }

    #[test]
    fn test_training_log_json_compat() {
        // Verify that the training log format matches what Python expects
        let entry = IterationLog {
            iteration: 0,
            selfplay_win_rate: 0.1,
            selfplay_positions: 150,
            avg_game_time: 1.5,
            policy_loss: 0.94,
            value_loss: 0.04,
            buffer_size: 150,
            iter_time: 30.0,
            eval_win_rate: None,
            eval_ci_lower: None,
            eval_ci_upper: None,
        };

        let json = serde_json::to_string(&entry).expect("serialize");
        // Should NOT contain eval fields when None
        assert!(!json.contains("eval_win_rate"));
        assert!(!json.contains("eval_ci_lower"));

        // With eval
        let entry_with_eval = IterationLog {
            eval_win_rate: Some(0.18),
            eval_ci_lower: Some(0.10),
            eval_ci_upper: Some(0.26),
            ..entry
        };

        let json2 = serde_json::to_string(&entry_with_eval).expect("serialize");
        assert!(json2.contains("eval_win_rate"));
        assert!(json2.contains("0.18"));

        // Verify round-trip
        let deserialized: IterationLog = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.iteration, 0);
        assert!(deserialized.eval_win_rate.is_none());

        let deserialized2: IterationLog = serde_json::from_str(&json2).expect("deserialize");
        assert!((deserialized2.eval_win_rate.unwrap() - 0.18).abs() < 1e-6);
    }
}
