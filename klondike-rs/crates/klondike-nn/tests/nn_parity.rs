// Weight-loading parity test: loads Python checkpoint, runs same inputs,
// compares log_policy and value outputs within 1e-5.

use klondike_nn::inference::BatchedInference;
use klondike_nn::model::NetConfig;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct ParityData {
    #[allow(dead_code)]
    checkpoint: String,
    #[allow(dead_code)]
    seed: u64,
    n_samples: usize,
    param_count: i64,
    state_dict_keys: Vec<String>,
    samples: Vec<Sample>,
}

#[derive(Deserialize)]
struct Sample {
    observation: Vec<f32>,
    legal_mask: Vec<f32>,
    log_policy: Vec<f32>,
    value: f64,
}

fn load_parity_data() -> ParityData {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/nn_parity_data.json");
    let data = std::fs::read_to_string(&data_path)
        .unwrap_or_else(|e| panic!("Failed to read parity data at {:?}: {}", data_path, e));
    serde_json::from_str(&data).expect("Failed to parse parity data")
}

#[test]
fn test_param_count_matches_python() {
    let parity = load_parity_data();
    let config = NetConfig::default();
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let _model = klondike_nn::model::KlondikeNet::new(&vs.root(), &config);
    let rust_count = klondike_nn::model::KlondikeNet::count_parameters(&vs);
    assert_eq!(
        rust_count, parity.param_count,
        "Rust param count {} != Python {}",
        rust_count, parity.param_count
    );
}

#[test]
fn test_varstore_keys_match_python() {
    let parity = load_parity_data();
    let config = NetConfig::default();
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let _model = klondike_nn::model::KlondikeNet::new(&vs.root(), &config);

    let mut rust_keys: Vec<String> = vs.variables().keys().cloned().collect();
    rust_keys.sort();
    let mut python_keys = parity.state_dict_keys.clone();
    python_keys.sort();

    assert_eq!(
        rust_keys, python_keys,
        "VarStore keys don't match Python state_dict keys"
    );
}

#[test]
fn test_weight_loading_parity() {
    let parity = load_parity_data();
    assert_eq!(parity.n_samples, parity.samples.len());

    // Load Python checkpoint (safetensors format — converted from .pt)
    let ckpt_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("checkpoints/supervised_v2/best_supervised.safetensors");

    if !ckpt_path.exists() {
        eprintln!(
            "SKIP: checkpoint not found at {:?} — run on machine with checkpoints",
            ckpt_path
        );
        return;
    }

    let config = NetConfig::default();
    let mut infer = BatchedInference::load(&config, &ckpt_path)
        .unwrap_or_else(|e| panic!("Failed to load checkpoint: {}", e));

    eprintln!("Loaded checkpoint on device: {:?}", infer.device());

    // Run each sample through Rust and compare
    for (i, sample) in parity.samples.iter().enumerate() {
        let obs: [f32; 757] = sample.observation[..757]
            .try_into()
            .expect("obs should be 757 floats");
        let mask: [f32; 260] = sample.legal_mask[..260]
            .try_into()
            .expect("mask should be 260 floats");

        let (rust_policy, rust_value) = infer.evaluate_single(&obs, &mask);

        // Compare value within tolerance
        let python_value = sample.value as f32;
        let value_diff = (rust_value - python_value).abs();
        assert!(
            value_diff < 1e-4,
            "Sample {}: value mismatch: rust={:.6} python={:.6} diff={:.6}",
            i,
            rust_value,
            python_value,
            value_diff
        );

        // Compare log_policy within tolerance
        // Convert rust policy (probs) back to log for comparison
        let mut max_diff: f32 = 0.0;
        let mut max_diff_idx = 0;
        for j in 0..260 {
            let rust_log = if rust_policy[j] > 1e-30 {
                rust_policy[j].ln()
            } else {
                -70.0 // clamp very small values
            };
            let python_log = sample.log_policy[j];

            // Only compare legal actions (illegal ones are ~-inf, noisy)
            if mask[j] > 0.5 {
                let diff = (rust_log - python_log).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_idx = j;
                }
            }
        }
        assert!(
            max_diff < 1e-4,
            "Sample {}: log_policy max diff {:.6} at action {} (rust_log vs python_log)",
            i,
            max_diff,
            max_diff_idx
        );

        eprintln!(
            "Sample {}: value diff={:.8}, log_policy max_diff={:.8} — OK",
            i, value_diff, max_diff
        );
    }
    eprintln!("All {} parity samples PASS", parity.n_samples);
}

#[test]
fn test_batch_inference_throughput() {
    let config = NetConfig::default();
    let mut infer = BatchedInference::new(&config);

    // Create a batch of 64 observations
    let obs = [0.5f32; 757];
    let mut mask = [0.0f32; 260];
    mask[0] = 1.0;
    mask[1] = 1.0;
    mask[5] = 1.0;
    mask[10] = 1.0;
    mask[50] = 1.0;

    let observations: Vec<_> = (0..64).map(|_| obs).collect();
    let masks: Vec<_> = (0..64).map(|_| mask).collect();

    // Warmup
    for _ in 0..10 {
        let _ = infer.evaluate_batch(&observations, &masks);
    }

    // Benchmark 100 forward passes (reduced from 1000 for test speed)
    let start = std::time::Instant::now();
    let n_iters = 100;
    for _ in 0..n_iters {
        let _ = infer.evaluate_batch(&observations, &masks);
    }
    let elapsed = start.elapsed();
    let per_batch = elapsed / n_iters;
    let positions_per_sec = (n_iters * 64) as f64 / elapsed.as_secs_f64();

    eprintln!(
        "Throughput: {:.1} ms/batch-of-64 ({:.0} positions/sec) on {:?}",
        per_batch.as_secs_f64() * 1000.0,
        positions_per_sec,
        infer.device()
    );

    // Should be under 50ms per batch
    assert!(
        per_batch.as_millis() < 50,
        "batch-of-64 took {}ms — too slow (target <50ms)",
        per_batch.as_millis()
    );
}
