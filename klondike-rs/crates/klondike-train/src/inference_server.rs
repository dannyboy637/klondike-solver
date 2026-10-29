// Batched GPU inference server for parallel self-play.
//
// Architecture: a single GPU thread collects inference requests from multiple
// rayon worker threads, batches them into one forward pass, and returns results.
// This avoids batch-1 kernel launch overhead (~0.7ms × 1000 calls/move).

use crossbeam_channel::{Receiver, Sender};
use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::encoding::OBSERVATION_DIM;
use klondike_core::env::KlondikeEnv;
use klondike_nn::model::KlondikeNet;
use klondike_search::eval_fn::EvalFn;
use tch::{Device, Tensor};

/// Request sent from a worker thread to the inference server.
pub struct InferenceRequest {
    observation: [f32; OBSERVATION_DIM],
    legal_mask: [f32; ACTION_SPACE_SIZE],
    response_tx: Sender<([f32; ACTION_SPACE_SIZE], f32)>,
}

/// Client handle for worker threads — implements EvalFn.
///
/// Each call to `evaluate()` sends the observation to the GPU server,
/// blocks until the batched result comes back, and returns (policy, value).
pub struct InferenceClient {
    request_tx: Sender<InferenceRequest>,
}

impl InferenceClient {
    pub fn new(request_tx: Sender<InferenceRequest>) -> Self {
        Self { request_tx }
    }
}

// crossbeam Sender is Send + Sync, so InferenceClient is too.
// No unsafe needed.

impl EvalFn for InferenceClient {
    fn evaluate(&self, env: &KlondikeEnv) -> ([f32; ACTION_SPACE_SIZE], f32) {
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);
        self.request_tx
            .send(InferenceRequest {
                observation: env.get_observation(),
                legal_mask: env.get_legal_mask(),
                response_tx: resp_tx,
            })
            .expect("inference server should be alive");
        resp_rx.recv().expect("inference server should respond")
    }
}

/// GPU inference server — owns no model, borrows it from the enclosing scope.
///
/// Use with `std::thread::scope()` so the server thread can borrow `&KlondikeNet`
/// without requiring `'static`.
pub struct InferenceServer<'a> {
    net: &'a KlondikeNet,
    device: Device,
    request_rx: Receiver<InferenceRequest>,
    max_batch_size: usize,
    // Pre-allocated staging buffers to avoid per-batch allocation
    obs_staging: Vec<f32>,
    mask_staging: Vec<f32>,
}

// SAFETY: InferenceServer runs on a single dedicated thread. The &KlondikeNet
// it borrows is used only for read-only inference (no_grad forward passes).
// During self-play, weights are frozen. The raw pointer in tch::Tensor wraps
// a reference-counted C++ at::Tensor that supports concurrent reads.
// Same safety argument as `unsafe impl Sync for NeuralEvalFn` in pipeline.rs.
unsafe impl Send for InferenceServer<'_> {}

impl<'a> InferenceServer<'a> {
    pub fn new(
        net: &'a KlondikeNet,
        device: Device,
        request_rx: Receiver<InferenceRequest>,
        max_batch_size: usize,
    ) -> Self {
        Self {
            net,
            device,
            request_rx,
            max_batch_size,
            obs_staging: Vec::with_capacity(max_batch_size * OBSERVATION_DIM),
            mask_staging: Vec::with_capacity(max_batch_size * ACTION_SPACE_SIZE),
        }
    }

    /// Run the server loop. Blocks until all senders are dropped.
    pub fn run(&mut self) {
        // Temporary buffer for response channels (avoids allocation per batch)
        let mut response_txs: Vec<Sender<([f32; ACTION_SPACE_SIZE], f32)>> =
            Vec::with_capacity(self.max_batch_size);

        loop {
            // Block until first request arrives (or channel disconnects)
            let first = match self.request_rx.recv() {
                Ok(req) => req,
                Err(_) => break, // All senders dropped — exit
            };

            self.obs_staging.clear();
            self.mask_staging.clear();
            response_txs.clear();

            // Add first request
            self.obs_staging.extend_from_slice(&first.observation);
            self.mask_staging.extend_from_slice(&first.legal_mask);
            response_txs.push(first.response_tx);

            // Drain pending requests without blocking (up to max_batch_size)
            while response_txs.len() < self.max_batch_size {
                match self.request_rx.try_recv() {
                    Ok(req) => {
                        self.obs_staging.extend_from_slice(&req.observation);
                        self.mask_staging.extend_from_slice(&req.legal_mask);
                        response_txs.push(req.response_tx);
                    }
                    Err(_) => break,
                }
            }

            let batch_size = response_txs.len();

            // Build tensors and run forward pass
            let obs_t = Tensor::from_slice(&self.obs_staging)
                .reshape([batch_size as i64, OBSERVATION_DIM as i64])
                .to_device(self.device);
            let mask_t = Tensor::from_slice(&self.mask_staging)
                .reshape([batch_size as i64, ACTION_SPACE_SIZE as i64])
                .to_device(self.device);

            let (log_policy, value) =
                tch::no_grad(|| self.net.forward(&obs_t, Some(&mask_t)));

            // Extract results on CPU
            let policy_probs = log_policy.exp().to_device(Device::Cpu);
            let value_cpu = value.to_device(Device::Cpu);

            // Send results back to each worker
            for i in 0..batch_size {
                let policy_row = policy_probs.get(i as i64);
                let policy_data: Vec<f32> =
                    Vec::try_from(&policy_row).expect("policy tensor extraction failed");

                let mut result_policy = [0.0f32; ACTION_SPACE_SIZE];
                result_policy.copy_from_slice(&policy_data);

                let val: f64 = value_cpu.double_value(&[i as i64, 0]);

                // Ignore send errors — worker may have been cancelled
                let _ = response_txs[i].send((result_policy, val as f32));
            }
        }
    }
}

/// Create a matched (client, server) pair for batched inference.
///
/// - `channel_bound`: capacity of the request channel (64 is a good default)
/// - `max_batch_size`: maximum requests to batch in one GPU call (64 is typical)
///
/// Usage with `std::thread::scope`:
/// ```ignore
/// std::thread::scope(|s| {
///     let (client, request_rx) = create_inference_channel(64);
///     s.spawn(|| {
///         let mut server = InferenceServer::new(&net, device, request_rx, 64);
///         server.run();
///     });
///     // Use client as EvalFn in rayon workers...
///     drop(client); // Signals server to exit
/// });
/// ```
pub fn create_inference_channel(
    channel_bound: usize,
) -> (InferenceClient, Receiver<InferenceRequest>) {
    let (tx, rx) = crossbeam_channel::bounded(channel_bound);
    (InferenceClient::new(tx), rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use klondike_nn::model::NetConfig;
    use rayon::prelude::*;
    use tch::nn;

    #[test]
    fn test_inference_server_correctness() {
        // Verify server+client produces same results as direct NeuralEvalFn
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig {
            hidden_dim: 64,
            n_blocks: 1,
            ..Default::default()
        };
        let net = KlondikeNet::new(&vs.root(), &config);

        let seeds: Vec<u64> = (0..8).collect();

        // Collect direct (batch-1) results
        let direct_results: Vec<([f32; ACTION_SPACE_SIZE], f32)> = seeds
            .iter()
            .map(|&s| {
                let env = KlondikeEnv::new(s, 1);
                let obs = env.get_observation();
                let mask = env.get_legal_mask();
                let obs_t = Tensor::from_slice(&obs)
                    .reshape([1, OBSERVATION_DIM as i64]);
                let mask_t = Tensor::from_slice(&mask)
                    .reshape([1, ACTION_SPACE_SIZE as i64]);
                let (log_policy, value) = tch::no_grad(|| net.forward(&obs_t, Some(&mask_t)));
                let policy_probs = log_policy.exp().squeeze_dim(0);
                let policy_data: Vec<f32> = Vec::try_from(&policy_probs).expect("extract");
                let mut policy = [0.0f32; ACTION_SPACE_SIZE];
                policy.copy_from_slice(&policy_data);
                let val = value.double_value(&[0, 0]) as f32;
                (policy, val)
            })
            .collect();

        // Collect server-batched results
        let server_results: Vec<([f32; ACTION_SPACE_SIZE], f32)> = std::thread::scope(|s| {
            let (client, request_rx) = create_inference_channel(16);
            let mut server = InferenceServer::new(&net, Device::Cpu, request_rx, 8);

            let server_handle = s.spawn(move || {
                server.run();
            });

            let results: Vec<_> = seeds
                .iter()
                .map(|&seed| {
                    let env = KlondikeEnv::new(seed, 1);
                    client.evaluate(&env)
                })
                .collect();

            drop(client);
            server_handle.join().expect("server thread panicked");
            results
        });

        // Compare
        for (i, (direct, server)) in direct_results.iter().zip(&server_results).enumerate() {
            // Value should match closely (float precision may differ slightly due to batching)
            assert!(
                (direct.1 - server.1).abs() < 1e-4,
                "seed {} value mismatch: direct={}, server={}",
                i,
                direct.1,
                server.1,
            );

            // Policy should match closely
            for a in 0..ACTION_SPACE_SIZE {
                assert!(
                    (direct.0[a] - server.0[a]).abs() < 1e-4,
                    "seed {} action {} policy mismatch: direct={}, server={}",
                    i,
                    a,
                    direct.0[a],
                    server.0[a],
                );
            }
        }
    }

    #[test]
    fn test_inference_server_multithreaded() {
        // Verify server handles concurrent requests from multiple threads
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig {
            hidden_dim: 64,
            n_blocks: 1,
            ..Default::default()
        };
        let net = KlondikeNet::new(&vs.root(), &config);

        // Use rayon for worker parallelism (same as production code)
        let (client, request_rx) = create_inference_channel(32);
        let mut server = InferenceServer::new(&net, Device::Cpu, request_rx, 16);

        let all_results: Vec<(u64, f32)> = std::thread::scope(|s| {
            let server_handle = s.spawn(move || {
                server.run();
            });

            // 32 evaluations in parallel via rayon
            let results: Vec<_> = (0..32u64)
                .into_par_iter()
                .map(|seed| {
                    let env = KlondikeEnv::new(seed, 1);
                    let (policy, value) = client.evaluate(&env);
                    let sum: f32 = policy.iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 0.01,
                        "seed {} policy sum {} != ~1.0",
                        seed,
                        sum,
                    );
                    assert!(
                        (0.0..=1.0).contains(&value),
                        "seed {} value {} out of range",
                        seed,
                        value,
                    );
                    (seed, value)
                })
                .collect();

            // Drop client to signal server exit, then join
            drop(client);
            server_handle.join().expect("server panicked");
            results
        });

        assert_eq!(all_results.len(), 32, "should have 32 total results");
    }
}
