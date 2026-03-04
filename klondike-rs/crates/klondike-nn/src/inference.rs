// BatchedInference: collects N positions, runs ONE GPU forward pass.
// Key performance innovation of the Rust port (no Python equivalent).

use crate::model::{KlondikeNet, NetConfig};
use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::encoding::OBSERVATION_DIM;
use tch::{nn, Device, Tensor};

/// Detect the best available device (CUDA > MPS > CPU).
pub fn auto_device() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else if tch::utils::has_mps() {
        Device::Mps
    } else {
        Device::Cpu
    }
}

/// Batched neural network inference engine.
///
/// Wraps a KlondikeNet with its VarStore and handles device placement,
/// tensor construction, and result extraction. Pre-allocates staging
/// buffers to avoid repeated heap allocation in hot paths.
pub struct BatchedInference {
    net: KlondikeNet,
    vs: nn::VarStore,
    device: Device,
    /// Pre-allocated staging buffer for flattened observations.
    obs_staging: Vec<f32>,
    /// Pre-allocated staging buffer for flattened legal masks.
    mask_staging: Vec<f32>,
}

impl BatchedInference {
    /// Create with random weights on the best available device.
    pub fn new(config: &NetConfig) -> Self {
        let device = auto_device();
        Self::new_on_device(config, device)
    }

    /// Create with random weights on a specific device.
    pub fn new_on_device(config: &NetConfig, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let net = KlondikeNet::new(&vs.root(), config);
        Self {
            net,
            vs,
            device,
            obs_staging: Vec::with_capacity(64 * OBSERVATION_DIM),
            mask_staging: Vec::with_capacity(64 * ACTION_SPACE_SIZE),
        }
    }

    /// Load weights from a .pt checkpoint file.
    pub fn load(config: &NetConfig, path: &std::path::Path) -> anyhow::Result<Self> {
        let device = auto_device();
        let mut vs = nn::VarStore::new(device);
        let net = KlondikeNet::new(&vs.root(), config);
        vs.load(path)?;
        Ok(Self {
            net,
            vs,
            device,
            obs_staging: Vec::with_capacity(64 * OBSERVATION_DIM),
            mask_staging: Vec::with_capacity(64 * ACTION_SPACE_SIZE),
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn vs(&self) -> &nn::VarStore {
        &self.vs
    }

    pub fn vs_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Evaluate a batch of positions.
    ///
    /// `observations`: N observation vectors of length 757.
    /// `legal_masks`: N legal action masks (1.0 = legal, 0.0 = illegal).
    ///
    /// Returns N `(policy_probs, value)` pairs.
    /// Policy is a proper probability distribution over legal actions only.
    ///
    /// Uses pre-allocated staging buffers to avoid per-call heap allocation.
    pub fn evaluate_batch(
        &mut self,
        observations: &[[f32; OBSERVATION_DIM]],
        legal_masks: &[[f32; ACTION_SPACE_SIZE]],
    ) -> Vec<([f32; ACTION_SPACE_SIZE], f32)> {
        let n = observations.len();
        assert_eq!(
            n,
            legal_masks.len(),
            "observations and legal_masks must have same length"
        );
        if n == 0 {
            return Vec::new();
        }

        // Reuse staging buffers (clear + extend avoids reallocation)
        self.obs_staging.clear();
        self.obs_staging.reserve(n * OBSERVATION_DIM);
        for obs in observations {
            self.obs_staging.extend_from_slice(obs);
        }

        self.mask_staging.clear();
        self.mask_staging.reserve(n * ACTION_SPACE_SIZE);
        for mask in legal_masks {
            self.mask_staging.extend_from_slice(mask);
        }

        // Create tensors on the target device
        let obs_tensor = Tensor::from_slice(&self.obs_staging)
            .reshape([n as i64, OBSERVATION_DIM as i64])
            .to_device(self.device);
        let mask_tensor = Tensor::from_slice(&self.mask_staging)
            .reshape([n as i64, ACTION_SPACE_SIZE as i64])
            .to_device(self.device);

        // Forward pass (no_grad for inference)
        let (log_policy, value) =
            tch::no_grad(|| self.net.forward(&obs_tensor, Some(&mask_tensor)));

        // Convert log_policy to probabilities and move to CPU in one step
        let policy_cpu = log_policy.exp().to_device(Device::Cpu);
        let value_cpu = value.to_device(Device::Cpu).view([-1]); // [n, 1] -> [n]

        // Extract results via contiguous copy (faster than per-row get())
        let policy_flat: Vec<f32> =
            Vec::try_from(&policy_cpu.view([-1])).expect("failed to extract policy tensor");
        let value_flat: Vec<f32> =
            Vec::try_from(&value_cpu).expect("failed to extract value tensor");

        let mut results = Vec::with_capacity(n);
        for (i, &val) in value_flat.iter().enumerate().take(n) {
            let mut policy = [0.0f32; ACTION_SPACE_SIZE];
            let start = i * ACTION_SPACE_SIZE;
            policy.copy_from_slice(&policy_flat[start..start + ACTION_SPACE_SIZE]);
            results.push((policy, val));
        }

        results
    }

    /// Evaluate a single position (convenience wrapper).
    pub fn evaluate_single(
        &mut self,
        observation: &[f32; OBSERVATION_DIM],
        legal_mask: &[f32; ACTION_SPACE_SIZE],
    ) -> ([f32; ACTION_SPACE_SIZE], f32) {
        self.evaluate_batch(
            std::slice::from_ref(observation),
            std::slice::from_ref(legal_mask),
        )
        .into_iter()
        .next()
        .expect("single evaluation should return one result")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_device() {
        let device = auto_device();
        match device {
            Device::Cpu | Device::Cuda(_) | Device::Mps => {}
            _ => panic!("unexpected device: {:?}", device),
        }
        eprintln!("auto_device: {:?}", device);
    }

    #[test]
    fn test_batched_inference_random_weights() {
        let config = NetConfig::default();
        let mut infer = BatchedInference::new(&config);

        // Create 10 observations and masks
        let mut observations = Vec::new();
        let mut masks = Vec::new();
        for _ in 0..10 {
            let mut obs = [0.0f32; OBSERVATION_DIM];
            let mut mask = [0.0f32; ACTION_SPACE_SIZE];
            obs[0] = 1.0;
            obs[100] = 0.5;
            mask[0] = 1.0;
            mask[1] = 1.0;
            mask[5] = 1.0;
            observations.push(obs);
            masks.push(mask);
        }

        let results = infer.evaluate_batch(&observations, &masks);
        assert_eq!(results.len(), 10);

        for (policy, value) in &results {
            // Value should be in [0, 1]
            assert!(
                *value >= 0.0 && *value <= 1.0,
                "value {} out of range",
                value
            );

            // Policy should sum to approximately 1.0
            let sum: f32 = policy.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "policy sum {} != 1.0", sum);

            // Illegal actions should have ~0 probability
            assert!(policy[2] < 0.001, "illegal action 2 has prob {}", policy[2]);
            assert!(policy[3] < 0.001, "illegal action 3 has prob {}", policy[3]);
        }
    }

    #[test]
    fn test_evaluate_single() {
        let config = NetConfig::default();
        let mut infer = BatchedInference::new(&config);

        let mut obs = [0.0f32; OBSERVATION_DIM];
        obs[0] = 1.0;
        let mut mask = [0.0f32; ACTION_SPACE_SIZE];
        mask[0] = 1.0;
        mask[10] = 1.0;

        let (policy, value) = infer.evaluate_single(&obs, &mask);
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.01);
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn test_empty_batch() {
        let config = NetConfig::default();
        let mut infer = BatchedInference::new(&config);
        let results = infer.evaluate_batch(&[], &[]);
        assert!(results.is_empty());
    }
}
