// KlondikeNet: MLP-ResNet with Fixup init, dual heads (policy + value).
// Ported from klondike/nn/model.py in Phase R3.

use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::encoding::OBSERVATION_DIM;
use tch::{nn, nn::Module, Kind, Tensor};

/// Configuration for KlondikeNet.
pub struct NetConfig {
    pub input_dim: i64,
    pub hidden_dim: i64,
    pub n_blocks: i64,
    pub action_dim: i64,
    pub use_auxiliary: bool,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self {
            input_dim: OBSERVATION_DIM as i64,
            hidden_dim: 512,
            n_blocks: 4,
            action_dim: ACTION_SPACE_SIZE as i64,
            use_auxiliary: true,
        }
    }
}

/// Residual block with two linear layers, Fixup init, and skip connection.
struct ResidualBlock {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ResidualBlock {
    fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        let fc1 = nn::linear(vs / "fc1", hidden_dim, hidden_dim, Default::default());
        let mut fc2 = nn::linear(vs / "fc2", hidden_dim, hidden_dim, Default::default());

        // Fixup initialization: zero fc2 weights and bias
        tch::no_grad(|| {
            let _ = fc2.ws.zero_();
            let _ = fc2.bs.as_mut().expect("fc2 must have bias").zero_();
        });

        Self { fc1, fc2 }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let residual = x;
        let h = self.fc1.forward(x).relu();
        let h = self.fc2.forward(&h);
        (h + residual).relu()
    }
}

/// MLP-ResNet for Klondike Solitaire.
///
/// Architecture:
/// - Input projection: Linear(757, 512) -> ReLU
/// - 4 residual blocks (512 hidden, Fixup init, no batch norm)
/// - Policy head: Linear(512, 260) -> masked log_softmax
/// - Value head: Linear(512, 256) -> ReLU -> Linear(256, 1) -> sigmoid
/// - Auxiliary heads: foundation count, face-down count (dormant)
///
/// ~2.75M parameters at default settings.
pub struct KlondikeNet {
    input_proj: nn::Linear,
    blocks: Vec<ResidualBlock>,
    policy_fc: nn::Linear,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,
    #[allow(dead_code)] // Dormant in Phase 3; activated in Phase 6
    aux_foundation: Option<nn::Linear>,
    #[allow(dead_code)] // Dormant in Phase 3; activated in Phase 6
    aux_facedown: Option<nn::Linear>,
}

impl KlondikeNet {
    /// Create a new KlondikeNet with the given config.
    ///
    /// VarStore parameter names match the Python state_dict keys exactly,
    /// enabling direct weight loading from Python-trained checkpoints.
    pub fn new(vs: &nn::Path, config: &NetConfig) -> Self {
        let input_proj = nn::linear(
            vs / "input_proj",
            config.input_dim,
            config.hidden_dim,
            Default::default(),
        );

        let mut blocks = Vec::new();
        for i in 0..config.n_blocks {
            let block_path = vs / "blocks" / i.to_string();
            blocks.push(ResidualBlock::new(&block_path, config.hidden_dim));
        }

        let policy_fc = nn::linear(
            vs / "policy_fc",
            config.hidden_dim,
            config.action_dim,
            Default::default(),
        );

        let value_fc1 = nn::linear(vs / "value_fc1", config.hidden_dim, 256, Default::default());
        let value_fc2 = nn::linear(vs / "value_fc2", 256, 1, Default::default());

        let aux_foundation = if config.use_auxiliary {
            Some(nn::linear(
                vs / "aux_foundation",
                config.hidden_dim,
                1,
                Default::default(),
            ))
        } else {
            None
        };
        let aux_facedown = if config.use_auxiliary {
            Some(nn::linear(
                vs / "aux_facedown",
                config.hidden_dim,
                1,
                Default::default(),
            ))
        } else {
            None
        };

        Self {
            input_proj,
            blocks,
            policy_fc,
            value_fc1,
            value_fc2,
            aux_foundation,
            aux_facedown,
        }
    }

    /// Forward pass. Returns (log_policy, value).
    ///
    /// - `obs`: (batch, input_dim) observation vectors
    /// - `legal_mask`: optional (batch, action_dim) binary mask of legal actions
    ///
    /// Returns:
    /// - `log_policy`: (batch, action_dim) log-probabilities (masked if mask provided)
    /// - `value`: (batch, 1) win probability in [0, 1]
    pub fn forward(&self, obs: &Tensor, legal_mask: Option<&Tensor>) -> (Tensor, Tensor) {
        // Shared trunk
        let mut x = self.input_proj.forward(obs).relu();
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // Policy head
        let logits = self.policy_fc.forward(&x);
        let logits = match legal_mask {
            Some(mask) => &logits + &((1.0 - mask) * (-1e9)),
            None => logits,
        };
        let log_policy = logits.log_softmax(-1, Kind::Float);

        // Value head
        let v = self.value_fc1.forward(&x).relu();
        let value = self.value_fc2.forward(&v).sigmoid();

        (log_policy, value)
    }

    /// Count total parameters in the VarStore.
    pub fn count_parameters(vs: &nn::VarStore) -> i64 {
        vs.variables().values().map(|t| t.numel() as i64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_model_creation() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig::default();
        let _model = KlondikeNet::new(&vs.root(), &config);
    }

    #[test]
    fn test_parameter_count() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig::default();
        let _model = KlondikeNet::new(&vs.root(), &config);

        let count = KlondikeNet::count_parameters(&vs);
        // input_proj: 757*512 + 512 = 388096
        // 4 blocks: 4 * (512*512 + 512 + 512*512 + 512) = 4 * 525312 = 2101248
        // policy_fc: 512*260 + 260 = 133380
        // value_fc1: 512*256 + 256 = 131328
        // value_fc2: 256*1 + 1 = 257
        // aux_foundation: 512*1 + 1 = 513
        // aux_facedown: 512*1 + 1 = 513
        // Total: 2755335
        assert_eq!(count, 2_755_335, "parameter count mismatch");
    }

    #[test]
    fn test_forward_pass() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig::default();
        let model = KlondikeNet::new(&vs.root(), &config);

        let batch_size = 4;
        let obs = Tensor::randn([batch_size, config.input_dim], (Kind::Float, Device::Cpu));
        let mask = Tensor::ones([batch_size, config.action_dim], (Kind::Float, Device::Cpu));

        let (log_policy, value) = model.forward(&obs, Some(&mask));

        assert_eq!(log_policy.size(), &[batch_size, config.action_dim]);
        assert_eq!(value.size(), &[batch_size, 1]);

        // log_policy should be valid log-probabilities (sum to ~1 after exp)
        let probs = log_policy.exp();
        let sums = probs.sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        let sums_vec: Vec<f32> = Vec::try_from(&sums).unwrap();
        for s in &sums_vec {
            assert!((s - 1.0).abs() < 1e-5, "policy should sum to 1, got {}", s);
        }

        // value should be in [0, 1]
        let val_vec: Vec<f32> = Vec::try_from(&value.view([-1])).unwrap();
        for v in &val_vec {
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "value should be in [0,1], got {}",
                v
            );
        }
    }

    #[test]
    fn test_forward_no_mask() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig::default();
        let model = KlondikeNet::new(&vs.root(), &config);

        let obs = Tensor::randn([2, config.input_dim], (Kind::Float, Device::Cpu));
        let (log_policy, value) = model.forward(&obs, None);

        assert_eq!(log_policy.size(), &[2, config.action_dim]);
        assert_eq!(value.size(), &[2, 1]);
    }

    #[test]
    fn test_varstore_keys() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig::default();
        let _model = KlondikeNet::new(&vs.root(), &config);

        let variables = vs.variables();
        let mut keys: Vec<String> = variables.keys().cloned().collect();
        keys.sort();

        let expected = vec![
            "aux_facedown.bias",
            "aux_facedown.weight",
            "aux_foundation.bias",
            "aux_foundation.weight",
            "blocks.0.fc1.bias",
            "blocks.0.fc1.weight",
            "blocks.0.fc2.bias",
            "blocks.0.fc2.weight",
            "blocks.1.fc1.bias",
            "blocks.1.fc1.weight",
            "blocks.1.fc2.bias",
            "blocks.1.fc2.weight",
            "blocks.2.fc1.bias",
            "blocks.2.fc1.weight",
            "blocks.2.fc2.bias",
            "blocks.2.fc2.weight",
            "blocks.3.fc1.bias",
            "blocks.3.fc1.weight",
            "blocks.3.fc2.bias",
            "blocks.3.fc2.weight",
            "input_proj.bias",
            "input_proj.weight",
            "policy_fc.bias",
            "policy_fc.weight",
            "value_fc1.bias",
            "value_fc1.weight",
            "value_fc2.bias",
            "value_fc2.weight",
        ];

        assert_eq!(
            keys, expected,
            "VarStore keys must match Python state_dict keys"
        );
    }

    #[test]
    fn test_fixup_init() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = NetConfig::default();
        let model = KlondikeNet::new(&vs.root(), &config);

        // All fc2 weights and biases in residual blocks should be zero (Fixup init)
        for block in &model.blocks {
            let ws_sum: f64 = block.fc2.ws.abs().sum(Kind::Float).try_into().unwrap();
            let bs_sum: f64 = block
                .fc2
                .bs
                .as_ref()
                .unwrap()
                .abs()
                .sum(Kind::Float)
                .try_into()
                .unwrap();
            assert!(
                ws_sum < 1e-10,
                "fc2 weights should be zero (Fixup), got sum={}",
                ws_sum
            );
            assert!(
                bs_sum < 1e-10,
                "fc2 bias should be zero (Fixup), got sum={}",
                bs_sum
            );
        }
    }
}
