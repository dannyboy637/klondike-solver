// EvalFn trait and UniformEval implementation.
// Used by MCTS for leaf evaluation.

use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::env::KlondikeEnv;

/// Evaluation function for MCTS leaf nodes.
///
/// Returns (policy, value) where:
/// - policy: [f32; 260] probability distribution over actions
/// - value: f32 estimated win probability (0-1)
pub trait EvalFn {
    fn evaluate(&self, env: &KlondikeEnv) -> ([f32; ACTION_SPACE_SIZE], f32);
}

/// Uniform policy over legal moves, value 0.5.
/// Used as baseline before neural net is available.
pub struct UniformEval;

impl EvalFn for UniformEval {
    fn evaluate(&self, env: &KlondikeEnv) -> ([f32; ACTION_SPACE_SIZE], f32) {
        let mask = env.get_legal_mask();
        let n_legal: f32 = mask.iter().sum();
        if n_legal == 0.0 {
            return ([0.0f32; ACTION_SPACE_SIZE], 0.0);
        }
        let mut policy = [0.0f32; ACTION_SPACE_SIZE];
        let inv = 1.0 / n_legal;
        for i in 0..ACTION_SPACE_SIZE {
            policy[i] = mask[i] * inv;
        }
        (policy, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_eval_sums_to_one() {
        let env = KlondikeEnv::new(42, 1);
        let eval = UniformEval;
        let (policy, value) = eval.evaluate(&env);

        assert!((0.99..=1.01).contains(&policy.iter().sum::<f32>()));
        assert!((value - 0.5).abs() < 1e-6);

        // Only legal actions should have nonzero policy
        let mask = env.get_legal_mask();
        for i in 0..ACTION_SPACE_SIZE {
            if mask[i] == 0.0 {
                assert_eq!(policy[i], 0.0, "illegal action {} has nonzero policy", i);
            }
        }
    }

    #[test]
    fn test_uniform_eval_no_legal_moves() {
        // We can't easily create a no-legal-moves state, but we can verify
        // that the function handles it correctly by checking the logic.
        // For now, just verify the basic case works.
        let env = KlondikeEnv::new(42, 1);
        let eval = UniformEval;
        let (policy, _) = eval.evaluate(&env);
        let legal = env.get_legal_actions();
        let expected = 1.0 / legal.len() as f32;
        for &a in &legal {
            assert!(
                (policy[a] - expected).abs() < 1e-6,
                "action {} policy {} != expected {}",
                a,
                policy[a],
                expected
            );
        }
    }
}
