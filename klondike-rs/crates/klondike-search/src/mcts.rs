// Arena-allocated MCTS with UCB1 selection.
// Ported from klondike/search/mcts.py in Phase R2.
//
// Key differences from Python:
// - Arena allocation (Vec<MctsNode> with u32 indices) instead of dict of heap nodes
// - Cache-friendly contiguous memory layout
// - clear() reuses allocated memory

use std::time::Instant;

use klondike_core::actions::{action_to_move, tuple_to_move, ACTION_SPACE_SIZE};
use klondike_core::env::KlondikeEnv;
use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::eval_fn::EvalFn;
use crate::SearchResult;

/// MCTS configuration.
#[derive(Clone, Debug)]
pub struct MctsConfig {
    pub n_simulations: u32,
    pub c_puct: f32,
    pub temperature: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_frac: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            n_simulations: 400,
            c_puct: 1.5,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_frac: 0.25,
        }
    }
}

/// Single node in the MCTS arena.
///
/// 20 bytes per node. Children are stored contiguously in the arena
/// starting at children_start, with children_count entries.
#[derive(Clone, Debug)]
pub struct MctsNode {
    pub visit_count: u32,
    pub value_sum: f32,
    pub prior: f32,
    pub children_start: u32,
    pub children_count: u16,
    pub action: u16,
}

impl MctsNode {
    fn new(prior: f32, action: u16) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            prior,
            children_start: 0,
            children_count: 0,
            action,
        }
    }

    #[inline]
    fn value(&self) -> f32 {
        if self.visit_count == 0 {
            return 0.0;
        }
        self.value_sum / self.visit_count as f32
    }

    #[inline]
    fn ucb_score(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let exploration =
            c_puct * self.prior * (parent_visits as f32).sqrt() / (1 + self.visit_count) as f32;
        self.value() + exploration
    }
}

/// Arena-allocated MCTS tree.
///
/// All nodes live in a contiguous Vec. Parent-child relationships
/// are encoded via (children_start, children_count) indices.
pub struct MctsArena {
    nodes: Vec<MctsNode>,
}

impl Default for MctsArena {
    fn default() -> Self {
        Self::new()
    }
}

impl MctsArena {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(4096),
        }
    }

    /// Clear the arena for reuse (keeps allocated memory).
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Allocate a root node and return its index.
    fn alloc_root(&mut self) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(MctsNode::new(0.0, u16::MAX));
        idx
    }

    /// Expand a node: add children for all legal moves.
    fn expand(
        &mut self,
        parent_idx: u32,
        legal_mask: &[f32; ACTION_SPACE_SIZE],
        policy: &[f32; ACTION_SPACE_SIZE],
    ) {
        let start = self.nodes.len() as u32;
        let mut count = 0u16;
        for action in 0..ACTION_SPACE_SIZE {
            if legal_mask[action] > 0.0 {
                self.nodes
                    .push(MctsNode::new(policy[action], action as u16));
                count += 1;
            }
        }
        self.nodes[parent_idx as usize].children_start = start;
        self.nodes[parent_idx as usize].children_count = count;
    }

    /// Select the child with highest UCB score.
    fn select_child(&self, parent_idx: u32, c_puct: f32) -> u32 {
        let parent = &self.nodes[parent_idx as usize];
        let start = parent.children_start as usize;
        let count = parent.children_count as usize;

        let mut best_idx = start as u32;
        let mut best_score = f32::NEG_INFINITY;

        for i in start..start + count {
            let score = self.nodes[i].ucb_score(parent.visit_count, c_puct);
            if score > best_score {
                best_score = score;
                best_idx = i as u32;
            }
        }

        best_idx
    }
}

/// Run MCTS search from the given environment state.
///
/// Returns a SearchResult with visit distribution and selected action.
/// The `rng` is used for Dirichlet noise and temperature-based action selection.
pub fn mcts_search(
    env: &KlondikeEnv,
    config: &MctsConfig,
    eval_fn: &dyn EvalFn,
    rng: &mut impl Rng,
) -> SearchResult {
    mcts_search_with_arena(&mut MctsArena::new(), env, config, eval_fn, rng)
}

/// Run MCTS search reusing an existing arena (avoids reallocation).
pub fn mcts_search_with_arena(
    arena: &mut MctsArena,
    env: &KlondikeEnv,
    config: &MctsConfig,
    eval_fn: &dyn EvalFn,
    rng: &mut impl Rng,
) -> SearchResult {
    let start = Instant::now();
    arena.clear();

    let legal_mask = env.get_legal_mask();
    let n_legal: usize = legal_mask.iter().filter(|&&v| v > 0.0).count();

    if n_legal == 0 {
        return SearchResult {
            action: -1,
            visit_counts: [0.0f32; ACTION_SPACE_SIZE],
            value: 0.0,
            determinizations_used: 0,
            simulations_per_det: config.n_simulations,
            wall_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        };
    }

    // Initialize root
    let root_idx = arena.alloc_root();

    // Evaluate root position
    let (mut policy, _root_value) = eval_fn.evaluate(env);

    // Mask and normalize policy
    mask_and_normalize(&mut policy, &legal_mask, n_legal);

    // Add Dirichlet noise at root for exploration
    if config.dirichlet_frac > 0.0 {
        let noise = sample_dirichlet(rng, config.dirichlet_alpha);
        for i in 0..ACTION_SPACE_SIZE {
            policy[i] =
                (1.0 - config.dirichlet_frac) * policy[i] + config.dirichlet_frac * noise[i];
        }
        // Re-mask and normalize after noise
        mask_and_normalize(&mut policy, &legal_mask, n_legal);
    }

    // Create root children
    arena.expand(root_idx, &legal_mask, &policy);

    // Pre-allocate path buffer — reused across all simulations
    let mut path: Vec<u32> = Vec::with_capacity(64);

    // Run simulations
    for _ in 0..config.n_simulations {
        simulate(arena, env, root_idx, config, eval_fn, &mut path);
        // CRITICAL: root.visit_count must be incremented here, not in backprop.
        arena.nodes[root_idx as usize].visit_count += 1;
    }

    // Extract visit distribution from root's children
    let mut visit_counts = [0.0f32; ACTION_SPACE_SIZE];
    let root = &arena.nodes[root_idx as usize];
    let start_child = root.children_start as usize;
    let count = root.children_count as usize;

    for i in start_child..start_child + count {
        let child = &arena.nodes[i];
        visit_counts[child.action as usize] = child.visit_count as f32;
    }

    // Select action based on temperature (log-space to avoid overflow)
    let best_action = select_action_temperature(&visit_counts, config.temperature, rng);

    // CRITICAL: root value from children, not root.value_sum
    let total_child_visits: u32 = (start_child..start_child + count)
        .map(|i| arena.nodes[i].visit_count)
        .sum();
    let total_child_value: f32 = (start_child..start_child + count)
        .map(|i| arena.nodes[i].value_sum)
        .sum();
    let root_value = if total_child_visits > 0 {
        total_child_value / total_child_visits as f32
    } else {
        0.0
    };

    SearchResult {
        action: best_action,
        visit_counts,
        value: root_value,
        determinizations_used: 0,
        simulations_per_det: config.n_simulations,
        wall_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

/// Single MCTS simulation: select -> expand -> evaluate -> backpropagate.
///
/// `path` is a reusable buffer for tracking the traversal path.
/// It is cleared at the start of each call and reused to avoid allocation.
fn simulate(
    arena: &mut MctsArena,
    env: &KlondikeEnv,
    root_idx: u32,
    config: &MctsConfig,
    eval_fn: &dyn EvalFn,
    path: &mut Vec<u32>,
) {
    let mut sim_env = env.clone();
    path.clear();
    let mut node_idx = root_idx;

    // Selection: walk down tree picking best UCB child
    while arena.nodes[node_idx as usize].children_count > 0 && !sim_env.is_done() {
        let child_idx = arena.select_child(node_idx, config.c_puct);
        path.push(child_idx);

        let action = arena.nodes[child_idx as usize].action as usize;
        let (move_type, card_idx) = action_to_move(action);
        let m = tuple_to_move(move_type, card_idx).expect("valid action in MCTS tree");
        let _ = sim_env.game_mut().do_move(&m);

        node_idx = child_idx;

        // If node hasn't been visited, it's a leaf — stop and expand
        if arena.nodes[node_idx as usize].visit_count == 0 {
            break;
        }
    }

    // Expansion + Evaluation
    let value;
    if sim_env.is_done() {
        value = if sim_env.game().is_win() { 1.0 } else { 0.0 };
    } else {
        let legal_mask = sim_env.get_legal_mask();
        let n_legal: usize = legal_mask.iter().filter(|&&v| v > 0.0).count();
        if n_legal == 0 {
            value = 0.0;
        } else {
            let (mut policy, v) = eval_fn.evaluate(&sim_env);
            value = v;

            // Mask and normalize
            mask_and_normalize(&mut policy, &legal_mask, n_legal);

            // Expand if this node has no children yet
            if arena.nodes[node_idx as usize].children_count == 0 {
                arena.expand(node_idx, &legal_mask, &policy);
            }
        }
    }

    // Backpropagation: update visit counts and values along the path.
    // root.visit_count is incremented by the caller (mcts_search), not here.
    for &child_idx in path.iter().rev() {
        arena.nodes[child_idx as usize].visit_count += 1;
        arena.nodes[child_idx as usize].value_sum += value;
    }
}

/// Mask a policy to legal moves and normalize.
fn mask_and_normalize(
    policy: &mut [f32; ACTION_SPACE_SIZE],
    legal_mask: &[f32; ACTION_SPACE_SIZE],
    n_legal: usize,
) {
    let mut total = 0.0f32;
    for i in 0..ACTION_SPACE_SIZE {
        policy[i] *= legal_mask[i];
        total += policy[i];
    }
    if total > 0.0 {
        let inv = 1.0 / total;
        for p in policy.iter_mut() {
            *p *= inv;
        }
    } else {
        // Fallback: uniform over legal moves
        let inv = 1.0 / n_legal as f32;
        for i in 0..ACTION_SPACE_SIZE {
            policy[i] = legal_mask[i] * inv;
        }
    }
}

/// Sample from Dirichlet(alpha, ..., alpha) with ACTION_SPACE_SIZE dimensions.
/// Uses Gamma sampling + normalization.
fn sample_dirichlet(rng: &mut impl Rng, alpha: f32) -> [f32; ACTION_SPACE_SIZE] {
    let gamma = Gamma::new(alpha as f64, 1.0).expect("valid gamma parameters");
    let mut samples = [0.0f32; ACTION_SPACE_SIZE];
    let mut sum = 0.0f32;
    for s in samples.iter_mut() {
        *s = gamma.sample(rng) as f32;
        sum += *s;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for s in samples.iter_mut() {
            *s *= inv;
        }
    }
    samples
}

/// Select action from visit counts using temperature (log-space to avoid overflow).
///
/// CRITICAL: uses log-space computation to prevent f64 overflow when
/// temperature is small (e.g. 0.05) and visit counts are large.
pub(crate) fn select_action_temperature(
    visit_counts: &[f32; ACTION_SPACE_SIZE],
    temperature: f32,
    rng: &mut impl Rng,
) -> i32 {
    if temperature < 0.01 {
        // Near-greedy: pick highest visit count
        let mut best_action = -1i32;
        let mut best_count = -1.0f32;
        for (i, &vc) in visit_counts.iter().enumerate() {
            if vc > best_count {
                best_count = vc;
                best_action = i as i32;
            }
        }
        return best_action;
    }

    // Log-space: log(n^(1/t)) = log(n)/t, then softmax
    let mut log_probs = [0.0f64; ACTION_SPACE_SIZE];
    let mut max_log = f64::NEG_INFINITY;

    for (i, &vc) in visit_counts.iter().enumerate() {
        if vc > 0.0 {
            log_probs[i] = (vc as f64).ln() / temperature as f64;
            if log_probs[i] > max_log {
                max_log = log_probs[i];
            }
        } else {
            log_probs[i] = -1e9;
        }
    }

    // Numerical stability: subtract max before exp
    let mut probs = [0.0f64; ACTION_SPACE_SIZE];
    let mut total = 0.0f64;
    for i in 0..ACTION_SPACE_SIZE {
        if visit_counts[i] > 0.0 {
            probs[i] = (log_probs[i] - max_log).exp();
            total += probs[i];
        }
    }

    if total <= 0.0 {
        // Fallback: greedy
        let mut best_action = -1i32;
        let mut best_count = -1.0f32;
        for (i, &vc) in visit_counts.iter().enumerate() {
            if vc > best_count {
                best_count = vc;
                best_action = i as i32;
            }
        }
        return best_action;
    }

    // Sample from distribution
    let inv = 1.0 / total;
    let r: f64 = rng.gen();
    let mut cumulative = 0.0f64;
    for i in 0..ACTION_SPACE_SIZE {
        if visit_counts[i] > 0.0 {
            cumulative += probs[i] * inv;
            if r < cumulative {
                return i as i32;
            }
        }
    }

    // Fallback (rounding errors): return last visited action
    for i in (0..ACTION_SPACE_SIZE).rev() {
        if visit_counts[i] > 0.0 {
            return i as i32;
        }
    }
    -1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_fn::UniformEval;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_mcts_basic_properties() {
        let env = KlondikeEnv::new(42, 1);
        let config = MctsConfig {
            n_simulations: 100,
            ..MctsConfig::default()
        };
        let eval = UniformEval;
        let mut rng = SmallRng::seed_from_u64(42);

        let result = mcts_search(&env, &config, &eval, &mut rng);

        // Action should be legal
        assert!(result.action >= 0, "action should be non-negative");
        let legal = env.get_legal_actions();
        assert!(
            legal.contains(&(result.action as usize)),
            "action {} should be in legal actions {:?}",
            result.action,
            legal
        );

        // Visit counts should sum to n_simulations
        let total_visits: f32 = result.visit_counts.iter().sum();
        assert_eq!(
            total_visits as u32, config.n_simulations,
            "total visits should equal n_simulations"
        );

        // Only legal actions should have visits
        let mask = env.get_legal_mask();
        for (i, &vc) in result.visit_counts.iter().enumerate() {
            if mask[i] == 0.0 {
                assert_eq!(vc, 0.0, "illegal action {} should have zero visits", i);
            }
        }

        // Value should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&result.value),
            "value {} should be in [0, 1]",
            result.value
        );

        // Wall time should be positive
        assert!(result.wall_time_ms > 0.0);
    }

    #[test]
    fn test_mcts_deterministic_same_seed() {
        let env = KlondikeEnv::new(42, 1);
        let config = MctsConfig {
            n_simulations: 50,
            ..MctsConfig::default()
        };
        let eval = UniformEval;

        let mut rng1 = SmallRng::seed_from_u64(123);
        let result1 = mcts_search(&env, &config, &eval, &mut rng1);

        let mut rng2 = SmallRng::seed_from_u64(123);
        let result2 = mcts_search(&env, &config, &eval, &mut rng2);

        assert_eq!(
            result1.action, result2.action,
            "same seed should produce same action"
        );
        assert_eq!(
            result1.visit_counts, result2.visit_counts,
            "same seed should produce same visit counts"
        );
    }

    #[test]
    fn test_mcts_no_dirichlet() {
        let env = KlondikeEnv::new(42, 1);
        let config = MctsConfig {
            n_simulations: 50,
            dirichlet_frac: 0.0,
            ..MctsConfig::default()
        };
        let eval = UniformEval;
        let mut rng = SmallRng::seed_from_u64(42);

        let result = mcts_search(&env, &config, &eval, &mut rng);

        // Should still work without Dirichlet noise
        assert!(result.action >= 0);
        let total: f32 = result.visit_counts.iter().sum();
        assert_eq!(total as u32, config.n_simulations);
    }

    #[test]
    fn test_mcts_greedy_temperature() {
        let env = KlondikeEnv::new(42, 1);
        let config = MctsConfig {
            n_simulations: 100,
            temperature: 0.001, // Near-greedy
            dirichlet_frac: 0.0,
            ..MctsConfig::default()
        };
        let eval = UniformEval;
        let mut rng = SmallRng::seed_from_u64(42);

        let result = mcts_search(&env, &config, &eval, &mut rng);

        // Greedy should pick the most visited action
        let max_visits = result.visit_counts.iter().cloned().fold(0.0f32, f32::max);
        assert_eq!(
            result.visit_counts[result.action as usize], max_visits,
            "greedy should pick the most visited action"
        );
    }

    #[test]
    fn test_mcts_arena_reuse() {
        let env = KlondikeEnv::new(42, 1);
        let config = MctsConfig {
            n_simulations: 50,
            ..MctsConfig::default()
        };
        let eval = UniformEval;
        let mut arena = MctsArena::new();

        let mut rng1 = SmallRng::seed_from_u64(42);
        let result1 = mcts_search_with_arena(&mut arena, &env, &config, &eval, &mut rng1);

        // Reuse arena for second search
        let mut rng2 = SmallRng::seed_from_u64(42);
        let result2 = mcts_search_with_arena(&mut arena, &env, &config, &eval, &mut rng2);

        assert_eq!(result1.action, result2.action);
        assert_eq!(result1.visit_counts, result2.visit_counts);
    }

    #[test]
    fn test_select_action_temperature_greedy() {
        let mut counts = [0.0f32; ACTION_SPACE_SIZE];
        counts[10] = 50.0;
        counts[20] = 30.0;
        counts[30] = 20.0;
        let mut rng = SmallRng::seed_from_u64(42);

        let action = select_action_temperature(&counts, 0.001, &mut rng);
        assert_eq!(action, 10, "greedy should pick highest count");
    }

    #[test]
    fn test_select_action_temperature_distribution() {
        let mut counts = [0.0f32; ACTION_SPACE_SIZE];
        counts[10] = 50.0;
        counts[20] = 50.0;
        let mut rng = SmallRng::seed_from_u64(42);

        // With temperature 1.0 and equal counts, both should be selected roughly equally
        let mut selections = [0u32; ACTION_SPACE_SIZE];
        for _ in 0..1000 {
            let action = select_action_temperature(&counts, 1.0, &mut rng);
            selections[action as usize] += 1;
        }

        // Both should have significant selections (roughly 500 each)
        assert!(
            selections[10] > 350,
            "action 10 selected {} times",
            selections[10]
        );
        assert!(
            selections[20] > 350,
            "action 20 selected {} times",
            selections[20]
        );
    }
}
