pub mod eval_fn;
pub mod hop;
pub mod mcts;
pub mod pimc;

use klondike_core::actions::ACTION_SPACE_SIZE;

/// Shared return type for all search functions.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Best action index (0-259), or -1 if no legal moves.
    pub action: i32,
    /// Visit counts or win rates per action. Shape: [ACTION_SPACE_SIZE].
    pub visit_counts: [f32; ACTION_SPACE_SIZE],
    /// Estimated win probability (0-1).
    pub value: f32,
    /// Number of determinizations used (0 for pure MCTS).
    pub determinizations_used: u32,
    /// MCTS simulations per determinization.
    pub simulations_per_det: u32,
    /// Wall-clock time in milliseconds.
    pub wall_time_ms: f64,
}
