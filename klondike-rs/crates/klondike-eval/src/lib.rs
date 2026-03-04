//! Evaluation and statistical tools for Klondike solvers.
//!
//! - [`stats`]: Wilson CI, SPRT sequential testing, McNemar paired comparison.
//! - [`benchmark`]: Parallel game runner with progress logging and CI reporting.

pub mod benchmark;
pub mod stats;
