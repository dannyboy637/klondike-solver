//! AlphaZero training pipeline for Klondike Solitaire.
//!
//! - [`selfplay`]: Parallel self-play game generation via rayon + PC-PIMC search.
//! - [`buffer`]: FIFO replay buffer with serde save/load.
//! - [`pipeline`]: Full training loop: selfplay → buffer → train → eval → checkpoint.

pub mod buffer;
pub mod inference_server;
pub mod pipeline;
pub mod selfplay;
