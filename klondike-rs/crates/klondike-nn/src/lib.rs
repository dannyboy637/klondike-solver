//! Neural network for Klondike Solitaire (tch-rs / PyTorch).
//!
//! - [`model::KlondikeNet`]: MLP-ResNet with Fixup init, dual policy+value heads (~2.75M params).
//! - [`inference::BatchedInference`]: Batched GPU/MPS inference with pre-allocated staging buffers.
//! - [`auto_device`]: Automatic device selection (CUDA > MPS > CPU).

pub mod inference;
pub mod model;

pub use inference::auto_device;
