// FIFO replay buffer for self-play training samples.
// Ported from klondike/alphazero/buffer.py in Phase R4.

use klondike_core::actions::ACTION_SPACE_SIZE;
use klondike_core::encoding::OBSERVATION_DIM;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// A single training sample from self-play.
///
/// Uses `Vec<f32>` instead of fixed-size arrays because serde doesn't
/// support `[T; N]` for N > 32 without extra dependencies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingSample {
    pub observation: Vec<f32>, // OBSERVATION_DIM (757)
    pub policy: Vec<f32>,      // ACTION_SPACE_SIZE (260)
    pub value: f32,
    pub legal_mask: Vec<f32>, // ACTION_SPACE_SIZE (260)
}

impl TrainingSample {
    /// Create a sample from fixed-size arrays (the typical internal representation).
    pub fn new(
        observation: [f32; OBSERVATION_DIM],
        policy: [f32; ACTION_SPACE_SIZE],
        value: f32,
        legal_mask: [f32; ACTION_SPACE_SIZE],
    ) -> Self {
        Self {
            observation: observation.to_vec(),
            policy: policy.to_vec(),
            value,
            legal_mask: legal_mask.to_vec(),
        }
    }
}

/// Persistent format for save/load — stores the buffer contents plus the
/// write index so that FIFO ordering is preserved across restarts.
#[derive(Serialize, Deserialize)]
struct BufferSnapshot {
    samples: Vec<TrainingSample>,
    index: usize,
    capacity: usize,
}

/// FIFO ring-buffer of training samples with random-access batch sampling.
///
/// Pre-allocates up to `capacity` slots. Once full, new samples overwrite the
/// oldest entries (ring-buffer semantics via `index % capacity`).
pub struct ReplayBuffer {
    samples: Vec<TrainingSample>,
    /// Total number of samples ever added (may exceed capacity).
    index: usize,
    capacity: usize,
    /// Number of slots currently occupied (capped at `capacity`).
    size: usize,
}

impl ReplayBuffer {
    /// Create a new empty buffer that can hold up to `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            index: 0,
            capacity,
            size: 0,
        }
    }

    /// Append a single sample. Overwrites the oldest entry when full.
    pub fn add(&mut self, sample: TrainingSample) {
        let slot = self.index % self.capacity;
        if slot < self.samples.len() {
            self.samples[slot] = sample;
        } else {
            // Still filling up — push rather than index.
            self.samples.push(sample);
        }
        self.index += 1;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// Append all samples from a completed game (all share the same outcome).
    pub fn add_game(&mut self, samples: Vec<TrainingSample>) {
        for s in samples {
            self.add(s);
        }
    }

    /// Draw `batch_size` samples uniformly at random (with replacement).
    ///
    /// Returns at most `min(batch_size, self.size)` references.
    /// Returns an empty vec if the buffer is empty.
    pub fn sample_batch<R: Rng>(&self, batch_size: usize, rng: &mut R) -> Vec<&TrainingSample> {
        if self.size == 0 {
            return Vec::new();
        }
        let n = batch_size.min(self.size);
        (0..n)
            .map(|_| {
                let idx = rng.gen_range(0..self.size);
                &self.samples[idx]
            })
            .collect()
    }

    /// Serialize the buffer contents to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), anyhow::Error> {
        let snapshot = BufferSnapshot {
            samples: self.samples[..self.size].to_vec(),
            index: self.index,
            capacity: self.capacity,
        };
        let file = fs::File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &snapshot)?;
        Ok(())
    }

    /// Deserialize a buffer from a JSON file produced by [`save`].
    pub fn load(path: &Path) -> Result<Self, anyhow::Error> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: BufferSnapshot = serde_json::from_reader(reader)?;
        let size = snapshot.samples.len();
        Ok(Self {
            samples: snapshot.samples,
            index: snapshot.index,
            capacity: snapshot.capacity,
            size,
        })
    }

    /// Number of samples currently stored.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Maximum number of samples the buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether the buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_sample(value: f32) -> TrainingSample {
        TrainingSample {
            observation: vec![value; OBSERVATION_DIM],
            policy: vec![0.0; ACTION_SPACE_SIZE],
            value,
            legal_mask: vec![1.0; ACTION_SPACE_SIZE],
        }
    }

    #[test]
    fn add_increases_len() {
        let mut buf = ReplayBuffer::new(10);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        buf.add(make_sample(1.0));
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());

        buf.add(make_sample(0.0));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn fifo_eviction() {
        let mut buf = ReplayBuffer::new(3);
        buf.add(make_sample(1.0));
        buf.add(make_sample(2.0));
        buf.add(make_sample(3.0));
        assert_eq!(buf.len(), 3);

        // Adding a 4th sample should evict the first (value=1.0).
        buf.add(make_sample(4.0));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.capacity(), 3);

        // Slot 0 should now hold value=4.0 (overwrote value=1.0).
        assert_eq!(buf.samples[0].value, 4.0);
        assert_eq!(buf.samples[1].value, 2.0);
        assert_eq!(buf.samples[2].value, 3.0);
    }

    #[test]
    fn sample_batch_returns_correct_count() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..50 {
            buf.add(make_sample(i as f32));
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let batch = buf.sample_batch(10, &mut rng);
        assert_eq!(batch.len(), 10);

        // Requesting more than buffer size should cap.
        let batch = buf.sample_batch(200, &mut rng);
        assert_eq!(batch.len(), 50);
    }

    #[test]
    fn sample_batch_empty_buffer() {
        let buf = ReplayBuffer::new(10);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let batch = buf.sample_batch(5, &mut rng);
        assert!(batch.is_empty());
    }

    #[test]
    fn add_game_works() {
        let mut buf = ReplayBuffer::new(100);
        let samples = vec![make_sample(1.0), make_sample(1.0), make_sample(1.0)];
        buf.add_game(samples);
        assert_eq!(buf.len(), 3);
        for s in &buf.samples {
            assert_eq!(s.value, 1.0);
        }
    }

    #[test]
    fn save_load_round_trip() {
        let mut buf = ReplayBuffer::new(10);
        buf.add(make_sample(0.5));
        buf.add(make_sample(1.0));
        buf.add(make_sample(0.0));

        let dir = std::env::temp_dir().join("klondike_buffer_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_buffer.json");

        buf.save(&path).expect("save should succeed");

        let loaded = ReplayBuffer::load(&path).expect("load should succeed");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.capacity(), 10);
        assert_eq!(loaded.samples[0].value, 0.5);
        assert_eq!(loaded.samples[1].value, 1.0);
        assert_eq!(loaded.samples[2].value, 0.0);
        // Index is preserved so FIFO continues correctly after reload.
        assert_eq!(loaded.index, buf.index);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_load_after_eviction() {
        let mut buf = ReplayBuffer::new(3);
        for i in 0..5 {
            buf.add(make_sample(i as f32));
        }
        assert_eq!(buf.len(), 3);

        let dir = std::env::temp_dir().join("klondike_buffer_evict_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_buffer.json");

        buf.save(&path).expect("save should succeed");
        let loaded = ReplayBuffer::load(&path).expect("load should succeed");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.index, 5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_from_arrays() {
        let obs = [0.5f32; OBSERVATION_DIM];
        let pol = [1.0 / ACTION_SPACE_SIZE as f32; ACTION_SPACE_SIZE];
        let mask = [1.0f32; ACTION_SPACE_SIZE];
        let sample = TrainingSample::new(obs, pol, 1.0, mask);
        assert_eq!(sample.observation.len(), OBSERVATION_DIM);
        assert_eq!(sample.policy.len(), ACTION_SPACE_SIZE);
        assert_eq!(sample.legal_mask.len(), ACTION_SPACE_SIZE);
        assert_eq!(sample.value, 1.0);
    }
}
