// Integration test: 100 parallel self-play games -> buffer -> save/load.
// Validates the full Phase R4 pipeline end-to-end.

use std::time::Instant;

use klondike_search::eval_fn::UniformEval;
use klondike_train::buffer::ReplayBuffer;
use klondike_train::selfplay::{play_games_parallel, SelfPlayConfig};
use rand::SeedableRng;

#[test]
fn test_100_games_buffer_save_load() {
    let config = SelfPlayConfig {
        n_determinizations: 2,
        mcts_simulations: 20,
        max_moves: 100,
        dirichlet_frac: 0.0, // No noise for speed
        ..SelfPlayConfig::default()
    };
    let eval = UniformEval;
    let seeds: Vec<u64> = (0..100).collect();

    let start = Instant::now();
    let games = play_games_parallel(&config, &eval, &seeds);
    let elapsed = start.elapsed();

    assert_eq!(games.len(), 100);

    let mut wins = 0;
    let mut total_samples = 0;

    // Add all games to buffer
    let mut buffer = ReplayBuffer::new(200_000);
    for game in &games {
        assert!(game.n_moves > 0, "seed {} should play moves", game.seed);
        assert_eq!(game.samples.len(), game.n_moves);
        if game.won {
            wins += 1;
        }
        total_samples += game.samples.len();
        buffer.add_game(game.samples.clone());
    }

    assert_eq!(buffer.len(), total_samples);

    // Save and load round-trip
    let dir = std::env::temp_dir().join("klondike_integration_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("buffer.json");

    buffer.save(&path).expect("save should succeed");
    let loaded = ReplayBuffer::load(&path).expect("load should succeed");

    assert_eq!(loaded.len(), buffer.len());
    assert_eq!(loaded.capacity(), buffer.capacity());

    // Sample a batch from loaded buffer
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let batch = loaded.sample_batch(64, &mut rng);
    assert_eq!(batch.len(), 64.min(loaded.len()));

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);

    eprintln!(
        "100 games: {wins}/100 wins, {total_samples} samples, {:.2}s ({:.0}ms/game)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / 100.0
    );

    // Performance check: 100 games with light config should complete in under 30s
    assert!(
        elapsed.as_secs() < 30,
        "100 games took {}s, expected <30s",
        elapsed.as_secs()
    );
}
