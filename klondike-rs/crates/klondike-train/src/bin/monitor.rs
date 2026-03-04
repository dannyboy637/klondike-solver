// Training log viewer — reads training_log.json and prints formatted summary.
// Ported from scripts/monitor_training.py.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "monitor", about = "View AlphaZero training progress")]
struct Args {
    /// Checkpoint directory containing training_log.json
    #[arg(default_value = "checkpoints/alphazero_rs")]
    checkpoint_dir: PathBuf,
}

/// Training log entry — compatible with pipeline.rs IterationLog.
#[derive(Deserialize)]
struct IterationLog {
    iteration: usize,
    selfplay_win_rate: f64,
    #[allow(dead_code)]
    selfplay_positions: Option<usize>,
    avg_game_time: Option<f64>,
    policy_loss: f64,
    value_loss: f64,
    buffer_size: usize,
    iter_time: Option<f64>,
    eval_win_rate: Option<f64>,
    eval_ci_lower: Option<f64>,
    eval_ci_upper: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let log_path = args.checkpoint_dir.join("training_log.json");

    let log_str = fs::read_to_string(&log_path)
        .with_context(|| format!("Failed to read {}", log_path.display()))?;

    let log: Vec<IterationLog> =
        serde_json::from_str(&log_str).context("Failed to parse training_log.json")?;

    if log.is_empty() {
        println!("No training iterations recorded yet.");
        return Ok(());
    }

    // Header
    println!(
        "{:>4} {:>8} {:>8} {:>8} {:>8} {:>7} {:>10}",
        "Iter", "SP Win%", "P Loss", "V Loss", "Buffer", "s/game", "Eval"
    );
    println!("{}", "-".repeat(65));

    // Rows
    for entry in &log {
        let sp = entry.selfplay_win_rate * 100.0;
        let spd = entry.avg_game_time.unwrap_or(0.0);

        let eval_str = match (
            entry.eval_win_rate,
            entry.eval_ci_lower,
            entry.eval_ci_upper,
        ) {
            (Some(ew), Some(lo), Some(hi)) => {
                format!("{:5.1}% ({:.0}-{:.0})", ew * 100.0, lo * 100.0, hi * 100.0)
            }
            _ => String::new(),
        };

        println!(
            "{:4} {:7.1}% {:8.4} {:8.4} {:8} {:6.1}s {}",
            entry.iteration,
            sp,
            entry.policy_loss,
            entry.value_loss,
            entry.buffer_size,
            spd,
            eval_str
        );
    }

    // Summary
    let last = log.last().expect("log is non-empty");
    let total_time: f64 = log.iter().map(|e| e.iter_time.unwrap_or(0.0)).sum();
    let evals: Vec<&IterationLog> = log.iter().filter(|e| e.eval_win_rate.is_some()).collect();

    println!("\n--- Summary ---");
    println!("Iterations: {}", log.len());
    println!("Total time: {:.1} hours", total_time / 3600.0);
    println!("Buffer: {} positions", last.buffer_size);
    println!("Latest policy loss: {:.4}", last.policy_loss);
    println!("Latest value loss: {:.4}", last.value_loss);

    if let Some(best) = evals.iter().max_by(|a, b| {
        a.eval_win_rate
            .unwrap()
            .partial_cmp(&b.eval_win_rate.unwrap())
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "Best eval: {:.1}% at iter {}",
            best.eval_win_rate.unwrap() * 100.0,
            best.iteration
        );
    }

    if log.len() >= 2 {
        let pl_trend = last.policy_loss - log[0].policy_loss;
        let direction = if pl_trend < 0.0 {
            "improving"
        } else {
            "worsening"
        };
        println!("Policy loss trend: {:+.4} ({})", pl_trend, direction);
    }

    Ok(())
}
