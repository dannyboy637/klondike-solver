// Convert checkpoint weights between formats and verify architecture match.
// Handles .safetensors and tch-compatible .pt files.
// Python pickle .pt files require manual conversion via safetensors first.

use anyhow::{Context, Result};
use clap::Parser;
use klondike_nn::model::{KlondikeNet, NetConfig};
use tch::{nn, Device};

#[derive(Parser)]
#[command(
    name = "convert-weights",
    about = "Convert and verify model checkpoints"
)]
struct Args {
    /// Input checkpoint path (.safetensors or .pt)
    #[arg(long)]
    input: String,

    /// Output checkpoint path
    #[arg(long)]
    output: String,

    /// Hidden dimension (must match input checkpoint)
    #[arg(long, default_value_t = 512)]
    hidden_dim: i64,

    /// Number of residual blocks (must match input checkpoint)
    #[arg(long, default_value_t = 4)]
    n_blocks: i64,

    /// Disable auxiliary heads
    #[arg(long)]
    no_auxiliary: bool,

    /// Only print expected parameter names and shapes (don't load/save)
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let config = NetConfig {
        hidden_dim: args.hidden_dim,
        n_blocks: args.n_blocks,
        use_auxiliary: !args.no_auxiliary,
        ..Default::default()
    };

    // Create model on CPU for portability
    let mut vs = nn::VarStore::new(Device::Cpu);
    let _net = KlondikeNet::new(&vs.root(), &config);

    let param_count = KlondikeNet::count_parameters(&vs);

    // Print expected architecture
    println!("Architecture:");
    println!("  input_dim:  {}", config.input_dim);
    println!("  hidden_dim: {}", config.hidden_dim);
    println!("  n_blocks:   {}", config.n_blocks);
    println!("  action_dim: {}", config.action_dim);
    println!("  auxiliary:   {}", config.use_auxiliary);
    println!("  parameters: {}", param_count);
    println!();

    // Print all parameter names and shapes
    println!("Expected parameters:");
    let mut variables: Vec<(String, Vec<i64>)> = vs
        .variables()
        .iter()
        .map(|(k, v)| (k.clone(), v.size()))
        .collect();
    variables.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, shape) in &variables {
        println!("  {:<35} {:?}", name, shape);
    }
    println!();

    if args.dry_run {
        println!("Dry run — no files loaded or saved.");
        return Ok(());
    }

    // Try to load input weights
    println!("Loading: {}", args.input);
    match vs.load(&args.input) {
        Ok(()) => {
            println!("Loaded successfully.");
        }
        Err(e) => {
            eprintln!("Failed to load checkpoint: {}", e);
            eprintln!();
            eprintln!("If the input is a Python pickle .pt file, convert it first:");
            eprintln!();
            eprintln!("  python -c \"");
            eprintln!("  import torch");
            eprintln!("  from safetensors.torch import save_file");
            eprintln!(
                "  state = torch.load('{}', map_location='cpu', weights_only=True)",
                args.input
            );
            eprintln!("  if 'model_state_dict' in state:");
            eprintln!("      state = state['model_state_dict']");
            eprintln!(
                "  save_file(state, '{}.safetensors')",
                args.input.trim_end_matches(".pt")
            );
            eprintln!("  \"");
            eprintln!();
            eprintln!("Then re-run with the .safetensors file as input.");
            return Err(e).context("Checkpoint load failed");
        }
    }

    // Save to output
    println!("Saving: {}", args.output);
    vs.save(&args.output)
        .with_context(|| format!("Failed to save to {}", args.output))?;

    println!();
    println!(
        "Converted successfully: {} parameters, {} -> {}",
        param_count, args.input, args.output
    );

    Ok(())
}
