# klondike-rs — AlphaZero Klondike Solitaire (Pure Rust)

## Commands
- Check: `cd klondike-rs && cargo check`
- Build: `cd klondike-rs && cargo build --release`
- Test all: `cd klondike-rs && cargo test --all`
- Test crate: `cd klondike-rs && cargo test -p klondike-core`
- Lint: `cd klondike-rs && cargo clippy --all-targets -- -D warnings`
- Format: `cd klondike-rs && cargo fmt --all -- --check`
- ALWAYS run `cargo check` after code changes before declaring done
- Run `cargo clippy` before completing any phase

## Conventions
- Error handling: `anyhow` for binaries, `Result<T, E>` with descriptive errors for library crates
- No `.unwrap()` in library crates; `.expect("reason")` only for true invariants
- Propagate errors with `?`; add context with `.context("what failed")`
- Avoid unnecessary `.clone()` — restructure ownership instead, especially in hot paths (MCTS loop, encoding, self-play)
- Prefer `&str` over `String` in function args, `&[T]` over `Vec<T>` when not consuming
- Define struct/trait signatures first, implement methods second
- Use `for` loops with mutable accumulators over complex iterator chains (easier to debug, add logging)
- Explicit `match` arms over wildcards — catch new variants at compile time

## Architecture
5-crate workspace: klondike-core (env/encoding/actions) -> klondike-search (MCTS/PIMC) -> klondike-nn (tch-rs model) -> klondike-eval (stats/benchmark) -> klondike-train (buffer/selfplay/pipeline/CLI)
Lonelybot is a direct path dependency — no PyO3, no FFI overhead.

## Testing
- Unit tests in same file (`#[cfg(test)] mod tests`)
- Integration tests in workspace `tests/` directory
- Encoding parity test (vs Python) is the single most critical test
- Fixed-seed deterministic tests for MCTS reproducibility
