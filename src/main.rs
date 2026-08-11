//! CLI entry point for the speculative-decoding benchmark.
//!
//! Runs the default `k × α` sweep and prints the results table. A fuller CLI
//! (flags, CSV export path) is added in a later iteration.

use speculative_decoding_bench::bench::{run_sweep, BenchConfig};

fn main() {
    let config = BenchConfig::default();
    match run_sweep(&config) {
        Ok(result) => {
            print!("{}", result.to_table());
        }
        Err(e) => {
            eprintln!("sweep failed: {e}");
            std::process::exit(1);
        }
    }
}
