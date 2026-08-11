//! # Speculative Decoding Benchmark
//!
//! A simulation harness and benchmark suite for **speculative decoding**, the
//! LLM-inference acceleration technique where a small, cheap *draft* model
//! proposes `k` tokens ahead and a large *target* model verifies them in a
//! single batched forward pass, accepting the longest correct prefix.
//!
//! This crate models the algorithm's **correctness** and **performance**
//! characteristics without requiring GPUs or real model weights:
//!
//! - [`sampling`] implements the exact accept/reject rule and the residual
//!   distribution that make speculative decoding provably distribution
//!   preserving.
//! - [`models`] provides configurable draft/target simulators with a tunable
//!   *agreement* parameter and simulated per-call latency.
//! - [`analytical`] implements the closed-form expected accepted tokens and
//!   expected speedup as functions of acceptance rate `α` and draft length `k`.
//! - [`bench`] sweeps `k × α`, measures empirical speedup, and finds the optimal
//!   `k` per acceptance rate.
//! - [`rng`] is a deterministic, std-only PRNG so every result is reproducible.
//!
//! ## Why distribution preservation is the core correctness property
//!
//! Speculative decoding is only useful if it produces the *same* output
//! distribution as sampling from the target model directly. A decoder that is
//! faster but changes the distribution is simply a broken decoder. The
//! accept/reject/residual construction guarantees the emitted tokens are
//! distributed exactly according to the target `p`, for *any* draft `q`.

pub mod analytical;
pub mod bench;
pub mod models;
pub mod rng;
pub mod sampling;

pub use analytical::{expected_accepted_tokens, expected_speedup};
pub use bench::{run_sweep, BenchConfig, SweepResult};
pub use models::{Model, SyntheticModel};
pub use rng::Rng;
pub use sampling::{
    residual_distribution, sample_from, speculative_step, SamplingError, StepOutcome,
};
