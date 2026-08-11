//! Benchmark sweep harness: simulate speculative-decoding rounds, measure the
//! empirical acceptance rate and speedup, and cross-check against the
//! closed-form [`crate::analytical`] model.
//!
//! ## Constructing an exact acceptance rate
//!
//! The per-token acceptance probability of speculative sampling is
//! `α = Σ_x min(p(x), q(x)) = 1 - TV(p, q)`, the overlap between the target and
//! draft distributions. To sweep `α` as a clean independent variable we build a
//! target/draft pair with *exactly* that overlap using a two-block construction:
//!
//! - The target `p` places all its mass on the first half of the vocabulary.
//! - The draft `q = α·p + (1-α)·u`, where `u` is uniform over the **second**
//!   half (a support disjoint from `p`).
//!
//! Then `Σ min(p, q) = Σ_{first half} min(p, α·p) = α`, so every drafted token is
//! accepted with probability exactly `α`. On rejection the residual
//! `norm(max(0, p - q))` reduces to `p`, so the loop stays distribution
//! preserving. This lets the empirical measurement be compared against the
//! analytical formula fed the same `α`.

use crate::analytical::{expected_accepted_tokens, expected_speedup, optimal_k};
use crate::models::{Model, SyntheticModel};
use crate::rng::Rng;
use crate::sampling::{sample_from, speculative_step, SamplingError, StepOutcome};
use std::time::Duration;

/// Configuration for a benchmark sweep.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Vocabulary size (must be at least 2 so the two-block split is non-empty).
    pub vocab: usize,
    /// Draft lengths to sweep, inclusive range endpoints `(k_min, k_max)`.
    pub k_range: (u32, u32),
    /// Target acceptance rates to sweep.
    pub alphas: Vec<f64>,
    /// Per-call cost of the draft relative to the target, `c_draft / c_target`.
    pub cost_ratio: f64,
    /// Number of simulated verification rounds per `(k, α)` cell.
    pub rounds: u64,
    /// Base RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            vocab: 64,
            k_range: (1, 8),
            alphas: vec![0.3, 0.5, 0.7, 0.9],
            cost_ratio: 0.2,
            rounds: 200_000,
            seed: 0xB0BA_CAFE,
        }
    }
}

/// One row of sweep results for a `(k, α)` cell.
#[derive(Debug, Clone, Copy)]
pub struct SweepRow {
    /// Draft length.
    pub k: u32,
    /// The acceptance rate the pair was constructed to have.
    pub alpha_target: f64,
    /// The acceptance rate actually measured from the simulation.
    pub alpha_measured: f64,
    /// Empirically measured tokens emitted per round.
    pub empirical_tokens: f64,
    /// Analytical expected tokens per round for `alpha_target`.
    pub analytical_tokens: f64,
    /// Empirically measured speedup over autoregressive decoding.
    pub empirical_speedup: f64,
    /// Analytical expected speedup for `alpha_target`.
    pub analytical_speedup: f64,
}

/// Full sweep output plus the optimal `k` per acceptance rate.
#[derive(Debug, Clone)]
pub struct SweepResult {
    /// All `(k, α)` rows.
    pub rows: Vec<SweepRow>,
    /// The optimal draft length per acceptance rate: `(alpha, best_k, speedup)`,
    /// computed from the analytical speedup.
    pub optimal_per_alpha: Vec<(f64, u32, f64)>,
    /// The cost ratio used for the sweep.
    pub cost_ratio: f64,
}

/// Build a target/draft pair whose per-token acceptance rate is exactly `alpha`.
///
/// Returns `(p, q)` distributions over `vocab` tokens. `vocab` is treated as at
/// least 2. See the module docs for the two-block construction.
#[must_use]
pub fn build_alpha_pair(vocab: usize, alpha: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let vocab = vocab.max(2);
    let alpha = alpha.clamp(0.0, 1.0);
    let half = vocab / 2;
    let second_len = vocab - half;

    // Target p: a peaked distribution over the FIRST half only.
    let mut rng = Rng::new(seed);
    let mut p = vec![0.0_f64; vocab];
    let mut p_total = 0.0;
    for slot in p.iter_mut().take(half) {
        let u = rng.next_f64();
        let w = u * u + 1e-6;
        *slot = w;
        p_total += w;
    }
    for slot in p.iter_mut().take(half) {
        *slot /= p_total;
    }

    // Draft q = alpha * p + (1 - alpha) * uniform(second half).
    let mut q = vec![0.0_f64; vocab];
    let u_mass = if second_len > 0 {
        (1.0 - alpha) / second_len as f64
    } else {
        0.0
    };
    for i in 0..vocab {
        if i < half {
            q[i] = alpha * p[i];
        } else {
            q[i] = u_mass;
        }
    }
    (p, q)
}

/// Aggregate counters from simulating rounds for a single `(k, α)` cell.
struct RoundStats {
    tokens_emitted: u64,
    positions_examined: u64,
    positions_accepted: u64,
}

/// Simulate `rounds` speculative-decoding verification rounds for a `(p, q)`
/// pair and draft length `k`.
fn simulate_rounds(
    p: &[f64],
    q: &[f64],
    k: u32,
    rounds: u64,
    rng: &mut Rng,
) -> Result<RoundStats, SamplingError> {
    let mut tokens_emitted = 0u64;
    let mut positions_examined = 0u64;
    let mut positions_accepted = 0u64;

    for _ in 0..rounds {
        let mut accepted_in_round = 0u32;
        let mut rejected = false;
        for _ in 0..k {
            let x = sample_from(q, rng)?;
            positions_examined += 1;
            let (_tok, outcome) = speculative_step(p, q, x, rng)?;
            match outcome {
                StepOutcome::Accepted => {
                    positions_accepted += 1;
                    accepted_in_round += 1;
                }
                StepOutcome::Resampled => {
                    // Residual resample yields the bonus token; round ends.
                    rejected = true;
                    break;
                }
            }
        }
        // Bonus token: on rejection the residual already produced it; if all k
        // were accepted the target emits one extra token from p.
        if !rejected {
            let _bonus = sample_from(p, rng)?;
        }
        tokens_emitted += u64::from(accepted_in_round) + 1;
    }

    Ok(RoundStats {
        tokens_emitted,
        positions_examined,
        positions_accepted,
    })
}

/// Run the full `k × α` sweep described by `config`.
///
/// # Errors
///
/// Propagates any [`SamplingError`] from the underlying simulation (which should
/// not occur for well-formed configs).
pub fn run_sweep(config: &BenchConfig) -> Result<SweepResult, SamplingError> {
    let (k_min, k_max) = config.k_range;
    let mut rows = Vec::new();

    for (ai, &alpha) in config.alphas.iter().enumerate() {
        for k in k_min..=k_max {
            // Distinct, deterministic seed per (alpha, k) cell.
            let cell_seed = config
                .seed
                .wrapping_add((ai as u64) << 32)
                .wrapping_add(u64::from(k).wrapping_mul(0x9E37_79B9));
            let (p, q) = build_alpha_pair(config.vocab, alpha, cell_seed);
            let mut rng = Rng::new(cell_seed ^ 0xDEAD_BEEF);
            let stats = simulate_rounds(&p, &q, k, config.rounds, &mut rng)?;

            let alpha_measured = if stats.positions_examined > 0 {
                stats.positions_accepted as f64 / stats.positions_examined as f64
            } else {
                0.0
            };
            let empirical_tokens = stats.tokens_emitted as f64 / config.rounds as f64;
            let analytical_tokens = expected_accepted_tokens(alpha, k);
            let round_cost = f64::from(k) * config.cost_ratio + 1.0;
            let empirical_speedup = empirical_tokens / round_cost;
            let analytical_speedup = expected_speedup(alpha, k, config.cost_ratio);

            rows.push(SweepRow {
                k,
                alpha_target: alpha,
                alpha_measured,
                empirical_tokens,
                analytical_tokens,
                empirical_speedup,
                analytical_speedup,
            });
        }
    }

    let mut optimal_per_alpha = Vec::new();
    for &alpha in &config.alphas {
        if let Some((k, s)) = optimal_k(alpha, config.cost_ratio, k_max) {
            optimal_per_alpha.push((alpha, k, s));
        }
    }

    Ok(SweepResult {
        rows,
        optimal_per_alpha,
        cost_ratio: config.cost_ratio,
    })
}

impl SweepResult {
    /// Render the sweep as a human-readable table.
    #[must_use]
    pub fn to_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:>3}  {:>7}  {:>9}  {:>10}  {:>10}  {:>10}  {:>10}\n",
            "k", "alpha", "alpha_hat", "emp_tok", "ana_tok", "emp_spd", "ana_spd"
        ));
        out.push_str(&"-".repeat(72));
        out.push('\n');
        for r in &self.rows {
            out.push_str(&format!(
                "{:>3}  {:>7.2}  {:>9.4}  {:>10.4}  {:>10.4}  {:>9.3}x  {:>9.3}x\n",
                r.k,
                r.alpha_target,
                r.alpha_measured,
                r.empirical_tokens,
                r.analytical_tokens,
                r.empirical_speedup,
                r.analytical_speedup,
            ));
        }
        out.push('\n');
        out.push_str(&format!(
            "Optimal k per acceptance rate (cost_ratio = {:.3}):\n",
            self.cost_ratio
        ));
        for (alpha, k, s) in &self.optimal_per_alpha {
            out.push_str(&format!(
                "  alpha = {alpha:.2}  ->  k* = {k}  (analytical speedup {s:.3}x)\n"
            ));
        }
        out
    }

    /// Render the sweep rows as CSV (with a header).
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "k,alpha_target,alpha_measured,empirical_tokens,analytical_tokens,empirical_speedup,analytical_speedup\n",
        );
        for r in &self.rows {
            out.push_str(&format!(
                "{},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                r.k,
                r.alpha_target,
                r.alpha_measured,
                r.empirical_tokens,
                r.analytical_tokens,
                r.empirical_speedup,
                r.analytical_speedup,
            ));
        }
        out
    }
}

/// Measure the empirical per-token acceptance rate for a [`SyntheticModel`]
/// draft/target pair over `rounds` single-token trials.
///
/// This exercises the tunable `agreement` knob end to end (unlike the exact-α
/// construction used by the sweep) and is used by tests to show the measured
/// acceptance rate moves monotonically with agreement.
///
/// # Errors
///
/// Propagates [`SamplingError`] from the underlying sampling calls.
pub fn measure_acceptance_rate(
    target: &SyntheticModel,
    draft: &SyntheticModel,
    trials: u64,
    seed: u64,
) -> Result<f64, SamplingError> {
    let mut rng = Rng::new(seed);
    let mut accepted = 0u64;
    let context = 0usize;
    let p = target.next_distribution(context);
    let q = draft.next_distribution(context);
    for _ in 0..trials {
        let x = sample_from(&q, &mut rng)?;
        let (_tok, outcome) = speculative_step(&p, &q, x, &mut rng)?;
        if outcome == StepOutcome::Accepted {
            accepted += 1;
        }
    }
    Ok(accepted as f64 / trials.max(1) as f64)
}

/// Convert a draft/target latency pair into the dimensionless cost ratio used by
/// the analytical model.
#[must_use]
pub fn cost_ratio_from_latencies(draft: Duration, target: Duration) -> f64 {
    let t = target.as_secs_f64();
    if t <= 0.0 {
        0.0
    } else {
        draft.as_secs_f64() / t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_pair_has_requested_overlap() {
        for &alpha in &[0.3, 0.5, 0.7, 0.9] {
            let (p, q) = build_alpha_pair(64, alpha, 123);
            let overlap: f64 = p.iter().zip(q.iter()).map(|(&a, &b)| a.min(b)).sum();
            assert!(
                (overlap - alpha).abs() < 1e-9,
                "alpha={alpha} overlap={overlap}"
            );
            let ps: f64 = p.iter().sum();
            let qs: f64 = q.iter().sum();
            assert!((ps - 1.0).abs() < 1e-9);
            assert!((qs - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn small_sweep_runs() {
        let cfg = BenchConfig {
            rounds: 2000,
            ..BenchConfig::default()
        };
        let res = run_sweep(&cfg).expect("sweep");
        assert_eq!(res.rows.len(), 4 * 8);
        assert_eq!(res.optimal_per_alpha.len(), 4);
    }
}
