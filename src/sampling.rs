//! Core speculative-sampling primitives: the accept/reject rule and the
//! residual (renormalized) distribution used on rejection.
//!
//! This module implements the exact acceptance rule from Leviathan et al.,
//! *"Fast Inference from Transformers via Speculative Decoding"* and Chen et
//! al., *"Accelerating Large Language Model Decoding with Speculative
//! Sampling"*:
//!
//! 1. The draft model produces distribution `q` and samples a token `x`.
//! 2. The target model produces distribution `p`.
//! 3. Accept `x` with probability `min(1, p(x) / q(x))`.
//! 4. On rejection, resample from the **residual distribution**
//!    `norm(max(0, p - q))`.
//!
//! The remarkable property this buys us is that the resulting samples are
//! distributed **exactly** according to the target distribution `p`, regardless
//! of how bad the draft `q` is. That distribution-preservation guarantee is the
//! whole point of speculative decoding: a faster decoder that changes the output
//! distribution is simply a broken decoder.

use crate::rng::Rng;

/// Numerical tolerance below which a residual distribution is treated as
/// all-zero (the degenerate case where `p <= q` everywhere they overlap).
const RESIDUAL_EPS: f64 = 1e-12;

/// Errors that can arise from the sampling primitives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingError {
    /// Two distributions passed together had mismatched lengths.
    LengthMismatch,
    /// A distribution was empty.
    Empty,
    /// A distribution had non-positive total mass and could not be sampled.
    DegenerateDistribution,
}

impl std::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch => write!(f, "distributions have mismatched lengths"),
            Self::Empty => write!(f, "distribution is empty"),
            Self::DegenerateDistribution => {
                write!(f, "distribution has non-positive total mass")
            }
        }
    }
}

impl std::error::Error for SamplingError {}

/// Compute the residual distribution `norm(max(0, p - q))`.
///
/// This is the distribution the target model resamples from when the draft
/// token is rejected. Elementwise we take `max(0, p[i] - q[i])`, then normalize
/// so the result sums to 1.
///
/// # The degenerate case
///
/// If `p[i] <= q[i]` for every `i` (so the elementwise max is all zeros), there
/// is no positive residual mass to normalize. This can happen when `q == p`
/// exactly, or when `q` dominates `p` everywhere. In that situation we fall back
/// to the target distribution `p` itself (normalized). This is the correct
/// behavior: mathematically the rejection branch is entered with probability
/// zero in the `q == p` case, so any proper distribution is acceptable there;
/// returning `norm(p)` keeps the output well-defined and distribution-preserving.
///
/// # Errors
///
/// Returns [`SamplingError::LengthMismatch`] if `p` and `q` differ in length,
/// [`SamplingError::Empty`] if they are empty, and
/// [`SamplingError::DegenerateDistribution`] if `p` itself has non-positive mass
/// (so even the fallback is undefined).
pub fn residual_distribution(p: &[f64], q: &[f64]) -> Result<Vec<f64>, SamplingError> {
    if p.len() != q.len() {
        return Err(SamplingError::LengthMismatch);
    }
    if p.is_empty() {
        return Err(SamplingError::Empty);
    }

    let mut residual: Vec<f64> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).max(0.0))
        .collect();

    let total: f64 = residual.iter().sum();

    if total <= RESIDUAL_EPS {
        // All-zero residual: fall back to the (normalized) target distribution.
        let p_total: f64 = p.iter().map(|&x| x.max(0.0)).sum();
        if p_total <= RESIDUAL_EPS {
            return Err(SamplingError::DegenerateDistribution);
        }
        return Ok(p.iter().map(|&x| x.max(0.0) / p_total).collect());
    }

    for r in &mut residual {
        *r /= total;
    }
    Ok(residual)
}

/// Outcome of a single speculative accept/reject step, useful for testing and
/// telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// The draft token was accepted.
    Accepted,
    /// The draft token was rejected and a token was resampled from the residual.
    Resampled,
}

/// Perform one speculative-sampling step for a single draft token.
///
/// Given the target distribution `p`, the draft distribution `q`, and the token
/// `x` that the draft *already sampled* from `q`, this returns the token that a
/// distribution-preserving speculative decoder emits, along with whether the
/// draft token was accepted or a residual resample occurred.
///
/// The acceptance test draws `u ~ U[0,1)` and accepts iff `u < p(x) / q(x)`
/// (equivalently, accepts with probability `min(1, p(x)/q(x))`). On rejection we
/// resample from [`residual_distribution`].
///
/// # Errors
///
/// Propagates errors from [`residual_distribution`], and returns
/// [`SamplingError::DegenerateDistribution`] if a residual resample is required
/// but the residual cannot be sampled.
pub fn speculative_step(
    p: &[f64],
    q: &[f64],
    x: usize,
    rng: &mut Rng,
) -> Result<(usize, StepOutcome), SamplingError> {
    if p.len() != q.len() {
        return Err(SamplingError::LengthMismatch);
    }
    if x >= p.len() {
        return Err(SamplingError::LengthMismatch);
    }

    let px = p[x];
    let qx = q[x];

    // Acceptance probability min(1, p(x)/q(x)). If q(x) == 0 the draft could not
    // have produced x under a correct sampler, but be defensive: treat a
    // positive p(x) as certain acceptance.
    let accept_prob = if qx <= 0.0 {
        if px > 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (px / qx).min(1.0)
    };

    let u = rng.next_f64();
    if u < accept_prob {
        return Ok((x, StepOutcome::Accepted));
    }

    // Rejected: resample from the residual distribution.
    let residual = residual_distribution(p, q)?;
    let token = rng
        .sample_categorical(&residual)
        .ok_or(SamplingError::DegenerateDistribution)?;
    Ok((token, StepOutcome::Resampled))
}

/// Draw a single token directly from distribution `p` (plain sampling).
///
/// This is the reference the speculative decoder must match distributionally.
///
/// # Errors
///
/// Returns [`SamplingError::DegenerateDistribution`] if `p` cannot be sampled.
pub fn sample_from(p: &[f64], rng: &mut Rng) -> Result<usize, SamplingError> {
    rng.sample_categorical(p)
        .ok_or(SamplingError::DegenerateDistribution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn residual_sums_to_one() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.2, 0.5, 0.3];
        let r = residual_distribution(&p, &q).expect("residual");
        let sum: f64 = r.iter().sum();
        assert!(approx(sum, 1.0, 1e-12), "residual sum was {sum}");
        // Only index 0 has positive p-q (0.3); others are clamped to zero.
        assert!(approx(r[0], 1.0, 1e-12));
        assert!(approx(r[1], 0.0, 1e-12));
        assert!(approx(r[2], 0.0, 1e-12));
    }

    #[test]
    fn residual_all_zero_falls_back_to_p() {
        // q == p everywhere => residual is all zeros => fall back to norm(p).
        let p = vec![0.25, 0.25, 0.5];
        let q = p.clone();
        let r = residual_distribution(&p, &q).expect("residual fallback");
        let sum: f64 = r.iter().sum();
        assert!(approx(sum, 1.0, 1e-12));
        assert!(approx(r[0], 0.25, 1e-12));
        assert!(approx(r[2], 0.5, 1e-12));
    }

    #[test]
    fn residual_q_dominates_p_falls_back() {
        // q strictly dominates p everywhere they differ; residual all zero.
        let p = vec![0.5, 0.5];
        let q = vec![0.9, 0.1];
        // index 0: max(0, -0.4)=0 ; index 1: max(0, 0.4)=0.4 -> not all zero.
        let r = residual_distribution(&p, &q).expect("residual");
        assert!(approx(r[1], 1.0, 1e-12));
    }

    #[test]
    fn residual_length_mismatch_errors() {
        let p = vec![0.5, 0.5];
        let q = vec![1.0];
        assert_eq!(
            residual_distribution(&p, &q),
            Err(SamplingError::LengthMismatch)
        );
    }

    #[test]
    fn residual_empty_errors() {
        let p: Vec<f64> = vec![];
        let q: Vec<f64> = vec![];
        assert_eq!(residual_distribution(&p, &q), Err(SamplingError::Empty));
    }

    #[test]
    fn accept_when_p_ge_q() {
        // p(x) >= q(x) => acceptance probability is 1, always accepted.
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let mut rng = Rng::new(1);
        for _ in 0..100 {
            let (tok, outcome) = speculative_step(&p, &q, 0, &mut rng).expect("step");
            assert_eq!(tok, 0);
            assert_eq!(outcome, StepOutcome::Accepted);
        }
    }
}
