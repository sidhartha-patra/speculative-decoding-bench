//! Configurable draft/target model simulators.
//!
//! Real speculative decoding pairs a small *draft* model with a large *target*
//! model. Here we simulate both as objects that produce a next-token
//! distribution over a synthetic vocabulary, with two knobs that matter for the
//! benchmark:
//!
//! - **Simulated latency** (`latency`): the wall-clock cost we attribute to one
//!   forward pass of this model. The draft is cheap; the target is expensive.
//! - **Agreement** (only for the draft): how closely the draft distribution `q`
//!   matches the target distribution `p`. Agreement is a real, tunable
//!   parameter in `[0, 1]` that demonstrably moves the measured acceptance rate:
//!   at `1.0` the draft reproduces the target exactly (acceptance ≈ 1), at `0.0`
//!   the draft is a uniform distribution (low acceptance).

use crate::rng::Rng;
use std::time::Duration;

/// A next-token model producing a categorical distribution over the vocabulary.
///
/// Implemented as a trait so draft and target models are interchangeable trait
/// objects and alternative simulators can be dropped in.
pub trait Model {
    /// The vocabulary size (number of possible next tokens).
    fn vocab_size(&self) -> usize;

    /// The simulated latency of a single forward pass.
    fn latency(&self) -> Duration;

    /// Produce the next-token distribution for a given context token.
    ///
    /// The returned vector has length [`Model::vocab_size`] and sums to 1. The
    /// `context` lets the distribution depend on the previous token so the
    /// synthetic stream is not i.i.d.; simulators may ignore it.
    fn next_distribution(&self, context: usize) -> Vec<f64>;
}

/// A synthetic next-token model over a fixed vocabulary.
///
/// The "true" distribution for a context is a deterministic, peaked categorical
/// derived from a seed and the context token (so it is stable across calls). A
/// draft model is then built from a target model by *blending* toward a uniform
/// distribution according to an `agreement` factor, which lets us dial the
/// acceptance rate up and down as an independent experimental variable.
#[derive(Debug, Clone)]
pub struct SyntheticModel {
    vocab: usize,
    seed: u64,
    latency: Duration,
    /// Blend factor in `[0, 1]`: 1.0 == the base (target) distribution,
    /// 0.0 == uniform. Values in between interpolate.
    agreement: f64,
}

impl SyntheticModel {
    /// Build a *target* model: the reference distribution with a given latency.
    ///
    /// `agreement` is fixed at `1.0` because the target defines the truth.
    #[must_use]
    pub fn target(vocab: usize, seed: u64, latency: Duration) -> Self {
        Self {
            vocab: vocab.max(1),
            seed,
            latency,
            agreement: 1.0,
        }
    }

    /// Build a *draft* model that approximates the target `base`.
    ///
    /// The draft shares the target's base distribution but blends it toward
    /// uniform by `1 - agreement`. `agreement` is clamped to `[0, 1]`. A cheaper
    /// `latency` reflects the draft being a smaller model.
    #[must_use]
    pub fn draft_from(base: &SyntheticModel, agreement: f64, latency: Duration) -> Self {
        Self {
            vocab: base.vocab,
            seed: base.seed,
            latency,
            agreement: agreement.clamp(0.0, 1.0),
        }
    }

    /// The base (target) distribution for a context, independent of agreement.
    ///
    /// Deterministic given `(seed, context)`: we seed a PRNG and draw peaked
    /// positive weights, then normalize. This yields a non-uniform, stable
    /// categorical that varies with the context token.
    fn base_distribution(&self, context: usize) -> Vec<f64> {
        let ctx_seed = self
            .seed
            .wrapping_mul(0x100_0000_01B3)
            .wrapping_add(context as u64);
        let mut rng = Rng::new(ctx_seed);
        let mut weights: Vec<f64> = (0..self.vocab)
            .map(|_| {
                // Square to sharpen the distribution into a few dominant tokens,
                // which is realistic for language-model next-token predictions.
                let u = rng.next_f64();
                u * u + 1e-6
            })
            .collect();
        let total: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= total;
        }
        weights
    }
}

impl Model for SyntheticModel {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn latency(&self) -> Duration {
        self.latency
    }

    fn next_distribution(&self, context: usize) -> Vec<f64> {
        let base = self.base_distribution(context);
        if (self.agreement - 1.0).abs() < f64::EPSILON {
            return base;
        }
        // Blend toward uniform: q = agreement * base + (1 - agreement) * uniform.
        let uniform = 1.0 / self.vocab as f64;
        let a = self.agreement;
        let mut blended: Vec<f64> = base.iter().map(|&b| a * b + (1.0 - a) * uniform).collect();
        // Renormalize to defend against floating-point drift.
        let total: f64 = blended.iter().sum();
        for w in &mut blended {
            *w /= total;
        }
        blended
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_sums_to_one() {
        let m = SyntheticModel::target(16, 42, Duration::from_micros(100));
        let d = m.next_distribution(3);
        let sum: f64 = d.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum was {sum}");
        assert_eq!(d.len(), 16);
    }

    #[test]
    fn agreement_one_matches_target() {
        let t = SyntheticModel::target(16, 7, Duration::from_micros(100));
        let d = SyntheticModel::draft_from(&t, 1.0, Duration::from_micros(10));
        let td = t.next_distribution(2);
        let dd = d.next_distribution(2);
        for (a, b) in td.iter().zip(dd.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn agreement_zero_is_uniform() {
        let t = SyntheticModel::target(8, 7, Duration::from_micros(100));
        let d = SyntheticModel::draft_from(&t, 0.0, Duration::from_micros(10));
        let dd = d.next_distribution(5);
        let uniform = 1.0 / 8.0;
        for w in dd {
            assert!((w - uniform).abs() < 1e-12);
        }
    }
}
