//! Deterministic, std-only pseudo-random number generator.
//!
//! We deliberately avoid external crates so the benchmark is reproducible and
//! self-contained. The generator is a **SplitMix64 → xoshiro256\*\*** pipeline:
//! `SplitMix64` is used to seed the state (the canonical way to initialize
//! xoshiro from a single 64-bit seed), and `xoshiro256**` provides the actual
//! stream. Both are well-studied public-domain algorithms by Sebastiano Vigna
//! and David Blackman (<https://prng.di.unimi.it/>).
//!
//! Given the same seed, every run produces an identical stream of values, which
//! is what makes the statistical tests in this crate non-flaky and the
//! benchmark results exactly reproducible.

/// A deterministic PRNG producing a reproducible stream from a 64-bit seed.
///
/// Implements `xoshiro256**` seeded via `SplitMix64`.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Create a new generator from a single 64-bit seed.
    ///
    /// The seed is expanded into the 256-bit internal state with `SplitMix64`,
    /// so even a seed of `0` yields a well-distributed state.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_sm = || {
            // SplitMix64
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let state = [next_sm(), next_sm(), next_sm(), next_sm()];
        Self { state }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // xoshiro256**
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Return a uniformly distributed `f64` in the half-open interval `[0, 1)`.
    ///
    /// Uses the top 53 bits of a 64-bit draw, matching the mantissa width of an
    /// IEEE-754 double so every representable value in `[0,1)` is reachable and
    /// the distribution is uniform.
    #[inline]
    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa: divide a 53-bit integer by 2^53.
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / 9_007_199_254_740_992.0) // 2^53
    }

    /// Sample an index from a categorical distribution given by `weights`.
    ///
    /// `weights` need not be normalized; only the relative magnitudes matter.
    /// Returns `None` if `weights` is empty or its total mass is not strictly
    /// positive (e.g. all zeros or non-finite), so callers must handle the
    /// degenerate case explicitly rather than silently biasing the result.
    #[must_use]
    pub fn sample_categorical(&mut self, weights: &[f64]) -> Option<usize> {
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 || weights.is_empty() {
            return None;
        }
        let mut threshold = self.next_f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            // Guard against negative weights contributing to the walk.
            if w > 0.0 {
                threshold -= w;
                if threshold < 0.0 {
                    return Some(i);
                }
            }
        }
        // Floating-point drift can leave threshold >= 0 at the end; return the
        // last positive-weight index as the mathematically-correct fallback.
        weights.iter().rposition(|&w| w > 0.0)
    }
}
