//! Closed-form analytical model of speculative-decoding performance.
//!
//! Following Leviathan et al., for a per-token acceptance probability `α`
//! (assumed i.i.d. across the `k` drafted positions) the **expected number of
//! tokens accepted per verification round** is:
//!
//! ```text
//! E[tokens] = (1 - α^(k+1)) / (1 - α)      for α < 1
//! ```
//!
//! Intuitively each drafted token is accepted with probability `α`; the round
//! stops at the first rejection, after which the target's residual sample still
//! yields one guaranteed token. The `+1` in the exponent accounts for that
//! bonus token produced by the target on every round.
//!
//! At `α = 1` the closed form is `0/0`; the limit is `k + 1` (all `k` drafts
//! accepted plus the bonus token). We handle that boundary explicitly.
//!
//! ## Expected speedup
//!
//! Let `c_target` be the cost of one target forward pass and `c_draft` the cost
//! of one draft forward pass. A speculative round runs the draft `k` times and
//! the target once (the batched verification), for a cost of
//! `k * c_draft + c_target`, and produces `E[tokens]` tokens. Autoregressive
//! decoding produces one token per `c_target`. Hence:
//!
//! ```text
//! speedup = E[tokens] / ((k * c_draft + c_target) / c_target)
//!         = E[tokens] * c_target / (k * c_draft + c_target)
//! ```

/// Expected number of tokens accepted (emitted) per verification round.
///
/// Implements `(1 - α^(k+1)) / (1 - α)` with the `α = 1` limit `k + 1` handled
/// explicitly. `alpha` is clamped to `[0, 1]`; `k` is the draft length (number
/// of tokens proposed per round) and should be `>= 1`.
///
/// For `k = 0` the expression correctly evaluates to `1.0` (only the target's
/// bonus token, i.e. plain autoregressive decoding).
#[must_use]
pub fn expected_accepted_tokens(alpha: f64, k: u32) -> f64 {
    let alpha = alpha.clamp(0.0, 1.0);
    // α = 1 limit: every draft accepted plus the bonus token => k + 1.
    if (1.0 - alpha).abs() < 1e-12 {
        return f64::from(k) + 1.0;
    }
    let num = 1.0 - alpha.powi(k as i32 + 1);
    num / (1.0 - alpha)
}

/// Expected speedup of speculative decoding over autoregressive decoding.
///
/// - `alpha`: per-token acceptance probability in `[0, 1]`.
/// - `k`: draft length (tokens proposed per round).
/// - `cost_ratio`: `c_draft / c_target`, the per-call cost of the draft model
///   relative to the target model (e.g. `0.1` when the draft is 10× cheaper).
///
/// Returns `E[tokens] / (k * cost_ratio + 1)`. `cost_ratio` is clamped to be
/// non-negative.
#[must_use]
pub fn expected_speedup(alpha: f64, k: u32, cost_ratio: f64) -> f64 {
    let cost_ratio = cost_ratio.max(0.0);
    let tokens = expected_accepted_tokens(alpha, k);
    let round_cost = f64::from(k) * cost_ratio + 1.0;
    tokens / round_cost
}

/// The draft length `k` in `1..=k_max` that maximizes [`expected_speedup`] for a
/// given `alpha` and `cost_ratio`, together with that speedup.
///
/// Returns `None` if `k_max == 0`.
#[must_use]
pub fn optimal_k(alpha: f64, cost_ratio: f64, k_max: u32) -> Option<(u32, f64)> {
    if k_max == 0 {
        return None;
    }
    let mut best: Option<(u32, f64)> = None;
    for k in 1..=k_max {
        let s = expected_speedup(alpha, k, cost_ratio);
        match best {
            Some((_, bs)) if bs >= s => {}
            _ => best = Some((k, s)),
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn alpha_one_limit_is_k_plus_one() {
        for k in 0..=8 {
            let t = expected_accepted_tokens(1.0, k);
            assert!(approx(t, f64::from(k) + 1.0, 1e-12), "k={k} got {t}");
        }
    }

    #[test]
    fn k_one_boundary() {
        // (1 - α^2) / (1 - α) = 1 + α.
        for &alpha in &[0.0, 0.3, 0.5, 0.7, 0.9] {
            let t = expected_accepted_tokens(alpha, 1);
            assert!(approx(t, 1.0 + alpha, 1e-12), "alpha={alpha} got {t}");
        }
    }

    #[test]
    fn alpha_zero_gives_one_token() {
        for k in 0..=8 {
            let t = expected_accepted_tokens(0.0, k);
            assert!(approx(t, 1.0, 1e-12), "k={k} got {t}");
        }
    }

    #[test]
    fn expected_tokens_monotonic_in_alpha() {
        let k = 4;
        let mut prev = expected_accepted_tokens(0.0, k);
        for i in 1..=10 {
            let alpha = f64::from(i) / 10.0;
            let t = expected_accepted_tokens(alpha, k);
            assert!(t >= prev - 1e-12, "not monotonic at alpha={alpha}");
            prev = t;
        }
    }

    #[test]
    fn speedup_matches_manual_formula() {
        // alpha=0.5, k=2, cost_ratio=0.1.
        // E[tokens] = (1 - 0.5^3)/(1-0.5) = (1-0.125)/0.5 = 1.75.
        // round_cost = 2*0.1 + 1 = 1.2. speedup = 1.75/1.2.
        let s = expected_speedup(0.5, 2, 0.1);
        assert!(approx(s, 1.75 / 1.2, 1e-12), "got {s}");
    }

    #[test]
    fn optimal_k_returns_some() {
        let (k, s) = optimal_k(0.9, 0.1, 8).expect("optimal");
        assert!((1..=8).contains(&k));
        assert!(s > 1.0);
    }
}
