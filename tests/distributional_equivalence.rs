//! Flagship integration tests for the speculative-decoding benchmark.
//!
//! These are the tests that justify the whole project:
//!
//! 1. **Distributional equivalence** — the full speculative accept/reject/
//!    residual loop with a deliberately *mismatched* draft `q` must produce the
//!    exact same output distribution as plain sampling from the target `p`. This
//!    is the core correctness property of speculative decoding.
//! 2. **Empirical-vs-analytical speedup** — the simulated speedup must track the
//!    closed-form prediction, cross-validating both the simulator and the
//!    formula.
//! 3. **Determinism** — identical seeds must yield identical results.

use speculative_decoding_bench::analytical::{expected_accepted_tokens, expected_speedup};
use speculative_decoding_bench::bench::{build_alpha_pair, run_sweep, BenchConfig};
use speculative_decoding_bench::rng::Rng;
use speculative_decoding_bench::sampling::{sample_from, speculative_step, StepOutcome};

/// Total variation distance between two empirical distributions.
fn total_variation(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f64>()
        * 0.5
}

/// Build a peaked target distribution `p` and a deliberately mismatched draft
/// `q` over the same vocabulary. `q` is intentionally a poor approximation of
/// `p` (different shape, reversed-ish mass) so the test genuinely exercises the
/// reject-and-resample path rather than trivially accepting everything.
fn mismatched_pair(vocab: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(0xF00D_1234);
    // Target p: peaked (squared uniforms), fully supported.
    let mut p: Vec<f64> = (0..vocab)
        .map(|_| {
            let u = rng.next_f64();
            u * u + 1e-3
        })
        .collect();
    let ptot: f64 = p.iter().sum();
    for x in &mut p {
        *x /= ptot;
    }

    // Draft q: a very different distribution. Reverse p and blend with a skew so
    // it is badly mismatched but still fully supported (no zero-probability
    // tokens, so every p-token is reachable through accept OR residual).
    let mut q: Vec<f64> = (0..vocab)
        .map(|i| {
            let rev = p[vocab - 1 - i];
            let skew = ((i as f64) + 1.0) / (vocab as f64);
            0.5 * rev + 0.5 * skew + 1e-3
        })
        .collect();
    let qtot: f64 = q.iter().sum();
    for x in &mut q {
        *x /= qtot;
    }
    (p, q)
}

#[test]
fn speculative_decoding_preserves_target_distribution() {
    let vocab = 12;
    let (p, q) = mismatched_pair(vocab);

    // Sanity: q really is mismatched from p.
    let mismatch = total_variation(&p, &q);
    assert!(
        mismatch > 0.15,
        "draft q is not mismatched enough (TV={mismatch:.3}); test would be trivial"
    );

    // Large trial count so the tolerance can be tight and the test non-flaky.
    let trials = 4_000_000u64;

    // (a) Plain sampling from p.
    let mut rng_plain = Rng::new(20260811);
    let mut counts_plain = vec![0u64; vocab];
    for _ in 0..trials {
        let t = sample_from(&p, &mut rng_plain).expect("plain sample");
        counts_plain[t] += 1;
    }

    // (b) Full speculative accept/reject/residual loop with mismatched q.
    let mut rng_spec = Rng::new(98765432);
    let mut counts_spec = vec![0u64; vocab];
    let mut resamples = 0u64;
    for _ in 0..trials {
        // Draft proposes a token from q, target verifies against p.
        let x = sample_from(&q, &mut rng_spec).expect("draft sample");
        let (tok, outcome) = speculative_step(&p, &q, x, &mut rng_spec).expect("spec step");
        if outcome == StepOutcome::Resampled {
            resamples += 1;
        }
        counts_spec[tok] += 1;
    }

    // The reject path must actually be exercised.
    let reject_rate = resamples as f64 / trials as f64;
    assert!(
        reject_rate > 0.1,
        "residual resample path barely used (rate={reject_rate:.3}); test is not meaningful"
    );

    let emp_plain: Vec<f64> = counts_plain
        .iter()
        .map(|&c| c as f64 / trials as f64)
        .collect();
    let emp_spec: Vec<f64> = counts_spec
        .iter()
        .map(|&c| c as f64 / trials as f64)
        .collect();

    // Statistical justification for the tolerance:
    //   Each per-token empirical frequency has standard error <= 0.5/sqrt(N).
    //   With N = 4e6, that is ~2.5e-4 per bin. Summed TV over 12 bins with a ~5σ
    //   band is comfortably under 6e-3. We also compare against the TWO
    //   independent-sampling baseline (plain-vs-plain) to anchor the tolerance
    //   empirically rather than by assertion alone.
    let tv_spec_vs_plain = total_variation(&emp_plain, &emp_spec);

    // Anchor: TV between two INDEPENDENT plain-sampling runs of the same size.
    let mut rng_plain2 = Rng::new(1122334455);
    let mut counts_plain2 = vec![0u64; vocab];
    for _ in 0..trials {
        let t = sample_from(&p, &mut rng_plain2).expect("plain sample 2");
        counts_plain2[t] += 1;
    }
    let emp_plain2: Vec<f64> = counts_plain2
        .iter()
        .map(|&c| c as f64 / trials as f64)
        .collect();
    let tv_plain_vs_plain = total_variation(&emp_plain, &emp_plain2);

    println!("TV(spec, plain)      = {tv_spec_vs_plain:.6}");
    println!("TV(plain, plain2)    = {tv_plain_vs_plain:.6}  (sampling-noise floor)");
    println!("reject/resample rate = {reject_rate:.4}");

    // The speculative distribution must be as close to plain-p as an independent
    // plain-p run is, up to a small slack, AND within an absolute tight bound.
    let tolerance = 6e-3;
    assert!(
        tv_spec_vs_plain < tolerance,
        "speculative decoding changed the output distribution: TV={tv_spec_vs_plain:.6} >= {tolerance}"
    );
    assert!(
        tv_spec_vs_plain < tv_plain_vs_plain + 3e-3,
        "spec TV ({tv_spec_vs_plain:.6}) far exceeds pure sampling-noise floor ({tv_plain_vs_plain:.6})"
    );
}

#[test]
fn empirical_speedup_tracks_analytical() {
    let cfg = BenchConfig {
        vocab: 64,
        k_range: (1, 8),
        alphas: vec![0.3, 0.5, 0.7, 0.9],
        cost_ratio: 0.2,
        rounds: 300_000,
        seed: 0x5EED_1234,
    };
    let res = run_sweep(&cfg).expect("sweep");

    for r in &res.rows {
        // Measured acceptance rate must match the constructed target alpha.
        assert!(
            (r.alpha_measured - r.alpha_target).abs() < 5e-3,
            "k={} alpha_target={} measured={} drifted",
            r.k,
            r.alpha_target,
            r.alpha_measured
        );
        // Empirical tokens/round must match the closed form.
        let ana_tok = expected_accepted_tokens(r.alpha_target, r.k);
        let tok_rel = (r.empirical_tokens - ana_tok).abs() / ana_tok;
        assert!(
            tok_rel < 0.01,
            "k={} alpha={}: empirical tokens {} vs analytical {} (rel {:.4})",
            r.k,
            r.alpha_target,
            r.empirical_tokens,
            ana_tok,
            tok_rel
        );
        // And hence empirical speedup must track analytical speedup.
        let ana_spd = expected_speedup(r.alpha_target, r.k, cfg.cost_ratio);
        let spd_rel = (r.empirical_speedup - ana_spd).abs() / ana_spd;
        assert!(
            spd_rel < 0.01,
            "k={} alpha={}: empirical speedup {} vs analytical {} (rel {:.4})",
            r.k,
            r.alpha_target,
            r.empirical_speedup,
            ana_spd,
            spd_rel
        );
    }
}

#[test]
fn acceptance_rate_high_when_q_equals_p() {
    // With q == p, acceptance must be ~1.0.
    let vocab = 32;
    let (p, _q) = build_alpha_pair(vocab, 0.9, 7);
    let q = p.clone();
    let mut rng = Rng::new(555);
    let trials = 200_000u64;
    let mut accepted = 0u64;
    for _ in 0..trials {
        let x = sample_from(&q, &mut rng).expect("sample");
        let (_t, o) = speculative_step(&p, &q, x, &mut rng).expect("step");
        if o == StepOutcome::Accepted {
            accepted += 1;
        }
    }
    let rate = accepted as f64 / trials as f64;
    assert!(
        rate > 0.999,
        "q==p acceptance rate should be ~1.0, got {rate}"
    );
}

#[test]
fn determinism_same_seed_same_result() {
    let cfg = BenchConfig {
        rounds: 5_000,
        ..BenchConfig::default()
    };
    let a = run_sweep(&cfg).expect("sweep a");
    let b = run_sweep(&cfg).expect("sweep b");
    assert_eq!(a.rows.len(), b.rows.len());
    for (ra, rb) in a.rows.iter().zip(b.rows.iter()) {
        assert_eq!(ra.k, rb.k);
        assert_eq!(ra.alpha_target, rb.alpha_target);
        assert_eq!(ra.alpha_measured, rb.alpha_measured);
        assert_eq!(ra.empirical_tokens, rb.empirical_tokens);
        assert_eq!(ra.empirical_speedup, rb.empirical_speedup);
    }

    // Raw PRNG stream determinism too.
    let mut r1 = Rng::new(42);
    let mut r2 = Rng::new(42);
    for _ in 0..1000 {
        assert_eq!(r1.next_f64(), r2.next_f64());
    }
}
