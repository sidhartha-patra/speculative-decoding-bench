# speculative-decoding-bench

A simulation harness and benchmark suite for **speculative decoding** — the LLM
inference-acceleration technique in which a small, cheap *draft* model proposes
`k` tokens ahead and a large *target* model verifies them in a single batched
forward pass, accepting the longest correct prefix.

This project models the algorithm's **correctness** and **performance**
characteristics without GPUs or real model weights. It lets an engineer reason
about the speedup / acceptance-rate / draft-length trade-off both **analytically**
(closed-form) and **empirically** (Monte-Carlo simulation), and — most
importantly — it verifies the property that makes speculative decoding usable at
all: it does not change the output distribution.

Everything is **std-only** (zero external crates), uses a **deterministic PRNG**
so every number here is reproducible, and hand-rolls timing/cost accounting.

---

## 1. What speculative decoding is, and why it matters

Autoregressive LLM decoding is fundamentally sequential: to emit `n` tokens you
run the large model `n` times, and each forward pass is memory-bandwidth bound —
you pay to stream billions of weights through the accelerator to produce a
*single* token. This sequential dependency is the dominant source of
per-request latency in production LLM serving.

**Speculative decoding** breaks the sequential bottleneck by trading cheap
compute for expensive latency:

1. A small **draft** model (cheap, fast) autoregressively proposes `k` candidate
   tokens.
2. The large **target** model scores all `k` proposals **in a single batched
   forward pass** (the proposals are known, so they can be verified in parallel
   rather than generated one at a time).
3. A verification rule accepts the longest correct prefix and emits one
   additional "bonus" token, then the loop repeats.

Because one expensive target pass now yields *multiple* tokens whenever the draft
guesses well, wall-clock latency drops. It is the dominant latency-optimization
technique in production LLM serving stacks — vLLM, TensorRT-LLM, and llama.cpp
all implement a form of it.

The trade-off is not free: every drafted token costs draft compute whether or not
it is accepted, so the *net* speedup depends on how often the draft agrees with
the target (the **acceptance rate** `α`) and how many tokens you draft per round
(`k`). That trade-off is exactly what this benchmark quantifies.

---

## 2. The exact acceptance rule (the math)

This crate implements the verification rule from Leviathan et al., *"Fast
Inference from Transformers via Speculative Decoding"* (ICML 2023), and Chen et
al., *"Accelerating Large Language Model Decoding with Speculative Sampling"*
(2023). For a single position:

- The draft produces a distribution `q(·)` and samples a token `x ~ q`.
- The target produces a distribution `p(·)`.
- **Accept** `x` with probability

  ```
  p_accept(x) = min(1, p(x) / q(x))
  ```

- **On rejection**, resample the token from the **residual distribution**

  ```
  p_residual(x) = norm(max(0, p(x) - q(x)))
  ```

  i.e. take the elementwise positive part of `p - q` and renormalize it to sum
  to 1.

### Why this is correct — distribution preservation

The remarkable theorem behind speculative decoding is that this accept/reject/
residual construction produces tokens distributed **exactly** according to the
target distribution `p`, *for any draft `q`*. Sketch: the probability of emitting
a given token `x` is

```
Pr[emit x] = q(x)·min(1, p(x)/q(x))              (drafted x and accepted)
           + (1 - β)·p_residual(x)               (some draft rejected, resampled)
         = min(p(x), q(x)) + (p(x) - min(p(x), q(x)))
         = p(x)
```

where `β = Σ_x min(p(x), q(x))` is the total acceptance mass. The accepted mass
contributes `min(p, q)` and the residual contributes exactly the remaining
`p - min(p, q)`, summing to `p`. A worse draft simply causes more rejections and
more residual resampling — it never biases the output.

### The degenerate residual case

If `p(x) ≤ q(x)` for every `x` (so `max(0, p - q)` is all zeros — this happens
when `q == p`, or when `q` dominates `p` everywhere), there is no positive
residual mass to normalize. The rejection branch is entered with probability zero
in that situation, so any proper distribution is valid there; we fall back to the
normalized target `norm(p)` to keep the output well-defined. This edge case is
handled explicitly in [`src/sampling.rs`](src/sampling.rs) and unit-tested.

---

## 3. Why distribution preservation is THE core correctness property

**A speculative decoder that is faster but changes the output distribution is
simply a broken decoder.** The entire value proposition is "same outputs as the
target model, produced faster." If the distribution drifts, you have silently
swapped in a *different, worse* model — any downstream evaluation, safety tuning,
or quality guarantee attached to the target model is void. Speed is worthless
without this invariant.

So the flagship test in this repository
([`tests/distributional_equivalence.rs`](tests/distributional_equivalence.rs))
is a direct, empirical check of exactly that invariant. Over **4,000,000**
fixed-seed trials it compares:

- **(a)** plain sampling directly from the target `p`, versus
- **(b)** the full speculative accept/reject/residual loop driven by a
  **deliberately mismatched** draft `q` (a reversed, skewed distribution with
  total-variation distance `> 0.15` from `p`, chosen so the reject/resample path
  is heavily exercised).

It then measures the total-variation distance between the two empirical output
distributions. Real captured result:

```
TV(spec, plain)      = 0.001176
TV(plain, plain2)    = 0.000593   (sampling-noise floor: two independent plain runs)
reject/resample rate = 0.3214
```

The speculative decoder — despite a badly wrong draft and a **32%** rejection
rate — produces an output distribution within `~2×` the pure Monte-Carlo noise
floor of plain target sampling, and comfortably under an absolute `6e-3`
tolerance. The tolerance is statistically justified: each per-token frequency has
standard error `≤ 0.5/√N ≈ 2.5e-4` at `N = 4e6`, so a summed-TV band of a few
`e-3` is a multi-sigma bound. Because the PRNG is fully deterministic, the test
is reproducible and non-flaky by construction.

That is the whole point of the project made executable: **the fast path and the
reference path are the same distribution.**

---

## 4. Architecture

| Module | Responsibility |
| --- | --- |
| [`src/rng.rs`](src/rng.rs) | Deterministic, std-only PRNG: `SplitMix64`-seeded `xoshiro256**`, plus uniform `f64` and categorical sampling. Reproducibility backbone. |
| [`src/sampling.rs`](src/sampling.rs) | The core algorithm: `min(1, p/q)` acceptance, the residual distribution `norm(max(0, p - q))`, the all-zero-residual fallback, and a single-token `speculative_step`. |
| [`src/models.rs`](src/models.rs) | `Model` trait + `SyntheticModel` draft/target simulators, with a tunable **agreement** knob (blend target → uniform) and a simulated per-call **latency**. |
| [`src/analytical.rs`](src/analytical.rs) | Closed-form expected accepted tokens and expected speedup, including the `α = 1` limit and `optimal_k`. |
| [`src/bench.rs`](src/bench.rs) | The sweep harness: exact-`α` distribution construction, round simulation, `k × α` sweep, CSV export, and per-`α` optimal-`k`. |
| [`src/main.rs`](src/main.rs) | CLI (`specbench`) that runs the default sweep and prints the results table. |

Public library API is documented with `///` doc comments; fallible paths return
`Result` with a typed `SamplingError` (no `unwrap()` on fallible paths in the
library).

---

## 5. The analytical model

Assuming the per-token acceptance probability `α` is i.i.d. across the `k`
drafted positions, the **expected number of tokens emitted per verification
round** is:

```
E[tokens] = (1 - α^(k+1)) / (1 - α)          for α < 1
```

Each drafted token is accepted with probability `α`; the round stops at the first
rejection, after which the target's residual sample still contributes one
guaranteed token — the `+1` in the exponent accounts for that bonus token
produced on every round.

At `α = 1` the closed form is `0/0`; the limit is `k + 1` (all `k` drafts
accepted plus the bonus token). This boundary is handled explicitly and
unit-tested, as is the `k = 1` boundary (`E[tokens] = 1 + α`).

**Expected speedup.** With `c_target` the cost of one target pass and `c_draft`
the cost of one draft pass, a round costs `k·c_draft + c_target` and yields
`E[tokens]` tokens, while autoregressive decoding yields one token per
`c_target`. Hence, with cost ratio `r = c_draft / c_target`:

```
speedup = E[tokens] / (k·r + 1)
```

This captures the real tension: larger `k` raises `E[tokens]` but also raises the
`k·r` wasted-draft penalty, so the optimal `k` is finite and depends on `α`.

---

## 6. Benchmark results (real, captured output)

Command: `cargo run --release` (the `specbench` CLI). Configuration:
`vocab = 64`, `cost_ratio r = 0.2` (draft is 5× cheaper than the target),
`200,000` simulated verification rounds per `(k, α)` cell, fixed seed.

Columns: `alpha` = target acceptance rate; `alpha_hat` = *measured* acceptance
rate from the simulation; `emp_tok` / `ana_tok` = empirical vs analytical tokens
per round; `emp_spd` / `ana_spd` = empirical vs analytical speedup.

```
  k    alpha  alpha_hat     emp_tok     ana_tok     emp_spd     ana_spd
------------------------------------------------------------------------
  1     0.30     0.3007      1.3007      1.3000      1.084x      1.083x
  2     0.30     0.3000      1.3902      1.3900      0.993x      0.993x
  3     0.30     0.3007      1.4184      1.4170      0.887x      0.886x
  4     0.30     0.3000      1.4251      1.4251      0.792x      0.792x
  5     0.30     0.3018      1.4313      1.4275      0.716x      0.714x
  6     0.30     0.3009      1.4301      1.4283      0.650x      0.649x
  7     0.30     0.3017      1.4319      1.4285      0.597x      0.595x
  8     0.30     0.2995      1.4276      1.4285      0.549x      0.549x
  1     0.50     0.4982      1.4982      1.5000      1.248x      1.250x
  2     0.50     0.5000      1.7498      1.7500      1.250x      1.250x
  3     0.50     0.4990      1.8732      1.8750      1.171x      1.172x
  4     0.50     0.4991      1.9347      1.9375      1.075x      1.076x
  5     0.50     0.5012      1.9732      1.9688      0.987x      0.984x
  6     0.50     0.4995      1.9824      1.9844      0.901x      0.902x
  7     0.50     0.5004      1.9937      1.9922      0.831x      0.830x
  8     0.50     0.5010      2.0002      1.9961      0.769x      0.768x
  1     0.70     0.6973      1.6973      1.7000      1.414x      1.417x
  2     0.70     0.6995      2.1894      2.1900      1.564x      1.564x
  3     0.70     0.7010      2.5367      2.5330      1.585x      1.583x
  4     0.70     0.6990      2.7684      2.7731      1.538x      1.541x
  5     0.70     0.6997      2.9374      2.9412      1.469x      1.471x
  6     0.70     0.6996      3.0571      3.0588      1.390x      1.390x
  7     0.70     0.6996      3.1374      3.1412      1.307x      1.309x
  8     0.70     0.7007      3.2059      3.1988      1.233x      1.230x
  1     0.90     0.8988      1.8987      1.9000      1.582x      1.583x
  2     0.90     0.9005      2.7110      2.7100      1.936x      1.936x
  3     0.90     0.9005      3.4420      3.4390      2.151x      2.149x
  4     0.90     0.9002      4.0954      4.0951      2.275x      2.275x
  5     0.90     0.9001      4.6888      4.6856      2.344x      2.343x
  6     0.90     0.9003      5.2208      5.2170      2.373x      2.371x
  7     0.90     0.8999      5.6896      5.6953      2.371x      2.373x
  8     0.90     0.8998      6.1241      6.1258      2.355x      2.356x

Optimal k per acceptance rate (cost_ratio = 0.200):
  alpha = 0.30  ->  k* = 1  (analytical speedup 1.083x)
  alpha = 0.50  ->  k* = 1  (analytical speedup 1.250x)
  alpha = 0.70  ->  k* = 3  (analytical speedup 1.583x)
  alpha = 0.90  ->  k* = 7  (analytical speedup 2.373x)
```

### What the numbers show

- **Empirical tracks analytical within < 1%.** Across all 32 cells the measured
  tokens-per-round and speedup match the closed-form prediction to under 1%
  relative error (e.g. `α = 0.9, k = 6`: `5.2208` vs `5.2170` tokens; `2.373x`
  vs `2.371x`). This is a genuine two-way cross-check: it validates *both* that
  the simulator implements the algorithm correctly *and* that the closed-form
  formula is right. `alpha_hat` also confirms the exact-`α` construction hits its
  target acceptance rate.
- **The optimal `k` is non-monotonic in `α` and strongly `α`-dependent.** At low
  acceptance (`α = 0.3`) the wasted draft compute dominates and the best choice
  is `k* = 1` — larger `k` is actively *slower than 1×* (down to `0.549x` at
  `k = 8`). As acceptance rises, deeper drafting pays off: `k* = 3` at `α = 0.7`,
  `k* = 7` at `α = 0.9`. Larger `k` wins **only while acceptance stays high
  enough to pay for the speculative compute** — precisely the real production
  trade-off practitioners tune.
- **Speedup < 1× is possible.** When `α` is low and `k` is large, speculative
  decoding is a net *loss*. The benchmark makes that failure mode explicit rather
  than assuming speculation always helps.

The sweep is also exported as CSV via `SweepResult::to_csv()`.

---

## 7. Building and running

The toolchain is standard stable Rust (edition 2021), no external dependencies.

```sh
cargo build --release
cargo test --release          # 17 unit + 4 integration = 21 tests
cargo run --release           # runs the default sweep and prints the table
cargo clippy -- -D warnings
cargo fmt --check
```

### Tests

- **`speculative_decoding_preserves_target_distribution`** — the flagship
  distributional-equivalence test (§3).
- **`empirical_speedup_tracks_analytical`** — asserts every sweep cell matches
  the closed form within 1% (tokens, acceptance, and speedup).
- **`acceptance_rate_high_when_q_equals_p`** — with `q == p`, acceptance ≈ 1.0.
- **`determinism_same_seed_same_result`** — identical seeds ⇒ identical results,
  down to the raw PRNG stream.
- Unit tests for the residual math (sums to 1, all-zero fallback, length/empty
  errors), the analytical `α = 1` / `k = 1` / `α = 0` boundaries and
  monotonicity, the agreement knob (`agreement = 1` ⇒ matches target,
  `agreement = 0` ⇒ uniform), and the exact-`α` construction's overlap.

---

## 8. Limitations & Scope

This project is a **simulation of the speculative-decoding *algorithm***, not a
transformer inference engine. Read the results as statements about the
algorithm's statistical and cost behavior, **not** about any particular hardware
or model. Specifically:

- **Synthetic distributions.** `p` and `q` are synthetic categorical
  distributions over a toy vocabulary, not real next-token logits from a language
  model. Real acceptance rates depend on how well a real draft model mimics a
  real target on real text, which this does not attempt to predict.
- **Simulated, dimensionless latency / cost.** "Cost" is a single scalar ratio
  `c_draft / c_target`. This deliberately ignores GPU kernel behavior, memory
  bandwidth, KV-cache read/write costs, attention cost growth with sequence
  length, quantization effects, and kernel-launch overhead — all of which matter
  on real accelerators.
- **No batching / serving dynamics.** Continuous batching, in-flight batching,
  paged attention, scheduler contention, and multi-request interference are out
  of scope. The cost model assumes verification of `k` proposals is exactly one
  target-pass equivalent, which is a first-order approximation of the real
  batched forward pass.
- **I.i.d. acceptance assumption.** The closed-form model (and the exact-`α`
  sweep construction) treat per-position acceptance as i.i.d. with a fixed `α`.
  Real acceptance is correlated across positions and context-dependent; tree- and
  Medusa-style multi-branch speculation are not modeled.
- **Greedy/prefix verification only.** A single linear draft chain is verified;
  token-tree verification and typical-acceptance relaxations are not implemented.

Within that scope, the value is precise: it implements the *provably
distribution-preserving* verification rule correctly, demonstrates the invariant
empirically, and shows the acceptance-rate × draft-length × cost trade-off with a
closed-form model that its own Monte-Carlo simulation confirms to within 1%.

---

## 9. References

- Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. arXiv:2211.17192.
- C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, J. Jumper.
  *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
  arXiv:2302.01318.
- D. Blackman, S. Vigna. *Scrambled Linear Pseudorandom Number Generators.*
  (xoshiro / SplitMix, <https://prng.di.unimi.it/>.)

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Sidhartha Patra.
