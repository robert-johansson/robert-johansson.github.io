---
layout: post
title: "GenMLX: From Prototype to GenJAX Parity in a Day"
date: 2026-02-19 11:00:00 +0100
categories: [probabilistic-programming]
tags: [clojurescript, mlx, gen, apple-silicon, bayesian]
share: false
---

[Yesterday]({% post_url 2026-02-18-genmlx %}) I introduced [GenMLX](https://github.com/robert-johansson/genmlx) — ~2,000 lines of ClojureScript implementing Gen's Generative Function Interface on Apple Silicon. Since then, quite a lot happened. The system is now ~5,700 lines and has reached inference parity with [GenJAX](https://github.com/probcomp/GenJAX).

## What changed

The numbers tell the story:

| | Yesterday | Today |
|---|---|---|
| Lines of code | ~2,000 | ~5,700 |
| Distributions | 13 | 22 |
| Combinators | 3 | 8 |
| Inference algorithms | 7 | 15+ |
| Test assertions | 165 | 337 |

## New inference algorithms

The original had MH, MALA, HMC, NUTS, importance sampling, SMC, and VI. The new additions bring GenMLX to full GenJAX parity and beyond:

- **Custom-proposal MH** — user-written proposal generative functions instead of prior proposals
- **Enumerative Gibbs** — exact conditional sampling for discrete variables with finite support
- **Involutive MCMC** — the general trans-dimensional MCMC framework from Gen.jl, now with optional Jacobian support
- **Conditional SMC** — SMC with a retained reference particle for particle MCMC
- **SMCP3** — Sequential Monte Carlo with Probabilistic Program Proposals, the most powerful inference algorithm in GenJAX
- **Programmable VI** — four variational objectives (ELBO, IWELBO, PWake, QWake) with two gradient estimators
- **Wake-sleep learning** — amortized inference via alternating wake and sleep phases
- **Composable inference kernels** — chain, repeat, cycle, mix, seed

```clojure
;; Compose inference kernels algebraically
(chain k1 k2 k3)                       ;; sequential
(repeat-kernel 10 k)                   ;; iterate
(cycle-kernels 30 [k1 k2])             ;; round-robin
(mix-kernels [[k1 0.7] [k2 0.3]])      ;; random mixture
(run-kernel {:samples 1000 :burn 200 :thin 2} k init-trace)
```

GenMLX now has NUTS, MALA, and standard ADVI, which GenJAX does not ship built-in.

## Vectorized execution: 29–122x speedup

The biggest performance win. Instead of running the model N times for N particles, GenMLX runs it once with `[N]`-shaped arrays. Because all score computation uses element-wise MLX operations, everything broadcasts automatically — no code transformation needed.

| Operation | Sequential | Batched | Speedup |
|---|---|---|---|
| `generate` (5 sites, N=100) | 122 ms | 2 ms | **61x** |
| Importance sampling (N=100) | 81 ms | 1 ms | **81x** |
| SMC init (N=100) | 65 ms | 1 ms | **65x** |

The key insight: GenMLX's handler state threading is shape-agnostic by construction. A score that starts as a scalar and accumulates scalar log-probabilities works identically when it starts as a scalar and accumulates `[N]`-shaped log-probabilities via broadcasting. So vectorized execution required no changes to the core handler — just sampling `[N]`-shaped arrays instead of scalars at each trace site.

## Five new combinators

Map, Unfold, and Switch were there from the start. New additions:

- **Scan** — state-threading sequential combinator (equivalent to GenJAX's `scan` / `jax.lax.scan`). More general than Unfold for models that need per-step outputs.
- **Mask** — gates execution on a boolean condition. Used internally by vectorized switch for all-branch execution with mask-selection.
- **Mix** — first-class mixture model combinator with mixing weights.
- **Contramap** — transforms arguments before passing to an inner generative function.
- **Dimap** — transforms both arguments and return values.

All combinators with update/regenerate support implement the full GFI, so they compose with all inference algorithms.

## Parametric edit interface

Generalizes `update` and `regenerate` into a single operation with typed EditRequests:

```clojure
(edit/edit gf trace (edit/constraint-edit new-constraints))
(edit/edit gf trace (edit/selection-edit selection))
(edit/edit gf trace (edit/proposal-edit fwd-proposal bwd-proposal))
```

Each edit type automatically generates its backward request for reversible kernels. This is what makes SMCP3 work — it uses the polymorphic edit interface so custom generative function types automatically participate in SMCP3 inference.

## Diff-aware incremental updates

MapCombinator now stores per-element scores as trace metadata. When called with a `vector-diff`, only changed elements are updated:

```clojure
(p/update-with-diffs model trace constraints (diff/vector-diff #{0}))
;; Only element 0 is re-executed — O(1) instead of O(n)
```

## Trainable parameters

`dyn/param` declares trainable parameters inside `gen` bodies:

```clojure
(def model
  (gen [obs-val]
    (let [mu (dyn/param :mu 0.0)]
      (dyn/trace :obs (dist/gaussian mu 1))
      mu)))
```

Gradients flow through MLX arrays, enabling standard optimization over model parameters.

## More distributions

From 13 to 22. New: truncated-normal, inverse-gamma, student-t, cauchy, geometric, negative-binomial, binomial, discrete-uniform, and mixture. A `defdist-transform` macro enables derived distributions via deterministic transforms (e.g., log-normal as exp of gaussian).

## Verification

337 test assertions across 16 test files:
- 165 Gen.jl compatibility tests
- 73 GenJAX compatibility tests
- 50 audit resolution tests (custom MH, involutive MCMC with Jacobians, combinator updates, VI, wake-sleep)
- 36 tests for edit interface, diff-aware updates, and trainable parameters
- 13 bug fix tests (update weights, vectorized switch, conditional SMC)

## The comparison

| System | Language | LOC | Distributions | Combinators | Inference | Hardware |
|---|---|---|---|---|---|---|
| Gen.jl | Julia | ~20,000 | 12 | 3 | 4 | CPU |
| GenJAX | Python/JAX | ~10,000 | 8 | 5 | 6+ | GPU/TPU (NVIDIA, Google) |
| GenMLX | ClojureScript | ~5,700 | 22 | 8 | 15+ | GPU (Apple Silicon) |

The size difference reflects design trade-offs — Gen.jl has a static DSL compiler, GenJAX has JAX compilation infrastructure. GenMLX gets its compactness from ClojureScript's persistent data structures, open multimethods, and MLX's lazy evaluation, which together replace significant infrastructure code.

The code is at [github.com/robert-johansson/genmlx](https://github.com/robert-johansson/genmlx).
