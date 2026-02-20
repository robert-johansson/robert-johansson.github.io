---
layout: post
title: "GenMLX: Vectorizing the Full Inference Stack"
date: 2026-02-20 12:15:00 +0100
categories: [probabilistic-programming]
tags: [clojurescript, mlx, gen, apple-silicon, bayesian]
share: false
---

[This morning's post]({% post_url 2026-02-20-genmlx-10k %}) was a snapshot at 10,800 lines. Since then, another 23 commits went in --- vectorizing the full inference stack, adding gradient estimators, neural network integration, and compiled Metal inference loops. The theme: make GenMLX fast for real workloads.

## Vectorized splice

The [previous vectorization work]({% post_url 2026-02-19-genmlx-progress %}) had a limitation: `splice` (sub-generative-function calls) didn't work in batched mode. Now it does. DynamicGF sub-GFs run under batched handlers, with nested splice (3+ levels) supported. This means hierarchical models vectorize end-to-end.

## Vectorized MCMC and SMC

N independent MH chains now run in parallel via MLX broadcasting --- enabling parallel tempering and R-hat convergence diagnostics from a single call.

For SMC, the model body runs *once* per timestep for all N particles. Multi-step batched particle filtering gives 24.5x speedup over sequential SMC (50 particles, 5 steps).

## Compiled Metal inference loops

The entire MH accept/reject chain compiles into a single Metal program via `mx/compile-fn`. Same for the VI optimization loop. No per-step JavaScript-to-Metal dispatch overhead --- the full iteration is one cached GPU dispatch.

A benchmark suite comparing GenMLX vs handcoded MLX for importance sampling and HMC at various scales confirms the abstraction overhead is minimal.

## Near-complete batch sampling

Native `dist-sample-n` now covers nearly every distribution. Added discrete-uniform, geometric, categorical, multivariate-normal, binomial, and student-t. Then a vectorized Marsaglia-Tsang implementation for gamma (with Ahrens-Dieter for alpha < 1) unlocked batch sampling for beta, inverse-gamma, and dirichlet. Only poisson remains sequential.

## VIMCO and ADEV gradient estimators

**VIMCO** --- multi-sample variational objective with leave-one-out baseline for tighter gradient estimates than single-sample ELBO.

**ADEV** --- sound automatic differentiation of expected values (Lew et al., POPL 2023). Integrates reparameterization and score-function estimators with `mx/grad`, with a dedicated ADEV handler for tracing through stochastic computation graphs.

## Prefix skipping for sequential models

Unfold and Scan now store per-step scores (and carries for Scan) as trace metadata. During `update`, the system finds the earliest constrained step and skips the unchanged prefix. This turns O(T^2) total work for SMC on time series into O(T).

## Custom gradient generative functions

`CustomGradientGF` wraps user-supplied forward/backward passes. The `IHasArgumentGrads` protocol indicates which arguments are differentiable, so gradient-based inference avoids unnecessary computation.

## Neural networks as generative functions

`NeuralNetGF` wraps MLX `nn.Module` as a deterministic generative function with full GFI support. Layer constructors (linear, sequential, relu, gelu, etc.) and training utilities using MLX's native `nn.valueAndGrad`.

Building on this, an amortized inference module provides VAE-style reparameterized ELBO training and neural importance sampling --- training a neural network guide, then using it as a proposal for importance sampling.

## Complete GFI coverage

All 9 combinators now implement the full GFI. Mask got update and regenerate. Contramap and Dimap got update-with-diffs. No more partial implementations.

## Running total

| Metric | Value |
|---|---|
| Commits today | 23 |
| Lines added | +3,933 |
| TODO items completed | 17 (41 of 66 done) |
| New source files | 5 |
| New test files | 9 |

The code is at [github.com/robert-johansson/genmlx](https://github.com/robert-johansson/genmlx).
