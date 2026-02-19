---
layout: post
title: "Toward a Lambda Calculus for GenMLX"
date: 2026-02-19 15:30:00 +0100
categories: [probabilistic-programming]
tags: [clojurescript, mlx, gen, lambda-calculus, formal-methods, apple-silicon]
share: false
---

The GenJAX POPL 2026 paper (Becker et al., [doi:10.1145/3776729](https://dl.acm.org/doi/abs/10.1145/3776729)) introduced **lambda\_GEN** --- a simply-typed lambda calculus that gives the Generative Function Interface a formal foundation. It uses quasi-Borel spaces for its denotational semantics, defines `simulate` and `assess` as source-to-source program transformations, and proves vectorization correct via logical relations.

[GenMLX](https://github.com/robert-johansson/genmlx) implements the same interface on Apple Silicon. I've been exploring what a **lambda\_MLX** --- a formal calculus grounding GenMLX --- would look like. The exercise turned out to be interesting because GenMLX's design choices lead to different formal structures than lambda\_GEN in several places.

## ClojureScript is already close

The gap between a formal lambda calculus and the implementation is smaller for ClojureScript than for Python/JAX. `gen` bodies are lambda abstractions. `dyn/trace` is an effect operation. Protocols are type classes. Multimethods are open dispatch. The core data structures --- persistent maps, vectors, records --- have direct formal counterparts.

## The handler architecture maps to algebraic effects

GenMLX's handler system decomposes each GFI operation into a **pure state transition** and a **thin mutable wrapper**:

```
transition : (State, Addr, Dist) -> (Value, State')
```

The state is an immutable map threaded through the model body. This is exactly the operational semantics of algebraic effect handlers (Plotkin & Pretnar 2009). In lambda\_MLX, I formalize this as a handler computation type `H(sigma, tau)` --- a state-passing monad where sigma is the handler state type:

```
sigma_sim = {key: Key, choices: gamma, score: R}
sigma_gen = {key: Key, choices: gamma, score: R, weight: R, constraints: gamma_obs}
sigma_upd = {key: Key, choices: gamma, score: R, weight: R, constraints: gamma_new,
             old-choices: gamma, discard: gamma_disc}
```

Each handler mode gets its own state type, making the invariants explicit. The denotation is straightforward: `[[H(sigma, tau)]] = [[sigma]] -> [[tau]] x [[sigma]]`.

## Broadcasting vs vmap: a semantic property instead of a syntactic transformation

This is the most interesting divergence. The POPL paper defines `vmap_n` as a **syntactic program transformation** on types and terms. It transforms a scalar model into a batched version and proves correctness via logical relations (Theorem 3.3, Corollary 3.4).

GenMLX does something different: it changes the **domain** of inputs, not the program. Instead of transforming `f` into `vmap_n{f}`, GenMLX calls `f` with `[N]`-shaped arrays and lets MLX broadcasting handle the rest. The model code is unchanged.

The correctness statement becomes: if the handler transitions are shape-agnostic and distributions broadcast correctly, then running the unmodified program with `[N]`-shaped inputs produces the same result as running N independent copies.

The proof is simpler than the paper's. The key lemma has three parts:

1. `dist-sample-n(d, key, N)` produces `[N]` independent samples from d
2. `dist-log-prob(d, v)` for `v : eta[N]` returns element-wise density evaluation via broadcasting
3. Score accumulation `sigma.score + lp` broadcasts correctly (scalar + [N] = [N])

Each component `score[i]` of the final `[N]`-shaped score then equals the score from a scalar execution with the i-th sampled values. The product measure factorization follows from independence of PRNG key splitting.

Same logical relations as Theorem 3.3, but the proof goes through handler shape-agnosticism rather than syntactic transformation correctness.

## lambda\_MLX must formalize more than lambda\_GEN

The POPL paper formalizes `simulate` and `assess`. GenMLX's full GFI needs additional program transformations:

```
simulate{G_gamma eta}   = P (gamma x eta x R)
assess{G_gamma eta}     = gamma -> eta x R
generate{G_gamma eta}   = gamma_obs -> P (gamma x eta x R)
update{G_gamma eta}     = (gamma x eta x R) -> gamma_new -> P (gamma x eta x R x gamma_disc)
edit{G_gamma eta}       = (gamma x eta x R) -> EditReq(gamma) -> P (gamma x eta x R x EditReq(gamma))
```

The **edit interface** is particularly clean to formalize. Three typed request constructors --- `ConstraintEdit`, `SelectionEdit`, `ProposalEdit` --- each with automatic backward request generation:

```
edit(tr, ConstraintEdit(c))             = (tr', w, ConstraintEdit(disc))
edit(tr, SelectionEdit(s))              = (tr', w, SelectionEdit(s))
edit(tr, ProposalEdit(f, fa, b, ba))    = (tr', w, ProposalEdit(b, ba, f, fa))
```

The backward request for a proposal edit swaps forward and backward --- this is what makes SMCP3's reversible kernels work.

**Diff-aware incremental updates** also need formalization. MapCombinator stores per-element scores and skips unchanged elements:

```
When delta = VectorDiff(S):
  for i in S union constrained_indices:
    update element i                -- O(|S|) work
  for i not in S:
    reuse cached (tr_i, score_i)    -- O(1) per element
```

This turns O(n) update into O(|S|) for sparse changes.

## Heterogeneous vs homogeneous grading

The paper's `cond` construct requires **homogeneous grading** --- all branches must have the same trace type so the selector can merge traces. GenMLX's Switch supports **heterogeneous grading** (different branch trace types), padding with absent values rather than NaN.

How to formalize this cleanly is an open question. Union types? Dependent types? The paper sidesteps it by requiring homogeneity.

## Open questions

Several things I haven't resolved:

1. **Dynamic vs static trace types.** lambda\_GEN's grading gamma is determined statically. GenMLX's dynamic DSL means trace types are determined at runtime. How to reconcile?

2. **Splice in batched mode.** Currently unsupported because sub-GF calls may have control flow dependent on `[N]`-shaped sampled values. The paper handles this via `cond` with `select_p` --- could GenMLX adopt something similar?

3. **Lazy evaluation as semantics.** MLX's lazy evaluation could be formalized as call-by-need with explicit forcing. Operations build thunks; `eval!` forces materialization. The fusion property --- consecutive lazy operations compile to a single GPU dispatch --- is analogous to the paper's observation that "JAX's program tracing performs partial evaluation on these lightweight effect handlers, thereby evaluating them away."

4. **The order of vmap and seed.** The paper notes "it is essential that seeding happens after vectorization." GenMLX's broadcasting approach sidesteps this entirely --- the PRNG key is split once and used directly.

## What lambda\_MLX would contribute

| Aspect | lambda\_GEN (POPL 2026) | lambda\_MLX |
|---|---|---|
| Core transformations | simulate, assess | + generate, update, regenerate, edit |
| Vectorization | vmap (syntactic transform) | Broadcasting (semantic property) |
| Correctness proof | Logical relations | Same relations, simpler proof |
| Effect handling | Lightweight handlers on JAX | Algebraic effect handlers as state monad |
| Stochastic branching | cond, homogeneous grading | Switch, heterogeneous grading |
| Incremental computation | Not formalized | Diff-aware update with change propagation |
| Edit interface | Not formalized | Typed EditReq with backward computation |
| Trainable parameters | Not formalized | param(k, default) as effect operation |

The main insight so far: GenMLX's purely functional design --- persistent data structures, explicit state threading, shape-agnostic handlers --- makes the formal treatment simpler in some places (broadcasting correctness) and richer in others (the full GFI surface area). Whether this leads to an actual paper remains to be seen, but the exercise of writing it down has already clarified the system's design.
