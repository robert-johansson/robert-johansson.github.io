---
layout: post
title: "GPU-Accelerated Probabilistic Programming in ClojureScript"
date: 2026-02-18 10:30:00 +0100
categories: [probabilistic-programming]
tags: [clojurescript, mlx, prob-cljs, apple-silicon, bayesian]
share: false
---

Today I integrated [MLX](https://ml-explore.github.io/mlx/) into [prob-cljs](https://github.com/robert-johansson/prob-cljs), my ClojureScript probabilistic programming library. The result: autograd, HMC, NUTS, and variational inference — all running on Metal GPU through Apple Silicon.

The core library stays zero-dependency. MLX is an optional add-on when running under [nbb](https://github.com/babashka/nbb).

## The problem

Pure ClojureScript has no automatic differentiation. That means no gradient-based inference — no HMC, no NUTS, no variational inference. You're stuck with Metropolis-Hastings and enumeration. These work, but they don't scale to the kind of models I want to build.

## The solution: MLX via node-mlx

Apple's [MLX](https://ml-explore.github.io/mlx/) framework provides autograd, lazy evaluation, and JIT compilation to Metal GPU programs. The Node.js bindings ([node-mlx](https://github.com/frost-beta/node-mlx)) expose all of this with ~20ns native call overhead.

The integration is three files:

```
src/prob/mlx/
  core.cljs       Tensors, arithmetic, linalg, autograd, vmap, compile
  dist.cljs       MLX-backed distributions with IDifferentiable protocol
  inference.cljs  HMC, NUTS, Variational Inference (ADVI)
```

## Why ClojureScript fits MLX well

ClojureScript's prefix notation turns out to be a better surface for MLX than JavaScript:

```javascript
// JavaScript — nested method calls
mx.add(a, mx.multiply(b, mx.subtract(c, d)))
```

```clojure
;; ClojureScript — reads naturally
(mx/add a (mx/multiply b (mx/subtract c d)))
```

And MLX's lazy evaluation maps directly to how we already think in Clojure — operations build up a deferred computation graph, and `eval!` forces it (like `doall`):

```clojure
(let [x (mx/random-normal [100 100])     ;; lazy
      y (mx/matmul x (mx/transpose x))   ;; lazy
      z (mx/mean (mx/diag y))]           ;; lazy
  (mx/eval! z)                           ;; now runs on GPU
  (mx/item z))                           ;; extract result
```

Higher-order function transforms compose via threading:

```clojure
(def fast-grad (-> log-posterior mx/grad mx/compile-fn))
```

## Autograd

This is the big one. Gradients that were previously impossible in pure ClojureScript:

```clojure
;; Gradient of f(x) = x²
(def grad-f (mx/grad (fn [x] (mx/multiply x x))))
(let [result (grad-f (mx/scalar 3.0))]
  (mx/eval! result)
  (mx/item result))
;; => 6.0  (df/dx = 2x, at x=3)

;; Second derivatives compose
(def f   (fn [x] (mx/multiply (mx/multiply x x) x)))  ;; x³
(def f'  (mx/grad f))                                   ;; 3x²
(def f'' (mx/grad f'))                                   ;; 6x
(let [result (f'' (mx/scalar 2.0))]
  (mx/eval! result)
  (mx/item result))
;; => 12.0
```

## Bayesian regression with HMC

Here's a complete example — Bayesian linear regression with Hamiltonian Monte Carlo, running leapfrog integration on GPU with JIT-compiled Metal kernels:

```clojure
(ns bayesian-regression
  (:require [prob.mlx.core :as mx]
            [prob.mlx.inference :as infer]))

;; Data: y ≈ 2x + 1
(def xs (mx/array [1 2 3 4 5]))
(def ys (mx/array [3.1 4.9 7.2 8.8 11.1]))

;; Log-posterior: params = [w, b]
(defn log-posterior [params]
  (let [w (mx/index params 0)
        b (mx/index params 1)
        y-hat (mx/add (mx/multiply w xs) b)
        residuals (mx/subtract ys y-hat)
        obs-lp (mx/multiply (mx/scalar -0.5)
                 (mx/sum (mx/multiply
                           (mx/divide residuals (mx/scalar 0.5))
                           (mx/divide residuals (mx/scalar 0.5)))))
        prior-w (mx/multiply (mx/scalar -0.5)
                  (mx/divide (mx/multiply w w) (mx/scalar 100)))
        prior-b (mx/multiply (mx/scalar -0.5)
                  (mx/divide (mx/multiply b b) (mx/scalar 100)))]
    (mx/add obs-lp (mx/add prior-w prior-b))))

;; HMC sampling on Metal GPU
(def samples
  (infer/hmc
    {:samples 1000
     :step-size 0.005
     :leapfrog-steps 20
     :burn 200}
    log-posterior
    (mx/zeros [2])))

(let [mean-arr (infer/sample-mean samples)
      std-arr  (infer/sample-std samples)]
  (mx/eval! mean-arr std-arr)
  (let [m (mx/->clj mean-arr)
        s (mx/->clj std-arr)]
    (println "w:" (first m) "±" (first s))    ;; ~2.0 ± 0.1
    (println "b:" (second m) "±" (second s)))) ;; ~1.0 ± 0.4
```

NUTS is also available — it adapts trajectory length automatically, so you don't need to tune `leapfrog-steps`.

## What this unlocks

| Capability | Before | After |
|---|---|---|
| Autograd | Not possible in pure CLJS | `mx/grad`, `mx/value-and-grad` |
| HMC | Not possible | Compiled Metal kernels |
| NUTS | Not possible | Adaptive tree depth |
| Variational inference | Not possible | ADVI + Adam |
| Multivariate Normal | Needed linalg | Cholesky-based, GPU-accelerated |
| Matrix ops | No Cholesky, SVD, etc. | Full linalg |

## Limitations

- **Apple Silicon only.** MLX requires M-series chips. The core `prob.*` library works everywhere.
- **nbb only.** Won't work in Scittle/browser (native addon).
- **Mean-field VI only.** Diagonal Gaussian guide — full-covariance or normalizing flows would be the next step.

The code is at [prob-cljs](https://github.com/robert-johansson/prob-cljs). Install MLX support with `npm install @frost-beta/mlx`.
