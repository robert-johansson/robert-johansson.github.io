---
layout: post
title: "Formalizing Psychoanalytic Theory as Bayesian Multi-Agent Reasoning"
date: 2026-02-18 21:00:00 +0100
categories: [computational-psychology]
tags: [psychoanalysis, bayesian, moser, memo, jax]
share: false
---

Tonight I'm writing about something different from the probabilistic programming posts earlier today. This project sits at the intersection of my two fields — psychology and computation — and it's the one I'm most excited about.

I've implemented Moser and von Zeppelin's (1991) cognitive-affective regulation architecture — a psychoanalytic theory of defense mechanisms, relational reasoning, and dream formation — as a family of Bayesian probabilistic programs. The entire model compiles to differentiable JAX array operations, which means it can eventually be fit to clinical data using gradient-based optimization.

The code is at [github.com/robert-johansson/moser-model](https://github.com/robert-johansson/moser-model).

## Why Moser?

Moser and von Zeppelin proposed one of the most detailed process-level accounts in psychoanalytic theory. Unlike many psychodynamic frameworks that remain at the level of metaphor, Moser described specific *mechanisms*: how wishes and internalized rules conflict, how the mind simulates the Other's response before acting, how defense activates when affect exceeds a threshold, how dreams progressively relax reality constraints. The constructs are precise — they were just never formalized computationally.

It turns out they map onto Bayesian multi-agent reasoning primitives with surprising directness:

| Moser's construct | Computational primitive |
|---|---|
| Wish-elements | Utility functions over action-response pairs |
| Rule-elements | Vulnerability-weighted shame penalties |
| Inner simulation (cxt planning) | Bayesian inference over the Other's response distribution |
| Defense (cxt stabilization) | Utility modification triggered by affect intensity |
| Ad hoc model of Other (cxt relation) | Recursive belief attribution (`thinks[]`) |
| Situation-sequences | Markov decision processes with discounted future value |
| Appraisals | Prediction errors computed against expectation baselines |
| Fantasy and dream | Counterfactual simulation with degraded reality constraints |

## The tool: memo

The implementation uses the [memo DSL](https://github.com/kach/memo) by Kartik Chandra — a domain-specific language for Bayesian recursive multi-agent reasoning that compiles to JAX. memo provides primitives that feel almost purpose-built for psychodynamic modeling:

- **`chooses`** — an agent selects from actions via softmax over utility
- **`thinks`** — recursive belief attribution (Self models Other who models Self)
- **`imagine`** — counterfactual simulation without real-world feedback
- **`observes`** — Bayesian conditioning on evidence
- **`wants`** / **`EU`** — expected utility maximization

Because everything compiles to JAX, the entire model is automatically differentiable.

## The model in six stages

The model builds cumulatively through six stages, each adding a component from Moser's architecture.

### Stage 1: Wish-rule conflict and defense

The minimal model. A subject has four possible actions — APPROACH (direct wish expression, maximum vulnerability), HINT (indirect, moderate risk), WITHDRAW (no engagement, no risk), and DEFLECT (redirection, minimal risk). The Other can ACCEPT, give an AMBIGUOUS response, or REJECT.

The subject runs an inner simulation: for each possible action, what is the Other likely to do, and what will I feel? The expected value integrates wish gratification against shame-weighted vulnerability:

```
Q(a) = E[ U_wish(a, r) - v(a) * shame * I(r = REJECT) ]
```

When shame exceeds a defense threshold, a penalty reshapes the action distribution — not by suppressing the wish, but by *transforming* it. APPROACH gets penalized; WITHDRAW and DEFLECT get bonuses. This is Moser's key insight: defense doesn't eliminate desire, it redirects its expression.

![Action distributions across stages]({{ site.url }}/images/fig1_action_distributions.png)

*Figure 1: Action probability distributions across model stages. Top left: low shame — APPROACH dominates (53%). Top middle: high shame — WITHDRAW dominates (53%), APPROACH drops to near zero. The shift from approach to withdrawal is the basic defensive pattern. Bottom row: Stages 2-3 add theory of mind and sequential planning, producing more nuanced strategies — notably DEFLECT (43%) emerges as the preferred defense in Stage 3 when planning ahead.*

### Stage 2: Theory of mind

Moser describes the "ad hoc model" — the subject's representation of the Other, who in turn has a model of the subject. Stage 2 adds this as recursive reasoning using memo's `thinks[]` block:

```
alice: thinks[
    other: chooses(pred_self in Action, ...),
    other: chooses(r in Response, ...)
]
```

Self models Other who models Self — one level of recursive belief attribution. The effect is subtle but consistent: theory of mind amplifies whatever the baseline tendency is. When conditions favor approach, ToM gives additional confidence ("the Other expects me to approach and will welcome it"). When conditions favor defense, ToM strengthens avoidance ("the Other sees my vulnerability").

### Stage 3: Sequential planning

Relations unfold over time. Stage 3 models this as a recursive Q-function with discounted future value — the subject plans not just the immediate action but the cascade of future situations it initiates.

The key emergent finding: DEFLECT replaces WITHDRAW as the preferred defensive strategy. When planning multiple steps ahead, the model discovers that maintaining a transformed relation (deflecting) is better than terminating it entirely (withdrawing). This is clinically significant — Moser distinguishes between primitive withdrawal and sophisticated defensive transformation, and the model finds this distinction on its own.

![Warmth x shame heatmaps]({{ site.url }}/images/fig2_heatmaps.png)

*Figure 2: Dominant action across the warmth x shame parameter space. Stage 1 (left) shows a clean two-region split: APPROACH (green) at low shame, WITHDRAW (red) at high shame. Stage 3 (middle) reveals DEFLECT (purple) emerging at high shame — the model discovers that transformed engagement beats withdrawal when planning ahead. Stage 4 (right) with continuous appraisals expands DEFLECT further, producing a richer behavioral landscape.*

### Stage 4: Computed appraisals

Stage 4 replaces the binary defense threshold with continuous affect monitoring. Three signals drive defense activation:

- **Prediction error** — surprise/disappointment: what happened vs. what was expected
- **Counterfactual contrast** — regret: what happened vs. what the best action would have yielded
- **Negem** — accumulated negative emotion from past relational injuries

Only negative signals matter (disappointment, not pleasant surprise), and they combine through a sigmoid function for smooth, graded defense activation.

The negem parameter captures something clinically important: patients with extensive relational trauma show defensive reactions to situations that wouldn't trouble securely-attached individuals, even when their current shame sensitivity is only moderate. Past injury lowers the threshold for future defense.

![Defense activation]({{ site.url }}/images/fig3_defense_activation.png)

*Figure 3: Defense activation mechanisms. Panel A shows the crossover from APPROACH to WITHDRAW as shame increases, with HINT and DEFLECT peaking as transitional strategies. Panel B shows how the defense threshold parameter controls onset — higher thresholds delay the penalty. Panel C demonstrates negem amplification: at moderate shame (0.4), P(APPROACH) ranges from 35% with no accumulated negative emotion down to 3% with high negem. Past relational injuries make current defense more likely.*

### Stage 5: Multi-agent internal structure

Stage 5 formalizes the wish-rule conflict as literal multi-agent reasoning within a single mind. Three internal agents interact:

- **Wish agent** — proposes actions maximizing affiliative utility, favors APPROACH
- **Rule agent** — proposes actions minimizing vulnerability, favors WITHDRAW
- **Regulator** — observes conflict intensity, selects a regulation mode

The regulator chooses between three contexts: PLANNING (balanced, both agents contribute equally), STABILIZATION (rule-dominant — defensive), and REORGANIZATION (wish-dominant with exploration — what happens in therapy or crisis when constraints loosen).

Conflict intensity is highest when both wish and rule are strong — the defining feature of neurotic conflict. The context selection uses soft Gaussian weighting rather than hard switching, so the three modes blend smoothly.

### Stage 6: Dreams

This is the most striking stage. Moser describes dreams as offline processing where reality-testing is relaxed, allowing freer transformation of relational scenarios. Stage 6 models this by introducing a `reality_testing` parameter that interpolates the Other's response between reality-constrained and wish-driven:

```
P_dream(r | a) = reality_testing * P_real(r | a)
                 + (1 - reality_testing) * P_wish(r)
```

In dreams, reality testing *degrades over time*, and faster under high shame and accumulated negative emotion. The result is progressive transformation freedom: actions that are forbidden while awake become available as the dream deepens.

![Dream vs waking]({{ site.url }}/images/fig4_dream_vs_waking.png)

*Figure 4: Dream vs. waking states. Panel A: Under high shame, the action distribution shifts dramatically as reality testing decreases — WITHDRAW dominance gives way to near-uniform distribution at RT=0.1 (deep dream). Panel B: P(APPROACH) across shame x reality-testing — high shame eliminates approach while awake (dark red, right side) but dreams restore it (green, left side). Panel C: Entropy (behavioral freedom) increases as reality testing decreases, with high-shame individuals showing the largest gain — those most constrained while awake benefit most from dreaming.*

The entropy result in Panel C is the computational signature of what Moser calls "transformation freedom." High-shame individuals go from minimal behavioral freedom while awake (dominated by WITHDRAW) to near-maximum freedom in deep dreams (all actions roughly equally viable). The dream's regulatory function is precisely to enable exploration that waking life forbids.

## What emerges without being programmed

Several clinically relevant patterns arise from the model's dynamics rather than being hand-coded:

**Strategic defense.** The sequential model (Stage 3) discovers on its own that DEFLECT is a better defensive strategy than WITHDRAW when planning ahead — maintaining a transformed relation beats exiting entirely.

**Trauma amplification.** The appraisal system (Stage 4) produces the clinical observation that relational trauma history lowers defense thresholds in all future relationships, without a separate mechanism for "stored affects." Negem simply shifts the baseline of affect intensity.

**Dream self-repair.** High-shame, high-negem individuals show the most dramatic transformation freedom in dreams (Stage 6) because reality-testing degradation is affect-dependent. The most constrained patients produce the most wish-fulfilling dreams — the dream serves a self-repair function.

## What this opens up

The entire model is differentiable end-to-end. JAX's `value_and_grad` can compute gradients of model predictions with respect to all parameters — shame sensitivity, warmth priors, defense thresholds, accumulated negative emotion. This means the model could, in principle, be fit to clinical data: action choices, reaction times, emotion ratings, linguistic markers from therapy transcripts.

That's the next step — moving from "the model produces clinically coherent patterns" to "the model's parameters can be estimated from individual patient data."

The implementation is a single Python file, 543 lines, using the memo DSL. The code and all figures are at [github.com/robert-johansson/moser-model](https://github.com/robert-johansson/moser-model).

### References

- Moser, U., & von Zeppelin, I. (1991). *Der geträumte Traum: Wie Träume entstehen und sich verändern*. Stuttgart: Kohlhammer.
- Baker, C. L., Saxe, R., & Tenenbaum, J. B. (2009). Action understanding as inverse planning. *Cognition*, 113(3), 329-349.
- Chandra, K., Chen, T., Tenenbaum, J. B., & Ragan-Kelley, J. (2025). A Domain-Specific Probabilistic Programming Language for Reasoning about Reasoning (or: a memo on memo). *Proceedings of the ACM on Programming Languages*, 9(OOPSLA2), 784-814.
- Houlihan, S. D., & Tenenbaum, J. B. (2024). Computational models of affect and appraisal. Working paper, MIT.
