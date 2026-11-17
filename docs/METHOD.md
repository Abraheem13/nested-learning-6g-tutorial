# Method

This document states the problem, the algorithm and the reasoning behind each
design choice, at the level of detail a reader needs to reimplement the method
from scratch. It is written to be read alongside Sections 2–4 of the chapter.

---

## 1. The problem: multi-timescale non-stationarity

A base station in a 6G RAN simultaneously observes signals that evolve at
separated rates:

| Signal | Physical origin | Order of magnitude |
|---|---|---|
| channel gain | fading | milliseconds |
| traffic demand | user population and app mix | seconds to minutes |
| SLA regime | operator re-tuning service contracts | minutes to hours |

The joint dynamics a learner must track therefore have **at least three
meaningful timescales**. We model this as a *multi-timescale non-stationary
Markov game*

```
G = ⟨ N, S_env, {A_i}, {r_i}, P, π_ρ, π_d, γ ⟩
```

where `P` is the single-step transition kernel and `π_ρ`, `π_d` are the regime
and demand drift kernels acting at rates `1/τ_slow` and `1/τ_med`.

The MTNS-MG is strictly **more general** than a stationary Markov game (`P`
depends on the drifting `ρ_t, d_t`), and strictly **more structured** than a
general non-stationary game: the drift is *hierarchical*. That structure is
exactly what a multi-timescale learner can exploit and a single-timescale
learner must ignore.

> **A caution on timescales.** The fast band here is a *modelled
> abstraction* of channel variation, not a claim that an xApp on the near-RT
> RIC (10 ms – 1 s) tracks fast fading. Real channel coherence times are
> shorter than the near-RT control loop. The drift this method addresses in a
> deployed setting is principally traffic and SLA/policy drift; the fast band
> represents short-horizon load and link-quality variation *within* the
> controller's reach. PHY coherence times and control-loop timescales should
> not be conflated.

---

## 2. The single-timescale bottleneck

Consider tracking a drifting optimum with constant step size `α`. The
steady-state error decomposes into two competing terms:

```
error(α)  ≈  C₁·α·σ²       (gradient-noise term: larger steps, noisier tracking)
           + C₂·Δ²/α²    (lag term: smaller steps, slower to follow the drift)
```

This is the classic **misadjustment-versus-lag** trade-off of adaptive
filtering (LMS), and it has an exact analogue in stochastic optimisation under
distributional drift. Minimising the sum gives an interior optimum `α*` and,
critically, a **non-zero floor**: no single step size drives the error to zero
while the target keeps moving.

Two independent literatures arrive at a *cube-root* exponent:

* For proximal SGD under drift, the optimal constant step size scales as
  `α* ∝ (Δ²/σ²)^{1/3}` with a matching lower bound on the irreducible tracking
  error (Cutler, Drusvyatskiy & Harchaoui, *JMLR* 2023).
* For two-timescale stochastic approximation, finite-sample analysis gives a
  best overall rate of `n^{−1/3}`, against `n^{−1/2}` for a single-timescale
  scheme in the stationary case (Dalal, Thoppe, Szörényi & Mannor, *COLT* 2018).

This is the **"1/3 barrier"**: a single rate cannot escape a cube-root wall
when the target drifts, and separating the rates is the principled way around
it. Whether the barrier binds in a given system is an empirical question; in
this benchmark it does not (a tuned single-rate learner matches the
multi-timescale one, `results/ROBUSTNESS.md`).

> Two precision notes. Konda & Tsitsiklis (*Ann. Appl. Prob.* 2004) give
> *asymptotic* rates via a CLT, not finite-time bounds. And the Dalal et al. `n^{−1/3}` is described by its
> authors as the best achievable *within their analysis*, not as a proven
> matching lower bound.

---

## 3. Nested Learning as a design primitive

Behrouz et al. (NeurIPS 2025) reframe a model as a hierarchy of nested
optimisation problems, each with its own context flow and **update frequency**.
Momentum, Adam and continuum-memory systems all become instances of one
principle:

> *Different components of a learning system should update at frequencies
> matched to the temporal structure of the signals they respond to.*

Applying that principle here: channel, demand and regime occupy three temporal
bands, so assign each to a parameter group with a matched rate, and anchor the
slow group with a memory trace.

> **This mapping is our interpretation, not a claim in the source.** Behrouz et
> al.'s Continuum Memory System is a set of learnable modules at geometrically
> spaced frequencies; the EMA anchor used here is a specific, simpler
> instantiation of the same idea.

---

## 4. The algorithm

### 4.1 Parameter partition

Each agent's parameters `θᵢ = (θᵢ^π, θᵢ^V)` are partitioned by **layer depth**:

```
θᵢ,₀ : input-layer weights   (slow)    ← observation→feature map, shared across regimes
θᵢ,₁ : hidden-layer weights  (medium)  ← intermediate features
θᵢ,₂ : output-layer weights  (fast)    ← the decision boundary / value head
```

The same partition applies to actor and critic, so the two heads share
timescales — letting one head race ahead of the other is a known source of
actor-critic instability.

The hypothesis this encodes: representation-level features should adapt slowly
because they encode observation-to-action-space regularities that hold across
regimes, whereas the decision boundary should adapt quickly because the optimal
action changes with the fast band.

### 4.2 Update rule

After each on-policy PPO update producing gradients `gᵢ = ∇_{θᵢ} L^PPO`:

```
(3)  θᵢ,ₗ ← θᵢ,ₗ − αₗ · Adamₗ(gᵢ,ₗ),        l ∈ {0,1,2},   α₀ ≪ α₁ ≪ α₂
(4)  θ̄ᵢ   ← λ·θ̄ᵢ + (1−λ)·θᵢ                 continuum-memory trace
(5)  θᵢ,₀ ← β·θᵢ,₀ + (1−β)·θ̄ᵢ,₀             stabilise the slow group
```

Adam operates *within* each group (adaptive per-parameter scaling); the
per-group rates provide separation *across* groups. This is precisely the
Nested Learning view in which momentum is itself a nested optimisation.

Defaults: `(α₀, α₁, α₂) = (1e-4, 5e-4, 2e-3)`, `λ = 0.995`, `β = 0.95`.
Setting `β = 1` disables stabilisation (the `no_ema` ablation).

### 4.3 Why this is a controlled experiment

Two facts make the ablation causally interpretable, and both are enforced by
tests:

1. **The partition is inert on its own.** Adam is element-wise, so three
   optimisers at a single shared rate with no anchor are *bit-identical* to one
   optimiser. Verified exactly (difference `0.0`) in
   `TestUpdateRuleInvariants::test_group_split_alone_changes_nothing`.
2. **No extra capacity.** Every method has 11,024 parameters per agent. The
   multi-timescale structure partitions one network; it never enlarges it.

Therefore, within a given configuration, any difference between `nested_marl`
and `nested_same_rate` is caused by the **rates**, and any difference from
`nested_no_ema` by the **anchor**. This makes the ablation internally valid; it
does not show that the configuration beats a tuned single-rate learner (it does
not, `results/ROBUSTNESS.md`).

### 4.4 Cost

Over independent PPO, the additions are: three small Adam states instead of one
(negligible memory), one EMA copy of the parameters, and two linear blending
operations per outer update. Inter-agent messages at execution: **zero**, since
each agent is an independent learner — a concrete operational advantage over
centralised-critic schemes, whose critic parameter count and control-plane
signalling both grow with the cohort.

---

## 5. What this method does *not* do

* **It is not hierarchical RL.** Options and goal-conditioned HRL abstract
  *actions* over time. This separates *learning rates* on shared parameters.
  The two axes are orthogonal and could be combined.
* **It is not meta-RL.** PEARL, RL² and VariBAD infer a task context and adapt
  in a hidden state or an inner loop. This is a fixed update rule with no task
  inference and no meta-training.
* **The partition is fixed a priori.** `scripts/run_review_analyses.py` shows
  that it matters and that a gradient signal-to-noise rule does not recover it.
  Learning the partition remains open.
* **The theory is local.** Finite-time two-timescale results are local
  guarantees under regularity assumptions; global convergence under
  non-convexity is open, as it is for essentially all deep RL.

---

## 6. Relation to continual learning

The retention-versus-plasticity framing connects the design to a documented
failure mode: deep networks progressively **lose plasticity** under continual
training (Dohare et al., *Nature*, 2024), and catastrophic forgetting under
sequential task exposure is well established (Kirkpatrick et al., *PNAS*,
2017).

In that framing the **EMA-anchored slow group is retention** and the **fast
group is plasticity**: a structural answer to "what should be remembered and
what should be adapted", as opposed to a regularisation penalty (EWC) or a reset
heuristic. This is a design rationale, not a result: in this simulator removing
the anchor leaves reward unchanged, and a tuned single-rate learner matches the
full method (`results/ROBUSTNESS.md`).
