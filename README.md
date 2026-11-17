# Nested Learning for AI-Native 6G Networks

[![tests](https://github.com/Abraheem13/nested-learning-6g-tutorial/actions/workflows/tests.yml/badge.svg)](https://github.com/Abraheem13/nested-learning-6g-tutorial/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reference implementation, case study and evaluation controls for
multi-timescale multi-agent reinforcement learning in 6G network slicing.

This repository accompanies the book chapter *"Nested Learning for AI-Native 6G
Networks: A Tutorial on Multi-Timescale Intelligence and How to Evaluate It"*
(Springer, *6G Communication: Opportunities, Challenges, and Future
Directions*). It is an **independent re-implementation** of the learner proposed
in Rashid et al., *IEEE OJ-COMS* 7:7624–7640 (2026); the original code of that
paper is not used here.

**Everything runs on a laptop CPU.** The per-seed run logs behind every result
are in `results/`, and every number, table and data figure in the chapter is
regenerated from them by `scripts/make_chapter_assets.py`. The manuscript
itself is not included here until the chapter is published.

---

## What the chapter finds (read this first)

* Against IPPO at the conventional learning rate `5e-4`, the multi-timescale
  learner looks clearly better (+12.6% first-40 reward).
* With the baseline's learning rate swept, single-rate IPPO at `5e-3` beats it.
* With an equal tuning budget the two are equivalent out of sample, and the
  advantage does not grow with drift or survive more environment noise.
* The hand-designed depth partition matters (alternatives are significantly
  worse) and a gradient-SNR rule does not recover it.
* The advantage seen under the conventional comparison is present with no
  drift at all and does not change when the drift bands are collapsed or
  widened.

So, in this benchmark, the mechanism is **not demonstrated**. See
`results/REPORT.md`, `ROBUSTNESS.md`, `NOISE_PROBE.md` and `REVIEW.md`.

---

## Quick start

```bash
git clone https://github.com/Abraheem13/nested-learning-6g-tutorial.git
cd nested-learning-6g-tutorial
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .

make test           # the test suite, including the update-rule invariants
make smoke          # ~1 minute: the pipeline end to end
```

## Reproducing the chapter

```bash
python -m pytest tests -q
python scripts/run_experiments.py --preset full --workers 4   # headline, ablation, severity, scalability
python scripts/make_report.py                                 # REPORT.md
python scripts/run_robustness.py --workers 4                  # controls 1-3, powered ablation
python scripts/run_noise_probe.py --workers 4                 # control 4
python scripts/run_review_analyses.py --workers 4             # partition sensitivity, drift structure
python scripts/make_chapter_assets.py                         # every number, table and figure in the chapter
python scripts/verify_chapter.py                              # checks numbers and stated claims against the logs
```

Runs are cached by content, so each step only trains what is missing. On an
Apple M-series laptop with four workers the full suite takes well under an hour.
Training is deterministic for a given machine and library stack (the test
suite checks this), but results are **not bit-identical across platforms**:
CPU kernels differ between x86 and ARM. The committed logs were produced on
Apple Silicon (Python 3.13, PyTorch 2.14); `make_chapter_assets.py` reads them
without retraining. `verify_chapter.py` also checks the manuscript when it is
present (`chapter/chapter.tex`), and `build_chapter.py` compiles it with the
Springer class; both are no-ops for the manuscript in this public copy.

### Presets

| Preset | Episodes | Seeds | Wall clock (4 cores) | Use for |
|---|---|---|---|---|
| `smoke` | 6 | 2 | ~1 min | checking the pipeline runs |
| `quick` | 40 | 5 | ~8 min | indicative trends while iterating |
| `full` | 80 | 10 | **~20 min** | the reported results |
| `extended` | 120 | 10 | ~45 min | checking conclusions aren't an artefact of `E=80` |

Runs are **cached and resumable**. Interrupt with Ctrl-C and re-launch: it picks
up where it stopped. Cache entries are content-addressed, so a run needed by two
experiments is computed once.

---

## What gets produced

```
results/
├── REPORT.md                  written summary of every finding, with the
│                              numbers that don't favour the method included
├── raw/*.csv|json             per-episode logs + a config/provenance sidecar
├── manifests/*.json           which runs constitute which experiment
├── tables/
│   ├── headline.tex           booktabs, \input-ready for the Springer template
│   ├── headline_significance.tex
│   ├── ablation.tex
│   ├── severity.tex
│   └── scalability.tex
└── figures/
    ├── drift_trajectory.pdf   the three drift bands (needs no training)
    ├── case_study.pdf         3-panel: efficiency, stability, scalability
    ├── learning_curves.pdf
    ├── switching_cost.pdf
    ├── ablation.pdf
    ├── severity_sweep.pdf
    ├── per_slice_sla.pdf
    └── scalability.pdf
```

The chapter's own tables and figures are built separately, from the same logs,
by `scripts/make_chapter_assets.py` (see *Reproducing the chapter*). The
reports `ROBUSTNESS.md`, `NOISE_PROBE.md` and `REVIEW.md`, with
`review_stats.json` and `chapter_numbers.json`, are written by the scripts that
produce them.

---

## The environment

`SliceEnv` implements a **Multi-Timescale Non-Stationary Markov Game**: `N`
base stations allocating PRBs across `S` slices under three drift bands.

| Band | Signal | Default period |
|---|---|---|
| fast | per-agent, per-slice channel gain (AR(1)) | every step |
| medium | per-slice demand multiplier | `τ_med = 15` steps |
| slow | SLA regime indicator | `τ_slow = 50` steps |

* **Action space** — `K = C(k+S-1, S-1) = 15` allocation templates on the
  discretised slice simplex (`k = 4`, `S = 3`).
* **Observation** — `oᵢ = [hᵢ | qᵢ | d | ρ] ∈ ℝ¹²`, strictly local.
* **Reward** — `rᵢ = Σₛ wₛ(ρ)·σᵢₛ − λ_sw‖aᵢᵗ − aᵢᵗ⁻¹‖₂`, so policy churn is
  intrinsically penalised.
* **Severity** — a single scalar `κ ∈ [0, 2.5]` controls how sharply demand
  differs between regimes. `κ = 0` recovers demand stationarity.
* **The drift clock never resets at episode boundaries.** Episodes are an
  artefact of the training loop; the network doesn't know about them.

No `gym` dependency — the environment is plain NumPy, so it installs anywhere.

---

## Methods

One learner implements all seven algorithms; they differ **only** by a
declarative `MethodSpec`. That is deliberate: it makes it structurally
impossible for two methods to diverge through an accidental implementation
detail rather than the mechanism under study.

| Method | Rates | Anchor | Role |
|---|---|---|---|
| `ippo` | `5e-4` | – | decentralised baseline |
| `mappo` | `5e-4` | – | centralised critic (CTDE) |
| `ewc_ippo` | `5e-4` | – | continual-learning baseline |
| **`nested_marl`** | `1e-4, 5e-4, 2e-3` | EMA, `β=0.95` | **proposed** |
| `nested_same_rate` | `5e-4 ×3` | EMA | removes *only* timescale separation |
| `nested_two_timescale` | `1e-4, 1e-4, 2e-3` | EMA | collapses slow+medium |
| `nested_no_ema` | `1e-4, 5e-4, 2e-3` | – | removes *only* the anchor |

### Why the ablation is causally interpretable

`tests/test_all.py::TestUpdateRuleInvariants` proves an exact identity:

> **Three Adam optimisers at a single shared rate, with no memory anchor, are
> bit-identical to one Adam optimiser over all parameters.**

Adam is element-wise, so partitioning parameters across optimisers at the same
rate cannot change the update. The partition is therefore *inert on its own*.
Any measured difference between Nested-MARL and its `same_rate` ablation must
come from the **rates** or the **anchor** — it cannot be an artefact of
splitting the network, and it cannot be extra capacity, because the parameter
count is identical (11,024 per agent in every method).

---

## Statistical practice

Deep RL results reverse under reseeding more often than the literature admits,
so the analysis follows current best practice (Henderson et al. 2018; Agarwal
et al. 2021):

* **Paired permutation tests** across seeds — distribution-free, exact by
  enumeration for `n ≤ 20`.
* **Holm–Bonferroni** correction across the family of baselines.
* **IQM** and **percentile-bootstrap CIs** alongside the mean, so one lucky
  seed cannot carry a claim.
* **Probability of improvement** with a bootstrap interval.
* **Three evaluation windows** — first-40 (sample efficiency), all-80
  (overall), last-20 (asymptotic) — reported *together*, so a
  sample-efficiency gain is never quoted as an asymptotic one.

**On p-value floors.** With `n` seeds the smallest attainable two-sided
permutation p-value is `3/(2ⁿ+1)`: 0.0029 at `n=10`, 0.09 at `n=5`. The report
flags any p-value sitting at that floor. A five-seed ablation *cannot* produce
a significant result at α=0.05 — which is why ablations are reported as effect
sizes and directions, not as significance claims.

---

## Repository layout

```
src/nested6g/
├── env/                 the simulator: actions, drift bands, MTNS-MG
├── agents/
│   ├── networks.py      actor-critic + depth-based parameter partition
│   ├── ppo.py           shared PPO core (identical across all methods)
│   ├── methods.py       declarative registry of every method and variant
│   └── learner.py       the multi-timescale update rule; alternative partitions
├── analysis/            permutation tests, Holm, bootstrap; tables; plots
├── experiments.py       the experiment grids
└── runner.py            resumable, parallel, content-addressed execution

scripts/
├── run_experiments.py      training CLI (headline, ablation, severity, scalability)
├── make_report.py          REPORT.md, tables, figures
├── run_robustness.py       controls 1-3 and the powered ablation
├── run_noise_probe.py      control 4
├── run_review_analyses.py  partition sensitivity and drift structure
├── make_chapter_assets.py  every number/table/data figure in the chapter
├── verify_chapter.py       manuscript vs run logs, and publisher constraints
└── build_chapter.py        flattened chapter compiled in svmult

results/                 committed run logs (raw/), manifests, reports
tests/                   test suite, including the bitwise update-rule invariants
.github/workflows/       continuous integration
```

---

## Provenance and honest scope

Please read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before
reporting any number from this repository. In short:

* This is a **clean-room implementation from the published system-model
  specification**, written to be readable and teachable. The architecture is
  verified against the reference parameter counts (11,024 per agent; a
  centralised critic growing 6,593 → 11,969 from `N=3` to `N=10`), and the
  qualitative picture of the original comparison reproduces; the additional
  controls then remove it. **Absolute numbers will not match the original
  paper**, because the service model, the RNG stream and many
  implementation details are necessarily re-specified here.
* The simulator is a **controlled abstraction**, not a channel-accurate
  emulator. It deliberately excludes physical-layer confounders in order to
  isolate the multi-timescale learning effect. It is not a substitute for
  ns-3, OpenAirInterface, or an O-RAN testbed.
* Every figure and table is generated from the logs. No number in the reports
  or the chapter is hand-entered.

---

## Requirements

Python ≥ 3.10, and: `torch` (CPU is fine), `numpy`, `pandas`, `matplotlib`,
`scipy`, `tabulate`. Install with `pip install -e .`, or
`pip install -r requirements.txt`.

Tested on Linux and macOS (Apple Silicon), Python 3.10–3.13.

## License

MIT — see [`LICENSE`](LICENSE).
