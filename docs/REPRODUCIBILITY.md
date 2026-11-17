# Reproducibility, provenance and honest scope

Read this before quoting any number from this repository.

---

## 1. What this code is

A **clean-room implementation written from the published system-model
specification**, organised for teaching rather than for matching a previous
result digit for digit.

**What is verified against the reference system:**

| Quantity | Reference | This code |
|---|---|---|
| Actor parameters per agent | 5,967 | 5,967 ✓ |
| Critic parameters per agent | 5,057 | 5,057 ✓ |
| Total per agent | 11,024 | 11,024 ✓ |
| Centralised critic, `N=3` | 6,593 | 6,593 ✓ |
| Centralised critic, `N=10` | 11,969 | 11,969 ✓ |
| Allocation templates `K` | 15 | 15 ✓ |
| Observation dimension `d_o` | 12 | 12 ✓ |

These are checked by `tests/test_all.py::TestNetworks`. Matching all of them is
strong evidence that the architecture and action space are reconstructed
correctly.

**What is necessarily re-specified**, because it is not fully determined by the
published description:

* the queueing and service model (capacity, arrival process, backlog carry-over);
* the exact functional form of the soft SLA indicator (throughput × latency ×
  reliability);
* the regime-conditional demand profile and the per-slice sensitivity vector;
* per-agent cell-load heterogeneity;
* the online-EWC formulation (Fisher decay, penalty coefficient);
* the RNG stream and network initialisation.

**Consequence: absolute values will differ from the original paper.** Report
the numbers this code produces, cite the original separately, and do not
present the two as the same experiment.

---

## 2. What reproduces qualitatively

Running `make full` should reproduce these *directional* findings. If one does
not, that is a result worth reporting, not a bug to tune away.

1. Nested-MARL leads on **sample efficiency** (first-40 reward) over IPPO,
   MAPPO and EWC-IPPO.
2. The advantage is **concentrated early**: the asymptotic (last-20) gap is
   markedly smaller than the first-40 gap, and may not reach significance.
3. The **`same_rate` ablation falls below the IPPO baseline** — the decisive
   control, since it isolates timescale separation from architectural depth.
4. Nested-MARL leads IPPO **at every drift severity** tested, with the
   relative advantage largest at mild drift.
5. MAPPO **degrades as the cohort grows**, consistent with the difficulty of
   learning one centralised value function over a state that grows with `N`
   under continuous drift.

> **These findings hold only against IPPO at the conventional rate `5e-4`.**
> The evaluation controls (`scripts/run_robustness.py`,
> `scripts/run_noise_probe.py`, `scripts/run_review_analyses.py`) show that a
> single-rate IPPO with a swept learning rate matches or beats Nested-MARL,
> that the advantage under the conventional comparison is present without
> drift and does not depend on band separation, and that raising the noise does
> not restore an advantage. The chapter reports those results as its main
> finding. Do not cite points 1-4 above without them.

### Findings that may not reproduce

Stated in advance so they are not quietly dropped:

* **Switching cost.** The original reports a 16.6% reduction versus IPPO. The
  full ten-seed run here gives an 18.0% reduction against the conventional
  baseline that is *not* significant (p = 0.085), and no difference under
  matched tuning (`results/ROBUSTNESS.md`).
* **EMA anchor.** The anchor is performance-neutral on reward in this
  simulator: removing it changes first-40 reward by 0.01. Whether it matters
  under longer horizons or more aggressive slow rates is untested here.

---

## 3. Determinism

* Every run is a pure function of its `RunSpec`. Same spec → same numbers, on
  the same platform and library versions.
* All environment stochasticity flows through one `numpy.random.Generator`;
  network initialisation through one `torch.manual_seed`.
* `torch.set_num_threads(1)` in every worker, so results do **not** depend on
  the worker count. Verified by `test_training_is_reproducible`.
* Each run writes a JSON sidecar recording the full environment config, the
  method spec, parameter counts, wall-clock time, and Python/torch/NumPy
  versions.

Cross-platform bit-identity is *not* guaranteed: BLAS kernels differ between
CPU architectures. Conclusions are stable; last-digit values may not be.

---

## 4. Statistical practice and its limits

* Comparisons are **paired by seed**, then tested with a two-sided paired
  permutation test (exact enumeration for `n ≤ 20`).
* P-values are **Holm-corrected** across the family of baselines.
* **IQM with bootstrap CIs** is reported alongside the mean.
* Three evaluation windows are reported **together**.

**The p-value floor.** With `n` seeds the smallest attainable two-sided p-value
is `3/(2ⁿ+1)`:

| Seeds | Floor |
|---|---|
| 5 | 0.094 |
| 10 | 0.0029 |
| 20 | 2.9 × 10⁻⁶ |

**A five-seed ablation cannot produce a significant result at α = 0.05.** This
is a property of the test, not of the data. Ablations are therefore reported as
effect sizes and directions; only the ten-seed headline and severity
experiments support significance claims. `REPORT.md` flags any p-value sitting
at the floor.

---

## 5. Scope limitations

1. **Single simulator.** A controlled abstraction, not ns-3, OpenAirInterface,
   or an O-RAN testbed. It deliberately excludes physical-layer confounders to
   isolate the multi-timescale effect. Its numbers are not RAN measurements.
2. **Timescale abstraction.** The three bands are a stylised model. Real
   channel coherence is faster than the near-RT RIC loop; see the caution in
   `docs/METHOD.md` §1.
3. **Scale.** Up to `N = 10` agents and `S = 5` slices. Production-scale
   deployments remain untested here.
4. **Proxy metrics.** SLA satisfaction, switching cost and queue backlog are
   simulator constructs. Validating them against real O-RAN E2 telemetry is a
   concrete next step, not something this repository does.
5. **Hand-set partition and rates.** The depth-based partition and the rates
   are fixed a priori. `scripts/run_review_analyses.py` shows the partition
   matters and that a gradient-SNR rule does not recover it; learning it is
   open.
6. **No theory in this repository.** The convergence results live in the
   companion paper; nothing here proves anything.

---

## 6. Reproducing from scratch

```bash
git clone https://github.com/Abraheem13/nested-learning-6g-tutorial.git
cd nested-learning-6g-tutorial
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make test           # the full test suite, ~20 s
make full           # ~20 min on 4 cores
make report         # tables, figures, REPORT.md
```

To confirm your environment is sane before committing an hour:

```bash
make smoke          # ~1 min; results are meaningless by design
```

Wall-clock reference: about 7 s per 80-episode run at `N=3` on one Apple M-series
core, and 15–20 s on a cloud x86 core. Everything in the chapter is about 600
distinct runs; with four workers that is well under an hour.

---
