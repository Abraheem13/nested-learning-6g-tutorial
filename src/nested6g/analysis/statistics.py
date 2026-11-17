"""Statistical tools for honest RL comparison.

Deep RL results are notoriously seed-sensitive, and point estimates over a
handful of runs routinely reverse under reseeding.  This module implements the
tests used throughout the repository:

* **Paired permutation test** -- an exact-in-the-limit, distribution-free test
  for a paired difference.  With ``n`` seeds, exact enumeration and the
  ``(k + 1) / (m + 1)`` correction used below, the smallest attainable
  two-sided p-value is ``3 / (2^n + 1)``; for ``n = 10`` that is
  ``3 / 1025 ~ 0.0029``, so a reported ``p = 0.003`` means "the most extreme
  value the test can produce", not "p is exactly 0.003".
* **Holm-Bonferroni correction** -- controls the family-wise error rate when
  one method is compared against several baselines.
* **Interquartile mean (IQM)** with **stratified bootstrap confidence
  intervals** -- a robust aggregate that discards the top and bottom 25% of
  runs, far less sensitive to single lucky or unlucky seeds than the mean.
* **Probability of improvement** -- the fraction of paired comparisons in which
  one method beats another, with a bootstrap interval.

References
----------
Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018.
Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TestResult",
    "paired_permutation_test",
    "holm_bonferroni",
    "interquartile_mean",
    "bootstrap_ci",
    "probability_of_improvement",
    "cohens_d",
    "summarise",
]


@dataclass
class TestResult:
    """Outcome of a paired comparison between two methods."""

    name: str
    mean_a: float
    mean_b: float
    delta: float
    delta_pct: float
    p_value: float
    p_holm: float | None
    ci_low: float
    ci_high: float
    effect_size: float
    n: int

    @property
    def significant(self) -> bool:
        p = self.p_holm if self.p_holm is not None else self.p_value
        return p < 0.05

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
    alternative: str = "two-sided",
) -> float:
    """Two-sided paired permutation test on ``mean(a - b)``.

    Under the null hypothesis of exchangeability, the sign of each paired
    difference is equally likely to be positive or negative.  We enumerate
    (or sample) sign flips and count how often the permuted statistic is at
    least as extreme as the observed one.

    Returns the p-value, with the standard ``(k + 1) / (m + 1)`` correction
    so that a p-value of exactly zero is never reported.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same shape)")
    rng = rng or np.random.default_rng(0)

    diff = a - b
    n = diff.size
    observed = float(diff.mean())

    if n <= 20:  # exact enumeration of all 2^n sign assignments
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        stats = (signs * diff).mean(axis=1)
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
        stats = (signs * diff).mean(axis=1)

    if alternative == "two-sided":
        count = int(np.sum(np.abs(stats) >= abs(observed) - 1e-12))
    elif alternative == "greater":
        count = int(np.sum(stats >= observed - 1e-12))
    elif alternative == "less":
        count = int(np.sum(stats <= observed + 1e-12))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")

    return (count + 1) / (stats.size + 1)


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction for family-wise error rate."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # enforce monotonicity
        corrected[key] = running
    return corrected


def interquartile_mean(x: np.ndarray) -> float:
    """Mean of the middle 50% of the sample (robust to outlier seeds)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    core = x[lo:hi]
    return float(core.mean()) if core.size else float(x.mean())


def bootstrap_ci(
    x: np.ndarray,
    statistic=np.mean,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for an arbitrary statistic."""
    x = np.asarray(x, dtype=float).ravel()
    rng = rng or np.random.default_rng(0)
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_resamples, n))
    stats = np.asarray([statistic(x[i]) for i in idx])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def probability_of_improvement(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 1_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """P(a > b) over all run pairs, with a bootstrap interval.

    Returns ``(point_estimate, ci_low, ci_high)``.  A value of 0.5 means the
    two methods are indistinguishable; 1.0 means ``a`` wins every comparison.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    rng = rng or np.random.default_rng(0)

    def _poi(u: np.ndarray, v: np.ndarray) -> float:
        comp = u[:, None] > v[None, :]
        ties = u[:, None] == v[None, :]
        return float((comp.sum() + 0.5 * ties.sum()) / comp.size)

    point = _poi(a, b)
    stats = np.empty(n_resamples)
    for k in range(n_resamples):
        ia = rng.integers(0, a.size, a.size)
        ib = rng.integers(0, b.size, b.size)
        stats[k] = _poi(a[ia], b[ib])
    return point, float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d (mean difference over the SD of the differences)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf" if d.mean() else 0.0)


def summarise(
    a: np.ndarray,
    b: np.ndarray,
    name: str,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> TestResult:
    """Full paired comparison of ``a`` (treatment) against ``b`` (baseline)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = float(a.mean() - b.mean())
    lo, hi = bootstrap_ci(a - b, np.mean, rng=rng)
    return TestResult(
        name=name,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        delta=delta,
        delta_pct=float(100.0 * delta / abs(b.mean())) if b.mean() else float("nan"),
        p_value=paired_permutation_test(a, b, n_permutations, rng),
        p_holm=None,
        ci_low=lo,
        ci_high=hi,
        effect_size=cohens_d(a, b),
        n=int(a.size),
    )
