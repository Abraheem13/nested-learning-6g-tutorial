"""Turn raw per-episode logs into per-seed summaries and publication tables.

Three evaluation windows separate *sample efficiency* from *asymptotic*
behaviour, which matters because the two can point in opposite directions:

=============  =====================================  =========================
Window         Episodes                               Question it answers
=============  =====================================  =========================
``first40``    1 .. 40                                How fast does it learn?
``all80``      1 .. E                                 Overall performance.
``last20``     E-19 .. E                              Where does it end up?
``last30``     E-29 .. E                              Final-phase stability.
=============  =====================================  =========================

Reporting all three is a deliberate guard against the common failure mode of
quoting whichever window flatters the proposed method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..agents.methods import METHODS
from .statistics import (
    TestResult,
    bootstrap_ci,
    holm_bonferroni,
    interquartile_mean,
    probability_of_improvement,
    summarise,
)

__all__ = [
    "WINDOWS",
    "per_seed_summary",
    "aggregate_table",
    "comparison_table",
    "learning_curves",
    "method_label",
]

#: window name -> (start, stop) as Python slice semantics on the episode axis.
#: ``None`` means "open end".
WINDOWS: dict[str, tuple[int | None, int | None]] = {
    "first40": (0, 40),
    "all80": (None, None),
    "last20": (-20, None),
    "last30": (-30, None),
}

_METRICS = ("reward", "sla", "queue", "switching", "regret")


def method_label(name: str) -> str:
    spec = METHODS.get(name)
    return spec.label if spec else name


def _window(series: pd.Series, name: str) -> np.ndarray:
    lo, hi = WINDOWS[name]
    return series.to_numpy()[lo:hi]


def per_seed_summary(
    df: pd.DataFrame,
    group_keys: tuple[str, ...] = ("method", "seed"),
    metrics: tuple[str, ...] = _METRICS,
) -> pd.DataFrame:
    """Collapse per-episode logs into one row per run.

    Each metric is summarised over every window in :data:`WINDOWS`, producing
    columns such as ``reward_first40`` and ``sla_last20``.
    """
    keys = [k for k in group_keys if k in df.columns]
    rows = []
    for key_vals, sub in df.sort_values("episode").groupby(keys, sort=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row: dict[str, float | str | int] = dict(zip(keys, key_vals))
        for metric in metrics:
            if metric not in sub.columns:
                continue
            for win in WINDOWS:
                row[f"{metric}_{win}"] = float(np.mean(_window(sub[metric], win)))
        for col in sub.columns:
            if col.startswith("sla_slice_"):
                row[f"{col}_last30"] = float(np.mean(_window(sub[col], "last30")))
        row["episodes"] = int(sub["episode"].max() + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_table(
    summary: pd.DataFrame,
    column: str,
    group: str = "method",
    order: tuple[str, ...] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean, SD, IQM and a bootstrap CI of ``column`` for each method."""
    rng = np.random.default_rng(seed)
    rows = []
    names = order or tuple(summary[group].unique())
    for name in names:
        vals = summary.loc[summary[group] == name, column].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        lo, hi = bootstrap_ci(vals, np.mean, rng=rng)
        iqm_lo, iqm_hi = bootstrap_ci(vals, interquartile_mean, rng=rng)
        rows.append(
            {
                group: name,
                "label": method_label(name) if group == "method" else str(name),
                "n": int(vals.size),
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "ci_low": lo,
                "ci_high": hi,
                "iqm": interquartile_mean(vals),
                "iqm_ci_low": iqm_lo,
                "iqm_ci_high": iqm_hi,
            }
        )
    return pd.DataFrame(rows)


def comparison_table(
    summary: pd.DataFrame,
    column: str,
    treatment: str,
    baselines: tuple[str, ...],
    group: str = "method",
    seed: int = 0,
    holm: bool = True,
) -> pd.DataFrame:
    """Paired comparison of one method against several baselines.

    Runs are paired by seed.  P-values are Holm-corrected across the family of
    baselines, and a bootstrap probability-of-improvement is reported alongside
    the significance test.
    """
    treat = summary[summary[group] == treatment].set_index("seed")[column]
    results: list[TestResult] = []
    poi: dict[str, tuple[float, float, float]] = {}

    for base in baselines:
        if base == treatment:
            continue
        other = summary[summary[group] == base].set_index("seed")[column]
        shared = treat.index.intersection(other.index)
        if len(shared) < 2:
            continue
        a = treat.loc[shared].to_numpy(dtype=float)
        b = other.loc[shared].to_numpy(dtype=float)
        results.append(summarise(a, b, name=base, seed=seed))
        poi[base] = probability_of_improvement(
            a, b, rng=np.random.default_rng(seed)
        )

    if holm and results:
        corrected = holm_bonferroni({r.name: r.p_value for r in results})
        for r in results:
            r.p_holm = corrected[r.name]

    rows = []
    for r in results:
        d = r.as_dict()
        d["label"] = method_label(r.name)
        d["poi"], d["poi_low"], d["poi_high"] = poi[r.name]
        rows.append(d)
    return pd.DataFrame(rows)


def learning_curves(
    df: pd.DataFrame,
    metric: str = "reward",
    group: str = "method",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Mean +/- 1 SD learning curve per method.

    Returns ``{name: (episodes, mean, sd)}``.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, sub in df.groupby(group):
        pivot = sub.pivot_table(index="episode", columns="seed", values=metric)
        out[str(name)] = (
            pivot.index.to_numpy(),
            pivot.mean(axis=1).to_numpy(),
            pivot.std(axis=1, ddof=0).to_numpy(),
        )
    return out
