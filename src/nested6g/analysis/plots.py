"""Publication-quality figures.

Every figure is emitted as both PDF (vector, for LaTeX inclusion) and PNG (for
quick inspection).  The palette is colour-blind safe and every series is also
distinguished by line style or hatch, so the figures survive greyscale
printing -- a requirement for most book publishers.

.. note::
   The chapter this repository supports permits **at most three illustrations**.
   :func:`figure_case_study` is designed as a single three-panel figure that
   discharges the whole "sample-efficiency, policy-stability and scalability"
   promise in one illustration slot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..agents.methods import METHODS  # noqa: E402
from .aggregate import learning_curves, method_label  # noqa: E402

__all__ = [
    "set_style", "save", "figure_drift", "figure_learning_curves",
    "figure_severity", "figure_ablation", "figure_switching",
    "figure_per_slice", "figure_scalability", "figure_case_study",
]

#: Colour-blind safe (Okabe-Ito derived) palette, keyed by method.
PALETTE = {
    "ippo": "#D55E00",
    "mappo": "#009E73",
    "ewc_ippo": "#8C6BB1",
    "nested_marl": "#0072B2",
    "nested_no_ema": "#56B4E9",
    "nested_same_rate": "#E69F00",
    "nested_two_timescale": "#CC79A7",
}
DASHES = {
    "ippo": (3, 2),
    "mappo": (5, 2),
    "ewc_ippo": (1, 1.5),
    "nested_marl": (),
    "nested_no_ema": (4, 1, 1, 1),
    "nested_same_rate": (2, 1),
    "nested_two_timescale": (6, 2, 1, 2),
}


def set_style() -> None:
    """Apply a compact, serif, print-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
        }
    )


def save(fig: plt.Figure, path: Path | str) -> Path:
    """Write a figure as PDF and PNG; return the PDF path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return pdf


def _style_for(name: str) -> dict:
    return {
        "color": PALETTE.get(name, "#444444"),
        "dashes": DASHES.get(name, ()),
        "label": method_label(name),
    }


# --------------------------------------------------------------------------- #
#  Environment illustration
# --------------------------------------------------------------------------- #
def figure_drift(steps: int = 400, kappa: float = 1.5, seed: int = 0) -> plt.Figure:
    """Three stacked panels showing the fast, medium and slow drift bands.

    This is the natural candidate for the chapter's Figure 1: it makes the
    central premise -- that the environment moves at three separated rates --
    visible before any algorithm is introduced.
    """
    from ..env import EnvConfig, SliceEnv, SLICE_NAMES

    env = SliceEnv(EnvConfig(kappa=kappa), seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)

    chan, dem, reg = [], [], []
    for _ in range(steps):
        env.step(rng.integers(0, env.n_actions, size=env.cfg.n_agents))
        st = env.drift.state
        chan.append(st.channel[0, 0])
        dem.append(st.demand.copy())
        reg.append(st.regime)

    dem = np.asarray(dem)
    t = np.arange(steps)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.6), sharex=True)

    axes[0].plot(t, chan, color="#0072B2", lw=0.9)
    axes[0].set_ylabel("Channel gain")
    axes[0].set_title("Fast band: channel fading (every step)", loc="left")

    for s in range(dem.shape[1]):
        axes[1].step(
            t, dem[:, s], where="post", lw=1.1,
            label=SLICE_NAMES[s] if s < len(SLICE_NAMES) else f"slice {s}",
        )
    axes[1].set_ylabel("Demand mult.")
    axes[1].set_title(
        rf"Medium band: per-slice demand (every $\tau_{{med}}={env.cfg.tau_med}$ steps)",
        loc="left",
    )
    axes[1].legend(ncol=3, loc="upper right")

    axes[2].step(t, reg, where="post", color="#009E73", lw=1.3)
    axes[2].set_yticks(range(env.cfg.n_regimes))
    axes[2].set_yticklabels([f"regime {r}" for r in range(env.cfg.n_regimes)])
    axes[2].set_ylabel("SLA regime")
    axes[2].set_xlabel("Environment step")
    axes[2].set_title(
        rf"Slow band: SLA regime (every $\tau_{{slow}}={env.cfg.tau_slow}$ steps)",
        loc="left",
    )

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Training figures
# --------------------------------------------------------------------------- #
def figure_learning_curves(
    df: pd.DataFrame, metric: str = "reward", ylabel: str = "Mean episodic reward"
) -> plt.Figure:
    """Mean +/- 1 SD learning curves across seeds."""
    curves = learning_curves(df, metric=metric)
    order = [m for m in METHODS if m in curves]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for name in order:
        ep, mu, sd = curves[name]
        st = _style_for(name)
        ax.plot(ep, mu, **st)
        ax.fill_between(ep, mu - sd, mu + sd, color=st["color"], alpha=0.12, lw=0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


def figure_switching(df: pd.DataFrame) -> plt.Figure:
    """Policy churn (mean switching cost) over training. Lower is more stable."""
    return figure_learning_curves(
        df, metric="switching", ylabel=r"Mean switching cost $\|a_t-a_{t-1}\|_2$"
    )


def figure_severity(summary: pd.DataFrame, column: str = "reward_first40") -> plt.Figure:
    """Absolute performance and paired advantage across drift severities."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

    for name in ("ippo", "nested_marl"):
        sub = summary[summary["method"] == name]
        if sub.empty:
            continue
        g = sub.groupby("kappa")[column]
        k = np.asarray(sorted(g.groups))
        mu = g.mean().reindex(k).to_numpy()
        sd = g.std(ddof=1).reindex(k).to_numpy()
        st = _style_for(name)
        axes[0].errorbar(k, mu, yerr=sd, marker="o", ms=4, capsize=2.5, **st)
    axes[0].set_xlabel(r"Drift severity $\kappa$")
    axes[0].set_ylabel("Mean reward, first 40 ep.")
    axes[0].set_title("(a) Absolute performance", loc="left")
    axes[0].legend(loc="upper right")

    treat = summary[summary["method"] == "nested_marl"]
    base = summary[summary["method"] == "ippo"]
    ks, mus, los, his = [], [], [], []
    for k in sorted(treat["kappa"].unique()):
        a = treat[treat["kappa"] == k].set_index("seed")[column]
        b = base[base["kappa"] == k].set_index("seed")[column]
        shared = a.index.intersection(b.index)
        if len(shared) < 2:
            continue
        d = (a.loc[shared] - b.loc[shared]).to_numpy(dtype=float)
        ks.append(k)
        mus.append(d.mean())
        sem = d.std(ddof=1) / np.sqrt(d.size)
        los.append(d.mean() - 1.96 * sem)
        his.append(d.mean() + 1.96 * sem)

    ks = np.asarray(ks)
    axes[1].axhline(0.0, color="#888888", lw=0.8, ls=":")
    axes[1].plot(ks, mus, marker="s", ms=4, color=PALETTE["nested_marl"])
    axes[1].fill_between(ks, los, his, color=PALETTE["nested_marl"], alpha=0.18, lw=0)
    axes[1].set_xlabel(r"Drift severity $\kappa$")
    axes[1].set_ylabel("Paired advantage (Nested $-$ IPPO)")
    axes[1].set_title("(b) Paired advantage, 95% CI", loc="left")

    fig.tight_layout()
    return fig


def figure_ablation(summary: pd.DataFrame, column: str = "reward_last30") -> plt.Figure:
    """Bar chart isolating each design ingredient."""
    order = ["ippo", "nested_same_rate", "nested_two_timescale",
             "nested_no_ema", "nested_marl"]
    order = [m for m in order if m in set(summary["method"])]

    means, errs, labels, colors = [], [], [], []
    for name in order:
        v = summary.loc[summary["method"] == name, column].to_numpy(dtype=float)
        means.append(v.mean())
        errs.append(v.std(ddof=1) / np.sqrt(v.size) * 1.96 if v.size > 1 else 0.0)
        labels.append(method_label(name).replace(" (", "\n("))
        colors.append(PALETTE.get(name, "#444444"))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor="black", lw=0.5)
    if "ippo" in order:
        ax.axhline(means[order.index("ippo")], color="#555555", ls="--", lw=0.9,
                   label="IPPO baseline")
        ax.legend(loc="lower right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Mean reward, final 30 ep.")
    lo = min(means) - max(errs) - 2
    ax.set_ylim(max(0.0, lo), max(means) + max(errs) + 2)
    fig.tight_layout()
    return fig


def figure_per_slice(df: pd.DataFrame) -> plt.Figure:
    """Per-slice SLA satisfaction over training."""
    from ..env import SLICE_NAMES

    cols = sorted(c for c in df.columns if c.startswith("sla_slice_"))
    fig, axes = plt.subplots(1, len(cols), figsize=(6.6, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for j, col in enumerate(cols):
        for name in ("ippo", "nested_marl"):
            sub = df[df["method"] == name]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="episode", columns="seed", values=col)
            mu, sd = piv.mean(axis=1), piv.std(axis=1, ddof=0)
            st = _style_for(name)
            axes[j].plot(piv.index, mu, **st)
            axes[j].fill_between(piv.index, mu - sd, mu + sd,
                                 color=st["color"], alpha=0.12, lw=0)
        title = SLICE_NAMES[j] if j < len(SLICE_NAMES) else f"slice {j}"
        axes[j].set_title(title, loc="center")
        axes[j].set_xlabel("Episode")
    axes[0].set_ylabel("SLA satisfaction")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    return fig


def figure_scalability(summary: pd.DataFrame, column: str = "reward_last30") -> plt.Figure:
    """Final-phase reward as the agent and slice counts grow."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=False)
    for j, n_slices in enumerate(sorted(summary["n_slices"].unique())):
        ax = axes[j] if j < len(axes) else axes[-1]
        sub_s = summary[summary["n_slices"] == n_slices]
        for name in ("ippo", "mappo", "nested_marl"):
            sub = sub_s[sub_s["method"] == name]
            if sub.empty:
                continue
            g = sub.groupby("n_agents")[column]
            n = np.asarray(sorted(g.groups))
            mu = g.mean().reindex(n).to_numpy()
            sd = g.std(ddof=1).reindex(n).to_numpy()
            st = _style_for(name)
            ax.errorbar(n, mu, yerr=sd, marker="o", ms=4, capsize=2.5, **st)
        ax.set_xlabel("Number of agents $N$")
        ax.set_title(f"$S = {n_slices}$ slices", loc="left")
        ax.set_xticks(sorted(sub_s["n_agents"].unique()))
    axes[0].set_ylabel("Mean reward, final 30 ep.")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  The single combined chapter figure
# --------------------------------------------------------------------------- #
def figure_case_study(
    headline: pd.DataFrame,
    severity_summary: pd.DataFrame | None = None,
    scalability_summary: pd.DataFrame | None = None,
    ablation: pd.DataFrame | None = None,
) -> plt.Figure:
    """Three-panel case-study figure: learning, stability, scalability.

    Designed to occupy a single illustration slot while covering all three
    behaviours the chapter promises to demonstrate.

    Panels (a) and (b) prefer the *ablation* frame when it is available,
    because it additionally contains the ``same_rate`` control -- the series
    that carries the causal argument. Showing IPPO, the control and the full
    method together is what makes the figure an argument rather than a
    scoreboard.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    curve_src = ablation if ablation is not None and not ablation.empty else headline
    series = ("ippo", "nested_same_rate", "nested_marl")

    # (a) sample efficiency ------------------------------------------------
    curves = learning_curves(curve_src, metric="reward")
    for name in series:
        if name not in curves:
            continue
        ep, mu, sd = curves[name]
        st = _style_for(name)
        axes[0].plot(ep, mu, **st)
        axes[0].fill_between(ep, mu - sd, mu + sd, color=st["color"], alpha=0.12, lw=0)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Mean episodic reward")
    axes[0].set_title("(a) Sample efficiency", loc="left")
    axes[0].legend(loc="lower right", fontsize=6.5)

    # (b) policy stability -------------------------------------------------
    curves = learning_curves(curve_src, metric="switching")
    for name in series:
        if name not in curves:
            continue
        ep, mu, sd = curves[name]
        st = dict(_style_for(name))
        st.pop("label")
        axes[1].plot(ep, mu, **st)
        axes[1].fill_between(ep, mu - sd, mu + sd, color=st["color"], alpha=0.12, lw=0)
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Switching cost")
    axes[1].set_title("(b) Policy stability", loc="left")

    # (c) scalability or severity -----------------------------------------
    ax = axes[2]
    drew_c = False
    if scalability_summary is not None and not scalability_summary.empty:
        base = scalability_summary[
            scalability_summary["n_slices"] == scalability_summary["n_slices"].min()
        ]
        for name in ("ippo", "mappo", "nested_marl"):
            sub = base[base["method"] == name]
            if sub.empty:
                continue
            g = sub.groupby("n_agents")["reward_last30"]
            n = np.asarray(sorted(g.groups))
            ax.errorbar(n, g.mean().reindex(n), yerr=g.std(ddof=1).reindex(n),
                        marker="o", ms=3.5, capsize=2, **_style_for(name))
        ax.set_xlabel("Number of agents $N$")
        ax.set_ylabel("Reward, final 30 ep.")
        ax.set_title("(c) Scalability", loc="left")
        ax.set_xticks(sorted(base["n_agents"].unique()))
        drew_c = True
    elif severity_summary is not None and not severity_summary.empty:
        for name in ("ippo", "nested_marl"):
            sub = severity_summary[severity_summary["method"] == name]
            if sub.empty:
                continue
            g = sub.groupby("kappa")["reward_first40"]
            k = np.asarray(sorted(g.groups))
            ax.errorbar(k, g.mean().reindex(k), yerr=g.std(ddof=1).reindex(k),
                        marker="o", ms=3.5, capsize=2, **_style_for(name))
        ax.set_xlabel(r"Drift severity $\kappa$")
        ax.set_ylabel("Reward, first 40 ep.")
        ax.set_title("(c) Drift severity", loc="left")
        drew_c = True

    if drew_c:
        ax.legend(loc="best", fontsize=6.5)
    else:
        # No third-panel data available: hide the axes rather than shipping a
        # blank frame, and shrink the figure to two panels.
        ax.set_axis_off()

    fig.tight_layout()
    return fig
