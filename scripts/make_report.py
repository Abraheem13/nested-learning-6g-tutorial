#!/usr/bin/env python3
"""Turn raw run logs into tables, figures and a written report.

Produces, under ``results/``:

* ``tables/*.csv``   -- machine-readable summaries
* ``tables/*.tex``   -- booktabs tables, ``\\input``-ready for the Springer template
* ``figures/*.pdf``  -- vector figures for LaTeX (plus PNG previews)
* ``REPORT.md``      -- a written summary of every finding, with the numbers
                        stated plainly, including the ones that do not favour
                        the proposed method

Every experiment is optional: whatever has been run is reported, and anything
missing is noted rather than silently skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nested6g.agents.methods import ABLATIONS, METHODS  # noqa: E402
from nested6g.analysis import aggregate as agg  # noqa: E402
from nested6g.analysis import plots  # noqa: E402
from nested6g.runner import RESULTS_ROOT, load_experiment  # noqa: E402

HEADLINE_ORDER = ("ippo", "mappo", "ewc_ippo", "nested_marl")
ABLATION_ORDER = ("ippo", "nested_same_rate", "nested_two_timescale",
                  "nested_no_ema", "nested_marl")


# --------------------------------------------------------------------------- #
#  LaTeX
# --------------------------------------------------------------------------- #
def to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_fmt: str = "%.3f",
    bold_max: str | None = None,
    bold_min: str | None = None,
) -> str:
    """Render a booktabs table, optionally bolding the best row of one column."""
    body = df.copy()
    for col in (bold_max, bold_min):
        if col and col in body.columns:
            vals = pd.to_numeric(body[col], errors="coerce")
            best = vals.idxmax() if col == bold_max else vals.idxmin()
            body[col] = [
                f"\\textbf{{{float_fmt % v}}}" if i == best else (float_fmt % v)
                for i, v in vals.items()
            ]
    latex = body.to_latex(
        index=False, escape=False, float_format=float_fmt,
        column_format="l" + "r" * (len(body.columns) - 1),
    )
    latex = latex.replace("\\toprule", "\\toprule").replace("_", r"\_")
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\footnotesize\n" + latex + "\\end{table}\n"
    )


def write_table(df: pd.DataFrame, stem: str, out: Path, **kw) -> None:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / f"{stem}.csv", index=False)
    (out / "tables" / f"{stem}.tex").write_text(to_latex(df, **kw))


def fmt(x: float, n: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def p_str(p: float, n_seeds: int) -> str:
    """Format a p-value, flagging when it is at the resolution floor."""
    floor = 3.0 / (2 ** n_seeds + 1)
    return f"{p:.4f}" + (" (floor)" if p <= floor * 1.01 else "")


# --------------------------------------------------------------------------- #
#  Experiments
# --------------------------------------------------------------------------- #
def do_headline(root: Path, out: Path, md: list[str]) -> pd.DataFrame | None:
    try:
        df = load_experiment("headline", root)
    except FileNotFoundError:
        md.append("## 1. Headline comparison\n\n_Not run._\n")
        return None

    summary = agg.per_seed_summary(df)
    n_seeds = summary.groupby("method")["seed"].nunique().max()
    order = tuple(m for m in HEADLINE_ORDER if m in set(summary["method"]))

    rows = []
    for m in order:
        s = summary[summary["method"] == m]
        rows.append({
            "Method": METHODS[m].label,
            "First-40": s["reward_first40"].mean(),
            "All-80": s["reward_all80"].mean(),
            "Last-20": s["reward_last20"].mean(),
            "SLA": s["sla_last30"].mean(),
            "Switching": s["switching_last30"].mean(),
            "Queue": s["queue_last30"].mean(),
        })
    table = pd.DataFrame(rows)
    write_table(
        table, "headline", out,
        caption=(f"Headline comparison at $\\kappa=1.5$ ($n={n_seeds}$ seeds). "
                 "Reward is averaged over the stated episode window; SLA, "
                 "switching cost and queue backlog over the final 30 episodes."),
        label="tab:headline", float_fmt="%.3f", bold_max="First-40",
    )

    comp = agg.comparison_table(
        summary, "reward_first40", "nested_marl",
        tuple(m for m in order if m != "nested_marl"),
    )
    write_table(
        comp[["label", "mean_a", "mean_b", "delta", "delta_pct",
              "p_value", "p_holm", "poi"]],
        "headline_significance", out,
        caption=("Paired comparison of Nested-MARL against each baseline on "
                 "first-40-episode reward. $p$-values are two-sided paired "
                 "permutation tests with Holm correction; PoI is the bootstrap "
                 "probability of improvement."),
        label="tab:headline-sig", float_fmt="%.4f",
    )

    plots.set_style()
    plots.save(plots.figure_learning_curves(df), out / "figures" / "learning_curves")
    plots.save(plots.figure_switching(df), out / "figures" / "switching_cost")
    if any(c.startswith("sla_slice_") for c in df.columns):
        plots.save(plots.figure_per_slice(df), out / "figures" / "per_slice_sla")

    # ---- narrative -------------------------------------------------------
    md.append(f"## 1. Headline comparison (kappa = 1.5, n = {n_seeds} seeds)\n")
    md.append(table.to_markdown(index=False, floatfmt=".3f") + "\n")
    best = table.loc[table["First-40"].idxmax(), "Method"]
    md.append(f"Best sample efficiency: **{best}**.\n")
    md.append("\nPaired tests against Nested-MARL (first-40 reward):\n")
    for _, r in comp.iterrows():
        verdict = "significant" if r["p_holm"] < 0.05 else "NOT significant"
        md.append(
            f"- vs **{r['label']}**: delta = {r['delta']:+.2f} "
            f"({r['delta_pct']:+.1f}%), p_holm = {p_str(r['p_holm'], n_seeds)} "
            f"-> {verdict}; P(improvement) = {r['poi']:.2f} "
            f"[{r['poi_low']:.2f}, {r['poi_high']:.2f}]"
        )
    md.append("")

    asym = agg.comparison_table(
        summary, "reward_last20", "nested_marl",
        tuple(m for m in order if m != "nested_marl"),
    )
    md.append("Asymptotic window (last-20 reward) -- reported so the "
              "sample-efficiency claim is not confused with an asymptotic one:\n")
    for _, r in asym.iterrows():
        md.append(f"- vs **{r['label']}**: delta = {r['delta']:+.2f}, "
                  f"p_holm = {p_str(r['p_holm'], n_seeds)}")
    md.append("")
    return df


def do_ablation(root: Path, out: Path, md: list[str]) -> pd.DataFrame | None:
    try:
        df = load_experiment("ablation", root)
    except FileNotFoundError:
        md.append("## 2. Ablation\n\n_Not run._\n")
        return None

    summary = agg.per_seed_summary(df)
    n_seeds = summary.groupby("method")["seed"].nunique().max()
    order = tuple(m for m in ABLATION_ORDER if m in set(summary["method"]))

    rows = []
    for m in order:
        s = summary[summary["method"] == m]
        rows.append({
            "Variant": METHODS[m].label,
            "First-40": s["reward_first40"].mean(),
            "All-80": s["reward_all80"].mean(),
            "Last-30": s["reward_last30"].mean(),
            "SLA": s["sla_last30"].mean(),
            "Switching": s["switching_last30"].mean(),
        })
    table = pd.DataFrame(rows)
    write_table(
        table, "ablation", out,
        caption=(f"Ablation at $\\kappa=1.5$ ($n={n_seeds}$ seeds). Each variant "
                 "removes exactly one ingredient of Nested-MARL."),
        label="tab:ablation", float_fmt="%.3f", bold_max="Last-30",
    )

    plots.set_style()
    plots.save(plots.figure_ablation(summary), out / "figures" / "ablation")

    md.append(f"## 2. Ablation (kappa = 1.5, n = {n_seeds} seeds)\n")
    md.append(table.to_markdown(index=False, floatfmt=".3f") + "\n")

    if "ippo" in order:
        base = summary[summary["method"] == "ippo"]["reward_last30"].mean()
        md.append(f"\nIPPO reference (last-30 reward): **{base:.2f}**\n")
        for m in ABLATIONS:
            if m not in order:
                continue
            v = summary[summary["method"] == m]["reward_last30"].mean()
            rel = "BELOW" if v < base else "above"
            md.append(f"- {METHODS[m].label}: {v:.2f} -> {rel} the IPPO baseline")
        md.append(
            "\nThe key control is **same rate (no timescale separation)**: "
            "it keeps three parameter groups, three optimisers and the EMA "
            "memory, and changes only the rates, so within this configuration "
            "it isolates the rates from architectural depth. These comparisons "
            "use the conventional baseline rate; `ROBUSTNESS.md` shows that a "
            "tuned single-rate IPPO matches or beats the full method.\n"
        )
    md.append("")
    return df


def do_severity(root: Path, out: Path, md: list[str]) -> pd.DataFrame | None:
    try:
        df = load_experiment("severity", root)
    except FileNotFoundError:
        md.append("## 3. Drift-severity sweep\n\n_Not run._\n")
        return None

    summary = agg.per_seed_summary(df, group_keys=("method", "kappa", "seed"))
    n_seeds = summary.groupby(["method", "kappa"])["seed"].nunique().max()

    rows = []
    for k in sorted(summary["kappa"].unique()):
        a = summary[(summary["method"] == "nested_marl") & (summary["kappa"] == k)]
        b = summary[(summary["method"] == "ippo") & (summary["kappa"] == k)]
        shared = np.intersect1d(a["seed"], b["seed"])
        av = a.set_index("seed").loc[shared, "reward_first40"].to_numpy()
        bv = b.set_index("seed").loc[shared, "reward_first40"].to_numpy()
        res = agg.summarise(av, bv, name=f"kappa={k}")
        rows.append({
            "kappa": k, "IPPO": bv.mean(), "Nested-MARL": av.mean(),
            "Delta": res.delta, "Delta %": res.delta_pct, "p": res.p_value,
        })
    table = pd.DataFrame(rows)
    write_table(
        table, "severity", out,
        caption=(f"Drift-severity sweep, mean reward over the first 40 episodes "
                 f"($n={n_seeds}$ seeds per cell). $p$ is a two-sided paired "
                 "permutation test."),
        label="tab:severity", float_fmt="%.3f",
    )

    plots.set_style()
    plots.save(plots.figure_severity(summary), out / "figures" / "severity_sweep")

    md.append(f"## 3. Drift-severity sweep (n = {n_seeds} seeds per cell)\n")
    md.append(table.to_markdown(index=False, floatfmt=".3f") + "\n")
    wins = int((table["Delta"] > 0).sum())
    md.append(f"\nNested-MARL leads at {wins} of {len(table)} severities. "
              f"Significant (p < 0.05) at "
              f"{int((table['p'] < 0.05).sum())} of {len(table)}.\n")
    md.append("")
    return summary


def do_scalability(root: Path, out: Path, md: list[str]) -> pd.DataFrame | None:
    try:
        df = load_experiment("scalability", root)
    except FileNotFoundError:
        md.append("## 4. Scalability\n\n_Not run._\n")
        return None

    summary = agg.per_seed_summary(
        df, group_keys=("method", "n_agents", "n_slices", "seed")
    )
    pivot = summary.pivot_table(
        index=["n_agents", "n_slices"], columns="method",
        values="reward_first40", aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={m: METHODS[m].label for m in METHODS
                                  if m in pivot.columns})
    write_table(
        pivot, "scalability", out,
        caption=("Scalability sweep at $\\kappa=1.5$: mean reward over the first "
                 "40 episodes as the agent count $N$ and slice count $S$ grow."),
        label="tab:scalability", float_fmt="%.2f",
    )

    plots.set_style()
    plots.save(plots.figure_scalability(summary), out / "figures" / "scalability")

    md.append("## 4. Scalability\n")
    md.append(pivot.to_markdown(index=False, floatfmt=".2f") + "\n")
    md.append("")
    return summary


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=RESULTS_ROOT)
    p.add_argument("--out", type=Path, default=None,
                   help="output root (default: same as --results)")
    args = p.parse_args(argv)
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)

    md: list[str] = [
        "# Experimental report\n",
        "Generated by `scripts/make_report.py`. Every number below is computed "
        "from the run logs in `results/raw/`; nothing is hand-entered.\n",
        "> **Reading the p-values.** With `n` seeds the smallest attainable "
        "two-sided permutation p-value is `3 / (2^n + 1)`: 0.0029 at n = 10, "
        "0.09 at n = 5. A p-value marked `(floor)` is at that resolution limit "
        "and means *as extreme as this test can report*, not *exactly this "
        "value*.\n",
    ]

    headline = do_headline(args.results, out, md)
    ablation = do_ablation(args.results, out, md)
    severity = do_severity(args.results, out, md)
    scalability = do_scalability(args.results, out, md)

    # Environment illustration + the combined chapter figure.
    plots.set_style()
    plots.save(plots.figure_drift(), out / "figures" / "drift_trajectory")
    if headline is not None:
        plots.save(
            plots.figure_case_study(headline, severity, scalability, ablation),
            out / "figures" / "case_study",
        )
        md.append(
            "## Figures\n\n"
            "`figures/case_study.pdf` is a single three-panel figure covering "
            "sample efficiency, policy stability and scalability -- built to fit "
            "one illustration slot.\n\n"
            "`figures/drift_trajectory.pdf` illustrates the three drift bands "
            "and needs no training results.\n"
        )

    (out / "REPORT.md").write_text("\n".join(md))
    print(f"Report written to {out / 'REPORT.md'}")
    print(f"Tables  -> {out / 'tables'}")
    print(f"Figures -> {out / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
