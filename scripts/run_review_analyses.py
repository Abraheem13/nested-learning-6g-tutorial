#!/usr/bin/env python3
"""Analyses requested in the review of the book chapter.

Two questions were raised by the reviewers. Both are answered by experiment
rather than argument, and both answers are written out exactly as measured.

**A. Partition design.** The three rate groups are assigned by layer depth, by
hand. Does that assignment matter, and can it be chosen automatically? Each
alternative below keeps the rates, the group sizes, the memory anchor and the
parameter count identical to the depth partition, and changes *only which
tensors go in which group*:

* ``inverted``  -- depth reversed: input layers fast, output layers slow;
* ``random``    -- tensors shuffled into groups of the same sizes;
* ``grad_snr``  -- data-driven: after a 5-episode warm-up at the medium rate,
  tensors are ranked by gradient signal-to-noise ratio and the highest-SNR
  third is made fast, the lowest-SNR third slow (see ``learner.py``).

Each is run at the reference rate triple and at the best triple from the
matched-budget control, and compared with the depth partition at the same
rates (paired by seed, Holm-corrected across the three alternatives).

**B. Drift severity and band separation.** The method is motivated by
*hierarchical* drift: bands separated in time. Two sweeps test whether its
advantage tracks that structure.

* Severity: kappa in {0, 0.5, 1, 1.5, 2} at the default separation
  (tau_slow = 50, tau_med = 15). All of these runs exist from earlier
  experiments and are reused, not retrained.
* Separation: at kappa = 1.5 the regime period is collapsed to tau_slow = 16,
  just above tau_med = 15, so that the slow and medium bands coincide, and
  widened to tau_slow = 150. Collapsing the bands necessarily also makes regime
  changes three times more frequent; the two cannot be separated in this
  simulator, and the chapter says so.

Two comparisons are made in every cell, on the same seeds: the conventional one
(reference nested configuration vs IPPO at 5e-4) and the tuned one (best of five
nested configurations vs best of five single rates, i.e. IPPO at 5e-3). The
dependence of the advantage on separation is tested as a paired
difference-in-differences across seeds.

Usage
-----
    python scripts/run_review_analyses.py --workers 8

Runs already in ``results/raw`` are reused (the depth-partition runs and the
``tau_slow = 50`` cells all exist from earlier experiments), so only the new
cells are trained. Writes::

    results/REVIEW.md               verdicts in plain language
    results/review_stats.json       every number quoted in the chapter
    results/review_partition.csv    per-seed summaries, analysis A
    results/review_scope.csv        per-seed summaries, analysis B

The chapter's tables and figure are built from ``review_stats.json`` by
``scripts/make_chapter_assets.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nested6g.agents.methods import (PARTITION_DEPTH_METHOD,  # noqa: E402
                                     PARTITION_KINDS, PARTITION_METHODS,
                                     PARTITION_TRIPLES)
from nested6g.analysis import aggregate as agg  # noqa: E402
from nested6g.analysis.statistics import holm_bonferroni, summarise  # noqa: E402
from nested6g.experiments import SEEDS  # noqa: E402
from nested6g.runner import (RESULTS_ROOT, RunSpec, load_experiment,  # noqa: E402
                             run_many, write_manifest)

KAPPA = 1.5
KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0)
TAU_HIER, TAU_COLLAPSED, TAU_WIDE = 50, 16, 150
TAUS = (TAU_COLLAPSED, TAU_HIER, TAU_WIDE)
SCOPE_PAIRS = {
    "conventional": ("nested_marl", "ippo"),
    "tuned": ("nested_cfg_high", "ippo_lr0p005"),
}
METRICS = ("reward_first40", "reward_last20")


# --------------------------------------------------------------------------- #
#  Grids
# --------------------------------------------------------------------------- #
def partition_specs(episodes: int, n_seeds: int) -> list[RunSpec]:
    out = []
    for t in PARTITION_TRIPLES:
        methods = [PARTITION_DEPTH_METHOD[t]] + [PARTITION_METHODS[(t, k)]
                                                 for k in PARTITION_KINDS]
        out += [RunSpec(experiment="review_partition", method=m, seed=s,
                        episodes=episodes, kappa=KAPPA)
                for m in methods for s in SEEDS[:n_seeds]]
    return out


def scope_specs(episodes: int, n_seeds: int) -> list[RunSpec]:
    """Severity sweep at the default band separation (every run already cached
    by earlier experiments) plus two new band separations at kappa = 1.5."""
    methods = sorted({m for pair in SCOPE_PAIRS.values() for m in pair})
    cells = [(k, TAU_HIER) for k in KAPPAS] + [(KAPPA, t) for t in TAUS if t != TAU_HIER]
    return [RunSpec(experiment="review_scope", method=m, seed=s,
                    episodes=episodes, kappa=k, tau_slow=t)
            for k, t in cells for m in methods for s in SEEDS[:n_seeds]]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _summaries(experiment: str, root: Path) -> pd.DataFrame:
    df = load_experiment(experiment, root)
    if "tau_slow" not in df.columns:
        df["tau_slow"] = TAU_HIER
    df["tau_slow"] = df["tau_slow"].fillna(TAU_HIER).astype(int)
    return agg.per_seed_summary(df, group_keys=("method", "tau_slow", "seed"))


def _vals(s: pd.DataFrame, method: str, col: str, tau: int = TAU_HIER) -> pd.Series:
    v = s[(s["method"] == method) & (s["tau_slow"] == tau)]
    return v.set_index("seed")[col]


def _paired(a: pd.Series, b: pd.Series, name: str):
    shared = a.index.intersection(b.index)
    return summarise(a.loc[shared].to_numpy(), b.loc[shared].to_numpy(), name)


def _agreement(root: Path, method: str, n_seeds: int, episodes: int) -> dict:
    """How often the data-driven partition puts a tensor where depth puts it."""
    same = total = 0
    fast_counts: dict[str, int] = {}
    for s in SEEDS[:n_seeds]:
        js = RunSpec(experiment="x", method=method, seed=s,
                     episodes=episodes, kappa=KAPPA).paths(root)[1]
        rep = json.loads(js.read_text()).get("partition") or []
        for agent in rep:
            for name, g in agent["groups"].items():
                depth = {"0": 0, "2": 1, "4": 2}[name.split(".")[1]]
                same += int(g == depth)
                total += 1
                if g == 2:
                    fast_counts[name] = fast_counts.get(name, 0) + 1
    return {"agreement": same / total if total else float("nan"),
            "n_assignments": total, "fast_counts": fast_counts}


# --------------------------------------------------------------------------- #
#  Analysis A
# --------------------------------------------------------------------------- #
def analyse_partition(root: Path, n_seeds: int, episodes: int,
                      stats: dict, md: list[str]) -> None:
    s = _summaries("review_partition", root)
    s.to_csv(root / "review_partition.csv", index=False)
    n = int(s.groupby("method")["seed"].nunique().min())
    A: dict = {}
    for t in PARTITION_TRIPLES:
        depth = PARTITION_DEPTH_METHOD[t]
        A[t] = {"depth": {m: float(_vals(s, depth, m).mean()) for m in METRICS}}
        for col in METRICS:
            res = {k: _paired(_vals(s, PARTITION_METHODS[(t, k)], col),
                              _vals(s, depth, col), k) for k in PARTITION_KINDS}
            holm = holm_bonferroni({k: r.p_value for k, r in res.items()})
            for k, r in res.items():
                A[t].setdefault(k, {})
                A[t][k][col] = float(r.mean_a)
                A[t][k][f"{col}_delta"] = float(r.delta)
                A[t][k][f"{col}_p_holm"] = float(holm[k])
        A[t]["grad_snr"].update(
            _agreement(root, PARTITION_METHODS[(t, "grad_snr")], n_seeds, episodes))
    stats["partition"] = {"n_seeds": n, **A}

    # ---- plain-language report -------------------------------------------
    md.append(f"## A. Partition design (n = {n} seeds, kappa = {KAPPA})\n")
    for t, lrs in PARTITION_TRIPLES.items():
        md.append(f"### {t} rates {lrs}\n")
        md.append(f"- depth (hand-designed): first-40 = "
                  f"{A[t]['depth']['reward_first40']:.2f}, last-20 = "
                  f"{A[t]['depth']['reward_last20']:.2f}")
        for k in PARTITION_KINDS:
            v = A[t][k]
            md.append(
                f"- {k}: first-40 = {v['reward_first40']:.2f} "
                f"(delta {v['reward_first40_delta']:+.2f}, p_holm "
                f"{v['reward_first40_p_holm']:.4f}); last-20 = "
                f"{v['reward_last20']:.2f} (delta {v['reward_last20_delta']:+.2f}, "
                f"p_holm {v['reward_last20_p_holm']:.4f})")
        g = A[t]["grad_snr"]
        md.append(f"- grad_snr agrees with the depth assignment on "
                  f"{100 * g['agreement']:.1f}% of {g['n_assignments']} tensor "
                  f"assignments (chance for an equal-size random partition: 33.3%).")
        md.append("")
    md.append("")


# --------------------------------------------------------------------------- #
#  Analysis B
# --------------------------------------------------------------------------- #
def _cell(s: pd.DataFrame, name: str, kappa: float, tau: int, col: str):
    a, b = SCOPE_PAIRS[name]
    sub = s[(s["kappa"] == kappa) & (s["tau_slow"] == tau)]
    va = sub[sub["method"] == a].set_index("seed")[col]
    vb = sub[sub["method"] == b].set_index("seed")[col]
    shared = va.index.intersection(vb.index)
    return va.loc[shared], vb.loc[shared]


def analyse_scope(root: Path, stats: dict, md: list[str]) -> None:
    df = load_experiment("review_scope", root)
    if "tau_slow" not in df.columns:
        df["tau_slow"] = TAU_HIER
    df["tau_slow"] = df["tau_slow"].fillna(TAU_HIER).astype(int)
    s = agg.per_seed_summary(df, group_keys=("method", "kappa", "tau_slow", "seed"))
    s.to_csv(root / "review_scope.csv", index=False)
    n = int(s.groupby(["method", "kappa", "tau_slow"])["seed"].nunique().min())
    col = "reward_first40"

    cells = [(k, TAU_HIER) for k in KAPPAS] + [(KAPPA, t) for t in TAUS if t != TAU_HIER]
    B: dict = {name: {} for name in SCOPE_PAIRS}
    diffs: dict = {name: {} for name in SCOPE_PAIRS}
    for name in SCOPE_PAIRS:
        for kappa, tau in cells:
            va, vb = _cell(s, name, kappa, tau, col)
            r = summarise(va.to_numpy(), vb.to_numpy(), f"{name}_{kappa}_{tau}")
            B[name][f"k{kappa:g}_ts{tau}"] = {
                "kappa": kappa, "tau_slow": tau, "nested": float(r.mean_a),
                "baseline": float(r.mean_b), "delta": float(r.delta),
                "delta_pct": float(r.delta_pct), "p": float(r.p_value),
                "ci_low": float(r.ci_low), "ci_high": float(r.ci_high)}
            diffs[name][(kappa, tau)] = va - vb
        # Paired difference-in-differences: does the advantage depend on band
        # separation? Same seeds, same kappa, only tau_slow differs.
        for tau in TAUS:
            if tau == TAU_HIER:
                continue
            d = summarise(diffs[name][(KAPPA, TAU_HIER)].to_numpy(),
                          diffs[name][(KAPPA, tau)].loc[
                              diffs[name][(KAPPA, TAU_HIER)].index].to_numpy(),
                          f"{name}_did_{tau}")
            B[name][f"did_ts{tau}"] = {"delta": float(d.delta), "p": float(d.p_value)}
    stats["scope"] = {"n_seeds": n, "taus": list(TAUS), "kappas": list(KAPPAS), **B}

    md.append(f"## B. Drift severity and band separation (n = {n} seeds)\n")
    md.append(f"tau_med = 15 throughout. tau_slow = {TAU_HIER} is the default; "
              f"{TAU_COLLAPSED} collapses the slow band onto the medium one; "
              f"{TAU_WIDE} widens the separation.\n")
    for name, (a, b) in SCOPE_PAIRS.items():
        md.append(f"### {name}: {a} vs {b}\n")
        for kappa, tau in cells:
            v = B[name][f"k{kappa:g}_ts{tau}"]
            md.append(f"- kappa {kappa:g}, tau_slow {tau}: nested {v['nested']:.2f} vs "
                      f"baseline {v['baseline']:.2f}, delta {v['delta']:+.2f} "
                      f"({v['delta_pct']:+.1f}%), p {v['p']:.4f}")
        for tau in TAUS:
            if tau != TAU_HIER:
                v = B[name][f"did_ts{tau}"]
                md.append(f"- advantage at tau_slow {TAU_HIER} minus at {tau}: "
                          f"{v['delta']:+.2f} (paired p {v['p']:.4f})")
        md.append("")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--results", type=Path, default=RESULTS_ROOT)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)

    grids = {"review_partition": partition_specs(args.episodes, args.seeds),
             "review_scope": scope_specs(args.episodes, args.seeds)}
    if not args.report_only:
        print("=" * 68)
        print(f"  Review analyses: {sum(len(g) for g in grids.values())} runs "
              f"({args.seeds} seeds, {args.episodes} episodes); cached runs reused")
        print("=" * 68, flush=True)
        for name, g in grids.items():
            run_many(g, root=args.results, workers=args.workers, desc=name)
            write_manifest(name, g, root=args.results)

    stats: dict = {"seeds": args.seeds, "episodes": args.episodes}
    md = ["# Review analyses\n",
          "Requested by the chapter reviewers. Numbers are as measured; see "
          "scripts/run_review_analyses.py for the design of each analysis.\n"]
    analyse_partition(args.results, args.seeds, args.episodes, stats, md)
    analyse_scope(args.results, stats, md)

    (args.results / "review_stats.json").write_text(json.dumps(stats, indent=2))
    (args.results / "REVIEW.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"Written: {args.results / 'REVIEW.md'}, review_stats.json, "
          f"review_partition.csv, review_scope.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
