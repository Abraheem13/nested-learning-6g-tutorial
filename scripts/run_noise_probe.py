#!/usr/bin/env python3
"""Control 4 - the precondition probe.

The tracking-noise trade-off has a floor only where a large step size is
genuinely costly. If the benchmark is too smooth for that floor to bind, then
raising the environment noise should make timescale separation pay. That is a
prediction the theory itself makes, so this script tests it.

Both noise sources -- the per-step channel noise and the per-refresh demand
jitter, whose defaults are both 0.10 -- are scaled together by 1x, 3x and 5x.
The 1x row is therefore the default environment, and its runs are the cached
runs of the other experiments (same content-addressed cache entries), so the
first row of this table must agree exactly with the baseline sweep.

The comparison set is the two strongest single-rate baselines from Control 1
against the reference nested configuration.

Correction note (review revision): an earlier version of this script scaled the
demand jitter as 1.5x the channel noise, so its "default" row was not the
default environment. This version fixes that and routes every run through the
shared runner so results are cached, resumable and recorded in a manifest.

Usage
-----
    python scripts/run_noise_probe.py --workers 4

Writes ``results/noise_probe.csv`` and ``results/NOISE_PROBE.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nested6g.agents.methods import METHODS  # noqa: E402
from nested6g.analysis import aggregate as agg  # noqa: E402
from nested6g.analysis.statistics import summarise  # noqa: E402
from nested6g.experiments import SEEDS  # noqa: E402
from nested6g.runner import (RESULTS_ROOT, RunSpec, load_experiment,  # noqa: E402
                             run_many, write_manifest)

#: Noise levels: the default, and three- and fivefold that value.
NOISE_LEVELS: tuple[float, ...] = (0.10, 0.30, 0.50)

#: The comparison set: the two strongest single-rate baselines from Control 1,
#: against the reference nested configuration.
PROBE_METHODS: tuple[str, ...] = ("ippo_lr0p005", "ippo_lr0p002", "nested_marl")


def specs(episodes: int, n_seeds: int) -> list[RunSpec]:
    return [RunSpec(experiment="noise_probe", method=m, seed=s,
                    episodes=episodes, kappa=1.5, noise=n)
            for n in NOISE_LEVELS for m in PROBE_METHODS for s in SEEDS[:n_seeds]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--results", type=Path, default=RESULTS_ROOT)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)

    grid = specs(args.episodes, args.seeds)
    if not args.report_only:
        print("=" * 66)
        print(f"  Control 4: precondition probe   {len(grid)} runs (cached reused)")
        print("=" * 66, flush=True)
        run_many(grid, root=args.results, workers=args.workers, desc="noise_probe")
        write_manifest("noise_probe", grid, root=args.results)

    df = load_experiment("noise_probe", args.results)
    if "noise" not in df.columns:
        df["noise"] = 0.10
    df["noise"] = df["noise"].fillna(0.10).round(4)
    s = agg.per_seed_summary(df, group_keys=("method", "noise", "seed"))
    s.to_csv(args.results / "noise_probe.csv", index=False)

    lines = [
        "# Control 4: precondition probe\n",
        f"Channel noise and demand jitter scaled together over {list(NOISE_LEVELS)} "
        f"(default 0.10); n = {args.seeds} seeds, {args.episodes} episodes. "
        "Cells are first-40 / last-20 mean reward.\n",
        "| Noise | " + " | ".join(METHODS[m].label for m in PROBE_METHODS) + " |",
        "|---|" + "---|" * len(PROBE_METHODS),
    ]
    best_gap = {}
    for n in NOISE_LEVELS:
        cells = []
        for m in PROBE_METHODS:
            v = s[(s["noise"] == round(n, 4)) & (s["method"] == m)]
            cells.append(f"{v['reward_first40'].mean():.2f} / "
                         f"{v['reward_last20'].mean():.2f}")
        tag = " (default)" if n == 0.10 else ""
        lines.append(f"| {n:.2f}{tag} | " + " | ".join(cells) + " |")
        nest = s[(s["noise"] == round(n, 4)) & (s["method"] == "nested_marl")]
        best = s[(s["noise"] == round(n, 4)) & (s["method"] == "ippo_lr0p005")]
        r = summarise(nest.set_index("seed")["reward_first40"].sort_index().to_numpy(),
                      best.set_index("seed")["reward_first40"].sort_index().to_numpy(),
                      f"noise{n}")
        best_gap[n] = r
    lines.append("")
    lines.append("Nested minus IPPO (alpha = 5e-3), first-40 reward:\n")
    for n, r in best_gap.items():
        lines.append(f"- noise {n:.2f}: delta {r.delta:+.2f} ({r.delta_pct:+.1f}%), "
                     f"p {r.p_value:.4f}")
    lines.append("")
    flips = sum(r.delta > 0 for r in best_gap.values())
    if flips == 0:
        lines.append("> **Result: the precondition is not restored.** The tuned "
                     "single-rate baseline leads at every noise level.\n")
    else:
        lines.append(f"> **Result: the ordering flips at {flips} of "
                     f"{len(NOISE_LEVELS)} noise levels.**\n")

    out = args.results / "NOISE_PROBE.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWritten to {out} and {args.results / 'noise_probe.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
