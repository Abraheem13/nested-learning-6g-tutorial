#!/usr/bin/env python3
"""Run the experiment suite.

Examples
--------
Verify everything works (about a minute)::

    python scripts/run_experiments.py --preset smoke

Reproduce the reported results on a laptop::

    python scripts/run_experiments.py --preset full --workers 4

Run a single experiment::

    python scripts/run_experiments.py --preset full --only headline --workers 4

Runs are cached: re-launching after an interruption resumes where it stopped.
Use ``--force`` to recompute.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nested6g.experiments import PRESETS, all_specs, estimate_runs  # noqa: E402
from nested6g.runner import RESULTS_ROOT, run_many, write_manifest  # noqa: E402

EXPERIMENTS = ("headline", "ablation", "severity", "scalability")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", default="full", choices=sorted(PRESETS),
                   help="statistical power vs wall-clock time (default: full)")
    p.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS),
                   help="run only these experiments")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="parallel worker processes (default: cores - 1)")
    p.add_argument("--results", type=Path, default=RESULTS_ROOT,
                   help="output directory (default: results/)")
    p.add_argument("--force", action="store_true", help="ignore cached runs")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without training")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    preset = PRESETS[args.preset]

    print("=" * 72)
    print(f"  Nested-MARL experiment suite   preset={preset.name}")
    print(f"  {preset.description}")
    print("=" * 72)

    counts = estimate_runs(preset)
    total = sum(counts[e] for e in args.only)
    for e in args.only:
        print(f"  {e:<14s} {counts[e]:4d} runs")
    print(f"  {'TOTAL':<14s} {total:4d} runs   episodes/run = {preset.episodes}"
          f"   workers = {args.workers}")
    print(f"  results -> {args.results.resolve()}")
    print("=" * 72, flush=True)

    if args.dry_run:
        return 0

    grids = all_specs(preset)
    t0 = time.perf_counter()
    for name in args.only:
        run_many(grids[name], root=args.results, workers=args.workers,
                 force=args.force, desc=name)
        write_manifest(name, grids[name], root=args.results)
    print(f"\nAll requested experiments finished in "
          f"{(time.perf_counter() - t0) / 60:.1f} min.")
    print("Next:  python scripts/make_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
