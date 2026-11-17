"""Console entry points installed by ``pip install -e .``.

After installation the two scripts below are available on the PATH::

    nested6g-run --preset full --workers 4
    nested6g-report

They are thin wrappers around ``scripts/run_experiments.py`` and
``scripts/make_report.py`` so that the repository works identically whether it
is used as an installed package or as a checked-out source tree.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _script(name: str) -> Path:
    """Locate a script relative to the repository root, if present."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "scripts" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not find scripts/{name}. The console entry points require the "
        "source tree; run the scripts directly instead."
    )


def run_main(argv: list[str] | None = None) -> int:
    sys.argv = ["nested6g-run", *(argv if argv is not None else sys.argv[1:])]
    runpy.run_path(str(_script("run_experiments.py")), run_name="__main__")
    return 0


def report_main(argv: list[str] | None = None) -> int:
    sys.argv = ["nested6g-report", *(argv if argv is not None else sys.argv[1:])]
    runpy.run_path(str(_script("make_report.py")), run_name="__main__")
    return 0
