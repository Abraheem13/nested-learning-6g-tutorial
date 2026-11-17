"""Experiment orchestration: reproducible, resumable, parallel.

Design goals
------------
1. **Every run is a pure function of its specification.**  A :class:`RunSpec`
   fully determines the result, so a run can be re-executed on any machine and
   produce byte-identical output.
2. **Resumable.**  Completed runs are written to disk and skipped on re-launch,
   so a long sweep survives an interrupted laptop.
3. **Embarrassingly parallel.**  Runs are independent; the pool size is the
   only thing that changes wall-clock time, never the results.
4. **Self-describing output.**  Each run writes a CSV of per-episode metrics
   plus a JSON sidecar recording the exact configuration, library versions and
   runtime, so results can be audited months later.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["RunSpec", "run_one", "run_many", "load_experiment",
           "write_manifest", "RESULTS_ROOT"]

RESULTS_ROOT = Path(os.environ.get("NESTED6G_RESULTS", "results"))


@dataclass(frozen=True)
class RunSpec:
    """Complete, hashable specification of a single training run."""

    experiment: str
    method: str
    seed: int
    episodes: int = 80
    kappa: float = 1.5
    n_agents: int = 3
    n_slices: int = 3
    horizon: int = 150
    tag: str = ""
    #: Regime period. Appears in the slug only when it differs from the default,
    #: so every run cached before this field existed keeps its cache entry.
    tau_slow: int = 50
    #: Standard deviation of both the channel noise and the demand jitter.
    #: Appears in the slug only when it differs from the default (0.10).
    noise: float = 0.10

    def slug(self) -> str:
        """Content-addressed name.

        Deliberately excludes ``experiment``: two experiments that request an
        identical run share the cached result instead of recomputing it.  For
        example the ablation study and the headline comparison both need IPPO
        at ``kappa=1.5``, and the severity sweep needs the same runs again at
        its middle severity -- so the suite computes each exactly once.
        """
        tag = f"__{self.tag}" if self.tag else ""
        if self.tau_slow != 50:
            tag += f"__ts{self.tau_slow}"
        if self.noise != 0.10:
            tag += f"__nz{self.noise:g}"
        return (
            f"{self.method}{tag}__k{self.kappa:g}__N{self.n_agents}"
            f"__S{self.n_slices}__T{self.horizon}__E{self.episodes}"
            f"__seed{self.seed}"
        )

    def paths(self, root: Path = RESULTS_ROOT) -> tuple[Path, Path]:
        d = root / "raw"
        return d / f"{self.slug()}.csv", d / f"{self.slug()}.json"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Single run
# --------------------------------------------------------------------------- #
def run_one(spec: RunSpec, root: Path = RESULTS_ROOT, force: bool = False,
            progress: bool = False) -> pd.DataFrame:
    """Execute one training run, writing results to disk.

    If the output already exists and ``force`` is False, the cached result is
    loaded and returned without recomputation.
    """
    csv_path, json_path = spec.paths(root)
    if csv_path.exists() and json_path.exists() and not force:
        return pd.read_csv(csv_path)

    # Imports are local so that worker processes in a Pool start cheaply and
    # so that `import nested6g.runner` does not pull in torch.
    import torch

    from .agents import MultiAgentLearner, PPOConfig, get_method
    from .env import EnvConfig, SliceEnv

    torch.set_num_threads(1)          # one thread per worker: pools scale linearly

    env_cfg = EnvConfig(
        n_agents=spec.n_agents,
        n_slices=spec.n_slices,
        horizon=spec.horizon,
        kappa=spec.kappa,
        slice_efficiency=_pad(EnvConfig.slice_efficiency, spec.n_slices, 1.0),
        base_load=_pad(EnvConfig.base_load, spec.n_slices, 30.0),
        demand_sensitivity=_pad(EnvConfig.demand_sensitivity, spec.n_slices, 1.0),
        n_regimes=min(3, spec.n_slices) if spec.n_slices >= 3 else spec.n_slices,
        tau_slow=spec.tau_slow,
        channel_noise_std=spec.noise,
        demand_noise_std=spec.noise,
    )
    env = SliceEnv(env_cfg, seed=spec.seed)
    spec_obj = get_method(spec.method)
    learner = MultiAgentLearner(
        env, spec_obj, PPOConfig(discount=env_cfg.discount), seed=spec.seed
    )

    t0 = time.perf_counter()
    history = learner.train(spec.episodes, progress=progress)
    elapsed = time.perf_counter() - t0

    frame = pd.DataFrame([h.as_dict() for h in history])
    for key, value in spec.as_dict().items():
        frame[key] = value

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "spec": spec.as_dict(),
                "method": {
                    "label": spec_obj.label,
                    "learning_rates": list(spec_obj.learning_rates),
                    "ema_lambda": spec_obj.ema_lambda,
                    "stabilise_beta": spec_obj.stabilise_beta,
                    "centralised_critic": spec_obj.centralised_critic,
                    "ewc_lambda": spec_obj.ewc_lambda,
                },
                "env": env_cfg.to_dict(),
                "params_per_agent": learner.n_parameters(),
                "optimisers_per_agent": learner.n_optimisers_per_agent(),
                "partition": getattr(learner, "partition_report", None),
                "elapsed_seconds": round(elapsed, 3),
                "platform": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "machine": platform.machine(),
                    "system": platform.system(),
                },
            },
            indent=2,
            default=str,
        )
    )
    return frame


def _pad(reference: Sequence[float], n: int, fill: float) -> tuple[float, ...]:
    """Extend or truncate a per-slice tuple to length ``n``."""
    ref = list(reference)
    if len(ref) >= n:
        return tuple(ref[:n])
    return tuple(ref + [fill] * (n - len(ref)))


# --------------------------------------------------------------------------- #
#  Many runs
# --------------------------------------------------------------------------- #
def _worker(args):
    spec, root, force = args
    try:
        run_one(spec, Path(root), force=force)
        return (spec.slug(), None)
    except Exception:                              # pragma: no cover
        import traceback
        return (spec.slug(), traceback.format_exc())


def run_many(
    specs: Iterable[RunSpec],
    root: Path = RESULTS_ROOT,
    workers: int = 1,
    force: bool = False,
    desc: str = "runs",
) -> None:
    """Execute a collection of runs, in parallel if requested."""
    specs = list(specs)
    todo = [s for s in specs if force or not s.paths(root)[0].exists()]
    cached = len(specs) - len(todo)
    print(f"[{desc}] {len(specs)} runs  ({cached} cached, {len(todo)} to run, "
          f"{workers} worker{'s' if workers != 1 else ''})", flush=True)
    if not todo:
        return

    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []

    if workers <= 1:
        for i, spec in enumerate(todo, 1):
            slug, err = _worker((spec, str(root), force))
            if err:
                failures.append((slug, err))
            _tick(i, len(todo), t0)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        payload = [(s, str(root), force) for s in todo]
        with ctx.Pool(processes=workers) as pool:
            for i, (slug, err) in enumerate(pool.imap_unordered(_worker, payload), 1):
                if err:
                    failures.append((slug, err))
                _tick(i, len(todo), t0)

    print()
    if failures:
        print(f"[{desc}] {len(failures)} FAILED:", flush=True)
        for slug, err in failures[:3]:
            print(f"  --- {slug} ---\n{err}", flush=True)
        raise RuntimeError(f"{len(failures)} run(s) failed")
    print(f"[{desc}] done in {time.perf_counter() - t0:.1f}s", flush=True)


def write_manifest(experiment: str, specs: Iterable[RunSpec],
                   root: Path = RESULTS_ROOT) -> Path:
    """Record which runs constitute an experiment.

    Because run files are content-addressed and shared between experiments, the
    manifest is what tells :func:`load_experiment` which of them to load.
    """
    specs = list(specs)
    d = root / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{experiment}.json"
    path.write_text(json.dumps(
        {"experiment": experiment,
         "n_runs": len(specs),
         "runs": [s.as_dict() | {"slug": s.slug()} for s in specs]},
        indent=2,
    ))
    return path


def _tick(i: int, total: int, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    eta = elapsed / i * (total - i)
    bar = "#" * int(30 * i / total)
    print(f"\r  [{bar:<30}] {i}/{total}  elapsed {elapsed:6.1f}s  eta {eta:6.1f}s",
          end="", flush=True)


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_experiment(experiment: str, root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Concatenate every completed run of an experiment into one long frame."""
    manifest = root / "manifests" / f"{experiment}.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no manifest for {experiment!r} at {manifest}. "
            "Run scripts/run_experiments.py first."
        )
    entries = json.loads(manifest.read_text())["runs"]
    frames = []
    for e in entries:
        csv = root / "raw" / f"{e['slug']}.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(
            f"manifest for {experiment!r} lists {len(entries)} runs but none "
            f"are present in {root / 'raw'}."
        )
    return pd.concat(frames, ignore_index=True)
