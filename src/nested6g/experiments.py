"""Declarative definitions of the four experiments.

Keeping the run grids in one place means the scripts in ``scripts/`` are thin
wrappers, and means the *same* seeds are reused across experiments wherever the
configuration coincides -- so the runner's cache is hit instead of recomputing
identical work.  For example the ablation study reuses the first five seeds of
the headline comparison for IPPO and Nested-MARL.

Experiments
-----------
``headline``      All methods at the reference severity. Answers: does it work?
``ablation``      Each ingredient removed in turn. Answers: *why* does it work?
``severity``      A sweep over drift severity. Answers: when does it work?
``scalability``   Agent and slice counts grown. Answers: does it keep working?
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents.methods import ABLATIONS, BASELINES, PROPOSED
from .runner import RunSpec

__all__ = [
    "SEEDS", "SEVERITIES", "SCALE_GRID", "PRESETS", "Preset",
    "headline_specs", "ablation_specs", "severity_specs", "scalability_specs",
    "all_specs", "estimate_runs",
]

#: Canonical seed list.  Ablations use ``SEEDS[:5]`` so their IPPO and
#: Nested-MARL runs are cache hits from the headline experiment.
SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021)

SEVERITIES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

#: (n_agents, n_slices) cells for the scalability sweep.
SCALE_GRID: tuple[tuple[int, int], ...] = (
    (3, 3), (6, 3), (10, 3),
    (3, 5), (6, 5), (10, 5),
)

REFERENCE_KAPPA = 1.5
DEFAULT_EPISODES = 80


@dataclass(frozen=True)
class Preset:
    """A trade-off between statistical power and wall-clock time."""

    name: str
    episodes: int
    n_seeds: int
    n_ablation_seeds: int
    severities: tuple[float, ...]
    scale_grid: tuple[tuple[int, int], ...]
    description: str


PRESETS: dict[str, Preset] = {
    "smoke": Preset(
        name="smoke", episodes=6, n_seeds=2, n_ablation_seeds=2,
        severities=(1.5,), scale_grid=((3, 3),),
        description="~1 minute. Verifies the pipeline runs end to end. "
                    "Results are meaningless; do not report them.",
    ),
    "quick": Preset(
        name="quick", episodes=40, n_seeds=5, n_ablation_seeds=3,
        severities=(0.5, 1.5), scale_grid=((3, 3), (6, 3)),
        description="~8 minutes on 4 cores. Indicative trends; "
                    "under-powered for significance claims.",
    ),
    "full": Preset(
        name="full", episodes=DEFAULT_EPISODES, n_seeds=10, n_ablation_seeds=5,
        severities=SEVERITIES, scale_grid=SCALE_GRID,
        description="The reported configuration: 80 episodes, 10 seeds. "
                    "~20 min on 4 cores, ~80 min on 1 core (195 unique runs).",
    ),
    "extended": Preset(
        name="extended", episodes=120, n_seeds=10, n_ablation_seeds=10,
        severities=SEVERITIES, scale_grid=SCALE_GRID,
        description="Longer horizon and more ablation seeds (~45 min on "
                    "4 cores); checks conclusions are not an artefact of E=80.",
    ),
}


# --------------------------------------------------------------------------- #
#  Grids
# --------------------------------------------------------------------------- #
def headline_specs(preset: Preset) -> list[RunSpec]:
    """Every method at the reference severity."""
    methods = list(BASELINES) + list(PROPOSED)
    return [
        RunSpec(
            experiment="headline", method=m, seed=s,
            episodes=preset.episodes, kappa=REFERENCE_KAPPA,
        )
        for m in methods
        for s in SEEDS[: preset.n_seeds]
    ]


def ablation_specs(preset: Preset) -> list[RunSpec]:
    """Ablations plus the two reference points, on a shared seed subset."""
    methods = ["ippo", *PROPOSED, *ABLATIONS]
    return [
        RunSpec(
            experiment="ablation", method=m, seed=s,
            episodes=preset.episodes, kappa=REFERENCE_KAPPA,
        )
        for m in methods
        for s in SEEDS[: preset.n_ablation_seeds]
    ]


def severity_specs(preset: Preset) -> list[RunSpec]:
    """IPPO vs Nested-MARL across the drift-severity sweep."""
    return [
        RunSpec(
            experiment="severity", method=m, seed=s,
            episodes=preset.episodes, kappa=k,
        )
        for m in ("ippo", "nested_marl")
        for k in preset.severities
        for s in SEEDS[: preset.n_seeds]
    ]


def scalability_specs(preset: Preset) -> list[RunSpec]:
    """Growing the agent cohort and the slice count."""
    n_seeds = max(3, preset.n_ablation_seeds)
    return [
        RunSpec(
            experiment="scalability", method=m, seed=s,
            episodes=preset.episodes, kappa=REFERENCE_KAPPA,
            n_agents=n, n_slices=s_count,
        )
        for m in ("ippo", "mappo", "nested_marl")
        for (n, s_count) in preset.scale_grid
        for s in SEEDS[:n_seeds]
    ]


def all_specs(preset: Preset) -> dict[str, list[RunSpec]]:
    return {
        "headline": headline_specs(preset),
        "ablation": ablation_specs(preset),
        "severity": severity_specs(preset),
        "scalability": scalability_specs(preset),
    }


def estimate_runs(preset: Preset) -> dict[str, int]:
    """Number of runs per experiment (before cache hits)."""
    return {k: len(v) for k, v in all_specs(preset).items()}
