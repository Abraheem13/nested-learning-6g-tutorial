"""Registry of the compared algorithms.

Every method is a *declarative* specification consumed by a single learner
implementation (:mod:`nested6g.agents.learner`).  This is deliberate: it makes
it structurally impossible for two methods to differ by an accidental
implementation detail rather than by the mechanism under study.

The seven entries fall into three families:

**Baselines**
    ``ippo``          Independent PPO, one Adam, one learning rate.
    ``mappo``         Centralised critic over the joint observation.
    ``ewc_ippo``      IPPO plus online Elastic Weight Consolidation.

**The proposed method**
    ``nested_marl``   Three depth-partitioned parameter groups updated at
                      separated rates, with an EMA continuum-memory anchor
                      stabilising the slow group.

**Ablations** (each removes exactly one ingredient)
    ``nested_no_ema``          Timescale separation, no memory anchor.
    ``nested_same_rate``       Memory anchor and three optimisers, but a single
                               shared rate -- isolates timescale separation from
                               architectural depth.
    ``nested_two_timescale``   Slow and medium groups collapsed -- tests whether
                               three levels are actually warranted.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MethodSpec", "METHODS", "get_method", "ABLATIONS", "BASELINES", "PROPOSED"]


@dataclass(frozen=True)
class MethodSpec:
    """Declarative description of one algorithm."""

    name: str
    label: str
    #: Learning rates.  Length 1 -> a single optimiser over all parameters;
    #: length 3 -> one optimiser per depth group, slow to fast.
    learning_rates: tuple[float, ...] = (5e-4,)
    #: Critic sees the joint observation of all agents (CTDE).
    centralised_critic: bool = False
    #: EMA decay for the continuum-memory trace; ``None`` disables the trace.
    ema_lambda: float | None = None
    #: Slow-group stabilisation coefficient.  ``1.0`` disables stabilisation.
    stabilise_beta: float = 1.0
    #: Online-EWC penalty coefficient.  ``0.0`` disables the penalty.
    ewc_lambda: float = 0.0
    #: Decay applied to the running Fisher estimate each episode.
    ewc_fisher_decay: float = 0.95
    family: str = "baseline"
    notes: str = ""
    #: How parameters are assigned to the three rate groups.
    #:   "depth"    -- by layer depth (input slow, output fast); the original,
    #:                 hand-designed partition, on the original code path.
    #:   "depth_pg" -- the same depth assignment realised on the per-tensor code
    #:                 path below; exists to prove that path is bit-identical.
    #:   "inverted" -- depth reversed (input fast, output slow).
    #:   "random"   -- tensors shuffled into groups of the same sizes, with a
    #:                 seed-dependent permutation.
    #:   "grad_snr" -- data-driven: after a warm-up, tensors are ranked by
    #:                 gradient signal-to-noise ratio and the highest-SNR third
    #:                 is made fast, the lowest slow (see learner.py).
    partition: str = "depth"
    #: Warm-up episodes before the grad_snr partition is fixed.
    partition_warmup: int = 5

    @property
    def is_nested(self) -> bool:
        return len(self.learning_rates) > 1

    @property
    def n_optimisers(self) -> int:
        return len(self.learning_rates)


#: Reference nested rates (alpha_0 << alpha_1 << alpha_2), slow -> fast.
NESTED_RATES: tuple[float, float, float] = (1e-4, 5e-4, 2e-3)
BASELINE_RATE: float = 5e-4


METHODS: dict[str, MethodSpec] = {
    # ------------------------------------------------------------ baselines
    "ippo": MethodSpec(
        name="ippo",
        label="IPPO (baseline)",
        learning_rates=(BASELINE_RATE,),
        family="baseline",
        notes="Independent PPO with Adam and a single learning rate.",
    ),
    "mappo": MethodSpec(
        name="mappo",
        label="MAPPO",
        learning_rates=(BASELINE_RATE,),
        centralised_critic=True,
        family="baseline",
        notes="Centralised-training / decentralised-execution actor-critic. "
              "Isolates the effect of centralised value estimation.",
    ),
    "ewc_ippo": MethodSpec(
        name="ewc_ippo",
        label="EWC-IPPO",
        learning_rates=(BASELINE_RATE,),
        ewc_lambda=100.0,
        family="baseline",
        notes="IPPO plus online Elastic Weight Consolidation, the canonical "
              "continual-learning regulariser.",
    ),
    # ------------------------------------------------------------- proposed
    "nested_marl": MethodSpec(
        name="nested_marl",
        label="Nested-MARL (ours)",
        learning_rates=NESTED_RATES,
        ema_lambda=0.995,
        stabilise_beta=0.95,
        family="proposed",
        notes="Three depth-partitioned groups at separated rates with an EMA "
              "continuum-memory anchor on the slowest group.",
    ),
    # ------------------------------------------------------------ ablations
    "nested_no_ema": MethodSpec(
        name="nested_no_ema",
        label="Nested-MARL, no EMA",
        learning_rates=NESTED_RATES,
        ema_lambda=0.995,
        stabilise_beta=1.0,          # beta = 1 disables stabilisation
        family="ablation",
        notes="Timescale separation retained, continuum-memory anchor removed.",
    ),
    "nested_same_rate": MethodSpec(
        name="nested_same_rate",
        label="Same rate (no timescale sep.)",
        learning_rates=(BASELINE_RATE, BASELINE_RATE, BASELINE_RATE),
        ema_lambda=0.995,
        stabilise_beta=0.95,
        family="ablation",
        notes="Three groups, three optimisers and the EMA memory, but a single "
              "shared rate. The decisive control: it holds architecture fixed "
              "and removes only timescale separation.",
    ),
    "nested_two_timescale": MethodSpec(
        name="nested_two_timescale",
        label="Nested-MARL, 2 timescales",
        learning_rates=(1e-4, 1e-4, 2e-3),
        ema_lambda=0.995,
        stabilise_beta=0.95,
        family="ablation",
        notes="Slow and medium groups collapsed onto one rate; tests whether "
              "the third level is empirically warranted.",
    ),
}

BASELINES = ("ippo", "mappo", "ewc_ippo")
PROPOSED = ("nested_marl",)
ABLATIONS = ("nested_same_rate", "nested_two_timescale", "nested_no_ema")


def get_method(name: str) -> MethodSpec:
    """Look up a method specification by name."""
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(
            f"unknown method {name!r}; available: {sorted(METHODS)}"
        ) from None


# --------------------------------------------------------------------------- #
#  IPPO learning-rate sweep: the tuned-baseline control
# --------------------------------------------------------------------------- #
# Registered at module import (not at runtime) so that spawned worker
# processes, which re-import this module from scratch, also see them.
SWEEP_RATES: tuple[float, ...] = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3)


def _sweep_name(lr: float) -> str:
    """Filesystem-safe method name for a swept rate, e.g. 1e-4 -> ippo_lr0p0001."""
    return f"ippo_lr{lr:g}".replace("-", "m").replace(".", "p")


SWEEP_METHODS: tuple[str, ...] = tuple(_sweep_name(lr) for lr in SWEEP_RATES)

for _lr in SWEEP_RATES:
    _n = _sweep_name(_lr)
    METHODS[_n] = MethodSpec(
        name=_n,
        label=f"IPPO ($\\alpha$={_lr:g})",
        learning_rates=(_lr,),
        family="baseline",
        notes=(f"Single-rate IPPO at {_lr:g}. Part of the tuned-baseline "
               "control, which tests whether the multi-timescale gain is "
               "merely a better choice of learning rate."),
    )
del _lr, _n


# --------------------------------------------------------------------------- #
#  Nested rate-configuration sweep: the matched-tuning-budget control
# --------------------------------------------------------------------------- #
# Sweeping only the baseline's learning rate while leaving the proposed method
# at a single hand-picked configuration is an unfair comparison -- in the
# baseline's favour. A valid tuned-vs-tuned test gives both families the same
# search budget. These five triples span the same region as SWEEP_RATES.
NESTED_CONFIGS: dict[str, tuple[float, float, float]] = {
    "nested_cfg_original":  (1e-4, 5e-4, 2e-3),   # the reference configuration
    "nested_cfg_shifted":   (5e-4, 2e-3, 5e-3),
    "nested_cfg_high":      (1e-3, 3e-3, 8e-3),
    "nested_cfg_widespan":  (2e-4, 2e-3, 1e-2),
    "nested_cfg_narrowhi":  (2e-3, 4e-3, 8e-3),
}
NESTED_SWEEP_METHODS: tuple[str, ...] = tuple(NESTED_CONFIGS)

for _n, _lrs in NESTED_CONFIGS.items():
    METHODS[_n] = MethodSpec(
        name=_n,
        label="Nested ({})".format(", ".join(f"{x:g}" for x in _lrs)),
        learning_rates=_lrs,
        ema_lambda=0.995,
        stabilise_beta=0.95,
        family="proposed",
        notes=("Nested-MARL rate configuration, part of the matched-tuning-"
               "budget control."),
    )
del _n, _lrs


# --------------------------------------------------------------------------- #
#  Review analysis A: is the hand-designed partition doing anything?
# --------------------------------------------------------------------------- #
# Each alternative partition is run at the two rate triples used in the chapter:
# the reference triple and the best triple of the matched-budget control. The
# depth-partition runs for both triples already exist ("nested_marl" and
# "nested_cfg_high"), so only the alternatives are new.
PARTITION_TRIPLES: dict[str, tuple[float, float, float]] = {
    "ref": (1e-4, 5e-4, 2e-3),
    "tuned": (1e-3, 3e-3, 8e-3),
}
PARTITION_DEPTH_METHOD: dict[str, str] = {"ref": "nested_marl", "tuned": "nested_cfg_high"}
PARTITION_KINDS: tuple[str, ...] = ("inverted", "random", "grad_snr")
PARTITION_METHODS: dict[tuple[str, str], str] = {}

for _t, _lrs in PARTITION_TRIPLES.items():
    for _k in PARTITION_KINDS:
        _n = f"part_{_t}_{_k}"
        PARTITION_METHODS[(_t, _k)] = _n
        METHODS[_n] = MethodSpec(
            name=_n,
            label=f"{_k} partition ({_t} rates)",
            learning_rates=_lrs,
            ema_lambda=0.995,
            stabilise_beta=0.95,
            family="review",
            partition=_k,
            notes="Review analysis A: partition sensitivity.",
        )
del _t, _lrs, _k, _n
