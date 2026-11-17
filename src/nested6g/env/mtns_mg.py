"""Multi-Timescale Non-Stationary Markov Game (MTNS-MG) for 6G slicing.

A lightweight, dependency-free simulator of ``N`` cooperating base stations,
each allocating physical resource blocks (PRBs) across ``S`` network slices
(by default eMBB / URLLC / mMTC) under non-stationarity at three separated
timescales.

Formally the game is the tuple

.. math::

    \\mathcal{G} = \\langle \\mathcal{N}, \\mathcal{S}_{env}, \\{A_i\\},
                   \\{r_i\\}, P, \\pi_\\rho, \\pi_d, \\gamma \\rangle

where ``P`` is the single-step transition kernel and ``pi_rho``, ``pi_d`` are
the regime and demand drift kernels acting at rates ``1/tau_slow`` and
``1/tau_med``.  The MTNS-MG is strictly more general than a stationary Markov
game (``P`` depends on the drifting ``rho_t, d_t``) but strictly more
*structured* than a general non-stationary game: the drift is hierarchical, and
it is exactly that structure a multi-timescale learner can exploit.

Design notes
------------
* **No episode reset of the drift clock.**  ``reset()`` clears queues and the
  action memory but the drift process keeps running, so an agent cannot learn a
  spurious "start of episode implies regime 0" shortcut.
* **Observations are strictly local.**  Agent ``i`` sees only its own channel
  and queue state plus the globally broadcast demand and regime indicator; it
  never sees another agent's state.  This keeps independent learners honest.
* **Deterministic given a seed.**  All stochasticity flows through a single
  ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .actions import n_templates, simplex_templates
from .drift import DriftConfig, DriftProcess

__all__ = ["EnvConfig", "SliceEnv", "StepInfo", "SLICE_NAMES"]

SLICE_NAMES = ("eMBB", "URLLC", "mMTC")


@dataclass
class EnvConfig:
    """Configuration of the MTNS-MG simulator.

    The defaults reproduce the canonical configuration: ``N=3`` agents,
    ``S=3`` slices, ``R=3`` regimes, ``K=15`` allocation templates,
    ``tau_med=15``, ``tau_slow=50`` and a horizon of ``T=150`` steps.
    """

    n_agents: int = 3
    n_slices: int = 3
    n_regimes: int = 3
    horizon: int = 150
    granularity: int = 4                  # -> K = C(k+S-1, S-1) = 15 for k=4, S=3

    # --- radio / service model ---------------------------------------------
    prb_total: float = 100.0              # PRBs available to each base station
    slice_efficiency: tuple[float, ...] = (1.00, 0.85, 1.15)
    base_load: tuple[float, ...] = (34.0, 30.0, 26.0)
    load_agent_spread: float = 0.15       # per-agent multiplicative heterogeneity
    queue_max: float = 400.0              # per (agent, slice) backlog cap
    queue_ref: float = 60.0               # latency-term scale in the SLA indicator
    reliability_floor: float = 0.05       # channel below this -> outage

    # --- reward -------------------------------------------------------------
    switching_penalty: float = 0.05       # lambda_sw
    discount: float = 0.95                # gamma (used by the learner, kept here
                                          # so the whole problem is one object)

    # --- drift --------------------------------------------------------------
    tau_med: int = 15
    tau_slow: int = 50
    kappa: float = 1.5
    demand_sensitivity: tuple[float, ...] = (0.70, 1.60, 0.70)
    channel_ar_coeff: float = 0.90
    channel_mean: float = 0.70
    channel_noise_std: float = 0.10
    demand_noise_std: float = 0.10

    def __post_init__(self) -> None:
        for name in ("slice_efficiency", "base_load", "demand_sensitivity"):
            val = getattr(self, name)
            if len(val) != self.n_slices:
                raise ValueError(
                    f"{name} has length {len(val)} but n_slices={self.n_slices}"
                )
        if self.n_regimes < 1:
            raise ValueError("n_regimes must be >= 1")

    # -- derived --------------------------------------------------------------
    @property
    def n_actions(self) -> int:
        """Number of allocation templates ``K``."""
        return n_templates(self.n_slices, self.granularity)

    @property
    def obs_dim(self) -> int:
        """Local observation dimension ``d_o = 3S + R``."""
        return 3 * self.n_slices + self.n_regimes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepInfo:
    """Per-step diagnostics returned alongside the reward."""

    sla: np.ndarray                # (N, S) per-slice SLA satisfaction in [0, 1]
    queue: np.ndarray              # (N, S) backlog after service
    switching: np.ndarray          # (N,)  ||a_t - a_{t-1}||_2 per agent
    regime: int
    demand: np.ndarray             # (S,)
    oracle_sla: float              # SLA an omniscient allocator would achieve
    served: np.ndarray             # (N, S) volume served this step
    offered: np.ndarray            # (N, S) volume offered this step


class SliceEnv:
    """The MTNS-MG environment.

    Example
    -------
    >>> env = SliceEnv(EnvConfig(), seed=0)
    >>> obs = env.reset()
    >>> obs.shape
    (3, 12)
    >>> actions = np.zeros(3, dtype=int)
    >>> obs, rewards, done, info = env.step(actions)
    >>> rewards.shape
    (3,)
    """

    def __init__(self, cfg: EnvConfig | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)

        self.templates = simplex_templates(self.cfg.n_slices, self.cfg.granularity)
        self.n_actions = self.templates.shape[0]
        self.obs_dim = self.cfg.obs_dim

        drift_cfg = DriftConfig(
            n_slices=self.cfg.n_slices,
            n_regimes=self.cfg.n_regimes,
            channel_ar_coeff=self.cfg.channel_ar_coeff,
            channel_mean=self.cfg.channel_mean,
            channel_noise_std=self.cfg.channel_noise_std,
            tau_med=self.cfg.tau_med,
            tau_slow=self.cfg.tau_slow,
            demand_noise_std=self.cfg.demand_noise_std,
            kappa=self.cfg.kappa,
        )
        self.drift = DriftProcess(drift_cfg, self.cfg.n_agents, self.rng)
        # Apply per-slice demand sensitivity: some slices (URLLC by default)
        # swing far more sharply across regimes than others.
        self._apply_demand_sensitivity()

        # Per-agent load heterogeneity, fixed for the lifetime of the env so
        # that agents face genuinely different -- but stationary -- cell loads.
        spread = self.cfg.load_agent_spread
        self._agent_load = self.rng.uniform(
            1.0 - spread, 1.0 + spread, size=(self.cfg.n_agents, 1)
        )

        self._queue = np.zeros((self.cfg.n_agents, self.cfg.n_slices))
        self._prev_alloc = np.full(
            (self.cfg.n_agents, self.cfg.n_slices), 1.0 / self.cfg.n_slices
        )
        self._t = 0

    # ------------------------------------------------------------------ setup
    def _apply_demand_sensitivity(self) -> None:
        """Rescale the regime-conditional demand profile per slice."""
        sens = np.asarray(self.cfg.demand_sensitivity, dtype=np.float64)
        prof = self.drift._profile                     # (R, S), mean-1 deviations
        self.drift._profile = np.clip(1.0 + (prof - 1.0) * sens[None, :], 0.05, None)

    # ------------------------------------------------------------------- core
    def reset(self) -> np.ndarray:
        """Clear queues and action memory; the drift clock keeps running."""
        self._queue[:] = 0.0
        self._prev_alloc[:] = 1.0 / self.cfg.n_slices
        self._t = 0
        return self._observe()

    def _observe(self) -> np.ndarray:
        """Local observations ``o_i = [h_i | q_i | d | rho]``, shape ``(N, d_o)``."""
        st = self.drift.state
        q_norm = self._queue / self.cfg.queue_max
        demand = np.tile(st.demand, (self.cfg.n_agents, 1))
        regime = np.tile(self.drift.regime_onehot(), (self.cfg.n_agents, 1))
        obs = np.concatenate([st.channel, q_norm, demand, regime], axis=1)
        return obs.astype(np.float32)

    def _service(self, alloc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Attempt service for every (agent, slice) pair.

        Returns ``(sla, served, offered)``.  Capacity is proportional to the
        allocated PRB share, the instantaneous channel gain and a per-slice
        spectral-efficiency factor; the backlog carries over between steps.
        """
        cfg = self.cfg
        st = self.drift.state

        offered = (
            np.asarray(cfg.base_load)[None, :]        # (1, S)
            * st.demand[None, :]                      # regime-driven multiplier
            * self._agent_load                        # (N, 1) cell heterogeneity
        )
        capacity = (
            cfg.prb_total
            * alloc                                    # (N, S) PRB share
            * st.channel                               # (N, S) channel gain
            * np.asarray(cfg.slice_efficiency)[None, :]
        )

        backlog = self._queue + offered
        served = np.minimum(backlog, capacity)
        self._queue = np.clip(backlog - served, 0.0, cfg.queue_max)

        # --- soft SLA indicator ------------------------------------------
        # Throughput component: fraction of the offered volume actually served.
        throughput = served / np.maximum(offered, 1e-9)
        throughput = np.clip(throughput, 0.0, 1.0)
        # Latency component: exponentially decaying in the residual backlog.
        latency = np.exp(-self._queue / cfg.queue_ref)
        # Reliability component: hard outage when the channel collapses.
        reliability = (st.channel > cfg.reliability_floor).astype(np.float64)

        sla = throughput * latency * reliability
        return sla, served, offered

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, StepInfo]:
        """Execute one joint action.

        Parameters
        ----------
        actions:
            Integer array of shape ``(N,)`` indexing allocation templates.

        Returns
        -------
        obs, rewards, done, info
        """
        actions = np.asarray(actions, dtype=int).reshape(self.cfg.n_agents)
        if np.any((actions < 0) | (actions >= self.n_actions)):
            raise ValueError(f"actions out of range [0, {self.n_actions})")

        alloc = self.templates[actions].astype(np.float64)     # (N, S)

        sla, served, offered = self._service(alloc)

        w = self.drift.sla_weights                             # (S,)
        utility = (sla * w[None, :]).sum(axis=1)               # (N,)
        switching = np.linalg.norm(alloc - self._prev_alloc, axis=1)
        rewards = utility - self.cfg.switching_penalty * switching

        # Oracle reference used for the SLA-regret curve (never seen by agents).
        oracle_sla = self._oracle_sla()

        self._prev_alloc = alloc
        self.drift.step()
        self._t += 1

        info = StepInfo(
            sla=sla,
            queue=self._queue.copy(),
            switching=switching,
            regime=self.drift.state.regime,
            demand=self.drift.state.demand.copy(),
            oracle_sla=oracle_sla,
            served=served,
            offered=offered,
        )
        done = self._t >= self.cfg.horizon
        return self._observe(), rewards, done, info

    # --------------------------------------------------------------- oracle
    def _oracle_sla(self) -> float:
        """Weighted SLA an omniscient, queue-free allocator would obtain.

        Computed on a *hypothetical* copy of the state: it allocates in exact
        proportion to weighted demand and is charged no switching cost.  Used
        only as the reference for cumulative SLA regret.
        """
        cfg = self.cfg
        st = self.drift.state
        alloc = self.drift.oracle_allocation()[None, :]
        offered = (
            np.asarray(cfg.base_load)[None, :] * st.demand[None, :] * self._agent_load
        )
        capacity = (
            cfg.prb_total * alloc * st.channel * np.asarray(cfg.slice_efficiency)[None, :]
        )
        served = np.minimum(offered, capacity)
        throughput = np.clip(served / np.maximum(offered, 1e-9), 0.0, 1.0)
        residual = np.clip(offered - served, 0.0, cfg.queue_max)
        latency = np.exp(-residual / cfg.queue_ref)
        sla = throughput * latency
        return float((sla * self.drift.sla_weights[None, :]).sum(axis=1).mean())

    # ------------------------------------------------------------------ misc
    @property
    def state_dim(self) -> int:
        """Dimension of the *joint* observation seen by a centralised critic."""
        return self.cfg.n_agents * self.obs_dim

    def describe(self) -> str:
        c = self.cfg
        return (
            f"MTNS-MG  N={c.n_agents}  S={c.n_slices}  R={c.n_regimes}  "
            f"K={self.n_actions}  d_o={self.obs_dim}  T={c.horizon}  "
            f"tau_med={c.tau_med}  tau_slow={c.tau_slow}  kappa={c.kappa}"
        )
