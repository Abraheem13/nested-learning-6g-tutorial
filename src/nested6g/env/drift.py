"""Three-band non-stationarity for the MTNS-MG.

The environment's non-stationarity is structured into three *disjoint temporal
bands* with well-separated rates ``tau_fast << tau_med << tau_slow``:

===========  ==================================  =======================
Band         Signal                              Default period
===========  ==================================  =======================
fast         per-agent, per-slice channel gain   every step (AR(1))
medium       per-slice demand multiplier         ``tau_med = 15`` steps
slow         SLA regime indicator                ``tau_slow = 50`` steps
===========  ==================================  =======================

A single drift *severity* scalar ``kappa`` controls how sharply the demand
distribution differs between regimes.  ``kappa = 0`` recovers a (demand-)
stationary environment; larger values sharpen the contrast between regimes and
therefore increase the reward for tracking the slow band correctly.

Crucially the drift clock is **continuous across episodes** -- there is no
reset at episode boundaries.  This mirrors real operation, in which an
orchestrator's "episode" is an artefact of the training loop and not something
the network is aware of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DriftConfig", "DriftProcess"]


@dataclass
class DriftConfig:
    """Configuration of the three drift bands."""

    n_slices: int = 3
    n_regimes: int = 3

    # --- fast band: channel -------------------------------------------------
    channel_ar_coeff: float = 0.90        # alpha_h in h <- a h + (1-a) mu + xi
    channel_mean: float = 0.70            # mu_h
    channel_noise_std: float = 0.10       # std of xi
    channel_clip: tuple[float, float] = (0.05, 1.0)

    # --- medium band: demand ------------------------------------------------
    tau_med: int = 15                     # demand refresh period (steps)
    demand_base: float = 1.0              # nominal multiplier
    demand_noise_std: float = 0.10        # per-refresh jitter

    # --- slow band: SLA regime ---------------------------------------------
    tau_slow: int = 50                    # regime dwell time (steps)

    # --- severity -----------------------------------------------------------
    kappa: float = 1.5                    # drift severity in [0, 2.5]

    def __post_init__(self) -> None:
        if self.tau_med >= self.tau_slow:
            raise ValueError("tau_med must be strictly smaller than tau_slow")
        if self.kappa < 0.0:
            raise ValueError("kappa must be non-negative")


@dataclass
class DriftState:
    """Mutable state of the drift processes."""

    step: int = 0
    regime: int = 0
    channel: np.ndarray = field(default=None)   # (N, S)
    demand: np.ndarray = field(default=None)    # (S,)


class DriftProcess:
    """Generates the fast/medium/slow signals driving the MTNS-MG.

    The three bands are produced by independent mechanisms so that an
    experimenter can ablate any one of them (e.g. freeze the regime to obtain a
    two-band environment) without disturbing the others.
    """

    def __init__(
        self,
        cfg: DriftConfig,
        n_agents: int,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.n_agents = n_agents
        self.rng = rng

        # Regime-conditional demand profile.  Regime r biases demand toward
        # slice r.  The contrast between the favoured slice and the others is
        # scaled by kappa: at kappa = 0 all slices are equal.
        self._profile = self._build_demand_profile()

        # SLA weight matrix w_s(rho): regime r up-weights slice r.  Rows are
        # normalised to sum to one so that per-step reward is comparable across
        # regimes and the reward scale does not depend on kappa.
        self._sla_weights = self._build_sla_weights()

        self.state = DriftState(
            step=0,
            regime=int(self.rng.integers(cfg.n_regimes)),
            channel=np.full((n_agents, cfg.n_slices), cfg.channel_mean, dtype=np.float64),
            demand=np.ones(cfg.n_slices, dtype=np.float64),
        )
        self._refresh_demand()

    # ------------------------------------------------------------------ setup
    def _build_demand_profile(self) -> np.ndarray:
        """Regime-conditional mean demand multipliers, shape ``(R, S)``."""
        R, S = self.cfg.n_regimes, self.cfg.n_slices
        k = self.cfg.kappa
        prof = np.ones((R, S), dtype=np.float64)
        for r in range(R):
            for s in range(S):
                # Favoured slice scales up with kappa, others scale down;
                # the row mean stays at demand_base for every kappa.
                prof[r, s] = 1.0 + k * (1.0 if s == (r % S) else -1.0 / (S - 1))
        prof = np.clip(prof, 0.05, None) * self.cfg.demand_base
        return prof

    def _build_sla_weights(self) -> np.ndarray:
        """SLA weights ``w_s(rho)``, shape ``(R, S)``, each row summing to 1."""
        R, S = self.cfg.n_regimes, self.cfg.n_slices
        w = np.full((R, S), 1.0, dtype=np.float64)
        for r in range(R):
            w[r, r % S] = 2.0            # the regime's priority slice
        return w / w.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------- step
    def _refresh_demand(self) -> None:
        mean = self._profile[self.state.regime]
        noise = self.rng.normal(0.0, self.cfg.demand_noise_std, size=mean.shape)
        self.state.demand = np.clip(mean + noise, 0.05, None)

    def _advance_channel(self) -> None:
        c = self.cfg
        xi = self.rng.normal(0.0, c.channel_noise_std, size=self.state.channel.shape)
        h = c.channel_ar_coeff * self.state.channel + (1.0 - c.channel_ar_coeff) * c.channel_mean + xi
        self.state.channel = np.clip(h, *c.channel_clip)

    def step(self) -> DriftState:
        """Advance all three bands by one environment step."""
        self.state.step += 1
        t = self.state.step

        self._advance_channel()                       # fast: every step
        if t % self.cfg.tau_slow == 0:                # slow: regime change
            self.state.regime = int((self.state.regime + 1 + self.rng.integers(self.cfg.n_regimes - 1))
                                    % self.cfg.n_regimes)
            self._refresh_demand()                    # regime change re-draws demand
        elif t % self.cfg.tau_med == 0:               # medium: demand refresh
            self._refresh_demand()
        return self.state

    # ----------------------------------------------------------------- access
    @property
    def sla_weights(self) -> np.ndarray:
        """Current SLA weight vector ``w(rho_t)``, shape ``(S,)``."""
        return self._sla_weights[self.state.regime]

    def regime_onehot(self) -> np.ndarray:
        v = np.zeros(self.cfg.n_regimes, dtype=np.float64)
        v[self.state.regime] = 1.0
        return v

    def oracle_allocation(self) -> np.ndarray:
        """Allocation an omniscient controller would pick under the current regime.

        Used only to compute the *SLA regret* baseline; never visible to agents.
        Proportional to the product of SLA weight and offered demand.
        """
        score = self.sla_weights * self.state.demand
        total = score.sum()
        if total <= 0:
            return np.full(self.cfg.n_slices, 1.0 / self.cfg.n_slices)
        return score / total
