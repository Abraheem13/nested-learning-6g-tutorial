"""The multi-agent learner.

One class implements every compared algorithm; the differences are entirely
driven by a :class:`~nested6g.agents.methods.MethodSpec`.  The learning loop is:

.. code-block:: text

    for each episode:
        roll out T steps with the current policies          (on-policy)
        for each agent, in parallel:
            compute GAE advantages and returns
            for n = 1..N_epochs:
                g <- grad of the PPO loss
                for l in {0, 1, 2}:                          Eq. (3)
                    theta_l <- theta_l - alpha_l * Adam_l(g_l)
            theta_bar <- lambda * theta_bar + (1-lambda) * theta      Eq. (4)
            theta_0   <- beta * theta_0 + (1-beta) * theta_bar_0      Eq. (5)

For a single-rate method the inner ``for l`` loop collapses to one optimiser and
Eqs. (4)-(5) are skipped, recovering standard IPPO exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..env.mtns_mg import SliceEnv
from .methods import MethodSpec
from .networks import ActorCritic
from .ppo import PPOConfig, RolloutBuffer, compute_gae, ppo_losses

__all__ = ["MultiAgentLearner", "EpisodeRecord"]


@dataclass
class EpisodeRecord:
    """Per-episode metrics logged during training."""

    episode: int
    reward: float                 # sum over steps of the mean per-agent reward
    sla: float                    # mean over agents, slices and steps
    sla_per_slice: list[float]    # mean per slice
    queue: float                  # mean backlog per (agent, slice)
    switching: float              # mean ||a_t - a_{t-1}||_2
    oracle_sla: float             # weighted SLA of an omniscient allocator
    regret: float                 # oracle_sla - achieved utility, summed
    entropy: float
    approx_kl: float

    def as_dict(self) -> dict[str, float]:
        d = {k: v for k, v in self.__dict__.items() if k != "sla_per_slice"}
        for i, v in enumerate(self.sla_per_slice):
            d[f"sla_slice_{i}"] = v
        return d


class _EWCState:
    """Online Elastic Weight Consolidation state for one agent.

    Maintains a running diagonal Fisher estimate and an anchor point.  Because
    the MTNS-MG has no explicit task boundaries, the Fisher is refreshed every
    episode with geometric decay -- the standard "online EWC" formulation.
    """

    def __init__(self, params: list[nn.Parameter], decay: float) -> None:
        self.decay = decay
        self.fisher = [torch.zeros_like(p) for p in params]
        self.anchor = [p.detach().clone() for p in params]

    def penalty(self, params: list[nn.Parameter]) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float32)
        for f, a, p in zip(self.fisher, self.anchor, params):
            total = total + (f * (p - a).pow(2)).sum()
        return 0.5 * total

    def consolidate(self, params: list[nn.Parameter], grads: list[torch.Tensor]) -> None:
        for i, (g, p) in enumerate(zip(grads, params)):
            contrib = torch.zeros_like(p) if g is None else g.detach().pow(2)
            self.fisher[i] = self.decay * self.fisher[i] + contrib
            self.anchor[i] = p.detach().clone()


class MultiAgentLearner:
    """Trains ``N`` independent (or centrally-critiqued) actor-critic agents."""

    def __init__(
        self,
        env: SliceEnv,
        spec: MethodSpec,
        ppo_cfg: PPOConfig | None = None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.spec = spec
        self.ppo = ppo_cfg or PPOConfig(discount=env.cfg.discount)
        self.device = torch.device(device)
        self.n_agents = env.cfg.n_agents

        torch.manual_seed(seed)
        critic_dim = env.state_dim if spec.centralised_critic else env.obs_dim

        self.nets: list[ActorCritic] = [
            ActorCritic(env.obs_dim, env.n_actions, hidden=64, critic_obs_dim=critic_dim)
            .to(self.device)
            for _ in range(self.n_agents)
        ]

        self.partition = getattr(spec, "partition", "depth")
        self.partition_report: list[dict] | None = None
        self._episodes_done = 0
        self._gstats: list[list[list[torch.Tensor]]] | None = None
        if self.partition == "depth":
            # Original code path: one Adam per depth group. Unchanged, so every
            # run produced before alternative partitions existed is still valid.
            self.optimisers: list[list[torch.optim.Optimizer]] = [
                self._build_optimisers(net) for net in self.nets
            ]
        else:
            self._init_partition(seed)

        # --- continuum-memory trace ----------------------------------------
        self.ema: list[list[torch.Tensor]] | None = None
        if spec.ema_lambda is not None:
            self.ema = [
                [p.detach().clone() for p in net.parameters()] for net in self.nets
            ]

        # --- online EWC ------------------------------------------------------
        self.ewc: list[_EWCState] | None = None
        if spec.ewc_lambda > 0.0:
            self.ewc = [
                _EWCState(list(net.parameters()), spec.ewc_fisher_decay)
                for net in self.nets
            ]

        self.buffers = [
            RolloutBuffer(env.cfg.horizon, env.obs_dim, critic_dim)
            for _ in range(self.n_agents)
        ]
        # Slow-group membership must be resolved per network: parameter object
        # identities differ between agents, so a single shared set would silently
        # stabilise only agent 0.
        if self.partition == "depth":
            self._slow_param_ids: list[set[int]] = [
                {id(p) for p in net.parameter_groups()[0]} for net in self.nets
            ]
        else:
            self._refresh_slow_ids()

    # --------------------------------------------------------------- setup
    def _build_optimisers(self, net: ActorCritic) -> list[torch.optim.Optimizer]:
        """One Adam per parameter group (nested) or a single Adam (baseline)."""
        lrs = self.spec.learning_rates
        if len(lrs) == 1:
            return [torch.optim.Adam(net.parameters(), lr=lrs[0], eps=1e-5)]
        groups = net.parameter_groups()
        if len(groups) != len(lrs):
            raise ValueError(
                f"{len(lrs)} learning rates but {len(groups)} parameter groups"
            )
        return [
            torch.optim.Adam(group, lr=lr, eps=1e-5)
            for group, lr in zip(groups, lrs)
        ]

    # ------------------------------------------------- alternative partitions
    def _depth_assignment(self, net: ActorCritic) -> list[int]:
        """Group index (0 slow .. 2 fast) of each tensor in ``net.parameters()``."""
        gid = net.group_index()
        return [gid[id(p)] for p in net.parameters()]

    def _init_partition(self, seed: int) -> None:
        """Build one Adam per agent with one parameter group per tensor.

        Adam is element-wise, so giving every tensor its own parameter group,
        each with the learning rate of its assigned timescale group, is
        mathematically the same update as one Adam per timescale group; the
        test suite checks the two paths are bit-identical for the depth
        assignment. Per-tensor groups make it possible to reassign a tensor to
        a different rate mid-training without discarding its Adam state, which
        the data-driven partition needs.
        """
        lrs = self.spec.learning_rates
        if len(lrs) != 3:
            raise ValueError("alternative partitions need three learning rates")
        kind = self.partition
        if kind not in ("depth_pg", "inverted", "random", "grad_snr"):
            raise ValueError(f"unknown partition {kind!r}")

        self._assign: list[list[int]] = []
        for i, net in enumerate(self.nets):
            depth = self._depth_assignment(net)
            if kind == "depth_pg":
                assign = depth
            elif kind == "inverted":
                assign = [2 - g for g in depth]
            elif kind == "random":
                # Same group sizes as the depth partition; tensors shuffled.
                sizes = [depth.count(g) for g in range(3)]
                rng = np.random.default_rng(10_007 * seed + i)
                order = rng.permutation(len(depth))
                assign = [0] * len(depth)
                pos = 0
                for g, n in enumerate(sizes):
                    for k in order[pos:pos + n]:
                        assign[int(k)] = g
                    pos += n
            else:  # grad_snr: every tensor at the medium rate during warm-up
                assign = [1] * len(depth)
            self._assign.append(assign)

        self.optimisers = []
        for i, net in enumerate(self.nets):
            groups = [{"params": [p], "lr": lrs[g]}
                      for p, g in zip(net.parameters(), self._assign[i])]
            self.optimisers.append([torch.optim.Adam(groups, lr=lrs[1], eps=1e-5)])

        if kind == "grad_snr":
            self._gstats = [
                [[torch.zeros_like(p), torch.zeros_like(p), torch.zeros(())]
                 for p in net.parameters()]
                for net in self.nets
            ]
        else:
            self._record_partition()

    def _refresh_slow_ids(self) -> None:
        """Slow group = tensors assigned to group 0. Empty during warm-up."""
        warming = self.partition == "grad_snr" and self._gstats is not None
        self._slow_param_ids = [
            set() if warming else
            {id(p) for p, g in zip(net.parameters(), self._assign[i]) if g == 0}
            for i, net in enumerate(self.nets)
        ]

    def _record_partition(self, snr: list[list[float]] | None = None) -> None:
        names = [n for n, _ in self.nets[0].named_parameters()]
        self.partition_report = [
            {"agent": i, "kind": self.partition,
             "groups": {name: int(g) for name, g in zip(names, self._assign[i])},
             **({"snr": {name: float(v) for name, v in zip(names, snr[i])}}
                if snr is not None else {})}
            for i in range(self.n_agents)
        ]

    def _accumulate_grad_stats(self, i: int, params: list[nn.Parameter]) -> None:
        for st, p in zip(self._gstats[i], params):
            if p.grad is None:
                continue
            g = p.grad.detach()
            st[0].add_(g)
            st[1].add_(g * g)
            st[2].add_(1.0)

    def _finalise_snr_partition(self) -> None:
        """Rank tensors by gradient SNR; highest third fast, lowest third slow.

        For each tensor, signal = mean over elements of (mean gradient)^2 and
        noise = mean over elements of the gradient variance, both taken over
        every PPO update of the warm-up. Their ratio is the drift-to-noise
        quantity that the chapter's cube-root step-size law (eq:cuberoot)
        says should set the step size (alpha* grows with it), used here only to *rank* tensors. Group sizes
        are kept equal to the depth partition so that only the assignment,
        not the group sizes, differs.
        """
        lrs = self.spec.learning_rates
        all_snr: list[list[float]] = []
        for i, net in enumerate(self.nets):
            snr = []
            for s_, ss_, n_ in self._gstats[i]:
                n = max(float(n_), 1.0)
                mean = s_ / n
                var = (ss_ / n - mean * mean).clamp_min(0.0)
                snr.append(float((mean * mean).mean() / (var.mean() + 1e-12)))
            depth = self._depth_assignment(net)
            sizes = [depth.count(g) for g in range(3)]
            order = sorted(range(len(snr)), key=lambda k: (snr[k], k))  # low -> high
            assign = [0] * len(snr)
            pos = 0
            for g, n in enumerate(sizes):
                for k in order[pos:pos + n]:
                    assign[k] = g
                pos += n
            self._assign[i] = assign
            for pg, g in zip(self.optimisers[i][0].param_groups, assign):
                pg["lr"] = lrs[g]
            all_snr.append(snr)
        self._gstats = None
        self._refresh_slow_ids()
        self._record_partition(all_snr)

    # ------------------------------------------------------------- rollout
    def _critic_input(self, obs: np.ndarray, agent: int) -> np.ndarray:
        return obs.reshape(-1) if self.spec.centralised_critic else obs[agent]

    @torch.no_grad()
    def collect(self) -> tuple[EpisodeRecord, np.ndarray]:
        """Run one episode on-policy and fill the rollout buffers."""
        env = self.env
        obs = env.reset()
        for buf in self.buffers:
            buf.clear()

        tot_reward = 0.0
        sla_steps, queue_steps, switch_steps = [], [], []
        oracle_steps, utility_steps = [], []

        done = False
        while not done:
            actions = np.empty(self.n_agents, dtype=np.int64)
            packed = []
            for i, net in enumerate(self.nets):
                o = torch.as_tensor(obs[i], device=self.device).unsqueeze(0)
                co = torch.as_tensor(
                    self._critic_input(obs, i), device=self.device
                ).unsqueeze(0)
                a, lp, v = net.act(o, co)
                actions[i] = int(a.item())
                packed.append((obs[i], self._critic_input(obs, i),
                               int(a.item()), float(lp.item()), float(v.item())))

            next_obs, rewards, done, info = env.step(actions)
            for i, (o, co, a, lp, v) in enumerate(packed):
                self.buffers[i].add(o, co, a, lp, v, float(rewards[i]), float(done))

            tot_reward += float(rewards.mean())
            sla_steps.append(info.sla.mean(axis=0))       # (S,)
            queue_steps.append(info.queue.mean())
            switch_steps.append(info.switching.mean())
            oracle_steps.append(info.oracle_sla)
            # achieved weighted utility (reward before the switching penalty)
            w = env.drift.sla_weights
            utility_steps.append(float((info.sla * w[None, :]).sum(axis=1).mean()))
            obs = next_obs

        sla_arr = np.asarray(sla_steps)                    # (T, S)
        record = EpisodeRecord(
            episode=-1,
            reward=tot_reward,
            sla=float(sla_arr.mean()),
            sla_per_slice=[float(x) for x in sla_arr.mean(axis=0)],
            queue=float(np.mean(queue_steps)),
            switching=float(np.mean(switch_steps)),
            oracle_sla=float(np.mean(oracle_steps)),
            regret=float(np.sum(np.asarray(oracle_steps) - np.asarray(utility_steps))),
            entropy=0.0,
            approx_kl=0.0,
        )
        return record, obs

    # -------------------------------------------------------------- update
    def update(self, last_obs: np.ndarray) -> dict[str, float]:
        """Apply the nested (or single-rate) PPO update to every agent."""
        diags: list[dict[str, float]] = []
        for i, net in enumerate(self.nets):
            diags.append(self._update_agent(i, net, last_obs))
            self._apply_memory(i, net)
        self._episodes_done += 1
        if (self._gstats is not None
                and self._episodes_done >= self.spec.partition_warmup):
            self._finalise_snr_partition()
        return {k: float(np.mean([d[k] for d in diags])) for k in diags[0]}

    def _update_agent(self, i: int, net: ActorCritic, last_obs: np.ndarray) -> dict[str, float]:
        data = self.buffers[i].as_tensors()

        with torch.no_grad():
            last_ci = torch.as_tensor(
                self._critic_input(last_obs, i), device=self.device
            ).unsqueeze(0)
            last_value = float(net.value(last_ci).item())

        advantages, returns = compute_gae(
            data["rewards"], data["values"], data["dones"],
            last_value, self.ppo.discount, self.ppo.gae_lambda,
        )
        if self.ppo.normalise_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        params = list(net.parameters())
        last_diag: dict[str, float] = {}
        last_grads: list[torch.Tensor] = []

        for _ in range(self.ppo.epochs):
            new_lp, entropy, values = net.evaluate(
                data["obs"], data["actions"], data["critic_obs"]
            )
            loss, diag = ppo_losses(
                new_lp, data["log_probs"], entropy, values,
                returns, advantages, self.ppo,
            )
            if self.ewc is not None:
                loss = loss + self.spec.ewc_lambda * self.ewc[i].penalty(params)

            for opt in self.optimisers[i]:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, self.ppo.max_grad_norm)
            if self._gstats is not None:
                self._accumulate_grad_stats(i, params)

            # Eq. (3): each parameter group steps with its own optimiser and
            # its own rate.  For a single-rate method this is one step.
            for opt in self.optimisers[i]:
                opt.step()

            last_diag = diag
            last_grads = [p.grad.detach().clone() if p.grad is not None else None
                          for p in params]

        if self.ewc is not None:
            self.ewc[i].consolidate(params, last_grads)
        return last_diag

    def _apply_memory(self, i: int, net: ActorCritic) -> None:
        """Eqs. (4)-(5): update the EMA trace, then stabilise the slow group."""
        if self.ema is None:
            return
        lam = float(self.spec.ema_lambda)
        beta = float(self.spec.stabilise_beta)

        with torch.no_grad():
            for k, p in enumerate(net.parameters()):
                # Eq. (4): continuum-memory trace over the full parameter vector.
                self.ema[i][k].mul_(lam).add_(p.detach(), alpha=1.0 - lam)
                # Eq. (5): pull the slow group toward the anchor.  beta = 1
                # leaves the parameters untouched (the "no EMA" ablation).
                if beta < 1.0 and id(p) in self._slow_param_ids[i]:
                    p.mul_(beta).add_(self.ema[i][k], alpha=1.0 - beta)

    # ---------------------------------------------------------------- train
    def train(self, episodes: int, progress: bool = False) -> list[EpisodeRecord]:
        """Train for ``episodes`` episodes; return the per-episode log."""
        history: list[EpisodeRecord] = []
        for ep in range(episodes):
            record, last_obs = self.collect()
            diag = self.update(last_obs)
            record.episode = ep
            record.entropy = diag.get("entropy", 0.0)
            record.approx_kl = diag.get("approx_kl", 0.0)
            history.append(record)
            if progress and (ep + 1) % 10 == 0:
                print(
                    f"    ep {ep + 1:3d}/{episodes}  reward {record.reward:7.2f}  "
                    f"SLA {record.sla:.3f}  switch {record.switching:.3f}",
                    flush=True,
                )
        return history

    # ----------------------------------------------------------------- misc
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.nets[0].parameters())

    def n_optimisers_per_agent(self) -> int:
        return len(self.optimisers[0])
