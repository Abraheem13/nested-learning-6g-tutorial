"""Actor-critic networks with a depth-based parameter-group partition.

Every agent carries an actor ``pi(a | o; theta^pi)`` and a critic
``V(o; theta^V)``, each a two-hidden-layer MLP with 64 ReLU units.  The
parameters of *both* heads are partitioned by **layer depth** into three
groups:

======  ==================  ================================================
Group   Layers              Intuition
======  ==================  ================================================
``0``   input linear        representation of the observation -- should be
                            *stable*, because the observation-to-feature map
                            is shared across regimes
``1``   hidden linear       intermediate features -- moderate plasticity
``2``   output linear       the decision boundary / value head -- should be
                            *fast*, because the optimal action changes with
                            the fast band
======  ==================  ================================================

The same partition is applied to the actor and the critic so that the two
heads share timescales; letting one head race ahead of the other is a known
source of actor-critic instability.

Parameter counts (``d_o = 12``, ``K = 15``, ``h = 64``) reproduce the reference
system exactly: actor 5,967 + critic 5,057 = **11,024** per agent, and a
centralised critic over the joint observation grows from 6,593 (``N = 3``) to
11,969 (``N = 10``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

__all__ = ["ActorCritic", "count_parameters"]


def _orthogonal_mlp(in_dim: int, hidden: int, out_dim: int, out_gain: float) -> nn.Sequential:
    """Two-hidden-layer MLP with orthogonal initialisation.

    Orthogonal init with a small output gain is standard PPO practice: it keeps
    the initial policy close to uniform and the initial value estimates small,
    which materially reduces seed variance.
    """
    layers = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )
    gains = [nn.init.calculate_gain("relu"), nn.init.calculate_gain("relu"), out_gain]
    for module, gain in zip([layers[0], layers[2], layers[4]], gains):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)
    return layers


class ActorCritic(nn.Module):
    """Independent-learner actor-critic with a three-way depth partition.

    Parameters
    ----------
    obs_dim:
        Dimension of the actor's (local) observation.
    n_actions:
        Size of the discrete action set ``K``.
    hidden:
        Width of both hidden layers.
    critic_obs_dim:
        Dimension of the critic's input.  Equal to ``obs_dim`` for an
        independent learner; set to the *joint* observation dimension to obtain
        a centralised critic (MAPPO).
    """

    N_GROUPS = 3

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 64,
        critic_obs_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.critic_obs_dim = critic_obs_dim if critic_obs_dim is not None else obs_dim

        self.actor = _orthogonal_mlp(obs_dim, hidden, n_actions, out_gain=0.01)
        self.critic = _orthogonal_mlp(self.critic_obs_dim, hidden, 1, out_gain=1.0)

    # --------------------------------------------------------------- forward
    def policy(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None):
        """Sample an action; return ``(action, log_prob, value)``."""
        dist = self.policy(obs)
        action = dist.sample()
        value = self.value(critic_obs if critic_obs is not None else obs)
        return action, dist.log_prob(action), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor,
                 critic_obs: torch.Tensor | None = None):
        """Return ``(log_prob, entropy, value)`` for stored transitions."""
        dist = self.policy(obs)
        value = self.value(critic_obs if critic_obs is not None else obs)
        return dist.log_prob(actions), dist.entropy(), value

    # ------------------------------------------------------- parameter groups
    def parameter_groups(self) -> list[list[nn.Parameter]]:
        """Partition every parameter by layer depth into ``N_GROUPS`` lists.

        Index ``0`` is the slowest (input layers), index ``2`` the fastest
        (output layers).  The partition is a *cover*: every parameter of the
        module appears in exactly one group.
        """
        groups: list[list[nn.Parameter]] = [[] for _ in range(self.N_GROUPS)]
        for head in (self.actor, self.critic):
            for depth, layer_idx in enumerate((0, 2, 4)):
                layer = head[layer_idx]
                groups[depth].extend([layer.weight, layer.bias])

        covered = sum(p.numel() for g in groups for p in g)
        total = sum(p.numel() for p in self.parameters())
        assert covered == total, f"partition covers {covered} of {total} parameters"
        return groups

    def group_index(self) -> dict[int, int]:
        """Map ``id(parameter) -> group index``, for masked EMA operations."""
        return {
            id(p): gi for gi, group in enumerate(self.parameter_groups()) for p in group
        }

    def describe(self) -> str:
        a = count_parameters(self.actor)
        c = count_parameters(self.critic)
        return f"actor={a:,}  critic={c:,}  total={a + c:,}"


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
