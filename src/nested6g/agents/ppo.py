"""Shared PPO machinery: rollout storage, GAE, and the clipped surrogate loss.

Every method compared in this repository -- IPPO, MAPPO, EWC-IPPO and the
Nested-MARL family -- uses *exactly* this PPO core.  The methods differ only in
(a) what the critic observes and (b) how the resulting gradient is applied to
the parameters.  Holding the loss fixed is what makes the comparison a
controlled experiment: any performance difference is attributable to the update
rule, not to a different objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["PPOConfig", "RolloutBuffer", "compute_gae", "ppo_losses"]


@dataclass
class PPOConfig:
    """PPO hyperparameters, identical across all compared methods."""

    discount: float = 0.95          # gamma
    gae_lambda: float = 0.90        # lambda_GAE
    clip_eps: float = 0.20          # epsilon
    epochs: int = 4                 # PPO epochs per update
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    normalise_advantages: bool = True


class RolloutBuffer:
    """Fixed-size on-policy buffer for one agent, one episode."""

    __slots__ = (
        "obs", "critic_obs", "actions", "log_probs", "values",
        "rewards", "dones", "_ptr", "capacity",
    )

    def __init__(self, capacity: int, obs_dim: int, critic_obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.critic_obs = np.zeros((capacity, critic_obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0

    def add(self, obs, critic_obs, action, log_prob, value, reward, done) -> None:
        i = self._ptr
        if i >= self.capacity:
            raise RuntimeError("RolloutBuffer overflow")
        self.obs[i] = obs
        self.critic_obs[i] = critic_obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self._ptr += 1

    def clear(self) -> None:
        self._ptr = 0

    def __len__(self) -> int:
        return self._ptr

    def as_tensors(self) -> dict[str, torch.Tensor]:
        n = self._ptr
        return {
            "obs": torch.from_numpy(self.obs[:n]),
            "critic_obs": torch.from_numpy(self.critic_obs[:n]),
            "actions": torch.from_numpy(self.actions[:n]),
            "log_probs": torch.from_numpy(self.log_probs[:n]),
            "values": torch.from_numpy(self.values[:n]),
            "rewards": torch.from_numpy(self.rewards[:n]),
            "dones": torch.from_numpy(self.dones[:n]),
        }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised Advantage Estimation.

    Returns ``(advantages, returns)`` where ``returns = advantages + values``.
    """
    n = rewards.shape[0]
    advantages = torch.zeros(n, dtype=torch.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(n)):
        non_terminal = 1.0 - dones[t].item()
        delta = rewards[t].item() + gamma * next_value * non_terminal - values[t].item()
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
        next_value = values[t].item()
    returns = advantages + values
    return advantages, returns


def ppo_losses(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    entropy: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    cfg: PPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate + value + entropy loss.

    Returns the scalar loss and a dict of diagnostics.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = 0.5 * (returns - values).pow(2).mean()
    entropy_loss = -entropy.mean()

    loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs)).mean().item()
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()

    diagnostics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }
    return loss, diagnostics
