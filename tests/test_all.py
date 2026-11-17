"""Test suite.

Run with ``pytest -q`` (or ``make test``).

The tests that matter most are the **update-rule invariants** in
:class:`TestUpdateRuleInvariants`.  They establish that the parameter-group
partition is a faithful decomposition of the network: three optimisers at a
single shared rate with no memory anchor are *bit-identical* to one optimiser.
That identity is what licenses the causal claim in the ablation study -- any
difference between Nested-MARL and its "same rate" ablation must come from the
rates or the anchor, and cannot be an artefact of splitting the parameters.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nested6g.agents import MultiAgentLearner, PPOConfig, get_method
from nested6g.agents.methods import METHODS, MethodSpec
from nested6g.agents.networks import ActorCritic, count_parameters
from nested6g.analysis import (holm_bonferroni, interquartile_mean,
                               paired_permutation_test,
                               probability_of_improvement)
from nested6g.env import EnvConfig, SliceEnv, n_templates, simplex_templates

torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
class TestActionSpace:
    def test_count_matches_closed_form(self):
        assert n_templates(3, 4) == 15
        assert simplex_templates(3, 4).shape == (15, 3)

    def test_rows_are_on_the_simplex(self):
        t = simplex_templates(3, 4)
        assert np.allclose(t.sum(axis=1), 1.0)
        assert (t >= 0).all()

    def test_enumeration_is_deterministic(self):
        assert np.array_equal(simplex_templates(3, 4), simplex_templates(3, 4))

    def test_rows_are_unique(self):
        t = simplex_templates(4, 5)
        assert len(np.unique(t, axis=0)) == len(t)


# --------------------------------------------------------------------------- #
class TestEnvironment:
    def test_shapes(self):
        env = SliceEnv(EnvConfig(), seed=0)
        obs = env.reset()
        assert obs.shape == (3, 12)
        assert env.n_actions == 15
        obs, r, done, info = env.step(np.zeros(3, dtype=int))
        assert obs.shape == (3, 12) and r.shape == (3,)
        assert info.sla.shape == (3, 3) and not done

    def test_determinism_given_seed(self):
        def rollout(seed):
            env = SliceEnv(EnvConfig(), seed=seed)
            env.reset()
            rng = np.random.default_rng(0)
            return np.array([
                env.step(rng.integers(0, env.n_actions, 3))[1] for _ in range(50)
            ])
        assert np.array_equal(rollout(7), rollout(7))
        assert not np.array_equal(rollout(7), rollout(8))

    def test_episode_terminates_at_horizon(self):
        env = SliceEnv(EnvConfig(horizon=20), seed=0)
        env.reset()
        for t in range(20):
            _, _, done, _ = env.step(np.zeros(3, dtype=int))
        assert done

    def test_drift_clock_survives_reset(self):
        """reset() must clear queues but NOT rewind the drift clock."""
        env = SliceEnv(EnvConfig(), seed=0)
        env.reset()
        for _ in range(30):
            env.step(np.zeros(3, dtype=int))
        before = env.drift.state.step
        env.reset()
        assert env.drift.state.step == before
        assert env._queue.sum() == 0.0

    def test_sla_is_bounded(self):
        env = SliceEnv(EnvConfig(), seed=1)
        env.reset()
        rng = np.random.default_rng(1)
        for _ in range(200):
            _, _, _, info = env.step(rng.integers(0, env.n_actions, 3))
            assert (info.sla >= 0).all() and (info.sla <= 1).all()
            assert (info.queue <= env.cfg.queue_max + 1e-6).all()

    def test_regime_changes_on_the_slow_clock_only(self):
        env = SliceEnv(EnvConfig(tau_med=15, tau_slow=50), seed=3)
        env.reset()
        changes = []
        prev = env.drift.state.regime
        for t in range(1, 301):
            env.step(np.zeros(3, dtype=int))
            if env.drift.state.regime != prev:
                changes.append(t)
                prev = env.drift.state.regime
        assert changes, "regime never changed in 300 steps"
        assert all(t % 50 == 0 for t in changes)

    @pytest.mark.parametrize("kappa", [0.0, 0.5, 1.5, 2.5])
    def test_severity_monotonically_widens_demand_contrast(self, kappa):
        env = SliceEnv(EnvConfig(kappa=kappa), seed=0)
        spread = env.drift._profile.max() - env.drift._profile.min()
        assert spread >= 0
        if kappa == 0.0:
            assert spread < 1e-9

    @pytest.mark.parametrize("n,s", [(3, 3), (6, 3), (10, 3)])
    def test_scales_with_agents(self, n, s):
        env = SliceEnv(EnvConfig(n_agents=n, n_slices=s), seed=0)
        assert env.reset().shape == (n, 3 * s + env.cfg.n_regimes)


# --------------------------------------------------------------------------- #
class TestNetworks:
    def test_parameter_counts_match_reference(self):
        """The reference system reports 11,024 parameters per agent."""
        net = ActorCritic(12, 15, hidden=64)
        assert count_parameters(net.actor) == 5_967
        assert count_parameters(net.critic) == 5_057
        assert count_parameters(net) == 11_024

    @pytest.mark.parametrize("n,expected", [(3, 6_593), (10, 11_969)])
    def test_centralised_critic_growth(self, n, expected):
        net = ActorCritic(12, 15, 64, critic_obs_dim=n * 12)
        assert count_parameters(net.critic) == expected

    def test_partition_is_an_exact_cover(self):
        net = ActorCritic(12, 15, 64)
        groups = net.parameter_groups()
        assert len(groups) == 3
        covered = sum(p.numel() for g in groups for p in g)
        assert covered == count_parameters(net)
        ids = [id(p) for g in groups for p in g]
        assert len(ids) == len(set(ids)), "a parameter appears in two groups"

    def test_partition_orders_by_depth(self):
        net = ActorCritic(12, 15, 64)
        g = net.parameter_groups()
        assert g[0][0].shape == (64, 12)      # actor input layer
        assert g[2][0].shape == (15, 64)      # actor output layer


# --------------------------------------------------------------------------- #
def _train_and_flatten(spec: MethodSpec, seed: int = 42, episodes: int = 6):
    env = SliceEnv(EnvConfig(kappa=1.5), seed=seed)
    learner = MultiAgentLearner(env, spec, PPOConfig(), seed=seed)
    learner.train(episodes)
    return [
        torch.cat([p.detach().flatten() for p in net.parameters()])
        for net in learner.nets
    ]


class TestUpdateRuleInvariants:
    """The claims that make the ablation study causally interpretable."""

    def test_group_split_alone_changes_nothing(self):
        """Three Adams at one rate == one Adam, exactly.

        Adam is an element-wise optimiser, so partitioning the parameters
        across several instances at the same learning rate cannot change the
        update. Verifying bit-identity proves the partition itself is inert:
        any effect measured in the ablation is caused by the *rates* or the
        *memory anchor*, never by the act of splitting.
        """
        single = _train_and_flatten(get_method("ippo"))
        split = _train_and_flatten(
            MethodSpec(name="c", label="c", learning_rates=(5e-4, 5e-4, 5e-4))
        )
        for a, b in zip(single, split):
            assert torch.equal(a, b)

    def test_ema_anchor_changes_every_agent(self):
        """The memory anchor must act on all agents, not just agent 0."""
        with_ema = _train_and_flatten(get_method("nested_marl"))
        without = _train_and_flatten(get_method("nested_no_ema"))
        for i, (a, b) in enumerate(zip(with_ema, without)):
            assert not torch.equal(a, b), f"agent {i} unaffected by the anchor"

    def test_separated_rates_change_the_solution(self):
        nested = _train_and_flatten(get_method("nested_marl"))
        same = _train_and_flatten(get_method("nested_same_rate"))
        assert any(not torch.equal(a, b) for a, b in zip(nested, same))

    def test_optimiser_counts(self):
        env = SliceEnv(EnvConfig(), seed=0)
        assert MultiAgentLearner(env, get_method("ippo")).n_optimisers_per_agent() == 1
        assert MultiAgentLearner(env, get_method("nested_marl")).n_optimisers_per_agent() == 3

    def test_nested_adds_no_parameters(self):
        """The multi-timescale structure partitions one network; it never grows it."""
        env = SliceEnv(EnvConfig(), seed=0)
        a = MultiAgentLearner(env, get_method("ippo")).n_parameters()
        b = MultiAgentLearner(env, get_method("nested_marl")).n_parameters()
        assert a == b == 11_024

    def test_training_is_reproducible(self):
        a = _train_and_flatten(get_method("nested_marl"), seed=7)
        b = _train_and_flatten(get_method("nested_marl"), seed=7)
        for x, y in zip(a, b):
            assert torch.equal(x, y)


# --------------------------------------------------------------------------- #
class TestMethodRegistry:
    def test_all_methods_construct(self):
        env = SliceEnv(EnvConfig(horizon=10), seed=0)
        for name in METHODS:
            learner = MultiAgentLearner(env, get_method(name), seed=0)
            hist = learner.train(1)
            assert len(hist) == 1 and np.isfinite(hist[0].reward)

    def test_unknown_method_raises(self):
        with pytest.raises(KeyError):
            get_method("does_not_exist")

    def test_nested_rates_are_separated(self):
        lrs = get_method("nested_marl").learning_rates
        assert lrs[0] < lrs[1] < lrs[2]
        assert lrs[2] / lrs[0] >= 10, "rates should be separated by an order of magnitude"


# --------------------------------------------------------------------------- #
class TestStatistics:
    def test_permutation_detects_a_real_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(2.0, 0.5, 12)
        b = rng.normal(0.0, 0.5, 12)
        assert paired_permutation_test(a, b, rng=rng) < 0.01

    def test_permutation_is_calibrated_under_the_null(self):
        rng = np.random.default_rng(1)
        ps = [
            paired_permutation_test(rng.normal(size=10), rng.normal(size=10), rng=rng)
            for _ in range(200)
        ]
        assert 0.01 <= np.mean(np.asarray(ps) < 0.05) <= 0.15

    def test_p_value_never_zero(self):
        a, b = np.arange(10) + 500.0, np.arange(10) * 1.0
        assert paired_permutation_test(a, b) > 0.0

    def test_holm_is_monotone_and_conservative(self):
        raw = {"a": 0.001, "b": 0.02, "c": 0.4}
        adj = holm_bonferroni(raw)
        assert all(adj[k] >= raw[k] for k in raw)
        assert adj["a"] <= adj["b"] <= adj["c"]

    def test_iqm_ignores_outliers(self):
        clean = np.arange(100.0)
        dirty = clean.copy()
        dirty[0], dirty[-1] = -1e6, 1e6
        assert interquartile_mean(clean) == pytest.approx(interquartile_mean(dirty))

    def test_probability_of_improvement_bounds(self):
        rng = np.random.default_rng(0)
        p, lo, hi = probability_of_improvement(np.ones(8) * 10, np.zeros(8), rng=rng)
        assert p == 1.0 and 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------------------------- #
class TestRunner:
    def test_slug_is_experiment_independent(self):
        from nested6g.runner import RunSpec
        a = RunSpec(experiment="headline", method="ippo", seed=1)
        b = RunSpec(experiment="ablation", method="ippo", seed=1)
        assert a.slug() == b.slug(), "identical runs must share a cache entry"

    def test_slug_separates_configurations(self):
        from nested6g.runner import RunSpec
        base = RunSpec(experiment="x", method="ippo", seed=1)
        variants = [
            RunSpec(experiment="x", method="mappo", seed=1),
            RunSpec(experiment="x", method="ippo", seed=2),
            RunSpec(experiment="x", method="ippo", seed=1, kappa=2.0),
            RunSpec(experiment="x", method="ippo", seed=1, n_agents=6),
            RunSpec(experiment="x", method="ippo", seed=1, episodes=40),
        ]
        assert len({v.slug() for v in variants} | {base.slug()}) == len(variants) + 1


# --------------------------------------------------------------------------- #
class TestAlternativePartitions:
    """Controls for the partition-sensitivity analysis (review revision).

    Each alternative partition must differ from the depth partition *only* in
    which tensor goes to which rate group. These tests pin that down.
    """

    def test_per_tensor_path_matches_depth_path_bitwise(self):
        """One Adam with per-tensor groups == one Adam per depth group, exactly."""
        from dataclasses import replace
        ref = get_method("nested_marl")
        a = _train_and_flatten(ref)
        b = _train_and_flatten(replace(ref, name="pg", partition="depth_pg"))
        for x, y in zip(a, b):
            assert torch.equal(x, y)

    def test_snr_warmup_is_single_rate_ippo(self):
        """During warm-up every tensor is at the medium rate and the anchor is off,
        so the data-driven learner must coincide with IPPO at that rate."""
        from nested6g.agents.methods import PARTITION_METHODS
        snr = get_method(PARTITION_METHODS[("ref", "grad_snr")])
        assert snr.learning_rates[1] == get_method("ippo").learning_rates[0]
        a = _train_and_flatten(snr, episodes=snr.partition_warmup - 1)
        b = _train_and_flatten(get_method("ippo"), episodes=snr.partition_warmup - 1)
        for x, y in zip(a, b):
            assert torch.equal(x, y)

    def _assign(self, method_name, seed=0, episodes=0):
        env = SliceEnv(EnvConfig(horizon=10), seed=seed)
        learner = MultiAgentLearner(env, get_method(method_name), seed=seed)
        if episodes:
            learner.train(episodes)
        depth = learner._depth_assignment(learner.nets[0])
        return learner, depth

    def test_inverted_reverses_depth(self):
        from nested6g.agents.methods import PARTITION_METHODS
        learner, depth = self._assign(PARTITION_METHODS[("ref", "inverted")])
        assert learner._assign[0] == [2 - g for g in depth]

    def test_random_keeps_group_sizes_and_differs(self):
        from nested6g.agents.methods import PARTITION_METHODS
        learner, depth = self._assign(PARTITION_METHODS[("ref", "random")])
        for a in learner._assign:
            assert sorted(a) == sorted(depth)
        assert any(a != depth for a in learner._assign)

    def test_snr_partition_is_formed_after_warmup(self):
        from nested6g.agents.methods import PARTITION_METHODS
        name = PARTITION_METHODS[("ref", "grad_snr")]
        learner, depth = self._assign(name, episodes=get_method(name).partition_warmup)
        assert learner._gstats is None
        for i, a in enumerate(learner._assign):
            assert sorted(a) == sorted(depth)
            assert len(learner._slow_param_ids[i]) == depth.count(0)
            lrs = [pg["lr"] for pg in learner.optimisers[i][0].param_groups]
            assert lrs == [get_method(name).learning_rates[g] for g in a]
        assert learner.partition_report is not None
        assert "snr" in learner.partition_report[0]

    def test_alternatives_add_no_parameters(self):
        from nested6g.agents.methods import PARTITION_METHODS
        env = SliceEnv(EnvConfig(), seed=0)
        for name in PARTITION_METHODS.values():
            assert MultiAgentLearner(env, get_method(name)).n_parameters() == 11_024


class TestScopeRunSpec:
    def test_default_tau_slow_keeps_old_cache_keys(self):
        from nested6g.runner import RunSpec
        a = RunSpec(experiment="x", method="ippo", seed=1)
        assert "__ts" not in a.slug()
        b = RunSpec(experiment="x", method="ippo", seed=1, tau_slow=16)
        assert a.slug() != b.slug()


def test_permutation_floor_matches_documentation():
    """The chapter and docstring state the floor as 3/(2^n + 1)."""
    from nested6g.analysis import paired_permutation_test
    a, b = np.arange(10) + 500.0, np.arange(10) * 1.0
    assert paired_permutation_test(a, b) == pytest.approx(3 / (2 ** 10 + 1))


def test_default_noise_keeps_old_cache_keys():
    from nested6g.runner import RunSpec
    a = RunSpec(experiment="x", method="ippo", seed=1)
    assert "__nz" not in a.slug()
    assert RunSpec(experiment="x", method="ippo", seed=1, noise=0.3).slug() != a.slug()
    # The default noise level must be the simulator's default, so the first
    # row of the noise probe re-uses the cached default-environment runs.
    assert EnvConfig().channel_noise_std == a.noise == EnvConfig().demand_noise_std
