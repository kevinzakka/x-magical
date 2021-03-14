"""Test benchmark envs, adapted from [1] and [2].

Note that test methods below are taken mostly verbatim from [1]. The reason
we can't import their package is that their tests require mujoco_py and will
throw an error (and subsequently fail all tests) if the dependency is not
installed.

References:
    [1]: https://github.com/HumanCompatibleAI/seals
    [2]: https://github.com/qxcv/magical
"""

import gym
import numpy as np
import pytest

import xmagical

# Register environments to fill ALL_REGISTERED_ENVS.
xmagical.register_envs()
# Keep this small to make test time reasonable.
N_ROLLOUTS = 2


def make_env_fixture(skip_fn):
    def f(env_name: str):
        env = None
        try:
            env = gym.make(env_name)
            yield env
        except Exception as e:
            raise e
        finally:
            if env is not None:
                env.close()

    return f


env = pytest.fixture(make_env_fixture(skip_fn=pytest.skip))


@pytest.mark.parametrize("env_name", xmagical.ALL_REGISTERED_ENVS)
class TestEnvs:
    """Simple tests to ensure environments behave properly."""

    def test_obs(self, env):
        """Test that obs is consistent with obs_space."""

        def _check_obs(obs, obs_space):
            if obs_space.shape:
                assert obs.shape == obs_space.shape
                assert obs.dtype == obs_space.dtype
            assert obs in obs_space

        obs = env.reset()
        _check_obs(obs, env.observation_space)
        act = env.action_space.sample()
        obs, _, _, _ = env.step(act)
        _check_obs(obs, env.observation_space)

    def test_seed(self, env):
        """Test that seeding an env generates identical rollouts."""

        def get_rollout(env, actions):
            ret = [(env.reset(), None, False, {})]
            for act in actions:
                ret.append(env.step(act))
            return ret

        def assert_equal_rollout(rollout_a, rollout_b):
            for step_a, step_b in zip(rollout_a, rollout_b):
                ob_a, rew_a, done_a, info_a = step_a
                ob_b, rew_b, done_b, info_b = step_b
                np.testing.assert_equal(ob_a, ob_b)
                assert rew_a == rew_b
                assert done_a == done_b
                np.testing.assert_equal(info_a, info_b)

        env.action_space.seed(0)
        actions = [env.action_space.sample() for _ in range(10)]

        # With the same seed, should always get the same result
        seeds = env.seed(42)
        assert isinstance(seeds, list)
        assert len(seeds) > 0
        rollout_a = get_rollout(env, actions)

        env.seed(42)
        rollout_b = get_rollout(env, actions)

        assert_equal_rollout(rollout_a, rollout_b)

    def test_rollout_len(self, env):
        """Test that envs generate rollouts of the correct length."""
        try:
            env.seed(7)
            env.action_space.seed(42)
            env.reset()
            for _ in range(N_ROLLOUTS):
                done = False
                traj_len = 0
                while not done:
                    action = env.action_space.sample()
                    _, _, done, _ = env.step(action)
                    traj_len += 1
                assert traj_len == env.max_episode_steps
                env.reset()
        finally:
            env.close()
