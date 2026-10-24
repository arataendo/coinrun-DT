import numpy as np
import pytest

pytest.importorskip("procgen")

from coinrun_dt.envs import NUM_ACTIONS, OBS_SHAPE, make_env, make_level_env
from coinrun_dt.runner import run_episodes


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env_indices):
        pass

    def act(self, obs):
        return self.rng.integers(0, NUM_ACTIONS, size=len(obs))

    def observe(self, rew, first):
        pass


def test_observation_shape():
    env = make_env(2)
    _, obs, first = env.observe()
    assert obs["rgb"].shape == (2,) + OBS_SHAPE
    assert obs["rgb"].dtype == np.uint8
    assert first.all()


def test_level_env_episodes_have_the_requested_seed():
    env = make_level_env(123, 3)
    episodes = run_episodes(env, RandomPolicy(), episodes_per_env=1, record_obs=True)
    assert len(episodes) == 3
    for ep in episodes:
        assert ep.level_seed == 123
        assert ep.obs.shape == (ep.length,) + OBS_SHAPE
        assert ep.success == (ep.ret > 0)
