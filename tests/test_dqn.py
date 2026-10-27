import numpy as np
import pytest
import torch

from coinrun_dt.dqn.model import DQNPolicy, QNetwork
from coinrun_dt.dqn.train import DQNConfig, ReplayBuffer, epsilon_floor, epsilons_at, td_loss
from coinrun_dt.envs import NUM_ACTIONS, OBS_SHAPE


def fill(buffer, rewards, dones):
    """環境1つ。観測には時刻を書き込んでおき、どの時刻が next_obs になったかを確かめられるようにする。"""
    for t, (r, d) in enumerate(zip(rewards, dones)):
        obs = np.full((1,) + OBS_SHAPE, t, dtype=np.uint8)
        buffer.add(obs, np.array([t % NUM_ACTIONS]), np.array([r], dtype=np.float32), np.array([d]))


def test_n_step_return_and_bootstrap():
    buf = ReplayBuffer(capacity_steps=16, num_envs=1, n_step=3, gamma=0.5)
    #          t:  0    1    2    3    4    5
    fill(buf, [1.0, 2.0, 4.0, 8.0, 0.0, 0.0], [0, 0, 0, 1, 0, 0])
    assert buf.num_transitions() == 3
    batch = buf.sample(200)
    t = batch["obs"][:, 0, 0, 0].astype(int)
    assert set(t) == {0, 1, 2}
    expected_ret = {0: 1 + 0.5 * 2 + 0.25 * 4, 1: 2 + 0.5 * 4 + 0.25 * 8, 2: 4 + 0.5 * 8}
    for i, ti in enumerate(t):
        assert batch["ret"][i] == pytest.approx(expected_ret[ti])
        assert batch["next_obs"][i, 0, 0, 0] == ti + 3
        assert batch["act"][i] == ti
    # t=0 は途中で終わらないので gamma^3 でブートストラップ、t=1, 2 は t=3 で終わるのでしない
    disc = dict(zip(t, batch["discount"]))
    assert disc[0] == pytest.approx(0.125)
    assert disc[1] == 0 and disc[2] == 0


def test_ring_buffer_wraps():
    buf = ReplayBuffer(capacity_steps=4, num_envs=1, n_step=1, gamma=0.9)
    fill(buf, [float(i) for i in range(10)], [0] * 10)
    batch = buf.sample(100)
    t = batch["obs"][:, 0, 0, 0].astype(int)
    assert set(t) == {6, 7, 8}          # 残っているのは 6〜9、次の観測まで揃うのは 6〜8
    assert np.all(batch["next_obs"][:, 0, 0, 0] == t + 1)
    assert np.allclose(batch["ret"], t)


def test_q_network_and_loss():
    model, target = QNetwork(), QNetwork()
    obs = torch.randint(0, 255, (4,) + OBS_SHAPE, dtype=torch.uint8)
    assert model(obs).shape == (4, NUM_ACTIONS)
    buf = ReplayBuffer(capacity_steps=8, num_envs=2, n_step=2, gamma=0.99)
    for t in range(8):
        buf.add(np.random.randint(0, 255, (2,) + OBS_SHAPE, dtype=np.uint8),
                np.array([1, 2]), np.array([0.0, 1.0], dtype=np.float32), np.array([False, t == 5]))
    loss = td_loss(model, target, buf.sample(8), torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)


def test_epsilon_schedule_and_policy():
    cfg = DQNConfig(num_envs=8, eps_start=1.0, eps_decay_steps=100, eps_base=0.4, eps_alpha=7.0)
    floor = epsilon_floor(cfg)
    assert floor[0] == pytest.approx(0.4) and floor[-1] == pytest.approx(0.4 ** 8)
    assert np.all(np.diff(floor) < 0)
    assert np.allclose(epsilons_at(0, cfg), 1.0)
    assert np.allclose(epsilons_at(50, cfg), (1.0 + floor) / 2)
    assert np.allclose(epsilons_at(1000, cfg), floor)
    policy = DQNPolicy(QNetwork(), torch.device("cpu"), epsilon=0.0)
    actions = policy.act(np.zeros((3,) + OBS_SHAPE, dtype=np.uint8))
    assert actions.shape == (3,) and len(set(actions)) == 1   # greedy なので同じ観測には同じ行動
