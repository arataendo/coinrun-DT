import numpy as np
import torch

from coinrun_dt.dt.model import DecisionTransformer, DTConfig
from coinrun_dt.dt.rollout import DTPolicy, dt_logits_along_trajectory
from coinrun_dt.envs import NUM_ACTIONS, OBS_SHAPE


def tiny_model(K=5):
    torch.manual_seed(0)
    return DecisionTransformer(DTConfig(context_len=K, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()


def random_batch(B=2, K=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        states=torch.randint(0, 256, (B, K) + OBS_SHAPE, dtype=torch.uint8, generator=g),
        actions=torch.randint(0, NUM_ACTIONS, (B, K), generator=g),
        rtg=torch.rand(B, K, 1, generator=g),
        timesteps=torch.arange(K).repeat(B, 1),
        mask=torch.ones(B, K, dtype=torch.bool),
    )


def test_output_shape():
    model = tiny_model()
    logits = model(**random_batch())
    assert logits.shape == (2, 5, NUM_ACTIONS)


@torch.no_grad()
def test_causality():
    model = tiny_model()
    batch = random_batch()
    base = model(**batch)

    # 時刻 3 の状態を変えても、時刻 0..2 のロジットは変わらない
    changed = dict(batch, states=batch["states"].clone())
    changed["states"][:, 3] = 255 - changed["states"][:, 3]
    out = model(**changed)
    torch.testing.assert_close(out[:, :3], base[:, :3])
    assert not torch.allclose(out[:, 3], base[:, 3])

    # 時刻 2 の行動は、時刻 2 のロジット（それを予測する位置）に影響しない
    changed = dict(batch, actions=batch["actions"].clone())
    changed["actions"][:, 2] = (changed["actions"][:, 2] + 1) % NUM_ACTIONS
    out = model(**changed)
    torch.testing.assert_close(out[:, :3], base[:, :3])


@torch.no_grad()
def test_padding_is_ignored():
    model = tiny_model()
    batch = random_batch()
    batch["mask"][:, :2] = False
    base = model(**batch)
    changed = dict(batch, states=batch["states"].clone(), rtg=batch["rtg"].clone())
    changed["states"][:, :2] = 0
    changed["rtg"][:, :2] = 5.0
    out = model(**changed)
    torch.testing.assert_close(out[:, 2:], base[:, 2:])


@torch.no_grad()
def test_policy_matches_trajectory_logits():
    """オンライン実行（DTPolicy）と軌跡に沿った一括計算が同じロジットを出す。K を超える長さで確認。"""
    K, T = 4, 9
    model = tiny_model(K)
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 256, (T, 1) + OBS_SHAPE, dtype=np.uint8)
    rewards = np.zeros(T, dtype=np.float32)
    rewards[5] = 10.0

    policy = DTPolicy(model, 1, target_return=10.0, reward_scale=0.1, device=torch.device("cpu"), sample=False)
    policy.reset(np.array([0]))
    actions, online = [], []
    for t in range(T):
        actions.append(int(policy.act(obs[t])[0]))
        online.append(policy.last_logits[0])
        policy.observe(rewards[t:t + 1], np.array([False]))

    batched = dt_logits_along_trajectory(model, obs[:, 0], np.array(actions), rewards, 10.0, 0.1,
                                         torch.device("cpu"), batch_size=4)
    torch.testing.assert_close(torch.stack(online), batched, rtol=1e-4, atol=1e-5)
