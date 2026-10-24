import numpy as np

from coinrun_dt.data.collect import EpisodeWriter
from coinrun_dt.data.dataset import TrajectoryDataset, compute_rtg
from coinrun_dt.envs import OBS_SHAPE
from coinrun_dt.runner import Episode


def make_episode(T, success, seed):
    rewards = np.zeros(T, dtype=np.float32)
    if success:
        rewards[-1] = 10.0
    obs = np.zeros((T,) + OBS_SHAPE, dtype=np.uint8)
    obs[:, 0, 0, 0] = np.arange(T)  # 時刻をピクセルに埋めて、窓の位置を確認する
    return Episode(actions=np.arange(T) % 15, rewards=rewards, level_seed=seed,
                   success=success, env_index=0, obs=obs)


def test_compute_rtg():
    np.testing.assert_allclose(compute_rtg(np.array([0.0, 0.0, 10.0])), [10.0, 10.0, 10.0])
    np.testing.assert_allclose(compute_rtg(np.array([1.0, 2.0, 3.0])), [6.0, 5.0, 3.0])


def test_dataset_windows(tmp_path):
    writer = EpisodeWriter(str(tmp_path))
    writer.write(make_episode(5, success=True, seed=1), ckpt_step=0)
    writer.write(make_episode(7, success=False, seed=2), ckpt_step=100)
    writer.close()

    K = 4
    ds = TrajectoryDataset(str(tmp_path), context_len=K, reward_scale=0.1)
    assert len(ds) == 12

    # 1本目の時刻 1 で終わる窓: 左に 2 つパディング
    item = ds[1]
    assert item["mask"].tolist() == [False, False, True, True]
    assert item["timesteps"].tolist() == [0, 0, 0, 1]
    assert item["states"][:, 0, 0, 0].tolist() == [0, 0, 0, 1]
    np.testing.assert_allclose(item["rtg"][2:, 0].numpy(), [1.0, 1.0])

    # 2本目の時刻 6 で終わる窓: パディングなし、1本目をまたがない
    item = ds[5 + 6]
    assert item["mask"].all()
    assert item["timesteps"].tolist() == [3, 4, 5, 6]
    np.testing.assert_allclose(item["rtg"][:, 0].numpy(), 0.0)

    # 2本目の先頭: 1本目の末尾を文脈に含めない
    item = ds[5]
    assert item["mask"].tolist() == [False, False, False, True]
    assert item["timesteps"].tolist() == [0, 0, 0, 0]


def test_writer_appends(tmp_path):
    for _ in range(2):
        writer = EpisodeWriter(str(tmp_path))
        writer.write(make_episode(3, success=False, seed=0), ckpt_step=0)
        writer.close()
    assert len(list((tmp_path / "episodes").glob("*.npz"))) == 2
