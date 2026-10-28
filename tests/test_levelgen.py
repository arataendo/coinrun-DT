import numpy as np
import pytest

from coinrun_dt.envs import make_level_env
from coinrun_dt.levelgen import MT19937, generate_level, level_grid_from_env


def test_mt19937_matches_std():
    g = MT19937(5489)            # std::mt19937 の既定の seed。10000 個目の値は C++ 規格で決まっている
    for _ in range(9999):
        g()
    assert g() == 4123659995


@pytest.mark.parametrize("mode", ["easy", "hard"])
def test_generated_grid_matches_procgen(mode):
    for seed in [0, 1, 7, 42, 1000, 123456]:
        env = make_level_env(seed, 1, mode)
        real = level_grid_from_env(env.get_state()[0])
        assert np.array_equal(real, generate_level(seed, mode).grid), seed


def test_features_are_consistent():
    for seed in range(50):
        info = generate_level(seed)
        assert info.dif in (1, 2, 3)
        assert info.dif <= info.num_sections <= 2 * info.dif - 1
        assert info.num_pits == 0 or info.dif == 3    # 穴は幅 8 以上の区間にしかできない
        assert info.num_enemies == 0 or info.danger_type == "enemy"   # easy では穴の中だけ
