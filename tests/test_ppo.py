import pytest

from coinrun_dt.ppo.train import PPOConfig, lr_at


def test_lr_schedule():
    const = PPOConfig(lr=1e-3, total_steps=1000, lr_schedule="constant")
    assert lr_at(0, const) == lr_at(900, const) == 1e-3
    linear = PPOConfig(lr=1e-3, total_steps=1000, lr_schedule="linear")
    assert lr_at(0, linear) == pytest.approx(1e-3)
    assert lr_at(250, linear) == pytest.approx(7.5e-4)
    assert lr_at(1000, linear) == 0.0 and lr_at(1200, linear) == 0.0
    with pytest.raises(ValueError):
        lr_at(0, PPOConfig(lr_schedule="cosine"))
