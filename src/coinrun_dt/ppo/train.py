"""PPO 学習（DT 用のデータを作るための方策）。

    python -m coinrun_dt.ppo.train --config configs/ppo_full.yaml

save_every_steps ごとに checkpoints/ppo/step_XXXXXXXXX.pt を保存する。
学習途中のチェックポイントは、後でデータ収集に使う（上手い方策と下手な方策を混ぜる）。
resume=true なら、out_dir の最新のチェックポイントから続ける（環境の状態は引き継がない）。
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from coinrun_dt.config import load_config
from coinrun_dt.envs import OBS_SHAPE, make_env
from coinrun_dt.ppo.model import ActorCritic, list_checkpoints
from coinrun_dt.utils import Logger, get_device, set_seed


@dataclass
class PPOConfig:
    total_steps: int = 25_000_000
    num_envs: int = 64
    num_steps: int = 256
    lr: float = 5e-4
    epochs: int = 3
    num_minibatches: int = 8
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_reward: bool = True
    distribution_mode: str = "easy"
    start_level: int = 0
    num_levels: int = 0          # 0 = ランダムなステージ
    save_every_steps: int = 1_000_000
    resume: bool = True          # out_dir にチェックポイントがあれば、最新のものから続ける
    out_dir: str = "checkpoints/ppo"
    log_dir: str = "runs/ppo"
    seed: int = 0
    device: str = "auto"


class RunningMeanStd:
    def __init__(self, eps: float = 1e-4):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, n = float(x.mean()), float(x.var()), x.size
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m2 = self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total
        self.var = m2 / total
        self.count = total


class RewardNormalizer:
    """割引リターンの標準偏差で報酬を割る（OpenAI baselines の VecNormalize と同じ）。"""

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, eps: float = 1e-8):
        self.ret = np.zeros(num_envs)
        self.rms = RunningMeanStd()
        self.gamma, self.clip, self.eps = gamma, clip, eps

    def __call__(self, rew: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + rew
        self.rms.update(self.ret)
        out = np.clip(rew / math.sqrt(self.rms.var + self.eps), -self.clip, self.clip)
        self.ret[done] = 0.0
        return out


def save_checkpoint(model: ActorCritic, optimizer: torch.optim.Optimizer,
                    reward_norm: Optional[RewardNormalizer], cfg: PPOConfig, step: int) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step_{step:09d}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "reward_rms": vars(reward_norm.rms) if reward_norm is not None else None,
                "step": step, "config": asdict(cfg)}, path)
    print(f"saved {path}", flush=True)
    return path


def try_resume(model: ActorCritic, optimizer: torch.optim.Optimizer,
               reward_norm: Optional[RewardNormalizer], cfg: PPOConfig, device: torch.device) -> int:
    ckpts = list_checkpoints(cfg.out_dir)
    if not cfg.resume or not ckpts:
        return 0
    step, path = ckpts[-1]
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if reward_norm is not None and ckpt.get("reward_rms"):
        vars(reward_norm.rms).update(ckpt["reward_rms"])
    print(f"resumed from {path} (step {step})", flush=True)
    return step


def train(cfg: PPOConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    env = make_env(cfg.num_envs, cfg.distribution_mode, cfg.start_level, cfg.num_levels,
                   rand_seed=cfg.seed)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    logger = Logger(cfg.log_dir)

    T, N = cfg.num_steps, cfg.num_envs
    batch_size = T * N
    minibatch_size = batch_size // cfg.num_minibatches

    obs_buf = torch.zeros((T, N) + OBS_SHAPE, dtype=torch.uint8)
    act_buf = torch.zeros((T, N), dtype=torch.long)
    logp_buf = torch.zeros((T, N))
    val_buf = torch.zeros((T, N))
    rew_buf = torch.zeros((T, N))
    done_buf = torch.zeros((T, N))

    reward_norm = RewardNormalizer(N, cfg.gamma) if cfg.normalize_reward else None
    _, obs, _ = env.observe()
    obs = obs["rgb"]
    ep_ret = np.zeros(N)
    recent_returns, recent_success, recent_lengths = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    ep_len = np.zeros(N, dtype=np.int64)

    global_step = try_resume(model, optimizer, reward_norm, cfg, device)
    next_save = 0 if global_step == 0 else (global_step // cfg.save_every_steps + 1) * cfg.save_every_steps
    num_updates = math.ceil(max(0, cfg.total_steps - global_step) / batch_size)
    for update in range(num_updates):
        if global_step >= next_save:
            save_checkpoint(model, optimizer, reward_norm, cfg, global_step)
            next_save += cfg.save_every_steps
        t0 = time.time()

        # ---- ロールアウト ----
        for t in range(T):
            obs_t = torch.from_numpy(obs)
            with torch.no_grad():
                logits, value = model(obs_t.to(device))
            dist = Categorical(logits=logits)
            action = dist.sample()
            env.act(action.cpu().numpy().astype(np.int32))
            rew, next_obs, first = env.observe()
            infos = env.get_info()

            obs_buf[t] = obs_t
            act_buf[t] = action.cpu()
            logp_buf[t] = dist.log_prob(action).cpu()
            val_buf[t] = value.cpu()

            ep_ret += rew
            ep_len += 1
            for i in np.nonzero(first)[0]:
                recent_returns.append(ep_ret[i])
                recent_lengths.append(ep_len[i])
                recent_success.append(float(infos[i]["prev_level_complete"]))
                ep_ret[i], ep_len[i] = 0.0, 0

            r = reward_norm(rew, first) if reward_norm is not None else rew
            rew_buf[t] = torch.as_tensor(r, dtype=torch.float32)
            done_buf[t] = torch.as_tensor(first, dtype=torch.float32)
            obs = next_obs["rgb"]
            global_step += N

        # ---- GAE ----
        with torch.no_grad():
            _, next_value = model(torch.from_numpy(obs).to(device))
        next_value = next_value.cpu()
        adv_buf = torch.zeros((T, N))
        last_gae = torch.zeros(N)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else val_buf[t + 1]
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * nv * not_done - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * not_done * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        b_obs = obs_buf.reshape((-1,) + OBS_SHAPE)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret = adv_buf.reshape(-1), ret_buf.reshape(-1)

        # ---- 更新 ----
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n_mb = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                logits, value = model(b_obs[idx].to(device))
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx].to(device))
                ratio = (logp - b_logp[idx].to(device)).exp()
                adv = b_adv[idx].to(device)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = -torch.min(ratio * adv,
                                     ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv).mean()
                v_loss = 0.5 * F.mse_loss(value, b_ret[idx].to(device))
                entropy = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    stats["pg_loss"] += pg_loss.item()
                    stats["v_loss"] += v_loss.item()
                    stats["entropy"] += entropy.item()
                    stats["approx_kl"] += ((ratio - 1) - ratio.log()).mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip_coef).float().mean().item()
                n_mb += 1

        metrics = {f"train/{k}": v / n_mb for k, v in stats.items()}
        if recent_returns:
            metrics["episode/return"] = float(np.mean(recent_returns))
            metrics["episode/success_rate"] = float(np.mean(recent_success))
            metrics["episode/length"] = float(np.mean(recent_lengths))
        metrics["perf/sps"] = batch_size / (time.time() - t0)
        logger.log(global_step, metrics)

    if num_updates > 0:
        save_checkpoint(model, optimizer, reward_norm, cfg, global_step)
    logger.close()


def main(argv=None) -> None:
    train(load_config(PPOConfig, argv, description=__doc__))


if __name__ == "__main__":
    main()
