"""DQN 学習（Double DQN + Dueling + n-step、一様な経験再生、環境ごとに違う ε）。

    python -m coinrun_dt.dqn.train --config configs/cpu/dqn.yaml

PPO とは別の種類のエージェントで、ステージごとの成功率（難易度の参照値）を取るために使う。
DT の学習データは PPO から作っているので、PPO の成功率だけを参照値にすると循環になる。
save_every_steps ごとに <out_dir>/step_XXXXXXXXX.pt を保存する。
resume=true なら最新のチェックポイントから続ける。経験再生のバッファは保存しないので、
再開直後は learning_starts ぶん貯め直してから学習に戻る。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from coinrun_dt.config import load_config
from coinrun_dt.envs import OBS_SHAPE, make_env
from coinrun_dt.dqn.model import QNetwork
from coinrun_dt.ppo.model import list_checkpoints
from coinrun_dt.utils import Logger, get_device, set_seed


@dataclass
class DQNConfig:
    total_steps: int = 10_000_000
    num_envs: int = 64
    buffer_steps: int = 2048        # 環境ごとに保持するステップ数（全体で buffer_steps × num_envs 遷移）
    learning_starts: int = 100_000  # この遷移数が貯まるまでは学習しない
    train_every: int = 1            # 何回の環境ステップ（num_envs 遷移）ごとに更新するか
    updates_per_train: int = 1
    batch_size: int = 128
    lr: float = 2.5e-4
    adam_eps: float = 1.5e-4
    gamma: float = 0.99
    n_step: int = 3
    target_update_every: int = 2000  # 勾配ステップ数
    # ε は eps_start から eps_decay_steps かけて、環境ごとの下限 eps_base^(1 + eps_alpha * i/(N-1)) まで下げる
    # （Ape-X と同じ。全環境を同じ小さな ε にすると、学習の初期に同じ行動を繰り返して動かなくなりやすい）
    eps_start: float = 1.0
    eps_decay_steps: int = 1_000_000
    eps_base: float = 0.4
    eps_alpha: float = 7.0
    reward_scale: float = 0.1        # コインの 10 を 1 にする
    max_grad_norm: float = 10.0
    distribution_mode: str = "easy"
    start_level: int = 0
    num_levels: int = 0              # 0 = ランダムなステージ
    save_every_steps: int = 1_000_000
    log_every_steps: int = 50_000
    resume: bool = True
    out_dir: str = "checkpoints/dqn"
    log_dir: str = "runs/dqn"
    seed: int = 0
    device: str = "auto"


class ReplayBuffer:
    """環境ごとに時間順に並べたリングバッファ。次の観測は obs[t+1] なので二重に持たない。

    done[t] は「時刻 t の行動でエピソードが終わった」ことを表す（gym3 の first と同じ）。
    """

    def __init__(self, capacity_steps: int, num_envs: int, n_step: int, gamma: float, seed: int = 0):
        T, N = capacity_steps, num_envs
        self.obs = np.zeros((T, N) + OBS_SHAPE, dtype=np.uint8)
        self.act = np.zeros((T, N), dtype=np.int64)
        self.rew = np.zeros((T, N), dtype=np.float32)
        self.done = np.zeros((T, N), dtype=bool)
        self.capacity, self.num_envs = T, N
        self.n_step, self.gamma = n_step, gamma
        self.pos, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, done: np.ndarray) -> None:
        self.obs[self.pos], self.act[self.pos] = obs, act
        self.rew[self.pos], self.done[self.pos] = rew, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def num_transitions(self) -> int:
        """n ステップ先の観測まで揃っている遷移の数。"""
        return max(0, self.size - self.n_step) * self.num_envs

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """n-step リターンを計算して返す。途中でエピソードが終わったら、そこで打ち切ってブートストラップしない。"""
        T = self.capacity
        start = (self.pos - self.size) % T
        t = (start + self.rng.integers(0, self.size - self.n_step, batch_size)) % T
        e = self.rng.integers(0, self.num_envs, batch_size)
        ret = np.zeros(batch_size, dtype=np.float32)
        alive = np.ones(batch_size, dtype=np.float32)
        for k in range(self.n_step):
            tk = (t + k) % T
            ret += alive * self.gamma ** k * self.rew[tk, e]
            alive *= 1.0 - self.done[tk, e]
        return {"obs": self.obs[t, e], "act": self.act[t, e], "ret": ret,
                "next_obs": self.obs[(t + self.n_step) % T, e],
                "discount": alive * self.gamma ** self.n_step}


# この下限より ε が小さい環境の成功率を、ほぼ greedy な方策の成功率として別に記録する
LOW_EPS = 0.01


def epsilon_floor(cfg: DQNConfig) -> np.ndarray:
    i = np.arange(cfg.num_envs)
    return cfg.eps_base ** (1 + cfg.eps_alpha * i / max(1, cfg.num_envs - 1))


def epsilons_at(step: int, cfg: DQNConfig) -> np.ndarray:
    """環境ごとの ε（num_envs,）。"""
    frac = min(1.0, step / max(1, cfg.eps_decay_steps))
    return cfg.eps_start + frac * (epsilon_floor(cfg) - cfg.eps_start)


def save_checkpoint(model: QNetwork, target: QNetwork, optimizer: torch.optim.Optimizer,
                    cfg: DQNConfig, step: int, grad_steps: int) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step_{step:09d}.pt"
    torch.save({"model": model.state_dict(), "target": target.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "grad_steps": grad_steps,
                "config": asdict(cfg)}, path)
    print(f"saved {path}", flush=True)
    return path


def td_loss(model: QNetwork, target: QNetwork, batch: Dict[str, np.ndarray],
            device: torch.device) -> torch.Tensor:
    obs = torch.from_numpy(batch["obs"]).to(device)
    next_obs = torch.from_numpy(batch["next_obs"]).to(device)
    act = torch.from_numpy(batch["act"]).to(device)
    ret = torch.from_numpy(batch["ret"]).to(device)
    discount = torch.from_numpy(batch["discount"]).to(device)
    q = model(obs).gather(1, act[:, None]).squeeze(1)
    with torch.no_grad():
        next_act = model(next_obs).argmax(-1, keepdim=True)    # Double DQN: 行動は online、値は target
        next_q = target(next_obs).gather(1, next_act).squeeze(1)
    return F.smooth_l1_loss(q, ret + discount * next_q)


def train(cfg: DQNConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    env = make_env(cfg.num_envs, cfg.distribution_mode, cfg.start_level, cfg.num_levels,
                   rand_seed=cfg.seed)
    model = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, cfg.n_step, cfg.gamma, cfg.seed)
    logger = Logger(cfg.log_dir)
    rng = np.random.default_rng(cfg.seed)

    global_step, grad_steps = 0, 0
    ckpts = list_checkpoints(cfg.out_dir)
    if cfg.resume and ckpts:
        global_step, path = ckpts[-1]
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        target.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        grad_steps = ckpt["grad_steps"]
        print(f"resumed from {path} (step {global_step})", flush=True)
    resume_step = global_step
    next_save = 0 if global_step == 0 else (global_step // cfg.save_every_steps + 1) * cfg.save_every_steps
    next_log = global_step + cfg.log_every_steps

    N = cfg.num_envs
    _, obs, _ = env.observe()
    obs = obs["rgb"]
    ep_ret = np.zeros(N)
    ep_len = np.zeros(N, dtype=np.int64)
    recent_returns, recent_success, recent_lengths = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    recent_success_low = deque(maxlen=100)
    low_eps = epsilon_floor(cfg) < LOW_EPS
    losses, q_means = [], []
    t0, steps_since_log = time.time(), 0
    it = 0
    while global_step < cfg.total_steps:
        if global_step >= next_save:
            save_checkpoint(model, target, optimizer, cfg, global_step, grad_steps)
            next_save += cfg.save_every_steps

        eps = epsilons_at(global_step, cfg)
        with torch.no_grad():
            action = model(torch.from_numpy(obs).to(device)).argmax(-1).cpu().numpy()
        explore = rng.random(N) < eps
        action[explore] = rng.integers(0, model.adv_head.out_features, explore.sum())
        env.act(action.astype(np.int32))
        rew, next_obs, first = env.observe()
        infos = env.get_info()
        buffer.add(obs, action, rew * cfg.reward_scale, first)

        ep_ret += rew
        ep_len += 1
        for i in np.nonzero(first)[0]:
            recent_returns.append(ep_ret[i])
            recent_lengths.append(ep_len[i])
            recent_success.append(float(infos[i]["prev_level_complete"]))
            if low_eps[i] and eps[i] < LOW_EPS:
                recent_success_low.append(float(infos[i]["prev_level_complete"]))
            ep_ret[i], ep_len[i] = 0.0, 0
        obs = next_obs["rgb"]
        global_step += N
        steps_since_log += N
        it += 1

        # 再開直後もバッファを貯め直してから学習する
        ready = buffer.num_transitions() >= cfg.learning_starts and global_step - resume_step >= cfg.learning_starts
        if ready and it % cfg.train_every == 0:
            for _ in range(cfg.updates_per_train):
                batch = buffer.sample(cfg.batch_size)
                loss = td_loss(model, target, batch, device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                grad_steps += 1
                losses.append(loss.item())
                if grad_steps % cfg.target_update_every == 0:
                    target.load_state_dict(model.state_dict())
            with torch.no_grad():
                q_means.append(model(torch.from_numpy(batch["obs"][:32]).to(device)).max(-1).values.mean().item())

        if global_step >= next_log:
            metrics = {"train/epsilon_mean": float(eps.mean()), "train/grad_steps": grad_steps}
            if losses:
                metrics["train/loss"] = float(np.mean(losses))
                metrics["train/q_max"] = float(np.mean(q_means))
            if recent_returns:
                metrics["episode/return"] = float(np.mean(recent_returns))
                metrics["episode/success_rate"] = float(np.mean(recent_success))
                metrics["episode/length"] = float(np.mean(recent_lengths))
            if recent_success_low:
                metrics[f"episode/success_rate_eps<{LOW_EPS}"] = float(np.mean(recent_success_low))
            metrics["perf/sps"] = steps_since_log / (time.time() - t0)
            logger.log(global_step, metrics)
            losses, q_means = [], []
            t0, steps_since_log = time.time(), 0
            next_log += cfg.log_every_steps

    if global_step > resume_step:
        save_checkpoint(model, target, optimizer, cfg, global_step, grad_steps)
    logger.close()


def main(argv=None) -> None:
    train(load_config(DQNConfig, argv, description=__doc__))


if __name__ == "__main__":
    main()
