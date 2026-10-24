"""DT のオンライン実行と、与えた軌跡に沿った DT の行動分布の計算。"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch.distributions import Categorical

from coinrun_dt.dt.model import DecisionTransformer


class DTPolicy:
    """目標リターンを条件に、N 並列の環境で DT を動かす（run_episodes 用）。

    各環境の直近 K ステップ分の (状態埋め込み, 行動, RTG, timestep) を右詰めで持つ。
    target_return は環境ごとに変えられる（スカラーまたは (N,) 配列）。
    """

    def __init__(self, model: DecisionTransformer, num_envs: int,
                 target_return: Union[float, np.ndarray], reward_scale: float,
                 device: torch.device, sample: bool = True, temperature: float = 1.0):
        self.model = model
        self.device = device
        self.sample = sample
        self.temperature = temperature
        self.reward_scale = reward_scale
        self.K = model.cfg.context_len
        self.max_timestep = model.cfg.max_timestep
        N, K, C = num_envs, self.K, model.cfg.n_embd

        target = np.broadcast_to(np.asarray(target_return, dtype=np.float32), (N,))
        self.target = torch.as_tensor(target * reward_scale, device=device)
        self.state_emb = torch.zeros((N, K, C), device=device)
        self.actions = torch.zeros((N, K), dtype=torch.long, device=device)
        self.rtg = torch.zeros((N, K, 1), device=device)
        self.timesteps = torch.zeros((N, K), dtype=torch.long, device=device)
        self.length = torch.zeros(N, dtype=torch.long, device=device)
        self.cur_rtg = self.target.clone()
        self.t = torch.zeros(N, dtype=torch.long, device=device)
        self.last_logits = None

    def reset(self, env_indices: np.ndarray) -> None:
        idx = torch.as_tensor(np.asarray(env_indices), dtype=torch.long, device=self.device)
        self.length[idx] = 0
        self.t[idx] = 0
        self.cur_rtg[idx] = self.target[idx]

    @staticmethod
    def _push(buf: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.cat([buf[:, 1:], value.unsqueeze(1)], dim=1)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
        self.state_emb = self._push(self.state_emb, self.model.embed_states(obs_t))
        self.rtg = self._push(self.rtg, self.cur_rtg.unsqueeze(-1))
        self.timesteps = self._push(self.timesteps, self.t)
        self.actions = self._push(self.actions, torch.zeros_like(self.t))  # 現在の行動は未定（因果マスクで見えない）
        self.length = (self.length + 1).clamp(max=self.K)
        mask = torch.arange(self.K, device=self.device).unsqueeze(0) >= (self.K - self.length).unsqueeze(1)

        logits = self.model.forward_from_embeddings(self.state_emb, self.actions, self.rtg,
                                                    self.timesteps, mask)[:, -1]
        self.last_logits = logits
        if self.sample:
            action = Categorical(logits=logits / self.temperature).sample()
        else:
            action = logits.argmax(-1)
        self.actions[:, -1] = action
        self.t = (self.t + 1).clamp(max=self.max_timestep)
        return action.cpu().numpy()

    def observe(self, rew: np.ndarray, first: np.ndarray) -> None:
        self.cur_rtg -= torch.as_tensor(rew, dtype=torch.float32, device=self.device) * self.reward_scale


@torch.no_grad()
def dt_logits_along_trajectory(model: DecisionTransformer, obs: np.ndarray, actions: np.ndarray,
                               rewards: np.ndarray, target_return: float, reward_scale: float,
                               device: torch.device, batch_size: int = 256) -> torch.Tensor:
    """軌跡の各時刻 t で、DT がその時点の文脈から出す行動ロジット (T, A) を返す。

    RTG は target_return から実際に得た報酬を引いていく（オンライン実行時と同じ条件付け）。
    DTPolicy で同じ軌跡をたどったときの last_logits と一致する。
    """
    model.eval()
    K = model.cfg.context_len
    T = len(actions)
    obs_t = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
    emb = torch.cat([model.embed_states(obs_t[i:i + batch_size]) for i in range(0, T, batch_size)])

    rew = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    rtg = (target_return - torch.cat([rew.new_zeros(1), rew.cumsum(0)[:-1]])) * reward_scale
    ts = torch.arange(T, device=device).clamp(max=model.cfg.max_timestep)
    act = torch.as_tensor(actions, dtype=torch.long, device=device)

    pad = K - 1
    emb_p = torch.cat([emb.new_zeros(pad, emb.shape[1]), emb])
    rtg_p = torch.cat([rtg.new_zeros(pad), rtg])
    ts_p = torch.cat([ts.new_zeros(pad), ts])
    act_p = torch.cat([act.new_zeros(pad), act])
    valid_p = torch.cat([torch.zeros(pad, dtype=torch.bool, device=device),
                         torch.ones(T, dtype=torch.bool, device=device)])

    out = []
    offsets = torch.arange(K, device=device)
    for start in range(0, T, batch_size):
        t = torch.arange(start, min(start + batch_size, T), device=device)
        idx = t.unsqueeze(1) + offsets.unsqueeze(0)   # パディング込みの配列で [t, t+K) = 元の [t-K+1, t]
        logits = model.forward_from_embeddings(emb_p[idx], act_p[idx], rtg_p[idx].unsqueeze(-1),
                                               ts_p[idx], valid_p[idx])
        out.append(logits[:, -1])
    return torch.cat(out)
