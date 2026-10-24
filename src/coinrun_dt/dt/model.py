"""Decision Transformer（Atari 版 DT と同じ構成: CNN 状態エンコーダ + GPT 型の因果 Transformer）。

トークン列は [R_1, s_1, a_1, R_2, s_2, a_2, ...]（長さ 3K）で、s_t のトークン位置から a_t を予測する。
状態の埋め込みはフレームごとに独立して計算できるので、embed_states と forward_from_embeddings に
分けてある。ロールアウト時は埋め込みをキャッシュし、CNN を毎ステップ K 回まわさずに済むようにする。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from coinrun_dt.envs import MAX_EPISODE_STEPS, NUM_ACTIONS


@dataclass
class DTConfig:
    num_actions: int = NUM_ACTIONS
    context_len: int = 30
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 128
    dropout: float = 0.1
    max_timestep: int = MAX_EPISODE_STEPS


class StateEncoder(nn.Module):
    """64x64x3 → n_embd（Atari DT の conv 8/4 → 4/2 → 3/1）。"""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),    # 64 → 15
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),   # 15 → 6
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),   # 6 → 4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, n_embd), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 64, 64, 3) uint8"""
        return self.net(obs.permute(0, 3, 1, 2).float() / 255.0)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, L, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, L, C)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: DTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        return x + self.mlp(self.ln2(x))


class DecisionTransformer(nn.Module):
    def __init__(self, cfg: DTConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.n_embd
        self.state_encoder = StateEncoder(C)
        self.rtg_embed = nn.Sequential(nn.Linear(1, C), nn.Tanh())
        self.action_embed = nn.Sequential(nn.Embedding(cfg.num_actions, C), nn.Tanh())
        self.time_embed = nn.Embedding(cfg.max_timestep + 1, C)
        self.pos_embed = nn.Parameter(torch.zeros(1, 3 * cfg.context_len, C))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(C)
        self.action_head = nn.Linear(C, cfg.num_actions, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def embed_states(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (..., 64, 64, 3) uint8 → (..., n_embd)"""
        lead = obs.shape[:-3]
        emb = self.state_encoder(obs.reshape((-1,) + obs.shape[-3:]))
        return emb.reshape(lead + (emb.shape[-1],))

    def forward(self, states: torch.Tensor, actions: torch.Tensor, rtg: torch.Tensor,
                timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        states: (B, K, 64, 64, 3) uint8, actions: (B, K) long, rtg: (B, K, 1) float,
        timesteps: (B, K) long, mask: (B, K) bool（True = 有効、左詰めパディング）
        → 行動ロジット (B, K, A)。位置 t のロジットは s_t までの情報だけで決まる。
        """
        return self.forward_from_embeddings(self.embed_states(states), actions, rtg, timesteps, mask)

    def forward_from_embeddings(self, state_emb: torch.Tensor, actions: torch.Tensor,
                                rtg: torch.Tensor, timesteps: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, C = state_emb.shape
        assert K <= self.cfg.context_len
        te = self.time_embed(timesteps.clamp(0, self.cfg.max_timestep))
        tokens = torch.stack([self.rtg_embed(rtg) + te, state_emb + te,
                              self.action_embed(actions) + te], dim=2).reshape(B, 3 * K, C)
        x = self.drop(tokens + self.pos_embed[:, :3 * K])
        attn_mask = self._attention_mask(mask, B, K, x.device)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_f(x)
        return self.action_head(x[:, 1::3])

    @staticmethod
    def _attention_mask(mask: Optional[torch.Tensor], B: int, K: int,
                        device: torch.device) -> torch.Tensor:
        """因果マスク + パディング位置への注意の禁止。True = 注意してよい。"""
        L = 3 * K
        causal = torch.ones(L, L, dtype=torch.bool, device=device).tril()
        if mask is None:
            return causal
        token_mask = mask.repeat_interleave(3, dim=1)                 # (B, L)
        m = causal.unsqueeze(0) & token_mask.unsqueeze(1)
        m = m | torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)  # 全マスク行の NaN 防止
        return m.unsqueeze(1)                                         # (B, 1, L, L)


def save_dt(path: Union[str, Path], model: DecisionTransformer, reward_scale: float,
            train_config: dict, iteration: int,
            optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": asdict(model.cfg),
                "reward_scale": reward_scale, "train_config": train_config, "iter": iteration,
                "optimizer": optimizer.state_dict() if optimizer is not None else None}, path)


def load_dt(path: Union[str, Path], device: torch.device) -> Tuple[DecisionTransformer, dict]:
    ckpt = torch.load(path, map_location=device)
    model = DecisionTransformer(DTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
