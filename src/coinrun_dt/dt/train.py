"""DT 学習。

    python -m coinrun_dt.dt.train --config configs/dt_full.yaml

eval_every イテレーションごとに、ランダムなステージで目標リターン eval_target_return を条件に
DT を動かし、成功率を記録する。resume=true なら最新の dt_iter_*.pt から続ける。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from coinrun_dt.config import load_config
from coinrun_dt.data.dataset import TrajectoryDataset
from coinrun_dt.dt.model import DecisionTransformer, DTConfig, save_dt
from coinrun_dt.dt.rollout import DTPolicy
from coinrun_dt.envs import make_env
from coinrun_dt.runner import run_episodes
from coinrun_dt.utils import Logger, get_device, set_seed


@dataclass
class DTTrainConfig:
    data_dir: str = "data/coinrun"
    # モデル
    context_len: int = 30
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 128
    dropout: float = 0.1
    # 最適化
    batch_size: int = 64
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_iters: int = 1000
    max_iters: int = 100_000
    grad_clip: float = 1.0
    reward_scale: float = 0.1     # CoinRun のコイン報酬 10 → RTG 1.0
    num_workers: int = 4
    # 評価
    eval_every: int = 5000
    eval_episodes: int = 64
    eval_target_return: float = 10.0
    distribution_mode: str = "easy"
    # 出力
    save_every: int = 20000
    resume: bool = True           # out_dir に dt_iter_*.pt があれば、最新のものから続ける
    out_dir: str = "checkpoints/dt"
    log_dir: str = "runs/dt"
    log_every: int = 100
    seed: int = 0
    device: str = "auto"


def make_optimizer(model: torch.nn.Module, cfg: DTTrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def lr_at(it: int, cfg: DTTrainConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def evaluate_random_levels(model: DecisionTransformer, cfg: DTTrainConfig,
                           device: torch.device, seed: int) -> dict:
    model.eval()
    env = make_env(cfg.eval_episodes, cfg.distribution_mode, num_levels=0, rand_seed=seed)
    policy = DTPolicy(model, env.num, cfg.eval_target_return, cfg.reward_scale, device)
    episodes = run_episodes(env, policy, episodes_per_env=1)
    model.train()
    return {"eval/success_rate": float(np.mean([e.success for e in episodes])),
            "eval/return": float(np.mean([e.ret for e in episodes])),
            "eval/length": float(np.mean([e.length for e in episodes]))}


def train(cfg: DTTrainConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    dataset = TrajectoryDataset(cfg.data_dir, cfg.context_len, cfg.reward_scale)
    print(f"dataset: {len(dataset)} steps, success_rate={dataset.success.mean():.3f}", flush=True)

    model = DecisionTransformer(DTConfig(context_len=cfg.context_len, n_layer=cfg.n_layer,
                                         n_head=cfg.n_head, n_embd=cfg.n_embd,
                                         dropout=cfg.dropout)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    optimizer = make_optimizer(model, cfg)
    logger = Logger(cfg.log_dir)
    out_dir = Path(cfg.out_dir)

    start_iter = 0
    ckpts = sorted(out_dir.glob("dt_iter_*.pt"))
    if cfg.resume and ckpts:
        ckpt = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"]
        print(f"resumed from {ckpts[-1]} (iter {start_iter})", flush=True)

    remaining = max(0, cfg.max_iters - start_iter)
    loader = []
    if remaining > 0:
        sampler = RandomSampler(dataset, replacement=True, num_samples=remaining * cfg.batch_size)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                            num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=cfg.num_workers > 0)

    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for it, batch in enumerate(loader, start=start_iter):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        for g in optimizer.param_groups:
            g["lr"] = lr_at(it, cfg)

        logits = model(batch["states"], batch["actions"], batch["rtg"], batch["timesteps"], batch["mask"])
        mask = batch["mask"].reshape(-1).float()
        loss_per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["actions"].reshape(-1),
                                   reduction="none")
        loss = (loss_per * mask).sum() / mask.sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        with torch.no_grad():
            acc = ((logits.argmax(-1).reshape(-1) == batch["actions"].reshape(-1)).float() * mask).sum() / mask.sum()
        running_loss += loss.item()
        running_acc += acc.item()
        n += 1

        step = it + 1
        if step % cfg.log_every == 0:
            logger.log(step, {"train/loss": running_loss / n, "train/action_acc": running_acc / n,
                              "train/lr": lr_at(it, cfg)})
            running_loss, running_acc, n = 0.0, 0.0, 0
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            logger.log(step, evaluate_random_levels(model, cfg, device, seed=cfg.seed + step))
        if cfg.save_every > 0 and step % cfg.save_every == 0:
            save_dt(out_dir / f"dt_iter_{step:07d}.pt", model, cfg.reward_scale, asdict(cfg), step,
                    optimizer)

    save_dt(out_dir / "dt_final.pt", model, cfg.reward_scale, asdict(cfg), cfg.max_iters)
    print(f"saved {out_dir / 'dt_final.pt'}", flush=True)
    logger.close()


def main(argv=None) -> None:
    train(load_config(DTTrainConfig, argv, description=__doc__))


if __name__ == "__main__":
    main()
