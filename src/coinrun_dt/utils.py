from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Logger:
    """metrics.jsonl に追記し、使えれば TensorBoard にも書く。"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_dir / "metrics.jsonl", "a", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(str(self.log_dir))
        except Exception:
            self._tb = None
        self._t0 = time.time()

    def log(self, step: int, metrics: Dict[str, float], echo: bool = True) -> None:
        record = {"step": int(step), "time": round(time.time() - self._t0, 1)}
        record.update({k: float(v) for k, v in metrics.items()})
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, float(v), step)
        if echo:
            body = " ".join(f"{k}={float(v):.4g}" for k, v in metrics.items())
            print(f"[step {step}] {body}", flush=True)

    def close(self) -> None:
        self._file.close()
        if self._tb is not None:
            self._tb.close()
