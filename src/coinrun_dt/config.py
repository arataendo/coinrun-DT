"""YAML 設定ファイルを dataclass に読み込む。

使い方:
    python -m coinrun_dt.ppo.train --config configs/ppo_small.yaml --set total_steps=100000 seed=1
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from typing import List, Optional, Type, TypeVar

import yaml

T = TypeVar("T")


def load_config(cls: Type[T], argv: Optional[List[str]] = None, description: str = "") -> T:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML 設定ファイル")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="設定の上書き（値は YAML として解釈される）")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--set の形式は KEY=VALUE: {item!r}")
        data[key] = yaml.safe_load(value)
    return config_from_dict(cls, data)


def config_from_dict(cls: Type[T], data: dict) -> T:
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__} に存在しない設定キー: {sorted(unknown)}")
    return cls(**data)
