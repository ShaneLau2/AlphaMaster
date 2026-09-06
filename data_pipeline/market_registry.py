"""M0: market registry 读写 —— 资产类别 policy → 市场列表 + OOS 复核腿。

registry 文件: data/market_registry.json
  policies.<name>: {group, min_bars, markets: [{symbol, timeframe, group, role, file, note}]}
  oos_review: [{symbol, timeframe, group, role="oos", file, note}]

role 语义: train(训练腿) / pending(文件缺失或未达标, 不可训练) / oos(仅复核)。
"""
from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = _PROJECT_ROOT / "data" / "market_registry.json"


def load_registry(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_REGISTRY
    if not p.exists():
        raise FileNotFoundError(f"registry 不存在: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    if "policies" not in d or not isinstance(d["policies"], dict):
        raise ValueError(f"registry 格式错误(缺 policies): {p}")
    return d


def list_policies(reg: dict) -> list[str]:
    return list(reg.get("policies", {}).keys())


def policy_config(reg: dict, policy: str) -> dict:
    pol = reg.get("policies", {}).get(policy)
    if not pol:
        raise KeyError(f"policy 不存在: {policy} (可选: {list_policies(reg)})")
    return pol


def policy_markets(reg: dict, policy: str, role: str | None = None) -> list[dict]:
    """返回 policy 的市场列表; role 过滤(train/pending/oos)。"""
    ms = list(policy_config(reg, policy).get("markets", []))
    if role is not None:
        ms = [m for m in ms if m.get("role") == role]
    return ms


def train_markets(reg: dict, policy: str) -> list[dict]:
    """训练腿 = role=train 且文件存在(文件缺失的 train 腿降级为 pending 并告警)。"""
    out: list[dict] = []
    for m in policy_markets(reg, policy, role="train"):
        if Path(m["file"]).exists():
            out.append(m)
        else:
            m2 = dict(m)
            m2["role"] = "pending"
            print(f"[registry] 训练腿文件缺失, 降级 pending: {m2['symbol']} {m2['file']}")
            out.append(m2)
    return out


def min_bars_for(reg: dict, policy: str) -> int:
    return int(policy_config(reg, policy).get("min_bars", 8000))


def oos_review_markets(reg: dict) -> list[dict]:
    """OOS 复核腿(跨标的, 供 P0.2/LMO 用); 文件缺失则跳过。"""
    out: list[dict] = []
    for m in reg.get("oos_review", []):
        if Path(m["file"]).exists():
            out.append(m)
        else:
            print(f"[registry] OOS 文件缺失, 跳过: {m.get('symbol')} {m['file']}")
    return out
