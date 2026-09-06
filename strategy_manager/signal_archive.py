"""线上档案 + 退化监控/回滚（P2）。

数据流：
1. 部署：引擎每次部署冠军时写入 strategies/champion_history.json（append-only，
   含 date/symbol/formula/val/holdout/data_fingerprint）——档案即冠军历史。
2. 实盘：live_trade.py 每根 bar 调 append_live_signal(symbol, ts, value) 追加信号。
3. 退化检测：check_decay 用信号档案回填已实现收益（open 序列 → target_ret），
   计算滚动真实 IC/Sharpe；连续 N 期下滑 → 告警，可回滚到上一版冠军。

所有读取都是容错式（文件缺失/损坏 → 安全默认），绝不抛错阻断实盘。
"""
from __future__ import annotations

import json
import math
import pathlib
from datetime import datetime, timezone
from typing import Any

import torch

from model_core.config import ModelConfig
from model_core.engine import _champion_history, _record_champion_event

_ARCHIVE_DIR = pathlib.Path("data") / "signal_archive"


# ── 档案读取 ────────────────────────────────────────────────────────────────

def deployments(symbol: str) -> list[dict]:
    """该品种全部冠军部署记录（新→旧）。"""
    hist = _champion_history()
    return [e for e in reversed(hist)
            if e.get("symbol") == symbol and e.get("event") == "deploy"]


def last_deployment(symbol: str) -> dict | None:
    rows = deployments(symbol)
    return rows[0] if rows else None


def previous_deployment(symbol: str) -> dict | None:
    rows = deployments(symbol)
    return rows[1] if len(rows) > 1 else None


# ── 信号档案（每 bar 追加）──────────────────────────────────────────────────

def append_live_signal(symbol: str, ts: int, value: float) -> None:
    """实盘每根 bar 追加公式信号（CSV 行: ts,value）。失败仅告警不抛错。"""
    try:
        _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace(":", "_").replace("/", "_")
        path = _ARCHIVE_DIR / f"{safe}.csv"
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(f"{int(ts)},{value!r}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[signal_archive] 追加失败 {symbol}: {exc}")


def load_signals(symbol: str) -> list[tuple[int, float]]:
    """读取信号档案（ts, value），旧→新；缺失/损坏返回空。"""
    try:
        safe = symbol.replace(":", "_").replace("/", "_")
        path = _ARCHIVE_DIR / f"{safe}.csv"
        if not path.exists():
            return []
        rows: list[tuple[int, float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                rows.append((int(parts[0]), float(parts[1])))
            except ValueError:
                continue
        return rows
    except Exception:  # noqa: BLE001
        return []


# ── 已实现性能回填 ──────────────────────────────────────────────────────────

def realized_performance(
    symbol: str,
    open_prices: torch.Tensor,      # [T] 或 [1, T] open 序列
    cost_rate: float | None = None,
    min_bars: int = 60,
) -> dict[str, Any] | None:
    """用信号档案 + 已实现 open 序列计算真实 IC / Sharpe。

    对齐语义与训练一致：target_ret[t] = log(open[t+2]/open[t+1])，
    IC = corr(signal[t], target_ret[t])，pnl = tanh(signal)×target − 换手×cost。
    """
    rows = load_signals(symbol)
    if len(rows) < min_bars:
        return None
    op = open_prices.reshape(-1).float()
    if op.numel() < 3:
        return None
    times = torch.arange(op.numel(), dtype=torch.int64)  # 档案可能只覆盖尾部：按 open 长度对齐

    sig_map = {ts: v for ts, v in rows}
    T = op.numel()
    target = torch.zeros(T)
    target[: T - 2] = torch.log(op[2:] / op[1:-1].clamp_min(1e-12))

    # 信号按 ts 与 open 索引对齐（用 open 的整数下标近似——调用方可传真实 time 数组）
    n_sig = min(len(rows), T - 2)
    sig_ts = torch.tensor([s for s, _ in rows[-n_sig:]], dtype=torch.int64)
    sig_v = torch.tensor([v for _, v in rows[-n_sig:]], dtype=torch.float32)
    # 对齐：target 的下标 = sig_ts - (sig_ts[0]) 近似（调用方应保证 ts 连续递增）
    if n_sig < min_bars:
        return None
    x = sig_v
    y = target[-n_sig:]
    xm, ym = x - x.mean(), y - y.mean()
    sx, sy = (xm ** 2).mean().sqrt(), (ym ** 2).mean().sqrt()
    ic = float((xm * ym).mean() / (sx * sy + 1e-9)) if sx > 1e-9 and sy > 1e-9 else 0.0

    pos = torch.tanh(sig_v)
    prev = torch.zeros_like(pos)
    prev[1:] = pos[:-1]
    cost = cost_rate if cost_rate is not None else float(ModelConfig.FINALIST_COST_RATE)
    pnl = pos * y - (pos - prev).abs() * cost
    ppy = 6240
    sharpe = float(pnl.mean() / (pnl.std() + 1e-9) * math.sqrt(ppy))
    return {
        "symbol": symbol,
        "bars_archived": len(rows),
        "bars_used": n_sig,
        "realized_ic": round(ic, 4),
        "realized_sharpe": round(sharpe, 3),
        "total_return_pct": round(float(pnl.sum()) * 100.0, 3),
        "cost_rate": cost,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


# ── 退化检测与回滚 ──────────────────────────────────────────────────────────

def check_decay(
    symbol: str,
    open_prices: torch.Tensor,
    min_bars: int = 60,
    window: int = 60,
    max_consecutive_declines: int = 3,
) -> dict[str, Any]:
    """连续 N 期真实 IC 下滑 → 告警（并给出回滚建议）。

    把档案切成长度为 window 的滚动块，逐块算真实 IC；
    若最近 N 块 IC 单调下滑 → decayed=True。
    """
    rows = load_signals(symbol)
    perf = realized_performance(symbol, open_prices, min_bars=min_bars)
    if perf is None:
        return {"symbol": symbol, "decayed": False,
                "reason": f"信号档案不足 {min_bars} 根", "perf": None}

    sig_v = torch.tensor([v for _, v in rows], dtype=torch.float32)
    op = open_prices.reshape(-1).float()
    T = op.numel()
    target = torch.zeros(T)
    target[: T - 2] = torch.log(op[2:] / op[1:-1].clamp_min(1e-12))
    n = sig_v.numel()
    if n < window * (max_consecutive_declines + 1):
        return {"symbol": symbol, "decayed": False,
                "reason": f"档案仅 {n} 根，不足 {window * (max_consecutive_declines + 1)} 根判定窗口",
                "perf": perf}

    def _ic_chunk(x: torch.Tensor, y: torch.Tensor) -> float:
        xm, ym = x - x.mean(), y - y.mean()
        sx, sy = (xm ** 2).mean().sqrt(), (ym ** 2).mean().sqrt()
        if sx < 1e-9 or sy < 1e-9:
            return 0.0
        return float((xm * ym).mean() / (sx * sy + 1e-9))

    y_used = target[-n:]
    ic_series: list[float] = []
    for i in range(0, n - window + 1, window):
        ic_series.append(_ic_chunk(sig_v[i:i + window], y_used[i:i + window]))

    declines = 0
    for i in range(1, len(ic_series)):
        if ic_series[i] < ic_series[i - 1]:
            declines += 1
            if declines >= max_consecutive_declines:
                break
        else:
            declines = 0
    decayed = declines >= max_consecutive_declines and len(ic_series) >= max_consecutive_declines + 1
    return {
        "symbol": symbol,
        "decayed": bool(decayed),
        "ic_series": [round(v, 4) for v in ic_series],
        "perf": perf,
        "consecutive_declines": declines,
        "max_consecutive_declines": max_consecutive_declines,
    }


def rollback_champion(symbol: str) -> dict[str, Any]:
    """回滚到上一版冠军：把 previous_deployment 的公式写回 strategies/best_{symbol}.json。"""
    prev = previous_deployment(symbol)
    save_path = pathlib.Path("strategies") / f"best_{symbol}.json"
    if prev is None or not prev.get("formula"):
        return {"ok": False, "reason": f"{symbol} 无上一版冠军可回滚（champion_history 不足）"}

    from model_core.vocab import VOCAB_VERSION
    names = FORMULA_VOCAB_TOKEN_NAMES()
    strategy = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "formula": list(prev["formula"]),
        "formula_decoded": " -> ".join(
            names[t] if 0 <= t < len(names) else f"?{t}" for t in prev["formula"]
        ),
        "best_score": prev.get("best_score"),
        "rollback_of": prev.get("ts"),
        "rollback_reason": "decay monitor",
        "holdout": prev.get("holdout"),
        "data_fingerprint": prev.get("data_fingerprint"),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(save_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(strategy, fp, indent=2, ensure_ascii=False)
    import os
    os.replace(tmp, save_path)
    _record_champion_event({
        "event": "rollback", "symbol": symbol,
        "formula": list(prev["formula"]),
        "rolled_back_to": prev.get("ts"),
        "reason": "decay monitor",
    })
    return {"ok": True, "symbol": symbol, "restored_formula": prev["formula"],
            "save_path": str(save_path)}


def FORMULA_VOCAB_TOKEN_NAMES() -> list[str]:
    from model_core.vocab import FORMULA_VOCAB
    return FORMULA_VOCAB.token_names