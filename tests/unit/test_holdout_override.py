"""可配置样本外预留（--holdout-bars N）与溯源偏好单测。

固化两点：
1. AlphaEngine._resolve_holdout_bars 允许 engine.holdout_override 覆盖默认
   ModelConfig.HOLDOUT_BARS（小数据集仍按 T//8 比例压缩，保证确定性）；
2. web.oos_provenance.strategy_provenance 优先读实际写入的 holdout_bars
   （train_range / 策略顶层 / data_source），缺省才按 ModelConfig 默认重算——
   否则用 --holdout-bars 3000 训出的策略会被溯源错标成 500 根。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_core.config import ModelConfig  # noqa: E402
from model_core.engine import AlphaEngine  # noqa: E402
from web.oos_provenance import strategy_provenance  # noqa: E402


def _mk_engine() -> AlphaEngine:
    # 绕过 __init__（需要 data_manager/torch 张量），只测纯逻辑方法
    eng = object.__new__(AlphaEngine)
    eng.holdout_override = None
    return eng


def _write_parquet(path: Path, n: int, start_ts: int, step_s: int = 300) -> None:
    ts = np.arange(start_ts, start_ts + n * step_s, dtype=np.int64)[:n]
    df = pd.DataFrame({
        "timestamp": ts,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_resolve_default_uses_modelconfig(tmp_path) -> None:
    eng = _mk_engine()
    # 默认 500（cap = 950k//8 很大，不触发）
    got = eng._resolve_holdout_bars(950_000)
    assert got == int(ModelConfig.HOLDOUT_BARS) == 500


def test_resolve_override_bigger_holdout() -> None:
    eng = _mk_engine()
    eng.holdout_override = 3000
    # 950k 全历史：cap=T//8≈118k → 3000 生效
    assert eng._resolve_holdout_bars(950_000) == 3000
    # 40000 根切片：cap=5000 → 3000 生效
    assert eng._resolve_holdout_bars(40_000) == 3000
    # 小数据按比例压缩：T=8000 → cap=1000
    assert eng._resolve_holdout_bars(8_000) == 1000
    eng.holdout_override = None


def test_provenance_prefers_stored_holdout_bars(tmp_path) -> None:
    data = tmp_path / "BTCUSDT_M5.parquet"
    _write_parquet(data, 4000, start_ts=1_700_000_000)
    strat = tmp_path / "best_X.json"
    strat.write_text(json.dumps({
        "formula": [1, 2, 3],
        "data_source": {"data_file": str(data), "bars": 4000, "mode": "full"},
        "train_range": {"data_file": str(data), "n_bars": 4000, "mode": "full",
                        "holdout_bars": 3000},
        "holdout_bars": 3000,
    }, ensure_ascii=False), encoding="utf-8")
    prov = strategy_provenance(strat)
    assert prov is not None
    # 顶层/溯源里写了 3000 → 不该按默认 500 重算（cap=min(5000,4000//8=500)）
    assert prov["holdout_bars"] == 3000


def test_provenance_fallback_to_default(tmp_path) -> None:
    data = tmp_path / "BTCUSDT_M5.parquet"
    _write_parquet(data, 4000, start_ts=1_700_000_000)
    strat = tmp_path / "best_Y.json"
    strat.write_text(json.dumps({
        "formula": [1, 2, 3],
        "data_source": {"data_file": str(data), "bars": 4000, "mode": "full"},
        "train_range": {"data_file": str(data), "n_bars": 4000, "mode": "full"},
    }, ensure_ascii=False), encoding="utf-8")
    prov = strategy_provenance(strat)
    assert prov is not None
    # 无任何存储值 → 按 ModelConfig 默认重算：min(500, 4000//8=500)
    assert prov["holdout_bars"] == 500
