"""统一可配置无信号阈值（观望区）单测。

覆盖：
1. live_signal.evaluate_signal 的 threshold 覆盖 → LONG/SHORT/FLAT 三档；
2. signal.compute_target_positions 的 min_abs 覆盖（训练侧默认 Config 不变）；
3. settings.sanitize_signal_threshold / 档位集合（0.05/0.3/0.5/0.8）；
4. engine.neutral_band_bonus 中性带正则（观望偏好，w=0 关闭）；
5. realtime factor_percentile + WatchTask.factor_pct_hist 暴露。
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import strategy_manager.signal as sig_mod  # noqa: E402
from model_core.engine import neutral_band_bonus  # noqa: E402
from strategy_manager.live_signal import DIR_FLAT, DIR_LONG, DIR_SHORT  # noqa: E402
from web.realtime_manager import WatchTask, factor_percentile  # noqa: E402


# ── 1. live_signal.evaluate_signal 阈值覆盖 ────────────────────────────────

def _raw_dict(T: int = 60) -> dict:
    import math

    import numpy as np

    px = np.linspace(100.0, 110.0, T)
    ts = [float(1_700_000_000 + i * 300) for i in range(T)]
    cols = {
        "open": [px],
        "high": [px + 0.5],
        "low": [px - 0.5],
        "close": [px],
        "volume": [np.full(T, 1.0)],
        "time": [ts],
    }
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in cols.items()}


def _eval_with_factor(factor: float, threshold: float | None, monkeypatch):
    """固定因子值跑 evaluate_signal（mock VM 输出，只测阈值分支）。"""
    import strategy_manager.live_signal as ls

    monkeypatch.setattr(ls, "_min_bars", lambda: 50)
    T = 60

    def fake_execute(formula, feats):
        return torch.full((1, T), factor, dtype=feats.dtype)

    monkeypatch.setattr(ls._VM, "execute", fake_execute)
    return ls.evaluate_signal([1], _raw_dict(T), threshold=threshold)


def test_evaluate_signal_threshold_flat_zone(monkeypatch) -> None:
    """factor=0.3（tanh≈0.29）：阈值 0.05 → LONG；0.5/0.8 → 观望 FLAT。"""
    res = _eval_with_factor(0.3, 0.05, monkeypatch)
    assert res["state"] == "ok" and res["direction"] == DIR_LONG
    assert res["threshold"] == 0.05
    res = _eval_with_factor(0.3, 0.5, monkeypatch)
    assert res["direction"] == DIR_FLAT
    res = _eval_with_factor(0.3, 0.8, monkeypatch)
    assert res["direction"] == DIR_FLAT


def test_evaluate_signal_threshold_short_and_strong(monkeypatch) -> None:
    """factor=-0.4（tanh≈-0.38）→ SHORT；factor=1.0（tanh≈0.76）在 0.5 挡仍 LONG。"""
    res = _eval_with_factor(-0.4, 0.3, monkeypatch)
    assert res["direction"] == DIR_SHORT
    res = _eval_with_factor(1.0, 0.5, monkeypatch)
    assert res["direction"] == DIR_LONG
    # 0.8 挡下 0.76 < 0.8 → FLAT
    res = _eval_with_factor(1.0, 0.8, monkeypatch)
    assert res["direction"] == DIR_FLAT


# ── 2. signal.compute_target_positions min_abs 覆盖 ───────────────────────

def test_compute_target_positions_min_abs_override() -> None:
    f = torch.tensor([0.1, 0.5, -0.8, 0.0])
    tanh = torch.tanh(f)
    # 显式 min_abs=0.3：|tanh|<0.3 的置零（0.1 → 0.0997 被清掉，-0.8 → -0.664 保留）
    pos = sig_mod.compute_target_positions(f, min_abs=0.3)
    assert pos[0].item() == 0.0
    assert abs(pos[1].item() - tanh[1].item()) < 1e-6
    assert abs(pos[2].item() - tanh[2].item()) < 1e-6
    assert pos[3].item() == 0.0
    # 默认 None → 训练侧 Config.MIN_TRADE_EXPOSURE=0.05：0.0997 保留
    pos_d = sig_mod.compute_target_positions(f)
    assert abs(pos_d[0].item() - tanh[0].item()) < 1e-6


def test_target_to_direction_override() -> None:
    assert sig_mod.target_to_direction(0.29, min_abs=0.3) == 0
    assert sig_mod.target_to_direction(0.29, min_abs=0.05) == 1
    assert sig_mod.target_to_direction(-0.35, min_abs=0.3) == -1


# ── 3. settings 档位消毒 ──────────────────────────────────────────────────

def test_sanitize_signal_threshold(monkeypatch) -> None:
    from web.settings import _DEFAULT, sanitize_signal_threshold

    assert sanitize_signal_threshold(0.05) == 0.05
    assert sanitize_signal_threshold(0.3) == 0.3
    assert sanitize_signal_threshold("0.5") == 0.5
    assert sanitize_signal_threshold(0.8) == 0.8
    # 非档位值 → 回退默认
    assert sanitize_signal_threshold(0.07) == _DEFAULT["signal_threshold"]
    assert sanitize_signal_threshold(None) == _DEFAULT["signal_threshold"]
    assert sanitize_signal_threshold("abc") == _DEFAULT["signal_threshold"]
    # 浮点容差
    assert sanitize_signal_threshold(0.30000000004) == 0.3
    # 默认键存在且为 0.05
    assert _DEFAULT["signal_threshold"] == 0.05


def test_resolve_signal_threshold_fallback(monkeypatch) -> None:
    import web.settings as ws

    monkeypatch.setattr(ws, "load_settings", lambda: {"signal_threshold": 0.5})
    assert ws.resolve_signal_threshold() == 0.5
    monkeypatch.setattr(ws, "load_settings", lambda: {"signal_threshold": 0.99})
    # 非法档位 → 回退 Config.MIN_TRADE_EXPOSURE（0.05）
    assert ws.resolve_signal_threshold() == 0.05


# ── 4. 中性带正则（观望偏好） ─────────────────────────────────────────────

def test_neutral_band_bonus() -> None:
    f = torch.tensor([[-0.4, 0.0, 0.6, 3.0, -10.0]])
    # |f| < 0.5 占 2/5 → w=1.0 → 0.4
    assert abs(neutral_band_bonus(f, 1.0, 0.5).item() - 0.4) < 1e-6
    # w=0 → 无奖励（默认关闭）
    assert neutral_band_bonus(f, 0.0, 0.5).item() == 0.0
    # 全在中性带 → 满奖励 w
    assert abs(neutral_band_bonus(torch.tensor([[0.1, -0.2, 0.0]]), 0.3).item() - 0.3) < 1e-6


def test_model_config_neutral_defaults_off() -> None:
    from model_core.config import ModelConfig

    assert getattr(ModelConfig, "NEUTRAL_BAND_W", 0.0) == 0.0
    assert getattr(ModelConfig, "NEUTRAL_BAND_HALF", 0.5) == 0.5


# ── 5. 实时因子历史分位 ───────────────────────────────────────────────────

def _task() -> WatchTask:
    return WatchTask(
        id="binance:BTCUSDT:5m:best_BTCUSDT",
        source="binance", symbol="BTCUSDT", timeframe="5m",
        strategy_file="strategies/best_BTCUSDT.json", strategy_name="best_BTCUSDT",
        formula=[1], vocab_version="v1", strategy_symbol="BTCUSDT",
        strategy_timeframe="M5", best_score=2.98, cadence_s=30,
    )


def test_factor_percentile_pure() -> None:
    buf = deque([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert factor_percentile(5.0, buf) == round(100.0 * 5 / 6, 1)  # 83.3
    assert factor_percentile(0.0, buf) == round(100.0 * 0 / 6, 1)  # 0.0
    assert factor_percentile(7.0, buf) == 100.0
    assert factor_percentile(3.0, deque()) is None


def test_signal_quality_exposes_pct_hist() -> None:
    t = _task()
    t.dir_hist = deque(["LONG"] * 3, maxlen=1000)  # window>0 才走完整返回
    t.factor_buf = deque([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], maxlen=2000)
    t.factor_pct_hist = deque([10.0, 20.0, 83.3], maxlen=50)
    sq = t._signal_quality()
    assert sq["factor_pct_hist"] == [10.0, 20.0, 83.3]