"""N×N 全组合矩阵：多目标帕累托/优于基线筛选 + 飞书完成摘要 单测。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from web import feishu_notify as fn

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("hold_matrix_mod", _ROOT / "scripts" / "hold_matrix.py")
_hm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hm)


def _row(combo, ret=0.0, sharpe=0.0, dd=-5.0, plr=1.0, **kw):
    r = {"combo": combo, "total_return": ret, "sharpe": sharpe,
         "max_drawdown": dd, "profit_loss_ratio": plr, "n_trades": 10}
    r.update(kw)
    return r


def test_pick_best_combo_skips_baseline_and_picks_by_sharpe() -> None:
    ranking = [
        _row("signal", ret=0.30, sharpe=9.0, dd=-0.01),          # 基线再高也不算「最优组合」
        _row("dd+chandelier", ret=0.05, sharpe=2.0, dd=-0.015),
        _row("be+time", ret=0.04, sharpe=1.8, dd=-0.02),
        _row("dd+be", ret=0.03, sharpe=None, dd=-0.01),          # 无夏普 → 跳过
    ]
    best = _hm.pick_best_combo(ranking)
    assert best is not None and best["combo"] == "dd+chandelier"


def test_pick_best_combo_break_tie_by_smaller_drawdown() -> None:
    ranking = [
        _row("dd+chandelier", sharpe=2.5, dd=-0.03),
        _row("be+time", sharpe=2.5, dd=-0.01),   # 同夏普 → 回撤更小者胜
    ]
    assert _hm.pick_best_combo(ranking)["combo"] == "be+time"


def test_pick_best_combo_none_when_only_baseline() -> None:
    assert _hm.pick_best_combo([_row("signal", sharpe=9.0)]) is None
    assert _hm.pick_best_combo([]) is None
    assert _hm.pick_best_combo([_row("risk", sharpe=None)]) is None


def test_pareto_front_picks_non_dominated() -> None:
    rows = [
        _row("a", ret=0.10, sharpe=1.5, dd=-0.02, plr=1.2),   # 最高收益但回撤大
        _row("b", ret=0.08, sharpe=2.0, dd=-0.01, plr=1.5),   # 均衡优（回撤 -1% < -2% = 更小）
        _row("c", ret=0.07, sharpe=1.8, dd=-0.02, plr=1.3),   # 每项都不如 b → 被支配
        _row("d", ret=0.02, sharpe=0.6, dd=-0.005, plr=2.2),  # 最小回撤最高盈亏比，但收益/夏普差
    ]
    front = {r["combo"] for r in _hm.pareto_front(rows)}
    # b 全面优于 c；a / b / d 互不支配（各自有至少一项最优）
    assert front == {"a", "b", "d"}


def test_beats_baseline_requires_all_dimensions_no_worse() -> None:
    base = _row("signal", ret=0.09, sharpe=1.6, dd=-0.03, plr=1.1)
    # 夏普/回撤/盈亏比都更好
    assert _hm.beats_baseline(_row("x", ret=0.05, sharpe=1.8, dd=-0.02, plr=1.4), base)
    # 盈亏比更差 → 不优于基线（即使回撤小）
    assert not _hm.beats_baseline(_row("y", ret=0.05, sharpe=1.8, dd=-0.02, plr=0.9), base)
    # 全等 → 无严格更优 → 不算优于
    assert not _hm.beats_baseline(dict(base), base)
    # 缺指标（None）→ False，不抛错
    assert not _hm.beats_baseline({"combo": "z", "sharpe": None, "max_drawdown": None, "profit_loss_ratio": None}, base)


def test_feishu_matrix_done_disabled_when_no_webhook(monkeypatch) -> None:
    monkeypatch.setattr(fn, "load_settings", lambda: {"feishu_enabled": True, "feishu_webhook_url": "", "feishu_secret": ""})
    ok, msg = fn.notify_hold_matrix_done({"ranking": [], "baseline_signal": {}, "focus_list": []})
    assert ok is False and "Webhook" in msg


def test_feishu_matrix_done_message_content(monkeypatch) -> None:
    captured: dict = {}

    def _fake_send(text: str, **kw):
        captured["text"] = text
        return True, "ok"

    monkeypatch.setattr(fn, "load_settings", lambda: {"feishu_enabled": True, "feishu_webhook_url": "http://x", "feishu_secret": ""})
    monkeypatch.setattr(fn, "send_text", _fake_send)
    matrix = {
        "data_file": "data/slices/BTCUSDT_M5.parquet",
        "bars": 5000,
        "window_bars": 3000,
        "window_start": 2000,
        "baseline_signal": _row("signal", ret=0.36, sharpe=3.2, dd=-0.096, plr=0.86),
        "ranking": [
            {**_row("signal", ret=0.36, sharpe=3.2, dd=-0.096, plr=0.86), "pareto": True, "beats_base": False, "focus": False, "label": "信号跟随"},
            {**_row("dd+be", ret=0.12, sharpe=2.9, dd=-0.02, plr=1.7), "pareto": True, "beats_base": True, "focus": True, "label": "保本 + 回撤熔断"},
            {**_row("chandelier+dd", ret=0.05, sharpe=1.1, dd=-0.01, plr=1.2), "pareto": False, "beats_base": False, "focus": False, "label": "吊灯 + 回撤熔断"},
        ],
        "pareto_front": [],
        "focus_list": [_row("dd+be", ret=0.12, sharpe=2.9, dd=-0.02, plr=1.7)],
    }
    ok, _ = fn.notify_hold_matrix_done(matrix)
    assert ok is True
    txt = captured["text"]
    assert "尾部 3000 根" in txt and "原始序列 bar 2000..6999" in txt  # 窗口标注
    assert "基线 信号跟随" in txt
    assert "Top3" in txt or "Top" in txt
    assert "★关注" in txt                                 # top5 里 focus 组合带标记
    assert "值得实盘关注" in txt and "回撤熔断 (DD)" in txt and "保本追踪" in txt
