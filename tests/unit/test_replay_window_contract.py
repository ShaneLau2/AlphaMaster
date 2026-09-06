"""回放窗口契约（2026-09-06 修复）：window_bars 是总切片长，前 800 根为
warm-up（run_replay 自第 800 根才开始交易）。

修复前 window_bars=800 是最小允许输入却返回空结果（ok=True, 0 笔,
sharpe 0）——误导；且 tail 对齐窗口的可交易样本随 window 变化
（window − 800），不同 window 的 Sharpe 不可比。现：
- window ≤ 800（或 ≤0）→ 400，明确说明 warm-up 占满窗口；
- 响应带 warmup_bars / usable_bars，供跨窗口可比。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
SF_H1 = str(ROOT / "strategies/best_BTCUSDT_H1.json")
DF_H1 = str(ROOT / "data/training/BTCUSDT_H1.parquet")


def _req(**kw):
    from web.app import PaperReplayRequest
    base = dict(data_file=DF_H1, strategy_file=SF_H1, policy_id="signal")
    base.update(kw)
    return PaperReplayRequest(**base)


def _reject(func, window):
    with pytest.raises(HTTPException) as ei:
        func(_req(window_bars=window))
    assert ei.value.status_code == 400
    detail = str(ei.value.detail)
    assert ("warm-up" in detail or "可交易" in detail
            or "正整数" in detail)  # ≤0 的窗口按契约一并拒绝


def test_replay_rejects_empty_warmup_only_window():
    """window ≤ 800 曾经返回 ok+0 交易空结果 → 现在必须 400 并说明原因。"""
    from web.app import api_paper_replay
    for w in (800, 700, 0, -5):
        _reject(api_paper_replay, w)


def test_replay_compare_rejects_empty_window_too():
    """replay-compare 共享同一窗口契约。"""
    from web.app import api_paper_replay_compare
    _reject(api_paper_replay_compare, 800)


def test_replay_reports_warmup_and_usable_bars():
    """1500 窗 → warmup 800、usable=700（stats 只覆盖这 700 根）。"""
    from web.app import api_paper_replay
    res = api_paper_replay(_req(window_bars=1500))
    assert res["ok"] is True
    assert res["warmup_bars"] == 800
    assert res["usable_bars"] == 700
    assert res["bars"] == 1500


def test_usable_bars_span_comparability():
    """不同 window 的可用样本 = window − 800，可直接比（不再是整段混同）。"""
    from web.app import api_paper_replay
    r12 = api_paper_replay(_req(window_bars=1200))
    r30 = api_paper_replay(_req(window_bars=3000))
    assert r12["usable_bars"] == 400 and r30["usable_bars"] == 2200
    assert r30["usable_bars"] - r12["usable_bars"] == 1800
    # OOS 分类不随窗口变化（同一尾部 holdout 500 根）
    assert r12["oos"]["n_holdout"] == r30["oos"]["n_holdout"] == 500
    assert r12["oos"]["status"] == r30["oos"]["status"] == "partial"


def test_replay_full_file_window_reports_usable():
    """window_bars=None → 整文件，usable = T − 800。"""
    import pyarrow.parquet as pq
    from web.app import api_paper_replay
    T = pq.read_table(DF_H1).num_rows
    res = api_paper_replay(_req(window_bars=None))
    assert res["ok"] is True
    assert res["bars"] == T
    assert res["usable_bars"] == T - 800
