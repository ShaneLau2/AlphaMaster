"""web/compare_report 单测：报告渲染 / 品种过滤 / 迷你口径标注。"""
from __future__ import annotations

import json

from web.compare_report import (
    light_summary,
    render_markdown,
    summary_for_symbol,
    summary_symbol,
)

SUMMARY = {
    "kind": "compare_ranges",
    "created_at": "2026-09-04T10:33:45+00:00",
    "source_file": "/x/data/slices/BTCUSDT_M5.parquet",
    "params": {"n_bars": 15000, "chunks": None, "steps": 10, "seeds": [42, 7], "regime": "vol"},
    "variants": {
        "tail": {
            "runs": [{"symbol": "BTCUSDT", "seed": 42, "best_score": 2.1634,
                      "holdout": {"val_score": -0.3691, "sharpe": -6.42, "score_ratio": -0.17, "passed": False}},
                     {"symbol": "BTCUSDT", "seed": 7, "best_score": 1.104,
                      "holdout": {"val_score": 6.97, "sharpe": 11.36, "score_ratio": 5.5, "passed": True}}],
            "best": {"symbol": "BTCUSDT", "seed": 42, "best_score": 2.1634,
                     "holdout": {"val_score": -0.3691, "sharpe": -6.42, "score_ratio": -0.17, "passed": False}},
            "holdout_passed": 1,
        },
        "spread": {
            "runs": [{"symbol": "BTCUSDT", "seed": 42, "best_score": 0.8747,
                      "holdout": {"val_score": 2.962, "sharpe": 8.19, "score_ratio": 3.39, "passed": True}}],
            "best": {"symbol": "BTCUSDT", "seed": 42, "best_score": 0.8747,
                     "holdout": {"val_score": 2.962, "sharpe": 8.19, "score_ratio": 3.39, "passed": True}},
            "holdout_passed": 1,
        },
    },
    "conclusion": ["仅 spread 通过 holdout 闸门：优先 spread（全历史分块）。"],
}


def test_summary_symbol_from_source():
    assert summary_symbol(SUMMARY) == "BTCUSDT"


def test_summary_for_symbol_matching_and_mismatch():
    assert summary_for_symbol("BTCUSDT") is not None
    assert summary_for_symbol("ETHUSDT") is None
    assert summary_for_symbol(None) is not None


def test_light_summary_shape():
    s = light_summary(SUMMARY)
    assert s["symbol"] == "BTCUSDT"
    assert s["mini"] is True  # steps=10 < 100
    assert s["tail"]["passed"] is False and s["spread"]["passed"] is True
    assert s["spread"]["holdout_val"] == 2.962
    assert s["spread"]["runs"] == 1


def test_render_markdown_contains_both_and_mini():
    md = render_markdown(SUMMARY)
    assert "tail（最近 N 根）" in md
    assert "spread（全历史分层分块）" in md
    assert "迷你口径" in md
    # 代表 = 跨 seed 中位 holdout 口径（fixture 落盘 conclusion 优先展示）
    assert "代表(seed)" in md and "跨seed中位" in md
    assert "仅 spread 通过 holdout 闸门" in md
    assert "| tail | 42 |" in md


def test_render_markdown_empty():
    assert "尚无对比实验结果" in render_markdown(None)


def test_light_summary_none():
    assert light_summary(None) is None
