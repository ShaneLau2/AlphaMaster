"""对比「代表口径」可选 + 分口径结论 + holdout 曲线叠加 相关单测。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web.compare_report import (  # noqa: E402
    CRITERIA, CRITERION_HOLDOUT_BEST, CRITERION_HOLDOUT_MEDIAN,
    CRITERION_IN_SAMPLE, conclusions_by_criterion, light_summary,
    pick_by_criterion, run_holdout_curve, summary_rep_criterion,
)


def _run(best_score: float, ho_val: float | None, seed: int,
         curve: bool = False) -> dict:
    ho = None
    if ho_val is not None:
        ho = {"val_score": ho_val, "passed": ho_val > 1.0, "sharpe": ho_val}
        if curve:
            ho["equity_curve"] = {
                "labels": [0, 1, 2, 3],
                "equity": [0.0, 0.5, -0.2, 1.0],
            }
    return {"seed": seed, "best_score": best_score, "holdout": ho}


def _summary() -> dict:
    tail_runs = [
        _run(2.0, 0.2, 7),      # holdout 低
        _run(1.8, 0.5, 42, curve=True),  # holdout 中位
        _run(2.4, 3.0, 123),    # in-sample best 且 holdout 高
    ]
    spr_runs = [
        _run(1.0, 2.5, 7, curve=True),
        _run(1.4, 4.0, 42),
        _run(1.2, 1.5, 123),
    ]
    return {
        "kind": "compare_ranges",
        "created_at": "2026-09-05T00:00:00+00:00",
        "source_file": "/x/BTCUSDT_M5.parquet",
        "params": {"n_bars": 8000, "steps": 10, "seeds": [7, 42, 123],
                   "regime": "vol", "rep_criterion": CRITERION_HOLDOUT_MEDIAN},
        "variants": {"tail": {"runs": tail_runs, "holdout_passed": 1},
                     "spread": {"runs": spr_runs, "holdout_passed": 3}},
        "conclusion": ["仅 spread 通过 holdout 闸门：优先 spread。"],
    }


def test_pick_by_criterion_in_sample_uses_best_score() -> None:
    runs = _summary()["variants"]["tail"]["runs"]
    rep = pick_by_criterion(runs, CRITERION_IN_SAMPLE)
    assert rep["seed"] == 123  # best_score 最高


def test_pick_by_criterion_holdout_median_uses_middle() -> None:
    runs = _summary()["variants"]["tail"]["runs"]
    rep = pick_by_criterion(runs, CRITERION_HOLDOUT_MEDIAN)
    assert rep["seed"] == 42  # holdout 0.2 / 0.5 / 3.0 → 中位 0.5


def test_pick_by_criterion_holdout_best() -> None:
    runs = _summary()["variants"]["tail"]["runs"]
    rep = pick_by_criterion(runs, CRITERION_HOLDOUT_BEST)
    assert rep["seed"] == 123  # holdout 最高 3.0


def test_conclusions_by_criterion_gives_each_criterion() -> None:
    cc = conclusions_by_criterion(_summary())
    assert set(cc.keys()) == set(CRITERIA)
    # 每种口径都有结论行
    for crit in CRITERIA:
        assert cc[crit], crit
    # in-sample best 口径：tail best_score 2.4 无 holdout → tail 不通过 → 仅 spread
    assert "spread" in cc[CRITERION_IN_SAMPLE][0]


def test_summary_rep_criterion_roundtrip_and_fallback() -> None:
    s = _summary()
    assert summary_rep_criterion(s) == CRITERION_HOLDOUT_MEDIAN
    assert summary_rep_criterion({"params": {}}) == CRITERION_HOLDOUT_MEDIAN
    assert summary_rep_criterion(None) == CRITERION_HOLDOUT_MEDIAN
    assert summary_rep_criterion(
        {"params": {"rep_criterion": CRITERION_IN_SAMPLE}}) == CRITERION_IN_SAMPLE


def test_run_holdout_curve_new_and_legacy() -> None:
    r = _run(1.0, 2.0, 7, curve=True)
    hc = run_holdout_curve(r)
    assert hc and hc["equity"][-1] == 1.0
    # 兼容旧格式：直接挂 run.holdout_curve
    r2 = {"holdout_curve": {"labels": [0, 1], "equity": [0.0, 1.0]}}
    hc2 = run_holdout_curve(r2)
    assert hc2 and hc2["equity"][-1] == 1.0
    assert run_holdout_curve(None) is None


def test_light_summary_carries_curves_and_dual_conclusions() -> None:
    ls = light_summary(_summary())
    assert ls is not None
    assert "conclusions_by_criterion" in ls
    assert ls["tail"]["holdout_curve"] is not None
    assert ls["spread"]["holdout_curve"] is not None
    assert ls["mini"] is True  # steps=10 < 100