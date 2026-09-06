"""对比实验代表口径单测：跨 seed 中位 holdout（替代 in-sample best）。"""
from __future__ import annotations

from web.compare_report import (
    conclusion_for,
    light_summary,
    median_holdout,
    pick_best,
    pick_representative,
    render_markdown,
)


def _run(seed, best, val=None, passed=False):
    ho = None if val is None else {"val_score": val, "passed": passed, "score_ratio": val}
    return {"seed": seed, "best_score": best, "holdout": ho}


def test_pick_representative_odd_uses_middle_holdout() -> None:
    runs = [_run(1, 9.0, val=-1.0), _run(2, 1.0, val=0.5), _run(3, 5.0, val=3.0)]
    rep = pick_representative(runs)
    assert rep["seed"] == 2  # holdout 中位数位置（0.5 中间），与 best 无关


def test_pick_representative_ignores_insample_lottery() -> None:
    # 2 seeds：in-sample best 是 seed7（彩票 holdout），代表必须是 seed42
    runs = [_run(42, 2.0, val=-0.37), _run(7, 1.1, val=6.97, passed=True)]
    rep = pick_representative(runs)
    assert rep["seed"] == 42


def test_pick_representative_even_tie_takes_lower() -> None:
    # 中位数恰为两点中点（-0.5 与 0.5 的均值 0）→ 悲观侧取较低 holdout
    runs = [_run(1, 1.0, val=-0.5), _run(2, 1.0, val=0.5)]
    assert pick_representative(runs)["seed"] == 1


def test_pick_representative_missing_holdout_side() -> None:
    # 偶数且一侧无 holdout 读数 → 取有读数的一侧，不选无代表 run
    runs = [_run(1, 5.0, val=None), _run(2, 1.0, val=2.0)]
    assert pick_representative(runs)["seed"] == 2


def test_pick_representative_all_missing_falls_back_to_best() -> None:
    runs = [_run(1, 1.0, val=None), _run(2, 3.0, val=None)]
    assert pick_representative(runs)["seed"] == 2


def test_median_holdout_mean_for_even() -> None:
    runs = [_run(1, 1.0, val=-0.37), _run(2, 1.0, val=6.97)]
    assert abs(median_holdout(runs) - (6.97 - 0.37) / 2) < 1e-9
    assert median_holdout([_run(1, 1.0, val=None)]) is None


def test_pick_best_still_insample() -> None:
    runs = [_run(42, 2.1634, val=-0.37), _run(7, 1.104, val=6.97)]
    assert pick_best(runs)["seed"] == 42  # 展示曲线仍用 in-sample best（与结论代表无关）


def test_conclusion_gate_and_order() -> None:
    t = _run(42, 2.0, val=-0.37)
    s = _run(42, 0.87, val=2.96, passed=True)
    lines = conclusion_for(t, s)
    assert any("spread" in l for l in lines)
    both = conclusion_for(_run(1, 1, val=1.0, passed=True), _run(2, 1, val=2.0, passed=True))
    assert any("spread" in l for l in both)  # spread 分更高
    none_ = conclusion_for(_run(1, 1, val=0.1), _run(2, 1, val=0.2))
    assert any("未通过" in l for l in none_)


def _summary(runs_tail, runs_spread, best_t=None, best_s=None):
    return {
        "kind": "compare_ranges",
        "created_at": "2026-09-04T10:33:45+00:00",
        "source_file": "/x/BTCUSDT_M5.parquet",
        "params": {"steps": 10, "seeds": [42, 7], "n_bars": 15000, "regime": "vol"},
        "variants": {
            "tail": {"runs": runs_tail, "best": best_t or runs_tail[0], "holdout_passed": 1},
            "spread": {"runs": runs_spread, "best": best_s or runs_spread[0], "holdout_passed": 1},
        },
        "conclusion": [],
    }


def test_render_markdown_representative_table() -> None:
    s = _summary(
        [_run(42, 2.1634, val=-0.3691), _run(7, 1.104, val=6.97, passed=True)],
        [_run(42, 0.8747, val=2.962, passed=True)],
    )
    md = render_markdown(s)
    assert "代表(seed)" in md and "跨seed中位" in md
    assert "中位口径" not in md  # chip 在前端，不在 markdown
    assert "不用 in-sample best 选代表" in md


def test_light_summary_uses_representative_and_median() -> None:
    s = _summary(
        [_run(42, 2.1634, val=-0.3691), _run(7, 1.104, val=6.97, passed=True)],
        [_run(42, 0.8747, val=2.962, passed=True)],
    )
    ls = light_summary(s)
    assert ls["tail"]["seed"] == 42  # 代表 = seed42（中位侧），不是 best 的 seed
    assert abs(ls["tail"]["median_holdout"] - 3.3004) < 0.01
    assert ls["spread"]["median_holdout"] == 2.962
    assert ls["tail"]["passed"] is False and ls["spread"]["passed"] is True
