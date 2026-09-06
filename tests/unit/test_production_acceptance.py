"""生产验收脚本（scripts/production_acceptance.py）纯函数单测：交易指标 / 时段切分 / 判定。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "pa_mod", _ROOT / "scripts" / "production_acceptance.py")
_pa = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pa)


# ── 交易级指标 ─────────────────────────────────────────────────
def test_trade_metrics_mixed_wins_and_losses() -> None:
    closed = [{"pnl": 0.02}, {"pnl": 0.02}, {"pnl": -0.01}]
    m = _pa.trade_metrics(closed)
    assert m["avg_pnl_per_trade"] == round(0.03 / 3, 6)
    # 期望值 = 2/3×0.02 − 1/3×(−0.01)（avg_loss 为负值）
    assert m["expectancy"] == round(2 / 3 * 0.02 - 1 / 3 * (-0.01), 6)
    assert m["avg_win"] == 0.02 and m["avg_loss"] == -0.01


def test_trade_metrics_empty() -> None:
    m = _pa.trade_metrics([])
    assert m["avg_pnl_per_trade"] is None and m["expectancy"] is None


def test_trade_metrics_all_wins_no_loss() -> None:
    m = _pa.trade_metrics([{"pnl": 0.01}, {"pnl": 0.02}])
    assert m["expectancy"] == 0.015 and m["avg_loss"] == 0.0


# ── 时段 → bar 区间（epoch 秒，含端点） ─────────────────────────
def test_segment_indices_inclusive_bounds() -> None:
    import numpy as np

    # 每天一根：2021-01-01 .. 2021-01-05（epoch 秒）
    import datetime as _dt

    days = []
    for d in range(1, 6):
        days.append(int(_dt.datetime(2021, 1, d, tzinfo=_dt.timezone.utc).timestamp()))
    ts = np.asarray(days, dtype=np.int64)
    lo, hi = _pa.segment_indices(ts, "2021-01-02", "2021-01-04")
    assert (lo, hi) == (1, 4)  # [bar1..bar3] 含端点，排他上界


def test_segment_indices_missing_range_ok() -> None:
    import numpy as np

    ts = np.asarray([1600000000, 1600003600], dtype=np.int64)
    lo, hi = _pa.segment_indices(ts, "2030-01-01", "2030-12-31")
    assert lo == hi == 2  # 空段（hi<=lo 调用方跳过）


# ── 分段裁剪（warm-up 头） ──────────────────────────────────────
def test_slice_window_prepends_warmup_head() -> None:
    import numpy as np

    d = {k: np.arange(10000, dtype=float) for k in
         ("factor", "open", "high", "low", "close")}
    sub, start = _pa.slice_window(d, 2000, 3000)
    assert start == 800
    assert sub["factor"].shape == (1800,)          # 1200..3000
    assert sub["factor"][0] == 1200.0 and sub["factor"][-1] == 2999.0


def test_slice_window_start_of_data_no_head() -> None:
    import numpy as np

    d = {k: np.arange(5000, dtype=float) for k in
         ("factor", "open", "high", "low", "close")}
    sub, start = _pa.slice_window(d, 100, 900)
    assert start == 100
    assert sub["factor"].shape == (900,)           # 0..900，无 warm-up 可裁


# ── Gate 2 · 数据可信度标签 ────────────────────────────────────
def test_confidence_label_tiers() -> None:
    import numpy as np

    def ts_for_years(y: float):
        start = 1600000000  # 2020-09
        return np.asarray([start, start + int(y * 365.25 * 86400)], dtype=np.int64)

    assert _pa.confidence_label(ts_for_years(2.3)) == ("provisional", 2.3)
    assert _pa.confidence_label(ts_for_years(4.0)) == ("medium", 4.0)
    assert _pa.confidence_label(ts_for_years(9.05)) == ("full", 9.05)
    # 边界：恰好 3 年 → medium；恰好 5 年 → medium（>5 才 full）
    assert _pa.confidence_label(ts_for_years(3.0))[0] == "medium"
    assert _pa.confidence_label(ts_for_years(5.0))[0] == "medium"
    assert _pa.confidence_label(ts_for_years(5.5))[0] == "full"


# ── 年代段规划（短历史回退三等分） ─────────────────────────────
def test_build_segments_uses_fixed_eras_when_covered() -> None:
    import numpy as np

    # 2017-08 → 2026-09（与 BTC H1 同跨度），每天一根
    import datetime as _dt

    start = int(_dt.datetime(2017, 8, 17, tzinfo=_dt.timezone.utc).timestamp())
    ts = np.asarray([start + i * 86400 for i in range(3300)], dtype=np.int64)
    segs = _pa.build_segments(ts)
    assert len(segs) == 4 and segs[0][2] == "2017–2020" and segs[-1][2] == "2025–2026"


def test_build_segments_falls_back_to_thirds_on_short_history() -> None:
    import numpy as np

    # ETH 类短历史：2024-05 → 2026-09（约 2.3 年）——固定年代段只覆盖 2 段 → 三等分
    import datetime as _dt

    start = int(_dt.datetime(2024, 5, 22, tzinfo=_dt.timezone.utc).timestamp())
    ts = np.asarray([start + i * 3600 for i in range(20000)], dtype=np.int64)
    segs = _pa.build_segments(ts)
    assert len(segs) == 3
    # 三段互不重叠且首段从数据起点开始
    lo1, hi1 = _pa.segment_indices(ts, segs[0][0], segs[0][1])
    lo2, hi2 = _pa.segment_indices(ts, segs[1][0], segs[1][1])
    lo3, hi3 = _pa.segment_indices(ts, segs[2][0], segs[2][1])
    assert lo1 == 0 and hi1 == lo2 and hi2 == lo3 and hi3 == 20000
    assert segs[0][2].startswith("第1段")


# ── 四测试判定 ──────────────────────────────────────────────────
def _run(sharpe: float, sortino: float | None = None, expectancy: float | None = None,
         total_return: float = 0.1, n_trades: int = 100, win_rate: float = 0.5,
         pl: float | None = 1.2) -> dict:
    return {"sharpe": sharpe, "sortino": sortino if sortino is not None else sharpe,
            "expectancy": expectancy if expectancy is not None else 0.01,
            "total_return": total_return, "n_trades": n_trades,
            "win_rate": win_rate, "profit_loss_ratio": pl}


def test_verdicts_all_pass() -> None:
    runs = {
        "oos": _run(1.2, sortino=1.5, expectancy=0.02),                       # ≥0.8 / ≥1.0 / >0
        "cost": [_run(1.57), _run(1.2), _run(0.9), _run(0.7, total_return=0.05)],  # 末档 Sharpe≥0.5 收益>0
        "threshold": [_run(1.4), _run(1.57), _run(1.5), _run(1.45)],          # 各档≥0.7×0.30档 且 ≥3 档≥0.5
        "time": [_run(2.1), _run(2.4), _run(0.8), _run(0.3)],                 # 3+ 段为正
    }
    v = _pa.verdicts(runs)
    assert all(x["passed"] for x in v.values())


def test_verdicts_fail_each() -> None:
    v = _pa.verdicts({
        "oos": _run(0.5, sortino=0.6),                                        # 双低
        "cost": [_run(1.5), _run(1.0), _run(0.4), _run(-0.3, total_return=-0.1)],  # 末档负
        "threshold": [_run(1.5), _run(1.57), _run(0.1), _run(0.2)],           # 两档塌陷
        "time": [_run(2.1), _run(2.4), _run(0.2), _run(-0.9)],                # 2025-26 段深负
    })
    assert v["oos"]["passed"] is False
    assert v["cost"]["passed"] is False
    assert v["threshold"]["passed"] is False
    assert v["time"]["passed"] is False
    assert all("evidence" in x for x in v.values())


def test_verdicts_time_needs_three_positive_segments() -> None:
    v = _pa.verdicts({
        "oos": _run(1.2),
        "cost": [_run(1.5), _run(1.2), _run(0.9), _run(0.7, total_return=0.05)],
        "threshold": [_run(1.4), _run(1.57), _run(1.5), _run(1.45)],
        "time": [_run(2.1), _run(-0.4), _run(0.8), _run(0.3)],                # 3 段正且无段<-0.5 → 过
    })
    assert v["time"]["passed"] is True
    v2 = _pa.verdicts({
        "oos": _run(1.2),
        "cost": [_run(1.5), _run(1.2), _run(0.9), _run(0.7, total_return=0.05)],
        "threshold": [_run(1.4), _run(1.57), _run(1.5), _run(1.45)],
        "time": [_run(2.1), _run(1.2), _run(-0.3), _run(-0.9)],               # 最差段 -0.9 < -0.5 → 不过
    })
    assert v2["time"]["passed"] is False