"""矩阵单组合资金曲线工具单测：滚动夏普（向量化 vs 朴素循环）、降采样、平段裁剪。"""
from __future__ import annotations

import math

import numpy as np

from web.hold_matrix_curves import downsample, rolling_sharpe, trim_leading_flat


def _naive_rolling(pnl, w, ppy) -> np.ndarray:
    p = np.asarray(pnl, dtype=float)
    n = p.size
    out = np.full(n, np.nan)
    for t in range(w - 1, n):
        seg = p[t - w + 1 : t + 1]
        sd = seg.std()
        if sd > 1e-12:
            out[t] = seg.mean() / sd * math.sqrt(ppy)
    return out


def test_rolling_sharpe_matches_naive() -> None:
    rng = np.random.default_rng(7)
    pnl = rng.normal(0.001, 0.01, 800)
    got = rolling_sharpe(pnl, window=60, ppy=1.0)
    exp = _naive_rolling(pnl, 60, 1.0)
    np.testing.assert_allclose(got[59:], exp[59:], rtol=1e-10, atol=1e-12)
    assert np.isnan(got[:59]).all()  # 不足窗口为 NaN


def test_rolling_sharpe_flat_segment_nan() -> None:
    # 三段：200 根全 0 → 200 根恒 0.02 → 100 根全 0。
    # 完整窗口落在“纯平台”内时 std=0 → NaN；只有跨越边界的窗口才有值。
    pnl = np.concatenate([np.zeros(200), np.full(200, 0.02), np.zeros(100)])
    out = rolling_sharpe(pnl, window=40, ppy=105195.0)
    assert np.isnan(out[:200]).all()        # 窗口完全在 0 平台内
    assert not np.isnan(out[200:239]).any()  # 跨越 0→0.02 边界
    assert np.isnan(out[239:400]).all()     # 窗口完全在 0.02 平台内
    assert not np.isnan(out[400:439]).any()  # 跨越 0.02→0 边界
    assert np.isnan(out[439:]).all()        # 窗口完全在尾部 0 平台内


def test_downsample_keeps_endpoints_and_bounds() -> None:
    a = np.arange(10_000, dtype=float)
    d = downsample(a, max_pts=900)
    assert d.size <= 900
    assert d[0] == 0.0 and d[-1] == 9999.0
    d2 = downsample(a, max_pts=50_000)
    assert d2.size == a.size


def test_trim_leading_flat() -> None:
    eq = np.array([1.0] * 500 + [1.0, 1.01, 1.02, 1.015])
    assert trim_leading_flat(eq) == 500  # 首个变动点(501)的前一格
    assert trim_leading_flat(np.array([1.0, 1.0])) == 0


def test_dd_intervals_merges_and_closes() -> None:
    from web.hold_matrix_curves import dd_intervals_from_events

    events = [
        {"bar": 100, "action": "exit", "state": 1, "dd_pct": -3.0},
        {"bar": 105, "action": "exit", "state": 2, "dd_pct": -5.2},   # 深档（state 升 2）并入同段
        {"bar": 120, "action": "enter", "state": 0, "dd_pct": -1.0},  # 收复 → 结束区间
        {"bar": 300, "action": "exit", "state": 1, "dd_pct": -2.5},   # 未收复 → end=None
    ]
    ivs = dd_intervals_from_events(events)
    assert len(ivs) == 2
    a, b = ivs
    assert (a["start"], a["end"]) == (100, 119)
    assert a["state"] == 2 and a["dd_pct"] == -5.2   # 段内取最深
    assert b["start"] == 300 and b["end"] is None
    assert dd_intervals_from_events([]) == []
    # 纯收复（无熔断）不产生区间
    assert dd_intervals_from_events([{"bar": 10, "state": 0}]) == []


def test_pnl_quintiles_buckets_and_segments() -> None:
    from web.hold_matrix_curves import pnl_quintiles

    # 确定性：大部分 0（空仓）+ 一段稳步上涨 + 一段急跌
    pnl = np.zeros(1000)
    pnl[200:400] = 0.001           # 200 根小赚
    pnl[400:500] = 0.0002          # 100 根小赚（低桶）
    pnl[700:750] = -0.01           # 50 根急亏
    pnl[900:950] = 0.02            # 50 根大赚
    q = pnl_quintiles(pnl)
    assert len(q["buckets"]) == 1000
    assert set(q["buckets"]) <= {0, 1, 2, 3, 4}
    st = {s["bucket"]: s for s in q["stats"]}
    assert len(st) == 5
    assert abs(sum(s["sum"] for s in q["stats"]) - float(pnl.sum())) < 1e-9
    # 最亏段应落在 700..749 附近（Kadane 找到的连续段，允许起点略前）
    w = q["worst_segment"]
    assert w["cum"] < -0.45 and w["bars"] >= 50
    # 最赚段应落在 900..949
    b = q["best_segment"]
    assert b["cum"] > 0.9 and b["bars"] >= 50
    # 空输入安全
    q0 = pnl_quintiles(np.array([]))
    assert q0["buckets"] == [] and q0["best_segment"] is None


def test_pnl_quintiles_sampled_alignment() -> None:
    from web.hold_matrix_curves import downsample_indices, pnl_quintiles

    rng = np.random.default_rng(3)
    pnl = rng.normal(0, 0.005, 3000)
    idx = downsample_indices(3000)
    q = pnl_quintiles(pnl, idx=idx)
    assert len(q["buckets"]) == len(idx)
    # 采样点的分桶必须与全量分位切法一致（同一 searchsorted 逻辑）
    qs = np.quantile(pnl, [0.2, 0.4, 0.6, 0.8])
    exp = np.searchsorted(qs, pnl, side="right").astype(int)
    assert q["buckets"] == [int(exp[i]) for i in idx]
