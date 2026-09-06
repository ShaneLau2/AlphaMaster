"""信号质量角标统计的单元测试（web.realtime_manager.WatchTask._signal_quality）。"""

from collections import deque

from web.realtime_manager import WatchTask


def _task() -> WatchTask:
    return WatchTask(
        id="binance:BTCUSDT:5m:best_BTCUSDT",
        source="binance",
        symbol="BTCUSDT",
        timeframe="5m",
        strategy_file="strategies/best_BTCUSDT.json",
        strategy_name="best_BTCUSDT",
        formula=[56, 105, 119, 94, 69, 71, 76, 105],
        vocab_version="v1",
        strategy_symbol="BTCUSDT",
        strategy_timeframe="M5",
        best_score=2.98,
        cadence_s=30,
    )


def test_empty_quality():
    sq = _task()._signal_quality()
    assert sq["window"] == 0
    assert sq["dir_run"] == 0
    assert sq["long_pct"] is None
    assert sq["factor_pct"] is None


def test_direction_percentages_and_run():
    t = _task()
    # 模拟 20 次收盘判断：15 LONG + 5 SHORT（无 FLAT），当前连续 LONG 4 根
    t.dir_hist = deque(["LONG"] * 11 + ["SHORT"] * 5 + ["LONG"] * 4, maxlen=1000)
    t.dir_run = 4
    sq = t._signal_quality()
    assert sq["window"] == 20
    assert sq["long_pct"] == 75.0
    assert sq["short_pct"] == 25.0
    assert sq["flat_pct"] == 0.0
    assert sq["dir_run"] == 4


def test_flat_percentages():
    t = _task()
    t.dir_hist = deque(["LONG"] * 5 + ["FLAT"] * 3 + ["SHORT"] * 2, maxlen=1000)
    sq = t._signal_quality()
    assert sq["window"] == 10
    assert sq["long_pct"] == 50.0
    assert sq["flat_pct"] == 30.0
    assert sq["short_pct"] == 20.0


def test_factor_percentile():
    t = _task()
    t.factor_value = 1.5
    t.factor_buf = deque([-2.0, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0], maxlen=2000)
    t.dir_hist = deque(["LONG"] * 7, maxlen=1000)  # 方向历史非空时才计算分位
    sq = t._signal_quality()
    # 7 个值中 ≤1.5 的有 5 个 → 71.4%
    assert sq["factor_pct"] == 71.4
    # 无因子值时不崩溃
    t.factor_value = None
    assert _task()._signal_quality()["factor_pct"] is None


def test_dir_run_accumulation_semantics():
    """dir_run 语义：同向 +1、翻转重置 1（由 _evaluate_task 维护，这里验证展示字段）。"""
    t = _task()
    t.dir_hist = deque(["LONG"] * 3, maxlen=1000)
    t.dir_run = 3
    assert t._signal_quality()["dir_run"] == 3
    # 翻转后的首个 LONG 应显示 1（模拟：方向从 SHORT 变 LONG，仅记录一次）
    t2 = _task()
    t2.dir_hist = deque(["SHORT"] * 6 + ["LONG"], maxlen=1000)
    t2.dir_run = 1
    assert t2._signal_quality()["dir_run"] == 1