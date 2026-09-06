"""实时监控新能力单测：因子僵化跟踪（跨新 bar）+ 可配置偏离入场告警。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web.data_sources.base import Bar  # noqa: E402


def _task(rt, policy_id: str = "signal", direction: str | None = None):
    from web.realtime_manager import WatchTask

    return WatchTask(
        id="binance:BTCUSDT:5m:best", source="binance", symbol="BTCUSDT",
        timeframe="5m", strategy_file="s.json", strategy_name="best",
        formula=[1], vocab_version=None, strategy_symbol="BTCUSDT",
        strategy_timeframe="5m", best_score=1.0, cadence_s=30,
        policy_id=policy_id, direction=direction,
    )


def _bar(ts: int, close: float) -> Bar:
    return Bar(ts=ts, open=close, high=close + 0.5, low=close - 0.5,
               close=close, volume=1.0)


def _signal(direction: str, factor: float) -> dict:
    return {
        "state": "ok", "direction": direction, "strength": 1.0,
        "position": 1.0, "factor_value": factor, "message": "",
        "bars_used": 10,
    }


def test_factor_flat_tracking_across_new_bars(monkeypatch) -> None:
    """跨新 bar 因子几乎不变才累计；同一根 bar 内不累计；价格移动单独计数。"""
    import web.realtime_manager as rt

    mgr = rt.RealtimeManager()
    t = _task(rt, policy_id="signal", direction="LONG")
    base_ts = 1_700_000_000
    calls = {"n": 0}
    closes_by_call = [100.0] * 6 + [101.0, 102.0]  # 前 6 根价格不动，后 2 根价格动

    def fake_bars(source, symbol, timeframe):  # noqa: ANN001
        i = calls["n"]
        calls["n"] += 1
        return [_bar(base_ts + i * 300, closes_by_call[min(i, len(closes_by_call) - 1)])]

    monkeypatch.setattr(rt, "fetch_cached_bars", fake_bars)
    monkeypatch.setattr(rt, "fetch_live_price", lambda *a, **k: None)
    monkeypatch.setattr(rt, "evaluate_signal", lambda formula, raw, **kw: _signal("LONG", 1.3919))

    for _ in range(6):
            mgr._evaluate_task(t)
    # 第一根播种，之后 5 根“新 bar”比较都几乎不变
    assert t.factor_flat_run == 5
    assert t.price_moved_run == 0  # 收盘价恒 100
    sq = t._signal_quality()
    assert sq["factor_flat_run"] == 5
    assert sq["price_moved_run"] == 0

    # 价格动（101→102）但因子仍不动 → price_moved_run 涨，flat 继续累计
    mgr._evaluate_task(t)
    mgr._evaluate_task(t)
    assert t.factor_flat_run == 7
    assert t.price_moved_run == 2

    # 因子终于变化（>0.001% 相对）→ flat 重置
    monkeypatch.setattr(rt, "evaluate_signal", lambda formula, raw, **kw: _signal("LONG", 1.40))
    mgr._evaluate_task(t)
    assert t.factor_flat_run == 0


def test_same_bar_no_accumulation(monkeypatch) -> None:
    """同一根 bar 上多次评估（输入相同）不增加 flat 计数。"""
    import web.realtime_manager as rt

    mgr = rt.RealtimeManager()
    t = _task(rt, policy_id="signal", direction="LONG")

    monkeypatch.setattr(rt, "fetch_cached_bars",
                        lambda *a, **k: [_bar(1_700_000_000, 100.0)])
    monkeypatch.setattr(rt, "fetch_live_price", lambda *a, **k: None)
    monkeypatch.setattr(rt, "evaluate_signal", lambda formula, raw, **kw: _signal("LONG", 1.3919))
    for _ in range(3):
        mgr._evaluate_task(t)
    assert t.factor_flat_run == 0  # 同一根 bar 三次评估 → 无跨 bar 比较


def test_deviation_alert_fires_once_and_recovers(monkeypatch) -> None:
    """偏离入场参考超阈值推一次；回到阈值一半内解除；再越线再推。"""
    import web.realtime_manager as rt
    import web.settings as st

    fired: list[tuple] = []

    class _Mgr(rt.RealtimeManager):
        def _notify_deviation(self, task, px, dev_pct, thr):  # noqa: ANN001
            fired.append((task.id, px, round(dev_pct, 2), thr))

    mgr = _Mgr()
    t = _task(rt, policy_id="chandelier", direction="LONG")
    t.dir_entry_price = 100.0
    t.live_price = 101.0  # +1%

    monkeypatch.setattr(st, "load_settings", lambda: {"rt_alert_dev_pct": 1.0})
    mgr._check_dev_alert(t)
    assert len(fired) == 1 and fired[0][1] == 101.0 and fired[0][2] == 1.0

    # 更偏离但仍在同一“越线段”内 → 不重复推
    t.live_price = 103.0
    mgr._check_dev_alert(t)
    assert len(fired) == 1

    # 回到阈值一半以内（+0.4% ≤ 0.5%）→ 解除去重
    t.live_price = 100.4
    mgr._check_dev_alert(t)
    assert len(fired) == 1

    # 再次越线 → 新告警
    t.live_price = 102.0  # +2%
    mgr._check_dev_alert(t)
    assert len(fired) == 2
    assert t.dev_alerted is True


def test_deviation_disabled_or_no_direction(monkeypatch) -> None:
    import web.realtime_manager as rt
    import web.settings as st

    fired: list[int] = []

    class _Mgr(rt.RealtimeManager):
        def _notify_deviation(self, task, px, dev_pct, thr):  # noqa: ANN001
            fired.append(1)

    monkeypatch.setattr(st, "load_settings", lambda: {"rt_alert_dev_pct": 0.0})
    mgr = _Mgr()
    t = _task(rt, policy_id="chandelier", direction="LONG")
    t.dir_entry_price = 100.0
    t.live_price = 150.0
    mgr._check_dev_alert(t)
    assert fired == []  # 阈值 0 = 关闭

    monkeypatch.setattr(st, "load_settings", lambda: {"rt_alert_dev_pct": 1.0})
    t2 = _task(rt, policy_id="chandelier", direction="FLAT")
    t2.dir_entry_price = 100.0
    t2.live_price = 102.0
    mgr._check_dev_alert(t2)
    assert fired == []  # 无方向不告警


def test_direction_anchor_resets_on_flip(monkeypatch) -> None:
    """方向翻转时重新锚定 dir_entry_price（走 _evaluate_task 全路径）。"""
    import web.realtime_manager as rt

    mgr = rt.RealtimeManager()
    t = _task(rt, policy_id="signal", direction=None)
    base_ts = 1_700_000_000
    steps = [
        (base_ts, 100.0, "LONG", 1.0),
        (base_ts + 300, 105.0, "LONG", 1.05),
        (base_ts + 600, 104.0, "SHORT", -1.0),
    ]
    state = {"i": 0}

    def fake_bars(source, symbol, timeframe):  # noqa: ANN001
        ts, close, _d, _f = steps[min(state["i"], len(steps) - 1)]
        return [_bar(ts, close)]

    def sig(formula, raw, **kw):  # noqa: ANN001
        _d, _f = None, None
        _ts, _c, _d, _f = steps[min(state["i"], len(steps) - 1)]
        state["i"] += 1
        return _signal(_d, _f)

    monkeypatch.setattr(rt, "fetch_cached_bars", fake_bars)
    monkeypatch.setattr(rt, "fetch_live_price", lambda *a, **k: None)
    monkeypatch.setattr(rt, "evaluate_signal", sig)

    mgr._evaluate_task(t)  # 首根：LONG，锚定 100
    assert t.dir_entry_price == 100.0
    mgr._evaluate_task(t)  # 第二根：仍 LONG → 锚定不变
    assert t.dir_entry_price == 100.0
    mgr._evaluate_task(t)  # 第三根：翻转 SHORT → 重新锚定 104
    assert t.dir_entry_price == 104.0
    assert t.factor_flat_run == 0  # 因子 1.0→1.05→-1.0 每次都大幅变化

def test_anchor_dev_history_records_per_bar_and_resets_on_flip(monkeypatch) -> None:
    """入场锚偏离历史：每根新收盘 bar 记一次偏离%，方向翻转时清空重锚。"""
    import web.realtime_manager as rt

    mgr = rt.RealtimeManager()
    t = _task(rt, policy_id="signal", direction=None)
    base_ts = 1_700_000_000
    steps = [
        (base_ts, 100.0, "LONG", 1.0),        # 1: 锚定 100
        (base_ts + 300, 101.0, "LONG", 1.0),  # 2: 偏离 +1%
        (base_ts + 600, 102.0, "LONG", 1.0),  # 3: 偏离 +2%
        (base_ts + 900, 104.0, "SHORT", -1.0),  # 4: 翻转 SHORT → 清空，锚定 104
        (base_ts + 1200, 104.5, "SHORT", -1.0),  # 5: 偏离 +0.4808%
    ]
    state = {"i": 0}

    def fake_bars(source, symbol, timeframe):  # noqa: ANN001
        ts, close, _d, _f = steps[min(state["i"], len(steps) - 1)]
        return [_bar(ts, close)]

    def sig(formula, raw, **kw):  # noqa: ANN001
        _ts, _c, _d, _f = steps[min(state["i"], len(steps) - 1)]
        state["i"] += 1
        return _signal(_d, _f)

    monkeypatch.setattr(rt, "fetch_cached_bars", fake_bars)
    monkeypatch.setattr(rt, "fetch_live_price", lambda *a, **k: None)
    monkeypatch.setattr(rt, "evaluate_signal", sig)

    mgr._evaluate_task(t)  # 1 LONG 锚 100，偏离 0%
    mgr._evaluate_task(t)  # 2 101 → +1%
    mgr._evaluate_task(t)  # 3 102 → +2%
    assert list(t.anchor_dev_hist) == [0.0, 1.0, 2.0]
    assert t.dir_entry_price == 100.0

    mgr._evaluate_task(t)  # 4 翻转 SHORT → 清空重锚 104
    assert t.dir_entry_price == 104.0
    assert list(t.anchor_dev_hist) == [0.0]

    mgr._evaluate_task(t)  # 5 104.5 → 偏离 0.480769…
    assert abs(t.anchor_dev_hist[-1] - (104.5 / 104.0 - 1.0) * 100.0) < 1e-6
    pub = t.to_public()
    assert isinstance(pub["anchor_dev_hist"], list)


def test_stale_feishu_fires_only_on_hard_plateau_and_once(monkeypatch) -> None:
    """硬钝化告警：连续 >=N 根几乎不变 + 轨迹平台恒定才推一次；因子变化后重置允许再推。"""
    import web.realtime_manager as rt
    import web.settings as st

    fired: list[tuple] = []

    class _Mgr(rt.RealtimeManager):
        def _notify_stale(self, task, flat_run, factor_val):  # noqa: ANN001
            fired.append((task.id, flat_run, round(factor_val, 6)))

    mgr = _Mgr()
    t = _task(rt, policy_id="signal", direction="LONG")
    base_ts = 1_700_000_000

    def mk(factor: float, offset: int = 0):
        # 每步一根新 bar，因子固定（平台恒定）；offset 保证 ts 单调递增
        state = {"i": 0}

        def fake_bars(source, symbol, timeframe):  # noqa: ANN001
            i = state["i"]
            state["i"] += 1
            return [_bar(base_ts + (offset + i) * 300, 100.0)]

        def sig(formula, raw, **kw):  # noqa: ANN001
            return _signal("LONG", factor)

        return fake_bars, sig

    monkeypatch.setattr(st, "load_settings", lambda: {"rt_alert_stale_bars": 3})
    fb, sg = mk(1.3919, offset=0)
    monkeypatch.setattr(rt, "fetch_cached_bars", fb)
    monkeypatch.setattr(rt, "evaluate_signal", sg)
    monkeypatch.setattr(rt, "fetch_live_price", lambda *a, **k: None)

    for _ in range(5):  # 第1根播种，之后 flat_run 逐步 1→4
        mgr._evaluate_task(t)
    assert t.factor_flat_run >= 3
    # 只推了一次（达到 3 后立即推，后续 bar 去重）
    assert len(fired) == 1 and fired[0][1] >= 3 and fired[0][2] == 1.3919
    assert t._stale_pushed is True

    # 因子恢复变化（非硬钝化漂移段也会先 flat 归零）→ 允许再次提醒
    fb2, sg2 = mk(1.40, offset=10)
    monkeypatch.setattr(rt, "fetch_cached_bars", fb2)
    monkeypatch.setattr(rt, "evaluate_signal", sg2)
    mgr._evaluate_task(t)
    assert t.factor_flat_run == 0 and t._stale_pushed is False

    # 再次平台恒定到阈值 → 推第二条
    fb3, sg3 = mk(1.40, offset=20)
    monkeypatch.setattr(rt, "fetch_cached_bars", fb3)
    monkeypatch.setattr(rt, "evaluate_signal", sg3)
    for _ in range(4):
        mgr._evaluate_task(t)
    assert len(fired) == 2


def test_stale_disabled_when_threshold_zero(monkeypatch) -> None:
    import web.realtime_manager as rt
    import web.settings as st

    fired: list[int] = []

    class _Mgr(rt.RealtimeManager):
        def _notify_stale(self, task, flat_run, factor_val):  # noqa: ANN001
            fired.append(1)

    mgr = _Mgr()
    t = _task(rt, policy_id="signal", direction="LONG")
    base_ts = 1_700_000_000
    state = {"i": 0}

    def fake_bars(source, symbol, timeframe):  # noqa: ANN001
        i = state["i"]
        state["i"] += 1
        return [_bar(base_ts + i * 300, 100.0)]

    def sig(formula, raw, **kw):  # noqa: ANN001
        return _signal("LONG", 1.3919)

    monkeypatch.setattr(st, "load_settings", lambda: {"rt_alert_stale_bars": 0})
    monkeypatch.setattr(rt, "fetch_cached_bars", fake_bars)
    monkeypatch.setattr(rt, "evaluate_signal", sig)
    monkeypatch.setattr(rt, "fetch_live_price", lambda *a, **k: None)

    for _ in range(6):
        mgr._evaluate_task(t)
    assert t.factor_flat_run >= 3
    assert fired == []  # 0 = 关闭
