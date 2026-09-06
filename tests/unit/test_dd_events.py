"""DD 事件日志 + 实时分析轻量 DD 跟踪器 单测。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_log_and_recent_roundtrip(monkeypatch, tmp_path):
    """log_dd_event 追加写 JSONL，recent_events 返回新→旧。"""
    import web.dd_events as de

    logf = tmp_path / "dd_gate_events.jsonl"
    monkeypatch.setattr(de, "LOG_FILE", logf)
    de.log_dd_event("paper", symbol="BTCUSDT", timeframe="5m", policy_id="dd+chandelier",
                    event="熔断", dd_pct=-4.2857, mark=100.5, peak=105.0,
                    prev_state=0, state=1)
    de.log_dd_event("paper", symbol="BTCUSDT", timeframe="5m", policy_id="dd+chandelier",
                    event="收复", dd_pct=-0.381, mark=104.6, peak=105.0,
                    prev_state=1, state=0)
    evs = de.recent_events(limit=10)
    assert len(evs) == 2
    assert evs[0]["event"] == "收复"       # 新→旧
    assert evs[1]["event"] == "熔断"
    assert evs[1]["dd_pct"] == -4.2857
    assert evs[1]["policy_id"] == "dd+chandelier"
    # 文件不存在 → 空
    monkeypatch.setattr(de, "LOG_FILE", tmp_path / "nope.jsonl")
    assert de.recent_events() == []


def test_log_never_raises(monkeypatch, tmp_path):
    import web.dd_events as de

    logf = tmp_path / "x" / "sub" / "dd_gate_events.jsonl"
    monkeypatch.setattr(de, "LOG_FILE", logf)  # 父目录不存在也会自动创建
    de.log_dd_event("realtime", symbol="X", timeframe="1m", policy_id="dd",
                    event="熔断", dd_pct=-3.0, mark=97.0, peak=100.0)
    assert logf.exists()
    # 坏参数不抛
    de.log_dd_event("realtime", symbol="X", timeframe="1m", policy_id="dd",
                    event="收复", dd_pct="bad", mark=None, peak=None)  # type: ignore[arg-type]


def test_realtime_tracker_transitions():
    """实时分析轻量 DD 跟踪：随收盘价走 熔断→收复 阶梯并记录转移。"""
    import web.dd_events as de
    import web.realtime_manager as rt

    from web.realtime_manager import WatchTask

    recorded = []

    def fake_log(scope, **kw):  # noqa: ANN001
        recorded.append((scope, kw.get("event"), kw.get("prev_state"), kw.get("state")))

    de.log_dd_event = fake_log  # _dd_track_step 在调用时 from web.dd_events import log_dd_event → 打模块属性即可

    task = WatchTask(
        id="binance:BTCUSDT:5m:best", source="binance", symbol="BTCUSDT",
        timeframe="5m", strategy_file="s.json", strategy_name="best",
        formula=[1], vocab_version=None, strategy_symbol="BTCUSDT",
        strategy_timeframe="5m", best_score=1.0, cadence_s=30,
        policy_id="dd+chandelier",
    )
    # 平稳 → 播种峰值，无转移
    mgr = rt.RealtimeManager()
    task.last_close = 105.0
    mgr._dd_track_step(task)
    assert task.dd_gate == 0 and task.dd_peak == 105.0
    # 急跌 -4.3% → 熔断（转移 0→1）
    task.last_close = 100.5
    mgr._dd_track_step(task)
    assert task.dd_gate == 1
    assert task.dd_pct is not None and task.dd_pct < -4.0
    # 收复 -0.38% → 闸恢复（转移 1→0）
    task.last_close = 104.6
    mgr._dd_track_step(task)
    assert task.dd_gate == 0
    assert recorded == [
        ("realtime", "熔断", 0, 1),
        ("realtime", "收复", 1, 0),
    ]
    # 非 dd 方案不跟踪（无峰值累计）
    task2 = WatchTask(
        id="x", source="binance", symbol="BTCUSDT", timeframe="5m",
        strategy_file="s", strategy_name="b", formula=[1], vocab_version=None,
        strategy_symbol=None, strategy_timeframe=None, best_score=None,
        cadence_s=30, policy_id="signal",
    )
    task2.last_close = 99.0
    mgr._dd_track_step(task2)
    assert task2.dd_gate == 0 and task2.dd_peak is None