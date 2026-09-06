"""实时分析 status() 并发现价刷新：预算封顶 + 并行不串行 单测。

背景：/api/realtime/status 原来串行 fetch 每个监控项的现价，TTL 刚过时
总耗时 ≈ Σ(单项网络延迟)，多个监控项会把轮询/交互拖慢到秒级；改为
_refresh_live_prices 用独立线程池并行 + 整批预算封顶。本文件锁定两个性质：
1) 慢于预算的任务不会拖住 status（预算封顶）；
2) 快于预算的多个任务并行完成（总耗时≈max，而非 Σ）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import web.realtime_manager as rt  # noqa: E402


def _task(rtm: rt, sym: str) -> rt.WatchTask:
    return rtm.WatchTask(
        id=f"binance:{sym}:5m:best", source="binance", symbol=sym,
        timeframe="5m", strategy_file="s.json", strategy_name="best",
        formula=[1], vocab_version=None, strategy_symbol=sym,
        strategy_timeframe="5m", best_score=1.0, cadence_s=30,
        policy_id="signal",
    )


def _install_mgr(monkeypatch, n: int, delay: float, budget: float):
    """构造带 n 个任务的 manager，patch 拉价动作为 sleep delay 并写回。"""
    monkeypatch.setattr(rt, "_PRICE_REFRESH_BUDGET_S", budget)
    mgr = rt.RealtimeManager()
    tasks = [_task(rt, f"BTCUSDT{i}") for i in range(n)]

    def fake_pull(t: rt.WatchTask) -> None:
        time.sleep(delay)
        t.live_price = 100.0 + len(t.symbol)
        t.live_price_ts = int(time.time())

    monkeypatch.setattr(rt, "_pull_live_price", fake_pull)
    return mgr, tasks


def test_parallel_completes_faster_than_serial_total(monkeypatch) -> None:
    """3 个各睡 0.7s 的任务：并行总耗时≈0.7s 而非 2.1s，且全部写回现价。"""
    mgr, tasks = _install_mgr(monkeypatch, n=3, delay=0.7, budget=1.8)
    t0 = time.monotonic()
    mgr._refresh_live_prices(tasks)
    elapsed = time.monotonic() - t0
    # 串行需 2.1s > 预算 1.8s（第三个任务写不回去）；只有并行能在预算内全部完成
    assert elapsed < 1.4, f"疑似串行：elapsed={elapsed:.2f}s"
    assert all(getattr(t, "live_price", None) is not None for t in tasks), "并行任务未全部写回"


def test_budget_caps_wait_when_tasks_slower_than_budget(monkeypatch) -> None:
    """任务睡 1.0s > 预算 0.3s：status 侧必须在预算附近返回，不被任务拖住。"""
    mgr, tasks = _install_mgr(monkeypatch, n=2, delay=1.0, budget=0.3)
    t0 = time.monotonic()
    mgr._refresh_live_prices(tasks)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.9, f"预算未封顶：elapsed={elapsed:.2f}s"
    # 后台线程仍在收尾；等其结束避免跨用例残留
    mgr._price_pool.shutdown(wait=True)
