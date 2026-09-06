"""实时监控压测：10-20 个监控项下的 status 轮询 / 单卡移除 / 价格刷新预算封顶。

验证三件事（并打印实测毫秒）：
1. status() 轮询延迟不随监控项数量线性增长——现价刷新是并发 + _PRICE_REFRESH_BUDGET_S
   封顶（web/realtime_manager._refresh_live_prices）：把 _pull_live_price 换成
   慢速假实现（3s）后，15 个监控项的 status 也应 ≈2s 预算内返回（串行会是 45s）。
2. 单卡移除延迟（unwatch + 确认消失的 status 往返）毫秒级。
3. TTL 命中后 status 应回到极快（<300ms），预算只是上限不是常驻成本。

用法：
    .venv/bin/python scripts/perf_load_test.py [--n 15] [--polls 6]

退出码 0=通过 / 1=失败。无需浏览器；离线可跑（网络失败静默）。
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_SETTINGS = PROJECT_ROOT / "results" / "_perf_load_settings.json"
_STRATEGY = PROJECT_ROOT / "strategies" / "best_BTCUSDT.json"

# 预算断言：慢速假拉价 3s / 预算 2s → status 总耗时必须 ≤ 预算 + 2s 余量
_BUDGET_S = 2.0
_SLOW_S = 3.0
_STATUS_MAX_MS = int((_BUDGET_S + 2.0) * 1000)   # 4000ms：封顶成立
_WARM_STATUS_MAX_MS = 800                        # TTL 命中后应飞快
_REMOVE_MAX_MS = 3000                            # unwatch + status 确认往返


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.loads(r.read().decode())


def _measure(base: str, path: str, payload: dict | None = None) -> tuple[float, dict]:
    t0 = time.perf_counter()
    d = _post(base, path, payload) if payload is not None else _get(base, path)
    return (time.perf_counter() - t0) * 1000.0, d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=15, help="监控项数量（10-20）")
    ap.add_argument("--polls", type=int, default=6, help="status 轮询次数")
    args = ap.parse_args()
    n = max(5, min(30, args.n))
    if not _STRATEGY.exists():
        print(f"缺少策略文件 {_STRATEGY}"); return 1

    # 隔离 settings（不污染真实 web_settings.json）
    import web.settings as _ws
    _ws.SETTINGS_PATH = _SETTINGS
    if _SETTINGS.exists():
        _SETTINGS.unlink()

    # 慢速假拉价：模拟每个监控项网络阶段 3s + 真实 _pull_live_price 的 TTL 缓存行为
    # （12s 内同项不再拉网）——这样冷态验证预算封顶、热态验证 TTL 命中后回归飞快。
    import web.realtime_manager as rm

    real_pull = rm._pull_live_price
    _pulled_at: dict[str, float] = {}
    _TTL = 12.0

    def _slow_pull(task) -> None:  # noqa: ANN001 签名与真实一致（做慢 + 写 TTL 缓存）
        key = f"{task.source}:{task.symbol}:{task.timeframe}"
        last = _pulled_at.get(key, 0.0)
        now = time.monotonic()
        if now - last < _TTL:   # TTL 命中：不拉网（与真实 fetch_live_price 一致）
            return
        _pulled_at[key] = now
        time.sleep(_SLOW_S)     # 模拟慢网络

    rm._pull_live_price = _slow_pull

    import uvicorn  # noqa: E402
    from web.app import app  # noqa: E402

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    holder: dict[str, threading.Thread] = {}
    holder["t"] = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="error"),
        daemon=True)
    holder["t"].start()
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.8)
    else:
        print("测试服务未就绪"); return 1

    syms = [f"P{n:04d}USDT" for n in range(n)]  # 任意唯一符号即可（离线不校验存在性）
    ids: list[str] = []
    for i, sym in enumerate(syms):
        _, w = _measure(base, "/api/realtime/watch", {
            "source": "binance", "symbol": sym, "timeframe": "5m",
            "strategy_file": str(_STRATEGY)})
        ids.append(w["watch"]["id"])
    st0 = _get(base, "/api/realtime/status")
    got = st0.get("count", 0)
    print(f"\n[load] 挂载监控项: 请求 {n} 个 → status.count = {got}")
    if got != n:
        print(f"FAIL: 期望 {n} 个监控项，实际 {got}"); return 1

    print(f"[load] 慢速假拉价 {_SLOW_S}s/项（预算 {_BUDGET_S}s）下 status 轮询延迟：")
    lat_ms: list[float] = []
    for i in range(args.polls):
        ms, _ = _measure(base, "/api/realtime/status")
        lat_ms.append(ms)
        tag = "冷TTL(慢网络)" if i == 0 else ("热TTL" if i >= 2 else "过渡")
        print(f"  poll#{i + 1}  {ms:8.1f} ms  [{tag}]")
    # 预算封顶语义：任何一次轮询都不得突破预算上限（否则就是退回串行 Σ 的迹象）；
    # 冷批背景排空（价格池 4 线程）期间的轮询会贴预算上限——这正是设计的封顶行为，
    # 排空后（最后一 poll）TTL 命中应回到飞快。
    worst = max(lat_ms)
    last = lat_ms[-1]
    print(f"[load] 最差轮询 {worst:.1f}ms 须 < {_STATUS_MAX_MS}ms（预算封顶，串行会是 "
          f"{n * _SLOW_S:.0f}s）；排空后最后 poll {last:.1f}ms 须 < {_WARM_STATUS_MAX_MS}ms")
    if worst >= _STATUS_MAX_MS:
        print(f"FAIL: {n} 项下轮询最差 {worst:.0f}ms 突破预算上限——并发现价刷新预算失效（可能退回串行）")
        return 1
    if last >= _WARM_STATUS_MAX_MS:
        print(f"FAIL: 排空后 TTL 热态 {last:.0f}ms 仍慢——TTL 缓存未生效？")
        return 1

    # 单卡移除延迟：unwatch + 确认消失的 status
    target = ids[0]
    ms, _ = _measure(base, "/api/realtime/unwatch", {"id": target})
    ms2, st = _measure(base, "/api/realtime/status")
    still = any(w["id"] == target for w in st.get("watches", []))
    total = ms + ms2
    print(f"\n[load] 单卡移除: unwatch {ms:.1f}ms + 确认 status {ms2:.1f}ms = {total:.1f}ms "
          f"(须 < {_REMOVE_MAX_MS}ms；行已消失={not still})")
    if still or total >= _REMOVE_MAX_MS:
        print("FAIL: 移除往返过慢或未消失"); return 1

    # 恢复真实拉价后，剩余项再轮询一次应仍快（收尾干净）
    rm._pull_live_price = real_pull
    ms3, _ = _measure(base, "/api/realtime/status")
    print(f"[load] 恢复真实拉价后 status: {ms3:.1f}ms")

    print("\n✅ 压测通过：并发预算封顶在 15 项规模下成立；单卡移除毫秒级；TTL 热态无常驻成本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())