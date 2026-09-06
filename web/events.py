"""SSE 事件中枢（进程内 pub/sub + 订阅驱动的变更 watcher）。

角色：把「客户端轮询实时/模拟盘状态」改成服务端推送——
- 客户端连上 ``GET /api/events``（text/event-stream）后，后台 watcher 开始工作；
- watcher 定期对各域做**轻量指纹**比对，只在变化时才构建完整快照并推送
  （event 名 = 域，payload = 与既有 REST 同构的 status 快照，前端渲染函数原样复用）；
- 实时分析另有 ~10s 一次的现价刷新（复用 manager 的短 TTL 网络拉取）；
- 没有任何订阅者时 watcher 自动停止，服务端零额外开销；
- 10s 无名 data 心跳保活（前端可见，代理/网关不丢）；慢客户端丢弃旧事件只留最新，防内存膨胀。

设计约束：单 uvicorn worker（localhost 部署满足）。多 worker 部署时 SSE 只会连到
其中一个进程，前端 reconcile 轮询仍能兜底自愈——但推送将不再全量。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

MAX_QUEUE = 8            # 每个订阅队列缓存的事件条数（超出丢旧保新）
HEARTBEAT_S = 10         # 无事件时的心跳间隔（太慢会让部分代理/网关以 15s 空闲超时掐断长连接）
WATCH_INTERVAL_S = 3.0   # 各域指纹检查节奏
PRICE_REFRESH_S = 10.0   # 实时分析现价强制刷新节奏（内部有短 TTL，不重复打网络）
GRACE_EMPTY_S = 8.0      # 订阅者清零后 watcher 多存活时长（防频繁启停）

_loop: asyncio.AbstractEventLoop | None = None
_subs: set[asyncio.Queue[str]] = set()
_subs_lock = threading.Lock()
_watcher_task: asyncio.Task | None = None
_watcher_last_sig: dict[str, str] = {}      # 域 → 上次已推送快照的完整签名
_watcher_last_full: dict[str, float] = {}   # 域 → 上次全量快照时间（现价到期刷新用）


def _r(v: Any, nd: int = 4) -> str:
    """浮点字段指纹化（None 保持空串），避免微小浮点噪声触发推送。"""
    if v is None:
        return ""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


# ─────────────────────────── 接入事件循环 ───────────────────────────
def bind_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    global _loop
    if loop is not None:
        _loop = loop


def _ensure_loop() -> None:
    """懒绑定：无 startup 绑定时的兜底（subscribe 一定在 loop 线程内调用）。"""
    global _loop
    if _loop is None or _loop.is_closed():
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            pass


def _frame(evt: str, payload: Any) -> str:
    return f"event: {evt}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


# ─────────────────────────── 订阅管理 ──────────────────────────────
async def subscribe() -> asyncio.Queue[str]:
    _ensure_loop()
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_QUEUE)
    with _subs_lock:
        was_empty = not _subs  # 必须先判空再加：否则恒为 False，watcher 永不启动
        _subs.add(q)
    if was_empty:
        _maybe_start_watcher()
    return q


def unsubscribe(q: asyncio.Queue[str]) -> None:
    with _subs_lock:
        _subs.discard(q)
    if not _subs:
        # 记录清空时刻；稍后再真正停 watcher，避免「一开一关」抖动
        _schedule_reaper()


def subscriber_count() -> int:
    with _subs_lock:
        return len(_subs)


def publish(evt: str, payload: Any = None) -> None:
    """线程安全发布（可在任意线程调用；内部转到事件循环再分发）。"""
    _ensure_loop()
    loop = _loop
    if loop is None or loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(_dispatch, _frame(evt, payload))
    except RuntimeError:
        pass


def _dispatch(frame: str) -> None:
    dead: list[asyncio.Queue[str]] = []
    with _subs_lock:
        queues = list(_subs)
    for q in queues:
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            # 慢客户端：丢最旧一条再入队（保最新状态）
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                dead.append(q)
    if dead:
        with _subs_lock:
            for q in dead:
                _subs.discard(q)


async def event_generator():
    """SSE 生成器：首次推送 hello，之后转发事件/心跳。"""
    _ensure_loop()
    q = await subscribe()
    yield _frame("hello", {"ok": True, "server_time": time.time()})
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=HEARTBEAT_S)
                yield msg
            except asyncio.TimeoutError:
                # 心跳用无名 data 帧（无 event: 行）而非注释：注释在浏览器 EventSource
                # 里不可见，无法据此判断连接死活，代理/网关也可能不转发纯注释；
                # 无名 data 帧触发 message 事件，前端据此刷新 liveness。
                yield f"data: {json.dumps({'hb': time.time()}, ensure_ascii=False)}\n\n"
    except asyncio.CancelledError:  # 客户端断开/请求取消
        pass
    finally:
        unsubscribe(q)


# ─────────────────────── 域 watcher（轻指纹→变化才推） ──────────────────
def _realtime_probe_sig() -> str:
    """实时分析：只读内存任务字段的轻指纹（不含数组/曲线/网络）。"""
    from web.realtime_manager import realtime_manager as rt

    with rt._lock:
        bits = [str(rt._running), str(len(rt._tasks))]
        for t in sorted(rt._tasks.values(), key=lambda x: x.id):
            bits.append(
                "|".join(
                    [
                        t.id, t.state, str(t.direction),
                        _r(t.strength, 4), _r(t.factor_value, 5),
                        _r(t.live_price, 4), _r(t.last_close, 5),
                        str(t.last_bar_ts), str(t.message), str(t.dd_gate),
                        str(int(t.updated_at or 0)),
                    ]
                )
            )
    return "\n".join(bits)


def _paper_probe_sig() -> str:
    """模拟实盘：账户/持仓/成交/监控项轻指纹。"""
    from web.paper_manager import paper_manager as pp

    with pp._lock:
        bits = [
            str(pp._running), str(len(pp._watches)), str(len(pp._positions)),
            str(getattr(pp, "_seq", 0)), str(pp.n_trades),
            _r(pp.cash, 2), _r(pp.realized_pnl, 2),
        ]
        for t in sorted(pp._watches.values(), key=lambda x: x.id):
            bits.append(
                "|".join(
                    [
                        t.id, t.state, str(t.direction), _r(t.strength, 4),
                        str(t.last_bar_ts), _r(t.last_close, 6), str(t.message),
                    ]
                )
            )
        for wid in sorted(pp._positions):
            p = pp._positions[wid]
            bits.append(
                "|".join(
                    [
                        wid, str(p.get("side")), _r(p.get("qty"), 8),
                        _r(p.get("entry_price"), 6), _r(p.get("stop_price"), 6),
                        _r(p.get("target_price"), 6), str(p.get("bars_held")),
                    ]
                )
            )
        eqv = getattr(pp, "_eq_val", None)
        if eqv:
            bits.append(f"eq:{len(eqv)}:{_r(eqv[-1], 4)}")
        trades = getattr(pp, "_trades", None)
        if trades:
            first = trades[0]
            seq = first.get("seq") if isinstance(first, dict) else getattr(first, "seq", None)
            bits.append(f"tr:{str(seq)}:{len(trades)}")
    return "\n".join(bits)


def _full_realtime() -> dict[str, Any]:
    from web.realtime_manager import realtime_manager as rt

    st = rt.status()
    return {k: st.get(k) for k in ("running", "count", "watches", "server_time", "nearest_seconds_to_next")}


def _full_paper() -> dict[str, Any]:
    from web.paper_manager import paper_manager as pp

    return pp.status()


def _probe_once() -> dict[str, str]:
    """阻塞式探测（在 executor 线程跑）：返回 {域: 指纹}。探测失败返回空（不推）。"""
    out: dict[str, str] = {}
    try:
        rt_active = False
        from web.realtime_manager import realtime_manager as rt

        with rt._lock:
            rt_active = bool(rt._running or rt._tasks)
        if rt_active:
            out["realtime"] = _realtime_probe_sig()
    except Exception:  # noqa: BLE001 探测失败不阻断其它域
        pass
    try:
        pp_active = False
        from web.paper_manager import paper_manager as pp

        with pp._lock:
            pp_active = bool(pp._running or pp._watches)
        if pp_active:
            out["paper"] = _paper_probe_sig()
    except Exception:  # noqa: BLE001
        pass
    return out


def _full_snapshot(kind: str) -> dict[str, Any] | None:
    try:
        if kind == "realtime":
            return _full_realtime()
        if kind == "paper":
            return _full_paper()
    except Exception:  # noqa: BLE001 快照失败静默（下一轮再试）
        return None
    return None


async def _watch_step() -> None:
    """单轮：executor 里探测轻指纹；变化（或实时现价到期强刷）→ executor 里取
    全量快照，与其完整签名比对——真变了才推送（现价到期强刷但没真变化则不推）。"""
    loop = asyncio.get_running_loop()
    now = time.monotonic()
    probes = await loop.run_in_executor(None, _probe_once)
    for kind, sig in probes.items():
        changed = sig != _watcher_last_sig.get(kind)
        price_due = kind == "realtime" and now - _watcher_last_full.get(kind, 0.0) >= PRICE_REFRESH_S
        if not (changed or price_due):
            continue
        payload = await loop.run_in_executor(None, _full_snapshot, kind)
        if payload is None:
            continue
        full_sig = _sig_from_payload(kind, payload)
        if not full_sig or full_sig == _watcher_last_sig.get(kind):
            _watcher_last_full[kind] = now  # 强刷无变化：只更新计时，不推重复快照
            continue
        _watcher_last_sig[kind] = full_sig
        _watcher_last_full[kind] = now
        publish(kind, payload)


def _sig_from_payload(kind: str, payload: dict[str, Any]) -> str:
    """由完整快照生成与轻指纹同语义的签名（用于“推前再确认”）。"""
    try:
        if kind == "realtime":
            bits = [str(payload.get("running")), str(payload.get("count"))]
            for w in payload.get("watches") or []:
                bits.append(
                    "|".join(
                        [
                            str(w.get("id")), str(w.get("state")), str(w.get("direction")),
                            _r(w.get("strength"), 4), _r(w.get("factor_value"), 5),
                            _r(w.get("live_price"), 4), _r(w.get("last_close"), 5),
                            str(w.get("last_bar_ts")), str(w.get("message")),
                            str(w.get("dd_gate")), str(int(w.get("updated_at") or 0)),
                        ]
                    )
                )
            return "\n".join(bits)
        if kind == "paper":
            acct = payload.get("account") or {}
            bits = [
                str(payload.get("running")), str(payload.get("count")),
                str(len(payload.get("positions") or [])),
                _r(acct.get("cash"), 2), _r(acct.get("realized_pnl"), 2),
                str(acct.get("n_trades")),
            ]
            for t in payload.get("watches") or []:
                bits.append(
                    "|".join(
                        [
                            str(t.get("id")), str(t.get("state")), str(t.get("direction")),
                            _r(t.get("strength"), 4), str(t.get("last_bar_ts")),
                            _r(t.get("last_close"), 6), str(t.get("message")),
                        ]
                    )
                )
            return "\n".join(bits)
    except Exception:  # noqa: BLE001
        pass
    return ""


async def _watch_loop() -> None:
    while subscriber_count() > 0:
        try:
            await _watch_step()
        except Exception:  # noqa: BLE001 watcher 异常绝不让事件流断掉
            pass
        await asyncio.sleep(WATCH_INTERVAL_S)


def _maybe_start_watcher() -> None:
    global _watcher_task
    loop = _loop
    if loop is None or loop.is_closed():
        return
    if _watcher_task is not None and not _watcher_task.done():
        return
    try:
        _watcher_task = loop.create_task(_watch_loop())
    except RuntimeError:
        pass


def _schedule_reaper() -> None:
    """订阅清零 → 延时停 watcher（避免频繁启停；必须在事件循环线程调用）。"""
    global _watcher_task
    loop = _loop
    if loop is None or not loop.is_running():
        return

    async def _reap() -> None:
        global _watcher_task
        await asyncio.sleep(GRACE_EMPTY_S)
        if subscriber_count() == 0 and _watcher_task is not None and not _watcher_task.done():
            _watcher_task.cancel()
            try:
                await _watcher_task
            except asyncio.CancelledError:
                pass
            _watcher_task = None

    try:
        loop.create_task(_reap())
    except RuntimeError:
        pass
