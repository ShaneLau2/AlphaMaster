"""SSE 事件中枢（web.events）单元测试：帧格式 / 发布-订阅往返 / 慢客户端丢旧保新。"""
from __future__ import annotations

import asyncio
import json

from web import events as E


def _run(coro):
    return asyncio.run(coro)


def test_frame_format() -> None:
    frame = E._frame("realtime", {"running": True, "count": 1})
    head, _, data = frame.partition("\n")
    assert head == "event: realtime"
    assert data.startswith("data: ")
    payload = json.loads(data[len("data: "):])
    assert payload["running"] is True


def test_publish_roundtrip() -> None:
    async def main() -> None:
        E.bind_loop(asyncio.get_running_loop())
        q = await E.subscribe()
        E.publish("paper", {"account": {"cash": 1.0}})
        frame = await asyncio.wait_for(q.get(), 1.0)
        assert frame.startswith("event: paper\n")
        payload = json.loads(frame.split("\n", 2)[1][len("data: "):])
        assert payload["account"]["cash"] == 1.0
        E.unsubscribe(q)
        await asyncio.sleep(0.01)

    _run(main())


def test_subscribe_after_unsubscribe_delivers() -> None:
    async def main() -> None:
        E.bind_loop(asyncio.get_running_loop())
        q1 = await E.subscribe()
        E.unsubscribe(q1)
        await asyncio.sleep(0.01)
        q2 = await E.subscribe()
        E.publish("realtime", {"count": 0})
        frame = await asyncio.wait_for(q2.get(), 1.0)
        assert "event: realtime" in frame
        E.unsubscribe(q2)
        await asyncio.sleep(0.01)

    _run(main())


def test_first_subscriber_starts_watcher_then_reaper_stops() -> None:
    """回归：首个订阅者必须拉起 watcher，清零后 reaper 停掉它。

    曾因 subscribe() 里 ``was_empty = not _subs`` 在 add 之后判空而恒为 False，
    watcher 永不启动——连接正常（hello/心跳），但任何变更都不会被推送。
    """
    async def main() -> None:
        E.bind_loop(asyncio.get_running_loop())
        E._watcher_task = None
        E._subs.clear()
        q = await E.subscribe()
        assert E._watcher_task is not None and not E._watcher_task.done(), (
            "首个订阅者应启动 watcher"
        )
        old_grace = E.GRACE_EMPTY_S
        E.GRACE_EMPTY_S = 0.2
        try:
            E.unsubscribe(q)
            await asyncio.sleep(0.5)
        finally:
            E.GRACE_EMPTY_S = old_grace
        assert E._watcher_task is None or E._watcher_task.done(), (
            "订阅清零后 reaper 应停掉 watcher"
        )

    _run(main())


def test_slow_client_drops_oldest_keeps_latest() -> None:
    """订阅队列设了 maxsize；爆发超过容量时丢最旧、保最新一条。"""
    async def main() -> None:
        E.bind_loop(asyncio.get_running_loop())
        q = await E.subscribe()
        # 不消费，连发 > MAX_QUEUE 条，验证队列有界且最后一条被保留
        n = E.MAX_QUEUE + 5
        for i in range(n):
            E.publish("x", {"i": i})
        # call_soon_threadsafe 已投递；留一拍让所有帧入队
        await asyncio.sleep(0.05)
        assert q.qsize() <= E.MAX_QUEUE
        frames = []
        while not q.empty():
            frames.append(q.get_nowait())
        last = json.loads(frames[-1].split("\n", 2)[1][len("data: "):])
        assert last["i"] == n - 1
        E.unsubscribe(q)
        await asyncio.sleep(0.01)

    _run(main())
