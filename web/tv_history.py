"""TradingView 历史 K 线直连客户端（不依赖 tvdatafeed 的单次请求限制）。

背景：tvdatafeed 文档称单次最多 5000 根，实测（匿名会话，2026-09）发现
`create_series` 请求量可以远超 5000，服务端按自身留存返回它能给的全部：

    1m  -> 10,000 根（约 7 天）
    15m -> ~6,207 根（约 2 个月）
    1h  -> ~9,876 根（约 1.7 年）
    1d  -> 14,841 根（1833 年至今，几乎全量）

留意：请求传 500,000 会被服务端直接丢弃（无响应），故请求量钳制在 50,000；
`request_more_data` / `get_series` 翻页经实测被拒绝/断连，因此「一次请求拿全部
深度」就是匿名会话的真实上限，再大的数字没有任何意义。
"""
from __future__ import annotations

import json
import logging
import random
import re
import string
import time
from typing import Any

from websocket import create_connection  # websocket-client 随 tvdatafeed 安装

logger = logging.getLogger(__name__)

_WS_URL = "wss://data.tradingview.com/socket.io/websocket"
_WS_HEADERS = json.dumps({"Origin": "https://data.tradingview.com"})
_WS_TIMEOUT = 8.0

# 周期 -> TV chart resolution
RESOLUTION = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "8h": "480",
    "12h": "720",
    "1d": "1D",
    "3d": "3D",
    "1w": "1W",
    "1M": "1M",
}

# 未显式交易所时的探测顺序（与 tradingview_source 一致，去掉空交易所）
_PROBE_EXCHANGES = ("OANDA", "FX_IDC", "TVC", "NASDAQ", "NYSE", "SZSE", "SSE", "BINANCE")

MAX_REQUEST = 50_000   # 服务端接受的上限（超过会被静默丢弃）
COLLECT_TIMEOUT = 40.0  # 等待 series_completed 的秒数（D1 全量 ~10s）

_EXCHANGE_CACHE: dict[str, str] = {}


def _rs(n: int) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n))


def _prepend_header(st: str) -> str:
    return f"~m~{len(st)}~m~{st}"


def _msg(func: str, params: list[Any]) -> str:
    return _prepend_header(json.dumps({"m": func, "p": params}, separators=(",", ":")))


class _Session:
    """一次性 WebSocket 会话：建连 → 请求 → 收完 series_completed → 关闭。"""

    def __init__(self) -> None:
        self.ws = create_connection(
            _WS_URL, headers=_WS_HEADERS, timeout=_WS_TIMEOUT
        )
        self._cs = "cs_" + _rs(12)
        self._raw = ""

    def _send(self, func: str, params: list[Any]) -> None:
        self.ws.send(_msg(func, params))

    def begin(self) -> None:
        self._send("set_auth_token", ["unauthorized_user_token"])
        self._send("chart_create_session", [self._cs, ""])

    def request_resolution(self, symbol: str, resolution: str, n_bars: int) -> None:
        self._send(
            "resolve_symbol",
            [
                self._cs,
                "symbol_1",
                '={"symbol":"' + symbol
                + '","adjustment":"splits","session":"regular"}',
            ],
        )
        self._send(
            "create_series",
            [self._cs, "s1", "s1", "symbol_1", resolution, n_bars],
        )
        self._send("switch_timezone", [self._cs, "exchange"])

    def collect_until_done(self, timeout: float = COLLECT_TIMEOUT) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                self.ws.settimeout(6.0)
                self._raw += self.ws.recv()
            except Exception:
                break
            if "series_completed" in self._raw:
                break

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass

    def parse_bars(self) -> list[tuple[int, float, float, float, float, float]]:
        """优先解析现代 timescale_update 格式；无结果时退回 legacy \"s\":[...] 格式。"""
        bars = _parse_modern_bars(self._raw)
        if not bars:
            bars = _parse_legacy_bars(self._raw)
        return bars


def _split_messages(raw: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    i = 0
    while True:
        m = re.match(r"~m~(\d+)~m~", raw[i:])
        if not m:
            break
        length = int(m.group(1))
        j = i + m.end()
        try:
            msgs.append(json.loads(raw[j : j + length]))
        except Exception:
            pass
        i = j + length
    return msgs


def _parse_modern_bars(raw: str) -> list[tuple[int, float, float, float, float, float]]:
    """现代格式：{"m":"timescale_update","p":[cs,{s1:{node,s:[{i,v:[ts,o,h,l,c,v]}]}}]}"""
    out: dict[int, tuple[float, float, float, float, float]] = {}
    for msg in _split_messages(raw):
        if msg.get("m") != "timescale_update":
            continue
        payload = msg.get("p") or []
        if len(payload) < 2 or not isinstance(payload[1], dict):
            continue
        for series in payload[1].values():
            for item in series.get("s", []) or []:
                v = item.get("v")
                if not v or len(v) < 6:
                    continue
                ts = int(float(v[0]))
                if 1_400_000_000 <= ts <= 2_200_000_000:
                    out[ts] = (float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5]))
    return _sanitize(out)


def _parse_legacy_bars(raw: str) -> list[tuple[int, float, float, float, float, float]]:
    """旧格式（部分后端仍用它）："s":["s1",[[ts,"ts",o,h,l,c,v],...],"s1"]"""
    m = re.search(r'"s":\[(.+?)\}\]', raw)
    if not m:
        return []
    out: dict[int, tuple[float, float, float, float, float]] = {}
    for chunk in m.group(1).split(',{"'):
        fields = re.split(r"\[|:|,|\]", chunk)
        if len(fields) < 10:
            continue
        try:
            ts, o, h, l, c, v = (float(fields[i]) for i in (4, 5, 6, 7, 8, 9))
        except ValueError:
            continue
        if 1_400_000_000 <= ts <= 2_200_000_000:
            out[int(ts)] = (o, h, l, c, v)
    return _sanitize(out)


def _sanitize(out: dict[int, tuple[float, float, float, float, float]]) -> list[tuple[int, float, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float, float]] = []
    for ts, (o, h, l, c, v) in out.items():
        if not (o > 0 and h > 0 and l > 0 and c > 0):
            continue
        rows.append((ts, o, h, l, c, float(v)))
    rows.sort(key=lambda r: r[0])
    return rows


def fetch_history(
    symbol: str,
    timeframe: str,
    n_bars: int,
) -> list[tuple[int, float, float, float, float, float]]:
    """拉取 {symbol} {timeframe} 的 K 线（升序），返回服务端能给到的全部深度。

    symbol 可带 EXCHANGE:CODE 前缀；不带则按 _PROBE_EXCHANGES 顺序自动探测
    （命中即缓存，后续同品种直接复用）。
    """
    resolution = RESOLUTION.get(timeframe)
    if resolution is None:
        raise ValueError(f"不支持的周期: {timeframe}")

    code = symbol.strip()
    exchange: str | None = None
    if ":" in code:
        exchange, code = (part.strip() for part in code.split(":", 1))
        exchange = exchange.upper()

    n = min(max(int(n_bars), 100), MAX_REQUEST)
    attempts = [exchange] if exchange else None
    if attempts is None:
        cached = _EXCHANGE_CACHE.get(symbol)
        attempts = ([cached] if cached else []) + list(_PROBE_EXCHANGES)

    last_err: Exception | None = None
    for ex in attempts:
        target = f"{ex}:{code}" if ex else code
        bars = _request_once(target, resolution, n)
        if bars:
            if exchange is None and ex:
                _EXCHANGE_CACHE[symbol] = ex
            return bars
        last_err = RuntimeError(f"{target} 无数据")
    raise RuntimeError(f"TradingView 未找到 {symbol}（可尝试 EXCHANGE:CODE 指定交易所）: {last_err}")


def _request_once(
    target: str, resolution: str, n_bars: int
) -> list[tuple[int, float, float, float, float, float]]:
    session = _Session()
    try:
        session.begin()
        session.request_resolution(target, resolution, n_bars)
        session.collect_until_done()
        return session.parse_bars()
    except Exception as exc:  # noqa: BLE001 单次请求失败不致命
        logger.warning("tv fetch failed for %s: %s", target, exc)
        return []
    finally:
        session.close()