"""Binance 数据源（公开行情 API，USDT 现货，endTime 翻页取全量历史）。"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

BASE = "https://api.binance.com"
_PAGE = 1000          # klines 单次上限
_SLEEP = 0.15         # 翻页间隔（限权重）
_PROBE_TTL = 60.0       # 可用性探测缓存秒数（成功时）
_PROBE_FAIL_TTL = 8.0    # 探测失败缓存秒数（网络抖动时快速恢复）

_TF = {  # 完整覆盖 Binance klines interval：1m/3m/5m/15m/30m/1h/2h/4h/6h/8h/12h/1d/3d/1w/1M
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}

_PRESETS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
]


def _normalize_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if ":" in s:
        s = s.split(":", 1)[1]
    s = s.strip()
    if not s or not s.isalnum():
        raise DataSourceUnavailable(f"无法识别 Binance 品种：{symbol}（示例 BTCUSDT / BINANCE:BTCUSDT）")
    return s


class BinanceSource(DataSource):
    kind = "binance"
    label = "Binance"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._probe_ok: bool | None = None
        self._probe_at = 0.0

    def available(self) -> tuple[bool, str]:
        now = time.time()
        ttl = _PROBE_TTL if self._probe_ok else _PROBE_FAIL_TTL
        if self._probe_ok is not None and now - self._probe_at < ttl:
            return self._probe_ok, ("Binance 公开行情 · 全量历史（自动翻页）" if self._probe_ok else "Binance API 不可达（网络受限或域名被墙）")
        try:
            self._get("/api/v3/time")
            self._probe_ok = True
            return (True, "Binance 公开行情 · 全量历史（自动翻页）")
        except Exception:
            self._probe_ok = False
            return (False, "Binance API 不可达（网络受限或域名被墙）")
        finally:
            self._probe_at = now

    def supported_timeframes(self) -> list[str]:
        return list(_TF.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def _get(self, path: str, params: dict[str, str] | None = None, retries: int = 3) -> list:
        query = urllib.parse.urlencode(params or {})
        url = f"{BASE}{path}?{query}"
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AlphaMaster/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_err = exc
                if exc.code in (429, 418):  # 限频：重试退避
                    time.sleep(min(10, 2 ** attempt))
                    continue
                raise DataSourceUnavailable(
                    f"Binance HTTP {exc.code}: {exc.read(200).decode('utf-8', 'ignore')}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(6, 2 ** attempt))
        raise DataSourceUnavailable(f"Binance 请求失败: {last_err}")

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True,
        progress_cb=None, end_time: int | None = None,
    ) -> list[Bar]:
        """拉最近 n 根；end_time（epoch 毫秒）给定时代替 now 作翻页锚点——
        用于增量归档：重复下载时锚定到档案最早 bar 之前，向更早历史扩展。
        """
        if timeframe not in _TF:
            raise DataSourceUnavailable(f"Binance 不支持周期 {timeframe}")
        sym = _normalize_symbol(symbol)
        bar = _TF[timeframe]
        want = max(n + 2, 20)
        now_ms = int(time.time() * 1000)
        anchor_ms = now_ms if end_time is None else int(end_time)

        with self._lock:
            rows: list[list] = []
            end_time: int | None = anchor_ms
            stagnant = 0
            while len(rows) < want:
                params: dict[str, str] = {"symbol": sym, "interval": bar, "limit": str(_PAGE)}
                if end_time is not None:
                    params["endTime"] = str(end_time)
                batch = self._get("/api/v3/klines", params)
                if not batch:
                    break

                prev_len = len(rows)
                rows.extend(batch)
                if progress_cb is not None:
                    progress_cb(len(rows))

                oldest_ts = int(batch[0][0])
                if end_time is not None and oldest_ts >= end_time:
                    stagnant += 1
                    if stagnant >= 2:
                        break
                else:
                    stagnant = 0
                end_time = oldest_ts - 1

                if len(batch) < _PAGE:
                    break
                time.sleep(_SLEEP)

        if not rows:
            raise DataSourceUnavailable(f"Binance 无数据：{symbol}")
        # 拉取成功即视为可达（探测可能恰好撞上首次连接的抖动）
        self._probe_ok = True
        self._probe_at = time.time()

        # klines 每条: [openTime, open, high, low, close, vol, closeTime, ...]
        seen: dict[int, list] = {}
        for k in rows:
            seen[int(k[0])] = k
        ordered = [seen[ts] for ts in sorted(seen)]

        # 剔除仍在形成的最后一根（其 closeTime 在未来）
        if drop_forming and ordered and len(ordered[-1]) > 6:
            while ordered and int(ordered[-1][6]) > now_ms:
                ordered.pop()

        keep = ordered[-n:] if n > 0 else ordered
        bars: list[Bar] = []
        for k in keep:
            bars.append(
                Bar(
                    ts=int(int(k[0]) // 1000),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5] or 0.0),
                )
            )
        return bars