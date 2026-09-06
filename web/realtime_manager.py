"""实时信号引擎：多品种 / 多周期并发调度。

- 每个「监控项」= (数据源, 品种, 周期, 策略因子)。
- 后台线程按周期自适应节奏轮询，出现新 bar 才重算，信号取最后已收盘 bar。
- 共享 (源,品种,周期) 的 K 线抓取结果做短 TTL 缓存，避免重复请求。
- 监控清单持久化到 web_settings，重启恢复。
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_core.vocab import VOCAB_VERSION
from strategy_manager.live_signal import (
    DIR_FLAT,
    DIR_LONG,
    DIR_SHORT,
    evaluate_signal,
    min_exposure,
)
from web.data_sources.base import Bar, bars_to_raw_dict
from web.hold_policy import combo_id, combo_parts, dd_ladder_step, params_for, sl_tp_levels
from web.data_sources.factory import SOURCE_KINDS, get_source
from web.settings import load_settings, resolve_signal_threshold, save_settings


def factor_percentile(fv: float, buf: deque) -> float | None:
    """因子在其历史样本中的百分位（0-100）：<= fv 的比例。"""
    if not buf:
        return None
    le = sum(1 for x in buf if x <= fv)
    return round(100.0 * le / len(buf), 1)


def _resolved_threshold() -> float:
    """统一无信号阈值：web_settings 优先，回退 Config.MIN_TRADE_EXPOSURE。"""
    try:
        return resolve_signal_threshold()
    except Exception:  # noqa: BLE001
        return min_exposure()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 因子僵化判定：跨新 bar 的因子相对变化 < 0.001%（1e-5）视为“几乎不变”；
# 连续 >= _FACTOR_FLAT_BARS 根新 bar 都满足时，卡片亮「因子僵化」诊断角标。
RT_FACTOR_FLAT_EPS = 1e-5
RT_FACTOR_FLAT_BARS = 5
# 偏离入场告警：|现价/方向锚定入场 - 1| 超过阈值% 推送；回到阈值一半以内才解除去重
RT_DEV_RECOVER_FACTOR = 0.5
RT_ALERT_DEV_PCT_DEFAULT = 0.5

# 每个周期的轮询节奏（秒）
_CADENCE = {
    "1m": 15, "5m": 30, "15m": 45, "30m": 60,
    "1h": 60, "4h": 120, "1d": 300, "1w": 600, "1M": 600,
}
# K 线周期长度（秒）；用于推算「下一根已收盘 bar」时间
_TF_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
    "1M": 2592000,  # 近似 30 天
}
_DEFAULT_CADENCE = 60
_N_BARS = 1000                # 每次拉取的历史 bar 数（实盘最低需 800：特征warm-up 200 + VM滚动归一化 500 + 余量）
_HISTORY_LEN = 60             # 保留的信号强度历史点数（供 sparkline）
_LIVE_PRICE_TTL = 12.0         # 现价（含正在形成 bar）短缓存，供卡片展示“数据在动”
_PRICE_REFRESH_BUDGET_S = 2.0   # status() 并发现价刷新的整批等待预算（秒），网络阶段并行 + 超时封顶
_VALID_KINDS = {k for k, _ in SOURCE_KINDS}


# ── 共享 K 线缓存（realtime / paper 两个引擎共用，避免同一 源/品种/周期 重复请求）──
_BAR_CACHE: dict[tuple[str, str, str], tuple[float, list]] = {}
_BAR_CACHE_LOCK = threading.Lock()


def _cadence_for(tf: str) -> int:
    return _CADENCE.get(tf, _DEFAULT_CADENCE)


def _next_bar_close_at(last_bar_open: int | None, timeframe: str, now: float | None = None) -> int | None:
    """根据最后已收盘 bar 的开盘时间，推算下次收盘（即下次信号更新）的 Unix 秒。

    若最后一根已收盘 bar 已过时太久，视为休市/断档，
    返回 None，避免在周末等时段虚构「几分钟后更新」的倒计时。

    注意：阈值需兼顾 A 股/期货午休（1.5~2h）等盘中暂停场景。
    对小周期（如 5m/15m）使用固定 2×period 会在午休期间误判为休市。
    因此取 max(2×period, 7200)（至少 2 小时），确保跨过午休而不误报。
    真正的跨日/周末休市（>2h 无新 bar）才正确显示"休市中"。
    """
    if last_bar_open is None:
        return None
    period = _TF_SECONDS.get(timeframe)
    if not period:
        return None
    now_i = int(now if now is not None else time.time())
    last_open = int(last_bar_open)
    last_close = last_open + period
    # 兼顾午休：阈值至少 2 小时，避免 A 股/期货午休期间误判
    tolerance = max(period * 2, 7200)
    # 仍未到收盘（常见于数据源时钟快于本机、或未剔除形成中 bar）
    if last_close > now_i:
        return last_close
    # 正常交易中：上一根收盘距今至多 tolerance 秒
    if now_i - last_close > tolerance:
        return None
    # last_open 开盘 → last_close 收盘；当前形成中的 bar 在 +2*period 收盘
    nxt = last_open + 2 * period
    while nxt <= now_i:
        nxt += period
        if nxt - last_close > tolerance:
            return None
    return nxt


def _ensure_closed_bars(bars: list, timeframe: str, now: float | None = None) -> list:
    """按本机时钟剔掉时间戳仍在未来的 K 线（双保险，防数据源时钟偏快）。

    各数据源的 fetch_bars(drop_forming=True) 已负责剔除「正在形成」的 bar；
    这里只删 ts > now 的明显异常 bar（数据源时钟偏快导致返回了未来 bar）。

    不再用 `ts + period > now` 判断：该式假设 ts=开盘时刻，对 A 股日线（15:00
    收盘但 period 按 86400 秒算）会在收盘当晚误删当天已收盘 bar——因为
    ts+86400 落到次日，恒大于当晚的 now。改用 ts>now 后，已收盘 bar 的 ts
    必然 <= now，不会被误删；而真正未收盘/未来的 bar 由各源 drop_forming 处理。
    """
    if not bars:
        return bars
    now_i = int(now if now is not None else time.time())
    out = list(bars)
    while out and int(out[-1].ts) > now_i:
        out.pop()
    return out


def load_strategy_meta(path: str) -> dict[str, Any]:
    """读取策略 JSON 的元信息（formula / vocab_version / symbol 等）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"formula": data, "vocab_version": "legacy", "symbol": None, "timeframe": None, "best_score": None}
    return {
        "formula": data.get("formula"),
        "vocab_version": data.get("vocab_version"),
        "symbol": data.get("symbol"),
        "timeframe": data.get("timeframe"),
        "best_score": data.get("best_score"),
    }


# 向后兼容别名（旧内部引用）
_load_strategy_meta = load_strategy_meta

# 模拟盘引擎复用同一套收盘口径
ensure_closed_bars = _ensure_closed_bars
cadence_for = _cadence_for


_LIVE_CACHE: dict[tuple[str, str, str], tuple[float, "Bar | None"]] = {}
_LIVE_CACHE_LOCK = threading.Lock()


def fetch_live_price(source: str, symbol: str, timeframe: str):
    """拉“最新价”（含正在形成的 bar，仅展示用，不参与信号）。

    信号只取已收盘 bar（与回测口径一致），但卡片需要一个会随行情变动的
    现价来证明监控是活的。带短 TTL 缓存（_LIVE_PRICE_TTL）。
    """
    key = (source, symbol, timeframe)
    now = time.monotonic()
    with _LIVE_CACHE_LOCK:
        cached = _LIVE_CACHE.get(key)
        if cached and (now - cached[0]) < _LIVE_PRICE_TTL:
            return cached[1]
    bar = None
    try:
        src = get_source(source)
        bars = src.fetch_bars(symbol, timeframe, 2, drop_forming=False)
        if bars:
            bar = bars[-1]
    except Exception:  # noqa: BLE001 现价失败不影响信号/主流程
        bar = None
    with _LIVE_CACHE_LOCK:
        _LIVE_CACHE[key] = (now, bar)
    return bar


def _pull_live_price(task: "WatchTask") -> None:
    """拉取并写回单个监控项的现价；失败静默（现价失败不影响信号/主流程）。

    放在独立函数里：status() 提交后即使超时放弃等待，线程仍会自行完成并
    写入 TTL 缓存，下一次轮询即命中。"""
    try:
        live = fetch_live_price(task.source, task.symbol, task.timeframe)
        if live is not None:
            task.live_price = float(live.close)
            task.live_price_ts = int(live.ts)
    except Exception:  # noqa: BLE001
        pass


def fetch_cached_bars(
    source: str, symbol: str, timeframe: str, *, drop_forming: bool = True
) -> list:
    """带短 TTL 缓存的 K 线抓取，realtime 与模拟盘引擎共用。

    返回升序已收盘 Bar 列表；同一 (源,品种,周期) 在 TTL 内复用缓存，
    避免两个引擎各自拉网络。
    """
    key = (source, symbol, timeframe)
    ttl = max(10.0, _cadence_for(timeframe) * 0.8)
    now = time.monotonic()
    with _BAR_CACHE_LOCK:
        cached = _BAR_CACHE.get(key)
        if cached and (now - cached[0]) < ttl:
            return cached[1]
    src = get_source(source)
    bars = src.fetch_bars(symbol, timeframe, _N_BARS, drop_forming=drop_forming)
    bars = _ensure_closed_bars(bars, timeframe)
    with _BAR_CACHE_LOCK:
        _BAR_CACHE[key] = (now, bars)
    return bars


@dataclass
class WatchTask:
    id: str
    source: str
    symbol: str
    timeframe: str
    strategy_file: str
    strategy_name: str
    formula: list[int]
    vocab_version: str | None
    strategy_symbol: str | None
    strategy_timeframe: str | None
    best_score: float | None
    cadence_s: int
    policy_id: str = "signal"       # 持仓管理方案（图表入场/止损/止盈参考线用）
    # 运行时状态
    state: str = "pending"          # pending|ok|insufficient|error
    direction: str | None = None
    strength: float | None = None
    position: float | None = None
    factor_value: float | None = None
    bars_used: int | None = None
    last_bar_ts: int | None = None
    updated_at: float | None = None
    message: str = ""
    warn: str = ""
    next_due: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    closes: deque = field(default_factory=lambda: deque(maxlen=240))  # [ts, close] 供价格图
    last_close: float | None = None
    live_price: float | None = None      # 最新价（含形成中 bar，仅展示）
    live_price_ts: int | None = None
    # 信号质量统计（每次收盘重算时累计，供“卡着不动”诊断）
    dir_run: int = 0                       # 当前方向连续根数（同向 +1，翻转重置 1）
    dir_hist: deque = field(default_factory=lambda: deque(maxlen=1000))    # 每次评估的方向（滚动窗口）
    factor_buf: deque = field(default_factory=lambda: deque(maxlen=2000))  # 每次评估的因子值
    # dd 叠加时的轻量回撤跟踪（信息性：实时卡无账户，不阻断任何操作，仅展示+记录转移）
    dd_gate: int = 0
    dd_peak: float | None = None
    dd_pct: float | None = None
    # 因子僵化诊断：连续“新 bar 上因子几乎不变”的根数 / 连续价格移动根数
    factor_flat_run: int = 0
    price_moved_run: int = 0
    _factor_bar_ts: int | None = None   # 上一个参与对比的新 bar ts
    _factor_bar_val: float | None = None
    # 距翻转裕度历史：每根新收盘 bar 记 |factor|−threshold（负值=观望带内），
    # 供前端画 50 根迷你趋势提前看因子是否逼近阈值
    flip_margin_hist: deque = field(default_factory=lambda: deque(maxlen=50))
    # 因子历史分位趋势：每根新收盘 bar 记当前因子在其历史样本中的百分位（0-100），
    # 与裕度线互补——因子本身单调变化也能一眼看出是逼近还是远离阈值
    factor_pct_hist: deque = field(default_factory=lambda: deque(maxlen=50))
    _flip_bar_ts: int | None = None
    # 每根新收盘 bar 的 [ts, close, factor] 轨迹（最近 50 根，供前端点“僵化”
    # 徽标画 因子 vs 价格 双轴小图：一眼区分 真钝化/贴边 还是 区间震荡）
    trace_hist: deque = field(default_factory=lambda: deque(maxlen=50))
    # 偏离入场告警：方向建立时锚定的入场参考价 + 本次越线是否已推送
    dir_entry_price: float | None = None
    dev_alerted: bool = False
    # 入场锚偏离历史：每根新收盘 bar 记一次价格相对锚的偏离%（最近 50 根，
    # 供前端在入场锚芯片下画迷你图——一眼区分 缓涨越过阈值 还是 单根跳穿）
    anchor_dev_hist: deque = field(default_factory=lambda: deque(maxlen=50))
    _dev_hist_bar_ts: int | None = None
    # 僵化飞书告警去重：硬钝化推送一次；因子恢复变化或方向翻转后才允许再次提醒
    _stale_pushed: bool = False

    def _signal_quality(self) -> dict[str, Any]:
        n = len(self.dir_hist)
        base = {
            "dir_run": self.dir_run,
            "long_pct": None, "short_pct": None,
            "flat_pct": None, "factor_pct": None, "window": n,
            "factor_flat_run": self.factor_flat_run,
            "price_moved_run": self.price_moved_run,
        }
        if n == 0:
            return base
        long_n = sum(1 for d in self.dir_hist if d == DIR_LONG)
        short_n = sum(1 for d in self.dir_hist if d == DIR_SHORT)
        flat_n = n - long_n - short_n
        fv = self.factor_value
        factor_pct = factor_percentile(fv, self.factor_buf) if fv is not None else None
        return {
            "dir_run": self.dir_run,
            "long_pct": round(100.0 * long_n / n, 1),
            "short_pct": round(100.0 * short_n / n, 1),
            "flat_pct": round(100.0 * flat_n / n, 1),
            "factor_pct": factor_pct,
            "window": n,
            "factor_flat_run": self.factor_flat_run,
            "price_moved_run": self.price_moved_run,
            "flip_margin_hist": list(self.flip_margin_hist),
            "factor_pct_hist": list(self.factor_pct_hist),
        }

    def to_public(self) -> dict[str, Any]:
        now = time.time()
        next_close = _next_bar_close_at(self.last_bar_ts, self.timeframe, now)
        live = next_close is not None
        plan = None
        if self.state == "ok" and self.last_close and self.direction in (DIR_LONG, DIR_SHORT):
            lv = sl_tp_levels(self.direction, self.last_close, self.policy_id)
            plan = {
                "entry": round(float(self.last_close), 10),
                "stop_price": lv["stop_price"],
                "target_price": lv["target_price"],
            }
        return {
            "id": self.id,
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_name": self.strategy_name,
            "strategy_symbol": self.strategy_symbol,
            "strategy_timeframe": self.strategy_timeframe,
            "best_score": self.best_score,
            "policy_id": self.policy_id,
            "dd_gate": self.dd_gate,
            "dd_peak": self.dd_peak,
            "dd_pct": self.dd_pct,
            "state": self.state,
            "direction": self.direction,
            "strength": self.strength,
            "position": self.position,
            "factor_value": self.factor_value,
            "bars_used": self.bars_used,
            "last_bar_ts": self.last_bar_ts,
            "last_close": self.last_close,
            "live_price": self.live_price,
            "live_price_ts": self.live_price_ts,
            "chart": {"pts": [list(p) for p in self.closes]},
            "plan": plan,
            "session_live": live,
            "next_bar_close_at": next_close,
            "seconds_to_next": (
                max(0, int(next_close - now)) if next_close is not None else None
            ),
            "updated_at": self.updated_at,
            "message": self.message,
            "tv_blocked": self.message == "TV_CONNECTIVITY_BLOCKED",
            "warn": self.warn,
            "threshold": _resolved_threshold(),
            "history": list(self.history),
            "dir_entry_price": self.dir_entry_price,
            "dev_alerted": self.dev_alerted,
            "trace_hist": [list(x) for x in self.trace_hist],
            "anchor_dev_hist": list(self.anchor_dev_hist),
            "signal_quality": self._signal_quality(),
        }

    def persist_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_file": self.strategy_file,
            "policy_id": self.policy_id,
        }


class RealtimeManager:
    def __init__(self) -> None:
        self._tasks: dict[str, WatchTask] = {}
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rt")
        # 现价刷新专用池：与信号评估池分开，避免 status() 的并发现价被
        # 评估任务占满 worker 而排队（评估拉 K 线可能更久）。
        self._price_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rt-price")
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()
        # K线缓存：(kind,symbol,tf) -> (monotonic_ts, bars)
        self._bar_cache: dict[tuple[str, str, str], tuple[float, list]] = {}
        self._loaded = False
        self._tv_blocked_until = 0.0

    # ── 持久化 ──────────────────────────────────────────────────────────
    def load_persisted(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        watches = load_settings().get("realtime_watches") or []
        for w in watches:
            try:
                self._add_task_internal(
                    w["source"], w["symbol"], w["timeframe"], w["strategy_file"],
                    policy_id=w.get("policy_id"), persist=False,
                )
            except Exception:
                continue
        if self._tasks:
            self._ensure_thread()

    def _persist(self) -> None:
        save_settings({"realtime_watches": [t.persist_dict() for t in self._tasks.values()]})

    # ── 增删 ────────────────────────────────────────────────────────────
    def add_watch(self, source: str, symbol: str, timeframe: str, strategy_file: str,
                  policy_id: str = "signal") -> dict[str, Any]:
        task = self._add_task_internal(
            source, symbol, timeframe, strategy_file,
            policy_id=policy_id, persist=True,
        )
        self._ensure_thread()
        return task.to_public()

    def _add_task_internal(
        self, source: str, symbol: str, timeframe: str, strategy_file: str,
        persist: bool, policy_id: str | None = None,
    ) -> WatchTask:
        source = (source or "").strip()
        symbol = (symbol or "").strip()
        timeframe = (timeframe or "").strip()
        if source not in _VALID_KINDS:
            raise ValueError(f"未知数据源: {source}")
        if not symbol:
            raise ValueError("请填写品种")
        src = get_source(source)
        if timeframe not in src.supported_timeframes():
            raise ValueError(f"{src.label} 不支持周期 {timeframe}")

        path = strategy_file
        if not Path(path).is_absolute():
            path = str((PROJECT_ROOT / path).resolve())
        if not Path(path).exists():
            raise ValueError(f"策略文件不存在: {strategy_file}")
        meta = _load_strategy_meta(path)
        if not meta.get("formula"):
            raise ValueError("策略文件缺少 formula")

        name = Path(path).stem
        task_id = f"{source}:{symbol}:{timeframe}:{name}"

        warn = ""
        if meta.get("vocab_version") and meta["vocab_version"] not in (VOCAB_VERSION, "legacy"):
            warn = f"词表版本不符（{meta['vocab_version']} vs {VOCAB_VERSION}），信号可能失真"
        elif meta.get("symbol") and meta["symbol"] != symbol:
            warn = f"该因子为 {meta['symbol']} 训练，跨品种运行仅供参考"

        task = WatchTask(
            id=task_id,
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            strategy_file=path,
            strategy_name=name,
            formula=[int(t) for t in meta["formula"]],
            vocab_version=meta.get("vocab_version"),
            strategy_symbol=meta.get("symbol"),
            strategy_timeframe=meta.get("timeframe"),
            best_score=meta.get("best_score"),
            cadence_s=_cadence_for(timeframe),
            policy_id=combo_id(policy_id),
            warn=warn,
            next_due=0.0,
        )
        with self._lock:
            self._tasks[task_id] = task
            if persist:
                self._persist()
        return task

    def remove_watch(self, task_id: str) -> bool:
        with self._lock:
            existed = self._tasks.pop(task_id, None) is not None
            if existed:
                self._persist()
        return existed

    def clear(self) -> None:
        with self._lock:
            self._tasks.clear()
            self._persist()

    # ── 状态 ────────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self._lock:
            tasks = list(self._tasks.values())
        # 轻量「现价」刷新：让卡片价格在两次信号判断（cadence）之间也动起来。
        # 排除 tradingview 等昂贵/易挂源；fetch_live_price 自带短 TTL，
        # 前端每 4s 轮询这里时最多每 ~12s 真正拉一次网络，失败静默。
        # 并发 + 预算封顶：多个监控项不再串行累加网络等待（见 _refresh_live_prices）。
        self._refresh_live_prices(tasks)
        with self._lock:
            watches = [t.to_public() for t in tasks]
        nearest = None
        for w in watches:
            s = w.get("seconds_to_next")
            if s is None:
                continue
            if nearest is None or s < nearest:
                nearest = s
        return {
            "running": self._running,
            "count": len(watches),
            "watches": watches,
            "server_time": time.time(),
            "nearest_seconds_to_next": nearest,
        }

    def _refresh_live_prices(self, tasks: list["WatchTask"]) -> None:
        """并发拉现价：每个监控项一个任务进专用池，整批等待预算封顶。

        旧实现串行循环，N 个监控项且 TTL 刚过时 status() 总耗时 ≈ Σ(单项网络
        延迟)，把前端轮询/页面交互拖慢；改为并行后总耗时 ≈ max(单项) ≤ 预算。
        超时的项本批跳过（线程自行收尾并写入短 TTL 缓存，下轮即可命中）。"""
        futs: list = []
        for t in tasks:
            if t.source == "tradingview":
                continue
            futs.append(self._price_pool.submit(_pull_live_price, t))
        if not futs:
            return
        # 预算内等待；剩余 futures 不取消——它们在后台完成并写缓存，
        # 保持引用避免执行中被回收（wait 返回的 not_done 仍持引用）。
        wait(futs, timeout=_PRICE_REFRESH_BUDGET_S)

    # ── 调度线程 ────────────────────────────────────────────────────────
    def start(self) -> None:
        self._ensure_thread()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="realtime", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                pass
            time.sleep(1.5)

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            due = [t for t in self._tasks.values() if now >= t.next_due]
        for task in due:
            with self._inflight_lock:
                if task.id in self._inflight:
                    continue
                self._inflight.add(task.id)
            # 预置下次到期，避免重复提交
            task.next_due = now + task.cadence_s
            self._executor.submit(self._evaluate_task, task)

    def _get_bars(self, source: str, symbol: str, timeframe: str):
        """带短 TTL 缓存的 K 线抓取（与模拟盘引擎共用同一缓存）。"""
        return fetch_cached_bars(source, symbol, timeframe)

    def _dd_track_step(self, task: WatchTask) -> None:
        """轻量回撤跟踪：与模拟盘同阶梯（dd_ladder_step），但实时卡无账户，
        只展示状态并记录状态转移，不阻断任何操作。"""
        try:
            if "dd" not in combo_parts(task.policy_id):
                return  # 仅跟踪叠加了 dd 的方案
            mark = float(task.last_close or 0.0)
            if mark <= 0.0:
                return
            pk = float(task.dd_peak) if task.dd_peak else mark
            dd = mark / pk - 1.0 if pk > 0.0 else 0.0
            prev_st = int(task.dd_gate or 0)
            _action, st = dd_ladder_step(dd, params_for("dd"), prev_st)
            task.dd_gate = st
            task.dd_peak = max(pk, mark)
            task.dd_pct = dd * 100.0
            if st != prev_st:
                from web.dd_events import log_dd_event

                if st != 0:
                    event_name = "深档熔断" if st == 2 else "熔断"
                else:
                    event_name = "收复"
                log_dd_event(
                    "realtime",
                    symbol=task.symbol, timeframe=task.timeframe,
                    policy_id=task.policy_id,
                    event=event_name, dd_pct=dd * 100.0, mark=mark, peak=pk,
                    prev_state=prev_st, state=st,
                    detail="实时分析卡（无账户，仅信息跟踪，不阻断操作）",
                )
        except Exception:  # noqa: BLE001 跟踪失败绝不影响信号
            pass

    def _evaluate_task(self, task: WatchTask) -> None:
        try:
            bars = self._get_bars(task.source, task.symbol, task.timeframe)
            if not bars:
                self._set_error(task, "未获取到 K 线")
                return
            last_ts = bars[-1].ts
            task.last_close = float(bars[-1].close)
            # 现价（形成中 bar）短缓存拉取；失败静默（不影响信号）
            try:
                live = fetch_live_price(task.source, task.symbol, task.timeframe)
                if live is not None:
                    task.live_price = float(live.close)
                    task.live_price_ts = int(live.ts)
            except Exception:  # noqa: BLE001
                pass
            # 新 bar 才追加价格序列（供价格走势图；最多保留 240 点）
            if not task.closes or task.closes[-1][0] != last_ts:
                for b in bars:
                    task.closes.append([int(b.ts), round(float(b.close), 8)])
                # 僵化诊断：新 bar 收盘价是否移动（连续移动根数）
                if len(task.closes) >= 2:
                    moved = task.closes[-1][1] != task.closes[-2][1]
                    task.price_moved_run = task.price_moved_run + 1 if moved else 0
            raw = bars_to_raw_dict(bars)
            result = evaluate_signal(task.formula, raw, threshold=_resolved_threshold())

            task.state = result.get("state", "error")
            task.message = result.get("message", "")
            task.bars_used = result.get("bars_used", len(bars))
            task.last_bar_ts = last_ts
            task.updated_at = time.time()
            # 轻量 DD 跟踪（仅当方案叠加了 dd；状态转移记入 dd 事件日志）
            if task.last_close:
                self._dd_track_step(task)
            if task.state == "ok":
                new_dir = result["direction"]
                prev_dir = task.direction
                task.direction = new_dir
                task.strength = result["strength"]
                task.position = result["position"]
                task.factor_value = result["factor_value"]
                task.history.append(round(result["strength"], 4))
                # 信号质量统计：连续方向根数 / 方向分布 / 因子历史分位
                if prev_dir == new_dir:
                    task.dir_run += 1
                else:
                    task.dir_run = 1
                task.dir_hist.append(new_dir)
                if task.factor_value is not None:
                    task.factor_buf.append(round(float(task.factor_value), 6))
                # 方向建立/翻转：重新锚定“入场参考价”（当前已收盘价）并清除偏离告警状态
                if prev_dir != new_dir:
                    task.dir_entry_price = task.last_close
                    task.dev_alerted = False
                    task.anchor_dev_hist.clear()  # 换方向后偏离图相对新锚重画
                    task._stale_pushed = False
                # 入场锚偏离历史：每根新收盘 bar 记一次相对当前方向锚的偏离%（有方向锚才记）
                if task.direction in (DIR_LONG, DIR_SHORT) and task.dir_entry_price:
                    if task._dev_hist_bar_ts != last_ts:
                        task._dev_hist_bar_ts = last_ts
                        _px = float(task.last_close) if task.last_close is not None else None
                        if _px is not None and _px > 0.0:
                            task.anchor_dev_hist.append(
                                round((_px / float(task.dir_entry_price) - 1.0) * 100.0, 6)
                            )
                # 因子僵化统计：只在“新 bar”上比较（同一根 bar 内多次评估输入相同）
                fv = task.factor_value
                if fv is not None:
                    fv_f = float(fv)
                    if task._factor_bar_ts is not None and last_ts != task._factor_bar_ts:
                        prev_f = task._factor_bar_val
                        if prev_f is not None:
                            denom = max(abs(prev_f), 1e-9)
                            if abs(fv_f - prev_f) / denom < RT_FACTOR_FLAT_EPS:
                                task.factor_flat_run += 1
                                self._maybe_stale_alert(task)  # 硬钝化达阈值推飞书（可配置）
                            else:
                                task.factor_flat_run = 0
                                task._stale_pushed = False  # 因子恢复变化 → 允许再次提醒
                    task._factor_bar_val = fv_f
                    task._factor_bar_ts = last_ts
                    # 距翻转裕度 + 因子历史分位：只在新收盘 bar 上各记一点
                    if last_ts != task._flip_bar_ts:
                        thr_v = _resolved_threshold()
                        task.flip_margin_hist.append(round(abs(fv_f) - thr_v, 6))
                        pct = factor_percentile(fv_f, task.factor_buf)
                        if pct is not None:
                            task.factor_pct_hist.append(pct)
                        # 因子轨迹采样点：同一根新收盘 bar 的 价格 + 因子 对齐记录
                        task.trace_hist.append([
                            int(last_ts),
                            round(float(task.last_close), 8) if task.last_close is not None else None,
                            fv_f,
                        ])
                        task._flip_bar_ts = last_ts
                # 已有上次方向且发生转折时推飞书（首次算出方向不打扰）
                if prev_dir and new_dir and prev_dir != new_dir:
                    self._notify_direction_flip(task, prev_dir, new_dir)
                # 偏离入场参考价告警（阈值可配置，飞书未启用时空转）
                self._check_dev_alert(task)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if task.source == "tradingview":
                try:
                    from web.data_sources.tradingview_connectivity import (
                        TV_CONNECTIVITY_BLOCKED,
                        check_tradingview_connectivity,
                    )

                    now_m = time.monotonic()
                    if now_m < self._tv_blocked_until:
                        msg = TV_CONNECTIVITY_BLOCKED
                    else:
                        ok, _detail = check_tradingview_connectivity(
                            timeout_s=12.0, max_attempts=1, retry_delay_s=0.0
                        )
                        if not ok:
                            self._tv_blocked_until = now_m + 120.0
                            msg = TV_CONNECTIVITY_BLOCKED
                        else:
                            self._tv_blocked_until = 0.0
                except Exception:
                    pass
            self._set_error(task, msg)
        finally:
            with self._inflight_lock:
                self._inflight.discard(task.id)

    def _check_dev_alert(self, task: WatchTask) -> None:
        """价格偏离“持仓方案入场参考”超过配置阈值时推送飞书（去重：回到阈值一半内才解除）。

        入场参考 = 当前方向建立时锚定的收盘价（task.dir_entry_price）；
        现价优先用形成中 bar 的 live_price，回退到最近收盘价。
        阈值 rt_alert_dev_pct ∈ (0, 50]，0/缺省 = 关闭。
        """
        try:
            from web.settings import load_settings

            s = load_settings()
            try:
                thr = float(s.get("rt_alert_dev_pct") or 0.0)
            except (TypeError, ValueError):
                thr = 0.0
            if not (thr > 0.0):
                return
            if task.direction not in (DIR_LONG, DIR_SHORT):
                return
            entry = task.dir_entry_price
            px = task.live_price if task.live_price is not None else task.last_close
            if not entry or entry <= 0.0 or px is None or px <= 0.0:
                return
            dev_pct = (px / entry - 1.0) * 100.0
            if abs(dev_pct) >= thr and not task.dev_alerted:
                task.dev_alerted = True
                self._notify_deviation(task, px, dev_pct, thr)
            elif abs(dev_pct) <= thr * RT_DEV_RECOVER_FACTOR:
                task.dev_alerted = False
        except Exception as exc:  # noqa: BLE001 告警失败绝不影响信号
            print(f"[偏离告警] ✗ {task.symbol} 检查异常: {exc}", flush=True)

    def _maybe_stale_alert(self, task: WatchTask) -> None:
        """硬钝化飞书提醒：连续 >= N 根新 bar 因子几乎不变（变化<0.001%）且
        最近轨迹点几乎严格恒定（硬钝化，非缓漂）时推送一次，带轨迹摘要。

        阈值 rt_alert_stale_bars（根数）来自 web_settings，0/缺省 = 关闭。
        去重：同一次钝化只推一条；因子恢复变化（flat_run 归零）或方向翻转后
        才允许再次提醒。
        """
        try:
            from web.settings import load_settings  # 局部导入：便于测试 monkeypatch 生效

            s = load_settings()
            try:
                n_req = int(float(s.get("rt_alert_stale_bars") or 0.0))
            except (TypeError, ValueError):
                n_req = 0
            if n_req < 1:
                return
            if task.factor_flat_run < n_req:
                return
            if task._stale_pushed:
                return
            pts = [
                p[2]
                for p in list(task.trace_hist)
                if p and len(p) >= 3 and p[2] is not None and isinstance(p[2], (int, float))
            ]
            if not pts:
                return
            win = pts[-min(n_req, len(pts)) :]
            if len(win) < 3:
                return
            hi = float(max(win))
            lo = float(min(win))
            # 硬钝化 = 平台几乎严格恒定（浮点量级，1e-9 相对）；缓漂/阶梯式摆动不算
            if not (hi - lo <= 1e-9 * max(1.0, abs(hi), abs(lo))):
                return
            task._stale_pushed = True
            self._notify_stale(task, int(task.factor_flat_run), float(hi))
        except Exception as exc:  # noqa: BLE001 告警失败绝不影响信号
            print(f"[僵化告警] ✗ {task.symbol} 检查异常: {exc}", flush=True)

    def _notify_stale(self, task: WatchTask, flat_run: int, factor_val: float) -> None:
        """向飞书推送硬钝化提醒（带最近轨迹摘要）。"""
        try:
            from web.feishu_notify import notify_realtime_stale

            ok, msg = notify_realtime_stale(
                symbol=task.symbol,
                timeframe=task.timeframe,
                strategy_name=task.strategy_name,
                factor_value=task.factor_value,
                flat_run=flat_run,
                factor_val=float(factor_val),
                last_close=task.last_close,
                price_moved_run=int(task.price_moved_run),
                trace_tail=[list(x) for x in list(task.trace_hist)[-6:]],
            )
            if ok:
                print(
                    f"[飞书通知] ✓ {task.symbol} 因子硬钝化 {flat_run} 根已推送",
                    flush=True,
                )
            else:
                print(
                    f"[飞书通知] ✗ {task.symbol} 因子硬钝化推送失败: {msg}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[飞书通知] ✗ {task.symbol} 硬钝化告警异常: {exc}", flush=True)

    def _notify_deviation(self, task: WatchTask, px: float, dev_pct: float, thr: float) -> None:
        """向飞书推送偏离入场参考告警。"""
        try:
            from web.feishu_notify import notify_realtime_deviation

            ok, msg = notify_realtime_deviation(
                symbol=task.symbol,
                timeframe=task.timeframe,
                strategy_name=task.strategy_name,
                direction=task.direction,
                factor_value=task.factor_value,
                entry_price=task.dir_entry_price,
                current_price=px,
                dev_pct=dev_pct,
                threshold_pct=thr,
            )
            if ok:
                print(
                    f"[飞书通知] ✓ {task.symbol} 偏离入场 {dev_pct:+.2f}% 已推送",
                    flush=True,
                )
            else:
                print(
                    f"[飞书通知] ✗ {task.symbol} 偏离入场 {dev_pct:+.2f}% 推送失败: {msg}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[飞书通知] ✗ {task.symbol} 偏离告警异常: {exc}", flush=True)

    def _notify_direction_flip(self, task: WatchTask, prev_dir: str, new_dir: str) -> None:
        """向飞书推送方向转折通知，带日志和重试。"""
        try:
            from web.feishu_notify import notify_direction_flip

            ok, msg = notify_direction_flip(
                symbol=task.symbol,
                timeframe=task.timeframe,
                strategy_name=task.strategy_name,
                prev_direction=prev_dir,
                new_direction=new_dir,
                strength=task.strength,
                factor_value=task.factor_value,
            )
            if ok:
                print(
                    f"[飞书通知] ✓ {task.symbol} {task.timeframe} "
                    f"{prev_dir}→{new_dir} 推送成功",
                    flush=True,
                )
            else:
                print(
                    f"[飞书通知] ✗ {task.symbol} {task.timeframe} "
                    f"{prev_dir}→{new_dir} 推送失败: {msg}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[飞书通知] ✗ {task.symbol} {task.timeframe} "
                f"{prev_dir}→{new_dir} 异常: {exc}",
                flush=True,
            )

    def _set_error(self, task: WatchTask, message: str) -> None:
        task.state = "error"
        task.message = message
        task.updated_at = time.time()


realtime_manager = RealtimeManager()
