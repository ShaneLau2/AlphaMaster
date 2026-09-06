"""web/paper_manager.py — 模拟实盘引擎（纸上交易，不真实下单）

把「实时分析」的收盘信号变成**离散订单 + 成交日志**的模拟盘：

- 每个监控项 = (数据源, 品种, 周期, 策略因子)；在 K 线收盘后按因子信号
  决定方向（LONG / SHORT / FLAT，阈值与实时分析一致）。
- 只在**方向翻转**时成交：以信号 bar 收盘价成交（按滑点调整），开/平仓各收
  一次手续费，成交流水与持仓/资金曲线全程记录并落盘，重启可恢复。
- 成交与回测同口径：只看已收盘 bar；引擎暂停期间错过的 bar 不补单，
  恢复后按最新收盘信号对账（启动即调仓）。

账户为「模拟货币」：每个持仓按 名义金额 × |tanh(因子)| 计仓，
多空盈亏按价格比例折算到账户货币，双向交易（可做空）假设。
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from web.data_sources.base import bars_to_raw_dict
from web.data_sources.factory import get_source
from web.realtime_manager import (
    _VALID_KINDS,
    cadence_for,
    ensure_closed_bars,
    fetch_cached_bars,
    load_strategy_meta,
)
from strategy_manager.live_signal import DIR_FLAT, DIR_LONG, DIR_SHORT, evaluate_signal
from web.hold_policy import (
    DD_STATE_DEEP,
    DD_STATE_GATED,
    DD_STATE_OK,
    EXIT_REASON_LABEL,
    atr_last,
    check_policy_exit,
    combo_id,
    combo_label,
    combo_parts,
    dd_ladder_step,
    params_for,
    policy_has_exits,
    sl_tp_levels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = PROJECT_ROOT / "data" / "paper_sim_state.json"

# 默认账户参数（可在界面/设置里改）
_DEFAULT_BALANCE = 100_000.0      # 起始资金（模拟货币）
_DEFAULT_NOTIONAL = 10_000.0      # 每份满仓名义金额
_DEFAULT_COMMISSION_PCT = 0.02    # 单边手续费 %
_DEFAULT_SLIPPAGE_PCT = 0.01      # 单边滑点 %

_MAX_TRADES = 400                # 保留的成交记录条数
_MAX_EQUITY_POINTS = 2000        # 资金曲线最多记录点数
_SNAPSHOT_MIN_INTERVAL = 120.0   # 引擎运行中无成交时的记点间隔（秒）


def _num(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# ─────────────────────────────────────────────────────────────────────
# 纯结算函数（便于单测）
# ─────────────────────────────────────────────────────────────────────

def clamp_strength(strength: float | None) -> float:
    """信号强度夹到 [0, 1]（|tanh(因子)|）。"""
    s = _num(strength, 0.0)
    return max(0.0, min(1.0, s))


def fill_price(price: float, side: str, slippage_pct: float) -> float:
    """按方向施加滑点的成交价：买入更贵、卖出更便宜。"""
    p = _num(price, 0.0)
    slip = max(0.0, _num(slippage_pct, 0.0)) / 100.0
    if side == DIR_LONG:
        return p * (1.0 + slip)
    if side == DIR_SHORT:
        return p * (1.0 - slip)
    return p


def fee_for_notional(notional_value: float, commission_pct: float) -> float:
    """按名义金额收取的单边手续费（非负）。"""
    nv = abs(_num(notional_value, 0.0))
    comm = max(0.0, _num(commission_pct, 0.0)) / 100.0
    return nv * comm


def open_position(
    side: str,
    price: float,
    strength: float,
    notional_base: float,
    commission_pct: float,
    slippage_pct: float,
) -> dict[str, float]:
    """按 名义金额 × 强度 开仓，返回 {qty, fill, notional_value, fee}。"""
    strength = clamp_strength(strength)
    base = max(0.0, _num(notional_base, 0.0))
    p = _num(price, 0.0)
    if p <= 0.0 or base <= 0.0 or strength <= 0.0:
        return {"qty": 0.0, "fill": 0.0, "notional_value": 0.0, "fee": 0.0}
    notional_value = base * strength
    fill = fill_price(p, side, slippage_pct)
    if fill <= 0.0:
        return {"qty": 0.0, "fill": 0.0, "notional_value": notional_value, "fee": 0.0}
    return {
        "qty": round(notional_value / fill, 10),
        "fill": round(fill, 10),
        "notional_value": round(notional_value, 4),
        "fee": round(fee_for_notional(notional_value, commission_pct), 6),
    }


def close_position(
    pos: dict[str, Any],
    price: float,
    commission_pct: float,
    slippage_pct: float,
) -> dict[str, float]:
    """按现价平掉一个持仓，返回 {pnl, fee, fill}。"""
    side = pos.get("side")
    qty = _num(pos.get("qty"), 0.0)
    entry = _num(pos.get("entry_price"), 0.0)
    notional_value = _num(pos.get("notional_value"), 0.0)
    # 平多 = 卖出（价格下移），平空 = 买入（价格上移），与开仓方向相反
    close_side = DIR_SHORT if side == DIR_LONG else DIR_LONG
    fill = fill_price(_num(price, 0.0), close_side, slippage_pct)
    if side == DIR_LONG:
        pnl = (fill - entry) * qty
    elif side == DIR_SHORT:
        pnl = (entry - fill) * qty
    else:
        pnl = 0.0
    return {"pnl": round(pnl, 6), "fee": round(fee_for_notional(notional_value, commission_pct), 6), "fill": round(fill, 10)}


def unrealized_pnl(pos: dict[str, Any], mark_price: float) -> float:
    """未实现盈亏（按现价 mark，不再叠加滑点）。"""
    side = pos.get("side")
    qty = _num(pos.get("qty"), 0.0)
    entry = _num(pos.get("entry_price"), 0.0)
    mark = _num(mark_price, 0.0)
    if side == DIR_LONG:
        return round((mark - entry) * qty, 6)
    if side == DIR_SHORT:
        return round((entry - mark) * qty, 6)
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# 监控项
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PaperWatch:
    """模拟盘监控项 = 数据源 + 品种 + 周期 + 策略因子。"""
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
    notional: float
    policy_id: str = "signal"      # 持仓管理方案：signal / risk / hybrid / be / time / chandelier / dd
    cooldown_side: str | None = None  # 止损/止盈出场后，同方向不再立刻重开，等信号先翻转
    dd_gate: int = 0              # dd 方案：DD_STATE_OK/GATED/DEEP（重启后重置，峰值重新累计）
    dd_peak: float | None = None  # dd 方案：该监控所见账户净值滚动峰值
    dd_pct: float | None = None   # dd 方案：当前相对峰值的回撤幅度（%，负值）
    # 运行时状态
    state: str = "pending"          # pending|ok|insufficient|error
    direction: str | None = None
    strength: float | None = None
    position: float | None = None
    factor_value: float | None = None
    last_bar_ts: int | None = None
    processed_bar_ts: int | None = None
    last_close: float | None = None
    updated_at: float | None = None
    message: str = ""
    warn: str = ""
    next_due: float = 0.0
    error: str = ""

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_name": self.strategy_name,
            "strategy_symbol": self.strategy_symbol,
            "strategy_timeframe": self.strategy_timeframe,
            "best_score": self.best_score,
            "notional": self.notional,
            "policy_id": self.policy_id,
            "cooldown": self.cooldown_side,
            "dd_gate": self.dd_gate,
            "dd_peak": self.dd_peak,
            "dd_pct": self.dd_pct,
            "state": self.state,
            "direction": self.direction,
            "strength": self.strength,
            "position": self.position,
            "factor_value": self.factor_value,
            "last_bar_ts": self.last_bar_ts,
            "last_close": self.last_close,
            "updated_at": self.updated_at,
            "message": self.message,
            "warn": self.warn,
            "seconds_to_next": (
                max(0, int(self.next_due - time.time())) if self.next_due else None
            ),
        }

    def persist_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_file": self.strategy_file,
            "notional": self.notional,
            "policy_id": self.policy_id,
        }


# ─────────────────────────────────────────────────────────────────────
# 模拟盘引擎
# ─────────────────────────────────────────────────────────────────────

class PaperTradingManager:
    """模拟盘单例：轮询已收盘 K 线 → 方向翻转时模拟开/平仓 → 记账落盘。"""

    def __init__(
        self,
        state_file: Path | str | None = None,
        *,
        starting_balance: float | None = None,
        commission_pct: float | None = None,
        slippage_pct: float | None = None,
        default_notional: float | None = None,
        max_position_pct: float | None = None,
    ) -> None:
        self.state_file = Path(state_file) if state_file is not None else STATE_FILE
        # 账户参数（None 时从 web_settings 读取；单测可直接注入固定值）
        self._cfg_balance = starting_balance
        self._cfg_commission = commission_pct
        self._cfg_slippage = slippage_pct
        self._cfg_notional = default_notional
        self._cfg_max_pos_pct = max_position_pct

        self._watches: dict[str, PaperWatch] = {}
        self._positions: dict[str, dict[str, Any]] = {}
        self._trades: deque[dict[str, Any]] = deque(maxlen=_MAX_TRADES)
        self._eq_ts: deque[int] = deque(maxlen=_MAX_EQUITY_POINTS)
        self._eq_val: deque[float] = deque(maxlen=_MAX_EQUITY_POINTS)

        self.cash: float = _DEFAULT_BALANCE
        self.realized_pnl: float = 0.0
        self.fees_paid: float = 0.0
        self.n_trades: int = 0
        self._seq: int = 0
        self._last_snapshot_at: float = 0.0
        self._milestone_band: int = 0    # 已通知过的盈亏档位（每跨起始资金 ±1% 提醒一次）
        self._started_at: float | None = None

        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="paper")
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()
        self._loaded = False

    # ── 配置 ──────────────────────────────────────────────────────────
    def config(self) -> dict[str, float]:
        """账户参数：优先显式注入，其次 web_settings（界面可改）。"""
        try:
            from web.settings import load_settings
            s = load_settings()
        except Exception:  # noqa: BLE001
            s = {}
        return {
            "starting_balance": _num(
                self._cfg_balance if self._cfg_balance is not None else s.get("paper_starting_balance"),
                _DEFAULT_BALANCE,
            ),
            "commission_pct": _num(
                self._cfg_commission if self._cfg_commission is not None else s.get("paper_commission_pct"),
                _DEFAULT_COMMISSION_PCT,
            ),
            "slippage_pct": _num(
                self._cfg_slippage if self._cfg_slippage is not None else s.get("paper_slippage_pct"),
                _DEFAULT_SLIPPAGE_PCT,
            ),
            "notional": _num(
                self._cfg_notional if self._cfg_notional is not None else s.get("paper_notional"),
                _DEFAULT_NOTIONAL,
            ),
            "max_position_pct": min(
                200.0,
                max(1.0, _num(
                    self._cfg_max_pos_pct
                    if self._cfg_max_pos_pct is not None else s.get("max_position_pct"),
                    100.0,
                )),
            ),
            # 统一无信号阈值：实时评估时生效（FLAT 观望区），随设置热更新
            "signal_threshold": _num(s.get("signal_threshold"), 0.05),
        }

    def _notional(self) -> float:
        return self.config()["notional"]

    # ── 持久化 ────────────────────────────────────────────────────────
    def load_persisted(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        with self._lock:
            for w in data.get("watches") or []:
                try:
                    self._add_watch_internal(
                        w["source"], w["symbol"], w["timeframe"], w["strategy_file"],
                        notional=_num(w.get("notional"), self._notional()),
                        policy_id=w.get("policy_id"),
                        persist=False,
                    )
                except Exception:
                    continue
            self.cash = _num(data.get("cash"), self.config()["starting_balance"])
            self.realized_pnl = _num(data.get("realized_pnl"), 0.0)
            self.fees_paid = _num(data.get("fees_paid"), 0.0)
            self.n_trades = int(data.get("n_trades") or 0)
            self._seq = int(data.get("seq") or 0)
            self._started_at = data.get("started_at")
            self._milestone_band = self._current_pnl_band()
            for pid, p in (data.get("positions") or {}).items():
                if pid in self._watches and isinstance(p, dict):
                    self._positions[pid] = p
            trades = data.get("trades") or []
            for t in reversed(trades[-_MAX_TRADES:]):
                if isinstance(t, dict):
                    self._trades.appendleft(t)
            eq = data.get("equity") or {}
            ts_list, val_list = eq.get("ts") or [], eq.get("equity") or []
            for t0, v0 in zip(ts_list, val_list):
                self._eq_ts.append(int(t0))
                self._eq_val.append(float(v0))
        if self._watches:
            self.start()

    def _save(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        with self._lock:
            data = {
                "version": 1,
                "watches": [w.persist_dict() for w in self._watches.values()],
                "cash": round(self.cash, 6),
                "realized_pnl": round(self.realized_pnl, 6),
                "fees_paid": round(self.fees_paid, 6),
                "n_trades": self.n_trades,
                "seq": self._seq,
                "started_at": self._started_at,
                "positions": self._positions,
                "trades": list(self._trades),
                "equity": {"ts": list(self._eq_ts), "equity": list(self._eq_val)},
                "updated_at": time.time(),
            }
        try:
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.state_file)
        except OSError:
            pass

    # ── 增删监控项 ────────────────────────────────────────────────────
    def add_watch(self, source: str, symbol: str, timeframe: str, strategy_file: str,
                  policy_id: str = "signal") -> dict[str, Any]:
        watch = self._add_watch_internal(
            source, symbol, timeframe, strategy_file,
            notional=self._notional(), policy_id=policy_id, persist=True,
        )
        self.start()
        return watch.to_public()

    def _add_watch_internal(
        self, source: str, symbol: str, timeframe: str, strategy_file: str,
        *, notional: float, persist: bool, policy_id: str | None = None,
    ) -> PaperWatch:
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
        meta = load_strategy_meta(path)
        if not meta.get("formula"):
            raise ValueError("策略文件缺少 formula")

        from model_core.vocab import VOCAB_VERSION

        name = Path(path).stem
        task_id = f"{source}:{symbol}:{timeframe}:{name}"

        warn = ""
        if meta.get("vocab_version") and meta["vocab_version"] not in (VOCAB_VERSION, "legacy"):
            warn = f"词表版本不符（{meta['vocab_version']} vs {VOCAB_VERSION}），信号可能失真"
        elif meta.get("symbol") and meta["symbol"] != symbol:
            warn = f"该因子为 {meta['symbol']} 训练，跨品种运行仅供参考"

        watch = PaperWatch(
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
            cadence_s=cadence_for(timeframe),
            notional=float(notional),
            policy_id=combo_id(policy_id),
            warn=warn,
            next_due=0.0,
        )
        with self._lock:
            self._watches[task_id] = watch
            if persist:
                self._save()
        return watch

    def remove_watch(self, task_id: str) -> bool:
        """移除监控项；若仍持有未平仓位，先按最新收盘价平掉。"""
        with self._lock:
            watch = self._watches.get(task_id)
            if not watch:
                return False
            self._close_position_internal(task_id, watch.last_close, reason="移除监控")
            del self._watches[task_id]
            self._save()
        if not self._watches:
            self.stop()
        return True

    def clear(self) -> None:
        with self._lock:
            for wid in list(self._positions):
                watch = self._watches.get(wid)
                if watch is not None:
                    self._close_position_internal(wid, watch.last_close, reason="清空监控")
            self._watches.clear()
            self._save()
        self.stop()

    # ── 启停 ──────────────────────────────────────────────────────────
    def start(self) -> None:
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                return
            self._running = True
            if self._started_at is None:
                self._started_at = time.time()
            self._thread = threading.Thread(target=self._loop, name="paper", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                pass
            time.sleep(1.0)

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            due = [t for t in self._watches.values() if now >= t.next_due]
        for task in due:
            with self._inflight_lock:
                if task.id in self._inflight:
                    continue
                self._inflight.add(task.id)
            task.next_due = now + task.cadence_s
            self._executor.submit(self._process_watch, task)

    # ── 行情 + 信号 → 订单 ────────────────────────────────────────────
    def _process_watch(self, watch: PaperWatch) -> None:
        try:
            bars = fetch_cached_bars(watch.source, watch.symbol, watch.timeframe)
            bars = ensure_closed_bars(bars, watch.timeframe)
            if not bars:
                self._set_error(watch, "未获取到 K 线")
                return
            last_ts = int(bars[-1].ts)
            last_bar = bars[-1]
            if watch.processed_bar_ts == last_ts:
                # 同一根已收盘 bar：无新信号，仅刷新市价（mark-to-market）
                watch.last_bar_ts = last_ts
                watch.last_close = float(last_bar.close)
                watch.updated_at = time.time()
                self._maybe_snapshot()
                return

            # 持仓计时：进入新已收盘 bar（仍在途）→ 持仓根数 +1
            pos = self._positions.get(watch.id)
            if pos is not None:
                pos["bars_held"] = int(_num(pos.get("bars_held"), 0)) + 1
            watch.last_bar_ts = last_ts
            watch.last_close = float(last_bar.close)
            watch.updated_at = time.time()
            last_close = watch.last_close

            # dd（行情回撤熔断）：按“当前已收盘 bar 相对滚动峰值行情价”判档，
            # 触档先平掉在途仓位，并暂停新开仓直到行情收复（与回放引擎同口径）。
            dd_action: str | None = None
            dd_event: dict[str, Any] | None = None
            if "dd" in combo_parts(watch.policy_id) and watch.processed_bar_ts is not None:
                dd_action, dd_event = self._dd_gate_step(watch)
            if dd_event:
                self._notify_dd_gate(watch, dd_event)
            if dd_action == "exit":
                pos = self._positions.get(watch.id)
                side = pos.get("side") if pos else None
                if pos is not None:
                    parts = combo_parts(watch.policy_id)
                    ctx = combo_label(watch.policy_id) if len(parts) > 1 else ""
                    reason_txt = f"回撤熔断（持仓管理{('·' + ctx) if ctx else ''}）"
                    self._close_position_internal(watch.id, last_close, reason=reason_txt)
                    watch.cooldown_side = side
                self._maybe_snapshot()
                return

            raw = bars_to_raw_dict(bars)
            result = evaluate_signal(watch.formula, raw, threshold=self.config()["signal_threshold"])
            watch.state = result.get("state", "error")
            watch.message = result.get("message", "")
            if watch.state == "ok":
                watch.processed_bar_ts = last_ts
                watch.direction = result.get("direction")
                watch.strength = result.get("strength")
                watch.position = result.get("position")
                watch.factor_value = result.get("factor_value")
                # 持仓管理硬出场（止损/止盈/移动止损/吊灯/时间）优先于信号翻转：
                # 同一根已收盘 bar 内先看是否触及风控价，命中则平仓并冷却同向重开。
                # dd 模块走上面独立的账户级阶梯，其余模块在此逐 bar 检查（组合可叠加）。
                if (self._positions.get(watch.id) is not None
                        and policy_has_exits(watch.policy_id)):
                    if self._policy_exit_check(watch, last_bar, bars):
                        watch.processed_bar_ts = last_ts
                        self._maybe_snapshot()
                        return
                self._reconcile(watch, watch.direction, watch.strength, last_close, last_ts)
            elif watch.state in ("insufficient", "pending"):
                watch.direction = None
                watch.strength = None
            else:
                watch.direction = None
                watch.strength = None
                watch.error = watch.message
            self._maybe_snapshot()
        except Exception as exc:  # noqa: BLE001
            self._set_error(watch, str(exc))
        finally:
            with self._inflight_lock:
                self._inflight.discard(watch.id)

    def _reconcile(
        self,
        watch: PaperWatch,
        direction: str | None,
        strength: float | None,
        price: float,
        bar_ts: int,
    ) -> None:
        """把最新收盘信号对账成离散订单（仅在方向翻转时成交）。"""
        with self._lock:
            pos = self._positions.get(watch.id)
            direction = direction if direction in (DIR_LONG, DIR_SHORT, DIR_FLAT) else DIR_FLAT
            # 冷却：止损/止盈出场后同方向需等信号先翻转（避免刚止损又追回去）
            cd = watch.cooldown_side
            if cd and direction != cd:
                watch.cooldown_side = None
                cd = None
            if pos is None and direction == DIR_FLAT:
                return
            if pos is None and direction in (DIR_LONG, DIR_SHORT):
                if cd == direction:
                    return
                # dd 熔断期内暂停新开仓（即使信号转强），待回撤收复（dd_gate==OK）
                if "dd" in combo_parts(watch.policy_id) and int(watch.dd_gate or 0) != DD_STATE_OK:
                    return
                self._open_position_internal(watch, direction, price, strength, bar_ts, reason="信号开仓")
                return
            if pos is not None and direction == DIR_FLAT:
                self._close_position_internal(watch.id, price, reason="信号转观望")
                return
            if pos is not None and pos.get("side") != direction:
                # 反手：先平旧仓再开新方向（两笔成交、各收一次费用）
                self._close_position_internal(watch.id, price, reason=f"反手 → 平仓")
                self._open_position_internal(watch, direction, price, strength, bar_ts, reason=f"反手 → 开{direction_label(direction)}")
                return
            # 同方向：保持不动（与实盘 runner 一致，不做同向加仓）

    def _open_position_internal(
        self,
        watch: PaperWatch,
        side: str,
        price: float,
        strength: float | None,
        bar_ts: int,
        *,
        reason: str,
    ) -> None:
        cfg = self.config()
        # 每笔投入上限（权益复合式）：名义 ≤ min(满仓名义金额, 当时账户权益 × 上限%)
        cap_frac = float(cfg.get("max_position_pct") or 100.0) / 100.0
        eq_now = self.equity()
        base = float(watch.notional or cfg["notional"])
        if base <= 0.0:
            return
        if eq_now > 0.0:
            base = min(base, eq_now * cap_frac)
        fill = open_position(
            side, price, strength, base,
            cfg["commission_pct"], cfg["slippage_pct"],
        )
        if fill["qty"] <= 0.0:
            return
        self.cash -= fill["fee"]
        self.fees_paid += fill["fee"]
        entry = float(fill["fill"])
        levels = sl_tp_levels(side, entry, watch.policy_id)
        pos = {
            "watch_id": watch.id,
            "side": side,
            "qty": fill["qty"],
            "entry_price": entry,
            "entry_bar_ts": bar_ts,
            "opened_at": time.time(),
            "notional_value": fill["notional_value"],
            "strength": clamp_strength(strength),
            "policy_id": watch.policy_id,
            "stop_price": levels["stop_price"],
            "target_price": levels["target_price"],
            "peak_fav": entry,
            "bars_held": 0,
        }
        self._positions[watch.id] = pos
        self._seq += 1
        action = "开多" if side == DIR_LONG else "开空"
        self._trades.appendleft({
            "ts": time.time(),
            "seq": self._seq,
            "watch_id": watch.id,
            "symbol": watch.symbol,
            "timeframe": watch.timeframe,
            "source": watch.source,
            "action": action,
            "price": round(fill["fill"], 8),
            "raw_price": round(_num(price, 0.0), 8),
            "qty": fill["qty"],
            "notional_value": round(fill["notional_value"], 4),
            "fee": round(fill["fee"], 4),
            "pnl": 0.0,
            "cash_after": round(self.cash, 2),
            "reason": reason,
        })
        self._snapshot_point(bar_ts)
        self._save()
        self._notify_trade(self._trades[0])

    def _close_position_internal(
        self, watch_id: str, price: float | None, *, reason: str,
    ) -> bool:
        pos = self._positions.get(watch_id)
        if pos is None:
            return False
        watch = self._watches.get(watch_id)
        mark = price if price is not None else (watch.last_close if watch else None)
        if mark is None or mark <= 0.0:
            return False
        cfg = self.config()
        flow = close_position(pos, mark, cfg["commission_pct"], cfg["slippage_pct"])
        self.cash += flow["pnl"] - flow["fee"]
        self.realized_pnl += flow["pnl"]
        self.fees_paid += flow["fee"]
        self.n_trades += 1
        self._seq += 1
        side = pos.get("side")
        action = "平多" if side == DIR_LONG else "平空"
        self._trades.appendleft({
            "ts": time.time(),
            "seq": self._seq,
            "watch_id": watch_id,
            "symbol": (watch.symbol if watch else ""),
            "timeframe": (watch.timeframe if watch else ""),
            "source": (watch.source if watch else ""),
            "action": action,
            "price": round(flow["fill"], 8),
            "raw_price": round(_num(mark, 0.0), 8),
            "qty": pos.get("qty"),
            "notional_value": round(_num(pos.get("notional_value"), 0.0), 4),
            "fee": round(flow["fee"], 4),
            "pnl": round(flow["pnl"], 2),
            "cash_after": round(self.cash, 2),
            "reason": reason,
        })
        del self._positions[watch_id]
        self._snapshot_point(int(pos.get("entry_bar_ts") or 0))
        self._save()
        self._notify_trade(self._trades[0])
        return True

    def _policy_exit_check(self, watch: PaperWatch, bar: Any,
                           bars: list[Any] | None = None) -> bool:
        """持仓管理方案的硬出场检查（只在已收盘 bar 上触发）。

        命中（止损/止盈/移动止损/保本/吊灯/时间）→ 平仓并记 reason；之后同方向
        进入冷却，待信号翻转后才能再开（避免刚止损又同向追回）。
        """
        with self._lock:
            pos = self._positions.get(watch.id)
            if pos is None:
                return False
            pid = pos.get("policy_id") or watch.policy_id
            if not policy_has_exits(pid):
                return False
            # chandelier：用最近一根已就绪 ATR（Wilder，含当前 bar）；未就绪回退静态兜底
            atr_now = None
            if pid == "chandelier" and bars:
                try:
                    per = int(params_for(pid).get("atr_period") or 14)
                    atr_now = atr_last(
                        [float(b.high) for b in bars],
                        [float(b.low) for b in bars],
                        [float(b.close) for b in bars],
                        per,
                    )
                except Exception:  # noqa: BLE001
                    atr_now = None
            fill, reason, peak = check_policy_exit(
                side=pos.get("side"),
                entry=_num(pos.get("entry_price"), 0.0),
                bar_open=float(bar.open),
                bar_high=float(bar.high),
                bar_low=float(bar.low),
                peak_fav=_num(pos.get("peak_fav"), pos.get("entry_price")),
                policy_id=pid,
                bar_close=float(getattr(bar, "close", watch.last_close or 0.0)
                                or watch.last_close or 0.0),
                bars_held=int(_num(pos.get("bars_held"), 0)),
                atr=atr_now,
            )
            pos["peak_fav"] = peak
            if fill is None or reason is None:
                return False
            side = pos.get("side")
            label = EXIT_REASON_LABEL.get(reason, reason)
            parts = combo_parts(pid)
            ctx = combo_label(pid) if len(parts) > 1 else ""
            reason_txt = f"{label}（持仓管理{('·' + ctx) if ctx else ''}）"
            self._close_position_internal(watch.id, fill, reason=reason_txt)
            watch.cooldown_side = side
            return True

    def _dd_gate_step(self, watch: PaperWatch) -> tuple[str, dict[str, Any] | None]:
        """dd 方案：行情回撤阶梯一步（在已收盘的新 bar 上，按当前收盘价 vs 滚动峰值）。

        返回 (action, event)：action ∈ {"ok"/"block"/"exit"}；event 仅在发生状态转移时
        非空（熔断 / 深档熔断 / 收复），供飞书单独告警。状态与峰值存 watch
        （重启后重置，峰值从重启后收盘价重新累计）。净值 = 现金 + 仓位×行情，
        单品种下行情回撤即净值回撤的同构量；现金态下行情收复后能自然恢复开仓，
        与回放引擎完全同口径。
        """
        with self._lock:
            mark = _num(watch.last_close, 0.0)
            if mark <= 0.0:
                return "ok", None  # 尚无行情价：不动闸（首个真实 bar 会播种峰值）
            pk = _num(watch.dd_peak, mark)
            dd = (mark / pk - 1.0) if pk > 0.0 else 0.0
            prev_st = int(watch.dd_gate or 0)
            action, st = dd_ladder_step(dd, params_for("dd"), prev_st)
            watch.dd_gate = st
            watch.dd_peak = max(pk, mark)
            watch.dd_pct = dd * 100.0  # 供卡片展示当前回撤幅度
        # 状态转移事件（锁外组装）
        event: dict[str, Any] | None = None
        if action == "exit":
            event = {
                "event": "深档熔断" if st == DD_STATE_DEEP else "熔断",
                "dd_pct": dd * 100.0, "mark": mark, "peak": pk,
                "prev_state": prev_st, "state": st,
            }
        elif prev_st in (DD_STATE_GATED, DD_STATE_DEEP) and st == DD_STATE_OK:
            event = {"event": "收复", "dd_pct": dd * 100.0, "mark": mark, "peak": pk,
                     "prev_state": prev_st, "state": st}
        return action, event

    # ── 飞书通知（与实时分析共用配置，未启用时零成本空转） ──────────────
    def _notify_trade(self, row: dict[str, Any]) -> None:
        """成交提醒；末尾顺带检查是否跨过盈亏里程碑。"""
        try:
            import web.feishu_notify as _fn

            ok, msg = _fn.notify_paper_trade(
                symbol=row.get("symbol") or "",
                timeframe=row.get("timeframe") or "",
                action=row.get("action") or "",
                price=row.get("price"),
                qty=row.get("qty"),
                notional_value=row.get("notional_value"),
                fee=row.get("fee"),
                pnl=row.get("pnl"),
                cash_after=row.get("cash_after"),
                equity=round(self.equity(), 2),
                reason=row.get("reason") or "",
            )
            if ok:
                print(f"[飞书通知] ✓ 模拟盘成交 {row.get('symbol')} {row.get('action')}", flush=True)
            else:
                print(f"[飞书通知] ✗ 模拟盘成交 {row.get('symbol')} {row.get('action')}: {msg}", flush=True)
        except Exception as exc:  # noqa: BLE001 通知失败绝不影响成交记账
            print(f"[飞书通知] ✗ 模拟盘通知异常: {exc}", flush=True)
        self._notify_milestone()

    # ── 飞书：DD 熔断 / 收复单独告警（正交组合里的 dd 模块事件） ──────
    def _notify_dd_gate(self, watch: PaperWatch, event: dict[str, Any]) -> None:
        # 每次状态转移先落 JSONL 事件日志（回溯每次熔断/收复）
        try:
            from web.dd_events import log_dd_event

            log_dd_event(
                "paper",
                symbol=watch.symbol, timeframe=watch.timeframe,
                policy_id=watch.policy_id,
                event=str(event.get("event") or "熔断"),
                dd_pct=event.get("dd_pct"),
                mark=event.get("mark"),
                peak=event.get("peak"),
                prev_state=int(event.get("prev_state") or 0),
                state=int(event.get("state") or 0),
            )
        except Exception:  # noqa: BLE001 日志失败绝不影响告警
            pass
        try:
            import web.feishu_notify as _fn

            ev = str(event.get("event") or "熔断")
            pct = event.get("dd_pct")
            p = params_for("dd")
            rec_hint = ""
            if ev == "收复":
                rec_hint = f"已收复（回撤 {pct:+.2f}%），恢复开仓"
            elif event.get("state") == DD_STATE_DEEP:
                thr = float(p.get("deep_recover_thr_pct") or 2.5)
                rec_hint = f"深档熔断：需收复到回撤 ≤{thr:.1f}% 才恢复开仓"
            else:
                thr = float(p.get("recover_thr_pct") or 1.0)
                rec_hint = f"熔断期暂停新开仓；需收复到回撤 ≤{thr:.1f}% 才恢复"
            pos = self._positions.get(watch.id)
            pos_txt = ("在途持仓" if pos else "无在途持仓")
            if pos:
                pos_txt += f"（{pos.get('side')}）"
            ok, msg = _fn.notify_paper_dd_gate(
                symbol=watch.symbol,
                timeframe=watch.timeframe,
                policy_label=combo_label(watch.policy_id),
                event=ev,
                dd_pct=pct,
                mark=event.get("mark"),
                peak=event.get("peak"),
                recover_hint=rec_hint,
                position=pos_txt,
            )
            if ok:
                print(f"[飞书通知] ✓ DD {ev} {watch.symbol}（{pct:+.2f}%）", flush=True)
            else:
                print(f"[飞书通知] ✗ DD {ev} {watch.symbol}: {msg}", flush=True)
        except Exception as exc:  # noqa: BLE001 通知失败绝不影响成交记账
            print(f"[飞书通知] ✗ DD 告警异常: {exc}", flush=True)

    def _milestone_step(self) -> float:
        """里程碑步长：起始资金的 1%（至少 $1）。"""
        bal = abs(self.config()["starting_balance"])
        return max(1.0, bal * 0.01)

    def _current_pnl_band(self) -> int:
        pnl = self.equity() - self.config()["starting_balance"]
        return int(abs(pnl) // self._milestone_step())

    def _notify_milestone(self) -> None:
        """净值每跨越一个新的 ±1% 档位推一次里程碑（频率受限，绝不刷屏）。"""
        try:
            band = self._current_pnl_band()
            if band <= self._milestone_band:
                return
            import web.feishu_notify as _fn

            ok, msg = _fn.notify_paper_milestone(
                equity=round(self.equity(), 2),
                starting_balance=self.config()["starting_balance"],
            )
            if ok:
                self._milestone_band = band
                print(f"[飞书通知] ✓ 模拟盘里程碑（第 {band} 档）", flush=True)
            else:
                print(f"[飞书通知] ✗ 模拟盘里程碑: {msg}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[飞书通知] ✗ 模拟盘里程碑异常: {exc}", flush=True)

    # ── 手工平仓 / 账户操作 ────────────────────────────────────────────
    def close_position(self, watch_id: str) -> bool:
        """手动平掉单个持仓（按该监控项最近一次市价）。"""
        with self._lock:
            watch = self._watches.get(watch_id)
            if watch is None or watch_id not in self._positions:
                return False
            return self._close_position_internal(watch_id, watch.last_close, reason="手动平仓")

    def close_all(self) -> int:
        """手动全平（未持仓的跳过）。"""
        count = 0
        with self._lock:
            for wid in list(self._positions):
                watch = self._watches.get(wid)
                if watch is None:
                    continue
                if self._close_position_internal(wid, watch.last_close, reason="手动全平"):
                    count += 1
        return count

    def reset(self, starting_balance: float | None = None) -> None:
        """重置账户：清空持仓/流水/资金曲线，资金回到起始值（保留监控项）。

        重置后每个监控项按「最新已收盘 bar 信号」重新对账（等价启动即调仓）。
        """
        with self._lock:
            self._positions.clear()
            self._trades.clear()
            self._eq_ts.clear()
            self._eq_val.clear()
            cfg = self.config()
            self.cash = _num(starting_balance, cfg["starting_balance"])
            self.realized_pnl = 0.0
            self.fees_paid = 0.0
            self.n_trades = 0
            self._seq = 0
            self._last_snapshot_at = 0.0
            self._milestone_band = 0
            for w in self._watches.values():
                w.processed_bar_ts = None
                w.state = "pending"
                w.direction = None
                w.strength = None
                w.cooldown_side = None
                w.next_due = 0.0
            self._snapshot_point(int(time.time()))
            self._save()

    # ── 资金曲线 / 快照 ────────────────────────────────────────────────
    def equity(self) -> float:
        """账户净值 = 现金 + 全部持仓未实现盈亏（按最近市价 mark）。"""
        unrealized = 0.0
        for wid, pos in self._positions.items():
            watch = self._watches.get(wid)
            mark = watch.last_close if watch is not None else None
            unrealized += unrealized_pnl(pos, mark) if mark else 0.0
        return self.cash + unrealized

    def _snapshot_point(self, bar_ts: int) -> None:
        eq = round(self.equity(), 6)
        ts = int(bar_ts or time.time())
        if self._eq_ts and self._eq_ts[-1] == ts:
            self._eq_ts.pop()
            self._eq_val.pop()
        self._eq_ts.append(ts)
        self._eq_val.append(eq)
        self._last_snapshot_at = time.time()

    def _maybe_snapshot(self) -> None:
        """引擎运行中无成交时，也定期记一个资金点（≥_SNAPSHOT_MIN_INTERVAL）。"""
        if time.time() - self._last_snapshot_at < _SNAPSHOT_MIN_INTERVAL:
            return
        self._snapshot_point(int(time.time()))
        self._save()
        self._notify_milestone()

    def _set_error(self, watch: PaperWatch, message: str) -> None:
        watch.state = "error"
        watch.message = message
        watch.updated_at = time.time()

    # ── 状态汇总 ───────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self._lock:
            watches = [w.to_public() for w in self._watches.values()]
            cfg = self.config()
            positions = []
            unrealized_total = 0.0
            for wid, pos in self._positions.items():
                watch = self._watches.get(wid)
                mark = watch.last_close if watch is not None else None
                upnl = unrealized_pnl(pos, mark) if mark else 0.0
                unrealized_total += upnl
                positions.append({
                    "watch_id": wid,
                    "symbol": (watch.symbol if watch else ""),
                    "timeframe": (watch.timeframe if watch else ""),
                    "source": (watch.source if watch else ""),
                    "strategy_name": (watch.strategy_name if watch else ""),
                    "side": pos.get("side"),
                    "policy_id": pos.get("policy_id") or (watch.policy_id if watch else "signal"),
                    "qty": round(_num(pos.get("qty"), 0.0), 10),
                    "entry_price": round(_num(pos.get("entry_price"), 0.0), 8),
                    "mark_price": round(mark, 8) if mark else None,
                    "stop_price": pos.get("stop_price"),
                    "target_price": pos.get("target_price"),
                    "notional_value": round(_num(pos.get("notional_value"), 0.0), 2),
                    "strength": round(_num(pos.get("strength"), 0.0), 4),
                    "opened_at": pos.get("opened_at"),
                    "unrealized_pnl": round(upnl, 2),
                })
            equity_val = round(self.cash + unrealized_total, 2)
            total_pnl = round(equity_val - cfg["starting_balance"], 2)
            return {
                "running": self._running,
                "count": len(watches),
                "watches": watches,
                "config": {
                    "starting_balance": round(cfg["starting_balance"], 2),
                    "notional": round(cfg["notional"], 2),
                    "commission_pct": round(cfg["commission_pct"], 4),
                    "slippage_pct": round(cfg["slippage_pct"], 4),
                    "max_position_pct": round(cfg.get("max_position_pct") or 100.0, 2),
                    "currency": "USD（模拟）",
                },
                "account": {
                    "cash": round(self.cash, 2),
                    "equity": equity_val,
                    "realized_pnl": round(self.realized_pnl, 2),
                    "unrealized_pnl": round(unrealized_total, 2),
                    "total_pnl": total_pnl,
                    "total_return_pct": round(total_pnl / cfg["starting_balance"] * 100.0, 4)
                    if cfg["starting_balance"] else 0.0,
                    "fees_paid": round(self.fees_paid, 2),
                    "n_trades": self.n_trades,
                    "n_open": len(positions),
                },
                "positions": positions,
                "trades": list(self._trades)[:100],
                "equity": {"ts": list(self._eq_ts), "equity": list(self._eq_val)},
                "started_at": self._started_at,
                "updated_at": time.time(),
            }


def direction_label(direction: str | None) -> str:
    """方向常量 → 中文标签（记录用）。"""
    if direction == DIR_LONG:
        return "多"
    if direction == DIR_SHORT:
        return "空"
    return "观望"


paper_manager = PaperTradingManager()

