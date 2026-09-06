"""web/paper_replay.py — 历史 Parquet 上的离散模拟（与模拟实盘同口径）。

对一根「策略因子序列 + OHLC 行情」做确定性逐 bar 撮合，语义与 web/paper_manager
一致：只在已收盘 bar 的收盘价处翻转成交、按滑点调整、开平各收一次手续费；
若启用持仓管理方案（risk/hybrid），每根新 bar 先检查止损/止盈/移动止损。

作用：
- 回测页选择「持仓管理策略 = 止盈止损/熔断」时，用本引擎替代连续 tanh 引擎，
  得到与模拟实盘同口径的绩效（总收益/Sharpe/Sortino/交易统计/资金曲线）。
- 顺带为「历史回放对比」提供与 live 引擎一致的回放内核。

单位约定：账户以 1.0 起步（现金=1.0，满仓名义=1.0），因此 PnL/收益即账户的
比例收益；费率按设置的 % 计。多空双向，可做空。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from strategy_manager.live_signal import DIR_FLAT, DIR_LONG, DIR_SHORT, min_exposure
from web.hold_policy import (
    DD_STATE_OK,
    EXIT_REASON_LABEL,
    atr_series,
    check_policy_exit,
    combo_id,
    combo_parts,
    dd_ladder_step,
    params_for,
    policy_has_exits,
    sl_tp_levels,
)

WARMUP_BARS = 800  # 与实时信号最小 bar 数一致（特征 warm-up + 滚动归一化）


def run_replay(
    *,
    factor: np.ndarray,           # [T] 每根 bar 的因子值（全量因果特征 + VM 一次算出）
    open_p: np.ndarray,
    high_p: np.ndarray,
    low_p: np.ndarray,
    close_p: np.ndarray,
    commission_pct: float = 0.02,
    slippage_pct: float = 0.01,
    policy_id: str = "signal",
    notional: float = 1.0,
    max_position_pct: float = 100.0,
    start_idx: int = WARMUP_BARS,
    threshold: float | None = None,
    track_dd: bool = False,
    periods_per_year: float = 105195.0,
) -> dict[str, Any]:
    """逐 bar 离散撮合。返回 dict：

    equity: np.ndarray[T]  账户净值（1+累计比例收益，bar 对齐）
    pnl:    np.ndarray[T]  每 bar 净值变化（sharpe/sortino/资金曲线用）
    trades: list[dict]     已平仓记录（pnl/fee/方向/出场原因/hold_bars）
    stats:  total_return / sharpe / sortino / n_trades / win_rate /
            avg_hold_bars / profit_loss_ratio / fees_total
    """
    T = int(close_p.shape[0])
    if T <= 0:
        raise ValueError("回放数据为空")
    thr = min_exposure() if threshold is None else float(threshold)
    pid = combo_id(policy_id)   # 保留 "A+B" 组合（dd 门控 + 其余模块逐 bar 出场）
    parts = combo_parts(pid)
    has_exit = policy_has_exits(pid)
    slip = max(0.0, float(slippage_pct)) / 100.0
    comm = max(0.0, float(commission_pct)) / 100.0
    # chandelier（吊灯 ATR 止损）需要逐 bar ATR；time 需要 bar 序号；dd 走账户级阶梯
    atr_arr = None
    if "chandelier" in parts:
        try:
            atr_arr = atr_series(high_p, low_p, close_p,
                                 int(params_for("chandelier").get("atr_period") or 14))
        except Exception:  # noqa: BLE001 ATR 计算失败回退静态兜底
            atr_arr = None
    dd_state = DD_STATE_OK
    # dd 阶梯用“行情滚动峰值回撤”（净值 = 现金 + 仓位×行情，单品种下与净值回撤同构；
    # 用行情价而非账户净值，现金态下也能自然“收复”后恢复开仓，与实盘引擎同口径）
    # s0 在下方定义，这里先用原始 start_idx 近似（峰值随后逐 bar 精确维护）
    _s0 = max(1, min(int(start_idx), T - 1))
    pk_close = float(np.max(close_p[:_s0])) if T > 0 else 1.0

    def _fill(price: float, side: str) -> float:
        return price * (1.0 + slip) if side == DIR_LONG else price * (1.0 - slip)

    cash = 1.0
    # 每笔投入上限（占当时账户权益 %）：开仓名义 = min(满仓名义, 当时权益 × 上限%)
    # × 强度。默认 100%：cash≤1.0 时即原行为；盈亏后权益偏离 1.0 时
    # 名义仍被「满仓名义(1.0 份)」封顶；上限调低（如 5%）则按权益×5% 复利式缩仓。
    cap_frac = min(200.0, max(1.0, float(max_position_pct or 100.0))) / 100.0
    n_open = 0.0            # 开仓名义金额（占账户比例）
    qty = 0.0
    side: str | None = None
    entry = 0.0
    entry_bar = 0
    peak_fav = 0.0
    cooldown_side: str | None = None
    closed: list[dict[str, Any]] = []
    equity = np.ones(T, dtype=np.float64)
    pnl = np.zeros(T, dtype=np.float64)
    fees_total = 0.0
    dd_events: list[dict] = []

    def _open(direction: str, price: float, strength: float, t: int) -> None:
        nonlocal cash, n_open, qty, side, entry, entry_bar, peak_fav, fees_total
        fill = _fill(price, direction)
        if not (fill > 0.0 and strength > 0.0 and notional > 0.0):
            return
        # 权益复合式：当时权益(=现金，单仓引擎开仓时无在途仓位) × 上限% × 强度
        allowed = min(float(notional), cash * cap_frac)
        n_open = allowed * min(1.0, strength)
        qty = n_open / fill
        side = direction
        entry = fill
        entry_bar = t
        peak_fav = entry
        cash -= n_open * comm
        fees_total += n_open * comm

    def _close(price: float, t: int, label: str) -> None:
        nonlocal cash, n_open, qty, side, fees_total
        if side == DIR_LONG:
            p = (price - entry) * qty
        elif side == DIR_SHORT:
            p = (entry - price) * qty
        else:
            p = 0.0
        fee = abs(qty * entry) * comm
        cash += p - fee
        fees_total += fee
        closed.append({
            "pnl": round(float(p), 10),
            "fee": round(float(fee), 10),
            "fill": round(float(price), 10),
            "bar": int(t),
            "side": side,
            "label": label,
            "hold_bars": int(t - entry_bar),
        })
        n_open = 0.0
        qty = 0.0
        side = None

    s0 = max(1, min(int(start_idx), T - 1))
    direction_prev = DIR_FLAT
    for t in range(s0, T):
        o, h, l, c = (float(open_p[t]), float(high_p[t]), float(low_p[t]), float(close_p[t]))
        if not (c > 0.0 and h >= l and h > 0.0 and o > 0.0):
            equity[t] = equity[t - 1]
            pnl[t] = 0.0
            continue

        exited = False
        dd_gate_action: str | None = None
        # 0) dd（行情回撤熔断）优先：按当前已收盘 bar 相对滚动峰值的回撤判档
        if "dd" in parts:
            pk = pk_close if pk_close > 0.0 else c
            dd_now = (c / pk - 1.0) if pk > 0.0 else 0.0
            dd_prev_state = dd_state
            dd_gate_action, dd_state = dd_ladder_step(
                dd_now, params_for("dd"), dd_state)
            # 只记录状态跃迁（进入熔断/加深/收复），避免熔断期逐 bar 刷事件
            if track_dd and dd_state != dd_prev_state:
                dd_events.append({
                    "bar": int(t),
                    "action": dd_gate_action,
                    "state": int(dd_state),
                    "dd_pct": round(100.0 * float(dd_now), 4),
                    "price": round(float(c), 2),
                })
            if dd_gate_action == "exit":
                # dd 是行情级风控闸：平仓但不设方向冷却（恢复由 dd_state 管），
                # 并重置翻转状态——熔断后要等一次新的方向翻转才重开，
                # 且熔断期内（block）不消耗翻转（避免收复后因方向未变而永远不重开）。
                if n_open > 0:
                    _close(c, t, EXIT_REASON_LABEL.get("dd", "回撤熔断"))
                direction_prev = DIR_FLAT
                exited = True

        # 1) 持仓管理硬出场（先于信号翻转；dd 熔断出场后跳过本根其余动作）
        if not exited and n_open > 0 and has_exit:
            bars_held = t - entry_bar
            atr_now = None
            if atr_arr is not None and t < len(atr_arr):
                v = float(atr_arr[t])
                atr_now = v if np.isfinite(v) else None
            fill_exit, reason, peak = check_policy_exit(
                side=side, entry=entry, bar_open=o, bar_high=h, bar_low=l,
                peak_fav=peak_fav, policy_id=pid,
                bar_close=c, bars_held=bars_held, atr=atr_now,
            )
            peak_fav = peak
            if fill_exit is not None and reason is not None:
                _close(fill_exit, t, EXIT_REASON_LABEL.get(reason, reason))
                cooldown_side = side if side else None
                exited = True

        # 2) 收盘信号翻转（dd 熔断/暂停态下禁止新开仓；block 期间不消费翻转）
        blocked = dd_gate_action in ("exit", "block")
        if not exited and not blocked:
            pos = math.tanh(float(factor[t])) if math.isfinite(float(factor[t])) else 0.0
            direction = DIR_LONG if pos >= thr else (DIR_SHORT if pos <= -thr else DIR_FLAT)
            if direction != direction_prev:
                if direction == DIR_FLAT and n_open > 0:
                    _close(c, t, "信号平仓")
                elif direction in (DIR_LONG, DIR_SHORT) and n_open > 0 and side != direction:
                    _close(c, t, "信号反手平")
                    _open(direction, c, abs(pos), t)
                elif direction in (DIR_LONG, DIR_SHORT) and n_open == 0:
                    if cooldown_side != direction:
                        _open(direction, c, abs(pos), t)
                direction_prev = direction
                if cooldown_side and direction != cooldown_side:
                    cooldown_side = None

        # 3) 按收盘价 mark-to-market
        if n_open > 0 and side:
            unreal = (c - entry) * qty if side == DIR_LONG else (entry - c) * qty
        else:
            unreal = 0.0
        eq = cash + unreal
        equity[t] = eq
        pnl[t] = eq - equity[t - 1]
        if c > pk_close:
            pk_close = c

        # 3) 按收盘价 mark-to-market
        if n_open > 0 and side:
            unreal = (c - entry) * qty if side == DIR_LONG else (entry - c) * qty
        else:
            unreal = 0.0
        eq = cash + unreal
        equity[t] = eq
        pnl[t] = eq - equity[t - 1]

    # 收尾：未平仓按最后一根收盘价强平
    if n_open > 0:
        _close(float(close_p[T - 1]), T - 1, "期末强平")
        equity[T - 1] = cash
        pnl[T - 1] = cash - equity[T - 2] if T - 2 >= 0 else cash - 1.0

    wins = [tr for tr in closed if tr["pnl"] > 0.0]
    losses = [tr for tr in closed if tr["pnl"] < 0.0]
    ppy = max(1.0, float(periods_per_year))
    m = float(pnl[s0:].mean())
    sd = float(pnl[s0:].std())
    sharpe = (m / sd * math.sqrt(ppy)) if sd > 1e-12 else 0.0
    down = pnl[s0:][pnl[s0:] < 0]
    ds = float(down.std()) if len(down) > 0 else 0.0
    ds = max(ds, abs(m), 1e-10)
    sortino = float(np.clip(m / ds * math.sqrt(ppy), -20, 20))
    pl_ratio = None
    if wins and losses:
        pl_ratio = float(np.mean([tr["pnl"] for tr in wins]) /
                         abs(np.mean([tr["pnl"] for tr in losses])))

    # 最大回撤：equity 相对滚动峰值的最大跌幅（负数；如 -0.082 = -8.2%）
    eq_act = np.asarray(equity[max(0, s0):], dtype=float)
    if eq_act.size:
        run_peak = np.maximum.accumulate(eq_act)
        run_peak_safe = np.where(run_peak > 0.0, run_peak, 1.0)
        max_drawdown = float((eq_act / run_peak_safe - 1.0).min())
    else:
        max_drawdown = 0.0

    stats = {
        "total_return": float(equity[-1] - 1.0),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "n_trades": len(closed),
        "win_rate": float(len(wins) / len(closed)) if closed else 0.0,
        "avg_hold_bars": float(np.mean([tr["hold_bars"] for tr in closed])) if closed else 0.0,
        "profit_loss_ratio": round(pl_ratio, 4) if pl_ratio is not None else None,
        "fees_total": round(float(fees_total), 8),
        "max_drawdown": round(max_drawdown, 4),
        "max_position_pct": float(cap_frac * 100.0),
        "signal_threshold": float(thr),
    }
    return {"equity": equity, "pnl": pnl, "trades": closed, "stats": stats,
            "dd_events": dd_events if track_dd else [],
            "policy_id": pid, "commission_pct": commission_pct, "slippage_pct": slippage_pct}
