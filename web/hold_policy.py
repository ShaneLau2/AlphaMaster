"""web/hold_policy.py — 持仓管理（出场风控）方案注册表与纯函数。

与 CLI 实盘 config.py 的 EXIT_MODE 语义一致，参数内置在方案里（不手填）。
引擎（模拟盘 / 历史回放 / 回测）只需调用 check_policy_exit 处理单根已收盘 bar；
UI 用 sl_tp_levels / list_hold_policies 展示方案与参考价位。

内置方案：
- signal   纯信号跟随：只在信号翻转/转弱时平仓（与训练/回测同口径），无止损止盈。
- risk     止盈止损保护：信号跟随 + 固定止损/止盈 + 浮盈达标后移动止损（追踪）。
- hybrid   熔断止损：信号跟随为主，亏损超阈值即强制平仓（紧急熔断），无固定止盈/追踪。
- be       保本追踪（利润保护）：先给固定保护，浮盈达标把止损抬到成本上方「保本」，
           再往上进入移动止损——先保本、后让利润奔跑（deepltrading stop_buffer/trailing 一族）。
- time     时间止损：持仓超时强制离场；「给了时间还没赚到目标」也提前走
           （deepltrading mtime / max-hold 一族）。需要引擎传入 bars_held 与 bar_close。
- chandelier 吊灯止损（ATR 追踪）：止损 = max(静态兜底, 持仓峰值 − k×ATR)，随波动缩放
           （deepltrading trailing/swing_high + ATR/vol 一族）。需要引擎传入 atr。
- dd       回撤熔断（DD 阶梯）：行情自滚动峰值回撤触档 → 平仓并暂停重开，收复才恢复
           （deepltrading「DD 阶梯暴露 / 行情回撤门控」的离散化：满仓/停仓两态 + 非对称
           恢复确认；单品种下行情回撤 = 持仓净值回撤的同构量）。由引擎调用 dd_ladder_step 驱动。
"""
from __future__ import annotations

from typing import Any

import numpy as np

# 方案参数（百分比数值，均为正数表示“幅度”）：
#   stop_loss_pct       固定止损幅度（相对入场价）
#   take_profit_pct     固定止盈幅度
#   trail_activation_pct 浮盈达该幅度后启动移动止损
#   trail_drop_pct       从峰值回撤该幅度触发移动止损
#   be_activation_pct    （be）浮盈达该幅度把止损抬到成本上方
#   be_buffer_pct        （be）保本后的缓冲（覆盖手续费）
#   max_hold_bars        （time）最多持仓 bar 数，超时强制离场
#   grace_bars           （time）宽限 bar 数：超过仍未赚到 min_profit 则提前离场
#   min_profit_pct       （time）宽限期后要求的最低浮盈
#   atr_period / atr_mult（chandelier）ATR 周期与吊灯倍数
#   exit_thr_pct / recover_thr_pct / deep_thr_pct / deep_recover_thr_pct
#                        （dd）净值回撤阶梯：触档平仓 / 收复恢复（普通档与深档分开）
HOLD_POLICIES: dict[str, dict[str, Any]] = {
    "signal": {
        "id": "signal",
        "name": "信号跟随",
        "kind": "signal",
        "desc": "只在信号翻转/转弱时平仓，无固定止盈止损，与训练/回测口径一致（默认）",
        "params": {},
    },
    "risk": {
        "id": "risk",
        "name": "止盈止损保护",
        "kind": "risk",
        "desc": "信号跟随 + 固定止盈(+4%) / 止损(-2%)，浮盈≥+3% 后回撤≥1.5% 触发移动止损",
        "params": {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "trail_activation_pct": 3.0,
            "trail_drop_pct": 1.5,
        },
    },
    "hybrid": {
        "id": "hybrid",
        "name": "熔断止损",
        "kind": "hybrid",
        "desc": "信号跟随为主，亏损 ≥-2% 即强制平仓（紧急熔断），无固定止盈/追踪",
        "params": {"stop_loss_pct": 2.0},
    },
    "be": {
        "id": "be",
        "name": "保本追踪",
        "kind": "protect",
        "desc": "先保本后奔跑：浮盈≥+1.5% 把止损抬到成本上方，浮盈≥+3% 后回撤≥1.5% 移动止损；未保本前保留 -3% 兜底",
        "params": {
            "stop_loss_pct": 3.0,
            "be_activation_pct": 1.5,
            "be_buffer_pct": 0.15,
            "trail_activation_pct": 3.0,
            "trail_drop_pct": 1.5,
        },
    },
    "time": {
        "id": "time",
        "name": "时间止损",
        "kind": "time",
        "desc": "持仓超过 96 根强制离场；超过 48 根仍未浮盈 ≥+1% 提前走——给时间让它工作，到期不兑现就走",
        "params": {
            "max_hold_bars": 96,
            "grace_bars": 48,
            "min_profit_pct": 1.0,
        },
    },
    "chandelier": {
        "id": "chandelier",
        "name": "吊灯止损 (ATR)",
        "kind": "atr",
        "desc": "止损 = max(静态兜底 -5%, 持仓峰值 − 3×ATR14)，随波动自适应上移，不让浮盈回吐过多",
        "params": {
            "stop_loss_pct": 5.0,
            "atr_period": 14,
            "atr_mult": 3.0,
        },
    },
    "dd": {
        "id": "dd",
        "name": "回撤熔断 (DD)",
        "kind": "dd",
        "desc": "行情自滚动峰值回撤 ≥3% 平仓并暂停开仓，收复到 ≤1% 内恢复；回撤 ≥6% 需收复到 ≤2.5%（阶梯式非对称，DD 暴露思路；单品种下与净值回撤同构）",
        "params": {
            "exit_thr_pct": 3.0,
            "recover_thr_pct": 1.0,
            "deep_thr_pct": 6.0,
            "deep_recover_thr_pct": 2.5,
        },
    },
}

DEFAULT_HOLD_POLICY = "signal"
DIR_LONG = "LONG"
DIR_SHORT = "SHORT"

# 出场原因 → 成交日志 reason 前缀
EXIT_REASON_LABEL = {
    "stop": "止损",
    "target": "止盈",
    "trail": "移动止损",
    "be": "保本止损",
    "time": "时间止损",
    "timeout": "持仓超时",
    "chandelier": "吊灯止损",
    "dd": "回撤熔断",
}

# dd 阶梯状态：0=正常（允许开仓） 1=普通档熔断（等待收复） 2=深档熔断（需更深收复）
DD_STATE_OK = 0
DD_STATE_GATED = 1
DD_STATE_DEEP = 2


def valid_policy(policy_id: str | None) -> str:
    """非法/空 policy 一律回落到默认 signal（旧数据兼容）。"""
    pid = (policy_id or "").strip().lower()
    return pid if pid in HOLD_POLICIES else DEFAULT_HOLD_POLICY


def combo_parts(policy_id: str | None) -> list[str]:
    """把策略 id 拆成模块列表（支持 "A+B+C" 组合；signal 与重复项剔除）。

    例如 "dd+chandelier" → ["dd", "chandelier"]；"risk" → ["risk"]；
    非法/空 → ["signal"]。引擎用它在组合里判断 dd 门控是否生效，
    check_policy_exit 用它逐模块求并集出场。
    """
    raw = (policy_id or "").strip().lower()
    if not raw:
        return [DEFAULT_HOLD_POLICY]
    parts = [p.strip() for p in raw.split("+") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p in HOLD_POLICIES and p != "signal" and p not in out:
            out.append(p)
    if not out:
        return [DEFAULT_HOLD_POLICY]
    return out


def combo_id(policy_id: str | None) -> str:
    """规范化组合串："be + time" → "be+time"；空/非法 → "signal"。"""
    parts = combo_parts(policy_id)
    if parts == [DEFAULT_HOLD_POLICY]:
        return DEFAULT_HOLD_POLICY
    return "+".join(parts)


def combo_label(policy_id: str | None) -> str:
    """组合的中文名，如 "dd+chandelier" → "回撤熔断 (DD) + 吊灯止损 (ATR)"。"""
    parts = combo_parts(policy_id)
    if len(parts) <= 1:
        return HOLD_POLICIES[parts[0]]["name"]
    return " + ".join(HOLD_POLICIES[p]["name"] for p in parts)


def get_policy(policy_id: str | None) -> dict[str, Any]:
    return HOLD_POLICIES[valid_policy(policy_id)]


def params_for(policy_id: str | None) -> dict[str, float]:
    """某方案的参数；signal 返回空 dict。"""
    return dict(HOLD_POLICIES[valid_policy(policy_id)].get("params") or {})


def list_hold_policies() -> list[dict[str, Any]]:
    """公开给 UI 的方案列表（含默认值与中文名）。"""
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "kind": p["kind"],
            "desc": p["desc"],
            "params": p["params"],
            "default": p["id"] == DEFAULT_HOLD_POLICY,
        }
        for p in HOLD_POLICIES.values()
    ]


def policy_has_exits(policy_id: str | None) -> bool:
    """该方案/组合是否含止损/止盈类硬出场（signal 不含；dd 含账户级出场）。"""
    parts = combo_parts(policy_id)
    return any(p != "signal" for p in parts)


def combo_params(policy_id: str | None) -> dict[str, float]:
    """组合参数合并（同名键后者覆盖）；signal/空返回空 dict。

    UI 参考价位、日志用；引擎的 dd 阶梯固定用 dd 模块自身参数。
    """
    merged: dict[str, float] = {}
    for p in combo_parts(policy_id):
        merged.update(HOLD_POLICIES[p].get("params") or {})
    return merged


def sl_tp_levels(side: str, entry: float, policy_id: str | None) -> dict[str, float | None]:
    """按入场价 + 方案参数算参考 止损/止盈 价位（UI 画线用；signal 全 None）。

    动态出场（be 保本抬价 / chandelier ATR / time / dd）给的是“静态兜底参考价”，
    实际成交价随行情上移——方案 desc 中已说明。
    """
    params = combo_params(policy_id)
    entry = float(entry or 0.0)
    if entry <= 0.0 or not params:
        return {"stop_price": None, "target_price": None}
    stop = target = None
    if params.get("stop_loss_pct"):
        sl = float(params["stop_loss_pct"]) / 100.0
        stop = entry * (1.0 - sl) if side == DIR_LONG else entry * (1.0 + sl)
    if params.get("take_profit_pct"):
        tp = float(params["take_profit_pct"]) / 100.0
        target = entry * (1.0 + tp) if side == DIR_LONG else entry * (1.0 - tp)
    return {
        "stop_price": round(stop, 10) if stop is not None else None,
        "target_price": round(target, 10) if target is not None else None,
    }


def _side_move(side: str, entry: float, mark: float) -> float:
    """价格相对入场的幅度（正=浮盈，负=浮亏），多空对称。"""
    if side == DIR_LONG:
        return (mark - entry) / entry
    return (entry - mark) / entry


# ── ATR（吊灯止损用，Wilder 平滑，与两引擎共用，保证回放/实盘口径一致）───
def atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """Wilder ATR 序列（长度同输入；前 period 根为 NaN）。"""
    period = max(2, int(period))
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = min(h.shape[0], l.shape[0], c.shape[0])
    if n < 2:
        return np.full(n, np.nan)
    h, l, c = h[:n], l[:n], c[:n]
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    out = np.full(n, np.nan)
    if n <= period:
        return out
    first = float(np.nanmean(tr[1: period + 1]))
    out[period] = first
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def atr_last(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> float | None:
    """序列最后一根已就绪 ATR（未就绪返回 None，调用方回退静态兜底）。"""
    s = atr_series(high, low, close, period)
    if s.shape[0] == 0:
        return None
    v = float(s[-1])
    return v if np.isfinite(v) else None


# ── 净值回撤熔断（dd）───
def dd_ladder_step(dd_value: float | None, params: dict[str, Any],
                   state: int) -> tuple[str, int]:
    """dd 阶梯一步判定（纯函数，两引擎共用，保证回放/实盘同语义）。

    dd_value：账户净值相对滚动峰值的回撤，负数（如 -0.04 = 回撤 4%）；None=无数据。
    state：DD_STATE_OK / GATED / DEEP（上一根已收盘 bar 后的状态）。
    返回 (action, next_state)，action ∈ {"ok","block","exit"}：
      - 正常态下回撤触档（≤ -exit_thr / ≤ -deep_thr）→ "exit"（引擎平掉在途持仓）并进入对应熔断态；
      - 熔断态下回撤收复（≥ -recover / ≥ -deep_recover）→ "ok" 放行（可开仓）；
      - 熔断态未收复 → "block"（暂停开仓）。
    回撤为正（新高）按 0 处理；None 保守视为 0（不误杀），但熔断态下 None 不恢复。
    """
    st = int(state) if int(state) in (DD_STATE_OK, DD_STATE_GATED, DD_STATE_DEEP) else DD_STATE_OK
    if dd_value is None:
        return ("block" if st != DD_STATE_OK else "ok", st)
    dd = min(0.0, float(dd_value))
    exit_thr = -abs(float(params.get("exit_thr_pct") or 0.0)) / 100.0
    recover_thr = -abs(float(params.get("recover_thr_pct") or 0.0)) / 100.0
    deep_thr = -abs(float(params.get("deep_thr_pct") or 0.0)) / 100.0
    deep_recover = -abs(float(params.get("deep_recover_thr_pct") or 0.0)) / 100.0

    if st == DD_STATE_OK:
        if dd <= deep_thr:
            return "exit", DD_STATE_DEEP
        if dd <= exit_thr:
            return "exit", DD_STATE_GATED
        return "ok", DD_STATE_OK
    if st == DD_STATE_GATED:
        if dd <= deep_thr:  # 熔断期内进一步加深 → 升级
            return "exit", DD_STATE_DEEP
        if dd >= recover_thr:
            return "ok", DD_STATE_OK
        return "block", DD_STATE_GATED
    # DEEP
    if dd >= deep_recover:
        return "ok", DD_STATE_OK
    return "block", DD_STATE_DEEP


def _dd_gate_dd_value(equity_now: float, equity_peak: float | None) -> float:
    """把 净值/峰值 折成回撤值（负数；峰值未初始化按现价=0 回撤）。"""
    eq = float(equity_now)
    pk = float(equity_peak) if equity_peak is not None else eq
    if pk <= 0.0 or eq <= 0.0:
        return 0.0
    return eq / pk - 1.0


def check_policy_exit(
    *,
    side: str,
    entry: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    peak_fav: float | None,
    policy_id: str | None,
    bar_close: float | None = None,
    bars_held: int | None = None,
    atr: float | None = None,
) -> tuple[float | None, str | None, float]:
    """对一根已收盘 bar 检查持仓管理方案的硬出场。

    返回 (成交价, 原因key, 新的峰值)。未触发时成交价=None，原因=None。
    peak_fav：多单=持仓以来最高价，空单=持仓以来最低价（初始可传 entry）。

    约定：
    - 只在「已收盘」bar 上检查（与信号口径一致）；用 high/low 判断是否触及。
    - risk/hybrid：同一根 bar 同时触及止损与止盈时按最保守的止损先成交；
      跳空穿过价位时按 bar 开盘价成交（止损滑向更差一侧、止盈滑向更好一侧）。
    - be / chandelier：只有止损类价位，同一根 bar 内按“更贴近现价者先触发”（多单取
      更高的止损价成交——价格下跌先过它）；跳空同样按开盘价成交。
    - time 类出场按收盘价成交（收盘才做决定），bar_close 缺省时跳过 time 逻辑。
    - bars_held / atr 是 time / chandelier 的上下文，由引擎传入。
    - dd 方案不做逐 bar 硬出场，返回空（由引擎用 dd_ladder_step 驱动账户级回撤）。
    """
    entry = float(entry)
    if entry <= 0.0:
        return None, None, float(peak_fav if peak_fav is not None else entry)

    # ── 组合策略（"A+B"）：逐模块求并集，同一根 bar 先触及者成交 ──
    parts = combo_parts(policy_id)
    if len(parts) > 1:
        return _check_combo_exit(
            parts=parts, side=side, entry=entry,
            bar_open=bar_open, bar_high=bar_high, bar_low=bar_low,
            peak_fav=peak_fav, bar_close=bar_close,
            bars_held=bars_held, atr=atr,
        )

    params = params_for(policy_id)
    if not params:
        return None, None, float(peak_fav if peak_fav is not None else entry)

    is_long = side == DIR_LONG
    # 更新峰值（多单看 high、空单看 low）
    if is_long:
        peak = max(float(peak_fav if peak_fav is not None else entry), float(bar_high))
    else:
        peak = min(float(peak_fav if peak_fav is not None else entry), float(bar_low))

    pid = valid_policy(policy_id)
    stop_pct = float(params.get("stop_loss_pct") or 0.0) / 100.0
    tp_pct = float(params.get("take_profit_pct") or 0.0) / 100.0
    act_pct = float(params.get("trail_activation_pct") or 0.0) / 100.0
    drop_pct = float(params.get("trail_drop_pct") or 0.0) / 100.0

    # 止损/止盈/吊灯价位计算
    stop_price = None
    if stop_pct:
        stop_price = entry * (1.0 - stop_pct) if is_long else entry * (1.0 + stop_pct)
    target_price = None
    if tp_pct:
        target_price = entry * (1.0 + tp_pct) if is_long else entry * (1.0 - tp_pct)

    hits: list[tuple[float, str]] = []  # (成交价, 原因)

    if pid in ("be", "chandelier"):
        # 只含止损类价位：be（兜底→保本→移动止损抬升）与 chandelier（兜底 vs 吊灯）
        # 多单取“最高（最贴近）”，空单取“最低”。同根巨幅 bar 先过哪个价就以哪个价成交。
        cand: list[tuple[float, str]] = []
        if stop_price is not None:
            cand.append((stop_price, "stop"))
        if pid == "be":
            be_act = float(params.get("be_activation_pct") or 0.0) / 100.0
            be_buf = float(params.get("be_buffer_pct") or 0.0) / 100.0
            if be_act and _side_move(side, entry, peak) >= be_act:
                be_stop = entry * (1.0 + be_buf) if is_long else entry * (1.0 - be_buf)
                cand.append((be_stop, "be"))
            if act_pct and drop_pct and _side_move(side, entry, peak) >= act_pct:
                cand.append((peak * (1.0 - drop_pct) if is_long else peak * (1.0 + drop_pct),
                             "trail"))
        else:  # chandelier
            mult = float(params.get("atr_mult") or 0.0)
            atr_v = None
            if atr is not None:
                try:
                    atr_v = float(atr)
                except (TypeError, ValueError):
                    atr_v = None
            if mult and atr_v is not None and np.isfinite(atr_v) and atr_v > 0.0 and peak:
                chand = peak - mult * atr_v if is_long else peak + mult * atr_v
                cand.append((chand, "chandelier"))
        if not cand:
            return None, None, peak
        eff_price, eff_reason = (max(cand, key=lambda x: x[0]) if is_long
                                 else min(cand, key=lambda x: x[0]))
        if is_long and bar_low <= eff_price:
            hits.append((min(eff_price, bar_open), eff_reason))
        elif not is_long and bar_high >= eff_price:
            hits.append((max(eff_price, bar_open), eff_reason))
    elif pid in ("risk", "hybrid"):
        # 保持既有保守语义：同一根 bar 巨幅波动按 止损 → 止盈 → 移动止损 排序成交
        if stop_price is not None:
            if is_long and bar_low <= stop_price:
                hits.append((min(stop_price, bar_open), "stop"))
            elif not is_long and bar_high >= stop_price:
                hits.append((max(stop_price, bar_open), "stop"))
        if target_price is not None and not hits:
            if is_long and bar_high >= target_price:
                hits.append((max(target_price, bar_open), "target"))
            elif not is_long and bar_low <= target_price:
                hits.append((min(target_price, bar_open), "target"))
        if not hits and act_pct and drop_pct and _side_move(side, entry, peak) >= act_pct:
            trail_price = peak * (1.0 - drop_pct) if is_long else peak * (1.0 + drop_pct)
            if is_long and bar_low <= trail_price:
                hits.append((min(trail_price, bar_open), "trail"))
            elif not is_long and bar_high >= trail_price:
                hits.append((max(trail_price, bar_open), "trail"))
    elif pid == "time":
        # 收盘价成交的时间类出场
        if bar_close is not None and bars_held is not None:
            close_v = float(bar_close)
            if close_v > 0.0:
                max_hold = int(params.get("max_hold_bars") or 0)
                grace = int(params.get("grace_bars") or 0)
                min_profit = float(params.get("min_profit_pct") or 0.0) / 100.0
                if max_hold > 0 and bars_held >= max_hold:
                    hits.append((close_v, "timeout"))
                elif grace > 0 and bars_held >= grace and _side_move(side, entry, peak) < min_profit:
                    hits.append((close_v, "time"))

    if not hits:
        return None, None, peak
    price, reason = hits[0]
    return round(float(price), 10), reason, peak


def _check_combo_exit(
    *,
    parts: list[str],
    side: str,
    entry: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    peak_fav: float | None,
    bar_close: float | None,
    bars_held: int | None,
    atr: float | None,
) -> tuple[float | None, str | None, float]:
    """组合策略逐模块求并集出场。

    - 每个模块独立调用 check_policy_exit（单 id 路径），收集其触发；
    - 价格类触发（stop/target/trail/be/chandelier）：同一根 bar 内多单取
      “最高触发价”先成交、空单取“最低触发价”（价格先过谁就按谁成交，与
      be/chandelier 既有语义一致）；跳空按开盘价；
    - 收盘类触发（time/timeout）：无价格触发时才生效，按 超时→时间止损 优先；
    - dd 模块逐 bar 无触发（由引擎账户级驱动），这里自动忽略。
    """
    price_hits: list[tuple[float, str]] = []
    time_hits: list[tuple[float, str]] = []
    peak_out = float(peak_fav if peak_fav is not None else entry)
    for p in parts:
        price, reason, peak = check_policy_exit(
            side=side, entry=entry,
            bar_open=bar_open, bar_high=bar_high, bar_low=bar_low,
            peak_fav=peak_fav, policy_id=p,
            bar_close=bar_close, bars_held=bars_held, atr=atr,
        )
        # 峰值统一取各模块中更极端的（多单更高、空单更低）
        if side == DIR_LONG:
            peak_out = max(peak_out, float(peak))
        else:
            peak_out = min(peak_out, float(peak))
        if price is None or reason is None:
            continue
        if reason in ("time", "timeout"):
            time_hits.append((float(price), reason))
        else:
            price_hits.append((float(price), reason))

    if price_hits:
        if side == DIR_LONG:
            price, reason = max(price_hits, key=lambda x: x[0])
        else:
            price, reason = min(price_hits, key=lambda x: x[0])
        return round(float(price), 10), reason, peak_out
    if time_hits:
        prio = {"timeout": 0, "time": 1}
        price, reason = min(time_hits, key=lambda x: (prio.get(x[1], 9), x[0]))
        return round(float(price), 10), reason, peak_out
    return None, None, peak_out
