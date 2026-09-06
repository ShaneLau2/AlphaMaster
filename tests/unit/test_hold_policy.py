"""持仓管理方案（止损/止盈/移动止损/熔断）纯函数 + 回放引擎单测。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from web.hold_policy import (
    EXIT_REASON_LABEL,
    DEFAULT_HOLD_POLICY,
    check_policy_exit,
    combo_id,
    combo_label,
    combo_parts,
    list_hold_policies,
    params_for,
    sl_tp_levels,
    valid_policy,
)
from web.paper_replay import run_replay

B = SimpleNamespace


def test_valid_policy_falls_back_to_signal() -> None:
    assert valid_policy(None) == "signal"
    assert valid_policy("") == "signal"
    assert valid_policy("RISK") == "risk"
    assert valid_policy("bogus") == DEFAULT_HOLD_POLICY


def test_list_hold_policies_has_seven_presets_with_defaults() -> None:
    rows = list_hold_policies()
    ids = [r["id"] for r in rows]
    assert ids == ["signal", "risk", "hybrid", "be", "time", "chandelier", "dd"]
    by = {r["id"]: r for r in rows}
    assert by["signal"]["default"] is True
    assert by["risk"]["params"]["stop_loss_pct"] == 2.0
    assert by["risk"]["params"]["take_profit_pct"] == 4.0
    assert "trail_activation_pct" in by["risk"]["params"]
    assert by["hybrid"]["params"] == {"stop_loss_pct": 2.0}
    assert by["be"]["params"]["be_activation_pct"] == 1.5
    assert "max_hold_bars" in by["time"]["params"]
    assert "atr_mult" in by["chandelier"]["params"]
    assert "exit_thr_pct" in by["dd"]["params"]
    assert all(r["default"] is False for r in rows if r["id"] != "signal")


def test_sl_tp_levels_long_short() -> None:
    lv = sl_tp_levels("LONG", 100.0, "risk")
    assert lv["stop_price"] == pytest.approx(98.0)
    assert lv["target_price"] == pytest.approx(104.0)
    lv2 = sl_tp_levels("SHORT", 100.0, "risk")
    assert lv2["stop_price"] == pytest.approx(102.0)
    assert lv2["target_price"] == pytest.approx(96.0)
    # signal 方案没有硬出场价位
    lv3 = sl_tp_levels("LONG", 100.0, "signal")
    assert lv3["stop_price"] is None and lv3["target_price"] is None


def test_check_stop_loss_long_hit_below_entry() -> None:
    # 入场 100，risk 方案止损 2% = 98；bar low 97.5 → 触发，成交 98
    fill, reason, peak = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=100.0, bar_high=100.5, bar_low=97.5,
        peak_fav=100.0, policy_id="risk",
    )
    assert fill == pytest.approx(98.0)
    assert reason == "stop"
    assert EXIT_REASON_LABEL[reason] == "止损"


def test_check_take_profit_long_hit_above_entry() -> None:
    fill, reason, _peak = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=100.0, bar_high=105.0, bar_low=99.5,
        peak_fav=100.0, policy_id="risk",
    )
    assert fill == pytest.approx(104.0)
    assert reason == "target"


def test_check_both_hit_same_bar_takes_stop_first() -> None:
    # 一根巨幅 bar 同时扫过止损与止盈：按最保守（止损先成交）
    fill, reason, _ = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=100.0, bar_high=108.0, bar_low=96.0,
        peak_fav=100.0, policy_id="risk",
    )
    assert reason == "stop"
    assert fill == pytest.approx(98.0)


def test_check_trailing_stop_after_activation() -> None:
    # 第一根 bar 冲高到 103.2（>激活 3% 但 < 止盈 4%），低点未跌破移动止损，未出场
    fill, reason, peak = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=100.0, bar_high=103.2, bar_low=102.0,
        peak_fav=100.0, policy_id="risk",
    )
    assert fill is None and reason is None
    assert peak == pytest.approx(103.2)
    # 第二根 bar 从峰值回撤 1.5% → 移动止损 103.2*0.985 ≈ 101.652
    fill2, reason2, _ = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=102.0, bar_high=102.0, bar_low=101.0,
        peak_fav=peak, policy_id="risk",
    )
    assert reason2 == "trail"
    assert fill2 == pytest.approx(101.652, abs=1e-3)


def test_gap_through_stop_fills_worse_side_open() -> None:
    # 跳空低开 96（< 止损 98）：止损单按开盘 96 成交（更差）
    fill, reason, _ = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=96.0, bar_high=96.5, bar_low=95.0,
        peak_fav=100.0, policy_id="risk",
    )
    assert fill == pytest.approx(96.0)
    assert reason == "stop"


def test_hybrid_only_emergency_stop_no_target() -> None:
    fill, reason, _ = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=100.0, bar_high=107.0, bar_low=99.0,
        peak_fav=100.0, policy_id="hybrid",
    )
    # 107 远超 2% 止盈线，但 hybrid 无止盈 → 不触发
    assert fill is None and reason is None
    fill2, reason2, _ = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=100.0, bar_high=100.2, bar_low=97.0,
        peak_fav=100.0, policy_id="hybrid",
    )
    assert reason2 == "stop" and fill2 == pytest.approx(98.0)


# ── 新方案纯函数：be / time / chandelier / dd ──────────────────────────

def test_be_breakeven_stop_after_activation() -> None:
    # 未激活（浮盈 <1.5%）：回撤到 97.5 由兜底止损 97 保护 → 不触发；到 96.5 触发
    f1, r1, p1 = check_policy_exit(
        side="LONG", entry=100.0, bar_open=100.0, bar_high=100.8, bar_low=97.5,
        peak_fav=100.0, policy_id="be",
    )
    assert f1 is None and r1 is None and p1 == pytest.approx(100.8)
    f2, r2, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=99.0, bar_high=99.0, bar_low=96.5,
        peak_fav=p1, policy_id="be",
    )
    assert r2 == "stop" and f2 == pytest.approx(97.0)
    # 浮盈 ≥1.5% → 止损抬到 100*1.0015 ≈ 100.15（保本+缓冲）
    f3, r3, p3 = check_policy_exit(
        side="LONG", entry=100.0, bar_open=100.0, bar_high=102.0, bar_low=101.0,
        peak_fav=100.0, policy_id="be",
    )
    assert r3 is None and p3 == pytest.approx(102.0)
    f4, r4, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=101.0, bar_high=101.0, bar_low=99.9,
        peak_fav=p3, policy_id="be",
    )
    assert r4 == "be" and f4 == pytest.approx(100.15, abs=1e-3)


def test_be_trail_after_activation_locks_more() -> None:
    # 冲高到 104（≥3% 激活移动止损）且已保本；回撤到 103 → 104*0.985≈102.44 触发
    fill, reason, peak = check_policy_exit(
        side="LONG", entry=100.0, bar_open=103.0, bar_high=104.0, bar_low=103.0,
        peak_fav=100.0, policy_id="be",
    )
    assert fill is None and reason is None and peak == pytest.approx(104.0)
    fill2, reason2, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=103.5, bar_high=103.5, bar_low=102.0,
        peak_fav=peak, policy_id="be",
    )
    assert reason2 == "trail" and fill2 == pytest.approx(102.44, abs=1e-3)


def test_time_policy_grace_and_timeout() -> None:
    # 50 根仍只浮盈 0.5%（<1%）→ 时间止损，收盘价成交
    f1, r1, p1 = check_policy_exit(
        side="LONG", entry=100.0, bar_open=100.0, bar_high=100.5, bar_low=99.8,
        peak_fav=100.0, policy_id="time", bar_close=100.2, bars_held=50,
    )
    assert r1 == "time" and f1 == pytest.approx(100.2) and p1 == pytest.approx(100.5)
    # 50 根但已浮盈 2% → 继续持有
    f2, r2, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=100.0, bar_high=102.0, bar_low=99.0,
        peak_fav=100.0, policy_id="time", bar_close=101.0, bars_held=50,
    )
    assert f2 is None and r2 is None
    # 超时 96 根 → 无论盈亏强制离场
    f3, r3, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=100.0, bar_high=103.0, bar_low=99.0,
        peak_fav=100.0, policy_id="time", bar_close=102.5, bars_held=96,
    )
    assert r3 == "timeout" and f3 == pytest.approx(102.5)
    # 上下文缺失（旧调用）→ 不触发
    f4, r4, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=100.0, bar_high=100.2, bar_low=99.8,
        peak_fav=100.0, policy_id="time",
    )
    assert f4 is None and r4 is None


def test_chandelier_atr_stop_overrides_floor() -> None:
    # 峰值 110、ATR=1 → 吊灯 107 > 兜底 95 → 回撤到 106 触发
    fill, reason, peak = check_policy_exit(
        side="LONG", entry=100.0, bar_open=105.0, bar_high=110.0, bar_low=108.0,
        peak_fav=100.0, policy_id="chandelier", atr=1.0,
    )
    assert fill is None and reason is None and peak == pytest.approx(110.0)
    fill2, reason2, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=108.0, bar_high=108.0, bar_low=106.0,
        peak_fav=peak, policy_id="chandelier", atr=1.0,
    )
    assert reason2 == "chandelier" and fill2 == pytest.approx(107.0)
    # 无 ATR（未就绪）→ 只用兜底 95
    fill3, reason3, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=106.0, bar_high=106.0, bar_low=105.0,
        peak_fav=peak, policy_id="chandelier",
    )
    assert fill3 is None and reason3 is None
    fill4, reason4, _ = check_policy_exit(
        side="LONG", entry=100.0, bar_open=96.0, bar_high=96.0, bar_low=93.0,
        peak_fav=100.0, policy_id="chandelier",
    )
    assert reason4 == "stop" and fill4 == pytest.approx(95.0)


def test_dd_ladder_step_transitions() -> None:
    from web.hold_policy import (
        DD_STATE_DEEP,
        DD_STATE_GATED,
        DD_STATE_OK,
        dd_ladder_step,
    )

    p = {"exit_thr_pct": 3.0, "recover_thr_pct": 1.0,
         "deep_thr_pct": 6.0, "deep_recover_thr_pct": 2.5}
    # 正常态：浅回撤不动；触普通档 / 深档 → exit
    assert dd_ladder_step(-0.02, p, DD_STATE_OK) == ("ok", DD_STATE_OK)
    assert dd_ladder_step(-0.04, p, DD_STATE_OK) == ("exit", DD_STATE_GATED)
    assert dd_ladder_step(-0.07, p, DD_STATE_OK) == ("exit", DD_STATE_DEEP)
    # 普通熔断态：未收复 block；收复 ok；加深升级 exit
    assert dd_ladder_step(-0.03, p, DD_STATE_GATED) == ("block", DD_STATE_GATED)
    assert dd_ladder_step(-0.005, p, DD_STATE_GATED) == ("ok", DD_STATE_OK)
    assert dd_ladder_step(-0.08, p, DD_STATE_GATED) == ("exit", DD_STATE_DEEP)
    # 深熔断态：需收复到 ≤2.5%
    assert dd_ladder_step(-0.03, p, DD_STATE_DEEP) == ("block", DD_STATE_DEEP)
    assert dd_ladder_step(-0.02, p, DD_STATE_DEEP) == ("ok", DD_STATE_OK)
    # 正回撤（新高）按 0；None 保守
    assert dd_ladder_step(0.05, p, DD_STATE_OK) == ("ok", DD_STATE_OK)
    assert dd_ladder_step(None, p, DD_STATE_OK) == ("ok", DD_STATE_OK)
    assert dd_ladder_step(None, p, DD_STATE_GATED) == ("block", DD_STATE_GATED)


def test_atr_series_wilder() -> None:
    from web.hold_policy import atr_last, atr_series

    # 恒定振幅 1.0（h-l=1）→ ATR 就绪后恒等于 1
    n = 60
    c = np.linspace(100, 100 + n * 0.1, n)
    h = c + 0.5
    l = c - 0.5
    s = atr_series(h, l, c, period=14)
    assert np.isnan(s[:14]).all()
    assert s[14] == pytest.approx(1.0, abs=1e-9)
    assert atr_last(h, l, c, 14) == pytest.approx(1.0, abs=1e-9)
    assert atr_last(h[:5], l[:5], c[:5], 14) is None


def test_sl_tp_levels_new_policies() -> None:
    lv = sl_tp_levels("LONG", 100.0, "be")
    assert lv["stop_price"] == pytest.approx(97.0)
    assert lv["target_price"] is None
    lv2 = sl_tp_levels("LONG", 100.0, "chandelier")
    assert lv2["stop_price"] == pytest.approx(95.0)
    lv3 = sl_tp_levels("LONG", 100.0, "time")
    assert lv3["stop_price"] is None and lv3["target_price"] is None
    lv4 = sl_tp_levels("LONG", 100.0, "dd")
    assert lv4["stop_price"] is None and lv4["target_price"] is None


# ── 回放引擎（纯数组，不联网） ───────────────────────────────────────────

def _signal_factor(T: int, warm: int, val: float) -> np.ndarray:
    f = np.zeros(T, dtype=np.float64)
    f[warm:] = val
    return f


def test_replay_signal_flip_open_close_accounting() -> None:
    T = 1200
    warm = 800
    base = np.full(T, 100.0)
    close = base.copy()
    close[warm:] = np.linspace(100.0, 110.0, T - warm)  # 单边上涨
    o = close.copy()
    h = close + 0.2
    l = close - 0.2
    f = _signal_factor(T, warm, 1.0)  # 强多头
    rep = run_replay(
        factor=f, open_p=o, high_p=h, low_p=l, close_p=close,
        commission_pct=0.02, slippage_pct=0.01, policy_id="signal",
        start_idx=warm,
    )
    st = rep["stats"]
    # 一路持多到最后强平 → 至少一笔交易，收益为正
    assert st["n_trades"] >= 1
    assert st["total_return"] > 0.0
    assert rep["equity"][-1] == pytest.approx(1.0 + st["total_return"])
    # 开/平各收一次手续费：名义 = 强度 tanh(1) ≈ 0.7616
    import math
    strength = math.tanh(1.0)
    assert st["fees_total"] == pytest.approx(strength * 2 * 0.02 / 100.0, abs=1e-6)


def test_replay_risk_policy_stop_triggers_before_flat() -> None:
    T = 1500
    warm = 800
    # 先上涨到入场 100 上方再多 8% 以上（突破 104 止盈应先出场？为验证止损用下跌段）
    px = np.full(T, 100.0)
    px[warm:warm + 5] = [100.0, 100.2, 100.4, 100.6, 100.8]   # 温和上行至 ~100.8
    px[warm + 5:] = np.linspace(100.8, 88.0, T - warm - 5)    # 之后暴跌
    o = px.copy()
    h = px + 0.3
    l = px - 0.3
    f = _signal_factor(T, warm, 1.0)  # 一直强多
    rep_signal = run_replay(
        factor=f, open_p=o, high_p=h, low_p=l, close_p=px,
        commission_pct=0.02, slippage_pct=0.01, policy_id="signal",
        start_idx=warm,
    )
    rep_risk = run_replay(
        factor=f, open_p=o, high_p=h, low_p=l, close_p=px,
        commission_pct=0.02, slippage_pct=0.01, policy_id="risk",
        start_idx=warm,
    )
    # risk：暴跌应触发止损（-2%），损失远小于 signal 一路扛到 88 的损失
    labels = [t["label"] for t in rep_risk["trades"]]
    assert any("止损" in lb for lb in labels)
    assert rep_risk["stats"]["total_return"] > rep_signal["stats"]["total_return"]
    assert rep_risk["stats"]["total_return"] > -0.05  # 2% 止损 + 手续费附近


# ── 回放：time / chandelier / dd 新方案 ─────────────────────────────────


def _mk_series(closes: list[float], warm: int = 800, amp: float = 0.05):
    pre = np.full(warm, 100.0)
    c = np.concatenate([pre, np.asarray(closes, dtype=float)])
    o = c.copy()
    h = c + amp
    l = c - amp
    f = np.concatenate([np.zeros(warm), np.ones(len(closes))])  # 全程强多
    return o, h, l, c, f


def test_replay_time_policy_exits_after_grace_without_profit() -> None:
    warm = 800
    o, h, l, c, f = _mk_series([100.0] * 120, warm)  # 横盘 0 浮盈
    rep = run_replay(factor=f, open_p=o, high_p=h, low_p=l, close_p=c,
                     commission_pct=0.02, slippage_pct=0.01,
                     policy_id="time", start_idx=warm)
    assert [t["label"] for t in rep["trades"]] == ["时间止损"]
    assert rep["trades"][0]["hold_bars"] == 48  # grace=48 到期不兑现就走


def test_replay_chandelier_exits_on_atr_trail() -> None:
    warm = 800
    cl = [100.0 + i * 0.2 for i in range(30)]          # 爬升至 ~105.8（TR=1）
    cl += [105.8] * 10
    cl += [105.8 - (i + 1) * 1.0 for i in range(10)]   # 阴跌回吐
    o, h, l, c, f = _mk_series(cl, warm, amp=0.5)      # h-l = 1 → ATR→1
    rep = run_replay(factor=f, open_p=o, high_p=h, low_p=l, close_p=c,
                     commission_pct=0.02, slippage_pct=0.01,
                     policy_id="chandelier", start_idx=warm)
    labels = [t["label"] for t in rep["trades"]]
    assert any("吊灯止损" in lb for lb in labels)
    # 吊灯比静态兜底(95)早触发：出场盈利为正（峰值 ~105.8 − 3×ATR=1）
    cut = rep["trades"][0]
    assert cut["pnl"] > 0.0


def test_replay_dd_cuts_on_drawdown_then_reopens_after_recovery() -> None:
    warm = 800
    # 上涨到 105 → 急跌到 99（>3% 回撤熔断）→ 低位徘徊 → 收复到 105 → 高位持有
    cl = ([100.0] * 5
          + [100.0 + (i + 1) * 5 / 15 for i in range(15)]   # →105
          + [105.0 - (i + 1) * 6 / 3 for i in range(3)]     # →99
          + [99.0] * 20                                      # 低位
          + [99.0 + (i + 1) * 6 / 20 for i in range(20)]    # →105 收复
          + [105.0] * 25                                     # 高位
          + [104.0] * 10)
    o, h, l, c, f = _mk_series(cl, warm)
    rep = run_replay(factor=f, open_p=o, high_p=h, low_p=l, close_p=c,
                     commission_pct=0.02, slippage_pct=0.01,
                     policy_id="dd", start_idx=warm)
    labels = [t["label"] for t in rep["trades"]]
    assert any("回撤熔断" in lb for lb in labels)
    cut = [t for t in rep["trades"] if "回撤熔断" in t["label"]][0]
    # 熔断后低位徘徊期不重开（仅一笔熔断离场 + 收复后的重开/期末强平）
    n_trades = len(rep["trades"])
    assert n_trades >= 2          # 熔断出场 + 收复后重开（期末强平）
    # 熔断出场后位置确实空了：cut 之后到收复前无新开仓（收益=0 区间），用 trade bar 序列验证
    bars_after_cut = [t["bar"] for t in rep["trades"] if t["bar"] > cut["bar"]]
    # 收复点（tail bar ~44 后）与熔断点之间 gap > 低位徘徊长度，说明未在低位重开
    assert all(b - cut["bar"] > 15 for b in bars_after_cut)
    assert rep["stats"]["total_return"] > 0.0


# ── 组合策略（A+B 正交叠加）───
def test_combo_parts_and_ids() -> None:
    assert combo_parts("dd+chandelier") == ["dd", "chandelier"]
    assert combo_parts("be + time") == ["be", "time"]
    assert combo_parts("signal+risk") == ["risk"]
    assert combo_parts("dd+dd") == ["dd"]
    assert combo_parts("bogus") == ["signal"]
    assert combo_parts("") == ["signal"]
    assert combo_id("be + time") == "be+time"
    assert combo_id("bogus") == "signal"
    assert "吊灯止损" in combo_label("dd+chandelier")
    assert combo_label("risk") == "止盈止损保护"


def test_combo_exit_union_first_touch_highest_stop() -> None:
    # 多单 "be+chandelier"：be 兜底97/保本100.15，chandelier 兜底95/吊灯100.2，
    # 浮盈 3.2% 后两者都武装移动止损 101.65 → 同一根 bar 先触及它（101.65）成交
    px, reason, _peak = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=102.2, bar_high=103.4, bar_low=100.9,
        peak_fav=103.2, policy_id="be+chandelier",
        atr=1.0,
    )
    assert reason == "trail"
    # 峰值随本根 high 更新到 103.4 → 移动止损 103.4×0.985=101.849，未跳空按该价成交
    assert px == pytest.approx(103.4 * 0.985, abs=0.01)


def test_combo_exit_time_fires_when_no_price_hit() -> None:
    # "risk+time"：无价格触发但 bars_held>=96 → 收盘价超时离场
    px, reason, _ = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=100.0, bar_high=100.4, bar_low=99.7,
        peak_fav=100.3, policy_id="risk+time",
        bar_close=99.8, bars_held=96,
    )
    assert reason == "timeout"
    assert px == pytest.approx(99.8)


def test_combo_exit_stop_beats_time_same_bar() -> None:
    # "hybrid+time"：同一根 bar 触及 2% 止损（98）且超时 → 止损优先
    px, reason, _ = check_policy_exit(
        side="LONG", entry=100.0,
        bar_open=100.0, bar_high=100.2, bar_low=97.9,
        peak_fav=100.2, policy_id="hybrid+time",
        bar_close=99.0, bars_held=120,
    )
    assert reason == "stop"
    assert px == pytest.approx(98.0)


def test_replay_combo_dd_plus_chandelier_dd_gate_active() -> None:
    # 组合 "dd+chandelier"：dd 熔断仍生效（上涨→急跌 >3% → 回撤熔断离场）
    warm = 800
    # 单根暴跌（gap）105→99：dd 阶梯在逐 bar 风控检查之前判档 → 回撤熔断先于吊灯成交
    cl = ([100.0] * 5
          + [100.0 + (i + 1) * 5 / 15 for i in range(15)]   # →105
          + [99.0]                                            # gap 暴跌（-5.7%）
          + [99.0] * 20
          + [99.0 + (i + 1) * 6 / 20 for i in range(20)]    # →105 收复
          + [105.0] * 25)
    o, h, l, c, f = _mk_series(cl, warm)
    rep = run_replay(factor=f, open_p=o, high_p=h, low_p=l, close_p=c,
                     commission_pct=0.02, slippage_pct=0.01,
                     policy_id="dd+chandelier", start_idx=warm)
    labels = [t["label"] for t in rep["trades"]]
    assert any("回撤熔断" in lb for lb in labels)


def test_replay_combo_dd_plus_chandelier_chandelier_exits() -> None:
    # 组合 "dd+chandelier"：小幅回吐（<3%，dd 不触发）时由吊灯离场
    warm = 800
    # 小幅回吐（-2.55% < dd 阈值，dd 不触发）且触及吊灯 → 吊灯离场
    cl = [100.0 + i * 0.2 for i in range(30)]          # →105.8
    cl += [105.8] * 10
    cl += [105.8 - (i + 1) * 0.9 for i in range(3)]    # 回吐至 ~103.1（<3%）
    o, h, l, c, f = _mk_series(cl, warm, amp=0.5)
    rep = run_replay(factor=f, open_p=o, high_p=h, low_p=l, close_p=c,
                     commission_pct=0.02, slippage_pct=0.01,
                     policy_id="dd+chandelier", start_idx=warm)
    labels = [t["label"] for t in rep["trades"]]
    assert any("吊灯止损" in lb for lb in labels)
    assert not any("回撤熔断" in lb for lb in labels)
