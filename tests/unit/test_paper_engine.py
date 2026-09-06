"""模拟实盘（纸上交易）引擎单测：结算函数 + 对账/记账/持久化。"""
from __future__ import annotations

import json

import pytest
from pytest import MonkeyPatch

from web.paper_manager import (
    DIR_FLAT,
    DIR_LONG,
    DIR_SHORT,
    PaperTradingManager,
    PaperWatch,
    clamp_strength,
    close_position,
    fee_for_notional,
    fill_price,
    open_position,
    unrealized_pnl,
)


# ── 纯结算函数 ──────────────────────────────────────────────────────

def test_fill_price_direction_slippage() -> None:
    assert fill_price(100.0, DIR_LONG, 0.01) == pytest.approx(100.01)
    assert fill_price(100.0, DIR_SHORT, 0.01) == pytest.approx(99.99)
    assert fill_price(100.0, DIR_LONG, 0.0) == pytest.approx(100.0)


def test_open_position_sizes_by_strength_and_fees() -> None:
    # 满仓名义 10000 × 强度 0.5 = 名义 5000
    r = open_position(DIR_LONG, 100.0, 0.5, 10000.0, commission_pct=0.02, slippage_pct=0.01)
    assert r["fill"] == pytest.approx(100.01)
    assert r["notional_value"] == pytest.approx(5000.0)
    assert r["qty"] == pytest.approx(5000.0 / 100.01)
    assert r["fee"] == pytest.approx(1.0)  # 5000 * 0.02%


def test_open_position_strength_zero_or_bad_price_returns_zero() -> None:
    assert open_position(DIR_LONG, 100.0, 0.0, 10000.0, 0.02, 0.01)["qty"] == 0.0
    assert open_position(DIR_LONG, 0.0, 0.5, 10000.0, 0.02, 0.01)["qty"] == 0.0
    assert open_position(DIR_SHORT, 100.0, 0.8, 0.0, 0.02, 0.01)["qty"] == 0.0


def test_close_position_long_pnl_and_fee() -> None:
    pos = {
        "side": DIR_LONG,
        "qty": 50.0,
        "entry_price": 100.0,  # 开仓已含滑点，此处直接对比
        "notional_value": 5000.0,
    }
    flow = close_position(pos, 110.0, commission_pct=0.02, slippage_pct=0.01)
    # 平仓卖价 = 110 * (1-0.01%) = 109.989；多单盈亏 = (卖出-买入)*数量
    assert flow["fill"] == pytest.approx(109.989)
    assert flow["pnl"] == pytest.approx((109.989 - 100.0) * 50.0)
    assert flow["fee"] == pytest.approx(1.0)  # 名义 5000 的单边手续费


def test_close_position_short_pnl() -> None:
    pos = {"side": DIR_SHORT, "qty": 10.0, "entry_price": 100.0, "notional_value": 1000.0}
    flow = close_position(pos, 90.0, commission_pct=0.0, slippage_pct=0.0)
    assert flow["pnl"] == pytest.approx((100.0 - 90.0) * 10.0)
    # 空单平仓买价 = 90 * (1+滑点) 由 slippage=0 保证


def test_unrealized_pnl_both_sides() -> None:
    long_pos = {"side": DIR_LONG, "qty": 2.0, "entry_price": 100.0, "notional_value": 0.0}
    assert unrealized_pnl(long_pos, 105.0) == pytest.approx(10.0)
    short_pos = {"side": DIR_SHORT, "qty": 2.0, "entry_price": 100.0, "notional_value": 0.0}
    assert unrealized_pnl(short_pos, 90.0) == pytest.approx(20.0)


def test_fee_and_strength_clamp() -> None:
    assert fee_for_notional(10_000.0, 0.02) == pytest.approx(2.0)
    assert clamp_strength(None) == 0.0
    assert clamp_strength(1.5) == 1.0
    assert clamp_strength(-0.2) == 0.0


# ── 对账/记账（直接驱动 manager，不启动引擎线程） ──────────────────────


def _manager(tmp_path) -> PaperTradingManager:
    return PaperTradingManager(
        state_file=tmp_path / "paper_sim_state.json",
        starting_balance=100_000.0,
        commission_pct=0.02,
        slippage_pct=0.01,
        default_notional=10_000.0,
    )


def _watch(mgr: PaperTradingManager, symbol: str = "BTCUSDT") -> PaperWatch:
    """在 tmp 目录写一个真实策略 JSON，并注册为监控项（便于持久化恢复测试）。"""
    strat_path = mgr.state_file.parent / f"best_{symbol}.json"
    if not strat_path.exists():
        strat_path.write_text(
            json.dumps({"vocab_version": None, "symbol": symbol, "formula": [1, 2, 3], "best_score": 1.0}),
            encoding="utf-8",
        )
    w = PaperWatch(
        id=f"binance:{symbol}:1h:best_{symbol}",
        source="binance",
        symbol=symbol,
        timeframe="1h",
        strategy_file=str(strat_path),
        strategy_name=f"best_{symbol}",
        formula=[1, 2, 3],
        vocab_version=None,
        strategy_symbol=symbol,
        strategy_timeframe="1h",
        best_score=1.0,
        cadence_s=60,
        notional=10_000.0,
    )
    mgr._watches[w.id] = w
    return w


def test_reconcile_open_and_close_flow(tmp_path) -> None:
    m = _manager(tmp_path)
    w = _watch(m)
    w.last_close = 100.0

    m._reconcile(w, DIR_LONG, 0.5, 100.0, bar_ts=1000)
    st = m.status()
    assert st["account"]["n_open"] == 1
    assert st["account"]["n_trades"] == 0
    pos = st["positions"][0]
    assert pos["side"] == DIR_LONG
    assert pos["entry_price"] == pytest.approx(100.01)  # 买价含滑点
    # 手续费 = 名义 5000 × 0.02% = 1
    assert st["account"]["fees_paid"] == pytest.approx(1.0)
    assert st["account"]["cash"] == pytest.approx(99_999.0)
    assert m._trades[0]["action"] == "开多"

    # 同方向：保持不动
    m._reconcile(w, DIR_LONG, 0.7, 105.0, bar_ts=1001)
    assert m.status()["account"]["n_open"] == 1

    # 转观望：平仓，实现盈利
    w.last_close = 110.0
    m._reconcile(w, DIR_FLAT, None, 110.0, bar_ts=1002)
    st = m.status()
    assert st["account"]["n_open"] == 0
    assert st["account"]["n_trades"] == 1
    assert st["account"]["realized_pnl"] > 0
    assert st["account"]["cash"] == pytest.approx(100_000.0 + st["account"]["realized_pnl"] - 2.0)
    assert m._trades[0]["action"] == "平多"


def test_reconcile_reversal_closes_then_opens(tmp_path) -> None:
    m = _manager(tmp_path)
    w = _watch(m)
    m._reconcile(w, DIR_LONG, 0.5, 100.0, bar_ts=1000)
    w.last_close = 95.0
    m._reconcile(w, DIR_SHORT, 0.6, 95.0, bar_ts=1001)
    st = m.status()
    # 反手 = 平多 + 开空 两笔（列表最新在前，含更早的开多）
    actions = [t["action"] for t in m._trades]
    assert actions == ["开空", "平多", "开多"]
    assert st["positions"][0]["side"] == DIR_SHORT
    assert st["account"]["n_open"] == 1
    assert st["account"]["n_trades"] == 1


def test_manual_close_and_remove_watch(tmp_path) -> None:
    m = _manager(tmp_path)
    w = _watch(m)
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 0.5, 100.0, bar_ts=1000)
    w.last_close = 102.0
    assert m.close_position(w.id) is True
    assert m.status()["account"]["n_open"] == 0

    # 移除监控时若仍持仓会自动平仓
    m._reconcile(w, DIR_SHORT, 0.5, 98.0, bar_ts=1001)
    assert m.remove_watch(w.id) is True
    assert w.id not in m._watches
    assert m.status()["account"]["n_open"] == 0


def test_persist_round_trip_restores_cash_and_positions(
    tmp_path, monkeypatch: MonkeyPatch
) -> None:
    m1 = _manager(tmp_path)
    w = _watch(m1, symbol="ETHUSDT")
    w.last_close = 2000.0
    m1._reconcile(w, DIR_LONG, 1.0, 2000.0, bar_ts=100)
    assert m1.status()["account"]["n_open"] == 1

    m2 = PaperTradingManager(
        state_file=tmp_path / "paper_sim_state.json",
        starting_balance=100_000.0,
        commission_pct=0.02,
        slippage_pct=0.01,
        default_notional=10_000.0,
    )
    # 恢复后不真正启动引擎线程（单测不联网）
    monkeypatch.setattr(m2, "start", lambda: None)
    m2.load_persisted()
    st = m2.status()
    assert st["account"]["cash"] == pytest.approx(m1.cash)
    assert st["account"]["fees_paid"] == pytest.approx(m1.fees_paid)
    assert len(st["positions"]) == 1
    assert st["positions"][0]["symbol"] == "ETHUSDT"
    # 持仓对应监控项存在
    assert st["count"] == 1


def test_reset_clears_account_keeps_watches(tmp_path) -> None:
    m = _manager(tmp_path)
    w = _watch(m)
    m._reconcile(w, DIR_LONG, 0.8, 100.0, bar_ts=100)
    assert m.status()["account"]["n_open"] == 1
    m.reset(starting_balance=50_000.0)
    st = m.status()
    assert st["account"]["equity"] == pytest.approx(50_000.0)
    assert st["account"]["n_open"] == 0
    assert st["count"] == 1  # 监控项保留
    assert len(st["trades"]) == 0
    assert len(st["equity"]["ts"]) == 1  # 记录了一个重置点


def test_equity_marks_open_position_at_last_close(tmp_path) -> None:
    m = _manager(tmp_path)
    w = _watch(m)
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 0.5, 100.0, bar_ts=1000)
    equity0 = m.equity()
    # 市价上涨 → 净值应上升（未实现盈利 > 0）
    w.last_close = 110.0
    assert m.equity() > equity0
    st = m.status()
    assert st["account"]["unrealized_pnl"] > 0


# ── 持仓管理方案（止损/止盈/移动止损 + 冷却重开） ──────────────────────


def test_policy_stop_exit_cooldown_then_reopen_after_flip(tmp_path) -> None:
    """risk 方案：新 bar 触发止损 → 平仓记原因 → 同方向冷却；信号翻转后可再开。"""
    from types import SimpleNamespace

    m = _manager(tmp_path)
    w = _watch(m)
    w.policy_id = "risk"
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)  # 开多（入场 ≈100.01）
    st = m.status()
    pos0 = st["positions"][0]
    assert pos0["stop_price"] is not None  # risk 方案开仓即带止损价
    assert pos0["policy_id"] == "risk"

    # 新 bar 低点跌破止损 98 → 触发，平仓原因含“止损”
    bar = SimpleNamespace(open=100.0, high=100.5, low=97.0)
    assert m._policy_exit_check(w, bar) is True
    st = m.status()
    assert st["account"]["n_open"] == 0
    assert m._trades[0]["action"] == "平多"
    assert "止损" in m._trades[0]["reason"]
    assert w.cooldown_side == DIR_LONG

    # 信号仍多 → 冷却中不开仓
    w.last_close = 101.0
    m._reconcile(w, DIR_LONG, 0.9, 101.0, bar_ts=1001)
    assert m.status()["account"]["n_open"] == 0

    # 信号转观望 → 冷却解除
    m._reconcile(w, DIR_FLAT, None, 100.5, bar_ts=1002)
    assert w.cooldown_side is None

    # 信号再转多 → 重新开仓
    m._reconcile(w, DIR_LONG, 0.8, 102.0, bar_ts=1003)
    assert m.status()["account"]["n_open"] == 1


def test_policy_target_exit_and_signal_policy_no_exit(tmp_path) -> None:
    from types import SimpleNamespace

    m = _manager(tmp_path)
    w = _watch(m)
    w.policy_id = "risk"
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)
    # 大阳线触止盈（104），未触止损
    bar = SimpleNamespace(open=100.0, high=105.0, low=99.5)
    assert m._policy_exit_check(w, bar) is True
    assert "止盈" in m._trades[0]["reason"]

    # signal 方案无硬出场
    m2 = _manager(tmp_path)
    w2 = _watch(m2)
    w2.policy_id = "signal"
    m2._reconcile(w2, DIR_LONG, 1.0, 100.0, bar_ts=2000)
    bar2 = SimpleNamespace(open=100.0, high=105.0, low=97.0)
    assert m2._policy_exit_check(w2, bar2) is False  # 不触发，交给信号翻转
    assert m2.status()["account"]["n_open"] == 1


# ── 新方案：be / time / chandelier / dd（管理引擎层） ────────────────────


def test_manager_be_policy_breakeven_after_activation(tmp_path) -> None:
    from types import SimpleNamespace

    m = _manager(tmp_path)
    w = _watch(m)
    w.policy_id = "be"
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)  # 开多，入场≈100.01
    pos = m._positions[w.id]
    assert pos["bars_held"] == 0
    # 冲高 102（浮盈≈1.99% ≥1.5% 激活保本）未跌破 → 不出场
    bar1 = SimpleNamespace(open=101.0, high=102.0, low=100.5)
    assert m._policy_exit_check(w, bar1) is False
    # 回撤到 99.9 < 保本价 ≈100.16 → 触发保本止损
    bar2 = SimpleNamespace(open=101.0, high=101.0, low=99.9)
    assert m._policy_exit_check(w, bar2) is True
    assert "保本" in m._trades[0]["reason"]
    assert m.status()["account"]["n_open"] == 0


def test_manager_time_policy_exits(tmp_path) -> None:
    from types import SimpleNamespace

    m = _manager(tmp_path)
    w = _watch(m)
    w.policy_id = "time"
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)
    pos = m._positions[w.id]
    pos["bars_held"] = 60  # 超过宽限 48 根仍未盈利
    w.last_close = 100.2
    bar = SimpleNamespace(open=100.0, high=100.3, low=99.8)
    assert m._policy_exit_check(w, bar) is True
    assert "时间止损" in m._trades[0]["reason"]
    assert m.status()["account"]["n_open"] == 0

    # 超时（≥96 根）即使盈利也离场
    m2 = _manager(tmp_path)
    w2 = _watch(m2)
    w2.policy_id = "time"
    w2.last_close = 100.0
    m2._reconcile(w2, DIR_LONG, 1.0, 100.0, bar_ts=2000)
    pos2 = m2._positions[w2.id]
    pos2["bars_held"] = 100
    w2.last_close = 105.0
    bar2 = SimpleNamespace(open=103.0, high=106.0, low=102.0)
    assert m2._policy_exit_check(w2, bar2) is True
    assert "超时" in m2._trades[0]["reason"]


def test_manager_chandelier_policy_uses_atr(tmp_path) -> None:
    from types import SimpleNamespace

    m = _manager(tmp_path)
    w = _watch(m)
    w.policy_id = "chandelier"
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)
    # 构造 16 根 h-l=1 的历史 bar → ATR14 就绪 = 1.0
    bars = [SimpleNamespace(high=100.0 + k * 0.1 + 0.5, low=100.0 + k * 0.1 - 0.5,
                            close=100.0 + k * 0.1) for k in range(16)]
    # 冲高 110（峰值 110），低点 108 未破吊灯（110−3×1=107）
    bar1 = SimpleNamespace(open=105.0, high=110.0, low=108.0)
    assert m._policy_exit_check(w, bar1, bars) is False
    # 回撤到 106 < 107 → 触发吊灯止损
    bar2 = SimpleNamespace(open=107.0, high=107.5, low=106.0)
    assert m._policy_exit_check(w, bar2, bars) is True
    assert "吊灯" in m._trades[0]["reason"]


def test_manager_dd_gate_cuts_blocks_then_reopens_after_recovery(tmp_path) -> None:
    m = _manager(tmp_path)
    w = _watch(m)
    w.policy_id = "dd"
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)  # 开多
    # 行情冲到 105 → 峰值 105，无回撤
    w.last_close = 105.0
    act, ev = m._dd_gate_step(w)
    assert act == "ok" and ev is None
    # 急跌到 99（−5.7%）→ 触档（熔断事件应单独产出，供飞书告警）
    w.last_close = 99.0
    act, ev = m._dd_gate_step(w)
    assert act == "exit" and ev is not None and ev["event"] == "熔断"
    assert w.dd_gate != 0
    # 引擎动作：平掉仓位并进入冷却（模拟 _process_watch 的 exit 分支）
    m._close_position_internal(w.id, w.last_close, reason="回撤熔断（持仓管理）")
    w.cooldown_side = DIR_LONG
    assert m.status()["account"]["n_open"] == 0
    # 低位信号仍强多 → dd 闸 block + 冷却 → 不开仓
    m._reconcile(w, DIR_LONG, 1.0, 99.0, bar_ts=1001)
    assert m.status()["account"]["n_open"] == 0
    # 行情收复到 104.5（−0.5% ≥ 收复线 −1%）→ 闸开（收复事件单独产出）
    w.last_close = 104.5
    act, ev = m._dd_gate_step(w)
    assert act == "ok" and ev is not None and ev["event"] == "收复"
    assert w.dd_gate == 0
    # 信号先转观望清冷却，再转多 → 重开
    m._reconcile(w, DIR_FLAT, None, 104.0, bar_ts=1002)
    m._reconcile(w, DIR_LONG, 1.0, 104.5, bar_ts=1003)
    assert m.status()["account"]["n_open"] == 1


def test_manager_dd_process_watch_exit_reason(tmp_path, monkeypatch: MonkeyPatch) -> None:
    """端到端（打桩行情源）：新 bar 触发回撤熔断 → 平仓原因含“回撤熔断”。"""
    from types import SimpleNamespace

    import web.paper_manager as pm
    from web.data_sources.base import Bar

    def _bars(closes: list[float], base_ts: int) -> list:
        out = []
        prev = 100.0
        for i, cl in enumerate(closes):
            out.append(Bar(ts=base_ts + i * 3600, open=prev, high=cl + 0.05,
                           low=cl - 0.05, close=cl, volume=1.0))
            prev = cl
        return out

    monkeypatch.setattr(pm, "ensure_closed_bars", lambda b, tf: b)
    monkeypatch.setattr(pm, "evaluate_signal", lambda formula, raw, **kw: {
        "state": "ok", "direction": DIR_LONG, "strength": 1.0, "position": 1.0,
        "factor_value": 10.0, "message": "",
    })

    m = _manager(tmp_path)
    w = _watch(m)
    w.policy_id = "dd"
    w.notional = 10_000.0
    w.processed_bar_ts = 1_700_000_000 - 3600  # 假装已处理过更早的 bar
    t0 = 1_700_000_000
    # 第一轮：平稳 105（峰值播种 + 开多）
    monkeypatch.setattr(pm, "fetch_cached_bars", lambda s, sym, tf: _bars([105.0] * 3, t0))
    m._process_watch(w)
    assert m.status()["account"]["n_open"] == 1
    assert w.dd_peak == pytest.approx(105.0)
    # 第二轮：急跌到 99 → 熔断平仓
    monkeypatch.setattr(pm, "fetch_cached_bars", lambda s, sym, tf: _bars([105.0, 101.0, 99.0], t0 + 4 * 3600))
    m._process_watch(w)
    st = m.status()
    assert st["account"]["n_open"] == 0
    assert st["account"]["n_trades"] >= 1
    assert "回撤熔断" in m._trades[0]["reason"]
    assert w.cooldown_side == DIR_LONG


# ── 飞书通知（patch send_text，不真发网络） ─────────────────────────────


def _enable_feishu(tmp_path) -> None:
    from web.settings import save_settings

    save_settings({
        "feishu_enabled": True,
        "feishu_webhook_url": "https://example.invalid/hook",
    })


def test_notify_trade_text_fields(tmp_path, monkeypatch: MonkeyPatch) -> None:
    """成交提醒文本应包含开/平动作、价格、盈亏与账户净值。"""
    from web import feishu_notify

    sent: list[str] = []
    monkeypatch.setattr(
        feishu_notify, "send_text", lambda text, **kw: (sent.append(text), (True, "ok"))[1]
    )
    _enable_feishu(tmp_path)

    m = _manager(tmp_path)
    w = _watch(m)
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)  # 开多（名义 10000 → 手续费 2）
    w.last_close = 105.0
    m._reconcile(w, DIR_FLAT, None, 105.0, bar_ts=1001)  # 平多

    texts = [t for t in sent if "模拟盘成交" in t]
    assert len(texts) == 2
    assert "动作：开多" in texts[0] and "动作：平多" in texts[1]
    assert "BTCUSDT" in texts[0] and "1h" in texts[0]
    assert "成交价：" in texts[0]
    assert "账户净值：" in texts[0]
    assert "手续费：" in texts[0]


def test_notify_disabled_no_crash(tmp_path, monkeypatch: MonkeyPatch) -> None:
    """飞书未启用时成交照常记账，通知静默失败。"""
    from web import feishu_notify

    calls: list[tuple] = []

    def fake_send(text, **kw):
        calls.append((text, kw))
        return (True, "ok")

    monkeypatch.setattr(feishu_notify, "send_text", fake_send)
    # 不启用飞书（fixture 已隔离 settings）
    m = _manager(tmp_path)
    w = _watch(m)
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)
    assert m.status()["account"]["n_open"] == 1  # 记账不受影响
    assert calls == []


def test_milestone_fires_once_per_band(tmp_path, monkeypatch: MonkeyPatch) -> None:
    """净值每跨 ±1% 新档位才提醒一次，同档不重复推送。"""
    from web import feishu_notify

    milestones: list[str] = []
    monkeypatch.setattr(
        feishu_notify,
        "send_text",
        lambda text, **kw: (
            milestones.append(text) if "里程碑" in text else None,
            (True, "ok"),
        )[1],
    )
    _enable_feishu(tmp_path)

    m = _manager(tmp_path)  # 起始 100k → 档位步长 1000
    w = _watch(m)
    w.last_close = 100.0
    m._reconcile(w, DIR_LONG, 1.0, 100.0, bar_ts=1000)
    assert milestones == []  # 净值 ≈99998，未跨档

    w.last_close = 120.0  # 名义 10000/100.01 ≈ 99.99 股 → 未实现 ≈ +2000 → 净值 >101k
    m._notify_milestone()  # 无成交也检查里程碑
    assert len(milestones) == 1
    assert "%" in milestones[0]

    m._notify_milestone()  # 仍同档 → 不再推送
    m._notify_trade(m._trades[0])  # 成交也不重复推
    assert len(milestones) == 1

