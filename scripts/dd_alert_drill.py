"""DD 告警演练：在隔离的模拟实盘账户上注入合成暴跌→收复行情，
完整走一遍生产代码路径（_process_watch → _dd_gate_step → _notify_dd_gate），
验证「熔断」与「收复」两条飞书告警（或打印未配置提示）。

用法：
  .venv/bin/python scripts/dd_alert_drill.py            # 人读输出
  .venv/bin/python scripts/dd_alert_drill.py --json     # JSON 转录（web 演练端点使用）

只使用临时目录状态，不影响正式模拟盘账户/监控。
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.data_sources.base import Bar  # noqa: E402
from web.paper_manager import (  # noqa: E402
    DIR_LONG,
    PaperTradingManager,
    PaperWatch,
)


def _bars(closes: list[float], base_ts: int) -> list:
    out = []
    prev = closes[0]
    for i, cl in enumerate(closes):
        out.append(Bar(ts=base_ts + i * 60, open=prev, high=max(prev, cl) + 0.05,
                       low=min(prev, cl) - 0.05, close=cl, volume=1.0))
        prev = cl
    return out


def _make_manager(tmp: Path) -> PaperTradingManager:
    return PaperTradingManager(
        state_file=tmp / "paper_sim_state.json",
        starting_balance=100_000.0,
        commission_pct=0.02,
        slippage_pct=0.01,
        default_notional=10_000.0,
    )


def _make_watch(mgr: PaperTradingManager, tmp: Path) -> PaperWatch:
    strat = tmp / "best_BTCUSDT.json"
    strat.write_text('{"vocab_version": null, "symbol": "BTCUSDT", "formula": [1, 2, 3], "best_score": 1.0}',
                     encoding="utf-8")
    w = PaperWatch(
        id="binance:BTCUSDT:5m:best_BTCUSDT",
        source="binance", symbol="BTCUSDT", timeframe="5m",
        strategy_file=str(strat), strategy_name="best_BTCUSDT",
        formula=[1, 2, 3], vocab_version=None,
        strategy_symbol="BTCUSDT", strategy_timeframe="5m",
        best_score=1.0, cadence_s=60, notional=10_000.0,
    )
    mgr._watches[w.id] = w
    return w


def run_drill() -> dict:
    """隔离 DD 演练：返回完整转录（步骤/事件/流水/飞书结果），不写任何生产状态。"""
    import web.paper_manager as pm

    # 打桩：行情源与信号源（只驱动 dd 阶梯，不依赖真实网络/模型）
    pm.ensure_closed_bars = lambda bars, tf: bars  # type: ignore[method-assign]
    pm.evaluate_signal = lambda formula, raw: {  # type: ignore[method-assign]
        "state": "ok", "direction": DIR_LONG, "strength": 1.0, "position": 1.0,
        "factor_value": 10.0, "message": "",
    }

    steps: list[dict] = []
    events: list[dict] = []
    feishu: list[dict] = []
    trades: list[dict] = []

    def _run_step(step_label: str, m: PaperTradingManager, w: PaperWatch) -> None:
        """执行一步 _process_watch，捕获该步的飞书打印行与状态。"""
        n_before = len(m._trades)
        feishu_out = io.StringIO()
        with contextlib.redirect_stdout(feishu_out):
            m._process_watch(w)
        st = m.status()
        for ln in feishu_out.getvalue().splitlines():
            if "[飞书通知]" in ln:
                feishu.append({"step": step_label, "line": ln.strip()})
        steps.append({
            "step": step_label,
            "n_open": st["account"]["n_open"],
            "dd_gate": w.dd_gate,
            "dd_pct": round(w.dd_pct, 3) if w.dd_pct is not None else None,
        })
        if len(m._trades) > n_before:
            tr = m._trades[0]
            trades.append({"action": tr.get("action"), "reason": tr.get("reason"),
                           "price": tr.get("price"), "pnl": tr.get("pnl")})

    with tempfile.TemporaryDirectory(prefix="dd_drill_") as td:
        tmp = Path(td)
        m = _make_manager(tmp)
        w = _make_watch(m, tmp)
        # 正交组合：dd + chandelier（验证组合上下文出现在平仓原因/告警里）
        w.policy_id = "dd+chandelier"
        w.processed_bar_ts = 1_700_000_000 - 3600
        t0 = 1_700_000_000

        # 1) 平稳 105 → 播种峰值并开多
        pm.fetch_cached_bars = lambda s, sym, tf: _bars([105.0] * 3, t0)  # type: ignore[method-assign]
        _run_step("① 平稳 105：开仓", m, w)

        # 2) 急跌到 ~100.5（自峰值 -4.3%，>3% 触档、<6% 深档）→ 普通熔断
        pm.fetch_cached_bars = lambda s, sym, tf: _bars([105.0, 103.0, 100.5], t0 + 4 * 3600)  # type: ignore[method-assign]
        _run_step("② 急跌 100.5：熔断触发", m, w)

        # 3) 收复到 104.6（自峰值 -0.38%，≥ 收复线 -1%）→ 闸恢复
        pm.fetch_cached_bars = lambda s, sym, tf: _bars([105.0, 103.0, 104.6], t0 + 8 * 3600)  # type: ignore[method-assign]
        _run_step("③ 收复 104.6：闸恢复", m, w)

        # 从 dd 事件日志捞本次演练的转移（隔离进程写共享日志，便于页面直接回溯）
        try:
            from web.dd_events import recent_events

            evs = recent_events(8)
            evs = [e for e in evs if e.get("scope") == "paper" and e.get("symbol") == "BTCUSDT"]
            events = [{"event": e.get("event"), "dd_pct": e.get("dd_pct"),
                       "mark": e.get("mark"), "peak": e.get("peak"),
                       "state": e.get("state")} for e in evs[:4]]
        except Exception:  # noqa: BLE001
            events = []

    return {
        "ok": True,
        "policy": "dd+chandelier",
        "account": "隔离临时账户（不影响正式模拟盘）",
        "steps": steps,
        "events": events,
        "trades": trades,
        "feishu": feishu,
        "note": "若实时分析页已启用并填好飞书 webhook，演练会真实推送「熔断」「收复」两条群消息。",
    }


def main() -> int:
    as_json = "--json" in sys.argv
    if as_json:
        sys.argv.remove("--json")
    out = run_drill()
    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print("== DD 告警演练（dd+chandelier · 临时账户，不影响正式状态）==")
    for s in out["steps"]:
        print(f"{s['step']}：n_open={s['n_open']}，dd_gate={s['dd_gate']}")
    for t in out["trades"]:
        print(f"   流水：{t.get('action')} · 原因：{t.get('reason')}")
    for e in out["events"]:
        print(f"   DD 事件：{e.get('event')}（回撤 {e.get('dd_pct')}%，现价 {e.get('mark')}，峰值 {e.get('peak')}）")
    for f in out.get("feishu", []):
        print(f"   {f['line']}")
    print("\n完成。若 实时分析页 已启用并填好飞书 webhook，应收到两条告警：")
    print("  【AlphaMaster 回撤熔断 (DD)】 事件：熔断（含回撤幅度/现价/仓位/恢复条件）")
    print("  【AlphaMaster 回撤熔断 (DD)】 事件：收复")
    print("未配置时上方打印的 ✗ 提示即为原因（通知失败绝不影响演练本身）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())