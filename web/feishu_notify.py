"""飞书自定义机器人通知（信号方向转折时推送文本）。

参考 PA_Agent 的 webhook + 可选签名校验写法。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from typing import Any

from web.settings import load_settings

_DIR_CN = {
    "LONG": "看涨",
    "SHORT": "看跌",
    "FLAT": "不确定",
}


def direction_cn(direction: str | None) -> str:
    if not direction:
        return "未知"
    return _DIR_CN.get(str(direction).upper(), str(direction))


def strength_cn(strength: float | None, direction: str | None) -> str:
    if direction == "FLAT" or direction is None:
        return "没把握"
    s = max(0.0, min(1.0, float(strength or 0.0)))
    if s < 0.2:
        return "一点把握"
    if s < 0.4:
        return "把握不大"
    if s < 0.6:
        return "一半把握"
    if s < 0.8:
        return "比较有把握"
    return "很有把握"


def _gen_sign(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def send_text(
    text: str,
    *,
    webhook_url: str | None = None,
    secret: str | None = None,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """向飞书群发送纯文本。返回 (ok, message)。"""
    settings = load_settings()
    url = (webhook_url if webhook_url is not None else settings.get("feishu_webhook_url") or "").strip()
    if not url:
        return False, "未配置 Webhook URL"
    sec = (secret if secret is not None else settings.get("feishu_secret") or "").strip()

    payload: dict[str, Any] = {
        "msg_type": "text",
        "content": {"text": text},
    }
    if sec:
        ts = int(time.time())
        payload["timestamp"] = str(ts)
        payload["sign"] = _gen_sign(sec, ts)

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)

    if data.get("code") == 0 or data.get("StatusCode") == 0:
        return True, "ok"

    code = data.get("code", data.get("StatusCode", "?"))
    msg = data.get("msg", data.get("StatusMessage", ""))
    hint = ""
    if code == 19021:
        hint = "（签名校验失败，请检查密钥或留空禁用签名）"
    elif code == 19024:
        hint = "（关键词校验失败，请检查机器人自定义关键词）"
    elif code == 19022:
        hint = "（IP 不在白名单）"
    return False, f"飞书返回 code={code} msg={msg}{hint}"


def notify_direction_flip(
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    prev_direction: str,
    new_direction: str,
    strength: float | None = None,
    factor_value: float | None = None,
) -> tuple[bool, str]:
    """信号方向发生转折时推送提醒。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    prev_cn = direction_cn(prev_direction)
    new_cn = direction_cn(new_direction)
    grasp = strength_cn(strength, new_direction)
    factor_s = f"{factor_value:+.4f}" if factor_value is not None else "—"

    text = (
        f"【AlphaMaster 信号转折】\n"
        f"{symbol} · {timeframe}\n"
        f"上次判断：{prev_cn}\n"
        f"本次判断：{new_cn}（{grasp}）\n"
        f"策略：{strategy_name}\n"
        f"因子：{factor_s}"
    )
    return send_text(text)


def notify_realtime_deviation(
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    direction: str | None,
    factor_value: float | None,
    entry_price: float | None,
    current_price: float | None,
    dev_pct: float | None,
    threshold_pct: float,
) -> tuple[bool, str]:
    """价格相对持仓方案入场参考偏离超过阈值时推送提醒（一次越线只推一条，
    回到阈值一半以内才允许再次提醒）。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    dir_cn = direction_cn(direction)
    f_s = f"{factor_value:+.4f}" if factor_value is not None else "—"
    e_s = f"{entry_price:,.2f}" if entry_price else "—"
    c_s = f"{current_price:,.2f}" if current_price else "—"
    d_s = f"{dev_pct:+.2f}%" if dev_pct is not None else "—"
    text = (
        f"【AlphaMaster 偏离入场告警】\n"
        f"{symbol} · {timeframe} · {strategy_name}\n"
        f"方向：{dir_cn} · 因子：{f_s}\n"
        f"入场参考：{e_s} → 现价：{c_s}\n"
        f"偏离：{d_s}（阈值 ±{threshold_pct:g}%）\n"
        f"注意：若已持有仓位请检查止损/止盈是否仍有效。"
    )
    return send_text(text)


def notify_realtime_stale(
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    factor_value: float | None = None,
    flat_run: int,
    factor_val: float,
    last_close: float | None = None,
    price_moved_run: int = 0,
    trace_tail: list | None = None,
) -> tuple[bool, str]:
    """因子硬钝化提醒：连续 N 根新 bar 几乎不变且已判定为平台恒定（硬钝化）时推送。
    带最近轨迹摘要。阈值 N（根数）由调用方按配置传入，此处只负责组稿发送。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    f_s = f"{factor_value:+.4f}" if factor_value is not None else "—"
    px_s = f"{last_close:,.6g}" if last_close else "—"
    moved_s = f"{price_moved_run} 根" if price_moved_run > 0 else "0（价格也基本没动）"
    lines: list[str] = [
        "【AlphaMaster 因子僵化告警】",
        f"{symbol} · {timeframe} · {strategy_name}",
        f"因子已连续 {flat_run} 根新 bar 几乎不变（相邻变化 < 0.001%），并判定为硬钝化（平台恒定 = {factor_val:+.6g}）。",
        f"引擎仍在正常重算——多半是公式饱和/钝化，行情小幅波动不足以改变下次信号，不是卡死。",
        f"当前因子：{f_s} · 最近收盘：{px_s}",
        f"同时段价格移动：{moved_s}",
    ]
    tail = [p for p in (trace_tail or []) if p and len(p) >= 3]
    if tail:
        fs = " → ".join(f"{p[2]:+.6g}" for p in tail)
        lines.append(f"轨迹摘要（最近 {len(tail)} 根因子）：{fs}")
    lines.append("若属预期饱和请忽略；需要更强的区分度建议提高无信号阈值或用中性带约束训练。")
    return send_text("\n".join(lines))


def notify_paper_trade(
    *,
    symbol: str,
    timeframe: str,
    action: str,
    price: float | None = None,
    qty: float | None = None,
    notional_value: float | None = None,
    fee: float | None = None,
    pnl: float | None = None,
    cash_after: float | None = None,
    equity: float | None = None,
    reason: str = "",
) -> tuple[bool, str]:
    """模拟盘成交提醒（开/平/反手），与信号提醒共用飞书配置。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    lines = ["【AlphaMaster 模拟盘成交】", f"{symbol} · {timeframe}", f"动作：{action}"]
    if price is not None:
        lines.append(f"成交价：{price:,.6g}")
    if qty is not None:
        lines.append(f"数量：{qty:,.6g}")
    if notional_value is not None:
        lines.append(f"名义金额：{notional_value:,.2f}")
    if fee is not None:
        lines.append(f"手续费：{fee:,.2f}")
    if pnl:
        lines.append(f"本次盈亏：{pnl:+,.2f}")
    if reason:
        lines.append(f"原因：{reason}")
    if equity is not None:
        cash_s = f" · 现金 {cash_after:,.2f}" if cash_after is not None else ""
        lines.append(f"账户净值：{equity:,.2f}{cash_s}")
    return send_text("\n".join(lines))


def notify_paper_milestone(
    *,
    equity: float,
    starting_balance: float,
) -> tuple[bool, str]:
    """模拟盘盈亏里程碑提醒（账户净值每跨越起始资金 ±1% 提醒一次）。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    pnl = float(equity or 0.0) - float(starting_balance or 0.0)
    ret = pnl / float(starting_balance) * 100.0 if starting_balance else 0.0
    text = (
        "【AlphaMaster 模拟盘里程碑】\n"
        f"账户净值：{equity:,.2f}\n"
        f"较起始：{pnl:+,.2f}（{ret:+.2f}%）"
    )
    return send_text(text)


def notify_paper_dd_gate(
    *,
    symbol: str,
    timeframe: str,
    policy_label: str = "",
    event: str = "熔断",
    dd_pct: float | None = None,
    mark: float | None = None,
    peak: float | None = None,
    recover_hint: str = "",
    position: str | None = None,
) -> tuple[bool, str]:
    """模拟盘回撤熔断（DD）状态机事件单独告警：熔断 / 深档熔断 / 收复。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    lines = ["【AlphaMaster 回撤熔断 (DD)】", f"{symbol} · {timeframe}"]
    if policy_label:
        lines.append(f"持仓方案：{policy_label}")
    lines.append(f"事件：{event}")
    if dd_pct is not None:
        lines.append(f"自峰值回撤：{dd_pct:+.2f}%")
    if position:
        lines.append(f"仓位：{position}")
    if mark is not None:
        lines.append(f"现价：{mark:,.6g}")
    if peak is not None:
        lines.append(f"滚动峰值：{peak:,.6g}")
    if recover_hint:
        lines.append(recover_hint)
    return send_text("\n".join(lines))


def notify_hold_matrix_done(matrix: dict[str, Any]) -> tuple[bool, str]:
    """N×N 全组合矩阵跑完后的飞书摘要：Top5（按夏普）vs 基线 + 帕累托/关注列表。

    matrix = results/hold_matrix_latest.json 的内容（由 hold_matrix_manager 在任务
    completed 瞬间调用一次；不配置 webhook 时优雅降级，不影响任务状态）。
    """
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    base = matrix.get("baseline_signal") or {}
    ranking = matrix.get("ranking") or []
    pareto = matrix.get("pareto_front") or []
    focus = matrix.get("focus_list") or []

    def fmt(key: str, v, signed: bool = True) -> str:
        n = v if isinstance(v, (int, float)) else None
        if n is None:
            return "—"
        if key == "max_drawdown":
            return f"{(n * 100):.1f}%"
        if key == "profit_loss_ratio":
            return f"{n:+.2f}" if signed else f"{n:.2f}"
        if key == "n_trades":
            return str(int(n))
        return f"{n * 100:+.2f}%" if key == "total_return" else f"{n:+.2f}"

    # 窗口标注（tail 尾部 N 根 / spread regime 分层抽样块）
    win = "全部历史"
    blocks = matrix.get("window_blocks") or []
    if matrix.get("window_mode") == "spread" and blocks:
        core = [b for b in blocks if b.get("kind") != "warmup"] or blocks
        parts = [f"bar {b['start']}..{b['end']}" for b in core]
        win = f"分层抽样（{matrix.get('regime') or 'vol'} · {matrix.get('chunks') or len(core)} 块）：" + " | ".join(parts)
    else:
        wb = matrix.get("window_bars")
        if wb:
            ws = matrix.get("window_start") or 0
            bars = matrix.get("bars") or 0
            win = f"尾部 {wb} 根（原始序列 bar {ws}..{ws + bars - 1}）"

    lines = ["【AlphaMaster 全组合矩阵完成】", f"窗口：{win}"]
    if matrix.get("data_file"):
        lines.append(f"数据：{matrix['data_file']}")

    def combo_cn(pid: str) -> str:
        try:
            from web.hold_policy import combo_label

            return combo_label(pid)
        except Exception:
            return pid

    base_s = (f"收益 {fmt('total_return', base.get('total_return'))} · 夏普 {fmt('sharpe', base.get('sharpe'), False)}"
              f" · 回撤 {fmt('max_drawdown', base.get('max_drawdown'))} · 盈亏比 {fmt('profit_loss_ratio', base.get('profit_loss_ratio'), False)}")
    lines.append(f"基线 信号跟随：{base_s}")

    if ranking:
        lines.append(f"\nTop{min(5, len(ranking))}（按夏普，全部 {len(ranking)} 个组合）：")
        for i, r in enumerate(ranking[:5], 1):
            tags = []
            if r.get("focus"):
                tags.append("★关注")
            elif r.get("pareto"):
                tags.append("帕累托")
            tag_s = (" [" + ",".join(tags) + "]") if tags else ""
            dd = r.get("max_drawdown")
            dd_s = f"{dd * 100:.1f}%" if isinstance(dd, (int, float)) else "—"
            lines.append(f"{i}. {combo_cn(r.get('combo'))} {tag_s}\n"
                         f"    收益 {fmt('total_return', r.get('total_return'))} · 夏普 {fmt('sharpe', r.get('sharpe'), False)} · "
                         f"回撤 {dd_s} · 盈亏比 {fmt('profit_loss_ratio', r.get('profit_loss_ratio'), False)}")

    if focus:
        names = "、".join(combo_cn(r.get("combo")) for r in focus[:6])
        lines.append(f"\n值得实盘关注（帕累托 ∩ 优于基线，{len(focus)} 个）：{names}")
    elif pareto:
        lines.append(f"\n帕累托前沿共 {len(pareto)} 个组合，但均未在 夏普/回撤/盈亏比 上全面优于基线")
    else:
        lines.append("\n（无帕累托/关注组合）")

    return send_text("\n".join(lines))

