"""飞书 DD 熔断/收复通知函数单测（未配置 webhook 时应优雅返回而非抛错）。"""
from __future__ import annotations

from web import feishu_notify as fn


def test_notify_paper_dd_gate_disabled_when_no_webhook(monkeypatch) -> None:
    monkeypatch.setattr(fn, "load_settings", lambda: {"feishu_enabled": True, "feishu_webhook_url": "", "feishu_secret": ""})
    ok, msg = fn.notify_paper_dd_gate(
        symbol="BTCUSDT", timeframe="5m", policy_label="吊灯止损 (ATR) + 回撤熔断 (DD)",
        event="熔断", dd_pct=-3.4, mark=97000.0, peak=100450.0,
        recover_hint="熔断期暂停新开仓；需收复到回撤 ≤1.0% 才恢复开仓", position="在途持仓（LONG）",
    )
    assert ok is False and "Webhook" in msg


def test_notify_paper_dd_gate_feishu_disabled(monkeypatch) -> None:
    monkeypatch.setattr(fn, "load_settings", lambda: {"feishu_enabled": False})
    ok, msg = fn.notify_paper_dd_gate(symbol="BTCUSDT", timeframe="5m", event="收复", dd_pct=-0.4)
    assert ok is False and "未启用" in msg


def test_paper_trade_reason_passthrough(monkeypatch) -> None:
    captured: dict = {}

    def _fake_send(text: str, **kw):
        captured["text"] = text
        return True, "ok"

    monkeypatch.setattr(fn, "load_settings", lambda: {"feishu_enabled": True, "feishu_webhook_url": "http://x", "feishu_secret": ""})
    monkeypatch.setattr(fn, "send_text", _fake_send)
    ok, _ = fn.notify_paper_trade(
        symbol="BTCUSDT", timeframe="5m", action="平多", price=97800.0,
        reason="吊灯止损（持仓管理·吊灯止损 (ATR) + 回撤熔断 (DD)）", pnl=-120.5, equity=99999.5,
    )
    assert ok is True and "吊灯止损（持仓管理·吊灯止损 (ATR) + 回撤熔断 (DD)）" in captured["text"]
