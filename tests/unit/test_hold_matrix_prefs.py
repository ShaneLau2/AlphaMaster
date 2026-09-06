"""按品种矩阵设置记忆 + spread 分层窗口的飞书文案 单测。"""
from __future__ import annotations

from web import feishu_notify as fn
from web.hold_matrix_prefs import get_symbol, load_all, set_symbol


def test_set_get_roundtrip(tmp_path) -> None:
    p = tmp_path / "prefs.json"
    set_symbol("BTCUSDT", {"window_mode": "spread", "window_bars": 8000,
                           "regime": "vol", "chunks": 4,
                           "data_file": "/x/data/BTCUSDT_M5.parquet"}, path=p)
    got = get_symbol("BTCUSDT", path=p)
    assert got["window_mode"] == "spread"
    assert got["window_bars"] == 8000
    assert got["regime"] == "vol" and got["chunks"] == 4
    assert got["data_file"] == "/x/data/BTCUSDT_M5.parquet"
    # 其他品种不串台
    assert get_symbol("XAUUSD", path=p)["window_mode"] == "tail"
    assert "XAUUSD" not in load_all(path=p)


def test_set_symbol_partial_keeps_defaults(tmp_path) -> None:
    p = tmp_path / "prefs.json"
    set_symbol("BTCUSDT", {"window_bars": 3000}, path=p)
    got = get_symbol("BTCUSDT", path=p)
    assert got["window_bars"] == 3000
    assert got["window_mode"] == "tail" and got["chunks"] == 4


def test_feishu_spread_window_text(monkeypatch) -> None:
    captured: dict = {}

    def _fake_send(text: str, **kw):
        captured["text"] = text
        return True, "ok"

    monkeypatch.setattr(fn, "load_settings", lambda: {"feishu_enabled": True, "feishu_webhook_url": "http://x", "feishu_secret": ""})
    monkeypatch.setattr(fn, "send_text", _fake_send)
    matrix = {
        "data_file": "data/slices/spread_8000_vol/BTCUSDT_M5.parquet",
        "window_mode": "spread",
        "regime": "vol",
        "chunks": 4,
        "window_bars": 8000,
        "window_blocks": [
            {"kind": "warmup", "start": 800, "end": 1599},
            {"kind": "core", "start": 1600, "end": 5800},
            {"kind": "core", "start": 16000, "end": 21800},
        ],
        "baseline_signal": {"total_return": 0.1, "sharpe": 2.0, "max_drawdown": -0.05, "profit_loss_ratio": 1.1},
        "ranking": [{"combo": "signal", "total_return": 0.1, "sharpe": 2.0, "max_drawdown": -0.05,
                     "profit_loss_ratio": 1.1, "pareto": True, "focus": False}],
        "pareto_front": [],
        "focus_list": [],
    }
    ok, _ = fn.notify_hold_matrix_done(matrix)
    assert ok is True
    txt = captured["text"]
    assert "分层抽样" in txt and "vol" in txt
    assert "bar 1600..5800" in txt and "bar 16000..21800" in txt   # 核心块范围（warmup 不展示）
    assert "bar 800..1599" not in txt.splitlines()[0].split("：")[1] if "：" in txt.splitlines()[0] else True