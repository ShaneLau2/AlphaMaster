"""settings.matrix_applied（矩阵最优“用户接受记录”）持久化单测。

背景：把用户接受过的矩阵最优 combo 记入 web_settings（与手动记忆 bt_prefs /
bt_hold_policy 分开），重启后据此恢复回测页「已应用」态。锁定：
1) 默认空 dict，文件缺失不报错；
2) 保存→读取 往返保留 combo 及元数据；
3) 非法记录（缺 combo / 非 dict / 非法数值）被清洗掉，不影响其它字段。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import web.settings as settings_mod  # noqa: E402


def _roundtrip(tmp_path, patch_key: dict):
    """把 settings 重定向到 tmp 文件并返回 load_settings。"""
    return None


def test_matrix_applied_default_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "web_settings.json")
    s = settings_mod.load_settings()
    assert s["matrix_applied"] == {}


def test_matrix_applied_save_load_roundtrip(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "web_settings.json")
    settings_mod.save_settings({
        "matrix_applied": {
            "BTCUSDT": {"combo": "dd+risk", "at": 123456.0, "sharpe": 1.23,
                        "max_drawdown": -0.031, "window_bars": 15000,
                        "window_mode": "tail"},
            "ETHUSDT": {"combo": "signal", "at": 1.0},  # signal 也算合法记录（组合 id 本身）
        }
    })
    s = settings_mod.load_settings()
    rec = s["matrix_applied"]
    assert rec["BTCUSDT"]["combo"] == "dd+risk"
    assert rec["BTCUSDT"]["sharpe"] == 1.23
    assert rec["BTCUSDT"]["max_drawdown"] == -0.031
    assert rec["BTCUSDT"]["window_bars"] == 15000
    assert rec["ETHUSDT"]["combo"] == "signal"


def test_matrix_applied_cleans_garbage(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "web_settings.json")
    (tmp_path / "web_settings.json").write_text(
        '{"matrix_applied": {'
        '"OK": {"combo": "risk"},'
        '"BAD1": {"combo": ""},'
        '"BAD2": "not-a-dict",'
        '"BAD3": {"combo": "dd", "sharpe": "abc", "max_drawdown": "x"}}}',
        encoding="utf-8",
    )
    rec = settings_mod.load_settings()["matrix_applied"]
    # BAD1（空 combo）与非 dict 被丢；含 combo 的记录保留（哪怕数值字段脏）
    assert set(rec.keys()) == {"OK", "BAD3"}
    assert rec["OK"]["combo"] == "risk"
    # 非法数值字段被丢、合法结构保留
    assert rec["BAD3"]["combo"] == "dd"
    assert "sharpe" not in rec["BAD3"]
    assert "max_drawdown" not in rec["BAD3"]
