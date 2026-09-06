"""防假中止孤儿守卫 + 按品种回测记忆 单测。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import model_core.supervise as sup  # noqa: E402


# ── 1. supervise：孤儿判定 ────────────────────────────────────────────────

def test_is_orphaned_parent_dead(monkeypatch) -> None:
    monkeypatch.setattr(sup.os, "getppid", lambda: 42)
    monkeypatch.setattr(sup.os, "kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    assert sup.is_orphaned() is True


def test_is_orphaned_parent_alive(monkeypatch) -> None:
    monkeypatch.setattr(sup.os, "getppid", lambda: 42)
    monkeypatch.setattr(sup.os, "kill", lambda pid, sig: None)
    assert sup.is_orphaned() is False


def test_is_orphaned_reparented_to_init() -> None:
    assert sup.is_orphaned.__wrapped__ if hasattr(sup.is_orphaned, "__wrapped__") else True
    # getppid()==1 → 已被 init/launchd 收养 → 孤儿
    orig = sup.os.getppid
    sup.os.getppid = lambda: 1  # type: ignore[method-assign]
    try:
        assert sup.is_orphaned() is True
    finally:
        sup.os.getppid = orig  # type: ignore[method-assign]


# ── 2. supervise：guard_sigterm 决策 ─────────────────────────────────────

def test_guard_sigterm_parent_alive_returns_false(monkeypatch) -> None:
    monkeypatch.setattr(sup, "is_orphaned", lambda: False)
    monkeypatch.setattr(sup, "attach_launchd", lambda *a, **k: None)
    assert sup.guard_sigterm("svc") is False  # 正常中止


def test_guard_sigterm_orphan_reattaches(monkeypatch) -> None:
    monkeypatch.setattr(sup, "is_orphaned", lambda: True)
    attached = {}
    monkeypatch.setattr(sup, "attach_launchd", lambda name, log=None: attached.update(name=name))
    assert sup.guard_sigterm("svc_x", log_path="/tmp/x.log") is True
    assert attached.get("name") == "svc_x"
    # 无服务名也可守卫（返回 True 但不必挂 launchd）
    assert sup.guard_sigterm() is True


# ── 3. supervise：launchd plist 落盘 ─────────────────────────────────────

def test_attach_launchd_writes_plist_and_bootstraps(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sup, "_launch_agents_dir", lambda: tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        import subprocess

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(sup.subprocess, "run", fake_run)
    monkeypatch.setattr(sup, "sys", type("S", (), {"platform": "darwin", "argv": ["prog.py", "--x"], "executable": "/usr/bin/python3"})())

    plist = sup.attach_launchd("train_BTCUSDT", log_path="/tmp/t.log")
    assert plist is not None and plist.exists()
    data = json.loads(plist.read_text(encoding="utf-8")) if plist.suffix == ".json" else None
    if data is None:  # plistlib binary → 用 plistlib 读
        import plistlib

        data = plistlib.loads(plist.read_bytes())
    assert data["Label"] == "com.alphamaster.train_BTCUSDT"
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ProgramArguments"][0] == "/usr/bin/python3"
    assert "ALPHAMASTER_UNDER_LAUNCHD" in data["EnvironmentVariables"]
    assert data["StandardOutPath"] == "/tmp/t.log"
    assert any(c[:2] == ["launchctl", "bootstrap"] for c in calls)


def test_attach_launchd_absolutizes_relative_argv0(monkeypatch, tmp_path) -> None:
    """launchd 不支持相对 program 路径（exit 78 EX_CONFIG）→ argv[0] 必须绝对化。"""
    monkeypatch.setattr(sup, "_launch_agents_dir", lambda: tmp_path)

    def fake_run(cmd, **kw):
        import subprocess

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(sup.subprocess, "run", fake_run)
    monkeypatch.setattr(sup, "sys", type("S", (), {
        "platform": "darwin", "argv": ["prog.py", "--x"],
        "executable": "/usr/bin/python3"})())

    plist = sup.attach_launchd("svc_rel", log_path="/tmp/r.log",
                               argv=[".venv/bin/python", "-u", "scripts/x.py"])
    import plistlib

    data = plistlib.loads(plist.read_bytes())
    p0 = data["ProgramArguments"][0]
    assert Path(p0).is_absolute()
    assert p0.endswith(".venv/bin/python")
    assert data["WorkingDirectory"] == str(Path(__file__).resolve().parents[2])


# ── 4. 按品种回测记忆（bt_prefs） ────────────────────────────────────────

def test_bt_prefs_roundtrip_and_defaults(tmp_path) -> None:
    from web.bt_prefs import get_symbol, load_all, save_all, set_symbol

    p = tmp_path / "bt_prefs.json"
    assert get_symbol("BTCUSDT", p)["hold_policy"] == "signal"  # 无记录 → 默认
    set_symbol("BTCUSDT", {"hold_policy": "dd+chandelier", "window_bars": 5000}, p)
    set_symbol("XAUUSD", {"hold_policy": "risk", "window_bars": 800}, p)
    got = get_symbol("BTCUSDT", p)
    assert got["hold_policy"] == "dd+chandelier"
    assert got["window_bars"] == 5000
    # 跨品种互不覆盖
    assert get_symbol("XAUUSD", p)["hold_policy"] == "risk"
    assert load_all(p)["BTCUSDT"]["window_bars"] == 5000
    # 非法窗口(<800)/组合清洗
    set_symbol("ETHUSDT", {"hold_policy": "  ", "window_bars": 100}, p)
    bad = get_symbol("ETHUSDT", p)
    assert bad["hold_policy"] == "signal"
    assert bad["window_bars"] is None
    # save_all 覆盖写入
    save_all({}, p)
    assert load_all(p) == {}


def test_bt_prefs_api_endpoints(monkeypatch, tmp_path) -> None:
    """/api/backtest/prefs GET（品种优先+全局兜底）与 PUT（按品种落盘）端到端。"""
    import importlib.util

    from web import bt_prefs as bp

    monkeypatch.setattr(bp, "PREFS_PATH", tmp_path / "bt_prefs.json")

    spec = importlib.util.spec_from_file_location(
        "am_web_app2", Path(__file__).resolve().parents[2] / "web" / "app.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # PUT：记录 BTCUSDT 的组合+窗口
    r = mod.api_backtest_prefs_put(mod.SettingsRequest(
        bt_prefs_symbol="BTCUSDT", bt_hold_policy="risk+dd", bt_window_bars=3000,
    ))
    assert r["ok"] is True
    assert r["prefs"]["hold_policy"] == "risk+dd"
    assert r["prefs"]["window_bars"] == 3000

    # GET：品种记录优先
    g = mod.api_backtest_prefs(symbol="BTCUSDT")
    assert g["from_symbol"] is True
    assert g["prefs"]["hold_policy"] == "risk+dd"

    # GET：无品种记录 → 全局设置兜底（from_symbol=False）
    monkeypatch.setattr(mod, "load_settings",
                        lambda: {"bt_hold_policy": "be", "bt_window_bars": 2000})
    g2 = mod.api_backtest_prefs(symbol="XAUUSD")
    assert g2["from_symbol"] is False
    assert g2["prefs"]["hold_policy"] == "be"
    assert g2["prefs"]["window_bars"] == 2000