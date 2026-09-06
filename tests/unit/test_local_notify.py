"""统一本地通知模块单测：ncprefs 宿主解析 + 非 macOS 判定 + 日志兜底。"""
from __future__ import annotations

import plistlib
from pathlib import Path

from web import local_notify as ln


def test_ncprefs_hosts_parses(tmp_path, monkeypatch) -> None:
    p = tmp_path / "com.apple.ncprefs.plist"
    p.write_bytes(plistlib.dumps({
        "apps": [
            {"bundle-id": "com.apple.Terminal", "path": "/System/Applications/Utilities/Terminal.app",
             "flags": 8396814, "auth": 0},
            {"bundle-id": "com.microsoft.VSCode", "path": "/Applications/Visual Studio Code.app",
             "flags": 8396814, "auth": 7},
            {"bundle-id": "com.apple.mail", "path": "/System/Applications/Mail.app", "flags": 310378510},
        ]
    }))
    monkeypatch.setattr(ln, "NCPREFS", p)
    hosts = ln._ncprefs_hosts()
    bids = [h["bundle_id"] for h in hosts]
    assert "com.apple.Terminal" in bids and "com.microsoft.VSCode" in bids
    assert "com.apple.mail" not in bids
    assert all("flags" in h for h in hosts)
    # 文件缺失 → 空列表
    monkeypatch.setattr(ln, "NCPREFS", tmp_path / "nope.plist")
    assert ln._ncprefs_hosts() == []


def test_diagnose_non_darwin(monkeypatch) -> None:
    monkeypatch.setattr(ln.sys, "platform", "linux")
    d = ln._diagnose()
    assert d["verdict"] == "non-darwin"


def test_notify_writes_log_non_darwin(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ln.sys, "platform", "linux")
    monkeypatch.setattr(ln, "NOTIFY_LOG", tmp_path / "notifications.log")
    res = ln.notify("测试", "hello 中文")
    assert res["posted"] is False and res["verdict"] == "non-darwin"
    assert (tmp_path / "notifications.log").exists()
    assert "hello 中文" in (tmp_path / "notifications.log").read_text(encoding="utf-8")


def test_notify_darwin_osascript_path(monkeypatch, tmp_path) -> None:
    """darwin + 无宿主 + UN 不可用 → posted=False 且带可操作 hint，仍落日志。"""
    monkeypatch.setattr(ln.sys, "platform", "darwin")
    monkeypatch.setattr(ln, "NCPREFS", tmp_path / "empty.plist")
    monkeypatch.setattr(ln, "PROBE_BIN", tmp_path / "no_probe")
    monkeypatch.setattr(ln, "NOTIFY_LOG", tmp_path / "notifications.log")

    class _R:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(ln.subprocess, "run", lambda *a, **k: _R())
    res = ln.notify("标题", "内容")
    assert res["posted"] is False
    assert res["verdict"] == "cli-unbundleable"
    assert res["hint"] and "系统设置→通知" in res["hint"]
    assert (tmp_path / "notifications.log").exists()


def test_notify_darwin_denied_when_rc_nonzero(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ln.sys, "platform", "darwin")
    monkeypatch.setattr(ln, "NOTIFY_LOG", tmp_path / "n.log")

    class _R:
        returncode = 1
        stderr = b"exec error"

    monkeypatch.setattr(ln.subprocess, "run", lambda *a, **k: _R())
    res = ln.notify("t", "m")
    assert res["posted"] is False
    assert "rc=1" in (res["hint"] or "")
