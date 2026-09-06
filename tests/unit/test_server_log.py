"""日志模块单元测试 (web/server_log.py)。

固化:
1. read_logs 按级别 ([ERROR] 前缀) / 关键词 (大小写不敏感) 过滤;
2. log_error 写入错误日志文件, 超过阈值后轮转保留历史备份;
3. debug_snapshot 兼容旧字段。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location("am_server_log_test", PROJECT_ROOT / "web" / "server_log.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


LOG = _load_module()

SERVER_LINES = [
    "2026-09-06 10:00:00 [INFO] GET /api/health -> 200 (1.2ms)",
    "2026-09-06 10:00:05 [WARNING] slow request detected",
    "2026-09-06 10:00:06 [ERROR] boom",
    "2026-09-06 10:00:07 [INFO] API token rotated",
]
ERROR_LINES = [
    "2026-09-06 10:00:06 [ERROR] boom",
    "2026-09-06 10:00:08 [ERROR] [client] uncaught TypeError: x is not a function",
]


@pytest.fixture()
def log_files(tmp_path, monkeypatch):
    server = tmp_path / "web_server.log"
    errors = tmp_path / "web_errors.log"
    server.write_text("\n".join(SERVER_LINES) + "\n", encoding="utf-8")
    errors.write_text("\n".join(ERROR_LINES) + "\n", encoding="utf-8")
    monkeypatch.setattr(LOG, "SERVER_LOG", server)
    monkeypatch.setattr(LOG, "ERROR_LOG", errors)
    return server, errors


def test_read_logs_returns_both_tails(log_files):
    snap = LOG.read_logs()
    assert snap["server_tail"] == SERVER_LINES
    assert snap["error_tail"] == ERROR_LINES
    assert snap["server_log"].endswith("web_server.log")


def test_read_logs_level_filter(log_files):
    snap = LOG.read_logs(level="error")
    assert snap["server_tail"] == ["2026-09-06 10:00:06 [ERROR] boom"]
    assert snap["error_tail"] == ERROR_LINES


def test_read_logs_search_case_insensitive(log_files):
    snap = LOG.read_logs(search="TOKEN")
    assert snap["server_tail"] == ["2026-09-06 10:00:07 [INFO] API token rotated"]
    assert snap["error_tail"] == []


def test_read_logs_level_and_search_combined(log_files):
    snap = LOG.read_logs(level="warning", search="slow")
    assert snap["server_tail"] == ["2026-09-06 10:00:05 [WARNING] slow request detected"]


def test_read_logs_unknown_level_ignored(log_files):
    snap = LOG.read_logs(level="bogus")
    assert snap["server_tail"] == SERVER_LINES


def test_debug_snapshot_preserves_legacy_fields(log_files):
    snap = LOG.debug_snapshot(1)
    assert set(snap) >= {"debug_mode", "server_log", "error_log", "server_tail", "error_tail"}
    assert len(snap["server_tail"]) <= 1


def test_log_error_writes_block_with_traceback(log_files, monkeypatch):
    server, errors = log_files
    try:
        raise ValueError("oops")
    except ValueError as exc:
        LOG.log_error("boom failed", exc)
    content = errors.read_text(encoding="utf-8")
    assert "[ERROR]" in content or "boom failed" in content
    assert "Traceback (most recent call last)" in content
    assert "ValueError: oops" in content


def test_log_error_rotates_when_over_threshold(log_files, monkeypatch):
    server, errors = log_files
    monkeypatch.setattr(LOG, "ERROR_LOG_MAX_BYTES", 10)
    errors.write_text("", encoding="utf-8")  # 从空文件开始, 保证只有第二次写触发轮转
    LOG.log_error("first block")
    LOG.log_error("second block")
    rotated = errors.with_suffix(".1")
    assert rotated.exists(), "超过阈值后旧错误日志应轮转到 web_errors.1"
    assert "first block" in rotated.read_text(encoding="utf-8")
    current = errors.read_text(encoding="utf-8")
    assert "second block" in current
    assert "first block" not in current