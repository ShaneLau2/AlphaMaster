"""API 令牌鉴权助手单元测试（web/app.py 的 _token_matches / _origin_is_trusted）。

背景：GitHub Pages 镜像页可直连本机后端后，所有 /api 请求必须携带令牌，
令牌发放端点 GET /api/auth/token 按来源白名单放行。本文件固化：
1. 令牌校验（Bearer 头 / ?token= 参数 / 错误令牌拒绝）；
2. 来源白名单（同源/环回/局域网同端口/已知镜像放行；公网域名/端口不匹配拒绝）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_app_module():
    spec = importlib.util.spec_from_file_location("am_web_app_token", PROJECT_ROOT / "web" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


APP = _load_app_module()


def _req(port: int | None = 8765):
    return SimpleNamespace(url=SimpleNamespace(port=port))


# ── 令牌校验 ──────────────────────────────────────────────────────────────

def test_token_matches_exact(monkeypatch):
    monkeypatch.setattr(APP, "API_TOKEN", "a" * 64)
    assert APP._token_matches("a" * 64) is True


def test_token_matches_rejects_wrong_or_empty(monkeypatch):
    monkeypatch.setattr(APP, "API_TOKEN", "a" * 64)
    assert APP._token_matches("b" * 64) is False
    assert APP._token_matches("") is False
    assert APP._token_matches(None) is False  # type: ignore[arg-type]


def test_auth_from_header_and_query(monkeypatch):
    monkeypatch.setattr(APP, "API_TOKEN", "tok123")
    req = SimpleNamespace(
        headers={"authorization": "Bearer tok123"},
        query_params={"token": ""},
    )
    assert APP._auth_from(req) == "tok123"
    req = SimpleNamespace(
        headers={},
        query_params={"token": "tok123"},
    )
    assert APP._auth_from(req) == "tok123"
    req = SimpleNamespace(
        headers={"authorization": "Basic abc"},
        query_params={"token": ""},
    )
    assert APP._auth_from(req) == ""


# ── 令牌发放来源白名单 ────────────────────────────────────────────────────

def test_origin_no_origin_allowed():
    assert APP._origin_is_trusted(_req(), "") is True


def test_origin_loopback_allowed():
    for origin in ("http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"):
        assert APP._origin_is_trusted(_req(), origin) is True, origin


def test_origin_mirror_allowed():
    assert APP._origin_is_trusted(_req(), "https://shanelau2.github.io") is True


def test_origin_public_domain_rejected():
    assert APP._origin_is_trusted(_req(), "https://evil.example.com") is False
    assert APP._origin_is_trusted(_req(), "https://shanelau2.github.io.evil.com") is False


def test_origin_private_ip_requires_same_port():
    assert APP._origin_is_trusted(_req(8765), "http://192.168.1.5:8765") is True
    assert APP._origin_is_trusted(_req(8765), "http://192.168.1.5:8080") is False
    assert APP._origin_is_trusted(_req(8765), "http://10.0.0.3:8765") is True
    assert APP._origin_is_trusted(_req(8765), "http://10.0.0.3:9999") is False


def test_origin_malformed_rejected():
    assert APP._origin_is_trusted(_req(), "not-a-url") is False
    assert APP._origin_is_trusted(_req(), "ftp://127.0.0.1") is False
    assert APP._origin_is_trusted(_req(), "https://") is False