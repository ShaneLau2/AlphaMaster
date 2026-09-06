"""Custom OpenAI-compatible base_url / model resolution."""
from __future__ import annotations

import pytest

from web.ai_providers import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    _clamp_max_tokens,
    _normalize_base_url,
    resolve_provider,
)


def test_normalize_base_url_defaults_and_strips() -> None:
    assert _normalize_base_url(None) == DEEPSEEK_BASE_URL
    assert _normalize_base_url("") == DEEPSEEK_BASE_URL
    assert _normalize_base_url("https://token.sensenova.cn/v1/") == (
        "https://token.sensenova.cn/v1"
    )


def test_normalize_base_url_rejects_non_http() -> None:
    with pytest.raises(ValueError, match="http://"):
        _normalize_base_url("token.sensenova.cn/v1")


def test_resolve_provider_custom_sensenova_url() -> None:
    resolved = resolve_provider(
        "deepseek",
        "sk-test-key",
        base_url="https://token.sensenova.cn/v1",
        model="deepseek-v4-flash",
    )
    assert resolved.provider == "deepseek"
    assert resolved.base_url == "https://token.sensenova.cn/v1"
    assert resolved.model == "deepseek-v4-flash"
    assert resolved.label == "SenseNova"
    assert resolved.api_key == "sk-test-key"


def test_resolve_provider_defaults_when_url_model_empty() -> None:
    resolved = resolve_provider("deepseek", "sk-test-key", base_url="", model="")
    assert resolved.base_url == DEEPSEEK_BASE_URL
    assert resolved.model == DEEPSEEK_MODEL
    assert resolved.label == "DeepSeek"


def test_clamp_max_tokens_sensenova() -> None:
    url = "https://token.sensenova.cn/v1"
    assert _clamp_max_tokens(url, "deepseek-v4-flash", 4096) == 4096
    assert _clamp_max_tokens(url, "deepseek-v4-flash", 999_999) == 65_536
    assert _clamp_max_tokens(url, "glm-5.2", 999_999) == 131_072
    assert _clamp_max_tokens(DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, 999_999) == 384_000


def test_clamp_max_tokens_global_cap() -> None:
    assert _clamp_max_tokens("https://api.example-proxy.com/v1", "gpt-4", 999_999) == 384_000
