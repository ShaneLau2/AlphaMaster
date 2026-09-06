"""Alias routing for openclaw / openclaw_wb auto-detect."""
from __future__ import annotations

from web.ai_providers import _alias_provider_from_key


def test_openclaw_wb_alias_wins_over_openclaw_prefix() -> None:
    assert _alias_provider_from_key("openclaw_wb") == "openclaw_wb"
    assert _alias_provider_from_key("openclaw_wb/auto") == "openclaw_wb"
    assert _alias_provider_from_key("OpenClaw_WB") == "openclaw_wb"


def test_openclaw_alias() -> None:
    assert _alias_provider_from_key("openclaw") == "openclaw"
    assert _alias_provider_from_key("openclaw/main") == "openclaw"
    assert _alias_provider_from_key("deepseek-key") is None
    assert _alias_provider_from_key("") is None
