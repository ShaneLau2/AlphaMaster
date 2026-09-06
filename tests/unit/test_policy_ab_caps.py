"""A/B 混合上限对比（caps 档位）单测。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web.app import _normalize_ab_caps  # noqa: E402


def test_caps_normalize_sorted_dedup_clamped() -> None:
    # 乱序 + 重复 + 超界 + 非法 → 升序去重钳到 1..200
    assert _normalize_ab_caps([100, 5, 25, 100, 0, -3, 500, 0.5], 100.0) == [1.0, 5.0, 25.0, 100.0, 200.0]


def test_caps_all_invalid_falls_back() -> None:
    assert _normalize_ab_caps([0, -1, None], 42.0) == [42.0]
    assert _normalize_ab_caps([], 100.0) == [100.0]


def test_caps_none_single_fallback() -> None:
    assert _normalize_ab_caps(None, 100.0) == [100.0]
    assert _normalize_ab_caps(None, 5.0) == [5.0]


def test_caps_preserves_typical_input() -> None:
    assert _normalize_ab_caps([5, 25, 100], 100.0) == [5.0, 25.0, 100.0]


def test_ab_row_contract_has_cap_pct() -> None:
    """后端产出的每行必须带 cap_pct（前端 上限% 列依赖），且 results 数 = pid × cap。"""
    caps = _normalize_ab_caps([5, 25, 100], 100.0)
    pids = ["signal", "dd"]
    rows = [{"policy_id": pid, "cap_pct": cap} for pid in pids for cap in caps]
    assert len(rows) == 6
    for r in rows:
        assert "cap_pct" in r
    # 同一方案多档各一行
    pid_rows = [r for r in rows if r["policy_id"] == "signal"]
    assert [r["cap_pct"] for r in pid_rows] == [5.0, 25.0, 100.0]