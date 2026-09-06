"""巡检结论回溯（champion_history / training_history）与公式解读 单测。"""
from __future__ import annotations

import json

import pytest

from web.train_inspector import (
    append_champion_inspect_event,
    build_inspect_event,
    build_stop_notice,
    merge_inspections_into_history,
    verdict_recommends_stop,
)

ENTRY = {
    "ts": "2026-09-05T01:00:00+00:00",
    "symbol": "BTCUSDT",
    "level": "danger",
    "title": "建议停止：400 步无新最优",
    "recommendation": "停滞超过阈值，可停止本轮。",
    "checks": [
        {"level": "danger", "text": "最优已停滞 400 步。"},
        {"level": "warn", "text": "分布塌缩：有效词汇≈1.2。"},
    ],
    "metrics": {
        "step": 1200, "total": 9000, "pct": 13.3, "best": 2.9, "stall": 400,
        "restarts": 2, "champion": "2.9818", "eff_vocab_avg": 1.2,
        "huge_noise": "x" * 5000,  # 不应被带进事件
    },
}


def _sample_history_file(tmp_path, symbol="BTCUSDT") -> str:
    p = tmp_path / f"training_history_{symbol}.json"
    p.write_text(json.dumps({"step": [1, 2, 3], "best_score": [1.0, 2.0, 3.0]}), encoding="utf-8")
    return str(p)


# ── build_inspect_event ───────────────────────────────────────────────

def test_build_inspect_event_compact_and_drops_noise() -> None:
    ev = build_inspect_event(ENTRY)
    assert ev is not None
    assert ev["event"] == "train_inspect"
    assert ev["symbol"] == "BTCUSDT"
    assert ev["metrics"] == {
        "step": 1200, "total": 9000, "pct": 13.3, "best": 2.9, "stall": 400,
        "restarts": 2, "champion": "2.9818", "eff_vocab_avg": 1.2,
    }
    assert "huge_noise" not in ev["metrics"]
    assert ev["checks"] == ["最优已停滞 400 步。", "分布塌缩：有效词汇≈1.2。"]


def test_build_inspect_event_requires_symbol() -> None:
    assert build_inspect_event({"level": "info", "title": "无任务"}) is None


# ── champion_history 追加/裁剪 ─────────────────────────────────────────

def test_append_champion_preserves_deploy_and_caps_inspects(tmp_path) -> None:
    p = tmp_path / "champion_history.json"
    p.write_text(json.dumps([
        {"event": "deploy", "symbol": "BTCUSDT", "ts": "old"},
    ]), encoding="utf-8")
    for i in range(5):
        e = dict(ENTRY)
        e["ts"] = f"2026-09-05T01:00:0{i}+00:00"
        e["metrics"] = dict(ENTRY["metrics"], step=100 + i)
        assert append_champion_inspect_event(p, e, max_inspect=3)
    data = json.loads(p.read_text(encoding="utf-8"))
    inspects = [x for x in data if x.get("event") == "train_inspect"]
    deploys = [x for x in data if x.get("event") == "deploy"]
    assert len(inspects) == 3
    assert len(deploys) == 1
    assert [x["metrics"]["step"] for x in inspects] == [102, 103, 104]


def test_append_champion_no_symbol_is_noop(tmp_path) -> None:
    p = tmp_path / "champion_history.json"
    assert not append_champion_inspect_event(p, {"level": "info", "title": "无任务"})


def test_append_champion_creates_file(tmp_path) -> None:
    p = tmp_path / "champion_history.json"
    assert append_champion_inspect_event(p, ENTRY)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data[0]["event"] == "train_inspect"
    assert data[0]["symbol"] == "BTCUSDT"


# ── training_history merge ─────────────────────────────────────────────

def test_merge_keeps_curves_and_prepends_inspections(tmp_path) -> None:
    hp = _sample_history_file(tmp_path)
    assert merge_inspections_into_history(hp, [ENTRY], max_keep=10) == 1
    data = json.loads(tmp_path.read_text if False else open(hp, encoding="utf-8").read())
    # 原有曲线 key 保留
    assert data["step"] == [1, 2, 3]
    assert data["best_score"] == [1.0, 2.0, 3.0]
    assert data["inspections"][0]["title"] == "建议停止：400 步无新最优"


def test_merge_caps_and_dedups_empty(tmp_path) -> None:
    hp = _sample_history_file(tmp_path)
    merge_inspections_into_history(hp, [ENTRY] * 12, max_keep=5)
    data = json.loads(open(hp, encoding="utf-8").read())
    assert len(data["inspections"]) == 5
    assert data["step"] == [1, 2, 3]
    # 空列表 no-op
    assert merge_inspections_into_history(hp, []) == 0


# ── 自动停止判定 / 文案 ───────────────────────────────────────────────

def test_verdict_recommends_stop() -> None:
    assert verdict_recommends_stop({"level": "danger", "title": "建议停止：400 步零提升"})
    assert not verdict_recommends_stop({"level": "danger", "title": "运行正常"})
    assert not verdict_recommends_stop({"level": "warn", "title": "建议停止：xxx"})
    assert not verdict_recommends_stop(None)
    assert not verdict_recommends_stop({})


def test_build_stop_notice_lines() -> None:
    text = build_stop_notice(ENTRY, "BTCUSDT")
    assert "建议停止" in text
    assert "步 1200/9000" in text
    assert "最优已停滞 400 步" in text
    assert "可停止本轮" in text
