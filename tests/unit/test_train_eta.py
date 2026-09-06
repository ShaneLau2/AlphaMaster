"""训练 ETA 估算单测：回归斜率 / 先验回退 / 剩余与预计结束 / 可信度分级。"""
from __future__ import annotations

from datetime import datetime, timezone

from web.train_eta import eta_from_samples, sample_regression


NOW = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


def test_regression_linear_pace() -> None:
    # 每 50 秒推进 5 步 → 10 s/步，线性完美 r²=1
    pace, r2 = sample_regression([(0.0, 100), (50.0, 105), (100.0, 110)])
    assert pace is not None and abs(pace - 10.0) < 1e-6
    assert r2 is not None and r2 > 0.999


def test_regression_rejects_stalled() -> None:
    # 步数无推进 → 无法估计 pace
    pace, r2 = sample_regression([(0.0, 100), (50.0, 100), (100.0, 100)])
    assert pace is None and r2 is None
    assert sample_regression([(0.0, 100)]) == (None, None)


def test_eta_session_regression() -> None:
    # 近期样本：50s 内 5 步 → 10 s/步；当前在 step 110，总 9000
    out = eta_from_samples(
        step=110,
        elapsed_seconds=100.0,
        train_steps=9000,
        samples=[(NOW.timestamp() - 100, 100), (NOW.timestamp() - 50, 105), (NOW.timestamp(), 110)],
        now=NOW,
    )
    assert out is not None
    assert out["pace_source"] == "session_regression"
    assert abs(out["seconds_per_step"] - 10.0) < 0.5
    assert out["remaining_steps"] == 9000 - 110
    assert abs(out["remaining_seconds"] - (9000 - 110) * 10) < 200
    assert out["estimated_finish_local"].endswith("2026") or "-" in out["estimated_finish_local"]


def test_eta_history_prior_fallback() -> None:
    # 无会话样本：已训 900 步累计耗时 9000s → 先验 10 s/步；剩 8100 步 ≈ 81000s
    out = eta_from_samples(
        step=900,
        elapsed_seconds=None,
        train_steps=9000,
        samples=[],
        history_total_seconds=9000.0,
        now=NOW,
    )
    assert out is not None
    assert out["pace_source"] == "history_prior"
    assert abs(out["seconds_per_step"] - 10.0) < 1e-6
    assert out["remaining_steps"] == 8100
    assert abs(out["remaining_seconds"] - 81000) < 1e-6


def test_eta_confidence_grades() -> None:
    low = eta_from_samples(
        step=110, elapsed_seconds=100.0, train_steps=9000,
        samples=[(NOW.timestamp() - 100, 100), (NOW.timestamp(), 110)], now=NOW,
    )
    assert low is not None and low["confidence"] == "low"
    many = eta_from_samples(
        step=110, elapsed_seconds=100.0, train_steps=9000,
        samples=[(NOW.timestamp() - 100 + i * 5, 100 + i * 2) for i in range(11)],
        now=NOW,
    )
    assert many is not None and many["confidence"] == "high"


def test_eta_none_without_data() -> None:
    assert eta_from_samples(step=None, elapsed_seconds=None) is None
    assert eta_from_samples(step=50, elapsed_seconds=None, samples=[], history_total_seconds=None) is None
