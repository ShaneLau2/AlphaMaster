"""OOS 溯源分类单测：训练溯源推断 + 窗口 真伪样本外 分桶与建议。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from web.oos_provenance import (classify_window, holdout_bars_for,
                                oos_status_label, strategy_provenance)


def _write_parquet(path, n: int, start_ts: int, step_s: int = 300) -> None:
    ts = np.arange(start_ts, start_ts + n * step_s, step_s, dtype=np.int64)[:n]
    df = pd.DataFrame({
        "timestamp": ts,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_strategy(path, data_file: str, n_bars: int, holdout: int | None = None):
    strat = {
        "symbol": "BTCUSDT",
        "formula": [1, 2, 3],
        "data_source": {
            "data_file": str(data_file),
            "bars": n_bars,
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-01-01T00:00:00+00:00",
            "source": "binance",
        },
        "train_range": {
            "data_file": str(data_file),
            "n_bars": n_bars,
            "mode": "tail",
            "holdout_bars": holdout,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strat), encoding="utf-8")


def test_holdout_bars_for_deterministic() -> None:
    from model_core.config import ModelConfig

    assert holdout_bars_for(40_000) == ModelConfig.HOLDOUT_BARS  # 500（cap 5000 不触发）
    # 小数据集：cap T//8 压缩
    assert holdout_bars_for(800) == 100           # min(500, 800//8)
    assert holdout_bars_for(1000) == 125          # min(500, 1000//8)
    assert 0 <= holdout_bars_for(100) <= 100


def test_strategy_provenance_infers_and_reads(tmp_path) -> None:
    data = tmp_path / "BTCUSDT_M5.parquet"
    _write_parquet(data, 40_000, start_ts=1_700_000_000)
    strat = tmp_path / "best_BTCUSDT.json"
    _make_strategy(strat, data, 40_000)
    prov = strategy_provenance(strat)
    assert prov is not None
    assert prov["n_bars"] == 40_000
    assert prov["holdout_bars"] == 500
    assert prov["source"] == "train_range"
    assert prov["holdout_start_ts"] == pytest.approx(1_700_000_000 + (40_000 - 500) * 300)
    assert prov["train_end_ts"] == pytest.approx(1_700_000_000 + (40_000 - 1) * 300)
    # 文件不存在/坏 JSON → None
    assert strategy_provenance(tmp_path / "nope.json") is None


def test_classify_window_partial_and_honest(tmp_path) -> None:
    """窗口同时含训练集内 + holdout → partial，且建议=真 holdout 尾部。"""
    data = tmp_path / "BTCUSDT_M5.parquet"
    n, hb = 40_000, 500
    _write_parquet(data, n, start_ts=1_700_000_000)
    strat = tmp_path / "best_BTCUSDT.json"
    _make_strategy(strat, data, n, holdout=hb)
    # 该 champion 的「15000 样本外」实际 = 14500 训练内 + 500 holdout
    oos = classify_window(strat, data, window_start=25_000, window_bars=15_000)
    assert oos["status"] == "partial"
    assert oos["n_in_sample"] == 14_500
    assert oos["n_holdout"] == 500
    assert oos["n_post_train"] == 0
    h = oos["honest"]
    assert h is not None and h["kind"] == "holdout-tail"
    assert (h["start_bar"], h["n_bars"]) == (39_500, 500)
    assert "部分样本内" in oos_status_label(oos)


def test_classify_pure_in_sample_and_pure_holdout(tmp_path) -> None:
    data = tmp_path / "BTCUSDT_M5.parquet"
    n, hb = 40_000, 500
    _write_parquet(data, n, start_ts=1_700_000_000)
    strat = tmp_path / "best_BTCUSDT.json"
    _make_strategy(strat, data, n, holdout=hb)
    assert classify_window(strat, data, 0, 5000)["status"] == "in-sample"
    oos = classify_window(strat, data, n - hb, hb)
    assert oos["status"] == "oos-holdout"
    assert oos["n_holdout"] == hb
    assert "holdout" in oos_status_label(oos)


def test_classify_post_train_new_data(tmp_path) -> None:
    """run 文件里含训练截止之后的 bar → oos-new / mixed + post-train 建议。"""
    data = tmp_path / "train.parquet"
    n, hb = 10_000, 500
    _write_parquet(data, n, start_ts=1_700_000_000)
    strat = tmp_path / "best_X.json"
    _make_strategy(strat, data, n, holdout=hb)
    # run 文件 = 训练文件 + 尾部多 200 根新 bar（时间轴连续）
    run = tmp_path / "concat.parquet"
    end_ts = 1_700_000_000 + (n - 1) * 300
    _write_parquet(run, n + 200, start_ts=1_700_000_000)
    oos = classify_window(strat, run, window_start=n - 100, window_bars=300)
    # 100 根样本内 + 200 根新 → mixed
    assert oos["status"] == "mixed"
    assert oos["n_post_train"] == 200
    assert oos["honest"]["kind"] == "post-train"
    # 只取新 bar → oos-new
    oos2 = classify_window(strat, run, window_start=n + 100, window_bars=100)
    assert oos2["status"] == "oos-new"
    assert oos2["n_post_train"] == 100
    assert end_ts < 1_700_000_000 + (n + 200 - 1) * 300  # run 确有更新 bar


def test_classify_missing_provenance(tmp_path) -> None:
    data = tmp_path / "BTCUSDT_M5.parquet"
    _write_parquet(data, 1000, start_ts=1_700_000_000)
    assert classify_window(None, data, 0, 500)["status"] == "unavailable"
    assert classify_window(tmp_path / "x.json", data, 0, 500)["status"] == "unavailable"
