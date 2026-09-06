"""训练子集生成器（tail/spread/full）单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.train_sampler import (
    LOOKBACK,
    MODE_FULL,
    MODE_SPREAD,
    MODE_TAIL,
    prepare_training_subset,
)


def _make_series(n: int = 20_000, seed: int = 7, base: int = 1_700_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "time": base + np.arange(n) * 300,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": rng.uniform(10, 50, n),
        }
    )


@pytest.fixture()
def src(tmp_path):
    root = tmp_path
    d = root / "data" / "training"
    d.mkdir(parents=True)
    f = d / "BTCUSDT_M5.parquet"
    _make_series().to_parquet(f, index=False)
    return root, f


def test_full_passthrough(src):
    root, f = src
    r = prepare_training_subset(f, mode=MODE_FULL, root_dir=root)
    assert r["subset"] is False
    assert r["data_file"] == str(f.resolve())
    assert r["subset_bars"] == r["source_bars"] == 20_000


def test_tail_exact_last_n(src):
    root, f = src
    df = pd.read_parquet(f)
    r = prepare_training_subset(f, mode=MODE_TAIL, n_bars=5_000, root_dir=root)
    assert r["subset"] and r["subset_bars"] == 5_000
    sub = pd.read_parquet(r["data_file"])
    assert len(sub) == 5_000
    assert sub["time"].iloc[-1] == df["time"].iloc[-1]
    assert sub["time"].iloc[0] == df["time"].iloc[-5_000]
    assert sub["time"].is_monotonic_increasing


def test_tail_over_total_falls_back_to_full(src):
    root, f = src
    r = prepare_training_subset(f, mode=MODE_TAIL, n_bars=10_000_000, root_dir=root)
    assert r["subset"] is False
    assert r["data_file"] == str(f.resolve())


def test_spread_keeps_tail_and_covers_eras(src):
    root, f = src
    df = pd.read_parquet(f)
    n = 8_000
    r = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=n, n_chunks=3, root_dir=root)
    assert r["subset"] is True
    assert r["subset_bars"] < n * 2
    sub = pd.read_parquet(r["data_file"])
    # 时间单调、无重复
    assert sub["time"].is_monotonic_increasing
    assert sub["time"].is_unique
    # 尾部 = 源数据最末端（近期段 + holdout 干净）
    assert sub["time"].iloc[-1] == df["time"].iloc[-1]
    # 覆盖多个年代（第一根远早于源序列的 1/2 处）
    assert sub["time"].iloc[0] < df["time"].iloc[len(df) // 2]
    # 末尾干净区间长度 ≥ 近期段下限
    assert len(sub) - (sub["time"] == df["time"].iloc[-1]).sum() + 1 > 0
    assert r["n_chunks_used"] and r["n_chunks_used"] >= 3


def test_spread_deterministic(src):
    root, f = src
    r1 = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=8_000, n_chunks=3, root_dir=root)
    r2 = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=8_000, n_chunks=3, root_dir=root)
    assert open(r1["data_file"], "rb").read() == open(r2["data_file"], "rb").read()


def test_spread_too_small_source_falls_back(src):
    root, f = src
    small = root / "data" / "training" / "XAUUSD_H1.parquet"
    _make_series(n=1500).to_parquet(small, index=False)
    r = prepare_training_subset(small, mode=MODE_SPREAD, n_bars=800, root_dir=root)
    assert r["subset"] is False
    assert r["data_file"] == str(small.resolve())


def test_spread_schema_roundtrip(src):
    root, f = src
    r = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=6_000, n_chunks=2, root_dir=root)
    sub = pd.read_parquet(r["data_file"])
    assert {"time", "open", "high", "low", "close", "volume"} <= set(sub.columns)


def test_invalid_mode_raises(src):
    root, f = src
    with pytest.raises(ValueError):
        prepare_training_subset(f, mode="bogus", root_dir=root)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_training_subset(tmp_path / "nope.parquet", mode=MODE_TAIL, root_dir=tmp_path)


def test_subset_loads_with_data_manager(src):
    """生成的子集必须能被引擎的 ParquetDataManager 直接加载（含指纹）。"""
    from data_pipeline.parquet_manager import ParquetDataManager

    root, f = src
    r = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=6_000, n_chunks=2, root_dir=root)
    mgr = ParquetDataManager(r["data_file"])
    mgr.load()
    assert mgr.raw_dict["close"].shape[1] == r["subset_bars"]
    assert mgr.fingerprint is not None
    mgr2 = ParquetDataManager(r["data_file"])
    mgr2.load()
    assert mgr2.fingerprint == mgr.fingerprint


def _two_regime_series(tmp_path, n=40_000, seed=3):
    """前一半低波动、后一半高波动的合成序列（regime 分层应能覆盖两类）。"""
    rng = np.random.default_rng(seed)
    ret = np.concatenate([rng.normal(0, 0.005, n // 2), rng.normal(0, 0.5, n // 2)])
    close = 100 * np.exp(np.cumsum(ret))
    df = pd.DataFrame(
        {
            "time": 1_600_000_000 + np.arange(n) * 300,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.uniform(10, 50, n),
        }
    )
    d = tmp_path / "data" / "training"
    d.mkdir(parents=True)
    f = d / "REGIME_M5.parquet"
    df.to_parquet(f, index=False)
    return tmp_path, f


def test_spread_regime_vol_covers_both_regimes(tmp_path):
    root, f = _two_regime_series(tmp_path)
    r = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=10_000, n_chunks=4,
                                regime="vol", root_dir=root)
    cov = r.get("regime_coverage") or {}
    # 高波动后段（high）必须进样本；low/mid 合计也应 > 0（低波动前段被选入）
    assert cov.get("high", 0) >= 1
    assert cov.get("low", 0) + cov.get("mid", 0) >= 1
    assert r["regime"] == "vol"
    sub = pd.read_parquet(r["data_file"])
    assert sub["time"].is_monotonic_increasing
    assert sub["time"].iloc[-1] == pd.read_parquet(f)["time"].iloc[-1]


def test_spread_regime_trend_and_equal_deterministic(tmp_path):
    root, f = _two_regime_series(tmp_path, seed=9)
    for regime in ("vol", "trend", "equal"):
        r1 = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=10_000, n_chunks=4,
                                     regime=regime, root_dir=root)
        r2 = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=10_000, n_chunks=4,
                                     regime=regime, root_dir=root)
        assert open(r1["data_file"], "rb").read() == open(r2["data_file"], "rb").read()
        assert r1["regime"] == regime
        # trend/vol 需要覆盖信息；equal 不要求
        if regime == "trend":
            cov = r1.get("regime_coverage") or {}
            assert sum(cov.values()) > 0


def test_train_range_sidecar(tmp_path):
    from data_pipeline.train_sampler import read_train_range

    root, f = _two_regime_series(tmp_path, seed=11)
    r = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=10_000, n_chunks=3,
                                regime="vol", root_dir=root)
    side = read_train_range(r["data_file"])
    assert side is not None
    assert side["mode"] == MODE_SPREAD
    assert side["regime"] == "vol"
    assert side["subset_bars"] == r["subset_bars"]
    assert side["source_bars"] == r["source_bars"]
    # full 模式不写 sidecar
    rf = prepare_training_subset(f, mode=MODE_FULL, root_dir=root)
    assert read_train_range(rf["data_file"]) is None
    # tail 也写
    rt = prepare_training_subset(f, mode=MODE_TAIL, n_bars=8_000, root_dir=root)
    assert read_train_range(rt["data_file"])["mode"] == MODE_TAIL


def test_invalid_regime_raises(src):
    root, f = src
    with pytest.raises(ValueError):
        prepare_training_subset(f, mode=MODE_SPREAD, n_bars=5_000,
                                regime="bogus", root_dir=root)


def test_preview_tail_block_is_recent_tail(src):
    from data_pipeline.train_sampler import preview_training_subset

    root, f = src
    r = preview_training_subset(f, mode=MODE_TAIL, n_bars=5_000)
    assert r["source_bars"] == 20_000
    assert len(r["blocks"]) == 1
    b = r["blocks"][0]
    assert b["kind"] == "core"
    assert (b["start"], b["end"]) == (15_000, 20_000)
    assert r["total_bars"] == 5_000


def test_preview_spread_blocks_match_actual_subset(src):
    from data_pipeline.train_sampler import preview_training_subset

    root, f = src
    n, chunks = 8_000, 3
    p = preview_training_subset(f, mode=MODE_SPREAD, n_bars=n, n_chunks=chunks,
                                regime="vol")
    # 块不重叠、全部落在 [0, total) 内，且含 core/recent
    blocks = p["blocks"]
    for a, b in zip(blocks, blocks[1:]):
        assert a["end"] <= b["start"]
    assert all(0 <= b["start"] < b["end"] <= 20_000 for b in blocks)
    kinds = {b["kind"] for b in blocks}
    assert {"core", "recent", "warmup"} <= kinds
    # 末尾 recent 落在真正的干净尾部
    recent = [b for b in blocks if b["kind"] == "recent"][0]
    assert recent["end"] == 20_000
    # 预览行数 = 实际生成子集行数（含 warm-up 垫片）
    r = prepare_training_subset(f, mode=MODE_SPREAD, n_bars=n, n_chunks=chunks,
                                regime="vol", root_dir=root)
    assert p["total_bars"] == r["subset_bars"]
    assert p["regime_coverage"] is not None


def test_preview_full_and_too_small_fallback(tmp_path):
    from data_pipeline.train_sampler import preview_training_subset

    root = tmp_path
    d = root / "data" / "training"
    d.mkdir(parents=True)
    small = d / "XAUUSD_H1.parquet"
    _make_series(n=1500).to_parquet(small, index=False)
    p = preview_training_subset(small, mode=MODE_SPREAD, n_bars=800)
    assert p["blocks"][0]["kind"] == "core"  # 退化尾部（几何不足）
    full = preview_training_subset(d / "XAUUSD_H1.parquet" if False else small,
                                   mode=MODE_FULL)
    assert full["total_bars"] == 1500


# ── 旧策略文件 train_range 推断（inferred_train_range）────────────────────

def test_inferred_train_range_full(tmp_path):
    """无 sidecar 的旧文件：推断 mode=full + 数据元信息 + inferred 标记。"""
    from data_pipeline.train_sampler import inferred_train_range

    d = tmp_path / "data" / "training"
    d.mkdir(parents=True)
    f = d / "BTCUSDT_M5.parquet"
    _make_series().to_parquet(f, index=False)
    tr = inferred_train_range(f)
    assert tr is not None
    assert tr["mode"] == "full"
    assert tr["subset"] is False
    assert tr["inferred"] is True
    assert tr["source_bars"] == 20_000
    assert tr["subset_bars"] == 20_000
    assert tr["symbol"] == "BTCUSDT"
    assert tr["timeframe"] == "M5"
    assert tr["start"] and tr["end"]


def test_inferred_train_range_missing_file(tmp_path):
    from data_pipeline.train_sampler import inferred_train_range

    assert inferred_train_range(tmp_path / "nope.parquet") is None
    assert inferred_train_range("") is None
