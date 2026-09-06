"""inspect_parquet_file 轻量读取路径（footer 元数据 + time 单列）单测。

背景：/api/data-file/browse 对大数据文件曾因全量 pd.read_parquet 耗时过长，
被前端 30s AbortController 中止（报“signal is aborted without reason”）。
轻量路径必须与旧全量口径逐字段一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from data_pipeline.parquet_manager import inspect_parquet_file  # noqa: E402


def _write(path: Path, df: pd.DataFrame) -> Path:
    df.to_parquet(path, index=False)
    return path


def test_light_path_matches_full_read_parity(tmp_path: Path) -> None:
    """同一文件：轻量路径与全量读算出的指标逐项一致（重复时间戳/缺列/区间/根数）。"""
    p = _write(tmp_path / "TESTBTC_M5.parquet", pd.DataFrame({
        "time": [1700000000 + i * 300 for i in range(1200)] + [1700000000],
        "open": list(range(1201)), "high": list(range(1201)),
        "low": list(range(1201)), "close": list(range(1201)),
    }))
    info = inspect_parquet_file(p)
    # 全量读对照
    df = pd.read_parquet(p)
    assert info["bars"] == len(df) == 1201
    assert info["bars_unique"] == int(df["time"].nunique()) == 1200
    assert info["duplicate_ts"] == 1
    assert info["missing_columns"] == ["volume"]
    assert info["start_date"] == "2023-11-14"
    assert info["end_date"] == "2023-11-19"
    assert info["valid"] is True and info["checks_ok"] is False  # 缺 volume 列


def test_light_path_with_volume_and_no_dup(tmp_path: Path) -> None:
    p = _write(tmp_path / "ETHUSD_M5.parquet", pd.DataFrame({
        "time": [1700000000 + i * 300 for i in range(1000)],
        "open": list(range(1000)), "high": list(range(1000)),
        "low": list(range(1000)), "close": list(range(1000)),
        "volume": list(range(1000)),
    }))
    info = inspect_parquet_file(p)
    assert info["duplicate_ts"] == 0
    assert info["missing_columns"] == []
    assert info["checks_ok"] is True


def test_light_path_missing_time_column(tmp_path: Path) -> None:
    """无 time 列：行数/列名来自元数据，缺列清单必须包含 time。"""
    p = _write(tmp_path / "XAUUSD_M5.parquet", pd.DataFrame({
        "open": list(range(900)), "close": list(range(900)),
    }))
    info = inspect_parquet_file(p)
    assert info["bars"] == 900
    assert "time" in info["missing_columns"]
    assert info["checks_ok"] is False