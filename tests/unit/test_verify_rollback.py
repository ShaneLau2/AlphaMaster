"""scripts/verify_champion_rollback 单测：记录解析 / 子集确定性 / rescore 判分。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_spec = importlib.util.spec_from_file_location(
    "vrb_mod", Path(__file__).resolve().parents[2] / "scripts" / "verify_champion_rollback.py"
)
vrb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vrb)  # type: ignore[union-attr]


def _make_series(n: int = 20_000, seed: int = 7, base: int = 1_700_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "time": base + np.arange(n) * 300,
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": rng.uniform(10, 50, n),
    })


# ── 记录解析 ──────────────────────────────────────────────────────────────

def _write_src(tmp_path: Path, n: int = 20_000) -> Path:
    d = tmp_path / "data" / "training"
    d.mkdir(parents=True)
    f = d / "BTCUSDT_M5.parquet"
    _make_series(n).to_parquet(f, index=False)
    return f


def test_record_from_sidecar(tmp_path):
    src = _write_src(tmp_path)
    from data_pipeline.train_sampler import prepare_training_subset

    r = prepare_training_subset(src, mode="tail", n_bars=5_000, root_dir=tmp_path)
    side = Path(r["data_file"]).parent / "train_range.json"
    rec = vrb.record_from_sidecar(side)
    assert rec and rec["kind"] == "subset"
    assert rec["mode"] == "tail"
    assert rec["n_bars"] == 5_000
    assert rec["recorded_subset_bars"] == 5_000
    assert rec["recorded_file"] and Path(rec["recorded_file"]).exists()
    assert rec["archived"]["formula"] is None


def test_record_from_compare_dict_shape(tmp_path):
    """variant 为 {best, holdout_passed, runs} 的标准对比输出格式。"""
    p = tmp_path / "compare_X.json"
    doc = {
        "source_file": "/x/BTCUSDT_M5.parquet",
        "params": {"n_bars": 15000, "chunks": None, "steps": 10, "seeds": [42, 7],
                   "regime": "vol"},
        "variants": {
            "tail": {
                "best": {"tag": "tail_s42", "seed": 42, "symbol": "BTCUSDT",
                         "best_score": 2.1634, "best_formula": [56, 105],
                         "source_bars": 15000, "holdout_bars": 500,
                         "holdout": {"val_score": -0.369, "passed": False}},
                "runs": [],
            },
        },
    }
    p.write_text(json.dumps(doc), encoding="utf-8")
    recs = vrb.record_from_compare(p)
    assert len(recs) == 1
    r = recs[0]
    assert r["kind"] == "champion" and r["label"] == "compare:tail"
    assert r["archived"]["formula"] == [56, 105]
    assert r["archived"]["best"] == 2.1634
    assert r["archived"]["holdout"]["val_score"] == -0.369
    assert r["recorded_file"] is None  # 对比子集通常已清理
    assert r["mode"] == "tail"


def test_record_from_compare_list_shape(tmp_path):
    """variant 为裸 run 列表的旧格式兜底。"""
    p = tmp_path / "compare_Y.json"
    doc = {
        "source_file": "/x/BTCUSDT_M5.parquet",
        "params": {"n_bars": 15000},
        "variants": {"spread": [{"tag": "spread_s7", "seed": 7, "symbol": "BTCUSDT",
                                 "best_score": 0.9, "best_formula": [3, 4],
                                 "holdout": {"val_score": 1.2, "passed": True}}]},
    }
    p.write_text(json.dumps(doc), encoding="utf-8")
    recs = vrb.record_from_compare(p)
    assert len(recs) == 1 and recs[0]["mode"] == "spread"


def test_record_from_strategy_with_train_range(tmp_path):
    p = tmp_path / "best_BTCUSDT.json"
    p.write_text(json.dumps({
        "symbol": "BTCUSDT", "formula": [1, 2, 3], "best_score": 2.98,
        "train_range": {"mode": "spread", "data_file": "/gone/spread_x/BTCUSDT_M5.parquet",
                        "source_file": "/src/BTCUSDT_M5.parquet",
                        "n_bars_requested": 100000, "regime": "vol", "subset_bars": 110000},
        "holdout": {"val_score": 1.5, "passed": True},
    }), encoding="utf-8")
    r = vrb.record_from_strategy(p)
    assert r and r["kind"] == "champion"
    assert r["mode"] == "spread"
    assert r["recorded_file"] is None          # 文件不存在
    assert r["recorded_subset_bars"] == 110000
    assert r["archived"]["formula"] == [1, 2, 3]
    assert r["archived"]["holdout"]["passed"] is True


def test_record_from_strategy_full_fallback(tmp_path):
    """无 train_range 的旧冠军 → mode=full + data_file 兜底。"""
    src = _write_src(tmp_path)
    p = tmp_path / "best_ZZZ.json"
    p.write_text(json.dumps({"symbol": "ZZZ", "formula": [9], "best_score": 1.2,
                             "data_file": str(src)}), encoding="utf-8")
    r = vrb.record_from_strategy(p)
    assert r and r["mode"] == "full"
    assert r["recorded_file"] == str(src)


# ── 子集确定性 ────────────────────────────────────────────────────────────

def _build_subset_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    """建一次 tail 子集（记录 + sidecar），再独立重生成到第二个 root。"""
    src = _write_src(tmp_path)
    from data_pipeline.train_sampler import prepare_training_subset

    rec = prepare_training_subset(src, mode="tail", n_bars=5_000, root_dir=tmp_path)
    side = Path(rec["data_file"]).parent / "train_range.json"
    # 独立重生成（相同参数、不同 root）→ 应产出相同内容
    root2 = tmp_path / "regen"
    r2 = prepare_training_subset(src, mode="tail", n_bars=5_000, root_dir=root2)
    return side, Path(rec["data_file"]), Path(r2["data_file"])


def test_determinism_pass_on_identical_regen(tmp_path):
    side, rec_file, regen_file = _build_subset_pair(tmp_path)
    rec = vrb.record_from_sidecar(side)
    rec["recorded_file"] = str(rec_file)
    got = vrb.check_determinism(rec, {"data_file": str(regen_file), "subset_bars": 5000})
    assert got["status"] == "PASS"
    assert got["max_close_delta"] == 0.0


def test_determinism_fail_on_tampered_file(tmp_path):
    side, rec_file, regen_file = _build_subset_pair(tmp_path)
    df = pd.read_parquet(regen_file)
    df.loc[df.index[-1], "close"] = df["close"].iloc[-1] + 1.0
    df.to_parquet(regen_file, index=False)
    rec = vrb.record_from_sidecar(side)
    rec["recorded_file"] = str(rec_file)
    got = vrb.check_determinism(rec, {"data_file": str(regen_file), "subset_bars": 5000})
    assert got["status"] == "FAIL"


def test_determinism_bar_parity_when_record_file_deleted(tmp_path):
    side, rec_file, _ = _build_subset_pair(tmp_path)
    rec = vrb.record_from_sidecar(side)
    rec["recorded_file"] = None            # 模拟对比实验子集已清理
    got = vrb.check_determinism(rec, {"data_file": str(rec_file), "subset_bars": 5000})
    assert got["status"] == "PASS"
    assert got["bars_regen"] == got["bars_recorded"] == 5000


# ── rescore 判分 ──────────────────────────────────────────────────────────

def _ho(val, sharpe, passed):
    return {"val_score": val, "sharpe": sharpe, "score_ratio": val / 2.0,
            "passed": passed}


def test_grade_rescore_pass_within_tol():
    rec = {"archived": {"holdout": _ho(2.962, 8.188, True)}}
    got = vrb.grade_rescore(rec, _ho(2.9605, 8.188, True), tol=0.02)
    assert got["status"] == "PASS"


def test_grade_rescore_fail_beyond_tol():
    rec = {"archived": {"holdout": _ho(2.962, 8.188, True)}}
    got = vrb.grade_rescore(rec, _ho(1.2, 0.5, False), tol=0.02)
    assert got["status"] == "FAIL"


def test_grade_rescore_info_without_archive():
    rec = {"archived": {"holdout": None}}
    got = vrb.grade_rescore(rec, _ho(0.88, 1.1, True), tol=0.02)
    assert got["status"] == "info"
    assert "现状读数" in got["note"]
