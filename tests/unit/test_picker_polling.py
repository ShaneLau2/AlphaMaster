"""文件/策略选择器的会话轮询契约验证（不弹原生对话框）。

数据文件与策略选择器都走同一套非阻塞会话制：
POST /browse → {dialog, session}；GET /browse-poll?session= → done 后返回完整检查信息。
本测试直接调用路由函数，monkeypatch poll_picker 模拟对话框完成/取消，
并验证数据文件检查摘要（去重根数/缺列/重复时间戳/时间跨度）随选中结果返回。
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

import web.app as app_mod
import web.settings as settings_mod
from model_core.vocab import VOCAB_VERSION


def _write_parquet(tmp: Path, name: str, n: int = 400) -> Path:
    base = int(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    df = pd.DataFrame(
        {
            "time": [base + i * 3600 for i in range(n)],
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [1] * n,
        }
    )
    p = tmp / name
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def iso(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把 settings 落盘重定向到临时目录，避免测试污染仓库 web_settings.json。"""
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "web_settings.json")
    return tmp_path


# ── 数据文件选择器 ─────────────────────────────────────────────────────────

def test_data_file_poll_done_returns_inspect_summary(iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq = _write_parquet(iso, "BTCUSDT_H1.parquet", 400)
    monkeypatch.setattr(app_mod, "poll_picker",
                        lambda sid: {"done": True, "path": str(pq), "error": None})
    res = app_mod.api_browse_data_file_poll(session="s1")
    assert res["done"] is True and res["cancelled"] is False
    assert res["symbol"] == "BTCUSDT" and res["timeframe"] == "H1"
    assert res["bars_unique"] == 400 and res["duplicate_ts"] == 0
    assert res["missing_columns"] == []
    assert res["start_date"] and res["end_date"]
    assert res["checks_ok"] is True


def test_data_file_poll_reports_duplicates_and_missing_columns(iso: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    pq = _write_parquet(iso, "AAA_H1.parquet", 400)
    df = pd.read_parquet(pq)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True).drop(columns=["volume"])
    dup.to_parquet(pq, index=False)
    monkeypatch.setattr(app_mod, "poll_picker",
                        lambda sid: {"done": True, "path": str(pq), "error": None})
    res = app_mod.api_browse_data_file_poll(session="s2")
    assert res["duplicate_ts"] == 1
    assert res["bars_unique"] == 400 and res["raw_bars"] == 401
    assert res["missing_columns"] == ["volume"]
    assert res["checks_ok"] is False


def test_data_file_poll_cancel(iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_mod, "poll_picker",
                        lambda sid: {"done": True, "path": None, "error": None})
    res = app_mod.api_browse_data_file_poll(session="s3")
    assert res["done"] is True and res["cancelled"] is True


def test_data_file_manual_path_skips_dialog(iso: Path) -> None:
    pq = _write_parquet(iso, "ETHUSDT_H1.parquet", 400)
    res = app_mod.api_browse_data_file(path=str(pq))
    assert res.get("session") is None and res.get("dialog") is None  # 直选接口
    assert res["bars_unique"] == 400


def test_slice_file_is_marked_with_full_archive(iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """切片/局部窗口文件在卡片上标注，并探测同名的全量档案。"""
    from data_pipeline import parquet_manager as pm

    # 全量档案：data/training/BTCUSDT_M5.parquet（1000 根桩）
    full_dir = iso / "training"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_pq = _write_parquet(full_dir, "BTCUSDT_M5.parquet", 1000)
    monkeypatch.setattr(pm, "TRAINING_DIR", full_dir)

    # 切片文件：slices 目录下同名（400 根）
    slices_dir = iso / "slices"
    slices_dir.mkdir(exist_ok=True)
    slice_pq = _write_parquet(slices_dir, "BTCUSDT_M5.parquet", 400)

    info = pm.inspect_parquet_file(slice_pq)
    assert info["is_slice"] is True
    assert info["slice_of"]["file"] == "BTCUSDT_M5.parquet"
    assert info["slice_of"]["bars"] == 1000
    assert info["slice_of"]["data_file"] == str(full_pq.resolve())

    # 全量档案本身不是切片
    full_info = pm.inspect_parquet_file(full_pq)
    assert full_info["is_slice"] is False
    assert "slice_of" not in full_info


def test_slice_name_marker_and_missing_full(iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """文件名显式带切片标记也识别；找不到全量档案时不报错。"""
    from data_pipeline import parquet_manager as pm

    # 隔离全量档案目录，避免误探测到仓库 data/training/ 下的真实文件
    monkeypatch.setattr(pm, "TRAINING_DIR", iso / "empty_training")
    p = _write_parquet(iso, "ETHUSDT_H1_slice.parquet", 400)
    info = pm.inspect_parquet_file(p)
    assert info["is_slice"] is True
    assert "slice_of" not in info  # 无同名全量档案时不携带 slice_of

    plain = _write_parquet(iso, "XAUUSD_H1.parquet", 400)
    assert pm.inspect_parquet_file(plain)["is_slice"] is False


def test_poll_unknown_session_raises_400() -> None:
    with pytest.raises(HTTPException) as ei:
        app_mod.api_browse_data_file_poll(session="definitely_missing")
    assert ei.value.status_code == 400


# ── 策略选择器（与数据文件同款会话轮询） ──────────────────────────────────

def _write_strategy(tmp: Path, name: str = "best_ZZ.json") -> Path:
    p = tmp / name
    p.write_text(
        json.dumps(
            {
                "vocab_version": VOCAB_VERSION,
                "symbol": "ZZ",
                "formula": [1, 2, 3],
                "best_score": 1.0,
                "data_source": {"file": "ZZ_H1.parquet",
                                "start": "2020-01-01", "end": "2026-01-01"},
            }
        ),
        encoding="utf-8",
    )
    return p


def _write_legacy_strategy(tmp: Path, name: str = "best_Legacy.json") -> Path:
    """老 schema：无 data_source，只有顶层 data_file（供归一化回归测试）。"""
    p = tmp / name
    p.write_text(
        json.dumps(
            {
                "vocab_version": VOCAB_VERSION,
                "symbol": "Legacy",
                "formula": [1, 2, 3],
                "best_score": 0.9,
                "data_file": str((tmp / "Legacy_H1.parquet").resolve()),
            }
        ),
        encoding="utf-8",
    )
    return p


def test_strategy_poll_done_returns_strategy_info(iso: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    strat = _write_strategy(iso)
    monkeypatch.setattr(app_mod, "poll_picker",
                        lambda sid: {"done": True, "path": str(strat), "error": None})
    res = app_mod.api_browse_strategy_file_poll(session="s4")
    assert res["done"] is True and res["cancelled"] is False
    assert res["symbol"] == "ZZ"
    assert res["data_source"]["file"] == "ZZ_H1.parquet"  # 回测页「训练数据来源」来源字段


def test_strategy_poll_cancel(iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_mod, "poll_picker",
                        lambda sid: {"done": True, "path": None, "error": None})
    res = app_mod.api_browse_strategy_file_poll(session="s5")
    assert res["done"] is True and res["cancelled"] is True


def test_strategy_poll_unknown_session_raises_400() -> None:
    with pytest.raises(HTTPException) as ei:
        app_mod.api_browse_strategy_file_poll(session="definitely_missing")
    assert ei.value.status_code == 400


def test_strategy_file_manual_path_skips_dialog(iso: Path) -> None:
    """回测页下拉切换策略：strategy-file/browse?path=… 直达（免对话框），与数据文件同款。"""
    strat = _write_strategy(iso)
    res = app_mod.api_browse_strategy_file(path=str(strat))
    assert res.get("session") is None and res.get("dialog") is None  # 直选接口
    assert res["symbol"] == "ZZ"
    assert res["strategy_file"] == str(strat.resolve())


def test_strategy_list_rows_carry_strategy_file_for_dropdown(
    iso: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/api/strategies 每行必须带完整绝对路径（下拉 option 的 value 用 strategy_file）。

    回归：曾只回传裸文件名 file，导致回测页策略下拉 option value 为空。
    """
    import web.progress as progress_mod

    monkeypatch.setattr(progress_mod, "STRATEGIES_DIR", iso)
    _write_strategy(iso, name="best_ZZ.json")
    _write_strategy(iso, name="best_YY.json")
    data = progress_mod.list_strategies()
    assert len(data) == 2
    for row in data:
        assert row["file"].startswith("best_")
        assert row["strategy_file"] == str((iso / row["file"]).resolve())
        assert Path(row["strategy_file"]).exists()
    # 路由层同样透传
    res = app_mod.api_strategies()
    rows = {r["file"]: r for r in res["strategies"]}
    assert rows["best_ZZ.json"]["strategy_file"] == str((iso / "best_ZZ.json").resolve())


def test_strategy_list_normalizes_legacy_data_file(
    iso: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """老 schema 策略（无 data_source，仅顶层 data_file）→ 归一化为 data_source.file。

    实时/回测页策略下拉的数据源标签取 data_source.file，老文件缺字段会显示为空。
    """
    import web.progress as progress_mod

    monkeypatch.setattr(progress_mod, "STRATEGIES_DIR", iso)
    _write_strategy(iso, name="best_ZZ.json")
    legacy = _write_legacy_strategy(iso)
    raw_legacy = json.loads(legacy.read_text(encoding="utf-8"))
    rows = {r["file"]: r for r in progress_mod.list_strategies()}
    old = rows["best_Legacy.json"]
    assert old["data_source"]["file"] == "Legacy_H1.parquet"  # 由顶层 data_file 推导
    assert old["data_source"]["data_file"] == raw_legacy["data_file"]
    new = rows["best_ZZ.json"]
    assert new["data_source"]["file"] == "ZZ_H1.parquet"  # 新 schema 原样保留
    # 老文件 data_source.file 与 data_file 顶层一致（供回测 fallback 溯源校验）
    assert Path(raw_legacy["data_file"]).name == old["data_source"]["file"]

def test_strategy_list_live_sidecar_enriches_champion_row(
    iso: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """存在 best_*.live.json 侧车时，已部署行附带 live 详情（分数/公式/更新时间）。"""
    import web.progress as progress_mod

    monkeypatch.setattr(progress_mod, "STRATEGIES_DIR", iso)
    _write_strategy(iso, name="best_ZZ.json")
    live = iso / "best_ZZ.live.json"
    live.write_text(json.dumps({"symbol": "ZZ", "formula": [1, 2, 3],
                                "best_score": 2.5, "train_steps": 100}),
                    encoding="utf-8")
    rows = {r["file"]: r for r in progress_mod.list_strategies()}
    row = rows["best_ZZ.json"]
    assert row["live"] is not None
    assert row["live"]["live_file"] == "best_ZZ.live.json"
    assert row["live"]["best_score"] == 2.5
    assert row["live"]["formula"] == [1, 2, 3]
    assert row["live"]["formula_decoded"]  # 解码后的人类可读串
    assert row["live"]["updated_at"]  # 更新时间（文件 mtime）
    assert not row.get("live_only")  # 已部署行不带 live_only 标记
    assert "best_ZZ.live.json" not in rows  # 侧车不单独成行


def test_strategy_list_live_only_row_when_no_champion(
    iso: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """尚无已部署冠军时（首次训练中途），*.live.json 以「训练中」行展示。"""
    import web.progress as progress_mod

    monkeypatch.setattr(progress_mod, "STRATEGIES_DIR", iso)
    live = iso / "best_YY.live.json"
    live.write_text(json.dumps({"symbol": "YY", "timeframe": "M5",
                                "formula": [4, 5, 6], "best_score": 3.2}),
                    encoding="utf-8")
    rows = progress_mod.list_strategies()
    assert len(rows) == 1
    row = rows[0]
    assert row["live_only"] is True
    assert row["file"] == "best_YY.live.json"
    assert row["strategy_file"] == str(live.resolve())
    assert row["best_score"] == 3.2
    assert row["live"]["live_file"] == "best_YY.live.json"
    assert row["live"]["updated_at"]
