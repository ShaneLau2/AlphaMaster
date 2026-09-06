"""增量归档语义测试：重复下载只追加新 bar，无新增时不重写文件/sidecar。

覆盖：首下 / binance·okx 向后扩展 / 拉满后 no-op / 同窗口 repeat no-op /
显式 replace 覆盖。全部用桩 fetch，不触网。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import web.data_download as dd

BASE_TS = 1_600_000_000  # epoch 秒


def _bars(start_sec: int, count: int, step: int = 3600) -> pd.DataFrame:
    ts = [BASE_TS + start_sec + i * step for i in range(count)]
    return pd.DataFrame(
        {"time": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """下载目录 + 桩 fetch：尾部窗口 300 根；带锚点请求时返回档案最早 bar 之前的数据。"""
    monkeypatch.setattr(dd, "DOWNLOAD_DIR", tmp_path)
    calls: list[dict] = []

    def fake_fetch(kind: str, symbol: str, tf: str, n: int,
                   progress_cb=None, anchor_ms: int | None = None) -> pd.DataFrame:
        calls.append({"kind": kind, "symbol": symbol, "anchor_ms": anchor_ms})
        if anchor_ms is not None:
            anchor_sec = anchor_ms // 1000
            count = anchor_sec - BASE_TS
            if count <= 0:
                return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            return _bars(0, max(0, min(count, 400)))
        return _bars(9700, 300 if n >= 5000 else n)

    monkeypatch.setattr(dd, "_fetch_source_bars", fake_fetch)
    env = type("Env", (), {"tmp": tmp_path, "calls": calls})()
    return env


def _mtime_ns(p: Path) -> int:
    return p.stat().st_mtime_ns


def _disk_rows(f: str) -> int:
    return len(pd.read_parquet(f))


def test_first_download_writes_archive(env) -> None:
    info = dd.download_symbol_bars("ZZZ", "1h", source="binance", n_bars=5000)
    assert info["n_bars"] == 300
    assert info["added_bars"] == 300
    assert info["merged"] is False
    assert info["no_change"] is False
    assert (env.tmp / "ZZZ_H1.parquet").exists()
    assert (env.tmp / "ZZZ_H1.parquet.meta.json").exists()


@pytest.mark.parametrize("source", ["binance", "okx"])
def test_repeat_extends_backward_past_archive_start(env, source: str) -> None:
    a1 = dd.download_symbol_bars("ZZZ", "1h", source=source, n_bars=5000)
    f = a1["data_file"]
    meta0 = json.loads(Path(f + ".meta.json").read_text(encoding="utf-8"))
    env.calls.clear()

    b1 = dd.download_symbol_bars("ZZZ", "1h", source=source, n_bars=5000)
    anchored = [c for c in env.calls if c["anchor_ms"] is not None]
    assert anchored, "重复下载应发起向后扩展请求"
    assert anchored[0]["anchor_ms"] == (BASE_TS + 9700) * 1000 - 1  # 严格早于档案最早 bar
    assert b1["n_bars"] == 700
    assert b1["added_bars"] == 400
    assert b1["no_change"] is False
    assert _disk_rows(f) == 700

    meta1 = json.loads(Path(f + ".meta.json").read_text(encoding="utf-8"))
    assert meta1["bars"] == 700
    assert meta1["end_date"] > meta0["end_date"]  # sidecar 区间随档案刷新


def test_repeat_at_provider_origin_is_noop(env) -> None:
    dd.download_symbol_bars("ZZZ", "1h", source="binance", n_bars=5000)
    for _ in range(24):  # 拉到数据源留存起点
        dd.download_symbol_bars("ZZZ", "1h", source="binance", n_bars=5000)
    f = str(env.tmp / "ZZZ_H1.parquet")
    side = f + ".meta.json"
    pf, sf = _mtime_ns(Path(f)), _mtime_ns(Path(side))

    info = dd.download_symbol_bars("ZZZ", "1h", source="binance", n_bars=5000)
    assert info["no_change"] is True
    assert _mtime_ns(Path(f)) == pf, "无新增时不应重写 parquet"
    assert _mtime_ns(Path(side)) == sf, "无新增时不应刷新 sidecar"


def test_same_window_repeat_noop_for_limited_source(env) -> None:
    d1 = dd.download_symbol_bars("YYY", "1h", source="tradingview", n_bars=5000)
    f, side = d1["data_file"], d1["data_file"] + ".meta.json"
    pf, sf = _mtime_ns(Path(f)), _mtime_ns(Path(side))

    d2 = dd.download_symbol_bars("YYY", "1h", source="tradingview", n_bars=5000)
    assert d2["no_change"] is True
    assert _mtime_ns(Path(f)) == pf
    assert _mtime_ns(Path(side)) == sf


def test_download_count_and_first_batch_tracked(env) -> None:
    """sidecar 累计下载次数/最早批次日：落盘 +1，首次日期不变，no-change 不计数。"""
    a1 = dd.download_symbol_bars("ZZZ", "1h", source="binance", n_bars=5000)
    side = Path(a1["data_file"] + ".meta.json")
    m1 = json.loads(side.read_text(encoding="utf-8"))
    assert m1["download_count"] == 1
    assert m1["first_downloaded_at"] == m1["downloaded_at"]
    first = m1["first_downloaded_at"]

    # 追加 400 根旧 bar → 计 2 次，最早批次日保持第一次
    dd.download_symbol_bars("ZZZ", "1h", source="binance", n_bars=5000)
    m2 = json.loads(side.read_text(encoding="utf-8"))
    assert m2["download_count"] == 2
    assert m2["first_downloaded_at"] == first

    # 拉到数据源起点后重复下载 no-change → 不重写 sidecar，计数不变
    for _ in range(30):
        dd.download_symbol_bars("ZZZ", "1h", source="binance", n_bars=5000)
    m3 = json.loads(side.read_text(encoding="utf-8"))
    assert m3["download_count"] == 2


def test_history_backfills_count_for_old_sidecars(env) -> None:
    """旧版 sidecar（无计数字段）→ 历史面板按 1 次 / 首次=下载日回填。"""
    f = env.tmp / "OLD_H1.parquet"
    f.write_bytes(b"not a parquet")  # 历史面板只读 sidecar，不读文件内容
    (env.tmp / "OLD_H1.parquet.meta.json").write_text(
        json.dumps({
            "file": "OLD_H1.parquet", "symbol": "OLD", "timeframe": "H1",
            "source": "binance", "source_label": "Binance",
            "downloaded_at": "2026-01-01T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    hist = dd.list_download_history(limit=50)
    row = next(r for r in hist["downloads"] if r["file"] == "OLD_H1.parquet")
    assert row["download_count"] == 1
    assert row["first_downloaded_at"] == "2026-01-01T00:00:00+00:00"
    assert row["data_file"] == str((env.tmp / "OLD_H1.parquet").resolve())


def test_history_rows_carry_data_file_for_dropdown(env) -> None:
    """历史行必须带完整绝对路径（前端下拉 option 的 value 用 data_file）。

    回归：曾经只回传裸文件名 file，导致下拉每个 option 的 value 为空串，
    选中文件后 change 事件静默复位、什么都不加载。
    """
    info = dd.download_symbol_bars("ZZZ", "1h", source="tradingview", n_bars=5000)
    f = Path(info["data_file"])
    hist = dd.list_download_history(limit=50)
    row = next(r for r in hist["downloads"] if r["file"] == "ZZZ_H1.parquet")
    # 契约：option value = data_file（绝对路径，可直接喂 browse?path=…）
    assert row["data_file"] == str(f.resolve())
    assert Path(row["data_file"]).exists()
    assert Path(row["data_file"]).name == "ZZZ_H1.parquet"


def test_explicit_replace_overwrites(env) -> None:
    dd.download_symbol_bars("ZZZ", "1h", source="tradingview", n_bars=5000)
    info = dd.download_symbol_bars("ZZZ", "1h", source="tradingview",
                                   n_bars=2000, mode="replace")
    assert info["n_bars"] == 2000
    assert info["merged"] is False
    assert info["no_change"] is False
    assert _disk_rows(info["data_file"]) == 2000


def test_unreadable_archive_is_flagged_not_silently_replaced(env) -> None:
    out = env.tmp / "ZZZ_H1.parquet"
    out.write_bytes(b"not a parquet")  # 损坏档案
    info = dd.download_symbol_bars("ZZZ", "1h", source="tradingview", n_bars=5000)
    assert info.get("archive_unreadable") is True


# ── backfill_to_origin（回溯到数据源起点） ────────────────────────────────

FLOOR = BASE_TS - 1200 * 3600  # 数据源留存起点（比 BASE 早 1200h）


def _backfill_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """桩 fetch：尾部窗口 300 根；锚点请求返回数据源起点 FLOOR 之前的分页。"""
    monkeypatch.setattr(dd, "DOWNLOAD_DIR", tmp_path)
    calls: list[dict] = []

    def fake_fetch(kind: str, symbol: str, tf: str, n: int,
                   progress_cb=None, anchor_ms: int | None = None) -> pd.DataFrame:
        calls.append({"kind": kind, "symbol": symbol, "anchor_ms": anchor_ms})
        if anchor_ms is not None:
            avail_end = anchor_ms // 1000 - 1
            if avail_end < FLOOR:
                return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            count = (avail_end - FLOOR) // 3600 + 1
            n_ret = min(count, 400)
            ts = [FLOOR + i * 3600 for i in range(n_ret)]
            return pd.DataFrame({"time": ts, "open": 1.0, "high": 1.0,
                                 "low": 1.0, "close": 1.0, "volume": 1})
        return _bars(9700, 300 if n >= 5000 else n)

    monkeypatch.setattr(dd, "_fetch_source_bars", fake_fetch)
    return type("Env", (), {"tmp": tmp_path, "calls": calls})()


@pytest.mark.parametrize("source", ["binance", "okx"])
def test_backfill_reaches_provider_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          source: str) -> None:
    e = _backfill_env(tmp_path, monkeypatch)
    info = dd.download_symbol_bars("ZZZ", "1h", source=source, n_bars=5000)  # 先有 300 根档案
    assert info["n_bars"] == 300

    bf = dd.backfill_to_origin("ZZZ", "1h", source=source, page_bars=5000)
    assert bf["reached_origin"] is True
    assert bf["backfilled_bars"] == 400
    assert bf["pages"] == 1  # 已有档案：1 个有效回溯页后到达数据源起点
    assert bf["n_bars"] == 700
    disk = pd.read_parquet(bf["data_file"])
    assert int(disk["time"].min()) == FLOOR
    meta = json.loads(Path(bf["data_file"] + ".meta.json").read_text(encoding="utf-8"))
    assert meta["bars"] == 700


def test_backfill_without_archive_starts_with_tail(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    e = _backfill_env(tmp_path, monkeypatch)
    bf = dd.backfill_to_origin("NEWS", "1h", source="okx", page_bars=5000)
    assert bf["reached_origin"] is True
    assert bf["backfilled_bars"] == 700  # 尾部 300 + 回溯 400
    assert bf["n_bars"] == 700
    assert bf["pages"] == 2


def test_backfill_reports_progress_and_phase(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    e = _backfill_env(tmp_path, monkeypatch)
    progress: list[int] = []
    phases: list[str] = []
    dd.download_symbol_bars("ZZZ", "1h", source="binance", n_bars=5000)
    dd.backfill_to_origin("ZZZ", "1h", source="binance", page_bars=5000,
                          progress_cb=progress.append, phase_cb=phases.append)
    assert len(progress) >= 1
    assert any("回溯" in p for p in phases)


@pytest.mark.parametrize("source", ["tradingview", "tongdaxin"])
def test_backfill_rejects_limited_sources(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch,
                                          source: str) -> None:
    e = _backfill_env(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="无法向更早历史翻页"):
        dd.backfill_to_origin("ZZZ", "1h", source=source, page_bars=5000)

# ---------- 服务重启自动续跑（断点续传语义） ----------


def _reset_dd_state() -> None:
    """清空模块级注册表/队列/锁，避免测试间互相污染。"""
    with dd._QUEUE_LOCK:
        dd._JOBS.clear()
        dd._QUEUE.clear()
    dd._LOADED = False
    dd._WORKER_STARTED = False


def _seed_jobs_file(path: Path, job: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({job["job_id"]: job}, ensure_ascii=False), encoding="utf-8")


def _running_job(job_id: str = "run1", **kw) -> dict:
    base = {
        "job_id": job_id, "symbol": "ZZZ", "timeframe": "1h",
        "source": "binance", "n_bars": 5000, "mode": "merge",
        "status": "running", "phase": "拉取 K 线（分页）", "bars_fetched": 12000,
        "started_at": 1000.0, "finished_at": None, "message": "",
        "result": None, "error": None, "resume_attempts": 0,
    }
    base.update(kw)
    return base


class TestRestartResume:
    def _restore(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dd, "JOBS_FILE", tmp_path / "download_jobs.json")
        monkeypatch.setattr(dd, "_ensure_worker", lambda: None)  # 不启动真实 worker（无网络）
        _reset_dd_state()
        dd._restore_from_disk()

    def test_running_job_auto_requeued_not_interrupted(self, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
        _seed_jobs_file(tmp_path / "download_jobs.json", _running_job())
        self._restore(tmp_path, monkeypatch)
        try:
            job = dd._JOBS["run1"]
            assert job.status == "queued", f"重启后应自动续跑而非中断，实际 {job.status}"
            assert job.resume_attempts == 1
            assert job.error is None
            assert "续跑" in job.phase
            assert dd._QUEUE and dd._QUEUE[0].job_id == "run1"
        finally:
            _reset_dd_state()

    def test_queued_job_still_passes_through(self, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
        _seed_jobs_file(tmp_path / "download_jobs.json",
                        _running_job(status="queued", phase="排队中"))
        self._restore(tmp_path, monkeypatch)
        try:
            assert dd._JOBS["run1"].status == "queued"
            assert dd._QUEUE and dd._QUEUE[0].job_id == "run1"
        finally:
            _reset_dd_state()

    def test_done_and_error_jobs_kept(self, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        jobs = {
            d["job_id"]: d for d in [
                _running_job("done1", status="done", phase="完成", result={"ok": True}),
                _running_job("err1", status="error", phase="失败", error="boom"),
            ]
        }
        _seed_jobs_file(tmp_path / "download_jobs.json", next(iter(jobs.values())))
        (tmp_path / "download_jobs.json").write_text(
            json.dumps(jobs, ensure_ascii=False), encoding="utf-8"
        )
        self._restore(tmp_path, monkeypatch)
        try:
            assert dd._JOBS["done1"].status == "done"
            assert dd._JOBS["err1"].status == "error"
            assert not any(j.job_id in ("done1", "err1") for j in dd._QUEUE)
        finally:
            _reset_dd_state()

    def test_resume_cap_marks_interrupted_after_max_attempts(self, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
        n = dd._MAX_RESUME_ATTEMPTS
        _seed_jobs_file(tmp_path / "download_jobs.json",
                        _running_job(resume_attempts=n))  # 已达上限 → 不再自动续跑
        self._restore(tmp_path, monkeypatch)
        try:
            job = dd._JOBS["run1"]
            assert job.status == "error"
            assert job.phase == "已中断"
            assert "自动续跑已达上限" in job.error
            assert dd._QUEUE == []
        finally:
            _reset_dd_state()

    def test_resume_attempts_persisted_roundtrip(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        _seed_jobs_file(tmp_path / "download_jobs.json",
                        _running_job(resume_attempts=1))
        self._restore(tmp_path, monkeypatch)
        try:
            job = dd._JOBS["run1"]
            assert job.status == "queued" and job.resume_attempts == 2
            dd._persist()
            data = json.loads((tmp_path / "download_jobs.json").read_text(encoding="utf-8"))
            assert data["run1"]["resume_attempts"] == 2
        finally:
            _reset_dd_state()
