"""联网下载 K 线 → 本地 parquet（供训练直接使用）。

数据源：tradingview（匿名 WebSocket）/ okx（翻页 REST）/ tongdaxin（免费行情服务器）。
请求数量可自选（100~100,000），服务端给多少取多少。

大任务走后台上传：POST /api/data/download 立即返回 job_id，
轮询 GET /api/data/download-status?job_id=… 获取 阶段/已拉取根数/耗时；
CLI 需要阻塞同步结果时用 ?sync=1（保持旧行为）。
"""
from __future__ import annotations

import datetime
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data_pipeline.parquet_manager import inspect_parquet_file
from web import tv_history
from web.data_sources.base import Bar

# 下载时可选的官方数据源（后端白名单，前端下拉来自 /api/realtime/sources）
SUPPORTED_SOURCES = ("tradingview", "binance", "okx", "tongdaxin")

# 下载落地文件的元信息（sidecar JSON，供卡片展示数据来源）
_META_SUFFIX = ".meta.json"
SOURCE_LABELS = {
    "tradingview": "TradingView",
    "binance": "Binance",
    "okx": "OKX",
    "tongdaxin": "通达信",
}


def write_parquet_meta(
    out: Path,
    sym: str,
    tf: str,
    source: str,
    *,
    bars: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """在 parquet 旁写 {file}.meta.json（失败不影响主流程）。

    bars / start_date / end_date 是下载完成时已知的快照（合并去重后的最终值），
    供「下载历史」面板直接展示，无需再解码头文件。

    download_count / first_downloaded_at：从已有 sidecar 推断每次 merge 的历史——
    每次真实落盘（新建/追加/回溯/覆盖）计数 +1，最早批次日保持不变。
    无新增（no_change）时不调用本函数，计数不受影响。
    """
    try:
        prev = read_parquet_meta(out) or {}
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta = {
            "file": out.name,
            "symbol": sym,
            "timeframe": tf,
            "source": source,
            "source_label": SOURCE_LABELS.get(source, source),
            "downloaded_at": now_iso,
            "download_count": int(prev.get("download_count") or 0) + 1,
            "first_downloaded_at": prev.get("first_downloaded_at") or now_iso,
        }
        if bars is not None:
            meta["bars"] = int(bars)
        if start_date:
            meta["start_date"] = start_date
        if end_date:
            meta["end_date"] = end_date
        Path(str(out) + _META_SUFFIX).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 元数据写失败不影响下载结果
        logging.getLogger(__name__).warning("write parquet meta failed for %s", out)


def _parquet_footer_summary(path: Path) -> dict[str, Any] | None:
    """轻量读 parquet footer：行数 + time 列 min/max（不解码数据，毫秒级）。

    旧版 sidecar 缺 bars/日期区间时用于补读。统计不可得时回退 None。
    """
    try:
        import pyarrow.parquet as pq

        meta = pq.read_metadata(str(path))
        names = meta.schema.names
        if "time" not in names:
            return {"bars": int(meta.num_rows)}
        idx = names.index("time")
        t_min: int | None = None
        t_max: int | None = None
        for rg in range(meta.num_row_groups):
            st = meta.row_group(rg).column(idx).statistics
            if st is None or not st.has_min_max:
                continue
            mn, mx = int(st.min), int(st.max)
            if t_min is None or mn < t_min:
                t_min = mn
            if t_max is None or mx > t_max:
                t_max = mx
        out: dict[str, Any] = {"bars": int(meta.num_rows)}
        if t_min is not None and t_max is not None:
            out["start"] = datetime.datetime.fromtimestamp(
                t_min, datetime.timezone.utc
            ).strftime("%Y-%m-%d")
            out["end"] = datetime.datetime.fromtimestamp(
                t_max, datetime.timezone.utc
            ).strftime("%Y-%m-%d")
        return out
    except Exception:  # noqa: BLE001 任何失败都回退（面板宁可缺字段也不报错）
        return None

def read_parquet_meta(path: str | Path) -> dict[str, Any] | None:
    """读取下载元信息（无 tag / 损坏时返回 None，绝不抛错）。"""
    try:
        meta_path = Path(str(path) + _META_SUFFIX)
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def list_download_history(limit: int = 20) -> dict[str, Any]:
    """扫描下载目录：parquet + meta sidecar → 下载历史（最新在前）与按数据源统计。

    - 行信息以 sidecar（{file}.meta.json）为准：source / downloaded_at /
      bars / start_date / end_date；旧版 sidecar 缺 bars/区间时用 pyarrow
      footer 统计轻量补读（不解码数据）。
    - 无 sidecar 的手工放置文件也会列出（source=None，文件名回退解析品种）。

    Returns:
        {downloads: [{file, data_file, symbol, timeframe, source,
                      source_label, downloaded_at, bars, start, end}],
         sources: [{source, label, files, bars, latest}],
         total_files, total_bars}
    """
    from data_pipeline.parquet_manager import parse_parquet_filename

    rows: list[dict[str, Any]] = []
    if DOWNLOAD_DIR.exists():
        for p in sorted(DOWNLOAD_DIR.glob("*.parquet")):
            meta = read_parquet_meta(p) or {}
            source = meta.get("source")
            # 旧版 sidecar 无计数字段时按 1 次下载、最早批次日=下载日回填（
            # 当时的每次落盘即一次下载，无法再细分，但首次日期是可信的）
            row: dict[str, Any] = {
                "file": p.name,
                # 完整路径：供前端下拉直接切换（browse?path=…），与
                # inspect_parquet_file 的 data_file 口径一致（绝对路径）
                "data_file": str(p.resolve()),
                "symbol": meta.get("symbol"),
                "timeframe": meta.get("timeframe"),
                "source": source,
                "source_label": meta.get("source_label")
                or SOURCE_LABELS.get(source, source),
                "downloaded_at": meta.get("downloaded_at"),
                "download_count": int(meta["download_count"])
                if meta.get("download_count") is not None
                else (1 if meta.get("downloaded_at") else None),
                "first_downloaded_at": meta.get("first_downloaded_at")
                or meta.get("downloaded_at"),
                "bars": meta.get("bars"),
                "start": meta.get("start_date"),
                "end": meta.get("end_date"),
            }
            if row["symbol"] is None:
                try:
                    row["symbol"], row["timeframe"] = parse_parquet_filename(p)
                except Exception:  # noqa: BLE001
                    pass
            if row["bars"] is None or row["start"] is None:
                ext = _parquet_footer_summary(p)
                if ext:
                    if row["bars"] is None:
                        row["bars"] = ext["bars"]
                    if row["start"] is None and ext.get("start"):
                        row["start"] = ext["start"]
                        row["end"] = ext["end"]
            rows.append(row)

    rows.sort(key=lambda r: r["downloaded_at"] or "", reverse=True)

    # 按数据源聚合（基于全部扫描文件）
    agg: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = r["source"] or "local"
        s = agg.setdefault(key, {
            "source": r["source"],
            "label": r["source_label"] or "本地文件",
            "files": 0,
            "bars": 0,
            "latest": None,
        })
        s["files"] += 1
        s["bars"] += int(r["bars"] or 0)
        if r["downloaded_at"] and (
            s["latest"] is None or r["downloaded_at"] > s["latest"]
        ):
            s["latest"] = r["downloaded_at"]
    sources = sorted(agg.values(), key=lambda s: -s["bars"])
    return {
        "downloads": rows[: max(1, min(int(limit), len(rows)))] if rows else [],
        "sources": sources,
        "total_files": len(rows),
        "total_bars": sum(int(r["bars"] or 0) for r in rows),
    }


_OKX_TF = {  # 周期 -> OKX history-candles bar 参数
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W", "1M": "1M",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "training"


def list_local_parquet_files() -> list[dict[str, Any]]:
    """扫描本地可直接用于回放/训练的 .parquet（data/training + data/slices 含一级子目录）。

    供历史回放页的数据文件下拉选择；行数与时间区间优先读 sidecar，
    缺失时用 pyarrow footer 轻量补读（不解码数据，毫秒级）。
    返回按 mtime 新→旧排序的 [{data_file, symbol, timeframe, bars, start, end,
    rel, size_mb}]。
    """
    from data_pipeline.parquet_manager import parse_parquet_filename

    candidates: list[Path] = []
    bases = [DOWNLOAD_DIR, PROJECT_ROOT / "data" / "slices"]
    for base in bases:
        if not base.exists():
            continue
        candidates.extend(sorted(base.glob("*.parquet")))
        if base.name == "slices":
            for sub in sorted(base.iterdir()):
                if sub.is_dir():
                    candidates.extend(sorted(sub.glob("*.parquet")))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        meta = read_parquet_meta(p) or {}
        symbol, tf = meta.get("symbol"), meta.get("timeframe")
        if not symbol:
            try:
                symbol, tf = parse_parquet_filename(p)
            except Exception:  # noqa: BLE001
                pass
        bars, start, end = meta.get("bars"), meta.get("start_date"), meta.get("end_date")
        if bars is None or start is None:
            ext = _parquet_footer_summary(p)
            if ext:
                if bars is None:
                    bars = ext["bars"]
                if start is None and ext.get("start"):
                    start, end = ext["start"], ext["end"]
        try:
            size_mb = round(p.stat().st_size / 1048576.0, 1)
            mtime = p.stat().st_mtime
        except OSError:
            size_mb, mtime = 0.0, 0.0
        rel = str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
        rows.append({
            "data_file": key,
            "rel": rel.replace("\\", "/"),
            "symbol": symbol or p.stem,
            "timeframe": tf or "",
            "bars": bars,
            "start": start,
            "end": end,
            "size_mb": size_mb,
            "mtime": mtime,
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows

# 周期 → 训练文件名周期 token（与 parse_parquet_filename 兼容的规范写法）
TF_TOKEN = {
    "1m": "M1",
    "3m": "M3",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "2h": "H2",
    "4h": "H4",
    "6h": "H6",
    "8h": "H8",
    "12h": "H12",
    "1d": "D1",
    "3d": "D3",
    "1w": "W1",
    "1M": "MN1",
}
SUPPORTED_TIMEFRAMES = tuple(TF_TOKEN)

# 数量上下限：少于 100 根没有训练意义；默认上限 100,000
MIN_BARS = 100
MAX_BARS = 100_000
DEFAULT_BARS = 5000

# Binance 30m 以下的周期（严格小于 30m）支持全量历史：1m 单品种可回溯数百万根，
# 上限开到 100 万根（后台任务自动翻页，1000 根/页）。其余周期上限保持 10 万。
DEEP_HISTORY_TIMEFRAMES = ("1m", "3m", "5m", "15m")
MAX_BARS_DEEP = 1_000_000


def bars_limit_for(source: str, timeframe: str) -> int:
    """按 数据源×周期 返回请求数量上限。"""
    if source == "binance" and timeframe in DEEP_HISTORY_TIMEFRAMES:
        return MAX_BARS_DEEP
    return MAX_BARS

_INVALID_SYMBOL_CHARS = {"/", "\\", "\x00"}


def validate_symbol(symbol: str) -> str:
    sym = (symbol or "").strip()
    if not sym:
        raise ValueError("品种不能为空")
    if len(sym) > 64:
        raise ValueError("品种名过长（≤64 字符）")
    if any(c in _INVALID_SYMBOL_CHARS for c in sym) or sym in {".", ".."}:
        raise ValueError("品种包含非法字符（不允许 / \\\\ .. 等）")
    return sym


def validate_timeframe(timeframe: str) -> str:
    tf = (timeframe or "").strip().lower()
    if tf not in TF_TOKEN:
        raise ValueError(
            f"不支持的周期: {timeframe or '(空)'}；可选: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return tf


def _bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": [int(b.ts) for b in bars],
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [float(b.volume or 0.0) for b in bars],
        }
    )
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df = df.astype({"time": "int64", "volume": "int64"})
    return df


def _fetch_source_bars(
    kind: str, symbol: str, tf: str, n: int,
    progress_cb: Callable[[int], None] | None = None,
    anchor_ms: int | None = None,
) -> pd.DataFrame:
    """按数据源拉取历史 K 线（服务端可用性 + 周期支持均在此校验）。

    anchor_ms（epoch 毫秒）仅 binance/okx 生效：代替「现在」作翻页锚点，
    用于向档案更早处增量扩展。
    """
    from web.data_sources.factory import get_source

    if kind == "tradingview":
        rows = tv_history.fetch_history(symbol, tf, n)
        if not rows:
            raise RuntimeError(
                "TradingView 未返回任何 K 线（网络不可达、品种不存在，或需 EXCHANGE:CODE 指定交易所）"
            )
        return _bars_to_df([Bar(ts=r[0], open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5]) for r in rows])

    if kind == "okx":
        # 走仓库自带的翻页下载器（可拉全量历史）而非 okx_source 的 300 根上限
        import download_okx_klines as dok

        inst_id = symbol.strip().upper()
        for quote in ("USDT", "USDC", "USD"):
            if not inst_id.endswith("-SWAP") and inst_id.endswith(quote):
                inst_id = f"{inst_id[:-len(quote)]}-{quote}-SWAP"
        if not inst_id.endswith("-SWAP"):
            inst_id = f"{inst_id}-USDT-SWAP"
        if tf not in _OKX_TF:
            raise ValueError(f"OKX 不支持周期 {tf}")
        df = dok.download_history(inst_id, _OKX_TF[tf], max_bars=n,
                                  progress_cb=progress_cb, after_start_ms=anchor_ms)
        if df is None or df.empty:
            raise RuntimeError(f"OKX 未返回任何 K 线：{symbol}")
        return df.rename(columns={"tick_volume": "volume"})

    if kind == "binance":
        # binance 支持 endTime 锚点翻页：增量归档可向档案更早处扩展
        inst = get_source("binance")
        ok, hint = inst.available()
        if not ok:
            raise RuntimeError(f"数据源不可用（{inst.label}）：{hint}")
        bars = inst.fetch_bars(symbol, tf, n, drop_forming=True,
                               progress_cb=progress_cb, end_time=anchor_ms)
        if not bars:
            raise RuntimeError(f"{inst.label} 未返回任何 K 线：{symbol}")
        return _bars_to_df(bars)

    # tongdaxin：走标准 DataSource 接口
    inst = get_source(kind)
    ok, hint = inst.available()
    if not ok:
        raise RuntimeError(f"数据源不可用（{inst.label}）：{hint}")
    tfs = inst.supported_timeframes()
    if tf not in tfs:
        raise ValueError(f"{inst.label} 不支持周期 {tf}；支持: {', '.join(tfs)}")
    bars = inst.fetch_bars(symbol, tf, n, drop_forming=True, progress_cb=progress_cb)
    if not bars:
        raise RuntimeError(f"{inst.label} 未返回任何 K 线：{symbol}")
    return _bars_to_df(bars)


_ARCHIVE_COLS = {"time", "open", "high", "low", "close", "volume"}


def _archive_path(sym: str, tf: str) -> Path:
    """下载落地文件名（跨平台安全；解析时按最后一个 _ 取周期 token）。"""
    safe_name = sym.replace(":", "_")
    return DOWNLOAD_DIR / f"{safe_name}_{TF_TOKEN[tf]}.parquet"


def _read_archive(out: Path) -> pd.DataFrame | None:
    """读取已有档案（合并增量用）。缺失/损坏/缺列时返回 None（不抛错）。"""
    if not Path(out).exists():
        return None
    try:
        old = pd.read_parquet(out)
        if not _ARCHIVE_COLS.issubset(old.columns):
            return None
        old = old.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        return old.astype({"time": "int64", "volume": "int64"})
    except Exception as exc:  # noqa: BLE001
        logger.warning("read archive failed for %s: %s", out, exc)
        return None


def _merge_dfs(*frames: pd.DataFrame) -> pd.DataFrame:
    """按 time 去重合并排序（增量归档的唯一写路径）。"""
    df = pd.concat(list(frames), ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df.astype({"time": "int64", "volume": "int64"})


def _archive_info_from_df(
    df: pd.DataFrame, sym: str, tf: str, out: Path,
    *, fetched_bars: int, added_bars: int, merged: bool, no_change: bool,
) -> dict[str, Any]:
    """从内存档案构造与 inspect_parquet_file 一致的信息（避免无新增时重读文件）。"""
    times = df["time"]
    t_min, t_max = int(times.min()), int(times.max())
    span_years = round((t_max - t_min) / (365.25 * 24 * 3600), 2) if t_max > t_min else 0.0
    start_date = datetime.datetime.fromtimestamp(
        t_min, datetime.timezone.utc
    ).strftime("%Y-%m-%d")
    end_date = datetime.datetime.fromtimestamp(
        t_max, datetime.timezone.utc
    ).strftime("%Y-%m-%d")
    return {
        "data_file": str(out),
        "filename": out.name,
        "symbol": sym,
        "timeframe": tf,
        "bars": int(len(df)),
        "years_h1": span_years,
        "valid": True,
        "message": "",
        "n_bars": int(len(df)),
        "fetched_bars": fetched_bars,
        "added_bars": added_bars,
        "new_bars": added_bars,  # 兼容旧字段名（UI 合并提示用）
        "merged": merged,
        "no_change": bool(no_change),
        "start_date": start_date,
        "end_date": end_date,
    }


def _save_and_inspect(
    df: pd.DataFrame, sym: str, tf: str, n: int, source: str = "", mode: str = "merge",
    old_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """增量归档：把本次拉取合并进已有 parquet，只追加新 bar 并刷新 sidecar 区间。

    mode=merge（默认）：与已有档案按 time 去重，历史只增不减；本次没有任何新 bar
        （重复下载同一尾部窗口）时不重写文件、不刷新 sidecar，返回 no_change=True。
    mode=replace（显式覆盖）：只用本次拉取的最近 n 根重写文件。

    old_df 由调用方预读传入（避免归档读取重复 IO）；未传时按需读取。
    """
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df = df.astype({"time": "int64", "volume": "int64"})
    fetched_bars = int(len(df))

    out = _archive_path(sym, tf)
    out.parent.mkdir(parents=True, exist_ok=True)
    had_archive = Path(out).exists()

    old = old_df
    if old is None and mode == "merge":
        old = _read_archive(out)

    merged = False
    if mode == "merge" and old is not None:
        combined = _merge_dfs(old, df)
        added_bars = int(len(combined)) - int(len(old))
        if added_bars == 0:
            # 重复下载同一窗口：档案已是这个窗口的超集 → 不重写文件/sidecar
            return _archive_info_from_df(
                old, sym, tf, out, fetched_bars=fetched_bars,
                added_bars=0, merged=True, no_change=True,
            )
        df = combined
        merged = True
    elif mode == "replace" and len(df) > n:
        df = df.iloc[-n:].reset_index(drop=True)  # 显式覆盖：只保留本次最近 n 根

    df.to_parquet(out, index=False)

    # sidecar 快照最终 bars/日期区间（供「下载历史」面板零 IO 展示）
    start_date = datetime.datetime.fromtimestamp(
        int(df["time"].iloc[0]), datetime.timezone.utc
    ).strftime("%Y-%m-%d")
    end_date = datetime.datetime.fromtimestamp(
        int(df["time"].iloc[-1]), datetime.timezone.utc
    ).strftime("%Y-%m-%d")
    write_parquet_meta(out, sym, tf, source, bars=int(len(df)),
                       start_date=start_date, end_date=end_date)

    prev_n = int(len(old)) if merged else 0  # 无既有档案（首下/覆盖）时全部视为新增
    info = _archive_info_from_df(
        df, sym, tf, out, fetched_bars=fetched_bars,
        added_bars=int(len(df)) - prev_n,
        merged=merged, no_change=False,
    )
    # 补充 inspect 口径字段（写入后文件即权威，但保留内存值避免二次读盘）
    info["start_date"] = start_date
    info["end_date"] = end_date
    if mode == "merge" and old is None and had_archive:
        # 档案存在但读不出（损坏/缺列）→ 无法安全增量，显式标记防静默覆盖
        info["archive_unreadable"] = True
    return info


def backfill_to_origin(
    symbol: str,
    timeframe: str,
    source: str = "binance",
    page_bars: int | None = None,
    progress_cb: Callable[[int], None] | None = None,
    phase_cb: Callable[[str], None] | None = None,
    max_pages: int = 500,
) -> dict[str, Any]:
    """回溯到数据源留存起点：反复向档案最早 bar 之前翻页，直到无更早数据。

    仅支持可时间锚定翻页的数据源（binance endTime / okx after）。每页落盘
    （中断可续），返回信息含 backfilled_bars / pages / reached_origin。

    progress_cb(bars) 汇报累计已拉取根数；phase_cb(text) 汇报阶段（如页数）。
    """
    sym = validate_symbol(symbol)
    tf = validate_timeframe(timeframe)
    kind = (source or "binance").strip().lower()
    if kind not in SUPPORTED_SOURCES:
        raise ValueError(f"不支持的数据源: {source or '(空)'}；可选: {', '.join(SUPPORTED_SOURCES)}")
    if kind not in ("binance", "okx"):
        raise ValueError(
            f"{SOURCE_LABELS.get(kind, kind)} 单次窗口上限，无法向更早历史翻页；"
            f"请改用 binance / okx"
        )
    page = min(max(int(page_bars or DEFAULT_BARS), MIN_BARS), bars_limit_for(kind, tf))

    out = _archive_path(sym, tf)
    archive = _read_archive(out)
    total_fetched = 0
    backfilled = 0
    pages = 0
    reached_origin = False

    if archive is None or archive.empty:
        # 无档案：先拉尾部窗口建立档案（第 1 页）
        if phase_cb:
            phase_cb("拉取初始窗口")
        df = _fetch_source_bars(kind, sym, tf, page, progress_cb=progress_cb)
        if df is None or df.empty:
            raise RuntimeError(f"数据源未返回任何 K 线：{symbol}")
        total_fetched += int(len(df))
        if progress_cb:
            progress_cb(total_fetched)
        info = _save_and_inspect(df, sym, tf, page, source=kind, mode="merge", old_df=None)
        backfilled += int(info["added_bars"])
        pages = 1
        archive = df
    else:
        info = _archive_info_from_df(
            archive, sym, tf, out, fetched_bars=0,
            added_bars=0, merged=True, no_change=True,
        )

    while pages < max_pages:
        old_min = int(archive["time"].min())
        if phase_cb:
            phase_cb(f"回溯更早历史（第 {pages + 1} 页，锚定 {datetime.datetime.fromtimestamp(old_min, datetime.timezone.utc).date()} 之前）")
        extra = _fetch_source_bars(
            kind, sym, tf, page, progress_cb=progress_cb,
            anchor_ms=old_min * 1000 - 1,  # 严格早于档案最早 bar
        )
        if extra is None or extra.empty:
            reached_origin = True
            break
        total_fetched += int(len(extra))
        if progress_cb:
            progress_cb(total_fetched)
        prev = archive
        combined = _merge_dfs(archive, extra)
        added = int(len(combined)) - int(len(archive))
        if added == 0:
            reached_origin = True
            break
        archive = combined
        backfilled += added
        pages += 1
        # 每页落盘：中断后可续跑，且 sidecar 区间随档案前移
        info = _save_and_inspect(archive, sym, tf, page, source=kind,
                                 mode="merge", old_df=prev)

    if pages >= max_pages:
        info["truncated_pages"] = True
    info.update({
        "pages": pages,
        "backfilled_bars": backfilled,
        "reached_origin": bool(reached_origin),
        "mode": "backfill",
    })
    return info


def _maybe_extend_backward(
    kind: str, sym: str, tf: str, n: int,
    df: pd.DataFrame, old: pd.DataFrame,
    progress_cb: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """增量扩展（binance / okx）：本次拉取若完全落在档案范围内（尾部重叠），
    再向档案最早 bar 之前拉一段，让重复下载真正累积更早历史。
    其余数据源（单次窗口上限）无法翻到更早，直接返回原结果。
    """
    if kind not in ("binance", "okx") or df is None or df.empty or old is None or old.empty:
        return df
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    old_min = int(old["time"].min())
    if int(df["time"].min()) < old_min:
        # 本次尾巴已覆盖到档案更早处 → 无需再向后补
        return df
    try:
        extra = _fetch_source_bars(
            kind, sym, tf, n, progress_cb=progress_cb,
            anchor_ms=old_min * 1000 - 1,  # 严格早于档案最早 bar
        )
    except Exception as exc:  # noqa: BLE001 扩展失败不阻断主结果
        logger.warning("backward extend failed for %s %s: %s", sym, tf, exc)
        return df
    if extra is None or extra.empty:
        return df  # 已到数据源留存起点
    merged = _merge_dfs(df, extra)
    logger.info(
        "[增量] %s %s 向后扩展 %d 根（档案最早 %s 之前）",
        sym, tf, int(len(merged)) - int(len(df)),
        datetime.datetime.fromtimestamp(old_min, datetime.timezone.utc).date(),
    )
    return merged


def download_symbol_bars(
    symbol: str,
    timeframe: str,
    source: str = "tradingview",
    n_bars: int | None = None,
    mode: str = "merge",
    progress_cb: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """增量归档：拉取 → 合并进已有 parquet → 刷新 sidecar 区间。

    与已有档案合并时只追加新 bar（重复下载同一窗口不重写文件）；
    binance 源在尾部窗口重叠时会锚定档案最早 bar 之前继续向后扩展。
    返回与选文件接口相同的检查信息（附 added_bars / no_change 标记）。
    """
    sym = validate_symbol(symbol)
    tf = validate_timeframe(timeframe)
    kind = (source or "tradingview").strip().lower()
    if kind not in SUPPORTED_SOURCES:
        raise ValueError(f"不支持的数据源: {source or '(空)'}；可选: {', '.join(SUPPORTED_SOURCES)}")
    if mode not in ("merge", "replace"):
        raise ValueError(f"不支持的写入模式: {mode}；可选 merge / replace")
    n = min(max(int(n_bars or DEFAULT_BARS), MIN_BARS), bars_limit_for(kind, tf))

    out = _archive_path(sym, tf)
    old = _read_archive(out) if mode == "merge" else None

    df = _fetch_source_bars(kind, sym, tf, n, progress_cb=progress_cb)
    if mode == "merge" and old is not None:
        df = _maybe_extend_backward(kind, sym, tf, n, df, old, progress_cb=progress_cb)

    info = _save_and_inspect(df, sym, tf, n, source=kind, mode=mode, old_df=old)
    meta = read_parquet_meta(Path(info["data_file"]))
    if meta:
        info["download_source"] = meta.get("source")
        info["downloaded_at"] = meta.get("downloaded_at")
    return info


# ---------- 后台下载任务（队列 + 磁盘持久化，Web UI 轮询进度） ----------

logger = logging.getLogger(__name__)

# 任务注册表落盘位置（重启后：完成的任务结果仍在；排队中的自动恢复；
# 进行中的任务「自动续跑」——增量归档把每次落盘的 parquet 当作检查点：
#   merge/backfill 的写入都是幂等去重合并，续跑时以档案为断点接着向后补，
#   不再需要人工重新提交（此前标记为“已中断”的死路）。
# 连续多次重启仍没跑完的任务（resume_attempts 达到上限）才标为失败。
JOBS_FILE = PROJECT_ROOT / "data" / "download_jobs.json"
_MAX_KEPT = 50
_MAX_RESUME_ATTEMPTS = 3   # 重启自动续跑上限：超过说明网络/服务不稳，不再自动重试
_PERSIST_TICK_INTERVAL = 2.0  # bars_fetched 高频变化时的落盘节流（秒）


@dataclass
class DownloadJob:
    job_id: str
    symbol: str
    timeframe: str
    source: str
    n_bars: int
    mode: str = "merge"          # merge（合并追加）/ replace（覆盖）/ backfill（回溯到数据源起点）
    status: str = "queued"      # queued / running / done / error
    phase: str = "排队中"
    bars_fetched: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    resume_attempts: int = 0      # 已自动续跑次数（重启恢复 running 任务时 +1）
    _last_persist: float = 0.0

    def snapshot(self, position: int = 0) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "phase": self.phase,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source,
            "n_bars": self.n_bars,
            "bars_fetched": self.bars_fetched,
            "position": position,
            "elapsed": round(time.time() - self.started_at, 1),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "resume_attempts": self.resume_attempts,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source,
            "n_bars": self.n_bars,
            "mode": self.mode,
            "status": self.status,
            "phase": self.phase,
            "bars_fetched": self.bars_fetched,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "resume_attempts": self.resume_attempts,
        }


_JOBS: dict[str, DownloadJob] = {}
_QUEUE: list[DownloadJob] = []
_QUEUE_LOCK = threading.Condition()
_LOADED = False
_RESTORE_LOCK = threading.Lock()


def _persist() -> None:
    """将任务注册表写到磁盘（原子替换）。"""
    try:
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = JOBS_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({j.job_id: j.to_dict() for j in _JOBS.values()}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(JOBS_FILE)
    except Exception as exc:  # noqa: BLE001 持久化失败不阻塞任务
        logger.warning("persist download jobs failed: %s", exc)


def _resumeable(job: DownloadJob) -> bool:
    """重启后可否自动续跑：增量/覆盖写入都幂等（merge/backfill 以档案为断点
    去重续补；replace 重拉同一尾部覆盖），无需人工介入。"""
    return bool(job.symbol and job.timeframe and job.n_bars >= MIN_BARS)


def _restore_from_disk() -> None:
    """启动时恢复任务注册表：done/error 原样保留；queued 重新入队自动续跑；

    运行中（running）的任务不再标死为「已中断」——转为排队自动续跑
    （resume_attempts +1，最多 _MAX_RESUME_ATTEMPTS 次；达到上限才标失败，
    因为多次重启仍无法完成多半是网络/服务本身的问题）。
    """
    global _LOADED
    with _RESTORE_LOCK:
        if _LOADED:
            return
        _LOADED = True
    if not JOBS_FILE.exists():
        return
    try:
        raw = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    revived: list[DownloadJob] = []
    for data in raw.values():
        if not isinstance(data, dict):
            continue
        job = DownloadJob(
            job_id=str(data.get("job_id") or uuid.uuid4().hex[:12]),
            symbol=str(data.get("symbol") or ""),
            timeframe=str(data.get("timeframe") or ""),
            source=str(data.get("source") or "tradingview"),
            n_bars=int(data.get("n_bars") or 5000),
        )
        mode_v = str(data.get("mode") or "merge")
        job.mode = mode_v if mode_v in ("merge", "replace", "backfill") else "merge"
        job.status = str(data.get("status") or "error")
        job.phase = str(data.get("phase") or "")
        job.bars_fetched = int(data.get("bars_fetched") or 0)
        job.started_at = float(data.get("started_at") or time.time())
        job.finished_at = data.get("finished_at")
        job.message = str(data.get("message") or "")
        job.result = data.get("result")
        job.error = data.get("error")
        job.resume_attempts = int(data.get("resume_attempts") or 0)
        revived.append(job)

    with _QUEUE_LOCK:
        for job in revived:
            _JOBS[job.job_id] = job
            if job.status == "queued":
                job.phase = "排队中"
                _QUEUE.append(job)  # 进程内重新排队，worker 启动后自动续跑
            elif job.status == "running":
                if _resumeable(job) and job.resume_attempts < _MAX_RESUME_ATTEMPTS:
                    # 自动续跑：merge/backfill 以已落盘档案为断点接着向后补
                    job.status = "queued"
                    job.resume_attempts += 1
                    job.error = None
                    job.phase = (
                        f"排队中（重启自动续跑 #{job.resume_attempts}，"
                        "以已有档案为断点）"
                    )
                    job.message = "上次运行被服务重启中断，已自动重新入队续跑"
                    _QUEUE.append(job)
                else:
                    job.status = "error"
                    job.phase = "已中断"
                    job.error = (
                        "服务重启中断了下载，且自动续跑已达上限"
                        f"（{_MAX_RESUME_ATTEMPTS} 次），请检查网络后重新提交"
                    )
                    job.finished_at = time.time()
        # 清理超出上限的已完成任务
        stale = [j for j in _JOBS.values() if j.status in ("done", "error")]
        for j in stale[: -_MAX_KEPT]:
            _JOBS.pop(j.job_id, None)
        _persist()
    _ensure_worker()


_WORKER_STARTED = False


def _ensure_worker() -> None:
    """惰性启动单一工作线程（按队列顺序逐个执行）。"""
    global _WORKER_STARTED
    with _QUEUE_LOCK:
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True
    threading.Thread(target=_worker_loop, daemon=True).start()


def _worker_loop() -> None:
    while True:
        with _QUEUE_LOCK:
            while not _QUEUE:
                _QUEUE_LOCK.wait()
            job = _QUEUE.pop(0)
        try:
            _run_job(job)
        except Exception as exc:  # noqa: BLE001 任务级兜底（_run_job 内部已兜，此为最后防线）
            logger.error("download job %s crashed: %s", job.job_id, exc)


def get_download_job(job_id: str) -> DownloadJob | None:
    with _QUEUE_LOCK:
        return _JOBS.get(job_id)


def list_download_jobs(limit: int = 30) -> list[dict[str, Any]]:
    """按「运行/排队 → 最近完成」顺序返回任务快照（含排队位置）。"""
    _restore_from_disk()
    with _QUEUE_LOCK:
        positions = {id(j): pos for pos, j in enumerate(_QUEUE, 1)}
        active: list[DownloadJob] = list(_QUEUE)
        for job in _JOBS.values():
            if job.status in ("queued", "running") and id(job) not in positions:
                active.append(job)  # 正在运行的任务（已被 worker 弹出队列）
        ordered = [job.snapshot(position=positions.get(id(job), 0)) for job in active]
        finished = [
            job.snapshot()
            for job in _JOBS.values()
            if job.status in ("done", "error") and id(job) not in positions
        ]
        finished.sort(key=lambda s: s.get("finished_at") or 0, reverse=True)
        return (ordered + finished)[:limit]


def submit_download(
    symbol: str, timeframe: str, source: str, n_bars: int, mode: str = "merge"
) -> str:
    """提交后台下载任务，立即返回 job_id；已有任务在跑时自动排队（串行执行）。"""
    _restore_from_disk()
    job = DownloadJob(
        job_id=uuid.uuid4().hex[:12],
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        n_bars=n_bars,
        mode=mode if mode in ("merge", "replace", "backfill") else "merge",
    )
    with _QUEUE_LOCK:
        _JOBS[job.job_id] = job
        _QUEUE.append(job)
        _QUEUE_LOCK.notify()  # worker 空闲则立刻执行，否则排队
        _persist()
    _ensure_worker()
    return job.job_id


def _run_job(job: DownloadJob) -> None:
    try:
        with _QUEUE_LOCK:
            job.status = "running"
            job.started_at = time.time()
            job.phase = "拉取 K 线（分页）" if job.source in ("binance", "okx", "tongdaxin") else "拉取 K 线"
            _persist()

        def _progress(bars: int) -> None:
            job.bars_fetched = max(job.bars_fetched, bars)
            now = time.time()
            if now - job._last_persist > _PERSIST_TICK_INTERVAL:
                job._last_persist = now
                _persist()

        def _phase(text: str) -> None:
            job.phase = text
            _persist()

        if job.mode == "backfill":
            info = backfill_to_origin(
                job.symbol, job.timeframe, job.source, page_bars=job.n_bars,
                progress_cb=_progress, phase_cb=_phase,
            )
        else:
            info = download_symbol_bars(
                job.symbol, job.timeframe, job.source, job.n_bars,
                mode=job.mode, progress_cb=_progress,
            )
        info["download_source"] = job.source
        meta = read_parquet_meta(info.get("data_file") or "")
        info["downloaded_at"] = (meta or {}).get("downloaded_at")
        try:
            from web.progress import save_settings

            save_settings({"last_data_file": info["data_file"]})
        except Exception:  # noqa: BLE001 设置保存失败不阻塞结果
            pass
        job.result = {"ok": True, "cancelled": False, "source": job.source, **info}
        job.status = "done"
        job.phase = "完成"
        job.finished_at = time.time()
        if job.mode == "backfill":
            job.message = (
                f"回溯完成：新增 {info.get('backfilled_bars', 0):,} 根，档案共 "
                f"{info['n_bars']:,} 根（{info.get('pages', 0)} 页）"
                + ("，已达数据源起点" if info.get("reached_origin") else "")
            )
        elif info.get("no_change"):
            job.message = f"已是最新，无新增（档案共 {info['n_bars']:,} 根，未改动文件）"
        elif job.mode == "replace":
            job.message = f"已覆盖为最近 {info['n_bars']:,} 根 K 线"
        elif info.get("added_bars", 0) > 0:
            job.message = f"增量追加 {info['added_bars']:,} 根（档案共 {info['n_bars']:,} 根）"
        else:
            job.message = f"已下载 {info['n_bars']:,} 根 K 线"
        _persist()
    except Exception as exc:  # noqa: BLE001 任务级兜底
        job.status = "error"
        job.error = str(exc)
        job.phase = "失败"
        job.finished_at = time.time()
        logger.error("download job %s failed: %s", job.job_id, exc)
        _persist()
    finally:
        with _QUEUE_LOCK:
            _QUEUE_LOCK.notify_all()  # 唤醒 worker 处理下一个排队任务
