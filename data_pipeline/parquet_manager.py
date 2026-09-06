"""Load training data from a single Parquet K-line file."""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

from config import Config
from model_core.features import FeatureEngineer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data" / "training"

# 切片文件目录名（位于其中的 parquet 视为局部窗口/切片，区别于同名的全量档案）
SLICE_DIR_NAMES = ("slices", "slice", "局部", "切片")
# 文件名显式带切片标记也视为切片（如 BTCUSDT_H1_slice.parquet）
SLICE_NAME_MARKERS = ("_slice", "_part", "_window")

# Canonical labels used across the project
_TIMEFRAMES = ("M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1", "D3", "W1", "MN1")

# Filename suffix aliases → canonical (case-insensitive keys)
_TF_ALIASES: dict[str, str] = {
    # M1
    "m1": "M1",
    "1m": "M1",
    "1min": "M1",
    "min1": "M1",
    # M3
    "m3": "M3",
    "3m": "M3",
    "3min": "M3",
    "min3": "M3",
    # M5
    "m5": "M5",
    "5m": "M5",
    "5min": "M5",
    "min5": "M5",
    # M15
    "m15": "M15",
    "15m": "M15",
    "15min": "M15",
    "min15": "M15",
    # M30
    "m30": "M30",
    "30m": "M30",
    "30min": "M30",
    "min30": "M30",
    # H1
    "h1": "H1",
    "1h": "H1",
    "60m": "H1",
    "60min": "H1",
    "min60": "H1",
    "60": "H1",
    # H2
    "h2": "H2",
    "2h": "H2",
    "120m": "H2",
    "120min": "H2",
    # H4
    "h4": "H4",
    "4h": "H4",
    "240m": "H4",
    "240min": "H4",
    "min240": "H4",
    "240": "H4",
    # H6
    "h6": "H6",
    "6h": "H6",
    "360m": "H6",
    "360min": "H6",
    # H8
    "h8": "H8",
    "8h": "H8",
    "480m": "H8",
    "480min": "H8",
    # H12
    "h12": "H12",
    "12h": "H12",
    "720m": "H12",
    "720min": "H12",
    # D1
    "d1": "D1",
    "1d": "D1",
    "day": "D1",
    "daily": "D1",
    "1440m": "D1",
    "1440min": "D1",
    # D3
    "d3": "D3",
    "3d": "D3",
    "3day": "D3",
    # W1
    "w1": "W1",
    "1w": "W1",
    "week": "W1",
    "weekly": "W1",
    # MN1 (month) — avoid bare "1m" which already maps to M1
    "mn1": "MN1",
    "1mo": "MN1",
    "1mon": "MN1",
    "month": "MN1",
    "monthly": "MN1",
}


def normalize_timeframe_token(token: str) -> str | None:
    """Map a filename timeframe token to canonical M1/M5/.../MN1."""
    raw = (token or "").strip()
    if not raw:
        return None
    key = raw.lower().replace("-", "").replace("_", "")
    if key in _TF_ALIASES:
        return _TF_ALIASES[key]
    upper = raw.upper()
    if upper in _TIMEFRAMES:
        return upper
    return None


def _compute_target_ret(open_tensor: torch.Tensor) -> torch.Tensor:
    """计算目标收益率张量（原数据管理器的静态方法内联至此）。

    target_ret[n, t] = log(open[n, t+2] / open[n, t+1])，对 t ∈ [0, T-3]
    最后两个位置（t = T-2, T-1）设为 0（边界）。
    """
    n, t = open_tensor.shape
    target = torch.zeros(n, t, dtype=torch.float32)

    if t >= 3:
        numerator = open_tensor[:, 2:]      # [N, T-2]
        denominator = open_tensor[:, 1:-1]  # [N, T-2]
        safe_denom = denominator.clone()
        safe_denom[safe_denom == 0] = 1.0
        log_ret = torch.log(numerator / safe_denom)  # [N, T-2]
        target[:, : t - 2] = log_ret
    return target


def parse_parquet_filename(path: str | Path) -> tuple[str, str]:
    """Parse ``{symbol}_{timeframe}.parquet``.

    Accepts canonical suffixes (``H1``) and common aliases (``60min``, ``1h``, ``5m``…).
    Examples: ``AAPL_H1.parquet``, ``002008_60min.parquet``, ``BTCUSDT_1h.parquet``.
    """
    name = Path(path).name
    if Path(path).suffix.lower() != ".parquet":
        raise ValueError(f"请选择 .parquet 文件；当前: {name}")
    stem = Path(path).stem
    if "_" not in stem:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet / 002008_60min.parquet；"
            f"当前: {name}"
        )
    symbol, tf_raw = stem.rsplit("_", 1)
    symbol = symbol.strip()
    timeframe = normalize_timeframe_token(tf_raw)
    if not symbol or timeframe is None:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet / 002008_60min.parquet；"
            f"支持周期别名: H1/60min/1h, M5/5min, D1/1d …；当前: {name}"
        )
    return symbol, timeframe


def _strip_slice_marker(name: str) -> str:
    """剥离文件名中的切片标记（_slice/_part/_window），用于品种/周期解析。

    例：``ETHUSDT_H1_slice.parquet`` → ``ETHUSDT_H1.parquet``
    """
    out = name
    for marker in SLICE_NAME_MARKERS:
        out = out.replace(marker, "")
    return out


def slice_info_for(
    p: Path, symbol: str, timeframe: str
) -> dict[str, Any]:
    """判断 parquet 是否为「切片/局部窗口」文件，并尝试定位其对应的全量档案。

    判定规则（目录或文件名命中任一标记）：
    - 路径任一段 ∈ {slices, slice, 局部, 切片}
    - 文件名含 _slice / _part / _window 标记

    全量档案定位：优先找 ``data/training/{symbol}_{timeframe}.parquet``（下载档案统一目录）；
    找不到时不报错，只返回 is_slice=True（卡片提示仍可用）。
    """
    parts = [part.lower() for part in p.parts]
    is_slice = any(part in SLICE_DIR_NAMES for part in parts) or any(
        marker in p.stem.lower() for marker in SLICE_NAME_MARKERS
    )
    if not is_slice:
        return {"is_slice": False}
    out: dict[str, Any] = {"is_slice": True}
    # 全量档案：data/training/{symbol}_{tf}.parquet（同一命名规则）
    full = TRAINING_DIR / f"{symbol}_{timeframe}.parquet"
    if full.exists() and full.resolve() != p.resolve():
        try:
            import pyarrow.parquet as pq

            full_bars = pq.read_metadata(full).num_rows
            out["slice_of"] = {
                "file": full.name,
                "data_file": str(full.resolve()),
                "bars": int(full_bars),
            }
        except Exception:  # noqa: BLE001 全量档案元数据读失败不影响切片标注
            out["slice_of"] = {"file": full.name, "data_file": str(full.resolve())}
    return out


def _read_parquet_light(path: Path) -> pd.DataFrame:
    """轻量读：优先 footer 元数据取行数/列名 + 只读 time 单列。

    检查/验证类调用（如 /api/data-file/browse）只需要行数、列名、
    time 列的 min/max/去重——全量读大文件会白白耗时数秒到数十秒。
    任一步失败（无 pyarrow / 元数据异常）回退到全量读，行为与旧版一致。
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415

        pf = pq.ParquetFile(str(path))
        names = list(pf.schema_arrow.names)
        num_rows = int(pf.metadata.num_rows)
        if "time" in names:
            df = pd.read_parquet(path, columns=["time"])
            if len(df) != num_rows:  # 列数异常时回退全量读
                raise ValueError("row count mismatch")
            for c in names:
                if c != "time":
                    df[c] = None
            return df
        # 无 time 列：仍用元数据行数构造（缺列检查交给调用方）
        return pd.DataFrame({c: pd.Series(dtype="float64", index=range(num_rows))
                             for c in names})
    except Exception:  # noqa: BLE001 回退全量读（与旧版行为一致）
        return pd.read_parquet(path)


def _inspect_parquet_file_impl(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.suffix.lower() != ".parquet":
        raise ValueError("请选择 .parquet 文件")

    # 切片文件名带 _slice/_part 标记：先剥离再解析品种/周期（如 ETHUSDT_H1_slice.parquet）
    symbol, timeframe = parse_parquet_filename(
        p.with_name(_strip_slice_marker(p.name))
    )
    # 快速路径：只读 Parquet footer 元数据 + time 单列（大文件从全量读 ~秒/十秒级
    # 降到 ~毫秒级，避免 /api/data-file/browse 验证路径时把请求拖到前端超时中止）。
    # 需要的数据只依赖 行数/列名/time 列的 min/max/去重 —— 全部可用元数据+单列算出来。
    df = _read_parquet_light(p)
    bars = len(df)
    if bars < Config.MIN_BARS:
        raise ValueError(
            f"数据不足: {bars} bars（至少需要 {Config.MIN_BARS}）"
        )

    # 从实际时间跨度计算年数（适用于所有周期，比固定公式更准确）
    years = None
    if "time" in df.columns and len(df) > 1:
        try:
            t_min = float(df["time"].min())
            t_max = float(df["time"].max())
            if t_max > t_min and t_max > 1_000_000_000:  # 合法的 Unix 时间戳
                span_seconds = t_max - t_min
                years = round(span_seconds / (365.25 * 24 * 3600), 2)
        except Exception:
            pass
    # 回退：H1 用固定公式（6240 根/年，24h 市场）
    if years is None and timeframe == "H1":
        years = round(bars / 6240, 2)

    # ── 检查摘要：去重后根数 / 缺列 / 重复时间戳 / 时间跨度 ──────────
    required_base = {"time", "open", "high", "low", "close"}
    has_vol = "volume" in df.columns or "tick_volume" in df.columns
    missing_columns = sorted(c for c in required_base if c not in df.columns)
    if not has_vol:
        missing_columns.append("volume")  # volume / tick_volume 任一即可
    bars_unique = raw_bars = bars
    duplicate_ts = 0
    if "time" in df.columns:
        bars_unique = int(df["time"].nunique())
        duplicate_ts = raw_bars - bars_unique
    start_date = end_date = None
    if "time" in df.columns and len(df) > 0:
        try:
            t_min = int(df["time"].min())
            t_max = int(df["time"].max())
            if t_max > 1_000_000_000:  # 合法 Unix 秒
                start_date = datetime.datetime.fromtimestamp(
                    t_min, datetime.timezone.utc
                ).strftime("%Y-%m-%d")
                end_date = datetime.datetime.fromtimestamp(
                    t_max, datetime.timezone.utc
                ).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            pass
    checks_ok = bool(
        duplicate_ts == 0 and not missing_columns
        and bars_unique > 0 and start_date and end_date
    )
    return {
        "data_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
        "raw_bars": raw_bars,
        "bars_unique": bars_unique,
        "duplicate_ts": duplicate_ts,
        "missing_columns": missing_columns,
        "start_date": start_date,
        "end_date": end_date,
        "checks_ok": checks_ok,
        "years_h1": years,
        "valid": True,
        "message": "",
        **slice_info_for(p, symbol, timeframe),
    }


# 文件检查结果缓存：按 (resolved path, mtime_ns, size) 作 key。UI 每 ~4s 轮询
# /api/overview 会反复 inspect 同一数据文件；文件不变时直接命中，避免每次读
# parquet footer/time 列。文件被改写(下载续写/切片)时 mtime/size 变化即失效。
_parquet_inspect_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
_PARQUET_INSPECT_MAX = 128


def invalidate_parquet_inspect_cache() -> None:
    _parquet_inspect_cache.clear()


def inspect_parquet_file(path: str | Path) -> dict[str, Any]:
    """mtime 缓存包装：数据文件不变时直接复用上次检查结果（行为等价）。"""
    p = Path(path)
    key: tuple[str, int, int] | None = None
    if p.exists():
        try:
            st = p.stat()
            key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
        except OSError:
            key = None
    if key is not None:
        hit = _parquet_inspect_cache.get(key)
        if hit is not None:
            return hit
    res = _inspect_parquet_file_impl(path)
    if key is not None:
        if len(_parquet_inspect_cache) >= _PARQUET_INSPECT_MAX:
            _parquet_inspect_cache.clear()
        _parquet_inspect_cache[key] = res
    return res


class ParquetDataManager:
    """Single-symbol data manager backed by one Parquet file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.symbol, self.timeframe = parse_parquet_filename(self.file_path)
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None
        self.fingerprint: str | None = None   # 数据版本指纹（P0：同版本 holdout 只消费一次）

    def load(self) -> None:
        df = pd.read_parquet(self.file_path)
        if len(df) < Config.MIN_BARS:
            raise ValueError(
                f"数据不足: {len(df)} bars（至少需要 {Config.MIN_BARS}）"
            )

        volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        required = ["time", "open", "high", "low", "close", volume_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet 缺少列: {missing}")

        sub = df[required].copy().rename(columns={volume_col: "volume"})

        # 兼容性修复：某些 A 股 parquet 导出工具把 Unix 秒时间戳误存为
        # "秒/1000"（数值被缩小 1000 倍，导致日期变成 1970 年）。
        # 若最大时间戳 < 1e7（1970-04-27 之前），则视为被除过 1000，乘回。
        if pd.api.types.is_numeric_dtype(sub["time"]) and sub["time"].max() < 10_000_000:
            sub["time"] = sub["time"] * 1000
            logger.info(
                f"[数据] {self.file_path.name} 时间戳被识别为秒/1000，"
                f"已乘 1000 恢复为 Unix 秒。"
            )

        sub = sub.sort_values("time")
        sub = sub[~sub["time"].duplicated(keep="last")]

        rows = {field: sub[field].values for field in ["open", "high", "low", "close", "volume"]}
        import numpy as np

        raw: dict[str, torch.Tensor] = {
            field: torch.tensor(np.array([rows[field]]), dtype=torch.float32)
            for field in ["open", "high", "low", "close", "volume"]
        }
        raw["time"] = torch.tensor(
            np.array([sub["time"].values.astype("int64")]),
            dtype=torch.int64,
        )

        self._raw_dict = raw
        self._target_ret = _compute_target_ret(raw["open"])
        # P0 数据指纹：行数 + 首末时间 + OHLC 内容 hash。数据一变指纹即变，
        # 同版本重训不会重置 holdout 单次消费状态。
        try:
            import hashlib

            t = raw["time"]
            h = hashlib.sha256()
            h.update(str(int(t.shape[1])).encode())
            h.update(str(int(t[0, 0])).encode())
            h.update(str(int(t[0, -1])).encode())
            for field in ["open", "high", "low", "close", "volume"]:
                h.update(raw[field].detach().cpu().numpy().tobytes())
            self.fingerprint = h.hexdigest()[:32]
        except Exception:  # noqa: BLE001 指纹失败不阻断数据加载
            self.fingerprint = None
        logger.info(
            f"[数据] 已加载 {self.symbol} {self.timeframe}，"
            f"共 {raw['open'].shape[1]} 根K线，文件 {self.file_path.name}"
        )

    def data_source_meta(self) -> dict[str, Any] | None:
        """数据溯源标记：本策略基于哪个 parquet 训出（symbol/timeframe/文件/日期跨度）。

        供训练写入 strategies/best_*.json 的 ``data_source`` 字段（策略表溯源展示）。
        ``load()`` 已执行时附带 bars / start / end / fingerprint；否则只给文件名级信息。

        Returns:
            {symbol, timeframe, file, data_file, bars?, start?, end?, fingerprint?}，
            文件不可用时返回 None。
        """
        file_path = Path(self.file_path)
        meta: dict[str, Any] = {
            "file": file_path.name,
            "data_file": str(file_path),
        }
        if self.symbol:
            meta["symbol"] = self.symbol
        if self.timeframe:
            meta["timeframe"] = self.timeframe
        if self._raw_dict is not None:
            t = self._raw_dict.get("time")
            if t is not None and t.numel() > 1:
                import datetime

                try:
                    t0 = int(t[0, 0])
                    t1 = int(t[0, -1])
                    meta["bars"] = int(t.shape[1])
                    meta["start"] = datetime.datetime.fromtimestamp(
                        t0, datetime.timezone.utc
                    ).isoformat()
                    meta["end"] = datetime.datetime.fromtimestamp(
                        t1, datetime.timezone.utc
                    ).isoformat()
                except (OverflowError, OSError, ValueError):
                    pass  # 时间戳异常不影响溯源标记其余字段
        if self.fingerprint:
            meta["fingerprint"] = self.fingerprint
        return meta

    @property
    def symbols(self) -> list[str]:
        return [self.symbol]

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        if self._raw_dict is None:
            raise RuntimeError("Call load() first")
        return self._raw_dict

    @property
    def feat_tensor(self) -> torch.Tensor:
        return FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self) -> torch.Tensor:
        if self._target_ret is None:
            raise RuntimeError("Call load() first")
        return self._target_ret

    @property
    def bar_time(self) -> torch.Tensor:
        raw = self.raw_dict
        if "time" in raw:
            return raw["time"][:, -1].long()
        return torch.zeros(1, dtype=torch.int64)
