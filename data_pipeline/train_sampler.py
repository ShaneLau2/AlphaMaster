"""训练子集生成：把整段历史缩减为可控的训练样本。

需求背景（训练 UI「训练数据范围」）：
- 用户不想每次都用下载的全量（如 100 万根）训练：特征/搜索开销大，且老年代
  行情的分布可能主导最终因子。
- 提供两种成熟做法：
  1. tail   最近 N 根（连续尾部切片）——「近期市场更像未来」，统计上最干净。
  2. spread 全历史分块抽取 N 根——按年代抽若干**足够长的连续块**，把不同市场
     形态（牛/熊/震荡、高/低波动）都纳入样本，避免模型只见过最近一种行情。
     - regime="equal"（默认保底）：各年代等长取块；
     - regime="vol"（默认）：块锚点按**滚动波动率分位**（low/mid/high）轮转选取，
       让低/中/高波动年代都进样本；
     - regime="trend"：按滚动收益（趋势强度）分位选块。

为什么不能随机抽互不相邻的单根 K 线：
本项目的特征全部是滚动/因果计算（FeatureEngineer + VM 滚动归一化，回看约
700~800 根）。跨时间缺口随机挑单根会让滚动统计失真，等于篡改数据语义。
时间序列 ML 的成熟做法（随机窗口/分块采样）是保持连续性：取若干连续块，每块
之前垫一段同年代真实行情作 warm-up（LOOKBACK），使块内滚动特征从头就有效。
本模块按此实现，且完全确定性（同样的参数 → 同样的文件 → 同样的数据指纹）。

溯源：每次生成子集都会在子集目录旁写 train_range.json（模式/参数/年代范围/
regime 覆盖），train_file._save_strategy 会把它并入策略 JSON 的 train_range 字段，
供 UI 与日后对比不同抽样训出的因子。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模式
MODE_FULL = "full"      # 全部数据（现状，默认）
MODE_TAIL = "tail"      # 最近 N 根（连续尾部切片）
MODE_SPREAD = "spread"  # 全历史分块抽取 N 根

# spread 的块选取方式
REGIME_EQUAL = "equal"  # 各年代等长取块（旧的纯等分行为）
REGIME_VOL = "vol"      # 按滚动波动率分位选块（默认）
REGIME_TREND = "trend"  # 按滚动收益/趋势分位选块
REGIMES = (REGIME_VOL, REGIME_TREND, REGIME_EQUAL)
DEFAULT_REGIME = REGIME_VOL

# 特征回看长度：给每块垫这么长的同年代前置行情，保证块内滚动特征有效
# （特征 warm-up ~360 + VM 滚动归一化 500 → 1000 保险）
LOOKBACK = 1000
# 单块核心长度下限（太短则样本少且垫片占比高）
CORE_MIN = 2000
# 结尾「近期连续段」下限：walk-forward 验证 + 尾部 holdout(500) 必须落在
# 真正干净的连续区间内，故在文件末尾保留一段足够长的近期行情。
RECENT_MIN = 4000
MAX_CHUNKS = 16
# regime 判定用的滚动窗口（对数收益窗口，按 bar 计）
REGIME_WINDOW = 50
SIDECAR_NAME = "train_range.json"


@dataclass
class SubsetResult:
    mode: str
    data_file: str            # 实际用于训练的文件（full 模式=源文件本身）
    subset: bool              # 是否生成了新子集文件
    source_file: str
    source_bars: int
    subset_bars: int
    n_bars_requested: int | None
    n_chunks: int | None
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "data_file": self.data_file,
            "subset": self.subset,
            "source_file": self.source_file,
            "source_bars": self.source_bars,
            "subset_bars": self.subset_bars,
            "n_bars_requested": self.n_bars_requested,
            "n_chunks": self.n_chunks,
            **self.meta,
        }


def _read_sorted(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if len(df) < 1:
        raise ValueError(f"数据文件为空: {path.name}")
    if "time" in df.columns:
        df = df.sort_values("time")
        df = df[~df["time"].duplicated(keep="last")]
    return df.reset_index(drop=True)


def _tail_slice(df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    return df.tail(n_bars).reset_index(drop=True)


# ── regime 度量与分桶 ─────────────────────────────────────────────────────
def _regime_classes(closes: np.ndarray, regime: str) -> np.ndarray:
    """给每根 bar 一个 0/1/2 分位类（按 metric 在有效区间内的三分位）。

    vol:   滚动波动率（对数收益 std，REGIME_WINDOW）
    trend: 滚动收益和（对数收益 sum，REGIME_WINDOW，>0 视为上行）
    返回长度 T 的 int8 数组；metric 未就绪的前段填 -1（不参与选块）。
    """
    w = REGIME_WINDOW
    n = closes.shape[0]
    logret = np.zeros(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret[1:] = np.log(closes[1:] / closes[:-1])
    metric = np.full(n, np.nan, dtype=np.float64)
    if n > w:
        from numpy.lib.stride_tricks import sliding_window_view

        sw = sliding_window_view(logret, w)  # sw[i] 覆盖 logret[i..i+w-1]
        if regime == REGIME_VOL:
            s = sw.std(axis=1)
            metric[w:] = s[1:]  # metric[t] = std(logret[t-w+1..t])（跳过含 arr[0]=0 的窗口）
        else:  # trend：滚动收益和
            s = sw.sum(axis=1)
            metric[w:] = s[1:]
    valid = metric[w:]
    q1, q2 = np.quantile(valid, [1 / 3, 2 / 3])
    cls = np.full(n, -1, dtype=np.int8)
    m = metric[w:]
    cls[w:] = np.where(m <= q1, 0, np.where(m <= q2, 1, 2))
    return cls


def _plan_spread_equal(old_avail: int, n_chunks: int, core_total: int,
                       lookback: int = LOOKBACK, core_min: int = CORE_MIN) -> list[tuple[int, int]]:
    """等分年代取块（旧行为）。返回 [(lead_start, core_end)]。"""
    for c in range(n_chunks, 0, -1):
        if c * (core_min + lookback) > old_avail:
            continue
        base, rem = divmod(core_total, c)
        chunks: list[tuple[int, int]] = []
        ok = True
        for k in range(c):
            core_len = base + (1 if k < rem else 0)
            epoch_end = round(old_avail * (k + 1) / c)
            core_start = epoch_end - core_len
            if core_start < 0:
                ok = False
                break
            if chunks and core_start - chunks[-1][1] < lookback:
                ok = False
                break
            lead = max(0, core_start - lookback)
            chunks.append((lead, epoch_end))
        if ok:
            return chunks
    return []


def _plan_spread_regime(old_avail: int, n_chunks: int, core_total: int,
                        classes: np.ndarray,
                        lookback: int = LOOKBACK) -> list[tuple[int, int]] | None:
    """按 regime 分位选块：块锚点尽量落在目标桶（0/1/2 轮转），保证覆盖率。

    用等分方案作**可行性骨架**：每块的 core_end 上限 = 等分方案的年代末（等分
    已保证几何可行、年代铺开），但允许 core_start 向前挪，在
    [上一块 core_end + lookback, 年代末 - core_len] 窗口内挑「离目标桶最近的
    最新 bar」。目标桶按 0/1/2 轮转，找不着 exact 桶时自动取最近桶（d 降级），
    因此只要对应桶在该年代存在就能覆盖；不存在也不至于整单失败。

    失败（等分骨架本身放不下）返回 None，调用方降级 tail。
    """
    scaffold = _plan_spread_equal(old_avail, n_chunks, core_total, lookback=lookback)
    if not scaffold:
        return None
    base, rem = divmod(core_total, n_chunks)
    target = 0
    lo = 0  # 上一块 core_end（下一块 lead 起点 ≥ lo）
    chunks: list[tuple[int, int]] = []
    for k in range(n_chunks):
        core_len = base + (1 if k < rem else 0)
        _lead, era_end = scaffold[k]            # 等分年代末 = 本块 core_end 上限
        s_lo = lo + lookback                    # lead 不越上一块 core
        s_hi = era_end - core_len               # 不越过本年代末（向后块让位）
        if s_hi < s_lo:                          # 骨架保证不会发生，防御一下
            return None
        best_s, best_dist = None, 10**9
        for s in range(s_hi, s_lo - 1, -1):
            c = int(classes[s]) if s < classes.shape[0] else -1
            d = abs(c - target) if c >= 0 else 10**9
            if d < best_dist:
                best_s, best_dist = s, d
                if d == 0:
                    break
        if best_s is None:
            return None
        chunks.append((best_s - lookback, best_s + core_len))
        lo = best_s + core_len
        target = (target + 1) % 3
    return chunks


def _spread_slice(df: pd.DataFrame, n_bars: int, n_chunks: int | None,
                  regime: str) -> tuple[pd.DataFrame, int, dict[str, Any]]:
    """全历史分块抽取：C 个连续块 + 结尾近期连续段。返回 (子集, 实际块数, meta)。"""
    total = len(df)
    recent = min(max(RECENT_MIN, n_bars // 4), n_bars, total)
    recent = max(0, recent)
    budget = n_bars - recent
    meta: dict[str, Any] = {"regime": regime}
    if budget <= 0 or total <= recent + LOOKBACK * 2:
        return _tail_slice(df, n_bars), 1, meta
    old_avail = total - recent - LOOKBACK
    if old_avail < CORE_MIN:
        return _tail_slice(df, n_bars), 1, meta

    n_chunks = max(1, min(int(n_chunks or 0) or max(2, round(budget / 8000)), MAX_CHUNKS))

    classes: np.ndarray | None = None
    if regime != REGIME_EQUAL:
        try:
            classes = _regime_classes(df["close"].to_numpy(dtype=np.float64), regime)
            classes = classes[:old_avail]
        except Exception:  # noqa: BLE001 regime 计算失败降级等分
            classes = None

    if classes is not None:
        plan = _plan_spread_regime(old_avail, n_chunks, budget, classes)
        if plan is None:
            plan = _plan_spread_equal(old_avail, n_chunks, budget)
        # 覆盖率统计（core 起始 ≈ lead + LOOKBACK 处的分位类；首块贴 0 时可能高估 1 根）
        cov = [0, 0, 0]
        for lead, _end in plan:
            s_est = lead + LOOKBACK
            c = int(classes[s_est]) if 0 <= s_est < len(classes) else -1
            if 0 <= c <= 2:
                cov[c] += 1
        meta["regime_coverage"] = {"low": cov[0], "mid": cov[1], "high": cov[2]}
        nsrc = [0, 0, 0]
        for c in classes:
            if 0 <= int(c) <= 2:
                nsrc[int(c)] += 1
        tot_src = max(1, sum(nsrc))
        meta["source_dist"] = {k: round(v / tot_src, 4) for k, v in
                               zip(("low", "mid", "high"), nsrc)}
    else:
        plan = _plan_spread_equal(old_avail, n_chunks, budget)
    if not plan:
        return _tail_slice(df, n_bars), 1, meta

    parts = [df.iloc[lead:end].reset_index(drop=True) for lead, end in plan]
    parts.append(df.iloc[old_avail:].reset_index(drop=True))  # LOOKBACK 前导 + recent
    out = pd.concat(parts, ignore_index=True)
    if "time" in out.columns:
        out = out[~out["time"].duplicated(keep="first")].sort_values("time")
    meta["n_chunks_used"] = len(plan) + 1
    return out.reset_index(drop=True), len(plan) + 1, meta


def _out_path(slices_root: Path, source: Path, mode: str, n_bars: int,
              n_chunks: int | None, regime: str | None) -> Path:
    parts = [mode, str(n_bars)]
    if mode == MODE_SPREAD:
        if regime and regime != REGIME_EQUAL:
            parts.append(regime)
        if n_chunks:
            parts.append(f"c{n_chunks}")
    return slices_root / "_".join(parts) / source.name


def read_train_range(subset_path: str | Path) -> dict[str, Any] | None:
    """读取子集旁 train_range.json（溯源）。不存在/损坏返回 None。"""
    p = Path(subset_path)
    side = p.parent / SIDECAR_NAME
    if not side.exists():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def inferred_train_range(data_file: str | Path, data_source: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """旧策略文件无 train_range 时，从 data_source/data_file 推断可展示的溯源。

    旧版训练（加 train_range 之前）不写 sidecar；这类文件参与溯源展示时用
    本函数现算：full 模式 + data_source 里的 bars/start/end/文件路径。
    推断结果带 ``inferred": True 标记，与真实 sidecar 区分。非 parquet/
    文件缺失时返回 None。
    """
    p = Path(str(data_file or ""))
    if not p.exists() or p.suffix.lower() != ".parquet":
        return None
    meta: dict[str, Any] = {
        "mode": "full",
        "subset": False,
        "data_file": str(p.resolve()),
        "source_file": str(p.resolve()),
        "regime": None,
        "n_bars_requested": None,
        "n_chunks_used": None,
        "inferred": True,
    }
    ds = data_source if isinstance(data_source, dict) else {}
    try:
        from data_pipeline.parquet_manager import inspect_parquet_file

        info = inspect_parquet_file(str(p))
        meta["source_bars"] = info.get("bars")
        meta["subset_bars"] = info.get("bars")
        meta["symbol"] = info.get("symbol") or ds.get("symbol")
        meta["timeframe"] = info.get("timeframe") or ds.get("timeframe")
        meta["start"] = str(info.get("start_date") or ds.get("start") or "")
        meta["end"] = str(info.get("end_date") or ds.get("end") or "")
    except Exception:  # noqa: BLE001 推断失败给最小字段
        meta["symbol"] = ds.get("symbol")
        meta["timeframe"] = ds.get("timeframe")
        meta["start"] = str(ds.get("start") or "")
        meta["end"] = str(ds.get("end") or "")
        meta["source_bars"] = ds.get("bars")
        meta["subset_bars"] = ds.get("bars")
    return meta


def prepare_training_subset(
    data_file: str | Path,
    mode: str = MODE_FULL,
    n_bars: int | None = None,
    n_chunks: int | None = None,
    regime: str | None = DEFAULT_REGIME,
    *,
    root_dir: str | Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """按 mode 生成训练子集，返回结果 dict（含最终 data_file 与溯源元信息）。

    mode:
      full    — 原样使用，不生成新文件。
      tail    — 最近 n_bars 根（n_bars 缺省=全部 → 等效 full）。
      spread  — 全历史分块共 n_bars 根；可选 n_chunks 控制年代块数，
                regime ∈ {vol(默认)/trend/equal} 控制分块方式。

    生成子集时会在其目录写 train_range.json sidecar（溯源用）。
    确定性：相同参数总是产出字节一致的文件（新指纹 → 独立 holdout 单次消费）。
    root_dir 仅供测试注入临时目录；默认写 PROJECT/data/slices/。
    """
    src = Path(data_file).expanduser()
    if not src.is_absolute():
        src = (PROJECT_ROOT / src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"数据文件不存在: {src}")

    base_root = Path(root_dir).expanduser() if root_dir else PROJECT_ROOT
    slices_root = base_root / "data" / "slices"

    mode = (mode or MODE_FULL).strip().lower()
    if mode not in (MODE_FULL, MODE_TAIL, MODE_SPREAD):
        raise ValueError(f"未知训练数据模式: {mode}（可选 full/tail/spread）")
    regime = (regime or DEFAULT_REGIME).strip().lower()
    if regime not in REGIMES:
        raise ValueError(f"未知分块方式: {regime}（可选 {'/'.join(REGIMES)}）")

    df = _read_sorted(src)
    total = len(df)
    meta: dict[str, Any] = {"n_chunks_used": None, "regime": regime}

    def _finalize(out_path: Path, subset: bool, subset_bars: int,
                  df_out: pd.DataFrame | None) -> dict[str, Any]:
        """写 sidecar + 元信息并返回结果 dict。"""
        try:
            from data_pipeline.parquet_manager import inspect_parquet_file

            info = inspect_parquet_file(out_path)
            meta["symbol"] = info.get("symbol")
            meta["timeframe"] = info.get("timeframe")
            meta["start"] = str(info.get("start_date") or "")
            meta["end"] = str(info.get("end_date") or "")
        except Exception:  # noqa: BLE001 元信息失败不影响子集可用
            pass
        res = SubsetResult(
            mode=mode,
            data_file=str(out_path.resolve()),
            subset=subset,
            source_file=str(src),
            source_bars=total,
            subset_bars=subset_bars,
            n_bars_requested=n_bars,
            n_chunks=n_chunks,
            meta=meta,
        ).to_dict()
        if subset:
            # 溯源 sidecar（full/降级不写，避免覆盖既有子集同目录）
            try:
                side = out_path.parent / SIDECAR_NAME
                side.parent.mkdir(parents=True, exist_ok=True)
                side.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            except OSError:
                pass
        return res

    if mode == MODE_FULL:
        return _finalize(src, False, total, None)
    if total < CORE_MIN:
        meta["note"] = f"源数据仅 {total} 根，不足以抽样，使用全部"
        return _finalize(src, False, total, None)

    if mode == MODE_TAIL:
        n = min(max(1, int(n_bars or total)), total)
        sub = _tail_slice(df, n)
        meta["n_chunks_used"] = 1
        if n >= total:
            meta["note"] = "请求根数 ≥ 全量，等效使用全部"
            return _finalize(src, False, total, None)
        out_path = _out_path(slices_root, src, MODE_TAIL, n, None, None)
        if overwrite or not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sub.to_parquet(out_path, index=False)
        return _finalize(out_path, True, len(sub), sub)

    # spread
    n = min(max(1, int(n_bars or total)), total)
    sub, used, spread_meta = _spread_slice(df, n, n_chunks, regime)
    meta.update(spread_meta)
    if len(sub) >= total:
        meta["note"] = "请求根数 ≥ 全量，等效使用全部"
        return _finalize(src, False, total, None)
    out_path = _out_path(slices_root, src, MODE_SPREAD, n,
                         n_chunks if n_chunks else None, regime)
    if overwrite or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(out_path, index=False)
    return _finalize(out_path, True, len(sub), sub)


def preview_training_subset(
    data_file: str | Path,
    mode: str = MODE_FULL,
    n_bars: int | None = None,
    n_chunks: int | None = None,
    regime: str | None = DEFAULT_REGIME,
) -> dict[str, Any]:
    """训练子集在整段历史上的取样区间预览（不写文件，供 UI 可视化）。

    返回 {mode, regime, source_file, source_bars, total_bars: 实际取样根数,
    blocks: [{kind: warmup|core|recent, start, end, regime_class}],
    regime_coverage, n_chunks_used}。kind：core=年代取样核心 / warmup=核心前的
    同年代前置行情（保证滚动特征有效）/ recent=末尾近期连续段（验证/holdout 干净区）。
    regime_class ∈ 0/1/2（low/mid/high），非 regime 段为 None。坐标均为源数据行号。
    """
    src = Path(data_file).expanduser()
    if not src.is_absolute():
        src = (PROJECT_ROOT / src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"数据文件不存在: {src}")
    mode = (mode or MODE_FULL).strip().lower()
    if mode not in (MODE_FULL, MODE_TAIL, MODE_SPREAD):
        raise ValueError(f"未知训练数据模式: {mode}（可选 full/tail/spread）")
    regime = (regime or DEFAULT_REGIME).strip().lower()
    if regime not in REGIMES:
        raise ValueError(f"未知分块方式: {regime}（可选 {'/'.join(REGIMES)}）")

    df = _read_sorted(src)
    total = len(df)
    out: dict[str, Any] = {
        "mode": mode, "regime": regime,
        "source_file": str(src.resolve()), "source_bars": total,
    }
    if mode == MODE_FULL or total < CORE_MIN:
        out["total_bars"] = total
        out["n_chunks_used"] = 1
        out["blocks"] = [{"kind": "core", "start": 0, "end": total, "regime_class": None}]
        out["regime_coverage"] = {}
        return out

    if mode == MODE_TAIL:
        n = min(max(1, int(n_bars or total)), total)
        out["total_bars"] = n
        out["n_chunks_used"] = 1
        out["blocks"] = [{"kind": "core", "start": total - n, "end": total,
                          "regime_class": None}]
        out["regime_coverage"] = {}
        return out

    # spread：与 _spread_slice 同一套几何，只出不写文件
    n = min(max(1, int(n_bars or total)), total)
    recent = min(max(RECENT_MIN, n // 4), n, total)
    recent = max(0, recent)
    budget = n - recent
    if budget <= 0 or total <= recent + LOOKBACK * 2:
        out["mode"] = MODE_TAIL  # 几何不足 → 实际退化为尾部切片（如实告知）
        out["total_bars"] = n
        out["n_chunks_used"] = 1
        out["blocks"] = [{"kind": "core", "start": total - n, "end": total,
                          "regime_class": None}]
        out["regime_coverage"] = {}
        return out
    old_avail = total - recent - LOOKBACK
    if old_avail < CORE_MIN:
        out["mode"] = MODE_TAIL
        out["total_bars"] = n
        out["n_chunks_used"] = 1
        out["blocks"] = [{"kind": "core", "start": total - n, "end": total,
                          "regime_class": None}]
        out["regime_coverage"] = {}
        return out

    used_chunks = max(1, min(int(n_chunks or 0) or max(2, round(budget / 8000)), MAX_CHUNKS))
    classes: np.ndarray | None = None
    if regime != REGIME_EQUAL:
        try:
            classes = _regime_classes(df["close"].to_numpy(dtype=np.float64), regime)[:old_avail]
        except Exception:  # noqa: BLE001 降级等分
            classes = None
    if classes is not None:
        plan = _plan_spread_regime(old_avail, used_chunks, budget, classes)
        if plan is None:
            plan = _plan_spread_equal(old_avail, used_chunks, budget)
    else:
        plan = _plan_spread_equal(old_avail, used_chunks, budget)
    if not plan:
        out["mode"] = MODE_TAIL
        out["total_bars"] = n
        out["n_chunks_used"] = 1
        out["blocks"] = [{"kind": "core", "start": total - n, "end": total,
                          "regime_class": None}]
        out["regime_coverage"] = {}
        return out

    cov = [0, 0, 0]
    blocks: list[dict[str, Any]] = []
    for lead, core_end in plan:
        core_start = lead + LOOKBACK
        c = int(classes[core_start]) if classes is not None and 0 <= core_start < len(classes) else None
        if c is not None and 0 <= c <= 2:
            cov[c] += 1
        if lead < core_start:
            blocks.append({"kind": "warmup", "start": lead, "end": core_start,
                           "regime_class": c})
        blocks.append({"kind": "core", "start": core_start, "end": core_end,
                       "regime_class": c})
    # 末尾近期段（含前置行情）：[old_avail, old_avail+LOOKBACK) 为 warmup
    blocks.append({"kind": "warmup", "start": old_avail, "end": old_avail + LOOKBACK,
                   "regime_class": None})
    blocks.append({"kind": "recent", "start": old_avail + LOOKBACK, "end": total,
                   "regime_class": None})
    # 实际子集根数 = 请求数 + 各块前置 warm-up（与 _spread_slice 生成的文件行数一致）
    out["total_bars"] = sum(max(0, b["end"] - b["start"]) for b in blocks)
    out["n_bars_requested"] = n
    out["n_chunks_used"] = len(plan) + 1
    out["regime_coverage"] = {"low": cov[0], "mid": cov[1], "high": cov[2]}
    out["blocks"] = blocks
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python data_pipeline/train_sampler.py <full|tail|spread> <parquet> [n_bars] [n_chunks] [regime]")
        sys.exit(1)
    _mode, _file = sys.argv[1], sys.argv[2]
    _n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
    _c = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else None
    _r = sys.argv[5] if len(sys.argv) > 5 else None
    print(json.dumps(prepare_training_subset(_file, mode=_mode, n_bars=_n,
                                             n_chunks=_c, regime=_r),
                     ensure_ascii=False, indent=2))
