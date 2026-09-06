"""样本外溯源感知：把训练真实 holdout 边界读出来，给回测/矩阵的窗口打“真伪 OOS”标签。

背景：系统曾在矩阵/回测文案里把“取数据文件尾部 N 根”一律写成“训练未见的新数据”。
实际上训练（含 walk-forward 选优）只预留了子集末尾 HOLDOUT_BARS 根当 holdout，
其余部分模型在训练时全见过——所以窗口可能“部分样本内”，标签是假的。

本模块提供：
- strategy_provenance(strategy_file)：读取/推断训练溯源（train_range / data_source /
  data_file），用与训练完全相同的 _holdout_bars_for 规则重算 holdout 根数，得到
  训练文件的 holdout 起始 ts 与训练截止 ts（都是绝对时间轴）。
- classify_window(...)：把某次回测窗口（run 文件 bar 区间）按 ts 桶切成
  样本内 / 训练 holdout / 训练后新数据 三段，给出诚实状态与建议窗口。
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────────────────────────────────────────────────────
# 时间序列读取（pyarrow 直读 ts 列 + lru 缓存；避免整表 pandas 开销）
# ────────────────────────────────────────────────────────────────────────
def _ts_column_name(path: Path) -> str:
    import pyarrow.parquet as pq

    schema = pq.read_schema(path, memory_map=True)
    names = schema.names
    for want in ("timestamp", "time", "ts", "datetime"):
        if want in names:
            return want
    return names[0]


@lru_cache(maxsize=8)
def _ts_cached(abs_path: str, mtime_ns: int, size: int) -> np.ndarray:
    import pyarrow.parquet as pq

    path = Path(abs_path)
    col = _ts_column_name(path)
    tab = pq.read_table(path, columns=[col], memory_map=True)
    return tab.column(col).to_numpy().astype(np.float64)


def ts_column(path: str | Path) -> np.ndarray:
    """读 parquet 的 ts 列（epoch 秒）。按 (path, mtime, size) 缓存。"""
    p = Path(path)
    st = p.stat()
    return _ts_cached(str(p.resolve()), st.st_mtime_ns, st.st_size)


def _fmt_ts(ts: float | None) -> str | None:
    if ts is None or ts != ts:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ────────────────────────────────────────────────────────────────────────
# 训练溯源（与 model_core.engine._holdout_bars_for 同规则的本地复制，
# 避免 import model_core.engine 把 torch 拖进纯 web 模块）
# ────────────────────────────────────────────────────────────────────────
def holdout_bars_for(T_full: int) -> int:
    cap = T_full // 8
    if cap < 100:
        cap = max(0, T_full - 300)
    from model_core.config import ModelConfig

    return max(0, min(int(ModelConfig.HOLDOUT_BARS), cap))


def strategy_provenance(strategy_file: str | Path) -> dict[str, Any] | None:
    """读策略文件 → 训练溯源。缺 train_range 时用 data_source/data_file 推断。

    返回：{strategy_file, data_file, n_bars, mode, holdout_bars,
           train_end_ts, holdout_start_ts, train_end_date, holdout_start_date, source}
    数据文件读不到/无法定位 → None（不硬失败，调用方降级为原文案）。
    """
    p = Path(strategy_file)
    if not p.exists():
        return None
    try:
        strat = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(strat, dict):
        return None

    tr = strat.get("train_range") or {}
    ds = strat.get("data_source") or {}
    data_file = str(
        tr.get("data_file") or ds.get("data_file") or strat.get("data_file") or ""
    ).strip()
    df = Path(data_file)
    if not df.is_absolute():
        df = PROJECT_ROOT / df
    if not data_file or not df.exists():
        return None
    try:
        n_bars = int(tr.get("n_bars") or ds.get("bars") or 0)
    except (TypeError, ValueError):
        n_bars = 0
    if n_bars <= 0:
        # 兜底：数文件根数
        try:
            ts = ts_column(df)
            n_bars = int(ts.size)
        except Exception:
            return None
    mode = str(tr.get("mode") or ds.get("mode") or ("tail" if ds.get("bars") else "full"))
    try:
        # 优先用实际训练时写入的预留根数（train_range / 策略顶层 / data_source），
        # 找不到才按默认规则重算——保证 --holdout-bars N 训出的策略溯源正确。
        hb_raw = (tr.get("holdout_bars") or strat.get("holdout_bars")
                  or ds.get("holdout_bars"))
        hb = int(hb_raw) if hb_raw is not None else holdout_bars_for(n_bars)
    except (TypeError, ValueError):
        hb = holdout_bars_for(n_bars)
    try:
        ts = ts_column(df)
    except Exception:
        return None
    end_ts = float(ts[-1])
    holdout_start = max(0, n_bars - hb)
    holdout_start_ts = float(ts[holdout_start]) if holdout_start < ts.size else end_ts
    return {
        "strategy_file": str(p),
        "data_file": str(df),
        "n_bars": n_bars,
        "mode": mode,
        "holdout_bars": hb,
        "train_end_ts": end_ts,
        "holdout_start_ts": holdout_start_ts,
        "train_end_date": _fmt_ts(end_ts),
        "holdout_start_date": _fmt_ts(holdout_start_ts),
        "source": "train_range" if tr.get("n_bars") else "inferred",
    }


# ────────────────────────────────────────────────────────────────────────
# 窗口分类
# ────────────────────────────────────────────────────────────────────────
def classify_window(
    strategy_file: str | Path | None,
    run_file: str | Path | None,
    window_start: int | None,
    window_bars: int | None,
) -> dict[str, Any]:
    """把一次回测窗口（run 文件 bar [window_start, +window_bars)）按 ts 分桶。

    返回 dict（不可用时 status='unavailable'，调用方应回退旧文案）：
      status: oos-holdout | oos-new | in-sample | partial(样本内+holdout 混合) |
              mixed(含训练后新数据) | unavailable
      n_in_sample / n_holdout / n_post_train: 窗口内三段的 bar 数
      train: 溯源信息（截断版）
      honest: 诚实窗口建议（run 文件 bar 空间）
    """
    base: dict[str, Any] = {
        "status": "unavailable",
        "n_in_sample": None,
        "n_holdout": None,
        "n_post_train": None,
        "honest": None,
        "train": None,
    }
    if not strategy_file or not run_file or not window_bars or window_bars <= 0:
        return base
    prov = strategy_provenance(strategy_file)
    if prov is None:
        return base
    try:
        run_ts = ts_column(run_file)
    except Exception:
        return base
    w0 = max(0, int(window_start or 0))
    w1 = min(int(run_ts.size), w0 + int(window_bars))
    if w1 <= w0:
        return base
    seg = run_ts[w0:w1]
    ho_lo, ho_hi = float(prov["holdout_start_ts"]), float(prov["train_end_ts"])
    n_in = int((seg < ho_lo).sum())
    n_ho = int(((seg >= ho_lo) & (seg <= ho_hi)).sum())
    n_post = int((seg > ho_hi).sum())

    if n_in == 0 and n_ho == 0 and n_post == 0:
        status = "unavailable"
    elif n_in == 0 and n_ho == 0 and n_post > 0:
        status = "oos-new"           # 纯训练后新数据
    elif n_in == 0 and n_post == 0 and n_ho > 0:
        status = "oos-holdout"       # 纯 holdout 尾部
    elif n_ho == 0 and n_post == 0 and n_in > 0:
        status = "in-sample"         # 全在训练集内
    elif n_post > 0 and (n_in > 0 or n_ho > 0):
        status = "mixed"             # 部分样本内/外 + 训练后新数据
    else:
        status = "partial"           # 样本内 + holdout 混合

    # 诚实窗口建议（run 文件 bar 空间）：
    same_file = str(Path(run_file).resolve()) == prov["data_file"]
    honest: dict[str, Any] | None = None
    if same_file and prov["holdout_bars"] > 0:
        honest = {
            "kind": "holdout-tail",
            "start_bar": max(0, prov["n_bars"] - prov["holdout_bars"]),
            "end_bar": prov["n_bars"] - 1,
            "n_bars": prov["holdout_bars"],
            "note": "训练真 holdout 尾部（选优/训练从未触碰）",
        }
    elif n_post > 0:
        # run 文件里有训练截止之后的新 bar → 建议从首个新 bar 起（含前 800 根 warm-up 提示）
        first_new = w0 + int(np.argmax(seg > ho_hi)) if n_post > 0 else None
        if first_new is not None:
            honest = {
                "kind": "post-train",
                "start_bar": first_new,
                "end_bar": w1 - 1,
                "n_bars": n_post,
                "note": "训练截止之后的新 bar（如需特征 warm-up，窗口需再向前含 ≥800 根）",
            }
    train_short = {k: prov[k] for k in
                   ("data_file", "n_bars", "mode", "holdout_bars",
                    "train_end_date", "holdout_start_date", "source")}
    return {
        "status": status,
        "n_in_sample": int(n_in),
        "n_holdout": int(n_ho),
        "n_post_train": int(n_post),
        "honest": honest,
        "train": train_short,
        "window_start": w0,
        "window_bars": int(window_bars),
    }


def oos_status_label(oos: dict[str, Any]) -> str:
    """分类 dict → 人类可读的简短标签（供 winTxt 用）。"""
    st = oos.get("status")
    n_in = oos.get("n_in_sample") or 0
    n_ho = oos.get("n_holdout") or 0
    n_post = oos.get("n_post_train") or 0
    if st == "oos-holdout":
        return f"样本外 ✓（恰为训练 holdout 尾部 {n_ho} 根，训练从未触碰）"
    if st == "oos-new":
        return f"样本外 ✓（{n_post} 根在训练截止之后，全新数据）"
    if st == "in-sample":
        return f"⚠ 样本内（{n_in} 根全部在训练集内，非样本外）"
    if st == "partial":
        return f"⚠ 部分样本内（{n_in} 根在训练集内，仅尾 {n_ho} 根为真 holdout）"
    if st == "mixed":
        return (f"⚠ 部分样本内 + 新数据（样本内 {n_in} / holdout {n_ho} / "
                f"训练后新数据 {n_post}）")
    return "无法判定样本外状态（缺训练溯源）"
