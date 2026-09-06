"""web/hold_matrix_curves.py — 全组合矩阵「单组合资金曲线/滚动夏普」侧车数据。

scripts/hold_matrix.py 每次跑 N×N 时，把每个组合的 equity/pnl 曲线额外落一份
npz + meta json（results/hold_matrix_curves_latest.*），主结果 JSON 保持轻量；
本模块提供读取/滚动夏普/降采样工具，供 /api/backtest/hold-matrix/curve 使用。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CURVES_NPZ = RESULTS_DIR / "hold_matrix_curves_latest.npz"
CURVES_META = RESULTS_DIR / "hold_matrix_curves_latest.json"


def rolling_sharpe(pnl, window: int, ppy: float) -> np.ndarray:
    """按 bar 窗口滚动的（年化）夏普，尾部不足窗口处为 NaN。

    sharpe_t = mean(pnl[t-w+1..t]) / std(...) * sqrt(ppy)；std≈0（空仓/平台）→ NaN。
    """
    p = np.asarray(pnl, dtype=float)
    n = p.size
    out = np.full(n, np.nan)
    w = max(2, int(window))
    if n < w + 1:
        return out
    # 滑窗两遍方差（np.var 先减均值再平方），常数平台（空仓 pnl≡0）精确得
    # var=0 → NaN；避免 E[x²]−E[x]² 的浮点对消造出虚假巨值夏普。
    from numpy.lib.stride_tricks import sliding_window_view

    seg = sliding_window_view(p, w)          # (n-w+1, w)
    mean = seg.mean(axis=1)
    std = seg.std(axis=1)
    safe = std > 1e-12
    vals = np.full(mean.shape, np.nan)
    vals[safe] = mean[safe] / std[safe] * math.sqrt(max(1.0, float(ppy)))
    out[w - 1:] = vals
    return out


def downsample_indices(n: int, max_pts: int = 900) -> list[int]:
    """与 downsample 完全同几何的采样索引（供分桶/标注对齐用）。"""
    n = int(n)
    if n <= max_pts or max_pts <= 1:
        return list(range(n))
    step = math.ceil((n - 1) / (max_pts - 1))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def dd_intervals_from_events(events: list[dict]) -> list[dict]:
    """DD 熔断事件（bar/state 跃迁）→ 熔断区间列表 [{start, end, state, dd_pct}]。

    state != 0 视为“熔断中”（1=熔断 / 2=深档熔断）；连续非零状态合并为一段；
    收复事件（state→0）结束区间；end=None = 一直熔断到窗口末尾。
    """
    if not events:
        return []
    intervals: list[dict] = []
    cur: dict | None = None
    for ev in sorted(events, key=lambda e: int(e.get("bar") or 0)):
        st = int(ev.get("state") or 0)
        bar = int(ev.get("bar") or 0)
        dd = ev.get("dd_pct")
        if st != 0:
            if cur is None:
                cur = {"start": bar, "state": st, "dd_pct": dd}
            else:
                cur["state"] = max(int(cur["state"] or 0), st)
                # dd_pct 为负百分比：越深（更负）越严重
                if dd is not None and (cur.get("dd_pct") is None or dd < cur["dd_pct"]):
                    cur["dd_pct"] = dd
        elif cur is not None:
            cur["end"] = bar - 1
            intervals.append(cur)
            cur = None
    if cur is not None:
        cur["end"] = None
        intervals.append(cur)
    return intervals


def pnl_quintiles(pnl, idx: list[int] | None = None) -> dict:
    """逐 bar 收益按分布五等分：每采样点分桶(0..4) + 分桶统计 + 最优/最差连续段。

    bucket 按 pnl 的 [20,40,60,80] 分位切（含大量 0 时低桶多为空仓/小赚，
    高桶才是真赢家）；best/worst 段用 Kadane 找累计收益最大/最小连续 bar 区间。
    """
    p = np.asarray(pnl, dtype=float)
    n = p.size
    if n == 0:
        return {"buckets": [], "stats": [], "best_segment": None, "worst_segment": None}
    qs = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
    bucket = np.searchsorted(qs, p, side="right").astype(int)  # 0..4
    idx = idx if idx is not None else list(range(n))
    sampled = [int(bucket[i]) for i in idx]
    stats = []
    total = float(p.sum())
    for b in range(5):
        m = bucket == b
        sm = float(p[m].sum())
        stats.append({
            "bucket": b,
            "n": int(m.sum()),
            "sum": round(sm, 8),
            "share": round(100.0 * sm / total, 2) if abs(total) > 1e-12 else None,
        })

    def _segment(sign: int) -> dict | None:
        best_sum, cur_sum, s0, best = 0.0, 0.0, 0, None
        for i in range(n):
            v = p[i] * sign
            if cur_sum + v > v:  # 延续
                cur_sum += v
            else:  # 重新起段
                cur_sum, s0 = v, i
            if cur_sum > best_sum:
                best_sum, best = cur_sum, (s0, i)
        if best is None:
            return None
        return {"start": int(best[0]), "end": int(best[1]),
                "cum": round(best_sum * sign, 8), "bars": int(best[1] - best[0] + 1)}

    return {
        "buckets": sampled,
        "stats": stats,
        "best_segment": _segment(1),
        "worst_segment": _segment(-1),
    }


def downsample(arr, max_pts: int = 900):
    """等步长降采样到 ≤ max_pts（保持端点）。"""
    a = np.asarray(arr)
    n = a.size
    if n <= max_pts or max_pts <= 1:
        return a
    step = math.ceil((n - 1) / (max_pts - 1))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return a[idx]


def trim_leading_flat(equity) -> int:
    """去掉开头完全不动（eq==eq[0]，warm-up/空仓等待）的段，返回起点索引。"""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0
    start = 0
    while start < eq.size - 1 and eq[start] == eq[0]:
        start += 1
    # 保留到首个变动点的前一格（避免从 0 收益突兀起笔）
    return max(0, start - 1)


def save_curves(curves: dict[str, np.ndarray], meta: dict[str, Any],
                npz_path: Path = CURVES_NPZ, meta_path: Path = CURVES_META,
                annotations: dict[str, dict] | None = None) -> None:
    """npz 每个组合存 {combo}_equity / {combo}_pnl 两把键；
    annotations: {combo: {trades, dd_events}} 并入 meta json（供买卖点/DD 区间标注）。"""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for pid, arrs in curves.items():
        eq, pn = arrs
        arrays[f"{pid}_equity"] = np.asarray(eq, dtype=float)
        arrays[f"{pid}_pnl"] = np.asarray(pn, dtype=float)
    np.savez(npz_path, **arrays)
    if annotations:
        meta = dict(meta)
        meta["annotations"] = annotations
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_annotations(combo: str) -> dict:
    """读单组合的 交易标注/DD 事件（meta json 内嵌；缺则空）。"""
    try:
        meta = json.loads(CURVES_META.read_text(encoding="utf-8"))
        ann = meta.get("annotations") or {}
        return ann.get(combo) or {}
    except (OSError, ValueError):
        return {}


def load_curve(combo: str):
    """读单组合曲线 → (equity, pnl, meta)；缺文件/键返回 None。"""
    if not CURVES_NPZ.exists() or not CURVES_META.exists():
        return None
    try:
        data = np.load(CURVES_NPZ, allow_pickle=False)
        meta = json.loads(CURVES_META.read_text(encoding="utf-8"))
        eq = data[f"{combo}_equity"]
        pn = data[f"{combo}_pnl"]
    except (KeyError, OSError, ValueError):
        return None
    return np.asarray(eq, dtype=float), np.asarray(pn, dtype=float), meta


def build_curve_response(cid: str, eq, pn, meta: dict, ann: dict) -> dict[str, Any]:
    """把 (equity, pnl, meta, 交易/DD 标注) 装配成曲线接口响应。

    基线切片的侧车数据与三轴网格非基线切片的按需重放共用同一装配逻辑，
    保证两条路径返回同构 JSON（labels/equity/rolling_sharpe/trades/
    dd_intervals/quintiles/segment_links）。
    """
    eq = np.asarray(eq, dtype=float)
    pn = np.asarray(pn, dtype=float)
    start = trim_leading_flat(eq)
    eq_t = eq[start:]
    pn_t = pn[start:]
    # 滚动夏普：窗口取 min(240, 总根数//40)，至少 30 根；空仓平段产出 NaN
    w = max(30, min(240, max(1, eq_t.size // 40)))
    w = min(w, max(2, eq_t.size // 2))
    roll = rolling_sharpe(pn_t, window=w, ppy=float(meta.get("ppy") or 105195.0))
    step_begin = int(meta.get("window_start") or 0) + start
    labels = [step_begin + int(i) for i in range(eq_t.size)]
    idx = downsample_indices(eq_t.size)
    eq_d = eq_t[idx]
    roll_d = roll[idx]
    lab_d = [float(labels[int(i)]) for i in idx]
    eq_list = [float(v) if v == v else None for v in eq_d.tolist()]
    roll_list = [None if (v != v) else float(v) for v in roll_d.tolist()]
    nav = eq_list[-1] if eq_list else 1.0

    # ── 买卖点标注 + DD 熔断区间 + 收益五等分热力（读侧车 annotations） ──
    w_start = int(meta.get("window_start") or 0)
    trades = []
    for tr in ann.get("trades") or []:
        eb = int(tr.get("eb") or 0) - start
        xb = int(tr.get("xb") or 0) - start
        if xb < 0:
            continue
        trades.append({
            "eb": max(0, eb),       # 相对曲线数组的入场 bar
            "xb": max(0, xb),       # 相对曲线数组的出场 bar
            "abs_eb": w_start + max(0, eb) + start,
            "abs_xb": w_start + max(0, xb) + start,
            "side": tr.get("side"),
            "pnl": tr.get("pnl"),
            "reason": tr.get("reason") or "",
        })
    # DD 事件 bar 同样是窗口内相对索引 → 换算成绝对 bar（与 labels 同基准）
    dd_intervals = []
    for iv in dd_intervals_from_events(ann.get("dd_events") or []):
        s = max(0, int(iv.get("start") or 0) - start)
        e = iv.get("end")
        dd_intervals.append({
            "start_abs": w_start + s + start,
            # end 与 start 同口径：事件 bar 是窗口相对索引，先裁掉开头平段再对齐绝对 bar
            "end_abs": (w_start + max(0, int(e) - start) + start) if e is not None else None,
            "state": iv.get("state"),
            "dd_pct": iv.get("dd_pct"),
        })
    quint = pnl_quintiles(pn_t, idx=idx)
    # ── 最优/最差段与交易·熔断的关联：段内每笔交易（入场→出场重叠）与 DD 熔断区间 ──
    l0 = w_start + start  # 段 start/end 是相对裁剪后曲线的 bar 索引 → 绝对 bar

    def _seg_links(seg: dict | None) -> dict | None:
        if not seg:
            return None
        s, e = int(seg["start"]), int(seg["end"])
        s_abs, e_abs = l0 + s, l0 + e
        hits = []
        for k, tr in enumerate(trades, 1):
            # 出场 bar 落在段内即算「该段的交易」（盈亏在出场 bar 兑现）
            if s_abs <= int(tr["abs_xb"]) <= e_abs:
                hits.append({
                    "k": k,
                    "side": tr.get("side"),
                    "reason": tr.get("reason") or "",
                    "pnl": tr.get("pnl"),
                    "xb_abs": int(tr["abs_xb"]),
                })
        dd_hits = []
        for j, iv in enumerate(dd_intervals, 1):
            iv_s = int(iv["start_abs"])
            iv_e = int(iv["end_abs"]) if iv.get("end_abs") is not None else e_abs
            if iv_s <= e_abs and iv_e >= s_abs:  # 区间相交
                dd_hits.append({
                    "j": j,
                    "state": iv.get("state"),
                    "dd_pct": iv.get("dd_pct"),
                    "start_abs": iv_s,
                    "end_abs": iv.get("end_abs"),
                })
        worst = None
        if hits:
            worst = min(hits, key=lambda h: float(h.get("pnl") or 0.0))
        brief = ""
        if hits:
            top = sorted(hits, key=lambda h: float(h.get("pnl") or 0.0))[:2]
            brief = " ".join(f"#{h['k']} {h.get('reason') or '?'} {100 * float(h.get('pnl') or 0):+.1f}%" for h in top)
        elif dd_hits:
            d0 = dd_hits[0]
            brief = f"第 {d0['j']} 次熔断（{'深档' if d0.get('state') == 2 else '熔断'} {d0.get('dd_pct') or 0:.1f}%）期间空仓"
        return {
            "start_abs": s_abs,
            "end_abs": e_abs,
            "n_trades": len(hits),
            "trades": hits[:6],
            "worst_trade": worst,
            "dd_intervals": dd_hits[:4],
            "brief": brief,
        }

    seg_links = {"best": _seg_links(quint.get("best_segment")),
                 "worst": _seg_links(quint.get("worst_segment"))}
    return {
        "available": True,
        "combo": cid,
        "labels": lab_d,
        "equity": eq_list,
        "rolling_sharpe": roll_list,
        "rolling_window": w,
        "bars": eq_t.size,
        "total_return": float(nav - 1.0),
        "trades": trades,
        "dd_intervals": dd_intervals,
        "quintiles": quint,
        "segment_links": seg_links,
        "meta": {"generated_at": meta.get("generated_at"), "window_bars": meta.get("window_bars"),
                  "window_start": meta.get("window_start"), "bars_total": meta.get("bars")},
    }
