"""web/combo_sweep_curves.py — 三轴网格任意一格 (组合 × 上限% × 无信号阈值) 的资金曲线。

基线切片（第一档上限 × 第一档阈值）读 hold_matrix_curves_latest.npz 侧车 —— 三轴
回测跑完自动落盘，与旧矩阵视图完全同构；非基线切片用同一离散撮合引擎
（web.paper_replay.run_replay，与 scripts/combo_sweep.py 同窗口/成本/因子，
确定性复现网格那一行）对单格现场重放，再走与侧车相同的曲线装配逻辑
（web.hold_matrix_curves.build_curve_response），两条路径返回同构 JSON。

窗口数据按 (strategy_file, data_file, window_bars, window_mode) 缓存 ——
首次点击要算特征（约 1-3 秒），之后同一切片内的点击只做 ~50ms 重放。
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRID_FILE = ROOT / "results" / "combo_sweep_latest.json"


def _read_grid() -> dict | None:
    """读最近一次三轴回测结果（缺文件/损坏 → None）。"""
    try:
        data = json.loads(GRID_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _abs(p: str) -> str:
    pp = Path(p)
    return str(pp if pp.is_absolute() else ROOT / pp)


@functools.lru_cache(maxsize=4)
def _load_window(strategy_file: str, data_file: str, window_bars: int | None,
                 window_mode: str) -> dict:
    """加载与三轴回测完全同口径的因子/行情窗口（按数据窗口缓存）。"""
    from scripts.hold_matrix import load_factor

    load_window = window_bars if window_mode == "tail" else None  # spread 子集即窗口
    return load_factor(strategy_file, data_file, load_window)


def slice_curve(combo: str, cap_pct: float | None = None,
                threshold: float | None = None) -> dict | None:
    """三轴网格中任意一格的资金曲线 + 滚动夏普（与 hold-matrix/curve 同构）。

    返回 None = 无三轴网格 / 未指定切片 / 命中基线切片 —— 调用方应回退到
    hold_matrix_curves_latest.npz 侧车路径（旧行为不变）；否则返回响应 dict。
    """
    from web.hold_matrix_curves import build_curve_response
    from web.hold_policy import combo_id

    cid = combo_id(combo)
    grid = _read_grid()
    if grid is None:
        return None
    if cap_pct is None or threshold is None:
        return None  # 未指定切片 → 基线 → 侧车
    cap, thr = float(cap_pct), float(threshold)
    base = grid.get("baseline_slice") or {}
    if abs(cap - float(base.get("cap_pct") or 0.0)) < 1e-9 and \
            abs(thr - float(base.get("threshold") or 0.0)) < 1e-9:
        return None  # 基线切片 → 侧车（与旧矩阵视图完全一致）
    # 校验该格在三轴网格的档位内；不在 → 明确报错而不是静默算别的切片
    caps = [float(c) for c in (grid.get("caps") or [])]
    thrs = [float(t) for t in (grid.get("thresholds") or [])]
    if not any(abs(cap - c) < 1e-9 for c in caps) or \
            not any(abs(thr - t) < 1e-9 for t in thrs):
        return {"available": False, "combo": cid,
                "error": f"该切片（上限 {cap:g}% × t={thr:g}）不在最近一次三轴回测的档位内 — 请按所需档位重跑三轴"}

    strategy_file = _abs(grid.get("strategy_file") or "")
    data_file = _abs(grid.get("data_file") or "")
    if not strategy_file or not data_file or not Path(data_file).exists():
        return {"available": False, "combo": cid, "error": "三轴回测的数据文件缺失，无法重放该切片曲线"}

    # ── 单格重放（与 scripts/combo_sweep.py 完全同参数） ──
    from model_core.backtest import estimate_periods_per_year
    from web.paper_replay import run_replay

    d = _load_window(strategy_file, data_file, grid.get("window_bars"),
                     grid.get("window_mode") or "tail")
    times = d["time"]
    ppy = estimate_periods_per_year(times) if times is not None else 105195.0
    rep = run_replay(
        factor=d["factor"], open_p=d["open"], high_p=d["high"],
        low_p=d["low"], close_p=d["close"],
        commission_pct=float(grid.get("commission_pct") or 0.02),
        slippage_pct=float(grid.get("slippage_pct") or 0.01),
        policy_id=cid, max_position_pct=cap, threshold=thr,
        track_dd=True,  # DD 熔断区间标注需要
        periods_per_year=float(ppy),
    )
    ann: dict[str, Any] = {
        "trades": [{
            "eb": max(0, int(t["bar"]) - int(t.get("hold_bars") or 0)),
            "xb": int(t["bar"]),
            "side": t.get("side"),
            "pnl": round(float(t.get("pnl") or 0.0), 8),
            "reason": t.get("label") or "",
        } for t in rep["trades"]],
        "dd_events": rep.get("dd_events") or [],
    }
    meta = {
        "generated_at": grid.get("generated_at"),
        "window_bars": grid.get("window_bars"),
        "window_start": grid.get("window_start"),
        "window_mode": grid.get("window_mode"),
        "window_blocks": grid.get("window_blocks"),
        "bars": grid.get("bars"),
        "ppy": float(ppy),
        "data_file": data_file,
        "cap_pct": cap,
        "threshold": thr,
    }
    return build_curve_response(cid, rep["equity"], rep["pnl"], meta, ann)