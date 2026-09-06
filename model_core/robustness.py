"""
model_core/robustness.py -- P3 冠军稳健性复核：成本 / 折叠 / 起点三轴敏感性

用途：`AlphaEngine._finalize_champion` 在【冠军未过 holdout 闸门】时，对 top-K
入围公式自动运行三轴敏感性分析，并把结论（verdict）随拒绝/警告记录写入
`champion_history.json`，供人工复核「这次拒绝是统计必然，还是单窗口坏运气」。

三轴定义：
  - cost  : 在 holdout 窗口按生产口径（tanh 仓位，pnl = pos·ret − |Δpos|·cost）
            扫成本倍数 [0, .5, 1, 2, 4]×base —— 冠军在 2× 成本下是否仍盈利、排名仍第一
  - fold  : 训练区滚动折叠取 n_folds±1 三种布局重算 wf val —— 冠军是否所有布局
            val>0 且多数布局仍排第一（防「换个折叠数冠军就换人」）
  - start : 丢弃训练区开头 5% / 10% / 20% 后重建训练窗口重算 wf val —— 冠军的
            优势是否依赖最早期那段数据

本模块只做纯计算 / 汇总（不 import engine，避免循环依赖），便于单测；
engine 负责数据切分与评估编排（见 engine._run_finalist_robustness）。
"""
from __future__ import annotations

import math
from typing import Any

try:  # 与 engine.py 同一降级策略：无 strategy_manager 时用 sign(tanh)
    from strategy_manager.signal import compute_target_positions_stateless as _positions
except ImportError:  # pragma: no cover
    def _positions(factors):  # type: ignore[no-redef]
        import torch as _torch
        return _torch.sign(_torch.tanh(factors))


# ── 阈值（引擎侧判定使用，集中定义便于单测对齐）────────────────────────────

# 冠军在单变体上 val / 盈利必须 > 0 才计入「稳健」
MIN_POSITIVE = 0.0
# 排名保持：冠军至少保持第一的比例（≥ 2/3 变体）
RANK_MAJORITY = 2 / 3
# start 轴：丢弃训练区开头后剩余窗口的最小根数（不足则跳过该变体）
MIN_REMAIN_BARS = 200


def production_pnl_stats(
    factors: Any,
    t_ret: Any,
    start: int,
    end: int,
    cost: float,
    periods_per_year: float,
) -> dict[str, float] | None:
    """在 [start, end) 窗口按生产口径（tanh 仓位 + 换手成本）回测一条因子。

    与 `_rigorous_holdout_pnl` 的 pnl 公式一致：pos = tanh 仓位函数，
    pnl = pos × target_ret − |Δpos| × cost；只做张量数学，返回 JSON 友好 dict。
    窗口为空 / 因子非法时返回 None。
    """
    pos = _positions(factors)
    prev = pos.roll(1, dims=1)
    prev[:, 0] = 0.0
    turnover = (pos - prev).abs()
    pnl = pos * t_ret - turnover * cost
    w = pnl[:, start:end]
    if w.numel() == 0:
        return None
    mean = float(w.mean())
    std = float(w.std()) + 1e-9
    ppy = max(1.0, float(periods_per_year))
    return {
        "sharpe": round(mean / std * math.sqrt(ppy), 3),
        "ann_ret_pct": round(mean * ppy * 100.0, 3),
        "total_return_pct": round(float(w.sum()) * 100.0, 3),
        "turnover": round(float(turnover[:, start:end].mean()), 4),
        "cost_rate": round(float(cost), 6),
    }


def build_variants(
    T: int,
    n_folds: int,
    gap: int,
    shift_fracs: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> list[dict[str, Any]]:
    """构造 fold / start 两轴的变体计划（纯描述符，供 engine 循环评估）。

    每项: {axis, n_folds, gap, shift, label}
      - axis=fold : n_folds ∈ {base-1, base, base+1}（≥2），shift=0
      - axis=start: n_folds=base, shift = round(frac × T)（T-shift ≥ MIN_REMAIN_BARS）
    """
    out: list[dict[str, Any]] = []
    for nf in sorted({n_folds - 1, n_folds, n_folds + 1}):
        if nf >= 2:
            out.append({
                "axis": "fold", "n_folds": nf, "gap": int(gap),
                "shift": 0, "label": f"folds={nf}",
            })
    for frac in shift_fracs:
        d = int(round(frac * T))
        if 0 < d <= T - MIN_REMAIN_BARS:
            out.append({
                "axis": "start", "n_folds": n_folds, "gap": int(gap),
                "shift": d, "label": f"drop_head={d} ({frac:.0%})",
            })
    return out


def rank_index(vals: list[float | None]) -> int | None:
    """返回最大值所在下标（None 视作 −∞）。全 None 时返回 None。"""
    best_i: int | None = None
    best_v: float | None = None
    for i, v in enumerate(vals):
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v, best_i = float(v), i
    return best_i


def annotate_rows(
    rows: list[dict[str, Any]],
    fml_count: int,
) -> list[dict[str, Any]]:
    """为每行的 flat f0..f{k-1} 数值列标注 champion（下标 0）的 val/rank。

    fml_count 控制读取哪些列（cost 行的 f0_ann 等带后缀键不会被误读）。
    """
    for row in rows:
        ordered = [row.get(f"f{i}") for i in range(fml_count)]
        champ_v = ordered[0]
        ri = rank_index(ordered)
        row["champ_val"] = round(champ_v, 4) if champ_v is not None else None
        row["champ_rank"] = (ri + 1) if ri is not None else None  # 1-based，None=全员无效
    return rows


def axis_verdict(
    rows: list[dict[str, Any]],
    *,
    label: str,
    need_rank_majority: bool = True,
) -> dict[str, Any]:
    """由标注后的变体行汇总单轴结论。

    规则：
      - pass_val   : 冠军在所有有效变体上 val > MIN_POSITIVE
      - pass_rank  : 冠军在 ≥ ceil(2/3 × 有效变体数) 上保持第一
      - passed     : 两者都满足（无有效变体时视为 not-applicable → False）
    """
    valid = [r for r in rows if r.get("champ_val") is not None]
    if not valid:
        return {"axis": label, "passed": False, "reason": "无有效变体（数据过短）",
                "n_valid": 0, "positive_frac": 0.0, "rank1_frac": 0.0}
    pos_frac = sum(1 for r in valid if r["champ_val"] > MIN_POSITIVE) / len(valid)
    ranks = [r.get("champ_rank") for r in valid]
    rank1 = sum(1 for rk in ranks if rk == 1)
    rank_frac = rank1 / len(valid)
    need = 1.0 if not need_rank_majority else RANK_MAJORITY
    pass_rank = rank_frac >= need
    pass_val = pos_frac == 1.0
    return {
        "axis": label,
        "passed": bool(pass_val and pass_rank),
        "n_valid": len(valid),
        "positive_frac": round(pos_frac, 3),
        "rank1_frac": round(rank_frac, 3),
        "reason": None if (pass_val and pass_rank) else (
            "冠军 val 存在 ≤0 变体" if not pass_val else "冠军排名保持率不足"
        ),
    }
