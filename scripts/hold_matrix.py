"""持仓管理策略正交组合矩阵实验（含默认 信号跟随）。

对全部方案（signal/risk/hybrid/be/time/chandelier/dd）做全组合，
每个格子 = 两种策略模块叠加（对角线 = 单策略；signal 与其他叠加 = 只保留另一个模块，
即“信号跟随 基线”行/列），用离散撮合引擎（web.paper_replay.run_replay，与模拟实盘/回测
同口径）在同一份数据、同一因子、同一成本下回放，产出 收益/夏普/交易数/最大回撤 矩阵 + 排名。

用法：
  .venv/bin/python scripts/hold_matrix.py \
      [--data-file data/slices/BTCUSDT_M5.parquet] \
      [--strategy-file strategies/best_BTCUSDT.json] \
      [--commission 0.02] [--slippage 0.01]

输出：
  results/hold_matrix_<时间戳>.{json,md,csv}  + results/hold_matrix_latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline.parquet_manager import ParquetDataManager  # noqa: E402
from model_core.backtest import estimate_periods_per_year  # noqa: E402
from model_core.features import FeatureEngineer  # noqa: E402
from model_core.vm import StackVM  # noqa: E402
from web.hold_matrix_curves import (save_curves,  # noqa: E402
                                    downsample_indices, pnl_quintiles)
from web.hold_policy import combo_id, combo_label  # noqa: E402
from web.paper_replay import run_replay  # noqa: E402

# 全部持仓管理方案：默认 信号跟随 也在矩阵里（作为基线的行/列）
POLICIES = ["signal", "risk", "hybrid", "be", "time", "chandelier", "dd"]

STAT_KEYS = ["total_return", "sharpe", "sortino", "max_drawdown", "n_trades", "win_rate",
             "avg_hold_bars", "profit_loss_ratio", "fees_total"]


def heat_summary_rows(run_order: list[str], curves: dict[str, tuple],
                      results: dict[str, dict], results_dir: Path,
                      max_pts: int = 900) -> list[dict]:
    """全部组合的收益五等分热力汇总（供选型对比表 results/hold_matrix_heat_summary.json）。

    每组合一行：五桶合计收益占比、最赚/最亏段（Kadane 连续区间）与其 bar 范围、
    段内主要出场原因（用 trades 的 bar 区间与段重叠判断，无需重跑）。
    """
    rows: list[dict] = []
    for pid in run_order:
        arrs = curves.get(pid)
        if arrs is None:
            continue
        eq = np.asarray(arrs[0], dtype=float)
        pn = np.asarray(arrs[1], dtype=float)
        quint = pnl_quintiles(pn, idx=downsample_indices(pn.size, max_pts))
        stats_map = results.get(pid) or {}
        trades = stats_map.get("_trades_raw") or []
        row: dict[str, Any] = {
            "combo": pid,
            "label": combo_label(pid),
            "total_return": stats_map.get("total_return"),
            "sharpe": stats_map.get("sharpe"),
            "max_drawdown": stats_map.get("max_drawdown"),
            "n_trades": stats_map.get("n_trades") or stats_map.get("trades"),
            "bucket_stats": quint.get("stats") or [],
        }
        l0 = 0
        for seg_key in ("best_segment", "worst_segment"):
            seg = quint.get(seg_key)
            cell: dict[str, Any] | None = None
            if seg:
                s, e = int(seg["start"]), int(seg["end"])
                # 段内出场的交易 → 主要出场原因（按笔数）
                reasons: dict[str, int] = {}
                for tr in trades:
                    xb = int(tr.get("bar") or 0)
                    if s <= xb <= e:
                        lb = str(tr.get("label") or "?")
                        reasons[lb] = reasons.get(lb, 0) + 1
                top_reasons = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
                cell = {
                    "start_bar": l0 + s,
                    "end_bar": l0 + e,
                    "bars": int(seg.get("bars") or (e - s + 1)),
                    "cum": seg.get("cum"),
                    "n_trades": int(sum(reasons.values())),
                    "top_reasons": [f"{k}×{n}" for k, n in top_reasons],
                }
            row[seg_key] = cell
        rows.append(row)
    return rows

# 多目标筛选的目标键：均为“越高越好”。max_drawdown 存为负值（-0.018 = -1.8%），
# 因此越接近 0 越好，直接比较即可（-0.01 > -0.02 = 回撤更小），无需取负。
PARETO_OBJ_KEYS = ["total_return", "sharpe", "profit_loss_ratio"]


def _obj_vec(r: dict) -> list[float]:
    vec = []
    for k in PARETO_OBJ_KEYS:
        v = r.get(k)
        vec.append(float(v) if isinstance(v, (int, float)) else -float("inf"))
    dd = r.get("max_drawdown")
    vec.append(float(dd) if isinstance(dd, (int, float)) else -float("inf"))  # 回撤（负值，越接近 0 越好）
    return vec


def pick_best_combo(ranking: list[dict]) -> dict | None:
    """非基线组合里按夏普取最优（并列取回撤更小者）；无可用组合返回 None。

    基线（signal）不算「最优组合」——它本就是默认；sharpe 缺失的组合跳过。
    """
    best: dict | None = None
    for r in ranking:
        if r.get("combo") == "signal":
            continue
        if r.get("sharpe") is None:
            continue
        if best is None or (r["sharpe"], r.get("max_drawdown") or -99.0) > \
                (best["sharpe"], best.get("max_drawdown") or -99.0):
            best = r
    return best


def _dominates(a: dict, b: dict) -> bool:
    """a 在所有目标上 ≥ b，且至少一项严格 >（a 全面不劣于 b）。"""
    va, vb = _obj_vec(a), _obj_vec(b)
    return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))


def pareto_front(rows: list[dict]) -> list[dict]:
    """多目标非劣解：不被任何其他组合支配（收益↑/夏普↑/盈亏比↑/最大回撤取负↑）。"""
    return [r for r in rows if not any(_dominates(o, r) for o in rows if o is not r)]


def beats_baseline(row: dict, baseline: dict) -> bool:
    """在 夏普/最大回撤/盈亏比 三维上均不劣于 信号跟随 基线，且至少一项严格更好。"""
    keys = ["sharpe", "max_drawdown", "profit_loss_ratio"]
    if not all(isinstance(row.get(k), (int, float)) and isinstance(baseline.get(k), (int, float)) for k in keys):
        return False
    ge = all(row[k] >= baseline[k] for k in keys)
    return ge and any(row[k] > baseline[k] for k in keys)


def load_factor(strategy_file: str, data_file: str, window: int | None):
    strat = json.load(open(strategy_file, encoding="utf-8"))
    formula = [int(t) for t in strat["formula"]]
    pm = ParquetDataManager(data_file)
    pm.load()
    raw = pm.raw_dict
    T = int(raw["close"].shape[1])
    start = 0
    if window:
        window = int(window)
        if window < 800:
            raise SystemExit(f"窗口过小（{window}）：特征 warm-up 需 ≥800 根")
        start = max(0, T - window)
    raw_s = {k: v[:, start:] for k, v in raw.items()}
    feats = FeatureEngineer.compute_features(raw_s)
    vm = StackVM()
    with torch.no_grad():
        factor = vm.execute(formula, feats)
    factor = factor[0].cpu().numpy().astype(float)
    return {
        "factor": factor,
        "open": raw_s["open"][0].numpy().astype(float),
        "high": raw_s["high"][0].numpy().astype(float),
        "low": raw_s["low"][0].numpy().astype(float),
        "close": raw_s["close"][0].numpy().astype(float),
        "time": raw_s.get("time"),
        "bars": factor.shape[0],
        "window_start": start,
        "data_file": str(pm.path) if hasattr(pm, "path") else data_file,
    }


def fmt_pct(v: float) -> str:
    return f"{v * 100.0:+.2f}%" if v is not None else "—"


def resolve_spread_window(data_file: str, window_bars: int | None, n_chunks: int,
                          regime: str
                          ) -> tuple[str | None, list[dict] | None, dict | None, bool]:
    """window_mode=spread 的窗口预处理：regime 分层抽样生成拼接子集。

    返回 (subset_file, blocks, regime_coverage, used)；used=False 表示不需要分层抽样
    （无 window_bars 或非 spread 模式），调用方保持原 data_file 即可。
    """
    if not window_bars or n_chunks <= 0:
        return None, None, None, False
    from data_pipeline.train_sampler import (prepare_training_subset,
                                             preview_training_subset)

    sub = prepare_training_subset(data_file, mode="spread",
                                  n_bars=window_bars,
                                  n_chunks=n_chunks, regime=regime)
    subset_file = sub["data_file"]
    # 块范围用同一几何的预览函数取（prepare 的结果不带 blocks）
    prev = preview_training_subset(data_file, mode="spread",
                                   n_bars=window_bars,
                                   n_chunks=n_chunks, regime=regime)
    blocks = [
        {"kind": b.get("kind"), "start": int(b["start"]), "end": int(b["end"]),
         "regime_class": b.get("regime_class")}
        for b in (prev.get("blocks") or []) if b.get("start") is not None
    ]
    coverage = prev.get("regime_coverage") or sub.get("regime_coverage")
    return subset_file, blocks, coverage, True


def main() -> int:
    ap = argparse.ArgumentParser(description="持仓管理正交组合矩阵（含默认 信号跟随）")
    ap.add_argument("--data-file", default="data/slices/BTCUSDT_M5.parquet")
    ap.add_argument("--strategy-file", default="strategies/best_BTCUSDT.json")
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--slippage", type=float, default=0.01)
    ap.add_argument("--window-bars", type=int, default=None)
    ap.add_argument("--window-mode", choices=["tail", "spread"], default="tail",
                    help="tail=尾部 N 根；spread=按波动率/趋势 regime 分层取块后拼接（样本外均衡）")
    ap.add_argument("--regime", default="vol", choices=["vol", "trend", "equal"])
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--max-position-pct", type=float, default=100.0,
                    help="每笔投入上限（占权益 %，默认 100）；仓位 = 强度 × 上限%")
    ap.add_argument("--threshold", type=float, default=None,
                    help="无信号阈值 |tanh(因子)| 档位（如 0.3/0.5/0.8）；缺省读 Config.MIN_TRADE_EXPOSURE")
    args = ap.parse_args()
    args.max_position_pct = min(200.0, max(1.0, args.max_position_pct))

    window_blocks = None
    regime_coverage = None
    window_n = args.window_bars
    load_window = args.window_bars
    if window_n and args.window_mode == "spread":
        # regime 分层抽样：复用训练采样器（同 warm-up 垫块/regime 分桶逻辑），
        # 得到多块拼接子集 + 原始序列 bar 范围溯源；回放在拼接序列上进行。
        subset_file, window_blocks, regime_coverage, used = resolve_spread_window(
            args.data_file, window_n, args.chunks, args.regime)
        print(f"[窗口] 分层抽样: {args.regime} · {args.chunks} 块 · 共 {window_n} 根 "
              f"→ {subset_file}", flush=True)
        print(f"[窗口] 块范围(原始序列): " + " | ".join(
            f"bar {b['start']}..{b['end']}" for b in window_blocks), flush=True)
        args.data_file = subset_file
        load_window = None  # 子集即窗口，不再截尾

    print(f"加载数据 {args.data_file} …", flush=True)
    d = load_factor(args.strategy_file, args.data_file, load_window)
    print(f"K线 {d['bars']} 根（window_start={d['window_start']}）", flush=True)

    times = d["time"]
    ppy = estimate_periods_per_year(times) if times is not None else 105195.0

    results: dict[str, dict] = {}
    curves: dict[str, tuple] = {}   # pid -> (equity, pnl) 原始曲线（单独存 npz，避免主 JSON 膨胀）
    annotations: dict[str, dict] = {}  # pid -> {trades, dd_events} 买卖点/DD 区间标注
    run_order = []
    # 只跑上三角（含对角线）去重后的组合；signal+X 归一为 X，自然去重
    for i, a in enumerate(POLICIES):
        for b in POLICIES[i:]:
            pid = combo_id(f"{a}+{b}")
            if pid in results:
                continue
            run_order.append(pid)
            rep = run_replay(
                factor=d["factor"], open_p=d["open"], high_p=d["high"],
                low_p=d["low"], close_p=d["close"],
                commission_pct=args.commission, slippage_pct=args.slippage,
                policy_id=pid, max_position_pct=args.max_position_pct,
                threshold=args.threshold,
                track_dd=True,  # 供前端画 DD 熔断区间底色
                periods_per_year=float(ppy),
            )
            curves[pid] = (np.asarray(rep["equity"], dtype=float),
                            np.asarray(rep["pnl"], dtype=float))
            # 标注精简版交易记录：入场/出场 bar、方向、盈亏、出场原因
            annotations[pid] = {
                "trades": [{
                    "eb": max(0, int(t["bar"]) - int(t.get("hold_bars") or 0)),
                    "xb": int(t["bar"]),
                    "side": t.get("side"),
                    "pnl": round(float(t.get("pnl") or 0.0), 8),
                    "reason": t.get("label") or "",
                } for t in rep["trades"]],
                "dd_events": rep.get("dd_events") or [],
            }
            stats = {k: rep["stats"].get(k) for k in STAT_KEYS}
            stats["label"] = combo_label(pid)
            stats["trades"] = len(rep["trades"])
            stats["exit_reasons"] = {}
            for t in rep["trades"]:
                stats["exit_reasons"][t["label"]] = stats["exit_reasons"].get(t["label"], 0) + 1
            stats["_trades_raw"] = rep["trades"]  # 热力汇总用（不写入主 JSON，见 out 构建处剥离）
            results[pid] = stats
            print(f"  {pid:24s} 收益 {fmt_pct(stats.get('total_return')):>8s}  "
                  f"夏普 {stats.get('sharpe') or 0:+.2f}  交易 {stats['trades']:>4d}", flush=True)

    # 基线 signal：已在矩阵里（signal 对角线 = combo_id("signal+signal") → "signal"）
    baseline = dict(results.get("signal") or {})

    # ── 22 组合五等分热力汇总（写 results/hold_matrix_heat_summary.json，供选型对比） ──
    try:
        heat_rows = heat_summary_rows(run_order, curves, results, ROOT / "results")
    except Exception as exc:  # noqa: BLE001 汇总失败不阻断主流程
        print(f"[warn] 热力汇总失败: {exc}", flush=True)
        heat_rows = []
    finally:
        # 剥离内部字段（不进主 JSON/排名）
        for pid in run_order:
            results[pid].pop("_trades_raw", None)

    # 排名（按夏普；只列去重后的实际组合）
    ranking = sorted(
        ({"combo": pid, **results[pid]} for pid in run_order),
        key=lambda r: (r.get("sharpe") is not None, r.get("sharpe") or -999.0),
        reverse=True,
    )

    # ── 多目标排序与帕累托筛选（收益↑ / 夏普↑ / 最大回撤↑ / 盈亏比↑） ──
    rows_all = [{"combo": pid, **results[pid]} for pid in run_order]
    pareto = pareto_front(rows_all)
    pareto_ids = {r["combo"] for r in pareto}
    beats_ids = {r["combo"] for r in rows_all if beats_baseline(r, baseline)}
    focus_ids = pareto_ids & beats_ids
    for r in ranking:
        r["pareto"] = r["combo"] in pareto_ids
        r["beats_base"] = r["combo"] in beats_ids
        r["focus"] = r["combo"] in focus_ids
    pareto_out = [dict(r) for r in pareto]
    focus_out = [dict(r) for r in ranking if r["combo"] in focus_ids]

    # 样本外溯源感知：tail 窗口按训练真 holdout 边界打 真伪 OOS 标签（须在 out 构建前）
    oos_info: dict | None = None
    if args.window_bars and not window_blocks:
        try:
            from web.oos_provenance import classify_window, oos_status_label

            oos_info = classify_window(args.strategy_file, d["data_file"],
                                       d["window_start"], args.window_bars)
            if oos_info and oos_info["status"] != "unavailable":
                win_txt = (f"样本外窗口：仅最后 {args.window_bars} 根（原始序列 bar "
                           f"{d['window_start']}..{d['window_start'] + d['bars'] - 1} · "
                           f"{oos_status_label(oos_info)}")
                if oos_info.get("honest"):
                    h = oos_info["honest"]
                    win_txt += (f" · 诚实 OOS 建议 bar {h['start_bar']}..{h['end_bar']}"
                                f"（{h['n_bars']} 根，{h['note']}")
                    win_txt += "）"
                win_txt += "）"
        except Exception:  # noqa: BLE001 溯源失败回退旧文案
            oos_info = None
    if oos_info is None:
        win_txt = "全部历史"
        if window_blocks:
            win_txt = (f"分层抽样窗口：{args.regime} · {args.chunks} 块 · 共 {sum(b['end'] - b['start'] + 1 for b in window_blocks if b['kind'] != 'warmup')} 根"
                       f"（原始序列 " + " | ".join(f"bar {b['start']}..{b['end']}" for b in window_blocks) + "）")
        elif args.window_bars:
            win_txt = (f"样本外窗口：仅最后 {args.window_bars} 根（原始序列 bar "
                       f"{d['window_start']}..{d['window_start'] + d['bars'] - 1}，训练未见的新数据）")

    # ── 同品种「最优组合」侧车（回测页默认持仓管理 + 最大回撤约束标注用） ──
    best_row = pick_best_combo(ranking)
    strategy_meta: dict[str, Any] = {}
    try:
        strategy_meta = json.loads(
            Path(args.strategy_file).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    best_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": strategy_meta.get("symbol") or "",
        "timeframe": strategy_meta.get("timeframe") or "",
        "strategy_file": args.strategy_file,
        "data_file": d["data_file"],
        "window_mode": args.window_mode,
        "window_bars": args.window_bars,
        "window_start": d["window_start"],
        "window_blocks": window_blocks,
        "combo": best_row["combo"] if best_row else None,
        "combo_label": best_row.get("label") if best_row else None,
        "sharpe": best_row.get("sharpe") if best_row else None,
        "total_return": best_row.get("total_return") if best_row else None,
        "max_drawdown": best_row.get("max_drawdown") if best_row else None,
        "profit_loss_ratio": best_row.get("profit_loss_ratio") if best_row else None,
        "n_trades": best_row.get("n_trades") if best_row else None,
        "baseline_signal_sharpe": baseline.get("sharpe"),
        "baseline_signal_return": baseline.get("total_return"),
        "note": "最优=非基线组合按夏普取优（并列取回撤更小）；max_drawdown 为负值（-0.018=-1.8%），用作该组合的约束参考",
    }
    (ROOT / "results" / "matrix_best_combo.json").write_text(
        json.dumps(best_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 热力汇总独立文件（聚合 22 组合五等分统计；主 JSON 也带一份 heat_summary）
    out_generated_at_iso = datetime.now(timezone.utc).isoformat()
    if heat_rows:
        heat_out = {
            "generated_at": out_generated_at_iso,
            "data_file": d["data_file"],
            "strategy_file": args.strategy_file,
            "bars": d["bars"],
            "window_bars": args.window_bars,
            "commission_pct": args.commission,
            "slippage_pct": args.slippage,
            "rows": heat_rows,
        }
        (ROOT / "results" / "hold_matrix_heat_summary.json").write_text(
            json.dumps(heat_out, ensure_ascii=False, indent=2), encoding="utf-8")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = {
        "generated_at": out_generated_at_iso,
        "data_file": d["data_file"],
        "strategy_file": args.strategy_file,
        "bars": d["bars"],
        "window_bars": args.window_bars,
        "window_start": d["window_start"],
        "window_mode": args.window_mode,
        "window_blocks": window_blocks,
        "regime": args.regime,
        "chunks": args.chunks,
        "regime_coverage": regime_coverage,
        "commission_pct": args.commission,
        "slippage_pct": args.slippage,
        "signal_threshold": args.threshold,  # 统一无信号阈值（None = Config 默认 0.05）
        "oos": oos_info,
        "policies": POLICIES,
        "baseline_signal": baseline,
        "cells": results,
        "ranking": ranking,
        "pareto_front": pareto_out,
        "focus_list": focus_out,
        "heat_summary": heat_rows,
    }
    for f in ("json", "md", "csv"):
        Path("results").mkdir(exist_ok=True)
    base = ROOT / "results" / f"hold_matrix_{ts}"
    with (ROOT / "results" / "hold_matrix_latest.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 资金曲线侧车（npz + meta json），供页面点行绘制 资金曲线/滚动夏普
    import shutil

    curves_meta = {
        "generated_at": out["generated_at"],
        "window_bars": args.window_bars,
        "window_start": d["window_start"],
        "window_mode": args.window_mode,
        "window_blocks": window_blocks,
        "bars": d["bars"],
        "ppy": float(ppy),
        "combos": run_order,
        "data_file": d["data_file"],
    }
    save_curves(curves, curves_meta, annotations=annotations)
    shutil.copyfile(ROOT / "results" / "hold_matrix_curves_latest.npz",
                    ROOT / "results" / f"hold_matrix_curves_{ts}.npz")
    shutil.copyfile(ROOT / "results" / "hold_matrix_curves_latest.json",
                    ROOT / "results" / f"hold_matrix_curves_{ts}.json")
    with base.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 组合语义与顺序无关（并集出场）→ 单元格统一按 POLICIES 顺序归一后查表，避免
    # “risk+hybrid / hybrid+risk”重复跑批造成的 KeyError
    _pidx = {p: i for i, p in enumerate(POLICIES)}

    def canon(a: str, b: str) -> str:
        return combo_id("+".join(sorted([a, b], key=lambda p: _pidx.get(p, 99))))

    # MD 报告
    def cell(a: str, b: str, key: str) -> str:
        v = results[canon(a, b)].get(key)
        return f"{v:+.2f}" if isinstance(v, (int, float)) else "—"

    md = [f"# 持仓管理 {len(POLICIES)}×{len(POLICIES)} 组合矩阵 · 含默认 信号跟随（{ts}）",
          "",
          f"- 数据：`{d['data_file']}`（{d['bars']} 根 · {win_txt}）",
          f"- 因子：`{args.strategy_file}`；成本：手续费 {args.commission}% / 滑点 {args.slippage}%",
          f"- 基线 signal：收益 {fmt_pct(baseline.get('total_return'))}，夏普 {baseline.get('sharpe') or 0:+.2f}，交易 {baseline['trades']}",
          "",
          "## 总收益矩阵（% · 行策略 + 列策略 叠加）",
          "",
          "| 行\\列 | " + " | ".join(POLICIES) + " |",
          "|" + "---|" * (len(POLICIES) + 1),
          ]
    for a in POLICIES:
        row = [f"**{a}**"] + [fmt_pct(results[canon(a, b)].get("total_return")) for b in POLICIES]
        md.append("| " + " | ".join(row) + " |")
    md += ["", "## 夏普矩阵",
           "", "| 行\\列 | " + " | ".join(POLICIES) + " |",
           "|" + "---|" * (len(POLICIES) + 1)]
    for a in POLICIES:
        row = [f"**{a}**"] + [cell(a, b, "sharpe") for b in POLICIES]
        md.append("| " + " | ".join(row) + " |")
    md += ["", "## 交易数矩阵",
           "", "| 行\\列 | " + " | ".join(POLICIES) + " |",
           "|" + "---|" * (len(POLICIES) + 1)]
    for a in POLICIES:
        row = [f"**{a}**"] + [str(results[canon(a, b)].get("n_trades") or "—") for b in POLICIES]
        md.append("| " + " | ".join(row) + " |")
    md += ["", "## 最大回撤矩阵（%）",
           "", "| 行\\列 | " + " | ".join(POLICIES) + " |",
           "|" + "---|" * (len(POLICIES) + 1)]
    for a in POLICIES:
        row = [f"**{a}**"] + [
            f"{results[canon(a, b)].get('max_drawdown') * 100:.1f}%"
            if results[canon(a, b)].get("max_drawdown") is not None else "—"
            for b in POLICIES]
        md.append("| " + " | ".join(row) + " |")
    md += ["", "## 排名（按夏普）", "",
           "| 排名 | 组合 | 收益 | 夏普 | 索提诺 | 最大回撤 | 交易 | 胜率 | 盈亏比 | 出场构成 |",
           "|--|--|--|--|--|--|--|--|--|--|"]
    for k, r in enumerate(ranking, 1):
        reasons = " ".join(f"{lb}×{n}" for lb, n in sorted(r["exit_reasons"].items()))
        mdd = r.get("max_drawdown")
        mdd_s = f"{mdd * 100:.1f}%" if mdd is not None else "—"
        md.append(f"| {k} | `{r['combo']}`（{r['label']}） | {fmt_pct(r.get('total_return'))} | "
                  f"{r.get('sharpe') or 0:+.2f} | {r.get('sortino') or 0:+.2f} | {mdd_s} | {r.get('n_trades') or 0} | "
                  f"{(r.get('win_rate') or 0) * 100:.0f}% | {r.get('profit_loss_ratio') or 0:.2f} | {reasons} |")
    md.append("")
    with base.with_suffix(".md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # CSV
    import csv
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "combo"] + STAT_KEYS)
        for a in POLICIES:
            for b in POLICIES:
                pid = canon(a, b)
                w.writerow([a, b, pid] + [results[pid].get(k) for k in STAT_KEYS])

    print(f"\n=== 排名前 8（基线 signal 收益 {fmt_pct(baseline.get('total_return'))} 夏普 {baseline.get('sharpe') or 0:+.2f}）===")
    for k, r in enumerate(ranking[:8], 1):
        mdd = r.get("max_drawdown")
        mdd_s = f"{mdd * 100:.1f}%" if mdd is not None else "—"
        print(f"{k:>2}. {r['combo']:24s} {r['label']:22s} "
              f"收益 {fmt_pct(r.get('total_return')):>8s}  夏普 {r.get('sharpe') or 0:+.2f}  "
              f"回撤 {mdd_s:>7s}  交易 {r.get('n_trades') or 0}")
    print(f"\n结果已写入 results/hold_matrix_{ts}.{{json,md,csv}} + hold_matrix_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())