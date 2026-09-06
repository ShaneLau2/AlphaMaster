"""三轴联合回测：持仓方案 × 每笔投入上限% × 无信号阈值。

对全部持仓方案（signal/risk/hybrid/be/time/chandelier/dd 的 22 个正交去重组合）×
每档投入上限 × 每档无信号阈值 做全网格离散撮合回放（web.paper_replay.run_replay，
与模拟实盘/回测同口径），同一份数据、同一因子、同一成本下一次跑完 ——
把「持仓正交组合矩阵」「阈值敏感性·观望带」「A/B·持仓方案对比」三个面板合并成一个三维扫描。

用法：
  .venv/bin/python scripts/combo_sweep.py \\
      [--data-file data/slices/BTCUSDT_M5.parquet] \\
      [--strategy-file strategies/best_BTCUSDT.json] \\
      [--commission 0.02] [--slippage 0.01] \\
      [--caps 10,25,100] [--thresholds 0.05,0.3,0.5,0.8]

输出：
  results/combo_sweep_<时间戳>.{json,md,csv} + results/combo_sweep_latest.json
  （基线切片 = 第一档上限 × 第一档阈值）另写 hold_matrix_latest.json + 资金曲线
  侧车 + matrix_best_combo.json + hold_matrix_heat_summary.json —— 前端矩阵视图
  无需改动即可渲染基线切片，其余切片由前端按 rows 客户端切片。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 复用 hold_matrix 的注册表/加载/帕累托/热力/最优组合逻辑（同一套口径）
from scripts.hold_matrix import (POLICIES, beats_baseline, fmt_pct,  # noqa: E402
                                 heat_summary_rows, load_factor, pareto_front,
                                 pick_best_combo, resolve_spread_window)
from model_core.backtest import estimate_periods_per_year  # noqa: E402
from web.hold_matrix_curves import save_curves  # noqa: E402
from web.hold_policy import combo_id, combo_label  # noqa: E402
from web.paper_replay import WARMUP_BARS, run_replay  # noqa: E402

STAT_KEYS = ["total_return", "sharpe", "sortino", "max_drawdown", "n_trades", "win_rate",
             "avg_hold_bars", "profit_loss_ratio", "fees_total"]


def normalize_caps(caps: list[float] | None, default: list[float] | None = None) -> list[float]:
    """上限%档位归一化：钳到 1..200、去重、升序；缺省/全非法时回退 default 或 [100]。"""
    if caps:
        out = sorted({min(200.0, max(1.0, float(c))) for c in caps
                      if c is not None and float(c) > 0})
        if out:
            return out
    return sorted(default or [100.0])


def normalize_thresholds(thresholds: list[float] | None,
                         default: list[float] | None = None) -> list[float]:
    """无信号阈值档位归一化：(0,1) 内、去重、升序；缺省/全非法时回退默认四档。"""
    if thresholds:
        out = sorted({round(float(t), 4) for t in thresholds
                      if t is not None and 0 < float(t) < 1})
        if out:
            return out
    return sorted(default or [0.05, 0.3, 0.5, 0.8])


def _best_key(r: dict) -> tuple:
    return (float(r["sharpe"]) if r.get("sharpe") is not None else float("-inf"),
            float(r.get("n_trades") or 0.0))


def best_row(rows: list[dict], where=None) -> dict | None:
    """在 rows 中按 夏普（并列取交易更多）选最优；只认有交易且夏普非 None 的行。"""
    cand = [r for r in rows if (r.get("n_trades") or 0) > 0 and r.get("sharpe") is not None
            and (where is None or where(r))]
    if not cand:
        return None
    return max(cand, key=_best_key)


def row_brief(r: dict) -> dict:
    """最优行的精简快照（供 best_overall / best_per_* 落盘）。"""
    keys = ["combo", "combo_label", "cap_pct", "threshold", "flat_share",
            "total_return", "sharpe", "sortino", "max_drawdown", "n_trades",
            "win_rate", "profit_loss_ratio", "avg_hold_bars"]
    return {k: r.get(k) for k in keys}


def combo_order() -> list[str]:
    """22 个上三角去重组合（含 信号跟随 基线），与 hold_matrix 同规则。"""
    order: list[str] = []
    for i, a in enumerate(POLICIES):
        for b in POLICIES[i:]:
            pid = combo_id(f"{a}+{b}")
            if pid not in order:
                order.append(pid)
    return order


def main() -> int:
    ap = argparse.ArgumentParser(description="三轴联合回测：持仓方案 × 上限% × 无信号阈值")
    ap.add_argument("--data-file", default="data/slices/BTCUSDT_M5.parquet")
    ap.add_argument("--strategy-file", default="strategies/best_BTCUSDT.json")
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--slippage", type=float, default=0.01)
    ap.add_argument("--window-bars", type=int, default=None)
    ap.add_argument("--window-mode", choices=["tail", "spread"], default="tail",
                    help="tail=尾部 N 根；spread=按波动率/趋势 regime 分层取块后拼接（样本外均衡）")
    ap.add_argument("--regime", default="vol", choices=["vol", "trend", "equal"])
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--caps", default="10,25,100",
                    help="每笔投入上限% 档位（逗号分隔，如 10,25,100；每个组合×每档各跑一行）")
    ap.add_argument("--thresholds", default="0.05,0.3,0.5,0.8",
                    help="无信号阈值 |tanh(因子)| 档位（逗号分隔，0<t<1）")
    args = ap.parse_args()

    caps = normalize_caps([float(c) for c in str(args.caps).replace("，", ",").split(",") if c.strip()])
    thr_levels = normalize_thresholds(
        [float(t) for t in str(args.thresholds).replace("，", ",").split(",") if t.strip()])
    print(f"[轴] 持仓方案 {len(POLICIES)}×{len(POLICIES)} → {len(combo_order())} 个组合 · "
          f"上限% {caps} · 阈值 {thr_levels} · 共 {len(combo_order()) * len(caps) * len(thr_levels)} 次回放",
          flush=True)

    window_blocks = None
    regime_coverage = None
    load_window = args.window_bars
    if args.window_bars and args.window_mode == "spread":
        subset_file, window_blocks, regime_coverage, used = resolve_spread_window(
            args.data_file, args.window_bars, args.chunks, args.regime)
        print(f"[窗口] 分层抽样: {args.regime} · {args.chunks} 块 · 共 {args.window_bars} 根 "
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

    # 信号强度序列（引擎同口径：pos = tanh(因子)），观望占比按 warm-up 之后统计
    pos_np = np.abs(np.tanh(d["factor"]))
    wu = max(800, int(WARMUP_BARS))
    seg = pos_np[wu:] if pos_np.size > wu else pos_np

    pids = combo_order()
    base_cap, base_thr = caps[0], thr_levels[0]  # 基线切片 = 第一档上限 × 第一档阈值

    rows: list[dict] = []
    curves: dict[str, tuple] = {}      # 基线切片 pid -> (equity, pnl)
    annotations: dict[str, dict] = {}  # 基线切片 pid -> {trades, dd_events}
    base_stats: dict[str, dict] = {}   # 基线切片 pid -> stats（含 exit_reasons，供矩阵 JSON）
    for pid in pids:
        for cap in caps:
            for t in thr_levels:
                is_base = (cap == base_cap and t == base_thr)
                rep = run_replay(
                    factor=d["factor"], open_p=d["open"], high_p=d["high"],
                    low_p=d["low"], close_p=d["close"],
                    commission_pct=args.commission, slippage_pct=args.slippage,
                    policy_id=pid, max_position_pct=cap, threshold=t,
                    track_dd=is_base,  # 只有基线切片需要 DD 区间标注（曲线视图用）
                    periods_per_year=float(ppy),
                )
                st = rep["stats"]
                flat_share = float(np.mean(seg < t)) if seg.size else 0.0
                n_flat = int(np.sum(seg < t)) if seg.size else 0
                pnls = [tr["pnl"] for tr in rep["trades"]]
                row: dict[str, Any] = {
                    "combo": pid,
                    "combo_label": combo_label(pid),
                    "cap_pct": float(cap),
                    "threshold": float(t),
                    "flat_share": round(flat_share, 4),
                    "flat_bars": n_flat,
                    "signal_bars": int(seg.size - n_flat) if seg.size else 0,
                    "total_return": st.get("total_return"),
                    "sharpe": st.get("sharpe"),
                    "sortino": st.get("sortino"),
                    "max_drawdown": st.get("max_drawdown"),
                    "n_trades": st.get("n_trades"),
                    "win_rate": st.get("win_rate"),
                    "profit_loss_ratio": st.get("profit_loss_ratio"),
                    "fees_total": st.get("fees_total"),
                    "avg_hold_bars": st.get("avg_hold_bars"),
                    "max_win": round(float(max(pnls)), 6) if pnls else None,
                    "max_single_loss": round(float(min(pnls)), 6) if pnls else None,
                }
                rows.append(row)
                if is_base:
                    curves[pid] = (np.asarray(rep["equity"], dtype=float),
                                   np.asarray(rep["pnl"], dtype=float))
                    annotations[pid] = {
                        "trades": [{
                            "eb": max(0, int(t2["bar"]) - int(t2.get("hold_bars") or 0)),
                            "xb": int(t2["bar"]),
                            "side": t2.get("side"),
                            "pnl": round(float(t2.get("pnl") or 0.0), 8),
                            "reason": t2.get("label") or "",
                        } for t2 in rep["trades"]],
                        "dd_events": rep.get("dd_events") or [],
                    }
                    st2 = {k: st.get(k) for k in STAT_KEYS}
                    st2["label"] = combo_label(pid)
                    st2["trades"] = len(rep["trades"])
                    st2["exit_reasons"] = {}
                    for t2 in rep["trades"]:
                        st2["exit_reasons"][t2["label"]] = st2["exit_reasons"].get(t2["label"], 0) + 1
                    st2["_trades_raw"] = rep["trades"]  # 热力汇总用（写矩阵 JSON 前剥离）
                    base_stats[pid] = st2
                print(f"  {pid:24s} cap={cap:g}% t={t:g} 收益 {fmt_pct(row['total_return']):>8s}  "
                      f"夏普 {row['sharpe'] or 0:+.2f}  交易 {row['n_trades'] or 0:>4d}", flush=True)

    # ── 三轴结果：最优（全局 / 每档阈值 / 每档上限）────────────────────────
    best_overall = best_row(rows)
    best_per_threshold = [
        {"threshold": t, "best": row_brief(b)} if (b := best_row(rows, lambda r, t=t: r["threshold"] == t)) else {"threshold": t, "best": None}
        for t in thr_levels
    ]
    best_per_cap = [
        {"cap_pct": c, "best": row_brief(b)} if (b := best_row(rows, lambda r, c=c: r["cap_pct"] == c)) else {"cap_pct": c, "best": None}
        for c in caps
    ]
    ranking = sorted(
        (dict(r) for r in rows if (r.get("n_trades") or 0) > 0 and r.get("sharpe") is not None),
        key=lambda r: (float(r["sharpe"]), float(r.get("n_trades") or 0.0)), reverse=True)[:60]

    # ── 基线切片 → 兼容 hold_matrix_latest.json（前端矩阵视图/曲线/帕累托原样可用）──
    baseline = dict(base_stats.get("signal") or {})
    heat_rows: list[dict] = []
    try:
        heat_rows = heat_summary_rows(pids, curves, base_stats, ROOT / "results")
    except Exception as exc:  # noqa: BLE001 热力汇总失败不阻断
        print(f"[warn] 热力汇总失败: {exc}", flush=True)
    finally:
        for pid in pids:
            base_stats[pid].pop("_trades_raw", None)
    base_ranking = sorted(
        ({"combo": pid, **base_stats[pid]} for pid in pids),
        key=lambda r: (r.get("sharpe") is not None, r.get("sharpe") or -999.0),
        reverse=True,
    )
    rows_all = [{"combo": pid, **base_stats[pid]} for pid in pids]
    pareto_ids = {r["combo"] for r in pareto_front(rows_all)}
    beats_ids = {r["combo"] for r in rows_all if beats_baseline(r, baseline)}
    focus_ids = pareto_ids & beats_ids
    for r in base_ranking:
        r["pareto"] = r["combo"] in pareto_ids
        r["beats_base"] = r["combo"] in beats_ids
        r["focus"] = r["combo"] in focus_ids

    # 样本外溯源感知（tail 窗口）
    oos_info: dict | None = None
    if args.window_bars and not window_blocks:
        try:
            from web.oos_provenance import classify_window
            oos_info = classify_window(args.strategy_file, d["data_file"],
                                       d["window_start"], args.window_bars)
        except Exception:  # noqa: BLE001
            oos_info = None
    if oos_info and oos_info["status"] == "unavailable":
        oos_info = None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    gen_iso = datetime.now(timezone.utc).isoformat()
    meta_common = {
        "generated_at": gen_iso,
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
    }

    # ── 主输出：combo_sweep_<ts>.{json,md,csv} + combo_sweep_latest.json ──
    out: dict[str, Any] = {
        **meta_common,
        "policies": POLICIES,
        "caps": caps,
        "thresholds": thr_levels,
        "baseline_slice": {"cap_pct": base_cap, "threshold": base_thr},
        "oos": oos_info,
        "rows": rows,
        "best_overall": row_brief(best_overall) if best_overall else None,
        "best_per_threshold": best_per_threshold,
        "best_per_cap": best_per_cap,
        "ranking": ranking,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with (ROOT / "results" / "combo_sweep_latest.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with (ROOT / "results" / f"combo_sweep_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── 兼容输出：基线切片 = hold_matrix_latest.json + 曲线侧车 + best/heat ──
    best_row_slice = pick_best_combo(base_ranking)
    strategy_meta: dict[str, Any] = {}
    try:
        strategy_meta = json.loads(Path(args.strategy_file).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    best_out = {
        "generated_at": gen_iso,
        "symbol": strategy_meta.get("symbol") or "",
        "timeframe": strategy_meta.get("timeframe") or "",
        "strategy_file": args.strategy_file,
        "data_file": d["data_file"],
        "window_mode": args.window_mode,
        "window_bars": args.window_bars,
        "window_start": d["window_start"],
        "window_blocks": window_blocks,
        "combo": best_row_slice["combo"] if best_row_slice else None,
        "combo_label": best_row_slice.get("label") if best_row_slice else None,
        "sharpe": best_row_slice.get("sharpe") if best_row_slice else None,
        "total_return": best_row_slice.get("total_return") if best_row_slice else None,
        "max_drawdown": best_row_slice.get("max_drawdown") if best_row_slice else None,
        "profit_loss_ratio": best_row_slice.get("profit_loss_ratio") if best_row_slice else None,
        "n_trades": best_row_slice.get("n_trades") if best_row_slice else None,
        "baseline_signal_sharpe": baseline.get("sharpe"),
        "baseline_signal_return": baseline.get("total_return"),
        "note": "最优=非基线组合按夏普取优（并列取回撤更小）；基线切片 = 第一档上限 × 第一档阈值",
    }
    (ROOT / "results" / "matrix_best_combo.json").write_text(
        json.dumps(best_out, ensure_ascii=False, indent=2), encoding="utf-8")
    if heat_rows:
        heat_out = {
            "generated_at": gen_iso,
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
    matrix_out: dict[str, Any] = {
        **meta_common,
        "signal_threshold": base_thr,  # 基线切片的无信号阈值
        "oos": oos_info,
        "policies": POLICIES,
        "baseline_signal": baseline,
        "cells": base_stats,
        "ranking": base_ranking,
        "pareto_front": [dict(r) for r in rows_all if r["combo"] in pareto_ids],
        "focus_list": [dict(r) for r in base_ranking if r["combo"] in focus_ids],
        "heat_summary": heat_rows,
    }
    with (ROOT / "results" / "hold_matrix_latest.json").open("w", encoding="utf-8") as f:
        json.dump(matrix_out, f, ensure_ascii=False, indent=2)
    curves_meta = {
        "generated_at": gen_iso,
        "window_bars": args.window_bars,
        "window_start": d["window_start"],
        "window_mode": args.window_mode,
        "window_blocks": window_blocks,
        "bars": d["bars"],
        "ppy": float(ppy),
        "combos": pids,
        "data_file": d["data_file"],
        "cap_pct": base_cap,
        "threshold": base_thr,
    }
    save_curves(curves, curves_meta, annotations=annotations)
    import shutil
    shutil.copyfile(ROOT / "results" / "hold_matrix_curves_latest.npz",
                    ROOT / "results" / f"hold_matrix_curves_{ts}.npz")
    shutil.copyfile(ROOT / "results" / "hold_matrix_curves_latest.json",
                    ROOT / "results" / f"hold_matrix_curves_{ts}.json")

    # ── MD 报告 ──
    def fmt_r(r: dict) -> str:
        mdd = r.get("max_drawdown")
        mdd_s = f"{mdd * 100:.1f}%" if mdd is not None else "—"
        return (f"`{r.get('combo') or '—'}`（{r.get('combo_label') or '—'}）· 上限 {r.get('cap_pct') or '—':g}% · "
                f"t={r.get('threshold') or '—':g} · 收益 {fmt_pct(r.get('total_return'))} · "
                f"夏普 {r.get('sharpe') or 0:+.2f} · 回撤 {mdd_s} · 交易 {r.get('n_trades') or 0} · "
                f"观望 {r.get('flat_share') or 0:.0%}")

    md = [f"# 三轴联合回测 · 持仓方案 × 上限% × 无信号阈值（{ts}）", "",
          f"- 数据：`{d['data_file']}`（{d['bars']} 根）· 因子：`{args.strategy_file}`",
          f"- 成本：手续费 {args.commission}% / 滑点 {args.slippage}% · "
          f"组合 {len(pids)} × 上限 {caps} × 阈值 {thr_levels} = {len(rows)} 次回放",
          f"- 基线切片：上限 {base_cap:g}% × t={base_thr:g}（hold_matrix_latest.json 与此切片一致）",
          ""]
    if best_overall:
        md += ["## 全局最优", "", f"- 🎯 {fmt_r(best_overall)}", ""]
    md += ["## 每档阈值最优", "",
           "| 阈值 | 最优（组合 · 上限 · 收益 · 夏普 · 回撤 · 交易 · 观望） |",
           "|--|--|"]
    for item in best_per_threshold:
        b = item.get("best")
        md.append(f"| t={item['threshold']:g} | {fmt_r(b) if b else '—'} |")
    md += ["", "## 每档上限最优", "",
           "| 上限% | 最优（组合 · 阈值 · 收益 · 夏普 · 回撤 · 交易） |",
           "|--|--|"]
    for item in best_per_cap:
        b = item.get("best")
        md.append(f"| {item['cap_pct']:g}% | {fmt_r(b) if b else '—'} |")
    md += ["", "## 排名前 20（按夏普）", "",
           "| 排名 | 组合 | 上限% | 阈值 | 收益 | 夏普 | 索提诺 | 最大回撤 | 交易 | 胜率 | 盈亏比 | 观望 |",
           "|--|--|--|--|--|--|--|--|--|--|--|--|"]
    for k, r in enumerate(ranking[:20], 1):
        mdd = r.get("max_drawdown")
        mdd_s = f"{mdd * 100:.1f}%" if mdd is not None else "—"
        md.append(f"| {k} | `{r['combo']}`（{r.get('combo_label') or ''}） | {r.get('cap_pct'):g}% | "
                  f"t={r.get('threshold'):g} | {fmt_pct(r.get('total_return'))} | {r.get('sharpe') or 0:+.2f} | "
                  f"{r.get('sortino') or 0:+.2f} | {mdd_s} | {r.get('n_trades') or 0} | "
                  f"{(r.get('win_rate') or 0) * 100:.0f}% | {r.get('profit_loss_ratio') or 0:.2f} | "
                  f"{r.get('flat_share') or 0:.0%} |")
    md.append("")
    with (ROOT / "results" / f"combo_sweep_{ts}.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ── CSV ──
    import csv
    csv_cols = ["combo", "combo_label", "cap_pct", "threshold", "flat_share"] + STAT_KEYS
    with (ROOT / "results" / f"combo_sweep_{ts}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for r in rows:
            w.writerow([r.get(c) for c in csv_cols])

    print(f"\n=== 三轴联合回测完成（{len(rows)} 行）：全局最优 ===")
    if best_overall:
        print(f"  🎯 {fmt_r(best_overall)}")
    print(f"结果已写入 results/combo_sweep_{ts}.{{json,md,csv}} + combo_sweep_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())