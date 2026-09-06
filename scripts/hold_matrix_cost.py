"""持仓组合成本敏感性：22 个组合 × 成本档 重跑 → 盈亏平衡点 / 钝感排名。

复用 scripts/hold_matrix.py 的因子/窗口加载与 web.paper_replay.run_replay 撮合引擎，
在同一数据、同一因子下把每个组合按多档成本（乘数 × 基准 佣金0.02%+滑点0.01%）重放，
产出：
  1. 每组合各成本档的 收益/夏普/最大回撤/盈亏比/交易数；
  2. 盈亏平衡成本乘数（收益由正转负的插值点，0~max 内线性插值，超过上界标注 >max）；
  3. 成本敏感斜率（收益对成本乘数 OLS 斜率）与「钝感排名」——斜率最接近 0（成本抬升
     时收益掉得最少）的方案最值得上实盘；
  4. 汇总写 results/hold_matrix_cost_latest.json / .md / .csv（回测页只读展示）。

用法：
  .venv/bin/python scripts/hold_matrix_cost.py \
      [--data-file data/slices/BTCUSDT_M5.parquet] \
      [--strategy-file strategies/best_BTCUSDT.json] \
      [--cost-mults 0,0.25,0.5,1,2,4] \
      [--window-bars N] [--window-mode tail|spread]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_hold_matrix_module():
    """import scripts/hold_matrix（复用其 POLICIES/load_factor/combo_id 等）。"""
    spec = importlib.util.spec_from_file_location("hm_src", ROOT / "scripts" / "hold_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hm_src"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _window_subset(hm, args) -> tuple[str, int | None]:
    """spread 模式 → 生成 regime 分层子集并返回 (子集文件, load_window=None)。"""
    if not (args.window_bars and args.window_mode == "spread"):
        return args.data_file, args.window_bars
    from data_pipeline.train_sampler import prepare_training_subset

    sub = prepare_training_subset(args.data_file, mode="spread", n_bars=args.window_bars,
                                  n_chunks=args.chunks, regime=args.regime)
    print(f"[窗口] 分层抽样 {args.regime} · {args.chunks} 块 · {args.window_bars} 根 "
          f"→ {sub['data_file']}", flush=True)
    return sub["data_file"], None


def _fmt_pct(v):
    return f"{v * 100.0:+.2f}%" if v is not None else "—"


def main() -> int:
    hm = _load_hold_matrix_module()
    ap = argparse.ArgumentParser(description="持仓组合成本敏感性（22 组合 × 成本档）")
    ap.add_argument("--data-file", default="data/slices/BTCUSDT_M5.parquet")
    ap.add_argument("--strategy-file", default="strategies/best_BTCUSDT.json")
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--slippage", type=float, default=0.01)
    ap.add_argument("--cost-mults", default="0,0.25,0.5,1,2,4",
                    help="成本档乘数（×基准），逗号分隔，必须含 0 与 1")
    ap.add_argument("--window-bars", type=int, default=None)
    ap.add_argument("--window-mode", choices=["tail", "spread"], default="tail")
    ap.add_argument("--regime", default="vol", choices=["vol", "trend", "equal"])
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--max-position-pct", type=float, default=100.0)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    args.max_position_pct = min(200.0, max(1.0, args.max_position_pct))
    mults = sorted({float(x) for x in args.cost_mults.split(",") if x.strip()})
    if not mults or 0.0 not in mults:
        print("错误: --cost-mults 必须包含 0（零成本参照）", file=sys.stderr)
        return 1
    base_comm = max(0.0, args.commission)
    base_slip = max(0.0, args.slippage)

    data_file, load_window = _window_subset(hm, args)
    print(f"加载数据 {data_file} …", flush=True)
    d = hm.load_factor(args.strategy_file, data_file, load_window)
    print(f"K线 {d['bars']} 根（window_start={d['window_start']}）", flush=True)

    from model_core.backtest import estimate_periods_per_year

    times = d["time"]
    ppy = estimate_periods_per_year(times) if times is not None else 105195.0

    # 上三角去重组合（与 hold_matrix 完全一致）
    run_order: list[str] = []
    for i, a in enumerate(hm.POLICIES):
        for b in hm.POLICIES[i:]:
            pid = hm.combo_id(f"{a}+{b}")
            if pid not in run_order:
                run_order.append(pid)

    # tier 名用乘数字符串（如 "0.0"/"1.0"）
    tier_rows: dict[str, dict[str, dict]] = {}
    stats: dict[str, dict[str, float]] = {pid: {} for pid in run_order}  # pid -> mult -> ret
    all_stats: dict[str, dict[str, dict[str, float]]] = {}
    for m in mults:
        key = f"{m:g}"
        comm = base_comm * m
        slip = base_slip * m
        print(f"\n[成本档 ×{m:g}] 佣金 {comm}% / 滑点 {slip}%", flush=True)
        per = {}
        for pid in run_order:
            rep = hm.run_replay(
                factor=d["factor"], open_p=d["open"], high_p=d["high"],
                low_p=d["low"], close_p=d["close"],
                commission_pct=comm, slippage_pct=slip,
                policy_id=pid, max_position_pct=args.max_position_pct,
                threshold=args.threshold,
                periods_per_year=float(ppy),
            )
            st = rep["stats"]
            row = {k: st.get(k) for k in hm.STAT_KEYS if k != "fees_total"}
            row["n_trades"] = len(rep["trades"])
            per[pid] = row
            stats[pid][key] = float(row.get("total_return") or 0.0)
            print(f"  {pid:24s} 收益 {_fmt_pct(row.get('total_return')):>9s} "
                  f"夏普 {row.get('sharpe') or 0:+.2f}  交易 {row['n_trades']:>4d}", flush=True)
        tier_rows[key] = per
        all_stats[key] = {pid: dict(per[pid]) for pid in run_order}

    # ── 每组合：盈亏平衡乘数 + 成本敏感斜率（收益对乘数 OLS）───────────────
    m_arr = np.asarray(mults, dtype=float)
    combos_out = []
    for pid in run_order:
        rets = np.asarray([stats[pid][f"{m:g}"] for m in mults], dtype=float)
        slope = float(np.polyfit(m_arr, rets, 1)[0]) if len(mults) >= 2 else float("nan")
        # 盈亏平衡：线性插值找 ret=0 的乘数
        be = None
        for i in range(len(mults) - 1):
            r0, r1 = rets[i], rets[i + 1]
            if r0 >= 0 >= r1:
                m0, m1 = mults[i], mults[i + 1]
                be = float(m0 + (m1 - m0) * (0 - r0) / (r1 - r0)) if r1 != r0 else float(m0)
                break
        if be is None:
            be = "∞" if rets[-1] > 0 else "<0"
        combos_out.append({
            "combo": pid,
            "label": hm.combo_label(pid),
            "return_by_mult": {f"{m:g}": float(r) for m, r in zip(mults, rets)},
            "sharpe_at_1x": all_stats.get("1", {}).get(pid, {}).get("sharpe"),
            "mdd_at_1x": all_stats.get("1", {}).get(pid, {}).get("max_drawdown"),
            "mdd_at_max": all_stats.get(f"{mults[-1]:g}", {}).get(pid, {}).get("max_drawdown"),
            "n_trades_1x": all_stats.get("1", {}).get(pid, {}).get("n_trades"),
            "breakeven_mult": be,
            "slope_per_mult": round(slope, 8),
        })

    # 钝感排名：成本乘数提高时收益掉得最少（斜率最大/最接近 0）者优先。
    # 盈亏平衡分桶：∞（任何成本档都为正）视为 +inf 排最前，<0（零成本已亏）视为 -inf 排最后。
    import math

    for c in combos_out:
        if isinstance(c["breakeven_mult"], float):
            be_val = c["breakeven_mult"]
        elif c["breakeven_mult"] == "∞":
            be_val = math.inf
        else:  # "<0"
            be_val = -math.inf
        c["_be_val"] = be_val
        c["insens_rank"] = None
    for rank, c in enumerate(
        sorted(combos_out, key=lambda x: (-x["slope_per_mult"], -x["_be_val"])), start=1
    ):
        c["insens_rank"] = rank
        c.pop("_be_val", None)

    ranking = sorted(combos_out, key=lambda x: (x["insens_rank"] is None, x["insens_rank"] or 999))
    baseline = all_stats.get("1", {}).get("signal", {})
    summary = {
        "n_combos": len(run_order),
        "tiers": mults,
        "least_sensitive_top5": [c["combo"] for c in ranking[:5]],
        "most_sensitive": ranking[-1]["combo"] if ranking else None,
        "signal_1x": {k: baseline.get(k) for k in
                      ("total_return", "sharpe", "max_drawdown", "n_trades", "profit_loss_ratio")},
    }

    # 落盘
    out = {
        "meta": {
            "data_file": str(Path(data_file).resolve()),
            "strategy_file": str(Path(args.strategy_file).resolve()),
            "window_bars": args.window_bars,
            "window_mode": args.window_mode,
            "commission_base_pct": base_comm,
            "slippage_base_pct": base_slip,
            "cost_mults": mults,
            "max_position_pct": args.max_position_pct,
            "threshold": args.threshold,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "tiers": all_stats,
        "combos": combos_out,
        "ranking": ranking,
        "summary": summary,
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (out_dir / f"hold_matrix_cost_{stamp}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "hold_matrix_cost_latest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown 表
    md = [f"# 持仓组合成本敏感性（{len(run_order)} 组合 × {len(mults)} 成本档）", "",
          f"- 数据 `{Path(data_file).name}` · 策略 `{Path(args.strategy_file).name}`",
          f"- 成本档 = 乘数 × 基准（佣金 {base_comm}% + 滑点 {base_slip}%）: "
          + ", ".join(f"×{m:g}" for m in mults),
          "",
          "| 组合 | " + " | ".join(f"收益@×{m:g}" for m in mults) + " | 盈亏平衡× | 钝感排名 |",
          "|---|---" * (len(mults) + 1) + "|---|"]
    for c in ranking:
        cells = " | ".join(_fmt_pct(c["return_by_mult"][f"{m:g}"]) for m in mults)
        be = f"{c['breakeven_mult']:g}" if isinstance(c["breakeven_mult"], float) else c["breakeven_mult"]
        rank = c["insens_rank"] if c["insens_rank"] else "—"
        md.append(f"| {c['combo']} | {cells} | {be} | {rank} |")
    md += ["", "## 钝感 Top5（成本抬升时收益掉得最少）"]
    for i, c in enumerate(ranking[:5], start=1):
        be = f"{c['breakeven_mult']:g}" if isinstance(c["breakeven_mult"], float) else c["breakeven_mult"]
        md.append(f"{i}. `{c['combo']}` — 斜率 {c['slope_per_mult']:+.2e}/× · 盈亏平衡 ×{be}")
    md += ["", "## 解读提示",
           "- 盈亏平衡 < 0：零成本下已亏损，成本再降也救不回，方案本身无 edge。",
           "- 盈亏平衡 > 最高档：成本抬到上限仍为正，方案对成本钝感，实盘最稳。",
           "- 把本文件并入回测页矩阵面板下方可展开查看；钝感排名 = 按斜率（越接近 0 越钝感）。"]
    (out_dir / "hold_matrix_cost_latest.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # CSV
    import csv
    with open(out_dir / "hold_matrix_cost_latest.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["combo", "label", *[f"ret_x{m:g}" for m in mults],
                    "breakeven_mult", "slope_per_mult", "insens_rank"])
        for c in ranking:
            be = c["breakeven_mult"] if isinstance(c["breakeven_mult"], float) else str(c["breakeven_mult"])
            w.writerow([c["combo"], c["label"],
                        *[round(c["return_by_mult"][f"{m:g}"], 6) for m in mults],
                        be, c["slope_per_mult"], c["insens_rank"] or ""])

    print(f"\n写入 results/hold_matrix_cost_latest.json/.md/.csv（{len(run_order)} 组合）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
