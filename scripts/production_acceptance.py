"""BTC H1 生产验收（Frozen Candidate → Production Baseline）。

固定策略文件与冻结配置（signal 基线 · 每笔投入上限 50% · 无信号阈值 0.30 ·
佣金 0.02% + 滑点 0.01% = 每边总成本 0.03%），全部复用 web.paper_replay.run_replay
（与模拟实盘同口径的离散撮合引擎），跑四项验收：

  ① Frozen OOS  —— 冻结点后的样本外代理。诚实口径：本模型 train_range.mode=full
                     （训练覆盖全量 79195 根），不存在真正的训练后 OOS；取尾部 20%
                     作「冻结后泛化」的最严格可用代理 + 训练 holdout（末 500 根）
                     参考值，两者都标注溯源状态（web.oos_provenance）。
  ② 成本压力    —— 固定模型不重训：每边总成本 0.03 / 0.05 / 0.10 / 0.20%
                     （滑点恒 0.01%，差额进佣金），看夏普是否仍健康。
  ③ 阈值稳定性  —— t ∈ {0.20, 0.30, 0.40, 0.50}，只测不优化：
                     0.30 附近一小片都健康（而非单点尖峰）才算通过。
  ④ 时间稳定性  —— 2017-2020 / 2021-2022 / 2023-2024 / 2025-2026 分段：
                     不靠单一牛市，多数年代段都为正。

先做「可复现性门禁」：全历史按冻结配置重放，须与冻结基线（results/
combo_sweep_latest.json signal@50%×0.30 行）一致（夏普 ±0.05、交易数 ±10），
否则直接失败退出。输出：
  results/production_acceptance_{ts}.{json,md} + production_acceptance_latest.*
  results/frozen_baseline_btc_h1.json —— 四项全过才写 status="frozen"，
  否则 status="blocked" + 未过项与依据。

用法：
  .venv/bin/python scripts/production_acceptance.py \
      [--strategy-file strategies/best_BTCUSDT_H1.json] \
      [--data-file data/training/BTCUSDT_H1.parquet]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_core.backtest import estimate_periods_per_year  # noqa: E402
from scripts.hold_matrix import load_factor  # noqa: E402
from web.paper_replay import WARMUP_BARS, run_replay  # noqa: E402

# ── 冻结配置（与用户验收口径一致：signal · 上限50% · t=0.30 · 每边总成本0.03%）──
FROZEN_POLICY = "signal"
FROZEN_CAP = 50.0
FROZEN_THRESHOLD = 0.30
BASE_COMM = 0.02   # 每边佣金 %（+滑点 0.01% = 总成本 0.03%）
BASE_SLIP = 0.01

# 可复现性门禁容差
REPRO_SHARPE_TOL = 0.05
REPRO_TRADES_TOL = 10

# ② 成本压力：每边总成本档（滑点恒 0.01%，差额进佣金）
FEE_TOTAL_LEVELS = [0.03, 0.05, 0.10, 0.20]
# ③ 阈值稳定性：只测不优化
THRESHOLD_LEVELS = [0.20, 0.30, 0.40, 0.50]
# ④ 时间稳定性：年代段 [start, end]（含端点，UTC）
SEGMENTS = [
    ("2017-01-01", "2020-12-31", "2017–2020"),
    ("2021-01-01", "2022-12-31", "2021–2022"),
    ("2023-01-01", "2024-12-31", "2023–2024"),
    ("2025-01-01", "2026-12-31", "2025–2026"),
]
# ① Frozen OOS：尾部 20% 代理 + 训练 holdout 参考
OOS_TAIL_FRAC = 0.20

# 验收通过线（保守、可解释；写进报告便于复核）
VERDICT = {
    "oos":      {"sharpe_min": 0.8, "sortino_min": 1.0, "expectancy_gt": 0.0},
    "cost":     {"sharpe_min_at_max_fee": 0.5, "return_gt_at_max_fee": 0.0},
    "threshold": {"min_sharpe_ratio_vs_frozen": 0.7, "healthy_min": 0.5,
                  "healthy_count_min": 3},
    "time":     {"segments_positive_min": 3, "segments_sharpe_floor": -0.5},
}


def trade_metrics(closed: list[dict]) -> dict:
    """交易级指标：avg pnl（占 1.0 名义）、期望值 = 胜率×均盈 − 败率×均亏。"""
    if not closed:
        return {"avg_pnl_per_trade": None, "expectancy": None,
                "avg_win": None, "avg_loss": None}
    pnls = np.asarray([float(t["pnl"]) for t in closed])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    wr = float((pnls > 0).mean())
    return {
        "avg_pnl_per_trade": round(float(pnls.mean()), 6),
        "expectancy": round(wr * avg_win - (1.0 - wr) * avg_loss, 6),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
    }


def segment_indices(ts_sec: np.ndarray, start_iso: str, end_iso: str) -> tuple[int, int]:
    """时间段 [start, end] → (起 bar, 止 bar]（含端点；epoch 秒时间列）。"""
    s = int(datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc).timestamp())
    e = int(datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc).timestamp())
    lo = int(np.searchsorted(ts_sec, s, side="left"))
    hi = int(np.searchsorted(ts_sec, e, side="right"))  # 含 end
    return lo, hi


def confidence_label(ts_sec: np.ndarray) -> tuple[str, float]:
    """Gate 2 · 数据可信度标签（不改 Gate 1 判定，仅加档位标注）。

    History < 3y → provisional；3–5y → medium confidence；>5y → full confidence。
    全历史跨度不足的品种即使 4/4 通过也只升 provisional，等数据累积后再升级。
    """
    years = (float(ts_sec[-1]) - float(ts_sec[0])) / (365.25 * 86400.0)
    if years < 3:
        return "provisional", round(years, 2)
    if years <= 5:
        return "medium", round(years, 2)
    return "full", round(years, 2)


def build_segments(ts_sec: np.ndarray) -> list[tuple[str, str, str]]:
    """年代段规划：固定四段有数据就沿用；数据覆盖不足 3 段（短历史品种，
    如 ETH 仅 2.3 年）时改按可用数据三等分，保证时间稳定性测试仍可比。"""
    fixed = [s for s in SEGMENTS
             if segment_indices(ts_sec, s[0], s[1])[1] > segment_indices(ts_sec, s[0], s[1])[0]]
    if len(fixed) >= 3:
        return fixed
    n = int(ts_sec.size)
    out = []
    for i in range(3):
        lo, hi = n * i // 3, n * (i + 1) // 3
        # 带时分秒往返，避免日期截断把分段边界挪走最多 ~1 根 bar
        s = datetime.fromtimestamp(int(ts_sec[lo]), tz=timezone.utc).isoformat()
        e = datetime.fromtimestamp(int(ts_sec[hi - 1]), tz=timezone.utc).isoformat()
        out.append((s, e, f"第{i + 1}段 · {s[:10]} → {e[:10]}"))
    return out


def run_one(d: dict, *, start_idx: int, commission_pct: float, slippage_pct: float,
            threshold: float, cap: float, ppy: float) -> dict:
    rep = run_replay(
        factor=d["factor"], open_p=d["open"], high_p=d["high"],
        low_p=d["low"], close_p=d["close"],
        commission_pct=commission_pct, slippage_pct=slippage_pct,
        policy_id=FROZEN_POLICY, max_position_pct=cap, threshold=threshold,
        start_idx=start_idx, track_dd=False, periods_per_year=float(ppy),
    )
    st = dict(rep["stats"])
    st.update(trade_metrics(rep["trades"]))
    return st


def slice_window(d: dict, lo: int, hi: int) -> tuple[dict, int]:
    """把 [lo, hi) 截成 (带 800 根特征 warm-up 头的子数组, 引擎 start_idx)。

    run_replay 只能控制起点不能控制终点，分段回放必须物理裁剪数组；
    特征在整段数据上先算好（因果），裁剪后引擎从子数组的 warm-up 头之后开跑。
    """
    wu = max(800, int(WARMUP_BARS))
    s = max(0, lo - wu)
    sub = {k: d[k][s:hi] for k in ("factor", "open", "high", "low", "close")}
    return sub, lo - s


def verdicts(runs: dict) -> dict[str, dict]:
    """四测试各自 verdict：passed / failed + 依据。runs 含各测试的指标 dict。"""
    v: dict[str, dict] = {}

    o = runs["oos"]
    oos_pass = (
        (o["sharpe"] or 0) >= VERDICT["oos"]["sharpe_min"]
        and (o["sortino"] or 0) >= VERDICT["oos"]["sortino_min"]
        and (o["expectancy"] or -1) > VERDICT["oos"]["expectancy_gt"]
    )
    v["oos"] = {
        "passed": bool(oos_pass),
        "evidence": f"OOS(尾部20%) Sharpe {o['sharpe']} / Sortino {o['sortino']} / "
                    f"期望值 {o['expectancy']}（门槛 {VERDICT['oos']}）",
    }

    fee_rows = runs["cost"]
    max_fee = fee_rows[-1]
    cost_pass = (
        (max_fee["sharpe"] or 0) >= VERDICT["cost"]["sharpe_min_at_max_fee"]
        and (max_fee["total_return"] or -1) > VERDICT["cost"]["return_gt_at_max_fee"]
    )
    v["cost"] = {
        "passed": bool(cost_pass),
        "evidence": f"0.20% 总成本档 Sharpe {max_fee['sharpe']} / 收益 "
                    f"{max_fee['total_return'] * 100:.1f}%（门槛 {VERDICT['cost']}）",
    }

    thr_rows = runs["threshold"]
    base_sh = thr_rows[1]["sharpe"] or 0.0  # 0.30 档 = 冻结基线
    ratios = [(t["sharpe"] or 0) / base_sh if base_sh else 0.0 for t in thr_rows]
    healthy = [(t["sharpe"] or 0) >= VERDICT["threshold"]["healthy_min"] for t in thr_rows]
    thr_pass = (
        min(ratios) >= VERDICT["threshold"]["min_sharpe_ratio_vs_frozen"]
        and sum(healthy) >= VERDICT["threshold"]["healthy_count_min"]
    )
    v["threshold"] = {
        "passed": bool(thr_pass),
        "evidence": f"各档夏普 {[t['sharpe'] for t in thr_rows]} · 相对0.30档比 "
                    f"{[round(r, 2) for r in ratios]} · ≥0.5 档数 {sum(healthy)}"
                    f"（门槛 {VERDICT['threshold']}）",
    }

    seg_rows = runs["time"]
    positives = sum(1 for t in seg_rows if (t["sharpe"] or 0) > 0)
    worst = min((t["sharpe"] or 0) for t in seg_rows)
    # 段数自适应：短历史三等分只有 3 段，门槛按实际段数取 min
    pos_min = min(VERDICT["time"]["segments_positive_min"], len(seg_rows))
    time_pass = positives >= pos_min and worst >= VERDICT["time"]["segments_sharpe_floor"]
    v["time"] = {
        "passed": bool(time_pass),
        "evidence": f"各段夏普 {[t['sharpe'] for t in seg_rows]} · 正段 {positives}/{len(seg_rows)} · "
                    f"最差段 {worst}（门槛 正段≥{pos_min} 且 无段<{VERDICT['time']['segments_sharpe_floor']}）",
    }
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC H1 生产验收：OOS / 成本 / 阈值 / 时间四测试")
    ap.add_argument("--strategy-file", default="strategies/best_BTCUSDT_H1.json")
    ap.add_argument("--data-file", default="data/training/BTCUSDT_H1.parquet")
    args = ap.parse_args()

    # ── 加载因子（全量一次，供所有测试共用） ──
    print(f"加载策略 {args.strategy_file} · 数据 {args.data_file} …", flush=True)
    d = load_factor(args.strategy_file, args.data_file, None)
    T = d["bars"]
    times = d["time"]
    ts_sec = np.asarray(times[0] if times.ndim > 1 else times, dtype=np.int64)
    ppy = estimate_periods_per_year(times) if times is not None else 105195.0
    print(f"K线 {T} 根 · {datetime.fromtimestamp(int(ts_sec[0]), tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(int(ts_sec[-1]), tz=timezone.utc).date()} · ppy={ppy:.0f}",
          flush=True)

    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_file": args.strategy_file,
        "data_file": args.data_file,
        "frozen_config": {"policy": FROZEN_POLICY, "cap_pct": FROZEN_CAP,
                          "threshold": FROZEN_THRESHOLD,
                          "commission_pct": BASE_COMM, "slippage_pct": BASE_SLIP,
                          "total_cost_per_side_pct": BASE_COMM + BASE_SLIP},
        "bars": T,
        "time_range": [datetime.fromtimestamp(int(ts_sec[0]), tz=timezone.utc).isoformat(),
                       datetime.fromtimestamp(int(ts_sec[-1]), tz=timezone.utc).isoformat()],
        "tests": {},
    }

    # ── 0) 可复现性门禁：全历史重放 ≈ 冻结基线（combo_sweep signal@50%×0.30） ──
    base = run_one(d, start_idx=WARMUP_BARS, commission_pct=BASE_COMM,
                   slippage_pct=BASE_SLIP, threshold=FROZEN_THRESHOLD, cap=FROZEN_CAP,
                   ppy=ppy)
    out["frozen_baseline_replay"] = base
    grid = None
    grid_path = ROOT / "results" / "combo_sweep_latest.json"
    try:
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    ref = None
    if grid:
        ref = next((r for r in grid.get("rows", []) if r["combo"] == "signal"
                    and abs(r["cap_pct"] - FROZEN_CAP) < 1e-9
                    and abs(r["threshold"] - FROZEN_THRESHOLD) < 1e-9), None)
    if ref is not None:
        out["frozen_baseline_ref"] = {k: ref.get(k) for k in
                                      ("total_return", "sharpe", "sortino",
                                       "max_drawdown", "n_trades", "win_rate")}
        d_sh = abs((base["sharpe"] or 0) - (ref.get("sharpe") or 0))
        d_tr = abs(base["n_trades"] - (ref.get("n_trades") or 0))
        out["reproducible"] = d_sh <= REPRO_SHARPE_TOL and d_tr <= REPRO_TRADES_TOL
        if not out["reproducible"]:
            print(f"[FAIL] 可复现性门禁未过：夏普差 {d_sh:.4f}（≤{REPRO_SHARPE_TOL}）、"
                  f"交易差 {d_tr}（≤{REPRO_TRADES_TOL}）", flush=True)
            print("冻结基线重放:", json.dumps(base, ensure_ascii=False)[:300], flush=True)
            print("网格参考行:", json.dumps(ref, ensure_ascii=False)[:300], flush=True)
            return 2
        print(f"[OK] 可复现性门禁：夏普 {base['sharpe']} ≈ {ref.get('sharpe')} · "
              f"交易 {base['n_trades']} ≈ {ref.get('n_trades')}", flush=True)
    else:
        out["reproducible"] = None  # 无网格参考 → 跳过门禁（仍输出基线重放）
        print("[warn] 无 combo_sweep_latest.json 参考行，跳过可复现性门禁", flush=True)

    # ── ① Frozen OOS（尾部 20% 代理 + 训练 holdout 参考） ──
    tail_start = T - int(T * OOS_TAIL_FRAC)
    oos = run_one(d, start_idx=max(WARMUP_BARS, tail_start), commission_pct=BASE_COMM,
                  slippage_pct=BASE_SLIP, threshold=FROZEN_THRESHOLD, cap=FROZEN_CAP,
                  ppy=ppy)
    holdout_bars = 500
    ho = run_one(d, start_idx=max(WARMUP_BARS, T - holdout_bars),
                 commission_pct=BASE_COMM, slippage_pct=BASE_SLIP,
                 threshold=FROZEN_THRESHOLD, cap=FROZEN_CAP, ppy=ppy)
    # 诚实口径：训练覆盖全量 → 代理窗口的溯源状态
    try:
        from web.oos_provenance import classify_window
        prov = classify_window(args.strategy_file, args.data_file, tail_start,
                               T - tail_start)
    except Exception:  # noqa: BLE001
        prov = None
    out["tests"]["oos"] = {
        "definition": ("训练覆盖全量数据（train_range.mode=full），无真正训练后 OOS；"
                       "以尾部 20% 作冻结后泛化代理，训练 holdout（末 500 根）仅参考"),
        "tail_frac": OOS_TAIL_FRAC,
        "tail_start": tail_start,
        "provenance": prov,
        "metrics": oos,
        "holdout_reference": {k: ho.get(k) for k in
                              ("total_return", "sharpe", "sortino", "max_drawdown",
                               "n_trades", "win_rate", "expectancy")},
    }
    print(f"[①] OOS 代理(尾部20%) 收益 {oos['total_return'] * 100:+.1f}% · 夏普 "
          f"{oos['sharpe']} · Sortino {oos['sortino']} · 回撤 {oos['max_drawdown'] * 100:.1f}% · "
          f"交易 {oos['n_trades']} · 期望值 {oos['expectancy']}", flush=True)

    # ── ② 成本压力 ──
    fee_rows = []
    for total in FEE_TOTAL_LEVELS:
        comm = max(0.0, total - BASE_SLIP)
        r = run_one(d, start_idx=WARMUP_BARS, commission_pct=comm, slippage_pct=BASE_SLIP,
                    threshold=FROZEN_THRESHOLD, cap=FROZEN_CAP, ppy=ppy)
        r["total_cost_per_side_pct"] = total
        fee_rows.append(r)
        print(f"[②] 总成本 {total:.2f}% 收益 {r['total_return'] * 100:+.1f}% · 夏普 "
              f"{r['sharpe']}", flush=True)
    out["tests"]["cost"] = {"levels_total_cost_pct": FEE_TOTAL_LEVELS,
                            "slippage_pct_fixed": BASE_SLIP, "rows": fee_rows}

    # ── ③ 阈值稳定性 ──
    thr_rows = []
    for t in THRESHOLD_LEVELS:
        r = run_one(d, start_idx=WARMUP_BARS, commission_pct=BASE_COMM,
                    slippage_pct=BASE_SLIP, threshold=t, cap=FROZEN_CAP, ppy=ppy)
        r["threshold"] = t
        thr_rows.append(r)
        print(f"[③] t={t:.2f} 收益 {r['total_return'] * 100:+.1f}% · 夏普 {r['sharpe']}",
              flush=True)
    out["tests"]["threshold"] = {"levels": THRESHOLD_LEVELS, "rows": thr_rows}

    # ── ④ 时间稳定性（段规划按数据覆盖自适应） ──
    seg_rows = []
    segments = build_segments(ts_sec)
    for start_iso, end_iso, label in segments:
        lo, hi = segment_indices(ts_sec, start_iso, end_iso)
        if hi <= lo:
            print(f"[warn] 段 {label} 无数据", flush=True)
            continue
        sub, seg_start = slice_window(d, lo, hi)
        r = run_one(sub, start_idx=seg_start, commission_pct=BASE_COMM,
                    slippage_pct=BASE_SLIP, threshold=FROZEN_THRESHOLD, cap=FROZEN_CAP,
                    ppy=ppy)
        r["segment"] = label
        r["bar_range"] = [lo, hi]
        r["date_range"] = [datetime.fromtimestamp(int(ts_sec[lo]), tz=timezone.utc).date().isoformat(),
                           datetime.fromtimestamp(int(ts_sec[hi - 1]), tz=timezone.utc).date().isoformat()]
        seg_rows.append(r)
        print(f"[④] {label} 收益 {r['total_return'] * 100:+.1f}% · 夏普 {r['sharpe']} · "
              f"回撤 {r['max_drawdown'] * 100:.1f}% · 交易 {r['n_trades']}", flush=True)
    out["tests"]["time"] = {"segments": segments, "rows": seg_rows}

    # ── 判定 + 冻结记录 ──
    runs = {"oos": oos, "cost": fee_rows, "threshold": thr_rows, "time": seg_rows}
    vd = verdicts(runs)
    out["verdicts"] = vd
    all_pass = all(v["passed"] for v in vd.values())
    out["overall_passed"] = all_pass
    # Gate 2 · 数据可信度（独立于 Gate 1 判定，只影响最终状态档位）
    conf, years = confidence_label(ts_sec)
    out["data_years"] = years
    out["confidence"] = conf
    print(f"数据可信度：{years} 年 → {conf}" + ("（<3y：4/4 通过也只升 provisional）" if conf == "provisional" else ""),
          flush=True)
    print("判定:", json.dumps({k: v["passed"] for k, v in vd.items()}, ensure_ascii=False),
          flush=True)

    # ── 落盘：验收报告 + 冻结记录 ──
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    json_out = json.dumps(out, ensure_ascii=False, indent=2, default=str)
    (results_dir / "production_acceptance_latest.json").write_text(json_out, encoding="utf-8")
    (results_dir / f"production_acceptance_{ts}.json").write_text(json_out, encoding="utf-8")

    strat = json.loads(Path(args.strategy_file).read_text(encoding="utf-8"))
    sym_slug = (strat.get("symbol") or "UNKNOWN").lower().replace("usdt", "")
    tf_slug = (strat.get("timeframe") or "x").lower()
    # Gate 1 全过才看 Gate 2：full → frozen（生产基线）；provisional/medium →
    # 只升 provisional/medium 档（候选），不直接进 Production，等数据累积后重跑升级。
    if not all_pass:
        freeze_status = "blocked"
    elif conf == "full":
        freeze_status = "frozen"
    elif conf == "medium":
        freeze_status = "medium"
    else:
        freeze_status = "provisional"
    freeze = {
        "status": freeze_status,
        "symbol": strat.get("symbol") or "BTCUSDT",
        "timeframe": strat.get("timeframe") or "H1",
        "strategy_file": args.strategy_file,
        "formula": strat.get("formula"),
        "formula_decoded": strat.get("formula_decoded"),
        "best_score": strat.get("best_score"),
        "frozen_at": out["generated_at"],
        "frozen_config": out["frozen_config"],
        "data_confidence": {"years": years, "label": conf,
                            "rule": "<3y → provisional · 3–5y → medium · >5y → full"},
        "acceptance": {
            "reproducible": out.get("reproducible"),
            "overall_passed": all_pass,
            "verdicts": {k: {"passed": v["passed"], "evidence": v["evidence"]}
                         for k, v in vd.items()},
        },
        "note": ("两层 Gate：Gate 1=四项验收全过才脱离 blocked；Gate 2=数据可信度"
                 "决定最终档位——full→frozen（生产基线）/ medium→medium / "
                 "provisional→provisional（候选，等数据累积后重跑升级）。"
                 "frozen 后不再重训；由 Paper Trading 累积真正的新样本外数据，"
                 "定期重跑 scripts/production_acceptance.py 刷新验收。"),
    }
    freeze_path = results_dir / f"frozen_baseline_{sym_slug}_{tf_slug}.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── MD 报告 ──
    def md_row(r: dict) -> str:
        return (f"| {r.get('segment') or r.get('threshold') or r.get('total_cost_per_side_pct')} "
                f"| {r['total_return'] * 100:+.2f}% | {r['sharpe']} | {r['sortino']} "
                f"| {r['max_drawdown'] * 100:.2f}% | {r['n_trades']} | "
                f"{r['win_rate'] * 100:.1f}% | {r['profit_loss_ratio'] or '—'} | "
                f"{r['expectancy'] or '—'} | {r['avg_pnl_per_trade'] or '—'} |")

    md = [
        "# BTC H1 生产验收 · Production Acceptance", "",
        f"- 冻结策略：`{args.strategy_file}` · 公式 `{strat.get('formula_decoded')}`",
        f"- 冻结配置：policy `signal` · 上限 {FROZEN_CAP:g}% · 无信号阈值 {FROZEN_THRESHOLD:g} · "
        f"每边总成本 {BASE_COMM + BASE_SLIP:.2f}%（佣金 {BASE_COMM}% + 滑点 {BASE_SLIP}%）",
        f"- 数据：`{args.data_file}` · {T} 根 · "
        f"{datetime.fromtimestamp(int(ts_sec[0]), tz=timezone.utc).date()} → "
        f"{datetime.fromtimestamp(int(ts_sec[-1]), tz=timezone.utc).date()}",
        f"- 生成：{out['generated_at']} · 引擎：`web.paper_replay.run_replay`（模拟实盘同口径）", "",
        f"## 0) 可复现性门禁", "",
        f"- 冻结基线重放：收益 {base['total_return'] * 100:+.2f}% · 夏普 {base['sharpe']} · "
        f"交易 {base['n_trades']}（参考网格行："
        f"{'夏普 ' + str(ref.get('sharpe')) + ' · 交易 ' + str(ref.get('n_trades')) if ref else '无'}）",
        f"- 复现：{'✅ 通过' if out.get('reproducible') else ('⚠️ 跳过（无参考行）' if out.get('reproducible') is None else '❌ 未过')}", "",
        f"## ① Frozen OOS", "",
        f"- 口径：训练覆盖全量数据（train_range.mode=full），无真正训练后 OOS；"
        f"取尾部 {int(OOS_TAIL_FRAC * 100)}%（bar {tail_start} → {T}）作冻结后泛化代理。"
        f"溯源：`{(prov or {}).get('status') or 'unavailable'}`",
        "| 指标 | 全历史(基线) | OOS 代理(尾部20%) |",
        "| --- | --- | --- |",
        f"| 收益 | {base['total_return'] * 100:+.2f}% | {oos['total_return'] * 100:+.2f}% |",
        f"| Sharpe | {base['sharpe']} | {oos['sharpe']} |",
        f"| Sortino | {base['sortino']} | {oos['sortino']} |",
        f"| 最大回撤 | {base['max_drawdown'] * 100:.2f}% | {oos['max_drawdown'] * 100:.2f}% |",
        f"| 交易数 | {base['n_trades']} | {oos['n_trades']} |",
        f"| 胜率 | {base['win_rate'] * 100:.1f}% | {oos['win_rate'] * 100:.1f}% |",
        f"| 期望值/笔 | {base['expectancy']} | {oos['expectancy']} |",
        f"- 训练 holdout（末 500 根，仅参考，有选择偏倚）："
        f"收益 {ho['total_return'] * 100:+.2f}% · 夏普 {ho['sharpe']} · "
        f"期望值 {ho['expectancy']}", "",
        f"## ② 成本压力（固定模型，每边总成本；滑点恒 {BASE_SLIP}%）", "",
        "| 总成本 | 收益 | Sharpe | Sortino | 回撤 | 交易 | 胜率 | 盈亏比 | 期望值/笔 | 均笔收益 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "".join(md_row(r) + "\n" for r in fee_rows), "",
        f"## ③ 阈值稳定性（只测不优化，上限 {FROZEN_CAP:g}%）", "",
        "| 阈值 | 收益 | Sharpe | Sortino | 回撤 | 交易 | 胜率 | 盈亏比 | 期望值/笔 | 均笔收益 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "".join(md_row(r) + "\n" for r in thr_rows), "",
        f"## ④ 时间稳定性（分年代段）", "",
        "| 年代 | 收益 | Sharpe | Sortino | 回撤 | 交易 | 胜率 | 盈亏比 | 期望值/笔 | 均笔收益 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "".join(md_row(r) + "\n" for r in seg_rows), "",
        "注：期望值/笔 = 胜率×均盈 − 败率×|均亏|（计数加权）；均笔收益 = 全部已平仓笔数简单平均。",
        "两者背离（期望值为正、均笔为负）说明亏损集中在少数大额单（左尾风险）。", "",
        "## 判定", "",
        "| 测试 | 结论 | 证据 |",
        "| --- | --- | --- |",
        "".join(f"| {k} | {'✅ 通过' if v['passed'] else '❌ 未过'} | {v['evidence']} |\n"
                for k, v in vd.items()),
        "",
        f"## Gate 2 · 数据可信度", "",
        f"- 数据跨度 {years} 年 → **{conf}**（规则：<3y → provisional · 3–5y → medium · >5y → full）",
        f"- 含义：即使 4/4 通过，{years} 年历史统计下的结果也不能与 9 年同级等价；"
        f"{years} 年 → 该品种最多升至 **{conf}** 档（候选），等数据累积后重跑升级。", "",
        f"**总体：{'✅ 四项全过 → ' + ('冻结基准（' + freeze_path.name + '）' if conf == 'full' else '候选档 ' + conf + '（' + freeze_path.name + '），等数据累积后重跑升级') if all_pass else '❌ 未全过 → 冻结记录 status=blocked，先解决未过项'}**", "",
        "冻结后：不再重训该品种；进入 Paper Trading 积累真正的新样本外数据，",
        "数据累积后重跑本脚本刷新验收（此时 ① 将出现真实的训练后 OOS；",
        "Gate 2 档位也会随数据跨度增加而升级）。",
    ]
    md_path = results_dir / "production_acceptance_latest.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    (results_dir / f"production_acceptance_{ts}.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"\n报告: {md_path}（{freeze_path.name} status={freeze['status']}）", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())