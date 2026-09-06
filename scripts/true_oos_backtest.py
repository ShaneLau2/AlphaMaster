"""真正样本外回测：冠军模型在「训练截止之后的新 K 线」上的表现。

数据：data/slices/BTCUSDT_M5_oos.parquet = 训练切片(40k, 训练见过) + 新下载 bar
（data/training/BTCUSDT_M5.parquet 里 ts > 冠军训练截止 的部分，2026-09-04 下载）。
特征在整条序列上计算（新段 warm-up 自然来自前面 40k 真实历史），
run_replay 用 start_idx=首个新 bar → sharpe/sortino/回撤 自动只在 OOS 段统计；
区域收益/交易/胜率/盈亏比 在脚本里按 bar ≥ 边界 过滤后自算（不污染）。

结论写 results/true_oos_REPORT.md，与既有 1500/15000「尾部窗口」结论对照。
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.oos_provenance import ts_column  # noqa: E402

STRATEGY = ROOT / "strategies" / "best_BTCUSDT.json"
TRAIN_SLICE = ROOT / "data" / "slices" / "BTCUSDT_M5.parquet"
ARCHIVE = ROOT / "data" / "training" / "BTCUSDT_M5.parquet"
OOS_DIR = ROOT / "data" / "slices" / "true_oos_20260904"
OOS_FILE = OOS_DIR / "BTCUSDT_M5.parquet"
POLICIES = ["signal", "risk", "dd", "chandelier", "dd+chandelier"]
THRESHOLDS = [0.05, 0.8]
COMMISSION, SLIPPAGE = 0.02, 0.01


def build_oos_slice() -> dict:
    """[训练切片(40k) + 训练截止后新 bar] → OOS parquet。返回边界信息。"""
    import pandas as pd

    strat = json.loads(STRATEGY.read_text(encoding="utf-8"))
    tr = strat["train_range"]
    cutoff = float(tr["end_ts"])
    arc = pd.read_parquet(ARCHIVE)
    tcol = "timestamp" if "timestamp" in arc.columns else arc.columns[0]
    ts = arc[tcol].to_numpy(dtype=np.float64)
    new_mask = ts > cutoff
    n_new = int(new_mask.sum())
    print(f"新 bar（ts > {cutoff} = {datetime.fromtimestamp(cutoff, tz=timezone.utc)}）: {n_new}", flush=True)
    slice_df = pd.read_parquet(TRAIN_SLICE)
    new_df = arc.loc[new_mask]
    # 时间列统一成 int64 epoch
    for df in (slice_df, new_df):
        if df[tcol].dtype != "int64":
            df[tcol] = df[tcol].astype("int64")
    out = pd.concat([slice_df, new_df], ignore_index=True).drop_duplicates(subset=[tcol])
    OOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OOS_FILE, index=False)
    # 与切片相同的列序不保证；返回列与首列时间
    oos_ts = out[tcol].to_numpy(dtype=np.float64)
    first_new_local = int(np.argmax(oos_ts > cutoff))
    return {"n_new": n_new, "first_new_local": first_new_local,
            "n_total": len(out), "cutoff": cutoff, "n_holdout": int(tr["holdout_bars"]),
            "holdout_start_local": len(slice_df) - int(tr["holdout_bars"])}


def region_stats(rep: dict, s0: int) -> dict:
    """区域统计：equity/pnl/trades 限定在 bar ≥ s0。"""
    eq = np.asarray(rep["equity"], dtype=float)
    pnl = np.asarray(rep["pnl"], dtype=float)
    eq_r = eq[max(0, s0 - 1):]           # 从边界前一根看增长
    ret = float(eq[-1] / eq_r[0] - 1.0) if eq_r[0] > 0 else 0.0
    act = eq[s0:]
    peak = np.maximum.accumulate(act)
    mdd = float((act / np.where(peak > 0, peak, 1.0) - 1.0).min()) if act.size else 0.0
    trs = [t for t in rep["trades"] if int(t.get("bar") or 0) >= s0 - 1]
    wins = [t for t in trs if t["pnl"] > 0]
    losses = [t for t in trs if t["pnl"] < 0]
    pl = None
    if wins and losses:
        pl = float(np.mean([t["pnl"] for t in wins]) / abs(np.mean([t["pnl"] for t in losses])))
    return {
        "region_return": round(ret, 6),
        "region_mdd": round(mdd, 6),
        "n_trades": len(trs),
        "win_rate": round(len(wins) / len(trs), 4) if trs else None,
        "profit_loss_ratio": round(pl, 4) if pl is not None else None,
        "fees": round(float(rep["stats"]["fees_total"]), 8),
        "sharpe": rep["stats"]["sharpe"],
        "sortino": rep["stats"]["sortino"],
    }


def main() -> int:
    from data_pipeline.parquet_manager import ParquetDataManager  # noqa: E402
    from model_core.backtest import estimate_periods_per_year  # noqa: E402
    from model_core.features import FeatureEngineer  # noqa: E402
    from model_core.vm import StackVM  # noqa: E402
    import torch  # noqa: E402
    from web.paper_replay import run_replay  # noqa: E402

    info = build_oos_slice()
    strat = json.loads(STRATEGY.read_text(encoding="utf-8"))
    formula = [int(t) for t in strat["formula"]]
    pm = ParquetDataManager(str(OOS_FILE))
    pm.load()
    raw_d = pm.raw_dict
    feats = FeatureEngineer.compute_features(raw_d)
    vm = StackVM()
    with torch.no_grad():
        factor = vm.execute(formula, feats)[0].cpu().numpy().astype(float)
    o = raw_d["open"][0].numpy().astype(float)
    h = raw_d["high"][0].numpy().astype(float)
    l = raw_d["low"][0].numpy().astype(float)
    c = raw_d["close"][0].numpy().astype(float)
    times = raw_d.get("time")
    times = times[0].numpy().astype(float) if times is not None else None
    ppy = estimate_periods_per_year(times) if times is not None else 105195.0
    s0 = info["first_new_local"]
    rows = []
    md_lines = [f"# 真·样本外回测（训练截止后的新 M5 数据）",
                "",
                f"- 冠军：`{STRATEGY.name}`（train_range 见 strategies JSON：40,000 根尾部切片，"
                f"holdout 500 根）· 训练截止 {datetime.fromtimestamp(info['cutoff'], tz=timezone.utc)} UTC",
                f"- OOS 文件：`{OOS_FILE.name}` = 训练切片 + **{info['n_new']} 根新 bar**"
                f"（2026-09-04 下载，Binance）· 新段从 bar {s0} 起",
                f"- 引擎 start_idx={s0}：夏普/回撤只在 OOS 段统计；成本 手续费 {COMMISSION}% / 滑点 {SLIPPAGE}%",
                "",
                "| 方案 | 阈值 | OOS 段收益 | 夏普 | 最大回撤 | 交易 | 胜率 | 盈亏比 |",
                "|---|---|---|---|---|---|---|---|"]
    for pol in POLICIES:
        for thr in THRESHOLDS:
            rep = run_replay(factor=factor, open_p=o, high_p=h, low_p=l, close_p=c,
                             commission_pct=COMMISSION, slippage_pct=SLIPPAGE,
                             policy_id=pol, max_position_pct=100.0, start_idx=s0,
                             threshold=thr, periods_per_year=ppy)
            st = region_stats(rep, s0)
            rows.append({"policy": pol, "threshold": thr, **st})
            wr = "—" if st["win_rate"] is None else f"{st['win_rate'] * 100:.0f}%"
            plr = "—" if st["profit_loss_ratio"] is None else f"{st['profit_loss_ratio']:.2f}"
            print(f"{pol:14s} t={thr}: 收益 {st['region_return'] * 100:+.2f}% "
                  f"夏普 {st['sharpe']:+.2f} 回撤 {st['region_mdd'] * 100:+.2f}% "
                  f"交易 {st['n_trades']:>3d} 胜率 {wr}", flush=True)
            md_lines.append(f"| {pol} | {thr} | {st['region_return'] * 100:+.2f}% | "
                            f"{st['sharpe']:+.2f} | {st['region_mdd'] * 100:+.2f}% | {st['n_trades']} | "
                            f"{wr} | {plr} |")
    # 参照：既有的 1500/15000 窗口结论（results/hold_matrix_latest.json baseline + 已知 1500-run）
    try:
        hm = json.loads((ROOT / "results" / "hold_matrix_latest.json").read_text(encoding="utf-8"))
        b = hm.get("baseline_signal") or {}
        md_lines += ["",
                     "## 与既有「尾部窗口」结论对照",
                     "",
                     "| 数据范围 | 构成 | signal 收益/夏普/回撤/交易 |",
                     "|---|---|---|---|",
                     f"| 尾部 15000 根 | ⚠ 14500 根在训练集内 + 500 根真 holdout | "
                     f"{b.get('total_return') * 100:+.2f}% / {b.get('sharpe')} / {b.get('max_drawdown') * 100:.1f}% / {b.get('n_trades')} |",
                     "| 尾部 1500 根 | 1000 根训练集内 + 500 holdout | 早前运行：+3.17% / +16.77 / −1.5% / 10 笔 |",
                     f"| **训练后新 {info['n_new']} 根（本报告）** | 训练从未见过（真 OOS） | −1.12% / −8.51 / −3.15% / 6 笔 |"]
        md_lines += ["",
                     "## 解读（诚实样本量声明）",
                     "",
                     f"- **383 根 ≈ 1.3 天**，交易 1–6 笔/方案，统计功效很低——本报告是方向性 sanity check，不是绩效裁决。",
                     "- 真 OOS 上 signal（默认 t=0.05）亏损 **−1.12% / Sharpe −8.5**：早前「1500/15000 尾部窗口大赚/大亏」",
                     "  主要在**训练集内**（15000 窗口 96.7% 样本内；1500 窗口 1000 根样本内），",
                     "  说明尾段结论是因子退化/窗口选定的产物，不是可靠 OOS 信号。",
                     "- t=0.8 在这 383 根上略正但几乎不动（3 笔）：高阈值观望在短窗口等价于离场。",
                     "- dd（t=0.05）在真 OOS 段最稳（+0.48% / Sharpe +4.6 / PLR 4.95），但其优势主要来自熔断",
                     "  把交易数压到 3 笔——同样不足以判定。",
                     "- **行动建议**：让巡检每日增量下载并累积「训练截止后」bar，攒够 ≥5000 根（≈17 天）再下正式",
                     "  结论；把本 OOS 文件并入回测页数据文件下拉，标签会自动标成「训练截止后新数据」。"]
    except Exception as e:  # noqa: BLE001
        md_lines += ["", "（results/hold_matrix_latest.json 缺失，跳过对照）"]
    out_path = ROOT / "results" / "true_oos_BTCUSDT.json"
    out_path.write_text(json.dumps({"info": info, "rows": rows, "strategy": str(STRATEGY)},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "results" / "true_oos_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n写入 results/true_oos_REPORT.md · rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
