"""
run_baseline_experiment.py — 基准数据集对比实验驱动

复现 strategies/STATUS_20250705.md 中 forex 组的「8 年训练配方」：
  数据配方：~50,000 根 H1（forex 49,998 bars / 8 年；基准集 BTCUSDT 50,000 根 / 5.7 年，24h 市场）
  训练配方：当前生产管线（train_file.py 冠军闸门 + 双统计校正 + holdout 单次消费），
            默认 REWARD_MODE=ftmo，步数取有界预算（--steps，默认 150，~55s/步）
  评估配方：生产配置回测（cost_rate=0.0003，tanh 连续仓位），全量数据

输出：report JSON + 与 forex 基线（年化+2.34% / Sharpe 0.64 / MDD 7.39% / 训练分 0.4851）的对比表。

用法：
    python scripts/run_baseline_experiment.py --data-file data/baseline/BTCUSDT_H1.parquet
        [--steps 150] [--seed 42] [--report data/baseline/experiment_report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: F401  (确保全局配置加载)
from model_core.config import ModelConfig
from model_core.backtest import estimate_periods_per_year
from model_core.features import FeatureEngineer
from model_core.vm import StackVM
from backtest_viz import BacktestEngine

import train_file

COST_RATE = 0.0003  # 与 run_backtest.py 生产默认一致（佣金 0.02% + 滑点 0.01%）


def compute_annualized(pnl, ppy: int) -> float:
    """几何年化收益（pnl 为对数收益）。"""
    m = float(pnl.mean())
    return math.expm1(m * ppy)


def max_drawdown(cum) -> float:
    """基于累计收益曲线的最大回撤（正数，百分比）。"""
    running_max = np_maximum_accumulate(cum)
    dd = running_max - cum
    peak = running_max.max() if len(running_max) else 0.0
    return float(dd.max()) if len(dd) and peak > 0 else 0.0


def np_maximum_accumulate(x):
    import numpy as np
    return np.maximum.accumulate(x)


def backtest_strategy(strat: dict, data_file: str) -> dict:
    """按 run_backtest.py 生产口径对单品种跑回测，返回指标。"""
    import numpy as np

    from data_pipeline.parquet_manager import ParquetDataManager

    mgr = ParquetDataManager(data_file)
    mgr.load()
    raw = mgr.raw_dict
    T = raw["open"].shape[1]
    times_all = raw.get("time", None)
    ppy = estimate_periods_per_year(times_all) if times_all is not None else 6240

    feat = FeatureEngineer.compute_features(raw)  # [N, F, T] 因果安全
    formula = [int(t) for t in strat["formula"]]
    engine = BacktestEngine(formula=formula, cost_rate=COST_RATE, periods_per_year=ppy)
    res = engine.run(raw, feat, [strat.get("symbol") or "SYM"])[0]

    pnl = np.asarray(res.pnl, dtype=np.float64)
    cum = np.asarray(res.cum_pnl, dtype=np.float64)
    sharpe = float(pnl.mean() / pnl.std(ddof=0) * math.sqrt(ppy)) if pnl.std(ddof=0) > 1e-12 else 0.0
    return {
        "T": int(T),
        "periods_per_year": int(ppy),
        "years": T / ppy if ppy else 0.0,
        "total_return": float(res.total_return),
        "annualized_pct": compute_annualized(pnl, ppy) * 100.0,
        "sharpe": sharpe,
        "sortino": float(res.sortino),
        "max_drawdown_pct": max_drawdown(cum) * 100.0,
        "n_trades": int(res.n_trades),
        "win_rate": float(res.win_rate),
        "cost_rate": COST_RATE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default="data/baseline/experiment_report.json")
    args = ap.parse_args()

    ModelConfig.TRAIN_STEPS = args.steps
    ModelConfig.REWARD_MODE = "ftmo"  # 与 train_file.py CLI 配方一致

    info = train_file.inspect_parquet_file(args.data_file)
    symbol = info["symbol"]
    print(f"\n{'='*66}\n  基准对比实验 — {symbol} {info['timeframe']}（有界复现 forex 8 年配方）\n{'='*66}")
    print(f"  数据: {args.data_file} ({info['bars']} bars)")
    print(f"  预算: {args.steps} 步 × batch {ModelConfig.BATCH_SIZE}   seed={args.seed}")
    print(f"  评估: 生产口径 cost_rate={COST_RATE}\n")

    t0 = time.time()
    engine = train_file.train_from_file(args.data_file, from_scratch=True, seed=args.seed)
    train_elapsed = time.time() - t0

    report = {
        "experiment": "STATUS_20250705 forex 8y 配方有界复现",
        "symbol": symbol,
        "data_file": str(Path(args.data_file).resolve()),
        "bars": info["bars"],
        "steps": args.steps,
        "seed": args.seed,
        "train_elapsed_s": round(train_elapsed, 1),
        "fingerprint": (engine.data_manager.fingerprint
                          if engine and getattr(engine, "data_manager", None) else None),
        "champion_outcome": getattr(engine, "champion_outcome", None) if engine else None,
        "holdout": getattr(engine, "holdout", None) if engine else None,
        "best_score": float(engine.best_score) if engine and engine.best_formula else None,
        "formula": engine._decode_formula(engine.best_formula) if engine and engine.best_formula else None,
    }

    strat_path = Path("strategies") / f"best_{symbol}.json"
    if strat_path.exists():
        strat = json.loads(strat_path.read_text(encoding="utf-8"))
        if strat.get("formula"):
            # 优先用策略自带 data_file；旧冠军 JSON 可能没有该字段时回退到实验数据文件
            bf = strat.get("data_file") or str(Path(args.data_file).resolve())
            report["backtest"] = backtest_strategy(strat, bf)
        else:
            report["backtest"] = None
    else:
        report["backtest"] = None

    # ── 对比表 ─────────────────────────────────────────────────────────
    ref = {
        "group": "forex",
        "bars": 49998,
        "years": 8,
        "score": 0.4851,
        "annualized_pct": 2.34,
        "sharpe": 0.64,
        "mdd_pct": 7.39,
    }
    bt = report.get("backtest") or {}
    print(f"\n{'='*66}\n  对比结果  {symbol}（本次） vs forex（STATUS 基线）\n{'='*66}")
    print(f"  {'指标':<16s} {'本次':>12s} {'forex 基线':>12s}")
    print(f"  {'─'*44}")
    rows = [
        ("K线数", f"{bt.get('T', '—')}", f"{ref['bars']}"),
        ("年数", f"{bt.get('years', 0):.2f}", f"{ref['years']}"),
        ("训练分", f"{report.get('best_score', 0):.4f}", f"{ref['score']}"),
        ("年化", f"{bt.get('annualized_pct', 0):+.2f}%", f"+{ref['annualized_pct']:.2f}%"),
        ("Sharpe", f"{bt.get('sharpe', 0):+.2f}", f"+{ref['sharpe']:.2f}"),
        ("MDD", f"{bt.get('max_drawdown_pct', 0):.2f}%", f"{ref['mdd_pct']:.2f}%"),
        ("Sortino", f"{bt.get('sortino', 0):+.2f}", "—"),
        ("交易数", f"{bt.get('n_trades', '—')}", "—"),
        ("胜率", f"{bt.get('win_rate', 0)*100:.1f}%", "—"),
    ]
    for name, cur, base in rows:
        print(f"  {name:<16s} {cur:>12s} {base:>12s}")
    print(f"{'='*66}")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n实验报告已保存: {out}")


if __name__ == "__main__":
    main()