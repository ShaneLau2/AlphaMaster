"""check_finalist_robustness.py — 生产化前对 finalists 的稳健性检查（P3）。

只对 top-K 入围公式跑（不把成本/折叠/起点自由度变成全搜索的选择依据）：
  1. cost × 2（成本压力）
  2. walk-forward 折数 ±1
  3. 9 个起点位移（起始日后移，仿 Freebuff 9 起点复核）

用法:
    python scripts/check_finalist_robustness.py \
        --finalists strategies/finalists_XAUUSD.json \
        --data-file data/training/XAUUSD_H1.parquet

输入由引擎训练结束时落盘（strategies/finalists_{symbol}.json）；
输出逐公式表格 + 冠军稳健性结论（champion 在 ≥80% 配置中保持 top-3 才算稳健）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.backtest import ContinuousBacktest
from model_core.config import ModelConfig
from model_core.engine import _build_walk_forward_folds
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB

MIN_ROBUST_FRACTION = 0.80  # 冠军需在至少 80% 配置中保持 top-3


def _rank_key(m: dict):
    return (float(m.get("sharpe", -99.0)), float(m.get("sortino", -99.0)),
            float(m.get("ann_ret", -99.0)))


def _production_backtest(fml, feat, t_ret, start, end, cost):
    """生产口径：tanh 仓位 + 换手成本（与引擎 _rigorous_holdout_pnl 一致）。"""
    vm = StackVM()
    bt = ContinuousBacktest()
    with torch.no_grad():
        res = vm.execute(fml, feat)
    if res is None or res.std() < 1e-4:
        return None
    from strategy_manager.signal import compute_target_positions_stateless
    pos = compute_target_positions_stateless(res)
    prev = torch.roll(pos, 1, dims=1)
    prev[:, 0] = 0.0
    turnover = torch.abs(pos - prev)
    pnl = pos * t_ret - turnover * cost
    w = pnl[:, start:end]
    ppy = max(1.0, float(bt.periods_per_year))
    return {
        "sharpe": float(w.mean() / (w.std() + 1e-9) * math.sqrt(ppy)),
        "sortino": float(bt._sortino(w)),
        "ann_ret": float(w.mean() * ppy) * 100.0,
        "total_return_pct": float(w.sum()) * 100.0,
    }


def _wf_val(fml, feat, t_ret, folds):
    """IC 门控的跨折 mean val（与训练 _eval_formula_task 同口径）。"""
    from model_core.engine import AlphaEngine
    vm = StackVM()
    bt = ContinuousBacktest()
    vals = []
    with torch.no_grad():
        res = vm.execute(fml, feat)
    if res is None or res.std() < 1e-4:
        return -5.0
    for fold in folds:
        with torch.no_grad():
            _, vl = bt.evaluate_fold(
                res, t_ret,
                fold["train_start"], fold["train_end"],
                fold["val_start"], fold["val_end"],
            )
            ic_v, _ = AlphaEngine._compute_ic(
                res[:, fold["val_start"]:fold["val_end"]],
                t_ret[:, fold["val_start"]:fold["val_end"]],
            )
            vals.append(float(AlphaEngine._apply_ic_gate(vl, ic_v)))
    return sum(vals) / len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalists", required=True, help="引擎落盘的 finalists JSON")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--cost-mult", type=float, default=2.0)
    ap.add_argument("--starts", type=int, default=9)
    args = ap.parse_args()

    payload = json.loads(Path(args.finalists).read_text(encoding="utf-8"))
    finalists = [f for f in payload.get("finalists", []) if f.get("fml")][: args.top]
    if not finalists:
        print("finalists JSON 为空（需先跑完一次训练）")
        sys.exit(1)
    cost_base = float(payload.get("cost_rate", ModelConfig.FINALIST_COST_RATE))
    sym = payload.get("symbol", "?")

    mgr = ParquetDataManager(args.data_file)
    mgr.load()
    T_full = mgr.target_ret.shape[1]
    holdout = int(payload.get("holdout_bars", ModelConfig.HOLDOUT_BARS))
    T = T_full - holdout
    feat = mgr.feat_tensor
    t_ret = mgr.target_ret

    # 基线：holdout 窗口生产回测（与引擎选冠军同口径）
    start, end = T_full - holdout, T_full - 2
    base_rows = []
    for f in finalists:
        m = _production_backtest(f["fml"], feat, t_ret, start, end, cost_base)
        if m:
            m["fml"] = f["fml"]
            m["val"] = f.get("val")
            base_rows.append(m)
    base_rows.sort(key=_rank_key, reverse=True)
    if not base_rows:
        print("全部 finalist 回测失败（公式过时/特征不匹配？）")
        sys.exit(1)

    champion_fml = base_rows[0]["fml"]

    # 配置矩阵：cost×2 / folds±1 / 9 起点位移
    configs: list[dict] = []
    for label, cost in (("base", cost_base), ("costx2", cost_base * args.cost_mult)):
        rows = []
        for f in finalists:
            m = _production_backtest(f["fml"], feat, t_ret, start, end, cost)
            if m:
                m["fml"] = f["fml"]
                rows.append(m)
        rows.sort(key=_rank_key, reverse=True)
        configs.append({"cfg": f"holdout-{label}", "rank": {
            tuple(f["fml"]): i + 1 for i, f in enumerate(rows)
        }})

    gap = getattr(ModelConfig, "WF_GAP", 20)
    for nf in (4, 6):
        folds = _build_walk_forward_folds(T, nf, gap=gap)
        vals = {tuple(f["fml"]): _wf_val(f["fml"], feat[:, :, :T], t_ret[:, :T], folds)
                for f in finalists}
        order = sorted(finalists, key=lambda f: vals[tuple(f["fml"])], reverse=True)
        configs.append({"cfg": f"wf-folds-{nf}", "rank": {
            tuple(f["fml"]): i + 1 for i, f in enumerate(order)
        }})

    step = max(1, T // (args.starts + 1))
    for k in range(1, args.starts + 1):
        off = k * step
        if off >= T - 500:
            break
        folds = _build_walk_forward_folds(T - off, 5, gap=gap)
        vals = {tuple(f["fml"]): _wf_val(f["fml"], feat[:, :, off:T], t_ret[:, off:T], folds)
                for f in finalists}
        order = sorted(finalists, key=lambda f: vals[tuple(f["fml"])], reverse=True)
        configs.append({"cfg": f"start-shift-{k}", "rank": {
            tuple(f["fml"]): i + 1 for i, f in enumerate(order)
        }})

    # 汇总
    print(f"\n=== finalists 稳健性报告 [{sym}]（成本 {cost_base}，{len(configs)} 个配置）===")
    print(f"{'配置':<18}{'冠军名次':>8}  冠军公式 {'<=3?':<6}")
    top3_counts = 0
    for cfg in configs:
        r = cfg["rank"].get(tuple(champion_fml), len(finalists))
        ok = r <= 3
        if ok:
            top3_counts += 1
        print(f"{cfg['cfg']:<18}{r:>8}  {('✓' if ok else '✗'):<6}")

    frac = top3_counts / len(configs)
    verdict = "稳健" if frac >= MIN_ROBUST_FRACTION else "不稳健（生产前需人工复核）"
    print(f"\n冠军保持 top-3 比例: {top3_counts}/{len(configs)} = {frac:.0%} → {verdict}")
    print(f"冠军公式: {' -> '.join(FORMULA_VOCAB.token_names[t] for t in champion_fml)}")

    print("\nTop-5 基线（holdout 生产回测）:")
    for i, row in enumerate(base_rows[:5], 1):
        print(f"  #{i} sharpe={row['sharpe']:.3f} sortino={row['sortino']:.3f} "
              f"年化={row['ann_ret']:+.1f}% val={row.get('val')}")
    sys.exit(0 if frac >= MIN_ROBUST_FRACTION else 2)


if __name__ == "__main__":
    main()