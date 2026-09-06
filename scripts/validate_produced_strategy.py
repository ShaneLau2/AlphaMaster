"""
validate_produced_strategy.py — 只读验证「已产出的策略」能否通过 walk-forward 与样本外验证。

背景：run_baseline_experiment.py（seed 42, 150 步）已于 08:16 完成并写报告
（data/baseline/experiment_report_20260903_0816_completed.json），冠军闸门判定
reject_restore（fold-SE 不足）。本脚本【复现引擎评估路径】做独立复核：

  1. walk-forward：用引擎同款 _build_walk_forward_folds + evaluate_fold + IC 门控，
     逐折给出验证分（注意 gap 在 49,500 bar 下坍缩为 0，与训练日志一致）；
  2. 样本外（holdout）：尾部 500 根（成熟 498 根）生产口径回测
     （tanh 仓位 + cost 0.0003 + 换手成本），复算闸门：分>0、保持率≥0.30、Sharpe≥0；
  3. 冠军 vs 次优 跨折 SE 裕度（champ vs runner 的逐折差 > 1×SE）；
  4. 全量生产回测对比表（vs STATUS forex 基线：年化+2.34% / Sharpe 0.64 / MDD 7.39%）。

只读：不写 holdout_state.json / champion_history.json / strategies/ 任何文件。

用法：
    .venv/bin/python scripts/validate_produced_strategy.py \
        --data-file data/baseline/BTCUSDT_H1.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.backtest import ContinuousBacktest, estimate_periods_per_year
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine, _build_walk_forward_folds
from model_core.vm import StackVM
from scripts.run_baseline_experiment import backtest_strategy
from strategy_manager.signal import compute_target_positions_stateless

# STATUS forex 基线（对比表）
REF = {
    "group": "forex", "bars": 49998, "years": 8,
    "score": 0.4851, "annualized_pct": 2.34, "sharpe": 0.64, "mdd_pct": 7.39,
}

# run #1（已完成）产出的三条候选：
#  - best: 训练最优（val 1.6397，step 68）
#  - candidate: 冠军候选（holdout Sharpe 9.705，被 fold-SE 拒绝）
#  - old: 闸门拒绝后恢复的旧冠军
FORMULAS = {
    "best(训练最优 val=1.6397)":      [56, 94, 109, 125, 4, 119, 87, 65],
    "candidate(冠军候选 val=1.6129)": [33, 98, 94, 90, 7, 91, 115, 79],
    "old(恢复的旧冠军)":              [56, 94, 2, 74, 87, 125, 117, 115],
}
BEST_VAL = {"best(训练最优 val=1.6397)": 1.6397, "candidate(冠军候选 val=1.6129)": 1.6129,
            "old(恢复的旧冠军)": 0.8208}


def wf_fold_vals(fml, feat, t_ret, folds) -> tuple[list[float], float]:
    """逐折 val（IC 门控，与 _eval_formula_task 同口径），返回 (per-fold, mean)。"""
    vm, bt = StackVM(), ContinuousBacktest()
    with torch.no_grad():
        res = vm.execute(fml, feat)
    if res is None or res.std() < 1e-4:
        return [], -5.0
    vals = []
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
    return vals, sum(vals) / len(vals)


def holdout_verify(fml, feat, t_ret, ppy: int, best_val: float) -> dict:
    """复刻引擎 _verify_holdout：尾部 500 根、排除最后 2 根边界。"""
    bt = ContinuousBacktest()
    bt.periods_per_year = ppy
    h = int(ModelConfig.HOLDOUT_BARS)
    start = t_ret.shape[1] - h
    end = t_ret.shape[1] - 2
    vm = StackVM()
    with torch.no_grad():
        res = vm.execute(fml, feat)
    if res is None or res.std() < 1e-4:
        return {"error": "formula 退化（None/常数）"}

    with torch.no_grad():
        _, ho_score = bt.evaluate_fold(res, t_ret, start, end, start, end)
        ic_ho, _ = AlphaEngine._compute_ic(res[:, start:end], t_ret[:, start:end])
        ho_adj = float(AlphaEngine._apply_ic_gate(ho_score, ic_ho))

        pos = compute_target_positions_stateless(res)
        prev = torch.roll(pos, 1, dims=1)
        prev[:, 0] = 0.0
        turnover = torch.abs(pos - prev)
        pnl = pos * t_ret - turnover * bt.cost_rate
        pnl_h = pnl[:, start:end]
        total_return_pct = float(pnl_h.sum()) * 100.0
        sharpe = float(pnl_h.mean() / (pnl_h.std() + 1e-9) * math.sqrt(ppy))
        sortino = float(bt._sortino(pnl_h))

    ratio = (ho_adj / best_val) if best_val and best_val > 0 else None
    reasons = []
    if ho_adj <= float(ModelConfig.HOLDOUT_MIN_SCORE):
        reasons.append(f"holdout 分 {ho_adj:.4f} ≤ {ModelConfig.HOLDOUT_MIN_SCORE}")
    if ratio is not None and ratio < float(ModelConfig.HOLDOUT_MIN_RATIO):
        reasons.append(f"保持率 {ratio:.3f} < {ModelConfig.HOLDOUT_MIN_RATIO}")
    if sharpe < float(ModelConfig.HOLDOUT_MIN_SHARPE):
        reasons.append(f"holdout Sharpe {sharpe:.3f} < {ModelConfig.HOLDOUT_MIN_SHARPE}")
    return {
        "bars": h, "start": start, "end": end, "mature_bars": end - start,
        "val_score": round(ho_adj, 4), "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3), "total_return_pct": round(total_return_pct, 3),
        "score_ratio": round(ratio, 4) if ratio is not None else None,
        "passed": len(reasons) == 0, "gate": reasons,
    }


def fold_se_check(champ_fml, runner_fml, feat, t_ret, folds) -> dict:
    """复刻 _fold_se_check：冠军 vs 次优 逐折 val 差 > k×SE（k=1.0）。"""
    c_vals, _ = wf_fold_vals(champ_fml, feat, t_ret, folds)
    r_vals, _ = wf_fold_vals(runner_fml, feat, t_ret, folds)
    if len(c_vals) != len(r_vals) or len(c_vals) < 3:
        return {"skipped": True}
    c = torch.tensor(c_vals)
    r = torch.tensor(r_vals)
    diff = c - r
    se = diff.std(unbiased=True) / math.sqrt(max(1, diff.numel()))
    k = float(ModelConfig.CHAMPION_SE_K)
    return {
        "champ_per_fold": [round(float(x), 4) for x in c.tolist()],
        "runner_per_fold": [round(float(x), 4) for x in r.tolist()],
        "mean_diff": round(float(diff.mean()), 4),
        "se": round(float(se), 4),
        "k": k,
        "passed": float(diff.mean()) > k * float(se),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", default="data/baseline/BTCUSDT_H1.parquet")
    args = ap.parse_args()

    mgr = ParquetDataManager(args.data_file)
    mgr.load()
    T_full = mgr.target_ret.shape[1]
    times = mgr.raw_dict.get("time")
    ppy = estimate_periods_per_year(times) if times is not None else 6240
    feat = mgr.feat_tensor
    t_ret = mgr.target_ret

    # 与训练同款：holdout 预留后建 WF 折
    T_train = T_full - int(ModelConfig.HOLDOUT_BARS)
    folds = _build_walk_forward_folds(T_train, 5, gap=int(getattr(ModelConfig, "WF_GAP", 20)))
    feat_tr = feat[:, :, :T_train]
    t_ret_tr = t_ret[:, :T_train]

    print(f"数据: {mgr.symbol} {mgr.timeframe}  T={T_full}  ppy={ppy}  指纹={mgr.fingerprint}")
    print(f"WF 折: {len(folds)} 折（训练窗口 {T_train} bar，gap={folds[0]['gap']}）\n")

    results = {}
    for label, fml in FORMULAS.items():
        print(f"{'='*70}\n[{label}]  {' -> '.join(map(str, fml))}\n{'='*70}")
        try:
            _, wf_mean = wf_fold_vals(fml, feat_tr, t_ret_tr, folds)
        except Exception as e:  # noqa: BLE001
            print(f"  WF 评估失败: {e}")
            wf_mean = float("nan")
        ho = holdout_verify(fml, feat, t_ret, ppy, BEST_VAL[label])
        print(f"  walk-forward 平均验证分 (IC 门控): {wf_mean:.4f}")
        print(f"  样本外(尾部{ho.get('bars')}根/成熟{ho.get('mature_bars')}根): "
              f"分={ho.get('val_score')} Sharpe={ho.get('sharpe')} "
              f"Sortino={ho.get('sortino')} 收益={ho.get('total_return_pct')}% "
              f"保持率={ho.get('score_ratio')} → "
              f"{'✅ 通过' if ho.get('passed') else '❌ 未过'}")
        for r in ho.get("gate", []):
            print(f"      - {r}")
        results[label] = {"wf_mean": wf_mean, "holdout": ho}

    # 冠军 vs 次优 跨折 SE（引擎拒绝的原因复核）
    print(f"\n{'='*70}\n跨折 SE 复核: candidate(冠军候选) vs best(训练最优)\n{'='*70}")
    se_r = fold_se_check(FORMULAS["candidate(冠军候选 val=1.6129)"],
                         FORMULAS["best(训练最优 val=1.6397)"],
                         feat_tr, t_ret_tr, folds)
    if se_r.get("skipped"):
        print("  折数不足，跳过")
    else:
        print(f"  candidate 逐折: {se_r['champ_per_fold']}")
        print(f"  best      逐折: {se_r['runner_per_fold']}")
        print(f"  mean_diff={se_r['mean_diff']}  SE={se_r['se']}  k={se_r['k']}  "
              f"→ {'✅ 裕度足够' if se_r['passed'] else '❌ 裕度不足（单折碰运气风险，引擎判定 reject 的原因）'}")

    # 全量生产回测对比表
    print(f"\n{'='*70}\n全量生产回测（cost={backtest_strategy.__globals__['COST_RATE']}，tanh 仓位） vs forex 基线\n{'='*70}")
    print(f"{'指标':<14s} {'本次 best':>12s} {'本次 candidate':>14s} {'forex 基线':>10s}")
    bt_best = backtest_strategy({"formula": FORMULAS["best(训练最优 val=1.6397)"], "symbol": "BTCUSDT"}, args.data_file)
    bt_cand = backtest_strategy({"formula": FORMULAS["candidate(冠军候选 val=1.6129)"], "symbol": "BTCUSDT"}, args.data_file)
    rows = [
        ("年化", f"{bt_best['annualized_pct']:+.2f}%", f"{bt_cand['annualized_pct']:+.2f}%", f"+{REF['annualized_pct']:.2f}%"),
        ("Sharpe", f"{bt_best['sharpe']:+.2f}", f"{bt_cand['sharpe']:+.2f}", f"+{REF['sharpe']:.2f}"),
        ("MDD", f"{bt_best['max_drawdown_pct']:.2f}%", f"{bt_cand['max_drawdown_pct']:.2f}%", f"{REF['mdd_pct']:.2f}%"),
        ("Sortino", f"{bt_best['sortino']:+.2f}", f"{bt_cand['sortino']:+.2f}", "—"),
        ("交易数", f"{bt_best['n_trades']}", f"{bt_cand['n_trades']}", "—"),
        ("胜率", f"{bt_best['win_rate']*100:.1f}%", f"{bt_cand['win_rate']*100:.1f}%", "—"),
        ("年数", f"{bt_best['years']:.2f}", f"{bt_cand['years']:.2f}", f"{REF['years']}"),
    ]
    for name, b, c, r in rows:
        print(f"  {name:<12s} {b:>12s} {c:>14s} {r:>10s}")


if __name__ == "__main__":
    main()