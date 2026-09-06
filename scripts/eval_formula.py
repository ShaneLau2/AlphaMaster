"""
scripts/eval_formula.py — 用与 live 训练完全相同的口径评估单条公式

口径 = AlphaEngine._eval_formula_task（REWARD_MODE=ftmo）：
  - 尾部 holdout 预留（_holdout_bars_for），其余 T 根构建 4 折滚动前推（gap=WF_GAP）
  - 每折 evaluate_fold 多目标 + OOS Sortino 门控，再按 IC 门控调整
  - reward = mean(REWARD_ALPHA * train_adj) - 重复惩罚；val = mean(val_adj) - 重复惩罚
  - 相关惩罚：因子池为空 → 0（live 训练中因子池非空时另有惩罚，见输出说明）
另附生产口径回测（cost_rate=0.0003，tanh 连续仓位，全量数据）。

用法：
    .venv/bin/python scripts/eval_formula.py \
        --data-file data/baseline/BTCUSDT_H1.parquet \
        --names "SAR_DIST TS_MAX_20 VWAP_DEV MAX TS_MAX_20 ICHIMOKU_TENKAN_DEV MAX TS_MAX_20"
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import Config  # noqa: F401  (确保全局配置加载)
from model_core.backtest import ContinuousBacktest, estimate_periods_per_year
from model_core.config import ModelConfig
from model_core.engine import (
    AlphaEngine,
    _build_walk_forward_folds,
    _holdout_bars_for,
    _repetition_penalty,
)
from model_core.features import FeatureEngineer
from model_core.vm import StackVM, validate_formula_structure
from model_core.vocab import FORMULA_VOCAB
from data_pipeline.parquet_manager import ParquetDataManager
from strategy_manager.signal import compute_target_positions_stateless

COST_RATE = 0.0003  # 与 run_backtest.py 生产默认一致


def compute_annualized(pnl, ppy: int) -> float:
    m = float(pnl.mean())
    return math.expm1(m * ppy)


def max_drawdown(cum) -> float:
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    return float(dd.max()) if len(dd) else 0.0


def production_backtest(tokens: list[int], data_file: str) -> dict:
    """与 run_baseline_experiment.backtest_strategy 同口径。"""
    from backtest_viz import BacktestEngine

    mgr = ParquetDataManager(data_file)
    mgr.load()
    raw = mgr.raw_dict
    times_all = raw.get("time", None)
    ppy = estimate_periods_per_year(times_all) if times_all is not None else 6240
    feat = FeatureEngineer.compute_features(raw)  # [N, F, T] 因果安全
    engine = BacktestEngine(formula=tokens, cost_rate=COST_RATE, periods_per_year=ppy)
    res = engine.run(raw, feat, ["SYM"])[0]
    pnl = np.asarray(res.pnl, dtype=np.float64)
    cum = np.asarray(res.cum_pnl, dtype=np.float64)
    sharpe = float(pnl.mean() / pnl.std(ddof=0) * math.sqrt(ppy)) if pnl.std(ddof=0) > 1e-12 else 0.0
    return {
        "T": int(raw["open"].shape[1]),
        "periods_per_year": int(ppy),
        "years": raw["open"].shape[1] / ppy if ppy else 0.0,
        "total_return": float(res.total_return),
        "annualized_pct": compute_annualized(pnl, ppy) * 100.0,
        "sharpe": sharpe,
        "sortino": float(res.sortino),
        "max_drawdown_pct": max_drawdown(cum) * 100.0,
        "n_trades": int(res.n_trades),
        "win_rate": float(res.win_rate),
        "cost_rate": COST_RATE,
    }


def walk_forward_eval(tokens: list[int], data_file: str) -> dict:
    """复刻 AlphaEngine._eval_formula_task 的 walk-forward 评分。"""
    mgr = ParquetDataManager(data_file)
    mgr.load()
    raw = mgr.raw_dict
    feat = FeatureEngineer.compute_features(raw)  # [N, F, T]
    t_ret = mgr.target_ret
    device = ModelConfig.DEVICE
    feat = feat.to(device)
    t_ret = t_ret.to(device)

    vm = StackVM()
    res = vm.execute(tokens, feat)
    if res is None:
        return {"valid": False, "reason": "VM 执行失败（栈结构非法或未知 token）"}
    if res.std() < 1e-4:
        return {"valid": False, "reason": "输出为常数（const），无法评分"}

    bt = ContinuousBacktest()  # cost_rate 读 Config.COST_RATE
    T_full = t_ret.shape[1]
    holdout = _holdout_bars_for(T_full)
    T = T_full - holdout
    folds = _build_walk_forward_folds(T, 5, gap=getattr(ModelConfig, "WF_GAP", 20))

    fold_tr, fold_vl, fold_ic = [], [], []
    for fold in folds:
        tr_sc, vl_sc = bt.evaluate_fold(
            res, t_ret,
            fold["train_start"], fold["train_end"],
            fold["val_start"], fold["val_end"],
        )
        ic_m, _ = AlphaEngine._compute_ic(
            res[:, fold["train_start"]:fold["train_end"]],
            t_ret[:, fold["train_start"]:fold["train_end"]],
        )
        tr_adj = AlphaEngine._apply_ic_gate(ModelConfig.REWARD_ALPHA * tr_sc, ic_m)
        ic_v, _ = AlphaEngine._compute_ic(
            res[:, fold["val_start"]:fold["val_end"]],
            t_ret[:, fold["val_start"]:fold["val_end"]],
        )
        vl_adj = AlphaEngine._apply_ic_gate(vl_sc, ic_v)
        fold_tr.append(tr_adj)
        fold_vl.append(vl_adj)
        fold_ic.append(ic_m.item())

    train_score = torch.stack(fold_tr).mean()
    val_score = torch.stack(fold_vl).mean()
    ic_full, _ = AlphaEngine._compute_ic(res, t_ret)

    reward = train_score
    val_score_out = val_score
    rp = _repetition_penalty(tokens)
    if rp > 0:
        reward = reward - rp
        val_score_out = val_score_out - rp

    exposure = compute_target_positions_stateless(res).abs().mean()

    return {
        "valid": True,
        "n_folds": len(folds),
        "T_full": int(T_full),
        "holdout_bars": int(holdout),
        "reward": float(reward),
        "val_score": float(val_score_out),
        "ic_full": float(ic_full),
        "ic_folds": [round(float(x), 4) for x in fold_ic],
        "repetition_penalty": float(rp),
        "exposure": float(exposure),
        "overfit_ratio": float(val_score_out / reward) if float(reward) > 0 else None,
        "const_output": bool(res.std() < 1e-4),
        "folds": [
            {"train": [f["train_start"], f["train_end"]], "val": [f["val_start"], f["val_end"]]}
            for f in folds
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--names", required=True, help="空格分隔的 token 名（feature/operator）")
    ap.add_argument("--label", default="", help="报告标签")
    args = ap.parse_args()

    names = args.names.strip().split()
    vocab = FORMULA_VOCAB.token_names
    name2id = {n: i for i, n in enumerate(vocab)}
    unknown = [n for n in names if n not in name2id]
    if unknown:
        print(f"[错误] 以下 token 不在词表中: {unknown}")
        print(f"       词表共 {len(vocab)} 个 token。相近可用项: " + ", ".join(
            n for n in vocab if any(u[:4].lower() in n.lower() for u in unknown)
        ))
        sys.exit(2)

    tokens = [name2id[n] for n in names]
    print("=" * 72)
    print(f"  单公式评估  {'[' + args.label + ']' if args.label else ''}")
    print(f"  公式: {' -> '.join(names)}")
    print(f"  tokens: {tokens}")
    print("=" * 72)

    viol = validate_formula_structure(tokens, vocab)
    if viol:
        print(f"  [结构告警] 感染模型违规 {len(viol)} 条:")
        for v in viol:
            print(f"    - {v}")
    else:
        print("  [结构] 通过 validate_formula_structure（无恒正感染违规）")

    ev = walk_forward_eval(tokens, args.data_file)
    if not ev["valid"]:
        print(f"  [无效] {ev['reason']}")
        sys.exit(1)

    print("\n  ── Walk-Forward 评分（与 live 训练同口径，REWARD_MODE=ftmo）──")
    print(f"  数据: T={ev['T_full']}（holdout 预留 {ev['holdout_bars']} 根），"
          f"{ev['n_folds']} 折滚动前推")
    for i, (f, icf) in enumerate(zip(ev["folds"], ev["ic_folds"])):
        print(f"    第{i+1}折: 训练{f['train']} 验证{f['val']}  IC={icf:.4f}")
    print(f"  奖励(train)   = {ev['reward']:+.4f}   （含重复惩罚 {ev['repetition_penalty']:.2f}）")
    print(f"  验证(val)     = {ev['val_score']:+.4f}")
    print(f"  全样本 IC     = {ev['ic_full']:+.4f}")
    print(f"  暴露度        = {ev['exposure']:.1%}  （稀疏门槛 5%）")
    ratio_txt = f"{ev['overfit_ratio']:.2f}" if ev['overfit_ratio'] is not None else "—(奖励≤0)"
    print(f"  过拟合比值    = {ratio_txt}  （跳过门槛: 验证 < 0.5×训练）")

    print("\n  ── 生产口径回测（全量数据，cost_rate=0.0003）──")
    pb = production_backtest(tokens, args.data_file)
    print(f"  T={pb['T']} ({pb['years']:.2f} 年, ppy={pb['periods_per_year']})")
    print(f"  总收益   = {pb['total_return']*100:+.2f}%")
    print(f"  年化     = {pb['annualized_pct']:+.2f}%")
    print(f"  Sharpe   = {pb['sharpe']:+.2f}")
    print(f"  Sortino  = {pb['sortino']:+.2f}")
    print(f"  MDD      = {pb['max_drawdown_pct']:.2f}%")
    print(f"  交易数   = {pb['n_trades']}  胜率 = {pb['win_rate']*100:.1f}%")

    print("\n  ── 对照（live 运行，BTCUSDT H1 同数据）──")
    print(f"  当前冠军(step45) = 1.4696  老冠军 = 0.8208  批均值 ≈ 0.3~0.5")
    verdict = []
    if ev["val_score"] <= 0:
        verdict.append("验证分 ≤ 0：不可用")
    if ev["overfit_ratio"] is not None and ev["overfit_ratio"] < 0.5:
        verdict.append("过拟合比值 < 0.5：live 引擎会跳过该公式")
    if ev["exposure"] < 0.05:
        verdict.append("暴露度 < 5%：live 引擎会跳过（仓位过稀疏）")
    if not verdict:
        verdict.append("通过 live 引擎的全部准入闸门（过拟合/稀疏）")
    print("  准入判定: " + "; ".join(verdict))
    print("=" * 72)


if __name__ == "__main__":
    main()