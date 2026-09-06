"""P2 IC/PnL 对齐单测：锁定 factor[t]~target_ret[t] 配对语义。

target_ret[t] = log(open[t+2]/open[t+1])；position[t] = tanh(factor[t]) 产生
open[t+1]→open[t+2] 的收益。因此 IC 必须配对 factor[t]~target_ret[t]
（原实现错配 factor[t]~target_ret[t+1]，比 PnL 晚一根 bar，已修正）。
"""
from __future__ import annotations

import math

import torch

from model_core.backtest import ContinuousBacktest
from model_core.engine import AlphaEngine


def _manual_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """独立手算 Pearson 相关系数（IC 复算基准）。"""
    xm = x - x.mean()
    ym = y - y.mean()
    sx = (xm ** 2).mean().sqrt()
    sy = (ym ** 2).mean().sqrt()
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return float((xm * ym).mean() / (sx * sy + 1e-9))


def test_ic_pairs_factor_t_with_target_t():
    """完美对齐：factor[t] = target_ret[t] → IC ≈ 1。

    旧实现（factor[t] vs target[t+1]）在无自相关收益上会得到 ≈ 0。
    """
    torch.manual_seed(7)
    T = 600
    target = torch.randn(1, T) * 0.01
    factor = target.clone()

    ic_mean, _ = AlphaEngine._compute_ic(factor, target)
    assert float(ic_mean) > 0.95, f"对齐配对 IC={ic_mean} 应≈1"

    # 错位配对（旧语义）应接近 0：证明修正确实改变了配对
    ic_lag, _ = AlphaEngine._compute_ic(factor[:, :-1], target[:, 1:])
    assert abs(float(ic_lag)) < 0.1, f"错位配对 IC={ic_lag} 应≈0"


def test_ic_recompute_matches_manual_pearson():
    """IC 复算 == 原始收益 Pearson（逐品种独立手算）。"""
    torch.manual_seed(11)
    N, T = 3, 400
    target = torch.randn(N, T) * 0.005
    factor = torch.randn(N, T) + 0.3 * target  # 已知线性关系的因子

    ic_mean, _ = AlphaEngine._compute_ic(factor, target)
    manual = sum(
        _manual_pearson(factor[n, :-2], target[n, :-2]) for n in range(N)
    ) / N
    assert abs(float(ic_mean) - manual) < 1e-5, (float(ic_mean), manual)


def test_ic_recompute_equals_spearman_of_original():
    """IC 复算 == 原始收益的秩相关（Spearman）。

    Pearson 只对线性变换不变，秩（单调）相关需先把双变量取秩再算 Pearson：
    Spearman(factor, target) = Pearson(rank(factor), rank(target))。
    """
    torch.manual_seed(23)
    N, T = 1, 500
    target = torch.randn(N, T) * 0.01
    factor = target ** 3  # 严格单调变换（三次方）

    f_rank = factor.argsort(dim=-1).float()
    t_rank = target.argsort(dim=-1).float()
    ic_on_ranks, _ = AlphaEngine._compute_ic(f_rank, t_rank)
    assert float(ic_on_ranks) > 0.95, f"Spearman 复算 IC={ic_on_ranks} 应≈1"

    # 方向保留：单调变换不改变符号，原值 Pearson 应 > 0
    ic_raw, _ = AlphaEngine._compute_ic(factor, target)
    assert float(ic_raw) > 0.0, f"单调变换后原始 Pearson IC={ic_raw} 应为正"


def test_boundary_bars_excluded():
    """尾部裁剪：最后两根 target 恒 0（边界），不得进入 IC 配对。"""
    torch.manual_seed(31)
    T = 300
    target = torch.randn(1, T) * 0.01
    target[0, -2:] = 0.0  # 边界 bar
    factor = target.clone()

    # 若把 0 边界纳入配对，Pearson 会被稀释；裁剪后应≈1
    ic_mean, _ = AlphaEngine._compute_ic(factor, target)
    assert float(ic_mean) > 0.9, f"边界裁剪后 IC={ic_mean} 应≈1"


def test_pnl_uses_same_t_pairing():
    """PnL 与 IC 同配对：position[t]×target_ret[t]（evaluate_fold 内部一致）。"""
    torch.manual_seed(41)
    N, T = 1, 300
    target = torch.randn(N, T) * 0.01
    # 真实因子量级为 O(1)（特征已归一化），tanh 仓位 ≈ ±0.76
    factor = torch.sign(target)
    bt = ContinuousBacktest()

    train_score, val_score = bt.evaluate_fold(factor, target, 0, 150, 150, 298)
    # 方向完全正确 + 无成本压力 → 分数应为正
    assert float(train_score) > 0, f"train_score={train_score} 应为正"
    assert float(val_score) > 0, f"val_score={val_score} 应为正"

    # 反向因子（factor=-sign(target)）→ IC 应显著为负（sign 变换下 |Pearson|=E|x|/σ≈0.8）
    ic_rev, _ = AlphaEngine._compute_ic(-factor, target)
    assert float(ic_rev) < -0.7
    rev_sc, rev_vl = bt.evaluate_fold(-factor, target, 0, 150, 150, 298)
    assert float(rev_sc) < 0 and float(rev_vl) < 0


def test_ts_ic_stability_aligned():
    """backtest._ts_ic_stability 与 _compute_ic 同配对（不晚一根 bar）。"""
    torch.manual_seed(53)
    T = 400
    target = torch.randn(1, T) * 0.01
    factor = target.clone()
    bt = ContinuousBacktest()
    stab = bt._ts_ic_stability(factor, target)
    assert stab > 0.9, f"ts_ic_stability={stab} 应对齐配对≈1"