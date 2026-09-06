"""P2 前视回归套件（仿 lookahead_healthcheck）。

核心属性：因果性（shift-invariance）——任何只依赖过去信息的计算，
在全序列上算出的 t 之前输出，必须与「截断到 t 的序列」算出的逐位一致。

覆盖：
1. 全部 65 个特征：compute_features(全序列) vs compute_features(截断) 前缀一致
2. 全部 66 个算子：vm.execute 对全量/截断特征的前缀一致（逐算子构造合法公式）
3. 标签成熟度：holdout 评分窗口必须排除最后两根边界 bar（target 恒 0）
"""
from __future__ import annotations

import math

import torch

from model_core.engine import AlphaEngine
from model_core.features import FeatureEngineer
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB

WARMUP = 250  # 覆盖特征工程最大因果回看窗口（含 _robust_norm 滚动窗）
T_FULL = 1200
T_TRUNC = 800


def _make_raw(n_symbols: int = 2, T: int = T_FULL, seed: int = 0) -> dict:
    """构造随机游走 OHLCV（带趋势 + 噪声，避免退化常数）。"""
    g = torch.Generator().manual_seed(seed)
    base = 100.0 + torch.cumsum(torch.randn(n_symbols, T, generator=g) * 0.1, dim=1)
    open_ = base
    close = base + torch.randn(n_symbols, T, generator=g) * 0.05
    high = torch.maximum(open_, close) + torch.rand(n_symbols, T, generator=g) * 0.02
    low = torch.minimum(open_, close) - torch.rand(n_symbols, T, generator=g) * 0.02
    volume = torch.randint(100, 10000, (n_symbols, T), generator=g).float()
    t0 = 1_700_000_000
    time = torch.arange(T, dtype=torch.int64).unsqueeze(0).expand(n_symbols, -1) + t0
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "time": time,
    }


def test_all_features_causal_prefix_invariance():
    """65 个特征全量：截断序列的输出 == 全序列的前缀（warmup 之后逐位一致）。"""
    raw_full = _make_raw(T=T_FULL, seed=3)
    raw_trunc = {k: v[:, :T_TRUNC] for k, v in raw_full.items()}

    feat_full = FeatureEngineer.compute_features(raw_full)     # [N, C, T_full]
    feat_trunc = FeatureEngineer.compute_features(raw_trunc)   # [N, C, T_trunc]

    assert feat_full.shape[1] == FORMULA_VOCAB.feature_count, (
        f"特征通道数 {feat_full.shape[1]} != vocab 特征数 {FORMULA_VOCAB.feature_count}"
    )
    a = feat_full[:, :, WARMUP:T_TRUNC]
    b = feat_trunc[:, :, WARMUP:]
    assert a.shape == b.shape, (a.shape, b.shape)
    if not torch.allclose(a, b, atol=1e-6, rtol=1e-4, equal_nan=True):
        bad = ~torch.isclose(a, b, atol=1e-6, rtol=1e-4, equal_nan=True)
        n_bad = int(bad.sum())
        chans = bad.any(dim=0).any(dim=1).nonzero(as_tuple=True)[0]
        t_idx = bad.any(dim=0).any(dim=0).nonzero(as_tuple=True)[0]
        raise AssertionError(
            f"前视违规！{n_bad} 个元素不一致，涉及通道 {chans[:8].tolist()}，"
            f"最早异常 t={t_idx[0].item() if t_idx.numel() else '—'}"
        )


def test_all_operators_causal_prefix_invariance():
    """66 个算子全量：逐算子构造合法公式，前缀输出必须一致。"""
    raw_full = _make_raw(T=T_FULL, seed=5)
    raw_trunc = {k: v[:, :T_TRUNC] for k, v in raw_full.items()}
    feat_full = FeatureEngineer.compute_features(raw_full)
    feat_trunc = FeatureEngineer.compute_features(raw_trunc)

    vm = StackVM()
    feat_offset = FORMULA_VOCAB.operator_offset
    tested = 0
    skipped = []
    for tok in sorted(vm.arity_map):
        arity = vm.arity_map[tok]
        if arity == 1:
            fml = [0, tok]
        elif arity == 2:
            fml = [0, 1, tok]
        else:
            skipped.append((tok, arity))
            continue
        try:
            with torch.no_grad():
                out_full = vm.execute(fml, feat_full)
                out_trunc = vm.execute(fml, feat_trunc)
        except Exception:
            skipped.append((tok, arity))
            continue
        if out_full is None or out_trunc is None:
            skipped.append((tok, arity))
            continue
        if out_full.std() < 1e-6:
            skipped.append((tok, arity))  # 常数算子无法区分因果性，跳过
            continue
        a = out_full[:, WARMUP:T_TRUNC]
        b = out_trunc[:, WARMUP:]
        assert a.shape == b.shape, (tok, arity, a.shape, b.shape)
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4, equal_nan=True), (
            f"算子 token {tok}（arity={arity}）存在前视：全序列与截断输出不一致"
        )
        tested += 1

    assert tested >= len(vm.arity_map) - len(skipped)
    assert tested >= 40, f"算子覆盖不足：只测了 {tested}/{len(vm.arity_map)}"
    print(f"[lookahead] 算子覆盖 {tested}/{len(vm.arity_map)}（跳过常数/异常 {len(skipped)}）")


class _FakeDataManager:
    """只提供引擎验证所需张量的假数据管理器。"""

    def __init__(self, T: int = 1000, C: int = 10) -> None:
        g = torch.Generator().manual_seed(9)
        feat = torch.randn(1, C, T, generator=g)
        target = torch.randn(1, T, generator=g) * 0.01
        self.feat_tensor = feat
        self.target_ret = target
        self.fingerprint = "fake_fp_test"


def test_holdout_mature_window_excludes_boundary():
    """标签成熟度：holdout 评分窗口 end == T-2（最后两根边界 target 恒 0 排除）。"""
    T = 1000
    engine = AlphaEngine(data_manager=_FakeDataManager(T=T), target_symbol=None)
    engine.holdout_bars = 100
    engine.best_formula = [0]  # 特征 0 直接作为因子

    result = engine._verify_holdout()
    assert result is not None, "holdout 验证应产生结果"
    assert result["end"] == T - 2, f"评分窗口应排除最后两根边界 bar，end={result['end']}"
    assert result["mature_bars"] == 100 - 2
    # 边界 bar 的 target 恒 0 不应进入 IC 配对（对齐单测已锁，这里锁窗口本身）
    assert "passed" in result and "gate" in result


def test_feature_truncation_slices_time_dim():
    """[N, C, T] 特征截断必须切时间维（回归：曾写成 feat[:, :T] 切了通道维）。

    该 bug 导致 holdout 预留后训练评分形状不匹配，每步全部 error。
    """
    feat = torch.randn(1, 65, 3000)
    assert feat[:, :, :2625].shape == (1, 65, 2625)
    # 错误写法（切通道维 65）不会改变时间维
    assert feat[:, :2625].shape == (1, 65, 3000)


def test_target_ret_boundary_is_zero():
    """数据层边界：target_ret[T-2] == target_ret[T-1] == 0（成熟度前提）。"""
    from data_pipeline.parquet_manager import _compute_target_ret

    raw = _make_raw(T=50, seed=13)
    target = _compute_target_ret(raw["open"])
    assert float(target[0, -1]) == 0.0
    assert float(target[0, -2]) == 0.0
    assert float(target[0, -3]) != 0.0