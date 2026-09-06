"""Tests for the P3 finalist-robustness sensitivity (model_core/robustness.py
+ engine._run_finalist_robustness)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import pytest
import torch

from model_core import robustness as rob


# ── 纯函数：production_pnl_stats ──────────────────────────────────────────────

class TestProductionPnlStats:
    def test_keys_and_finite(self):
        torch.manual_seed(0)
        factors = torch.randn(2, 120)
        t_ret = torch.randn(2, 120) * 0.001
        st = rob.production_pnl_stats(factors, t_ret, 10, 110, 0.0003, 6240)
        assert st is not None
        for key in ("sharpe", "ann_ret_pct", "total_return_pct", "turnover", "cost_rate"):
            assert key in st and math.isfinite(st[key])

    def test_empty_window_returns_none(self):
        torch.manual_seed(1)
        f = torch.randn(1, 50)
        t = torch.randn(1, 50)
        assert rob.production_pnl_stats(f, t, 40, 40, 0.0003, 6240) is None

    def test_cost_rate_recorded(self):
        f = torch.ones(1, 50)
        t = torch.zeros(1, 50)
        st = rob.production_pnl_stats(f, t, 0, 50, 0.001, 6240)
        assert st["cost_rate"] == 0.001


# ── 纯函数：build_variants / rank_index / annotate_rows / axis_verdict ─────────

class TestBuildVariants:
    def test_fold_and_start_axes(self):
        vs = rob.build_variants(T=600, n_folds=5, gap=20)
        fold = [v for v in vs if v["axis"] == "fold"]
        start = [v for v in vs if v["axis"] == "start"]
        # n_folds ∈ {4,5,6}
        assert [v["n_folds"] for v in fold] == [4, 5, 6]
        # 3 个头部丢弃变体，均保留 ≥ MIN_REMAIN_BARS 根
        assert len(start) == 3
        for v in start:
            assert 600 - v["shift"] >= rob.MIN_REMAIN_BARS
        # 变体带 label，且默认折叠参数(base)在列
        assert any(v["n_folds"] == 5 and v["shift"] == 0 for v in fold)

    def test_small_T_skips_deep_shifts(self):
        # T=240: 20% = 48 丢弃后剩 192 < 200 → 该变体被跳过
        vs = rob.build_variants(T=240, n_folds=5, gap=20)
        start = [v for v in vs if v["axis"] == "start"]
        assert len(start) == 2

    def test_fold_offset_capped_at_two(self):
        vs = rob.build_variants(T=600, n_folds=2, gap=20)
        fold = [v for v in vs if v["axis"] == "fold"]
        assert all(v["n_folds"] >= 2 for v in fold)


class TestRankAndVerdicts:
    def test_rank_index(self):
        assert rob.rank_index([1.0, 5.0, 3.0]) == 1
        assert rob.rank_index([None, -2.0]) == 1
        assert rob.rank_index([None, None]) is None

    def test_annotate_rows(self):
        rows = [
            {"label": "a", "f0": 1.0, "f1": 0.5},
            {"label": "b", "f0": None, "f1": None},
            {"label": "c", "f0": 0.3, "f1": 0.9},
        ]
        rob.annotate_rows(rows, 2)
        assert rows[0]["champ_val"] == 1.0 and rows[0]["champ_rank"] == 1
        assert rows[1]["champ_val"] is None and rows[1]["champ_rank"] is None
        assert rows[2]["champ_val"] == 0.3 and rows[2]["champ_rank"] == 2

    def test_axis_verdict_pass(self):
        rows = [{"champ_val": v, "champ_rank": r} for v, r in
                [(1.0, 1), (0.5, 1), (2.0, 2), (0.1, 1)]]  # 3/4 保持第一
        v = rob.axis_verdict(rows, label="fold")
        assert v["passed"] is True and v["axis"] == "fold"

    def test_axis_verdict_fail_on_negative_val(self):
        rows = [{"champ_val": v, "champ_rank": r} for v, r in
                [(1.0, 1), (-0.2, 1), (0.5, 1)]]
        v = rob.axis_verdict(rows, label="fold")
        assert v["passed"] is False and v["reason"] is not None

    def test_axis_verdict_fail_on_rank_flip(self):
        rows = [{"champ_val": v, "champ_rank": r} for v, r in
                [(1.0, 1), (0.3, 2), (0.4, 2)]]  # 仅 1/3 保持第一
        v = rob.axis_verdict(rows, label="start")
        assert v["passed"] is False

    def test_axis_verdict_no_valid_rows(self):
        v = rob.axis_verdict([{"champ_val": None, "champ_rank": None}], label="fold")
        assert v["passed"] is False and "无有效变体" in v["reason"]


# ── engine 集成：_run_finalist_robustness 在合成数据上可运行并产出三轴 ───────

class TestEngineRobustnessRunner:
    @pytest.fixture
    def engine(self):
        from model_core.engine import AlphaEngine
        # 轻量构建：关 LoRD（仅需 vm/bt 与评估编排）
        eng = AlphaEngine(data_manager=None, use_lord_regularization=False,
                          n_folds=5, target_symbol="ZZROBUST", seed=7)
        yield eng
        del eng

    def _synthetic(self):
        torch.manual_seed(11)
        T_full = 620
        feat = torch.randn(1, 5, T_full) * 0.5 + 1e-3  # [N, C, T]，C=5 ≥ 特征 token 0..4
        t_ret = torch.randn(1, T_full) * 0.002
        return feat, t_ret

    def test_runner_produces_three_axes(self, engine):
        feat, t_ret = self._synthetic()
        engine.holdout_bars = 20
        # holdout 窗口 [600, 618)，训练区 T = 600
        metrics = [
            {"fml": [0, 1, 65], "val": 1.5, "sharpe": 1.0, "sortino": 1.0, "ann_ret": 5.0},
            {"fml": [2, 3, 65], "val": 1.0, "sharpe": 0.5, "sortino": 0.5, "ann_ret": 2.0},
        ]
        rep = engine._run_finalist_robustness(metrics, feat, t_ret, 600, 618)
        assert rep is not None
        assert set(rep["axes"]) == {"fold", "start", "cost"}
        assert rep["verdict"] in ("robust", "brittle")
        assert rep["passed_axes"] in (0, 1, 2, 3)
        assert len(rep["fold_start_variants"]) >= 4   # fold(≥2) + start 变体
        assert rep["fold_start_variants"][0]["f0"] is not None
        assert len(rep["cost_rows"]) >= 3

    def test_runner_returns_none_on_single_finalist(self, engine):
        feat, t_ret = self._synthetic()
        engine.holdout_bars = 20
        metrics = [{"fml": [0, 1, 65], "val": 1.5, "sharpe": 1.0}]
        # 少于 2 个 finalists → 无排名可比，返回 None
        assert engine._run_finalist_robustness(metrics, feat, t_ret, 600, 618) is None

    def test_runner_tolerates_invalid_formula(self, engine):
        feat, t_ret = self._synthetic()
        engine.holdout_bars = 20
        metrics = [
            {"fml": [0, 1, 65], "val": 1.5, "sharpe": 1.0},
            {"fml": [200, 300], "val": 0.9, "sharpe": 0.4},  # 非法 token → vm 拒绝
        ]
        rep = engine._run_finalist_robustness(metrics, feat, t_ret, 600, 618)
        # 非法公式应被当作 None 处理而不抛异常；champion 行仍产出
        assert rep is not None
        assert rep["fold_start_variants"][0]["f0"] is not None
        assert rep["fold_start_variants"][0]["f1"] is None
