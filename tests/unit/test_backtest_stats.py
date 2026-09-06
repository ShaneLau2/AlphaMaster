"""回测报告指标（最大回撤）+ 组合持仓管理归一化 + 买入持有基准导出的单测。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from run_backtest import calc_max_drawdown, export_equity_json
from web.hold_policy import combo_id, combo_label, combo_parts


# ── 最大回撤（策略回测报告特征之一） ────────────────────────────────
class TestMaxDrawdown:
    def test_monotonic_up_is_zero(self) -> None:
        cum = np.linspace(0.0, 0.5, 100)
        assert calc_max_drawdown(cum) == 0.0

    def test_flat_is_zero(self) -> None:
        assert calc_max_drawdown(np.zeros(50)) == 0.0

    def test_peak_then_trough(self) -> None:
        # equity: 峰值 1.3 → 谷底 1.0 → -23.1%（cum = equity - 1）
        eq = np.array([1.0, 1.05, 1.3, 1.22, 1.12, 1.0, 1.02])
        assert round(calc_max_drawdown(eq - 1.0), 4) == pytest.approx(-0.2308)

    def test_recovery_does_not_hide_deeper_prior_dd(self) -> None:
        # 先深回撤再创新高：最大回撤仍取历史最深谷
        eq = np.array([1.0, 1.5, 1.1, 0.9, 1.6, 1.55])
        assert round(calc_max_drawdown(eq - 1.0), 4) == pytest.approx(-0.4)

    def test_empty_input(self) -> None:
        assert calc_max_drawdown(np.array([])) == 0.0


# ── 买入持有基准导出（export_equity_json 的 buy_hold 字段） ─────────
class TestBuyHoldExport:
    def _rm(self, buy_hold: np.ndarray | None = None) -> dict:
        rm: dict = {
            "BTCUSDT": {
                "pnl": np.array([0.01, 0.02, -0.01]),
                "cum_pnl": np.array([0.01, 0.03, 0.02]),
                "sharpe": 1.0,
                "sortino": 1.2,
                "total_return": 0.02,
                "profit_loss_ratio": 1.5,
            }
        }
        if buy_hold is not None:
            rm["BTCUSDT"]["buy_hold"] = buy_hold
        return rm

    def test_buy_hold_serialized_with_return(self, tmp_path) -> None:
        bh = np.array([0.0, 0.05, 0.1])
        p = export_equity_json(self._rm(buy_hold=bh), str(tmp_path), periods_per_year=8752)
        s = json.load(open(p))["symbols"]["BTCUSDT"]
        assert s["buy_hold"] == [0.0, 0.05, 0.1]
        assert s["buy_hold_return"] == 0.1
        # 与策略 equity 等长、同采样
        assert len(s["buy_hold"]) == len(s["equity"])

    def test_buy_hold_absent_yields_none(self, tmp_path) -> None:
        # 旧路径（results_map 无 buy_hold 键）：字段为 None，前端跳过基准线
        p = export_equity_json(self._rm(), str(tmp_path), periods_per_year=8752)
        s = json.load(open(p))["symbols"]["BTCUSDT"]
        assert s["buy_hold"] is None
        assert s["buy_hold_return"] is None

    def test_portfolio_buy_hold_is_mean_of_symbols(self, tmp_path) -> None:
        rm = self._rm(buy_hold=np.array([0.0, 0.04, 0.08]))
        rm["ETHUSDT"] = {
            "pnl": np.array([-0.01, 0.01, 0.02]),
            "cum_pnl": np.array([-0.01, 0.0, 0.02]),
            "sharpe": 0.8,
            "sortino": 0.9,
            "total_return": 0.02,
            "profit_loss_ratio": 1.4,
            "buy_hold": np.array([0.0, 0.02, 0.04]),
        }
        p = export_equity_json(rm, str(tmp_path), periods_per_year=8752)
        port = json.load(open(p))["portfolio"]
        assert port["buy_hold"] == [0.0, 0.03, 0.06]  # (0.04+0.02)/2, (0.08+0.04)/2
        assert port["buy_hold_return"] == 0.06

    def test_log_kind_buy_hold_return_is_real_expm1(self, tmp_path) -> None:
        # signal/连续口径: buy_hold 是累计对数收益 (log), 卡片展示真实收益率需 expm1。
        # 价格 1.0→1.5 (log≈0.4055): 真实收益应≈0.5, 而不是 0.4055。
        rm = self._rm(buy_hold=np.log(np.array([1.0, 1.2, 1.5])))
        rm["BTCUSDT"]["buy_hold_kind"] = "log"
        p = export_equity_json(rm, str(tmp_path), periods_per_year=8752)
        s = json.load(open(p))["symbols"]["BTCUSDT"]
        assert s["buy_hold"] == pytest.approx([0.0, 0.182322, 0.405465], abs=1e-5)  # 曲线保持 log 原生
        assert s["buy_hold_return"] == pytest.approx(0.5, abs=1e-5)  # 卡片为真实 %

    def test_missing_kind_defaults_to_simple(self, tmp_path) -> None:
        # 旧 results_map（无 buy_hold_kind）→ simple 语义，行为与历史一致
        bh = np.array([0.0, 0.1, 0.2])
        p = export_equity_json(self._rm(buy_hold=bh), str(tmp_path), periods_per_year=8752)
        s = json.load(open(p))["symbols"]["BTCUSDT"]
        assert s["buy_hold_return"] == 0.2


# ── 组合持仓管理（正交叠加，含默认 信号跟随） ────────────────────────
class TestComboNormalization:
    def test_signal_pair_resolves_to_other_module(self) -> None:
        # “信号跟随 + 回撤熔断” = 只保留 dd（信号无出场，求并集不变）
        assert combo_id("signal+dd") == "dd"
        assert combo_id("be+signal") == "be"

    def test_signal_alone_is_baseline(self) -> None:
        assert combo_id("signal") == "signal"
        assert combo_id("signal+signal") == "signal"
        assert combo_parts("signal+signal") == ["signal"]

    def test_two_module_stack(self) -> None:
        assert combo_id("chandelier+dd") == "chandelier+dd"
        assert combo_id(" dd + chandelier ") == "dd+chandelier"
        assert combo_parts("dd+be+time") == ["dd", "be", "time"]

    def test_combo_label_readable(self) -> None:
        assert "吊灯" in combo_label("dd+chandelier")
        assert "信号跟随" in combo_label("signal")
