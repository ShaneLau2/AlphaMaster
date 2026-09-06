"""阈值敏感性（观望带）选档逻辑单测。

固化 /api/backtest/threshold-sweep 的：
1. 每行字段契约（前端表格列依赖 threshold/flat_share/n_trades/...）；
2. `_sweep_pick_best` 选档：夏普最高、并列取交易多者、忽略无交易/无夏普行。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web.app import ThresholdSweepRequest, _sweep_pick_best  # noqa: E402


def _row(threshold: float, sharpe: float | None, n_trades: int | None) -> dict:
    return {
        "threshold": threshold,
        "flat_share": 0.1,
        "total_return": 0.01,
        "sharpe": sharpe,
        "n_trades": n_trades,
        "max_drawdown": -0.01,
        "win_rate": 0.5,
    }


def test_pick_best_highest_sharpe() -> None:
    rows = [_row(0.05, 0.5, 30), _row(0.3, 2.4, 12), _row(0.5, -0.8, 40)]
    best, second = _sweep_pick_best(rows)
    assert best is not None and best["threshold"] == 0.3
    assert second is not None and second["threshold"] == 0.05


def test_pick_best_tie_breaks_by_more_trades() -> None:
    rows = [_row(0.05, 1.2, 30), _row(0.3, 1.2, 60), _row(0.5, 1.2, 4)]
    best, _ = _sweep_pick_best(rows)
    assert best is not None and best["threshold"] == 0.3


def test_pick_best_ignores_no_trades_and_null_sharpe() -> None:
    rows = [
        _row(0.05, None, 30),     # 无夏普 → 不算候选
        _row(0.3, 9.9, 0),        # 无交易 → 不算候选（夏普再高也不可信）
        _row(0.5, 3.3, 7),        # 唯一有效行
        _row(0.8, 1.1, 2),
    ]
    best, _ = _sweep_pick_best(rows)
    assert best is not None and best["threshold"] == 0.5


def test_pick_best_all_empty_returns_none() -> None:
    assert _sweep_pick_best([]) == (None, None)
    assert _sweep_pick_best([_row(0.05, 5.0, 0), _row(0.3, None, 8)]) == (None, None)


def test_sweep_request_model_fields() -> None:
    """回归：请求模型必须含阈值扫描所需字段（thresholds 可缺省走四档）。"""
    req = ThresholdSweepRequest(strategy_file="strategies/best_BTCUSDT.json")
    assert req.thresholds is None  # 缺省 → 后端用 0.05/0.3/0.5/0.8
    req2 = ThresholdSweepRequest(strategy_file="x", thresholds=[0.1, 0.9])
    assert req2.thresholds == [0.1, 0.9]


def test_sweep_row_contract_has_frontend_columns() -> None:
    """后端产出的每行必须带前端表格列读取的字段。"""
    row = _row(0.05, 1.2, 30) | {"max_drawdown": -0.08, "profit_loss_ratio": 1.8,
                                 "sortino": 0.9, "fees_total": 0.001, "flat_bars": 5,
                                 "signal_bars": 45}
    for k in ("threshold", "flat_share", "n_trades", "total_return", "sharpe",
              "max_drawdown", "profit_loss_ratio", "win_rate"):
        assert k in row
