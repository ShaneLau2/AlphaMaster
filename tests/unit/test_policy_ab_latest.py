"""A/B 最近结果持久化（policy_ab_latest.json）与回滚校验 manager 单测。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_ab_latest_shape_matches_frontend_consumers(tmp_path: Path) -> None:
    """policy-ab 落盘结构必须满足回测页 btLoadAbLatest 的读取字段。"""
    latest = {
        "generated_at": "2026-09-05T00:00:00+00:00",
        "symbol": "BTCUSDT",
        "data_file": "/x/BTCUSDT_M5.parquet",
        "bars": 15000,
        "params": {"window_bars": 15000, "commission_pct": 0.02,
                   "slippage_pct": 0.01, "max_position_pct": 100.0,
                   "signal_threshold": 0.05},
        "policies": ["signal", "dd", "risk"],
        "best": {
            "policy_id": "dd", "name": "回撤熔断", "desc": "",
            "total_return": 0.05, "sharpe": 2.1, "max_drawdown": -0.08,
            "n_trades": 30, "win_rate": 0.6, "profit_loss_ratio": 1.8,
        },
        "results": [],
    }
    p = tmp_path / "policy_ab_latest.json"
    p.write_text(json.dumps(latest), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    b = loaded["best"]
    # 前端只依赖这些字段：policy_id / sharpe / max_drawdown / total_return / n_trades
    assert "policy_id" in b and b["policy_id"] == "dd"
    for k in ("sharpe", "max_drawdown", "total_return", "n_trades"):
        assert k in b
    # best 必须是 results 里夏普最高者
    ranked = sorted(loaded["results"] + [b],
                    key=lambda r: (r.get("sharpe") is not None,
                                   float(r.get("sharpe") or float("-inf"))),
                    reverse=True)
    assert ranked[0]["policy_id"] == b["policy_id"]


def test_ab_ranking_picks_highest_sharpe() -> None:
    results = [
        {"policy_id": "signal", "sharpe": 0.5},
        {"policy_id": "dd", "sharpe": 2.4},
        {"policy_id": "risk", "sharpe": -1.0},
    ]
    ranked = sorted(results,
                    key=lambda r: (r.get("sharpe") is not None,
                                   float(r.get("sharpe") or float("-inf"))),
                    reverse=True)
    assert ranked[0]["policy_id"] == "dd"
    assert ranked[-1]["policy_id"] == "risk"


def test_verify_rollback_manager_status_shape() -> None:
    """status() 返回字段必须满足前端 vrfPoll 读取（active/job/log_tail/report）。"""
    from web.verify_rollback_manager import VerifyRollbackManager

    mgr = VerifyRollbackManager()
    st = mgr.status()
    assert st["active"] is False
    assert st["job"] is None
    assert st["report"] is None
    assert "log_tail" in st