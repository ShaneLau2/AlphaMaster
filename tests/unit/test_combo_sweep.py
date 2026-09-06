"""三轴联合回测（持仓方案 × 上限% × 无信号阈值）：档位归一化 + 选优逻辑 单测。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("combo_sweep_mod", _ROOT / "scripts" / "combo_sweep.py")
_cs = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cs)

# 进度统计的日志行格式与 combo_sweep.py 的 print 保持一致（组合名 24 宽右对齐）
# 模块内含 @dataclass，exec 前必须先注册进 sys.modules（否则 dataclass 装饰器查不到模块）
import importlib.util as _ilu
import sys as _sys
_mgr_spec = _ilu.spec_from_file_location("combo_sweep_mgr", _ROOT / "web" / "combo_sweep_manager.py")
_mgr = _ilu.module_from_spec(_mgr_spec)
assert _mgr_spec.loader is not None
_sys.modules["combo_sweep_mgr"] = _mgr
_mgr_spec.loader.exec_module(_mgr)


def _row(combo="dd", cap=25.0, t=0.3, ret=0.05, sharpe=2.0, n_trades=50, **kw):
    r = {"combo": combo, "combo_label": combo, "cap_pct": cap, "threshold": t,
         "total_return": ret, "sharpe": sharpe, "max_drawdown": -0.02,
         "profit_loss_ratio": 1.2, "n_trades": n_trades, "flat_share": 0.3}
    r.update(kw)
    return r


# ── 上限% 档位归一化 ────────────────────────────────────────────
def test_normalize_caps_dedupes_sorts_and_clamps() -> None:
    # 300→200（上限钳制）；0.5→1.0（下限钳制，与 _normalize_ab_caps 同规则）
    assert _cs.normalize_caps([100.0, 5.0, 5.0, 300.0, 0.5, 25.0]) == [1.0, 5.0, 25.0, 100.0, 200.0]


def test_normalize_caps_fallback_on_empty_or_invalid() -> None:
    assert _cs.normalize_caps(None) == [100.0]
    assert _cs.normalize_caps([]) == [100.0]
    assert _cs.normalize_caps([0.0, -3.0]) == [100.0]
    assert _cs.normalize_caps([], default=[10.0, 25.0, 100.0]) == [10.0, 25.0, 100.0]


# ── 无信号阈值 档位归一化 ───────────────────────────────────────
def test_normalize_thresholds_filters_to_open_interval() -> None:
    assert _cs.normalize_thresholds([0.8, 0.05, 0.3, 0.3, 1.0, -0.1]) == [0.05, 0.3, 0.8]


def test_normalize_thresholds_fallback() -> None:
    assert _cs.normalize_thresholds(None) == [0.05, 0.3, 0.5, 0.8]
    assert _cs.normalize_thresholds([1.5, 0.0]) == [0.05, 0.3, 0.5, 0.8]


# ── 选优（全局 / 每档阈值 / 每档上限 共用 best_row）──────────────
def test_best_row_picks_highest_sharpe() -> None:
    rows = [_row(combo="a", sharpe=1.0), _row(combo="b", sharpe=3.0), _row(combo="c", sharpe=2.0)]
    assert _cs.best_row(rows)["combo"] == "b"


def test_best_row_skips_no_trade_and_no_sharpe() -> None:
    rows = [
        _row(combo="a", sharpe=9.0, n_trades=0),          # 无交易 → 跳过
        _row(combo="b", sharpe=None, n_trades=10),        # 无夏普 → 跳过
        _row(combo="c", sharpe=2.5, n_trades=5),
    ]
    assert _cs.best_row(rows)["combo"] == "c"
    assert _cs.best_row([_row(n_trades=0), _row(sharpe=None)]) is None


def test_best_row_tie_break_by_more_trades() -> None:
    rows = [_row(combo="a", sharpe=2.0, n_trades=3), _row(combo="b", sharpe=2.0, n_trades=9)]
    assert _cs.best_row(rows)["combo"] == "b"


def test_best_row_with_where_filter() -> None:
    rows = [_row(combo="a", t=0.05, sharpe=1.0), _row(combo="b", t=0.3, sharpe=3.0)]
    assert _cs.best_row(rows, lambda r: r["threshold"] == 0.05)["combo"] == "a"
    assert _cs.best_row(rows, lambda r: r["threshold"] == 0.5) is None


# ── 组合清单 ────────────────────────────────────────────────────
def test_combo_order_has_22_unique_combos_including_baseline() -> None:
    order = _cs.combo_order()
    assert len(order) == 22
    assert len(set(order)) == 22
    assert order[0] == "signal"                       # 基线在第一位
    assert "chandelier+dd" in order                   # 正交组合在内（combo_id 按注册表顺序归一）
    assert "be+dd" in order
    assert all("+" not in pid or pid.count("+") == 1 for pid in order)


def test_row_brief_keeps_selection_keys() -> None:
    b = _cs.row_brief(_row(combo="dd+be"))
    assert set(b) == {"combo", "combo_label", "cap_pct", "threshold", "flat_share",
                      "total_return", "sharpe", "sortino", "max_drawdown", "n_trades",
                      "win_rate", "profit_loss_ratio", "avg_hold_bars"}
    assert b["combo"] == "dd+be"
    assert "max_single_loss" not in b                 # 精选快照不带单笔明细


# ── 进度统计日志行 ─────────────────────────────────────────────────
def test_progress_line_regex_matches_padded_combo_rows() -> None:
    # combo_sweep.py 的实际输出：组合名 24 宽右对齐，pid 与 cap= 间 1+ 空格
    line = "  chandelier+dd            cap=100% t=0.3 收益   -3.97%  夏普 -4.80  交易   50"
    assert _mgr._COMBO_LINE_RE.match(line) is not None
    assert _mgr._COMBO_LINE_RE.match("  signal                   cap=50% t=0.3 收益   -0.98%") is not None


def test_progress_line_regex_ignores_header_and_summary() -> None:
    assert _mgr._COMBO_LINE_RE.match("[轴] 持仓方案 7×7 → 22 个组合 · 上限%") is None
    assert _mgr._COMBO_LINE_RE.match("加载数据 data/slices/BTCUSDT_M5.parquet …") is None
    assert _mgr._COMBO_LINE_RE.match("=== 三轴联合回测完成（44 行）：全局最优 ===") is None
    assert _mgr._COMBO_LINE_RE.match("结果已写入 results/combo_sweep_latest.json") is None


# ── 任意切片按需资金曲线（combo_sweep_curves.slice_curve） ────────
def _fake_grid(**kw) -> dict:
    g = {
        "baseline_slice": {"cap_pct": 10.0, "threshold": 0.3},
        "caps": [10.0, 100.0],
        "thresholds": [0.05, 0.3],
        "strategy_file": "strategies/best_BTCUSDT.json",
        "data_file": "data/slices/d.parquet",
        "window_bars": 5000,
        "window_mode": "tail",
        "commission_pct": 0.02,
        "slippage_pct": 0.01,
        "generated_at": "2026-09-06T00:00:00+00:00",
    }
    g.update(kw)
    return g


def test_slice_curve_none_when_grid_missing(monkeypatch) -> None:
    import web.combo_sweep_curves as csc

    monkeypatch.setattr(csc, "_read_grid", lambda: None)
    assert csc.slice_curve("signal", 10.0, 0.3) is None
    assert csc.slice_curve("signal") is None


def test_slice_curve_baseline_delegates_to_sidecar(monkeypatch) -> None:
    import web.combo_sweep_curves as csc

    monkeypatch.setattr(csc, "_read_grid", lambda: _fake_grid())
    assert csc.slice_curve("signal", 10.0, 0.3) is None    # 命中基线切片 → 侧车
    assert csc.slice_curve("signal", None, None) is None   # 未指定切片 → 侧车


def test_slice_curve_invalid_axis_reports_error(monkeypatch) -> None:
    import web.combo_sweep_curves as csc

    monkeypatch.setattr(csc, "_read_grid", lambda: _fake_grid())
    out = csc.slice_curve("signal", 77.0, 0.3)
    assert out is not None and out.get("available") is False
    assert "不在" in out.get("error", "")
    out2 = csc.slice_curve("signal", 10.0, 0.99)
    assert out2.get("available") is False


def test_slice_curve_replays_non_baseline_cell(monkeypatch, tmp_path) -> None:
    import numpy as np
    import web.combo_sweep_curves as csc
    import web.hold_matrix_curves as hmc
    import web.paper_replay as pr

    fake = tmp_path / "d.parquet"
    fake.write_text("x")
    monkeypatch.setattr(csc, "_read_grid",
                        lambda: _fake_grid(data_file=str(fake)))
    monkeypatch.setattr(csc, "_abs", lambda p: p)
    called: dict = {}

    def fake_load(strategy_file, data_file, window_bars, window_mode):
        called["load"] = (strategy_file, data_file, window_bars, window_mode)
        return {"factor": np.zeros(1000), "open": np.ones(1000), "high": np.ones(1000),
                "low": np.ones(1000), "close": np.ones(1000), "time": None}

    monkeypatch.setattr(csc, "_load_window", fake_load)

    def fake_replay(**kw):
        called["replay"] = kw
        return {"equity": np.ones(1000), "pnl": np.zeros(1000),
                "trades": [{"bar": 900, "hold_bars": 5, "side": "long",
                             "pnl": 0.01, "label": "tp"}],
                "dd_events": [{"bar": 100, "state": 1, "dd_pct": -0.05}]}

    monkeypatch.setattr(pr, "run_replay", fake_replay)

    def fake_build(cid, eq, pn, meta, ann):
        called["build"] = (cid, meta["cap_pct"], meta["threshold"],
                            len(ann["trades"]), len(ann["dd_events"]))
        return {"available": True, "combo": cid, "equity": [1.0]}

    monkeypatch.setattr(hmc, "build_curve_response", fake_build)
    out = csc.slice_curve("be", 100.0, 0.05)
    assert out == {"available": True, "combo": "be", "equity": [1.0]}
    assert called["load"] == ("strategies/best_BTCUSDT.json", str(fake), 5000, "tail")
    assert called["replay"]["policy_id"] == "be"
    assert called["replay"]["max_position_pct"] == 100.0
    assert called["replay"]["threshold"] == 0.05
    assert called["replay"]["track_dd"] is True
    assert called["build"] == ("be", 100.0, 0.05, 1, 1)  # 交易/DD 标注进入装配


def test_slice_curve_rejects_missing_data_file(monkeypatch) -> None:
    import web.combo_sweep_curves as csc

    monkeypatch.setattr(csc, "_read_grid", lambda: _fake_grid(data_file="nope.parquet"))
    out = csc.slice_curve("be", 100.0, 0.05)
    assert out is not None and out.get("available") is False
    assert "数据文件缺失" in out.get("error", "")


# ── 曲线装配（侧车/按需共用 build_curve_response） ───────────────
def test_build_curve_response_shape_and_annotations() -> None:
    import numpy as np
    from web.hold_matrix_curves import build_curve_response

    n = 3000
    eq = np.ones(n)
    pn = np.zeros(n)
    for k in range(1, 21):  # 20 笔阶梯盈利，模拟活跃段
        s = 100 * k
        eq[s:] *= 1.001
        pn[s] = 0.001
    ann = {"trades": [{"eb": 100, "xb": 150, "side": "long", "pnl": 0.01, "reason": "tp"}],
           "dd_events": [{"bar": 500, "state": 1, "dd_pct": -0.03},
                          {"bar": 900, "state": 0, "dd_pct": None}]}
    meta = {"generated_at": "2026-09-06T00:00:00+00:00", "window_bars": 3000,
            "window_start": 100, "bars": 3000, "ppy": 105195.0}
    out = build_curve_response("signal", eq, pn, meta, ann)
    assert out["available"] is True and out["combo"] == "signal"
    assert len(out["labels"]) == len(out["equity"]) == len(out["rolling_sharpe"])
    assert 0 < len(out["labels"]) <= 900
    assert out["bars"] > 0
    assert len(out["trades"]) == 1 and out["trades"][0]["reason"] == "tp"
    # DD 区间换算：事件 bar 是窗口相对索引 → 绝对 bar（labels 同基准）
    assert len(out["dd_intervals"]) == 1
    assert out["dd_intervals"][0]["start_abs"] == 600   # window_start 100 + 窗口内 500
    assert out["dd_intervals"][0]["end_abs"] == 999    # window_start 100 + 窗口内 899
    assert out["quintiles"]["best_segment"] is not None
    assert out["meta"]["window_start"] == 100
    # 空 annotation 不炸
    out2 = build_curve_response("signal", np.ones(500), np.zeros(500), meta, {})
    assert out2["available"] is True and out2["trades"] == [] and out2["dd_intervals"] == []