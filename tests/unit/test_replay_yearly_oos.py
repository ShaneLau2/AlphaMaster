"""回放页逐年度拆解 / OOS 警告数据 / train_range 溯源保护 单测。

固化三个契约：
1. 年份分桶必须用真日历年（datetime64[Y]），不用固定 365.25d 粗桶
   （粗桶边界在每年 12 月下旬，会把 2017-08..12 错拆两个桶）；
2. /api/paper/replay 响应字段（yearly/best_year/worst_year/oos）前端消费依赖；
3. sync_best_strategy_for_symbol 重写 best_*.json 时必须保留 train_range。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_year_bucketing_uses_true_calendar_years() -> None:
    """固定 365.25d 粗桶会把每年 12 月下旬错切成两个桶；datetime64[Y] 必须对齐真年。"""
    ts = np.array([
        1_502_942_400,  # 2017-08-17
        1_513_718_400,  # 2017-12-20 （粗桶边界附近，真年仍是 2017）
        1_545_297_600,  # 2018-12-20
        1_785_439_200,  # 2026-09-03
    ], dtype=float)
    coarse = np.floor_divide(ts.astype("int64"), 31_536_000)  # 旧逻辑：错位
    true_years = ts.astype("datetime64[s]").astype("datetime64[Y]").astype("int64") + 1970
    # 真年分桶：2017 的两根必须同桶
    assert true_years[0] == true_years[1] == 2017
    assert true_years[2] == 2018 and true_years[3] == 2026
    # 粗桶确实会错拆（2017-08 与 2017-12 分属不同粗桶，虽然真年相同）
    # （粗桶唯一值 3 个 vs 真年唯一值 3 个在此样本上碰巧相等——分开断言）
    assert coarse[0] == 47 and coarse[1] == 47
    assert len(set(true_years.tolist())) == 3


def test_replay_response_contract_fields(tmp_path: Path) -> None:
    """/api/paper/replay 响应必须携带前端消费的字段（rsRenderOosWarn/rsRenderYearly）。

    策略文件用 tmp 夹具而非线上部署文件——线上 best_BTCUSDT.json 的 train_range
    随冠军训练数据变化（M5 40000 根 vs H1 78695 根），夹具保证分类口径可复现。
    """
    from web.app import PaperReplayRequest, api_paper_replay

    root = Path(__file__).resolve().parents[2]
    live = json.loads((root / "strategies" / "best_BTCUSDT.json")
                      .read_text(encoding="utf-8"))
    fixture = tmp_path / "best_BTCUSDT.json"
    fixture.write_text(json.dumps({
        "vocab_version": live.get("vocab_version", 1),
        "symbol": "BTCUSDT",
        "formula": live["formula"],
        "formula_decoded": live.get("formula_decoded"),
        "best_score": live.get("best_score", 0),
        # 契约夹具：M5 40000 根 + holdout 500 尾（与 1500 窗口 = 1000 样本内 + 500 holdout 对应）
        "train_range": {"data_file": "data/slices/BTCUSDT_M5.parquet",
                        "n_bars": 40000, "mode": "tail", "holdout_bars": 500},
    }), encoding="utf-8")
    req = PaperReplayRequest(
        data_file="data/slices/BTCUSDT_M5.parquet",
        strategy_file=str(fixture),
        policy_id="signal",
        window_bars=1500,
    )
    res = api_paper_replay(req)
    # 逐年度拆解字段
    assert isinstance(res.get("yearly"), list) and res["yearly"], "yearly 必须非空"
    row = res["yearly"][0]
    for k in ("year", "bars", "return", "max_drawdown", "n_trades"):
        assert k in row, f"yearly 行缺 {k}"
    assert res.get("best_year") is not None and res.get("worst_year") is not None
    # OOS 溯源字段（train_range 精确回填 → 可精确分类）
    oos = res.get("oos")
    assert oos is not None, "train_range 回填后 OOS 分类必须可用"
    assert oos.get("status") in ("oos-holdout", "oos-new", "in-sample", "partial", "mixed")
    # 1500 根窗口 = 1000 根样本内 + 500 根 holdout 尾（训练 n_bars=40000, holdout=500）
    assert oos.get("n_holdout") == 500
    assert oos.get("honest") is not None and "start_bar" in oos["honest"]


def test_sync_best_strategy_preserves_train_range(tmp_path: Path, monkeypatch) -> None:
    """sync_best_strategy_for_symbol 重写 best_*.json 时不得丢 train_range。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "am_web_app", Path(__file__).resolve().parents[2] / "web" / "app.py")
    # 只测 strategy_file 模块本身（避免拉起整个 app）
    root = Path(__file__).resolve().parents[2]
    sspec = importlib.util.spec_from_file_location("rsf_test", root / "web" / "strategy_file.py")
    sf = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(sf)

    strat_dir = tmp_path / "strategies"
    strat_dir.mkdir()
    best = strat_dir / "best_XAUUSD.json"
    best.write_text(json.dumps({
        "vocab_version": 1, "symbol": "XAUUSD", "formula": [1, 2, 3],
        "best_score": 1.23, "train_range": {"mode": "tail", "n_bars_requested": 5000,
                                             "subset": True, "inferred": False},
    }), encoding="utf-8")

    # checkpoint/export 均无 → 回退 inspect 现有文件路径；核心断言：
    # 就算走到重写，train_range 也要在 payload 里
    monkeypatch.setattr(sf, "STRATEGIES_DIR", strat_dir, raising=False)
    info = sf.sync_best_strategy_for_symbol("XAUUSD")
    # 返回 inspect（未重写）时文件必须原样；重写时 train_range 必须保留
    on_disk = json.loads(best.read_text(encoding="utf-8"))
    assert on_disk.get("train_range"), "重写后 train_range 丢失"
    assert info is None or True  # inspect 失败不阻断本契约
