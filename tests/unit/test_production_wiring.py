"""生产接线 (Option A, 2026-09-06)：BTCUSDT 默认指向冻结的 H1 制品。

冻结裁决：production = strategies/best_BTCUSDT_H1.json（H1 冠军）；
best_BTCUSDT.json 是 M5 时代制品（保留、可手动选，不作为 symbol 默认）。
这些契约把「默认解析/展示/导出」与数据文件的 timeframe 对齐：

1. resolve_strategy_file：显式 last_strategy_file > best_{sym}_{tf}.json > best_{sym}.json
2. get_symbol_progress / get_strategy_for_export：timeframe 上下文下不再把
   M5 分数当“BTC 的策略”
3. /api/realtime/strategies：每个制品行指向它自己的文件（H1 行不得指向 M5）
4. /api/paper/replay 加载 H1 制品时按 H1 train_range 分类（无 M5 错配）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _write_strategy(dirpath: Path, name: str, *, symbol: str, timeframe: str | None,
                    score: float, formula: list[int]) -> Path:
    p = dirpath / name
    p.write_text(json.dumps({
        "vocab_version": "v_test",
        "symbol": symbol,
        "timeframe": timeframe,
        "formula": formula,
        "best_score": score,
    }), encoding="utf-8")
    return p


def _fixture_dir(tmp_path: Path) -> Path:
    d = tmp_path / "strategies"
    d.mkdir()
    _write_strategy(d, "best_BTCUSDT.json", symbol="BTCUSDT", timeframe="M5",
                    score=2.98, formula=[1, 2])
    _write_strategy(d, "best_BTCUSDT_H1.json", symbol="BTCUSDT", timeframe="H1",
                    score=1.78, formula=[50, 84, 41, 65])
    return d


# ── 1) 默认解析优先级：显式 > tf 匹配 > 旧式 best_{sym}.json ─────────────
def test_resolve_prefers_timeframe_matched_file(tmp_path, monkeypatch):
    import web.progress as wp
    import web.strategy_file as sf
    d = _fixture_dir(tmp_path)
    monkeypatch.setattr(wp, "STRATEGIES_DIR", d, raising=False)
    monkeypatch.setattr(sf, "STRATEGIES_DIR", d, raising=False)

    # H1 数据上下文（无显式选择）→ best_BTCUSDT_H1.json
    got = sf.resolve_strategy_file("", "BTCUSDT", "H1")
    assert got == str((d / "best_BTCUSDT_H1.json").resolve()), got
    # 无 timeframe → 旧式 M5 制品（回退语义不变）
    got = sf.resolve_strategy_file("", "BTCUSDT")
    assert got == str((d / "best_BTCUSDT.json").resolve()), got
    # tf 限定文件缺失 → 回退旧式
    got = sf.resolve_strategy_file("", "BTCUSDT", "M15")
    assert got == str((d / "best_BTCUSDT.json").resolve()), got
    # 显式已保存路径（指向 M5）优先于 tf 匹配
    got = sf.resolve_strategy_file(str(d / "best_BTCUSDT.json"), "BTCUSDT", "H1")
    assert got == str((d / "best_BTCUSDT.json").resolve()), got


# ── 2) progress / export 在 H1 上下文读 H1 制品 ─────────────────────────
def test_progress_and_export_timeframe_aware(tmp_path, monkeypatch):
    import web.progress as wp
    d = _fixture_dir(tmp_path)
    monkeypatch.setattr(wp, "STRATEGIES_DIR", d, raising=False)

    # H1 上下文：策略分数来自 H1 制品，M5 分数不得冒充“BTC 的策略”
    # （best_formula 反映的是 checkpoint 训练进度，非策略制品；策略侧看
    #  strategy_score / has_strategy / holdout）
    p = wp.get_symbol_progress("BTCUSDT", "H1")
    assert p.strategy_score == 1.78
    assert p.has_strategy is True
    # 无 timeframe：旧式回退（M5）
    assert wp.get_symbol_progress("BTCUSDT").strategy_score == 2.98
    # 导出：H1 上下文导出 H1 公式
    payload = wp.get_strategy_for_export("BTCUSDT", "H1")
    assert payload["formula"] == [50, 84, 41, 65]
    assert payload["timeframe"] == "H1"
    assert wp.get_strategy_for_export("BTCUSDT")["formula"] == [1, 2]


# ── 3) /api/realtime/strategies：行 → 自己的文件 ───────────────────────
def test_realtime_strategies_rows_point_to_own_file(tmp_path, monkeypatch):
    import web.app as app
    import web.progress as wp
    d = _fixture_dir(tmp_path)
    monkeypatch.setattr(wp, "STRATEGIES_DIR", d, raising=False)

    rows = app.api_realtime_strategies()["strategies"]
    by_tf = {r["timeframe"]: r for r in rows if r.get("symbol") == "BTCUSDT"}
    assert set(by_tf) == {"M5", "H1"}
    assert by_tf["H1"]["strategy_file"].endswith("best_BTCUSDT_H1.json")
    assert by_tf["M5"]["strategy_file"].endswith("best_BTCUSDT.json")
    # H1 行绝不指向 M5 文件（静默分叉的原始缺陷）
    assert by_tf["H1"]["strategy_file"] != by_tf["M5"]["strategy_file"]


# ── 4) _strategy_context：H1 数据文件 → 默认策略 = H1 制品 ─────────────
def test_strategy_context_h1_data_defaults_to_h1_artifact(tmp_path, monkeypatch):
    import web.app as app
    import web.progress as wp
    import web.strategy_file as sf
    d = _fixture_dir(tmp_path)
    monkeypatch.setattr(wp, "STRATEGIES_DIR", d, raising=False)
    monkeypatch.setattr(sf, "STRATEGIES_DIR", d, raising=False)
    h1_parquet = str(Path(__file__).resolve().parents[2] /
                     "data/training/BTCUSDT_H1.parquet")
    monkeypatch.setattr(app, "load_settings", lambda: {
        "last_data_file": h1_parquet, "last_strategy_file": ""})

    ctx = app._strategy_context()
    assert ctx["last_strategy_file"].endswith("best_BTCUSDT_H1.json")
    assert ctx["strategy_file"]["timeframe"] == "H1"


# ── 5) replay 加载 H1 制品：OOS 按 H1 train_range 分类 ─────────────────
def test_replay_loads_h1_artifact(tmp_path, monkeypatch):
    from web.app import PaperReplayRequest, api_paper_replay
    root = Path(__file__).resolve().parents[2]
    # 用真实部署制品（H1 冻结冠军）走真实回放路径
    sf_h1 = str(root / "strategies/best_BTCUSDT_H1.json")
    df_h1 = str(root / "data/training/BTCUSDT_H1.parquet")
    res = api_paper_replay(PaperReplayRequest(
        data_file=df_h1, strategy_file=sf_h1, policy_id="signal", window_bars=1500))
    assert res["ok"] is True
    oos = res["oos"]
    # H1 train_range 分类：n_holdout=500、honest 尾窗在 H1 数据上
    assert oos["n_holdout"] == 500 and oos["n_post_train"] == 0
    assert oos["honest"]["kind"] == "holdout-tail"
    assert str(oos["train"]["data_file"]).endswith("BTCUSDT_H1.parquet")
    assert oos["train"]["holdout_bars"] == 500