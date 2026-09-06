"""web/formula_read 单测：token 解码 + 中缀表达式 + 摘要（用合成词表，不依赖 model_core）。"""
from __future__ import annotations

import json

from web.formula_read import (
    _build_expression,
    _build_summary,
    _load_context,
    interpret_formula,
    read_formula_file,
)

# 合成词表：3 特征（RET20 / ATR / RSI14，前 3 个 token）+ 算子从 token 3 起
_SYNTH = {
    "token_names": ["RET20", "ATR", "RSI14", "TS_MEAN_5", "SUB", "NEG", "DELTA", "GATE"],
    "operator_offset": 3,
    "op_arity_by_token": {3: 1, 4: 2, 5: 1, 6: 1, 7: 3},
    "feature_cat_by_name": {"RET20": "trend", "ATR": "volatility", "RSI14": "reversal"},
}


def _synth_ctx():
    return _SYNTH


def test_expression_reconstruction(monkeypatch) -> None:
    monkeypatch.setattr("web.formula_read._load_context", _synth_ctx)
    # RET20 → TS_MEAN_5(RET20)；再 SUB RET20：TS_MEAN_5(RET20) − RET20
    expr, parts = _build_expression([0, 3, 0, 4], _SYNTH)
    assert "滚动均值5(RET20) − RET20" in expr
    kinds = [p["kind"] for p in parts]
    assert kinds == ["feature", "operator", "feature", "operator"]
    assert parts[1]["note"] == "滚动均值5"


def test_gate_arity3_reconstruction(monkeypatch) -> None:
    monkeypatch.setattr("web.formula_read._load_context", _synth_ctx)
    # GATE(RET20, ATR, RSI14)：token 顺序 RET20 ATR RSI14 GATE
    expr, parts = _build_expression([0, 1, 2, 7], _SYNTH)
    assert expr.startswith("门控(cond=RET20")
    assert "x=ATR" in expr and "y=RSI14" in expr


def test_summary_counts_features_and_ops(monkeypatch) -> None:
    monkeypatch.setattr("web.formula_read._load_context", _synth_ctx)
    expr, parts = _build_expression([0, 3, 0, 4], _SYNTH)
    s = _build_summary([0, 3, 0, 4], parts)
    assert "趋势" in s
    assert "滚动均值5" in s and "−" in s
    assert "共 4 个 token" in s


def test_interpret_formula_full(monkeypatch) -> None:
    monkeypatch.setattr("web.formula_read._load_context", _synth_ctx)
    r = interpret_formula([0, 3, 0, 4])
    assert r is not None
    assert r["decoded"] == "RET20 → TS_MEAN_5 → RET20 → SUB"
    assert "− RET20" in r["expression"]
    assert r["token_count"] == 4


def test_interpret_formula_empty_or_bad_token(monkeypatch) -> None:
    monkeypatch.setattr("web.formula_read._load_context", _synth_ctx)
    assert interpret_formula([]) is None
    assert interpret_formula(None) is None
    # 越界 token 不回抛
    r = interpret_formula([0, 99, 4])
    assert r is not None


def test_read_formula_file_live_preferred(tmp_path) -> None:
    (tmp_path / "best_BTCUSDT.json").write_text(
        json.dumps({"formula": [0, 1, 2]}), encoding="utf-8"
    )
    (tmp_path / "best_BTCUSDT.live.json").write_text(
        json.dumps({"formula": [2, 2, 2]}), encoding="utf-8"
    )
    assert read_formula_file(tmp_path, "BTCUSDT") == [2, 2, 2]
    assert read_formula_file(tmp_path, "ETHUSDT") is None


def test_read_formula_file_broken(tmp_path) -> None:
    (tmp_path / "best_BTCUSDT.live.json").write_text("not json", encoding="utf-8")
    assert read_formula_file(tmp_path, "BTCUSDT") is None


def test_real_vocab_load_smoke() -> None:
    """真实 model_core 词表可用时跑一次最小解读（慢但验证接线）。"""
    ctx = _load_context()
    assert ctx["operator_offset"] > 0
    assert len(ctx["token_names"]) > ctx["operator_offset"]
