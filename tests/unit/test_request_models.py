"""回测启动 window_bars 回归 + Start*/Replay* 请求模型与 handler 字段一致性审计。

背景：web/app.py 曾出现 `'StartBacktestRequest' object has no attribute
'window_bars'` 500——handler 引用了模型上不存在的字段（pydantic 默认忽略
未知输入，运行期才 AttributeError）。本文件固化两类防线：
1. /api/backtest/start 的 window_bars 往返（模型字段存在 + 端到端传递）；
2. 静态扫描 web/app.py：每个带 `req: <Model>` 的 handler，其函数体引用的
   `req.<field>` 必须在模型字段里（任何新字段/改名漏改会在此处直接 FAIL）。
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── 1. window_bars 回归 ──────────────────────────────────────────────────

def _load_app_module():
    spec = importlib.util.spec_from_file_location("am_web_app", PROJECT_ROOT / "web" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_start_backtest_request_has_window_bars():
    """回归：StartBacktestRequest 必须声明 window_bars（历史 500 根因）。"""
    from web.app import StartBacktestRequest

    assert "window_bars" in StartBacktestRequest.model_fields
    req = StartBacktestRequest(strategy_file="strategies/best_BTCUSDT.json", window_bars=5000)
    assert req.window_bars == 5000
    # 缺省 → None（回测全历史）
    assert StartBacktestRequest(strategy_file="x").window_bars is None


def test_backtest_start_echoes_window(monkeypatch):
    """端到端：handler 应把 window_bars 传进 backtest_manager.start 并回显。"""
    mod = _load_app_module()
    captured = {}

    class FakeJob:
        def to_dict(self):
            return {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return FakeJob()

    monkeypatch.setattr(mod.backtest_manager, "start", fake_start)
    monkeypatch.setattr(mod, "_inspect_strategy_or_http", lambda f: {"strategy_file": f})
    monkeypatch.setattr(mod, "inspect_parquet_file", lambda p: {"data_file": str(p), "valid": True})
    monkeypatch.setattr(mod, "save_settings", lambda d: d)
    monkeypatch.setattr(mod, "load_settings", lambda: {"max_position_pct": 100.0})

    res = mod.api_backtest_start(mod.StartBacktestRequest(
        strategy_file="strategies/best_BTCUSDT.json",
        data_file="data/slices/BTCUSDT_M5.parquet",
        window_bars=12345,
    ))
    assert captured["window_bars"] == 12345
    assert res["window_bars"] == 12345
    # 统一无信号阈值：请求缺省时回退设置默认 0.05，并透传给 manager
    assert captured["signal_threshold"] == 0.05
    assert res["signal_threshold"] == 0.05
    mod.api_backtest_start(mod.StartBacktestRequest(
        strategy_file="strategies/best_BTCUSDT.json",
        data_file="data/slices/BTCUSDT_M5.parquet",
        window_bars=12345,
        signal_threshold=0.8,
    ))
    assert captured["signal_threshold"] == 0.8
    # 非法（≤0）窗口归一为 None
    mod.api_backtest_start(mod.StartBacktestRequest(
        strategy_file="strategies/best_BTCUSDT.json",
        data_file="data/slices/BTCUSDT_M5.parquet",
        window_bars=0,
    ))
    assert captured["window_bars"] is None


# ── 2. Start*/Replay* 家族字段一致性静态审计 ─────────────────────────────

def _parse_models(src: str) -> dict[str, set[str]]:
    models: dict[str, set[str]] = {}
    for m in re.finditer(r"class (\w+)\(BaseModel\):(.*?)(?=\nclass |\n@|\ndef |\Z)", src, re.S):
        models[m.group(1)] = set(re.findall(r"^ {4}(\w+)\s*:", m.group(2), re.M))
    return models


def _parse_handlers(src: str) -> list[tuple[str, str, str]]:
    """返回 [(route, model, handler-body)]，仅含带 `req: <Model>` 的 handler。"""
    out = []
    pat = re.compile(
        r"@app\.(?:post|get|put|delete)\(\"([^\"]+)\"\)\s*\n"
        r"def (\w+)\(([^)]*)\)(?:\s*->\s*[^:]+)?:(.*?)(?=\n@app\.|\Z)",
        re.S,
    )
    for m in pat.finditer(src):
        route, _fname, sig, body = m.group(1), m.group(2), m.group(3), m.group(4)
        mt = re.search(r"req:\s*(\w+)", sig)
        if mt:
            out.append((route, mt.group(1), body))
    return out


def test_start_family_models_share_cost_fields():
    """Start* / Replay* 家族的公共字段保持同名同义（窗口/成本/上限）。"""
    from web.app import (
        HoldMatrixRunRequest,
        PaperReplayRequest,
        PolicyABRequest,
        StartBacktestRequest,
    )

    for f in ("commission_pct", "slippage_pct", "window_bars", "max_position_pct",
              "signal_threshold"):
        assert f in StartBacktestRequest.model_fields, f
        assert f in HoldMatrixRunRequest.model_fields, f
        assert f in PolicyABRequest.model_fields, f
    for f in ("commission_pct", "slippage_pct", "window_bars", "max_position_pct",
              "signal_threshold"):
        assert f in PaperReplayRequest.model_fields, f
    # 回测/矩阵都带 hold_policy / window_mode+regime 家族
    assert "hold_policy" in StartBacktestRequest.model_fields
    assert {"window_mode", "regime", "chunks"} <= set(HoldMatrixRunRequest.model_fields)


def test_all_handler_req_fields_exist_on_models():
    """静态审计：任何 handler 引用的 req.<field> 必须在对应模型里（防 500 回归）。"""
    src = (PROJECT_ROOT / "web" / "app.py").read_text(encoding="utf-8")
    models = _parse_models(src)
    handlers = _parse_handlers(src)
    assert len(handlers) >= 15, "handler 解析异常（数量过少）"
    problems = []
    for route, model, body in handlers:
        fields = models.get(model)
        assert fields is not None, f"{route}: 模型 {model} 不存在"
        for f in sorted(set(re.findall(r"\breq\.(\w+)", body))):
            if f not in fields:
                problems.append(f"{route}: req.{f} 不在 {model} 字段里")
    assert not problems, "字段不一致（会 500）:\n" + "\n".join(problems)


def test_audit_catches_missing_field(tmp_path):
    """审计逻辑自检：人为构造字段缺失应被查出（防审计本身失效）。"""
    src = '''
class FakeReq(BaseModel):
    a: int | None = None

@app.post("/fake")
def fake_handler(req: FakeReq) -> dict:
    return {"v": req.missing_field}
'''
    models = _parse_models(src)
    handlers = _parse_handlers(src)
    assert handlers and models["FakeReq"] == {"a"}
    route, model, body = handlers[0]
    used = set(re.findall(r"\breq\.(\w+)", body))
    assert used == {"missing_field"}
    assert "missing_field" not in models[model]
