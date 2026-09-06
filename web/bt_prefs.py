"""web/bt_prefs.py — 回测页「持仓组合 + 样本外窗口」按品种记忆。

{品种: {hold_policy, window_bars}} 存为 results/bt_prefs.json；
同一品种切换模型/重跑回测时自动带出该品种上次的组合与窗口，
跨品种各留一份互不覆盖。全局 web_settings 的 bt_hold_policy /
bt_window_bars 仍作为「无品种记录时的兜底默认」。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFS_PATH = PROJECT_ROOT / "results" / "bt_prefs.json"

_lock = threading.Lock()

_DEFAULTS: dict[str, Any] = {
    "hold_policy": "signal",
    "window_bars": None,
}


def _clean(prefs: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(_DEFAULTS)
    if prefs:
        if prefs.get("hold_policy"):
            from web.hold_policy import combo_id

            out["hold_policy"] = combo_id(str(prefs["hold_policy"]))
        wb = prefs.get("window_bars")
        try:
            wb_i = int(wb) if wb is not None else None
        except (TypeError, ValueError):
            wb_i = None
        out["window_bars"] = wb_i if wb_i is not None and wb_i >= 800 else None
    return out


def load_all(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else PREFS_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): _clean(v) for k, v in data.items() if isinstance(v, dict)}
    except (OSError, ValueError):
        pass
    return {}


def save_all(prefs: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    p = Path(path) if path else PREFS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")


def get_symbol(symbol: str, path: Path | None = None) -> dict[str, Any]:
    return _clean(load_all(path).get(symbol or ""))


def set_symbol(symbol: str, prefs: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    if not symbol:
        return {}
    with _lock:
        allp = load_all(path)
        allp[symbol] = _clean(prefs)
        save_all(allp, path)
        return allp[symbol]