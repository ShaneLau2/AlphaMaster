"""web/hold_matrix_prefs.py — 全组合矩阵按品种记忆的工具栏设置。

{品种: {data_file, window_mode, window_bars, regime, chunks}} 存为
results/hold_matrix_prefs.json；切换品种回测时前端自动带出该品种上次设置。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFS_PATH = PROJECT_ROOT / "results" / "hold_matrix_prefs.json"

_lock = threading.Lock()

_DEFAULTS: dict[str, Any] = {
    "window_mode": "tail",
    "window_bars": None,
    "regime": "vol",
    "chunks": 4,
    "data_file": None,
}


def _clean(prefs: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(_DEFAULTS)
    if prefs:
        for k in _DEFAULTS:
            if prefs.get(k) is not None:
                out[k] = prefs[k]
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