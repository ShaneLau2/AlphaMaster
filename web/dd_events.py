"""web/dd_events.py — DD 熔断/收复事件日志（JSONL，追加写，供回溯每次熔断/收复）。

模拟实盘与实时分析两条引擎共用：状态转移（OK→熔断→深档→收复→OK）时调用
`log_dd_event`，事件与当时的 回撤幅度/现价/峰值/恢复条件 全部落盘，
方便事后审计「什么时候熔断、什么时候收复、当时行情多少」。

日志文件：logs/dd_gate_events.jsonl（按时间追加；读取端可用 web 页面/脚本查询）。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_ROOT / "logs" / "dd_gate_events.jsonl"

_lock = threading.Lock()


def _ensure_dir() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_dd_event(
    scope: str,
    *,
    symbol: str,
    timeframe: str,
    policy_id: str,
    event: str,
    dd_pct: float | None = None,
    mark: float | None = None,
    peak: float | None = None,
    prev_state: int = 0,
    state: int = 0,
    detail: str = "",
) -> None:
    """追加一条 DD 事件。scope ∈ {"paper", "realtime"}（真实闸门/信息跟踪）。"""
    try:
        row: dict[str, Any] = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": scope,
            "symbol": symbol,
            "timeframe": timeframe,
            "policy_id": policy_id,
            "event": event,
            "dd_pct": round(float(dd_pct), 4) if isinstance(dd_pct, (int, float)) else None,
            "mark": round(float(mark), 8) if isinstance(mark, (int, float)) else None,
            "peak": round(float(peak), 8) if isinstance(peak, (int, float)) else None,
            "prev_state": int(prev_state),
            "state": int(state),
        }
        if detail:
            row["detail"] = detail
        with _lock:
            _ensure_dir()
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 日志失败绝不影响交易/闸门
        pass


def recent_events(limit: int = 50) -> list[dict[str, Any]]:
    """读取最近 N 条事件（新→旧）。文件不存在/损坏时返回 []。"""
    try:
        if not LOG_FILE.exists():
            return []
        with _lock:
            lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for ln in reversed(lines):
            try:
                out.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                continue
            if len(out) >= limit:
                break
        return out
    except Exception:  # noqa: BLE001
        return []