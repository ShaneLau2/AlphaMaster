"""Persisted UI settings for the training web console."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "web_settings.json"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"

_DEFAULT = {
    "last_data_file": "",
    "last_strategy_file": "",
    "debug_mode": False,
    # 背景动效(bg.js 神经网络/数学符号画布): False=关(默认,页面更流畅省电); True=开
    "bg_animation": False,
    "ai_provider": "deepseek",
    "ai_api_key": "",
    # OpenAI 兼容网关（DeepSeek / SenseNova 等）：可自定义 base_url + model
    "ai_base_url": "https://api.deepseek.com",
    "ai_model": "deepseek-v4-flash",
    # 回测单边成本（单位 %）：手续费 0.02% + 滑点 0.01% ≈ 常见加密货币轻度成本
    "bt_commission_pct": 0.02,
    "bt_slippage_pct": 0.01,
    # 回测页记忆化：上次持仓组合（可含 + 组合）与样本外窗口根数
    "bt_hold_policy": "signal",
    "bt_window_bars": None,
    # 回测页「矩阵最优 → 新回测默认」的用户接受记录（与手动记忆区分开）：
    # {品种: {combo, at, sharpe, max_drawdown, window_bars, window_mode}}；
    # 记录的是用户对某次 N×N 最优组合的明确接受，重启后据此恢复「已应用」态。
    "matrix_applied": {},
    # 实时分析监控清单：[{source, symbol, timeframe, strategy_file}, ...]
    "realtime_watches": [],
    # 模拟实盘（纸上交易）账户参数
    "paper_starting_balance": 100000.0,   # 起始资金（模拟货币）
    "paper_notional": 10000.0,            # 每份满仓名义金额
    "paper_commission_pct": 0.02,         # 单边手续费 %
    "paper_slippage_pct": 0.01,           # 单边滑点 %
    # 每笔投入上限（占当时账户权益 %，回测/模拟实盘共用）：
    # 单笔名义 = 当时权益 × 上限% × 强度 |tanh(因子)|（回测离散/连续引擎）；
    # 模拟实盘再受「满仓名义金额」货币额封顶。100% = 权益的整份。
    "max_position_pct": 100.0,
    # 统一无信号阈值：|tanh(因子)| 小于该值时方向为 FLAT（观望区）。
    # 回测/回放/实时/模拟实盘共用；可选 0.05 / 0.3 / 0.5 / 0.8。
    "signal_threshold": 0.05,
    # 飞书机器人（信号转折提醒，仅文本）
    "feishu_enabled": False,
    "feishu_webhook_url": "",
    "feishu_secret": "",
    "rt_alert_dev_pct": 0.5,        # 实时监控：价格偏离“持仓方案入场参考”告警阈值 %（0=关闭）
    "rt_alert_stale_bars": 0,       # 实时监控：因子硬钝化飞书告警阈值（连续根数，0=关闭）
    # tqsdk 天勤量化账号（国内期货实时数据源）
    "tqsdk_user": "七斗居士",
    "tqsdk_password": "ghhkphs8",
}


SIGNAL_THRESHOLDS = (0.05, 0.3, 0.5, 0.8)  # 可选无信号阈值档位


def sanitize_signal_threshold(value) -> float:
    """阈值只允许档位集合；非法/缺失回退默认 0.05。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _DEFAULT["signal_threshold"]
    if v in SIGNAL_THRESHOLDS:
        return v
    # 容差匹配（如 0.30000000000004），否则回退默认
    for cand in SIGNAL_THRESHOLDS:
        if abs(v - cand) < 1e-9:
            return cand
    return _DEFAULT["signal_threshold"]


_MATRIX_APPLIED_FIELDS = ("combo", "at", "sharpe", "max_drawdown", "window_bars", "window_mode")


def _clean_matrix_applied(raw) -> dict:
    """清洗 matrix_applied：{品种: {combo, at, sharpe, max_drawdown, ...}}。

    只保留含非空 combo 的记录；数值字段丢失/非法时跳过该字段。"""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for sym, rec in raw.items():
        if not isinstance(rec, dict):
            continue
        combo = str(rec.get("combo") or "").strip()
        if not combo:
            continue
        row: dict = {"combo": combo}
        for k in _MATRIX_APPLIED_FIELDS[1:]:
            v = rec.get(k)
            if v is None:
                continue
            if k in ("sharpe", "max_drawdown"):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
            row[k] = v
        sym = str(sym).strip()
        if sym:
            out[sym] = row
    return out


def resolve_signal_threshold() -> float:
    """统一无信号阈值：web_settings 优先，回退 Config.MIN_TRADE_EXPOSURE。"""
    try:
        return sanitize_signal_threshold(load_settings().get("signal_threshold"))
    except Exception:  # noqa: BLE001
        try:
            from config import Config
            return float(getattr(Config, "MIN_TRADE_EXPOSURE", 0.05))
        except Exception:  # noqa: BLE001
            return 0.05


def _as_pct(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0:
        return default
    return v


def _is_ephemeral_data_path(path: str) -> bool:
    """Stale pytest temp parquet paths (not valid training data)."""
    norm = str(path or "").replace("\\", "/").lower()
    if "pytest-of-" not in norm:
        return False
    return (
        "/appdata/local/temp/" in norm
        or norm.startswith("/tmp/")
        or "/temp/" in norm
    )


def _is_production_settings_path() -> bool:
    try:
        return SETTINGS_PATH.resolve() == (PROJECT_ROOT / "web_settings.json").resolve()
    except OSError:
        return False


def _is_usable_data_file(path: str) -> bool:
    p = Path(str(path or "").strip())
    return p.is_file() and p.suffix.lower() == ".parquet"


def _should_replace_last_data_file(path: str) -> bool:
    cur = str(path or "").strip()
    if not cur:
        return True
    if not Path(cur).is_file():
        return True
    return _is_ephemeral_data_path(cur)


def _data_file_from_strategy_json(path: str) -> str | None:
    p = Path(str(path or "").strip())
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    candidate = str(data.get("data_file") or "").strip()
    if _is_usable_data_file(candidate):
        return str(Path(candidate).resolve())
    return None


def _recover_last_data_file(current: dict) -> str:
    cur = str(current.get("last_data_file") or "").strip()
    if cur and not _should_replace_last_data_file(cur):
        return str(Path(cur).resolve())

    for strategy_path in (
        str(current.get("last_strategy_file") or "").strip(),
        *(str(p) for p in sorted(STRATEGIES_DIR.glob("best_*.json"))
          if p.is_file() and not p.name.endswith(".live.json")),
    ):
        if not strategy_path:
            continue
        candidate = _data_file_from_strategy_json(strategy_path)
        if candidate:
            return candidate
    return cur


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return dict(_DEFAULT)
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT)
    out = dict(_DEFAULT)
    out.update({k: v for k, v in data.items() if k in _DEFAULT})
    out["debug_mode"] = bool(out.get("debug_mode", False))
    out["bg_animation"] = bool(out.get("bg_animation", False))
    out["last_strategy_file"] = str(out.get("last_strategy_file") or "").strip()
    out["ai_provider"] = str(out.get("ai_provider") or "deepseek").strip().lower()
    if out["ai_provider"] not in ("deepseek", "openclaw", "openclaw_wb"):
        out["ai_provider"] = "deepseek"
    out["ai_api_key"] = str(out.get("ai_api_key") or "").strip()
    out["ai_base_url"] = str(
        out.get("ai_base_url") or _DEFAULT["ai_base_url"]
    ).strip()
    out["ai_model"] = str(out.get("ai_model") or _DEFAULT["ai_model"]).strip()
    out["bt_commission_pct"] = _as_pct(
        out.get("bt_commission_pct"), _DEFAULT["bt_commission_pct"]
    )
    out["bt_slippage_pct"] = _as_pct(
        out.get("bt_slippage_pct"), _DEFAULT["bt_slippage_pct"]
    )
    watches = out.get("realtime_watches")
    if not isinstance(watches, list):
        watches = []
    cleaned = []
    for w in watches:
        if not isinstance(w, dict):
            continue
        src = str(w.get("source") or "").strip()
        sym = str(w.get("symbol") or "").strip()
        tf = str(w.get("timeframe") or "").strip()
        sf = str(w.get("strategy_file") or "").strip()
        if src and sym and tf and sf:
            cleaned.append(
                {"source": src, "symbol": sym, "timeframe": tf, "strategy_file": sf}
            )
    out["realtime_watches"] = cleaned
    out["feishu_enabled"] = bool(out.get("feishu_enabled", False))
    out["feishu_webhook_url"] = str(out.get("feishu_webhook_url") or "").strip()
    out["feishu_secret"] = str(out.get("feishu_secret") or "").strip()
    out["rt_alert_dev_pct"] = _as_pct(out.get("rt_alert_dev_pct"), _DEFAULT["rt_alert_dev_pct"])
    if out["rt_alert_dev_pct"] > 50.0:
        out["rt_alert_dev_pct"] = 50.0
    try:
        out["rt_alert_stale_bars"] = max(0, int(out.get("rt_alert_stale_bars") or 0))
    except (TypeError, ValueError):
        out["rt_alert_stale_bars"] = 0
    out["tqsdk_user"] = str(out.get("tqsdk_user") or "").strip()
    out["tqsdk_password"] = str(out.get("tqsdk_password") or "").strip()
    out["paper_starting_balance"] = _as_pct(
        out.get("paper_starting_balance"), _DEFAULT["paper_starting_balance"]
    )
    out["paper_notional"] = _as_pct(out.get("paper_notional"), _DEFAULT["paper_notional"])
    out["paper_commission_pct"] = _as_pct(
        out.get("paper_commission_pct"), _DEFAULT["paper_commission_pct"]
    )
    out["paper_slippage_pct"] = _as_pct(
        out.get("paper_slippage_pct"), _DEFAULT["paper_slippage_pct"]
    )
    v_pos = _as_pct(out.get("max_position_pct"), _DEFAULT["max_position_pct"])
    out["max_position_pct"] = min(200.0, max(1.0, v_pos))
    out["signal_threshold"] = sanitize_signal_threshold(out.get("signal_threshold"))
    out["matrix_applied"] = _clean_matrix_applied(out.get("matrix_applied"))
    recovered = _recover_last_data_file(out)
    if recovered != out.get("last_data_file") and _is_production_settings_path():
        out["last_data_file"] = recovered
        if recovered:
            SETTINGS_PATH.write_text(
                json.dumps(out, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    elif recovered != out.get("last_data_file"):
        out["last_data_file"] = recovered
    return out


def save_settings(data: dict) -> dict:
    current = load_settings()
    if "last_data_file" in data:
        path = str(data["last_data_file"] or "").strip()
        if (
            path
            and _is_ephemeral_data_path(path)
            and _is_production_settings_path()
        ):
            data = {k: v for k, v in data.items() if k != "last_data_file"}
        else:
            current["last_data_file"] = path
    if "last_strategy_file" in data:
        current["last_strategy_file"] = str(data["last_strategy_file"] or "").strip()
    if "debug_mode" in data:
        current["debug_mode"] = bool(data["debug_mode"])
    if "bg_animation" in data:
        current["bg_animation"] = bool(data["bg_animation"])
    if "ai_provider" in data:
        provider = str(data["ai_provider"] or "deepseek").strip().lower()
        current["ai_provider"] = (
            provider if provider in ("deepseek", "openclaw", "openclaw_wb") else "deepseek"
        )
    if "ai_api_key" in data:
        current["ai_api_key"] = str(data["ai_api_key"] or "").strip()
    if "ai_base_url" in data:
        url = str(data["ai_base_url"] or "").strip()
        current["ai_base_url"] = url or _DEFAULT["ai_base_url"]
    if "ai_model" in data:
        model = str(data["ai_model"] or "").strip()
        current["ai_model"] = model or _DEFAULT["ai_model"]
    if "bt_commission_pct" in data:
        current["bt_commission_pct"] = _as_pct(
            data["bt_commission_pct"], _DEFAULT["bt_commission_pct"]
        )
    if "bt_slippage_pct" in data:
        current["bt_slippage_pct"] = _as_pct(
            data["bt_slippage_pct"], _DEFAULT["bt_slippage_pct"]
        )
    if "bt_hold_policy" in data:
        pid = str(data["bt_hold_policy"] or "signal").strip().lower() or "signal"
        from web.hold_policy import combo_id

        current["bt_hold_policy"] = combo_id(pid)
    if "bt_window_bars" in data:
        try:
            wb = int(data["bt_window_bars"])
        except (TypeError, ValueError):
            wb = 0
        current["bt_window_bars"] = wb if wb >= 800 else None
    if "realtime_watches" in data:
        watches = data["realtime_watches"]
        if not isinstance(watches, list):
            watches = []
        cleaned = []
        for w in watches:
            if not isinstance(w, dict):
                continue
            src = str(w.get("source") or "").strip()
            sym = str(w.get("symbol") or "").strip()
            tf = str(w.get("timeframe") or "").strip()
            sf = str(w.get("strategy_file") or "").strip()
            if src and sym and tf and sf:
                cleaned.append(
                    {
                        "source": src,
                        "symbol": sym,
                        "timeframe": tf,
                        "strategy_file": sf,
                    }
                )
        current["realtime_watches"] = cleaned
    if "feishu_enabled" in data:
        current["feishu_enabled"] = bool(data["feishu_enabled"])
    if "feishu_webhook_url" in data:
        current["feishu_webhook_url"] = str(data["feishu_webhook_url"] or "").strip()
    if "feishu_secret" in data:
        current["feishu_secret"] = str(data["feishu_secret"] or "").strip()
    if "rt_alert_dev_pct" in data:
        v = _as_pct(data["rt_alert_dev_pct"], _DEFAULT["rt_alert_dev_pct"])
        current["rt_alert_dev_pct"] = min(50.0, max(0.0, v))
    if "rt_alert_stale_bars" in data:
        try:
            current["rt_alert_stale_bars"] = max(0, int(data["rt_alert_stale_bars"]))
        except (TypeError, ValueError):
            current["rt_alert_stale_bars"] = 0
    if "tqsdk_user" in data:
        current["tqsdk_user"] = str(data["tqsdk_user"] or "").strip()
    if "tqsdk_password" in data:
        current["tqsdk_password"] = str(data["tqsdk_password"] or "").strip()
    if "paper_starting_balance" in data:
        current["paper_starting_balance"] = _as_pct(
            data["paper_starting_balance"], _DEFAULT["paper_starting_balance"]
        )
    if "paper_notional" in data:
        current["paper_notional"] = _as_pct(
            data["paper_notional"], _DEFAULT["paper_notional"]
        )
    if "paper_commission_pct" in data:
        current["paper_commission_pct"] = _as_pct(
            data["paper_commission_pct"], _DEFAULT["paper_commission_pct"]
        )
    if "paper_slippage_pct" in data:
        current["paper_slippage_pct"] = _as_pct(
            data["paper_slippage_pct"], _DEFAULT["paper_slippage_pct"]
        )
    if "max_position_pct" in data:
        v = _as_pct(data["max_position_pct"], _DEFAULT["max_position_pct"])
        current["max_position_pct"] = min(200.0, max(1.0, v))
    if "signal_threshold" in data:
        current["signal_threshold"] = sanitize_signal_threshold(data["signal_threshold"])
    if "matrix_applied" in data:
        current["matrix_applied"] = _clean_matrix_applied(data["matrix_applied"])
    SETTINGS_PATH.write_text(
        json.dumps(current, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return current
