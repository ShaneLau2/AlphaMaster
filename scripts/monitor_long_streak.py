"""LONG 段持续监控：因子跌破阈值或方向翻转时通知。

监控实时分析页 5m BTCUSDT 监控项：
- factor < 1.0（跌破参考阈值）→ 通知；
- direction 离开 LONG（翻 SHORT/FLAT）→ 通知；
- 通知内容含当时价格与 81166 入场参考的对比。

通知通道（依次尝试，全部失败也不影响继续轮询）：
1. 飞书 Webhook（web.settings 里的 feishu_enabled / feishu_webhook_url / feishu_secret）
2. macOS 本地通知（osascript display notification）
3. logs/long_monitor.log 日志（总会写）

用法：
  .venv/bin/python scripts/monitor_long_streak.py [--once] [--test] [--poll 30]
  --once  检查一次即退出（配合 cron / launchd 的周期调用）
  --test  强制发一条测试通知后退出（验证通道）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WATCH_ID = "binance:BTCUSDT:5m:best_BTCUSDT"
FACTOR_FLOOR = 1.0        # 因子跌破该值视为 LONG 段降温
REF_PRICE = 81166.0       # 因子越过 1.0 开始 LONG 段时的参考价格
DEFAULT_POLL = 30         # 秒
BASE_URL = "http://127.0.0.1:8765"
LOG = ROOT / "logs" / "long_monitor.log"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def fetch_status(base: str, timeout_s: float = 15.0) -> dict:
    req = urllib.request.Request(f"{base}/api/realtime/status")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def find_watch(status: dict, watch_id: str) -> dict | None:
    for w in status.get("watches", []):
        if w.get("id") == watch_id:
            return w
    return None


def notify_local(title: str, message: str) -> bool:
    """macOS 本地通知（统一模块：投递 + 权限诊断 + 日志兜底）。"""
    if sys.platform != "darwin":
        return False
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from web.local_notify import notify as local_notify

        res = local_notify(title, message)
        if not res.get("posted") and res.get("hint"):
            print(f"[本地通知被拦截] {res.get('hint')}", flush=True)
        return bool(res.get("posted"))
    except Exception:  # noqa: BLE001
        return False


def notify_feishu(text: str) -> tuple[bool, str]:
    try:
        from web.feishu_notify import send_text
        return send_text(text)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def build_message(w: dict, reason: str) -> str:
    factor = w.get("factor_value")
    direction = w.get("direction")
    price = w.get("live_price") or w.get("last_close")
    price_s = f"{price:,.2f}" if isinstance(price, (int, float)) else "—"
    diff = ""
    if isinstance(price, (int, float)):
        pct = (price - REF_PRICE) / REF_PRICE * 100.0
        diff = f"（较 81166 参考 {'+' if pct >= 0 else ''}{pct:.2f}%）"
    tf = w.get("timeframe", "5m")
    sym = w.get("symbol", "BTCUSDT")
    factor_s = f"{factor:.4f}" if isinstance(factor, (int, float)) else "—"
    return (
        f"[AlphaMaster LONG 监控] {sym} {tf}：{reason}\n"
        f"当前方向={direction} 因子={factor_s}（阈值 {FACTOR_FLOOR}）\n"
        f"价格={price_s} {diff}"
    )


def check_once(args) -> int:
    """单次检查。触发条件满足 → 通知并返回 1；否则返回 0。"""
    try:
        status = fetch_status(args.base)
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] 无法获取实时状态（稍后重试）: {exc}")
        return 0

    w = find_watch(status, args.watch)
    if w is None:
        log(f"[warn] 未找到监控项 {args.watch}（监控项列表: "
            f"{[x.get('id') for x in status.get('watches', [])]}）")
        return 0

    direction = w.get("direction")
    factor = w.get("factor_value")
    factor = float(factor) if isinstance(factor, (int, float)) else None

    # 触发判定
    triggered = False
    reason = ""
    if direction == "LONG":
        if factor is not None and factor < FACTOR_FLOOR:
            triggered = True
            reason = f"因子跌破 {FACTOR_FLOOR}（{factor:.4f}）"
    else:
        triggered = True
        reason = f"方向离开 LONG（当前 {direction}）"

    if not triggered:
        f_s = f"{factor:.4f}" if factor is not None else "—"
        log(f"[ok] 仍为 LONG · 因子 {f_s} · 继续监控（每 {args.poll}s）")
        return 0

    msg = build_message(w, reason)
    log(f"[触发] {reason} -> {msg}")
    notify_local("AlphaMaster · LONG 段监控", msg)
    ok, fb = notify_feishu(msg)
    log(f"[飞书] {'✓ 推送成功' if ok else f'✗ 推送失败: {fb}'}")
    return 1


def main() -> int:
    global FACTOR_FLOOR
    ap = argparse.ArgumentParser(description="LONG 段因子/方向监控")
    ap.add_argument("--once", action="store_true", help="检查一次即退出")
    ap.add_argument("--test", action="store_true", help="强制发测试通知后退出")
    ap.add_argument("--poll", type=int, default=DEFAULT_POLL, help="轮询间隔秒")
    ap.add_argument("--watch", default=WATCH_ID)
    ap.add_argument("--base", default=BASE_URL)
    ap.add_argument("--factor-floor", type=float, default=FACTOR_FLOOR)
    args = ap.parse_args()

    FACTOR_FLOOR = args.factor_floor

    if args.test:
        fake = {
            "symbol": "BTCUSDT", "timeframe": "5m",
            "direction": "LONG", "factor_value": 0.95,
            "live_price": 82350.0,
        }
        msg = build_message(fake, "测试通知：因子跌破 1.0（示例）")
        log(f"[test] {msg}")
        notify_local("AlphaMaster · LONG 段监控（测试）", msg)
        ok, fb = notify_feishu(msg)
        log(f"[test][飞书] {'✓ 推送成功' if ok else f'✗ 推送失败: {fb}'}")
        return 0

    if args.once:
        return check_once(args)

    log(f"LONG 段监控启动：{args.watch} · 因子阈值 {FACTOR_FLOOR} · 轮询 {args.poll}s")
    while True:
        try:
            if check_once(args):
                return 0  # 已触发并通知，退出（launchd KeepAlive 不会重启正常退出）
        except Exception as exc:  # noqa: BLE001
            log(f"[warn] 轮询异常（不中断）: {exc}")
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())