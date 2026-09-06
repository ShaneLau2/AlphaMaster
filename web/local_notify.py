"""统一 macOS 本地通知 + 权限诊断（替代散落的 osascript 直调）。

事实约束（2026-09 实测）：
1. `osascript display notification` 对无 GUI 宿主/未授权应用会“静默成功”（rc=0 但横幅被丢）。
2. `UNUserNotificationCenter` 在无 .app bundle 的纯 CLI 里直接崩溃
   （bundleProxyForCurrentProcess is nil）——必须由已打包 GUI 应用调用才有意义。
3. 本机通知权限表（com.apple.ncprefs）里没有 Freebuff/Terminal 等宿主 → 权限从未授予，
   这是横幅被静默拦截的根因。

因此本模块：
- probe_permission()：给出一份诚实的状态报告（osascript rc、UN 是否可用、
  通知权限表里宿主条目、判定 verdict=ok / not-granted / denied / cli-un-bundleable）。
- notify()：osascript 投递 + 无论如何写 logs/notifications.log；被拦截时附可操作提示；
  环境变量 ALPHAMASTER_SPEAK_ALERTS=1 时追加 `say` 语音兜底（无需任何权限）。
- CLI：`python -m web.local_notify [--probe|--send "msg"]`。
"""
from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NCPREFS = Path.home() / "Library" / "Preferences" / "com.apple.ncprefs.plist"
NOTIFY_LOG = PROJECT_ROOT / "logs" / "notifications.log"
PROBE_BIN = PROJECT_ROOT / ".cache" / "tools" / "notify_probe"
HOST_HINTS = ("freebuff", "terminal", "iterm", "code", "python", "alphamaster", "launchd")


def _log(msg: str) -> None:
    try:
        NOTIFY_LOG.parent.mkdir(exist_ok=True)
        with NOTIFY_LOG.open("a", encoding="utf-8") as fp:
            fp.write(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def _ncprefs_hosts() -> list[dict]:
    """通知权限表里与潜在宿主相关的条目（bundle-id + flags + path）。"""
    if not NCPREFS.exists():
        return []
    try:
        with NCPREFS.open("rb") as fp:
            data = plistlib.load(fp)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for app in data.get("apps", []) or []:
        bid = str(app.get("bundle-id", ""))
        path = str(app.get("path", ""))
        if any(h in bid.lower() or h in path.lower() for h in HOST_HINTS):
            out.append({"bundle_id": bid, "path": path[:120],
                        "flags": app.get("flags"), "auth": app.get("auth")})
    return out


def _osascript_rc(title: str, msg: str) -> tuple[int, str]:
    if sys.platform != "darwin":
        return -1, "not darwin"
    safe = msg[:180].replace('"', "'")
    stitle = (title or "通知")[:60].replace('"', "'")
    try:
        r = subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe}" with title "{stitle}"'],
            capture_output=True, timeout=10, check=False)
        return r.returncode, (r.stderr or b"").decode(errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        return -1, str(exc)


def _un_probe() -> dict:
    """UNUserNotificationCenter 状态；纯 CLI 会崩 → 明确标注不可用（诚实降级）。"""
    if sys.platform != "darwin" or not PROBE_BIN.exists():
        return {"available": False, "reason": "非 darwin 或探测二进制缺失"}
    try:
        r = subprocess.run([str(PROBE_BIN)], capture_output=True, timeout=20, check=False)
        if r.returncode in (0, 2):
            txt = (r.stdout or b"").decode(errors="replace").strip() or '{"error":"no output"}'
            try:
                st = json.loads(txt.splitlines()[-1])
                st["available"] = True
                return st
            except ValueError:
                return {"available": False, "reason": f"输出不可解析: {txt[:80]}"}
        return {"available": False, "reason": f"rc={r.returncode}",
                "stderr": (r.stderr or b"").decode(errors="replace")[:200]}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)[:200]}


def _diagnose() -> dict:
    """verdict ∈ ok / not-granted / cli-unbundleable / non-darwin。"""
    if sys.platform != "darwin":
        return {"verdict": "non-darwin", "detail": "仅 macOS 支持本地通知"}
    hosts = _ncprefs_hosts()
    un = _un_probe()
    # UN 在纯 CLI 不可用 + 权限表里没有宿主 → 横幅几乎必然被静默丢弃
    if not hosts and not un.get("available"):
        verdict = "cli-unbundleable"
        detail = ("宿主应用（Freebuff/Terminal 等）未出现在系统通知权限表，且本进程是无 bundle 的 CLI，"
                  "UNUserNotificationCenter 不可用 → osascript 横幅会被静默丢弃")
    elif hosts and any(h.get("flags") for h in hosts):
        verdict = "ok"
        detail = f"通知权限表含宿主条目 {len(hosts)} 个；osascript 横幅取决于宿主系统设置"
    else:
        verdict = "not-granted"
        detail = "宿主应用存在但未授权/被关 → 通知被拦截"
    return {"verdict": verdict, "detail": detail, "hosts": hosts, "un": un}


def probe_permission() -> dict:
    """全面诊断（含一次 osascript 测试投递）→ 供 --probe / 设置页展示。"""
    diag = _diagnose()
    if sys.platform == "darwin":
        rc, err = _osascript_rc("AlphaMaster 通知探测",
                                f"通知通道测试 probe-{int(datetime.now().timestamp())}")
        diag["osascript_rc"] = rc
        diag["osascript_stderr"] = err[:200] or None
    else:
        diag["osascript_rc"] = None
    return diag


def notify(title: str, message: str) -> dict:
    """投递本地通知。永远落日志；被拦截时给出可操作提示。

    返回 {"posted": bool, "verdict": str, "log": str, "hint": str|None}。
    """
    _log(f"{title}: {message}")
    if sys.platform != "darwin":
        return {"posted": False, "verdict": "non-darwin", "log": str(NOTIFY_LOG),
                "hint": "非 macOS，仅记日志"}
    diag = _diagnose()
    rc, err = _osascript_rc(title, message)
    posted = rc == 0
    hint = None
    if not posted:
        posted = False
        hint = f"osascript rc={rc} {err or ''} — 已落日志 logs/notifications.log"
    elif diag.get("verdict") in ("cli-unbundleable", "not-granted"):
        posted = False
        hint = ("横幅权限未授予（宿主未在 系统设置→通知 中授权），横幅会被系统静默丢弃；"
                f"通知已落 logs/notifications.log。开启：系统设置→通知→允许宿主应用横幅；"
                "或设 ALPHAMASTER_SPEAK_ALERTS=1 用语音兜底")
    if os.environ.get("ALPHAMASTER_SPEAK_ALERTS") == "1":
        try:
            subprocess.run(["say", f"{title}. {message}"], timeout=20, check=False)
        except Exception:  # noqa: BLE001
            pass
    return {"posted": posted, "verdict": diag.get("verdict"), "log": str(NOTIFY_LOG),
            "hint": hint}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "--probe":
        print(json.dumps(probe_permission(), ensure_ascii=False, indent=2))
        return 0
    if argv[0] == "--send":
        msg = argv[1] if len(argv) > 1 else "测试通知"
        res = notify("AlphaMaster", msg)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["posted"] else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
