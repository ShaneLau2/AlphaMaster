"""Native file picker for local training UI — 非阻塞会话版。

设计：选择对话框与请求解耦——POST /browse 立即返回 session id，
前端轮询 poll 端点直到用户选完/取消。彻底解决旧版的竞态：
「前端 12s 客户端超时 vs 对话框 20s 服务端上限」——用户慢慢翻目录时
请求被前端掐断，但对话框仍在后台，重复点击会叠出多个窗口并刷「网络错误」。

现在：唯一一个选择窗口，能开多久开多久（上限 10 分钟防泄漏），
重复点击返回同一个会话，不再制造错误。

平台：Windows 优先 comdlg32.GetOpenFileNameW（不依赖 tkinter/Tcl/Tk），
macOS 把 tkinter 对话框放进独立子进程【主线程】（Tcl/Tk 非线程安全，
在 uvicorn 工作线程里调 Tk 会进程级死锁），其余平台用后台线程跑 Tk。
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

_MAX_PICKER_SECONDS = 600.0  # 对话框最长存活：超时自动杀（防窗口/进程泄漏）

# macOS 子进程版选择器：在子进程【主线程】里跑 Tk，结果写入 result_path + stdout
_PICKER_SUBPROCESS = r'''
import base64
import json
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    print(json.dumps({"error": "当前环境不支持图形文件选择（缺少 tkinter）"}))
    raise SystemExit(1)

if __name__ == "__main__":
    # argv: title, filetypes(JSON, base64), initialdir, result_path
    title = sys.argv[1] if len(sys.argv) >= 2 else "选择文件"
    filetypes = []
    if len(sys.argv) >= 3:
        try:
            filetypes = json.loads(base64.b64decode(sys.argv[2]).decode())
        except Exception:
            filetypes = []
    initialdir = sys.argv[3] if len(sys.argv) >= 4 else ""
    result_path = sys.argv[4] if len(sys.argv) >= 5 else ""
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askopenfilename(
            title=title, filetypes=filetypes, initialdir=initialdir or None
        )
        out = {"path": path or None}
    except Exception as exc:
        out = {"error": str(exc)}
    if result_path:
        try:
            Path(result_path).write_text(json.dumps(out), encoding="utf-8")
        except Exception:
            pass
    print(json.dumps(out))
'''


def _pick_with_win32(
    title: str,
    filetypes: Sequence[tuple[str, str]],
    initialdir: str | None = None,
) -> str | None:
    """Windows 系统原生打开对话框（ctypes + comdlg32）。"""
    import ctypes
    from ctypes import wintypes

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    OFN_FILEMUSTEXIST = 0x00001000
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_EXPLORER = 0x00080000
    OFN_NOCHANGEDIR = 0x00000008
    OFN_HIDEREADONLY = 0x00000004

    # Filter: "Label\0pattern\0Label2\0pattern2\0\0"
    parts: list[str] = []
    for label, pattern in filetypes:
        parts.append(label)
        parts.append(pattern)
    filter_buf = ctypes.create_unicode_buffer("\0".join(parts) + "\0")

    max_path = 32768
    file_buf = ctypes.create_unicode_buffer(max_path)

    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter = ctypes.cast(filter_buf, wintypes.LPCWSTR)
    ofn.nFilterIndex = 1
    ofn.lpstrFile = ctypes.cast(file_buf, wintypes.LPWSTR)
    ofn.nMaxFile = max_path
    ofn.lpstrTitle = title
    if initialdir:
        ofn.lpstrInitialDir = initialdir
    ofn.Flags = (
        OFN_EXPLORER
        | OFN_FILEMUSTEXIST
        | OFN_PATHMUSTEXIST
        | OFN_NOCHANGEDIR
        | OFN_HIDEREADONLY
    )

    comdlg32 = ctypes.WinDLL("comdlg32", use_last_error=True)
    GetOpenFileNameW = comdlg32.GetOpenFileNameW
    GetOpenFileNameW.argtypes = [ctypes.POINTER(OPENFILENAMEW)]
    GetOpenFileNameW.restype = wintypes.BOOL

    ok = GetOpenFileNameW(ctypes.byref(ofn))
    if not ok:
        return None
    path = file_buf.value.strip()
    return path or None


def _pick_with_tkinter(
    title: str,
    filetypes: Sequence[tuple[str, str]],
    initialdir: str | None = None,
) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError("当前环境不支持图形文件选择（缺少 tkinter）")

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    path = filedialog.askopenfilename(
        title=title, filetypes=list(filetypes), initialdir=initialdir
    )
    root.destroy()
    return path or None


# ---------- 会话注册表：一个时刻最多一个选择窗口 ----------

_PICKERS: dict[str, dict[str, Any]] = {}
_PICKERS_LOCK = threading.Lock()


def _unlink_quiet(path: str) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def _sweep_expired() -> None:
    """清理已完成/超时的会话（调用方须持有 _PICKERS_LOCK）。"""
    now = time.monotonic()
    for sid in list(_PICKERS):
        p = _PICKERS[sid]
        if p["done"] or now - p["started"] > _MAX_PICKER_SECONDS:
            proc = p.get("proc")
            if proc is not None and proc.poll() is None:
                proc.kill()
            _unlink_quiet(p.get("result_path") or "")
            _PICKERS.pop(sid, None)


def _thread_pick(sid: str, fn: Any, *args: Any) -> None:
    """后台线程里跑阻塞式选择器（Windows comdlg32 / Linux Tk）。"""
    try:
        path = fn(*args)
        with _PICKERS_LOCK:
            if sid in _PICKERS and not _PICKERS[sid]["done"]:
                _PICKERS[sid]["path"] = path
                _PICKERS[sid]["done"] = True
    except Exception as exc:  # noqa: BLE001
        with _PICKERS_LOCK:
            if sid in _PICKERS:
                _PICKERS[sid]["error"] = str(exc)
                _PICKERS[sid]["done"] = True


def start_picker(
    title: str,
    filetypes: Sequence[tuple[str, str]],
    initialdir: str | None = None,
) -> tuple[str, bool]:
    """启动文件选择会话。返回 (session_id, created)。

    已有一个选择窗口打开时返回 (现存 session_id, False)——前端应跟随
    同一个会话轮询，而不是再弹一个。
    """
    with _PICKERS_LOCK:
        _sweep_expired()
        live = [sid for sid, p in _PICKERS.items() if not p["done"]]
        if live:
            return live[0], False
        sid = uuid.uuid4().hex[:12]
        _PICKERS[sid] = {
            "done": False,
            "path": None,
            "error": None,
            "started": time.monotonic(),
            "proc": None,
            "result_path": "",
        }

    if sys.platform == "darwin":
        # macOS：Tk 必须在主线程；独立子进程弹窗（不阻塞服务线程）
        fd, res_path = tempfile.mkstemp(prefix="alphamaster_picker_", suffix=".json")
        os.close(fd)
        payload = base64.b64encode(json.dumps(list(filetypes)).encode()).decode()
        with _PICKERS_LOCK:
            _PICKERS[sid]["result_path"] = res_path
            _PICKERS[sid]["proc"] = subprocess.Popen(
                [
                    sys.executable, "-c", _PICKER_SUBPROCESS,
                    title, payload, initialdir or "", res_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif sys.platform == "win32":
        threading.Thread(
            target=_thread_pick,
            args=(sid, _pick_with_win32, title, filetypes, initialdir or ""),
            daemon=True,
        ).start()
    else:
        threading.Thread(
            target=_thread_pick,
            args=(sid, _pick_with_tkinter, title, filetypes, initialdir or ""),
            daemon=True,
        ).start()
    return sid, True


def poll_picker(sid: str) -> dict[str, Any]:
    """查询选择会话状态：{"done": bool, "path": str|None, "error": str|None}。"""
    with _PICKERS_LOCK:
        entry = _PICKERS.get(sid)
        if entry is None:
            return {"done": True, "path": None, "error": "选择会话不存在（服务可能已重启）"}
        proc = entry.get("proc")
        if proc is not None and proc.poll() is not None and not entry["done"]:
            # 子进程退出（选完/取消/被杀）→ 读结果文件
            try:
                data = json.loads(Path(entry["result_path"]).read_text(encoding="utf-8"))
            except Exception:
                data = {}
            entry["path"] = data.get("path")
            entry["error"] = data.get("error")
            entry["done"] = True
        if not entry["done"] and time.monotonic() - entry["started"] > _MAX_PICKER_SECONDS:
            # 对话框长挂（如被系统吞掉）——杀进程、按取消处理
            if proc is not None and proc.poll() is None:
                proc.kill()
            entry["done"] = True
            entry["error"] = "选择器超时（10 分钟），已自动关闭"
        return {
            "done": entry["done"],
            "path": entry.get("path"),
            "error": entry.get("error"),
        }