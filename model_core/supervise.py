"""防假中止：SIGTERM 到达时若父进程已消失（孤儿化），先原地把自己挂到
launchd（KeepAlive 包裹）再继续运行，而不是退出。

背景：长训练 / 对比实验跑在终端 / SSH / web 子进程里时，会话被回收
（终端关闭、ssh 断开、launchd 整组回收）会先杀掉父进程，随后给子进程
补一发 SIGTERM。子进程没有父进程可问，一律当成“用户中止”退出 →
假中止（历史症状：训练在 step≈1 被“训练被中止”、exit 78）。

规则：
- 父进程仍活着（web 停止按钮 / 终端 Ctrl+C）→ 走正常中止流程；
- 父进程已消失（孤儿化）→ 写 ~/Library/LaunchAgents/com.alphamaster.<name>.plist
  （RunAtLoad + KeepAlive，ProgramArguments = 当前完整命令行），bootstrap 到
  launchd，然后忽略这发 SIGTERM 继续跑。若之后真被系统杀掉，launchd 会自动重启。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def is_orphaned() -> bool:
    """父进程是否已消失：getppid≤1（已被 init/launchd 收养）或 kill(ppid,0) 报不存在。"""
    try:
        ppid = os.getppid()
    except Exception:  # noqa: BLE001
        return True
    if ppid <= 1:
        return True
    try:
        os.kill(ppid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except Exception:  # noqa: BLE001
        return True


def _launch_agents_dir() -> Path:
    home = os.environ.get("HOME") or str(Path.home())
    return Path(home) / "Library" / "LaunchAgents"


def attach_launchd(service_name: str, log_path: str | None = None,
                   argv: list[str] | None = None) -> Path | None:
    """原地重挂 launchd：写 <label>.plist（RunAtLoad+KeepAlive，当前命令行 +
    ALPHAMASTER_UNDER_LAUNCHD=1 环境标记）并 bootstrap。进程本身不退出；
    返回 plist 路径，失败返回 None（绝不抛异常阻塞调用方）。

    argv 缺省 = [sys.executable, *sys.argv]（重挂自己）；
    也可显式传要托管的命令（用于把长任务交给 launchd 运行）。
    """
    if sys.platform == "win32":
        return None
    label = f"com.alphamaster.{service_name}"
    plist = _launch_agents_dir() / f"{label}.plist"
    try:
        import plistlib

        env = dict(os.environ)
        env["ALPHAMASTER_UNDER_LAUNCHD"] = "1"
        wd = str(Path(__file__).resolve().parents[1])  # 项目根
        prog_args = list(argv if argv is not None else [sys.executable, *sys.argv])
        # launchd 不支持相对 program 路径（相对路径 spawn 直接失败，last exit=78 EX_CONFIG）
        if prog_args and not Path(prog_args[0]).is_absolute():
            prog_args[0] = str(Path(wd) / prog_args[0])
        payload: dict = {
            "Label": label,
            "ProgramArguments": prog_args,
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "WorkingDirectory": wd,
            "EnvironmentVariables": env,
        }
        if log_path:
            payload["StandardOutPath"] = str(log_path)
            payload["StandardErrorPath"] = str(log_path)
        plist.parent.mkdir(parents=True, exist_ok=True)
        plist.write_bytes(plistlib.dumps(payload))
        uid = os.getuid()
        r = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist)],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            # 已 bootstrap 过会报 5/37；尝试旧式 load 兜底（幂等）
            subprocess.run(
                ["launchctl", "load", str(plist)],
                capture_output=True, text=True, timeout=15,
            )
        return plist
    except Exception:  # noqa: BLE001 挂载失败绝不阻塞训练
        return None


def guard_sigterm(service_name: str | None = None, log_path: str | None = None) -> bool:
    """SIGTERM 处理器内调用。

    Returns:
        True  = 孤儿化假中止 → 已重挂 launchd，调用方应忽略信号继续跑；
        False = 父进程仍在（正常中止：web 停止按钮 / 终端 Ctrl+C）→ 正常退出。
    """
    if not is_orphaned():
        return False
    if service_name:
        attach_launchd(service_name, log_path)
    return True


def install_orphan_sigterm_handler(
    service_name: str,
    log_path: str | None = None,
) -> None:
    """进程级 SIGTERM/SIGINT 处理器：孤儿化时重挂 launchd 并忽略信号；
    否则按 128+sig 退出（与现有“中止=退出码 128+信号”语义一致）。"""
    import signal

    def _handler(sig, frame):  # noqa: ANN001
        if guard_sigterm(service_name, log_path):
            return  # 孤儿化：已重挂 launchd，继续跑
        raise SystemExit(128 + int(sig))

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError, AttributeError):
        pass  # 非主线程 / 无信号环境跳过