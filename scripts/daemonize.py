"""daemonize.py — 把长任务脱离当前会话后台运行（双 fork + setsid + 日志重定向）。

本环境 launchctl bootstrap 不稳定（I/O error），长任务改用纯 daemon 化：
- 双 fork：脱离工具 shell 的进程组/会话，shell 退出不影响它；
- stdio 重定向到日志文件；
- 立即返回（同步调用者不等任务）。

用法:
  python scripts/daemonize.py <log_path> -- <cmd...>

示例:
  python scripts/daemonize.py logs/nb.log -- \\
      .venv/bin/python -u scripts/neutral_band_experiment.py --steps 300
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = sys.argv[1:]
    if "--" not in args or len(args) < 2:
        print(__doc__)
        return 2
    log_path = args[0]
    cmd = args[args.index("--") + 1:]
    log_p = Path(log_path)
    if not log_p.is_absolute():
        log_p = PROJECT_ROOT / log_p
    log_p.parent.mkdir(parents=True, exist_ok=True)

    # 第一层 fork：父进程立即返回
    pid1 = os.fork()
    if pid1 > 0:
        print(f"[daemonize] pid={pid1} 已脱离会话后台运行: {' '.join(cmd)}\n"
              f"  日志 -> {log_p}")
        return 0
    # 子进程：新会话 + 第二层 fork（确保不是会话首进程，避免重获控制终端）
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)
    os.chdir(str(PROJECT_ROOT))
    fd = os.open(str(log_p), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    if fd > 2:
        os.close(fd)
    # 解析命令：argv[0] 相对路径按项目根解析
    prog = cmd[0]
    if not Path(prog).is_absolute():
        prog = str(PROJECT_ROOT / prog)
    os.execv(prog, [prog, *cmd[1:]])


if __name__ == "__main__":
    sys.exit(main())