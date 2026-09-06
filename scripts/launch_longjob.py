"""把长任务交给 launchd 托管（RunAtLoad + KeepAlive），脱离终端/工具会话存活。

用法:
  python scripts/launch_longjob.py <service_name> <log_path> -- <cmd...>

示例:
  python scripts/launch_longjob.py nb_exp logs/nb.log -- \\
      .venv/bin/python -u scripts/neutral_band_experiment.py --steps 300

进程立即由 launchd 启动；若被杀会按 KeepAlive 自动重启。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core.supervise import attach_launchd  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    if "--" not in args or len(args) < 3:
        print(__doc__)
        return 2
    svc = args[0]
    log_path = args[1]
    cmd = args[args.index("--") + 1:]
    log_p = Path(log_path)
    if not log_p.is_absolute():
        log_p = PROJECT_ROOT / log_p
    plist = attach_launchd(svc, log_path=str(log_p), argv=cmd)
    if plist is None:
        print(f"[launch_longjob] 挂载失败: {svc}")
        return 1
    print(f"[launch_longjob] 已挂载 {svc} -> {plist}\n  cmd: {' '.join(cmd)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())