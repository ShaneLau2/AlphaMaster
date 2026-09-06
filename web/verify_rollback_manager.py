"""web/verify_rollback_manager.py — 冠军回滚校验的子进程管理。

训练页「回滚校验」按钮 → POST /api/training/verify-rollback 起一个
scripts/verify_champion_rollback.py 子进程（可带 --retrain-steps），
stdout 写 logs/verify_rollback_*.log；前端轮询 status 拿状态与最新报告
（results/verify_champion_rollback_<stamp>.md / .json）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from utils.train_logging import strip_ansi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"


class VerifyState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class VerifyJob:
    state: VerifyState = VerifyState.RUNNING
    pid: int | None = None
    log_path: str = ""
    stamp: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    strategy_file: str | None = None
    retrain_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "pid": self.pid,
            "log_path": self.log_path,
            "stamp": self.stamp,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "strategy_file": self.strategy_file,
            "retrain_steps": self.retrain_steps,
        }


class VerifyRollbackManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: VerifyJob | None = None
        self._log_fp = None

    # ── 状态 ────────────────────────────────────────────────────────
    def _refresh(self) -> None:
        if self._job and self._job.state == VerifyState.RUNNING:
            code = self._proc.poll() if self._proc is not None else 0
            if code is not None:
                self._job.state = VerifyState.DONE if code == 0 else VerifyState.FAILED
                self._job.exit_code = code
                self._job.error = None if code == 0 else f"exit={code}，详见日志"
                self._job.finished_at = datetime.now(timezone.utc).isoformat()
                self._proc = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh()
            out: dict[str, Any] = {"active": False, "job": None, "report": None,
                                   "log_tail": []}
            if self._job:
                out["active"] = self._job.state == VerifyState.RUNNING
                out["job"] = self._job.to_dict()
                out["log_tail"] = self.tail_log(80)
                # 报告（md + json）读取该 job 自己的产物
                if self._job.state in (VerifyState.DONE, VerifyState.FAILED) and self._job.stamp:
                    out["report"] = self._read_report(self._job.stamp)
            return out

    def _read_report(self, stamp: str) -> dict[str, Any] | None:
        md = RESULTS_DIR / f"verify_champion_rollback_{stamp}.md"
        js = RESULTS_DIR / f"verify_champion_rollback_{stamp}.json"
        # 脚本用本地时间戳命名（manager 的 stamp 是 UTC）→ 精确命中失败时
        # 回退到“本次任务开始后最新生成”的报告文件
        if not md.exists() and self._job and self._job.started_at:
            try:
                from datetime import datetime as _dt_utc

                start = _dt_utc.fromisoformat(self._job.started_at).timestamp()
                cands = [p for p in RESULTS_DIR.glob("verify_champion_rollback_*.md")
                         if p.stat().st_mtime >= start - 2]
                if cands:
                    md = max(cands, key=lambda p: p.stat().st_mtime)
                    js = md.with_suffix(".json")
            except (OSError, ValueError):
                pass
        try:
            return {
                "stamp": stamp,
                "markdown": md.read_text(encoding="utf-8") if md.exists() else "",
                "json_exists": js.exists(),
            }
        except OSError:
            return None

    # ── 启动 ────────────────────────────────────────────────────────
    def start(self, strategy_file: str, retrain_steps: int = 0,
              seed: int = 42, restore_missing: bool = False) -> VerifyJob:
        with self._lock:
            self._refresh()
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("已有回滚校验任务在运行")

            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"verify_rollback_{stamp}.log"
            sf_p = Path(strategy_file)
            if not sf_p.is_absolute():
                sf_p = PROJECT_ROOT / sf_p

            cmd = [sys.executable, "-u", "scripts/verify_champion_rollback.py",
                   "--records", str(sf_p.resolve()), "--seed", str(seed)]
            if retrain_steps and retrain_steps > 0:
                cmd += ["--retrain-steps", str(int(retrain_steps))]
            if restore_missing:
                cmd += ["--no-restore"]

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            self._proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, stdout=self._log_fp,
                stderr=subprocess.STDOUT, env=env,
            )
            self._job = VerifyJob(
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                stamp=stamp,
                started_at=datetime.now(timezone.utc).isoformat(),
                strategy_file=str(sf_p.resolve()),
                retrain_steps=int(retrain_steps or 0),
            )
            return self._job

    def tail_log(self, max_lines: int = 80) -> list[str]:
        if not self._job or not self._job.log_path:
            return []
        p = PROJECT_ROOT / self._job.log_path
        if not p.exists():
            return []
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            return [strip_ansi(l) for l in lines[-max_lines:]]
        except OSError:
            return []


# 模块级单例（app.py 持有引用）
_manager: VerifyRollbackManager | None = None


def get_manager() -> VerifyRollbackManager:
    global _manager
    if _manager is None:
        _manager = VerifyRollbackManager()
    return _manager