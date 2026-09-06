"""范围对比实验（tail vs spread）子进程管理器。

与训练/回测管理器一致：把 scripts/compare_ranges.py 作为子进程跑（串行执行两
个变体短训），日志写 logs/compare_*.log，完成后读 results/compare_latest.json
返回给前端。单实例：同一时间只允许一个对比实验。
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RESULT_PATH = PROJECT_ROOT / "results" / "compare_latest.json"


class CmpState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class CompareJob:
    params: dict[str, Any]
    state: CmpState = CmpState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "params": self.params,
            "pid": self.pid,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "error": self.error,
        }


class CompareManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: CompareJob | None = None
        self._log_fp = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh()
            out = {"active": False, "job": None, "log_tail": [], "result": None,
                   "from_file": False}
            if self._job:
                out["active"] = self._job.state == CmpState.RUNNING
                out["job"] = self._job.to_dict()
                out["log_tail"] = self.tail_log(120)
                out["result"] = self._read_result()
            else:
                # 无运行中任务时，展示最近一次（含 CLI/后台）实验的 results/compare_latest.json
                res = self._read_result()
                if res:
                    out["result"] = res
                    out["from_file"] = True
                    out["log_tail"] = self._tail_result_log(res)
            return out

    def _tail_result_log(self, res: dict[str, Any]) -> list[str]:
        """结果 JSON 自带路径时回显日志尾（无则空）。"""
        try:
            lp = res.get("log_path") or res.get("log")
            if lp:
                p = PROJECT_ROOT / lp
                if p.exists():
                    return p.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        except OSError:
            pass
        return []

    def _read_result(self) -> dict[str, Any] | None:
        try:
            if RESULT_PATH.exists():
                data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            pass
        return None

    def _refresh(self) -> None:
        if self._job and self._job.state == CmpState.RUNNING:
            if self._proc is None or self._proc.poll() is not None:
                code = self._proc.poll() if self._proc is not None else 0
                self._job.state = CmpState.FAILED if code else CmpState.DONE
                if code and self._log_fp:
                    pass  # 错误细节见日志
                self._job.error = None if code == 0 else f"exit={code}，详见日志"
                self._proc = None

    def start(self, data_file: str, n_bars: int, chunks: int | None,
              steps: int, seeds: list[int], regime: str = "vol",
              rep_criterion: str | None = None) -> CompareJob:
        with self._lock:
            self._refresh()
            if self._job and self._job.state == CmpState.RUNNING:
                raise RuntimeError("已有对比实验在运行")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe = Path(data_file).stem.replace(".", "_")
            log_path = LOG_DIR / f"compare_{safe}_{ts}.log"
            result_json = Path(os.environ.get("ALPHA_RESULTS_DIR", str(PROJECT_ROOT / "results"))) / "compare_latest.json"

            cmd = [
                sys.executable, "-u", "scripts/compare_ranges.py",
                "--data-file", data_file,
                "--n-bars", str(n_bars),
                "--steps", str(steps),
                "--seeds", ",".join(str(s) for s in seeds),
                "--regime", regime,
                "--out-dir", "results",
            ]
            if rep_criterion:
                cmd += ["--rep-criterion", rep_criterion]
            if chunks:
                cmd += ["--chunks", str(chunks)]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            self._proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, stdout=self._log_fp,
                stderr=subprocess.STDOUT, env=env,
            )
            self._job = CompareJob(
                params={"data_file": data_file, "n_bars": n_bars,
                        "chunks": chunks, "steps": steps, "seeds": seeds,
                        "regime": regime, "rep_criterion": rep_criterion},
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            # 结果文件由子进程收尾写入 results/compare_latest.json
            if result_json.exists():
                try:
                    result_json.unlink()
                except OSError:
                    pass
            return self._job

    def stop(self) -> bool:
        with self._lock:
            self._refresh()
            if self._proc is None or self._proc.poll() is not None:
                return False
            try:
                self._proc.send_signal(signal.SIGTERM)
                return True
            except OSError:
                return False

    def tail_log(self, max_lines: int = 120) -> list[str]:
        if not self._job or not self._job.log_path:
            return []
        p = PROJECT_ROOT / self._job.log_path
        if not p.exists():
            return []
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[-max_lines:]
        except OSError:
            return []


compare_manager = CompareManager()
