"""web/hold_matrix_manager.py — 持仓管理 N×N 全组合矩阵的子进程管理。

镜像 backtest_manager 的设计：把 scripts/hold_matrix.py（全组合回放，
写 results/hold_matrix_latest.json + 带时间戳文件）放进子进程执行，
stdout 写入 logs/hold_matrix_web_*.log；前端轮询 /api/backtest/hold-matrix/status
拿状态与进度（按日志里「已跑完的组合行数 / 预期总组合数」估算）。
"""
from __future__ import annotations

import os
import re
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

# hold_matrix.py 的策略列表与“去重后要跑的组合数”保持一致
# （7 个方案的上三角去重组合 = 22；若未来注册表变动需同步）
_POLICY_IDS = ["signal", "risk", "hybrid", "be", "time", "chandelier", "dd"]
_TOTAL_COMBOS = 22

# 日志里一行一个组合的输出形如：  risk  收益 +27.16%  夏普 +2.69  交易 559
_COMBO_LINE_RE = re.compile(r"^\s{2}\S+(\+\S+)?\s+收益\s")


class MatrixJobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class MatrixJob:
    state: MatrixJobState = MatrixJobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    strategy_file: str | None = None
    data_file: str | None = None
    window_bars: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "pid": self.pid,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "strategy_file": self.strategy_file,
            "data_file": self.data_file,
            "window_bars": self.window_bars,
        }


class HoldMatrixManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: MatrixJob | None = None
        self._log_fp = None
        self._stopped_by_user = False

    # ── 状态 ────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            job_dict = self._job.to_dict() if self._job else None
        tail = self.tail_log(60)
        done = 0
        for line in tail:
            if _COMBO_LINE_RE.match(line):
                done += 1
        return {
            "active": self._job is not None and self._job.state == MatrixJobState.RUNNING,
            "job": job_dict,
            "combos_done": done,
            "combos_total": _TOTAL_COMBOS,
            "log_tail": tail,
        }

    def start(
        self,
        strategy_file: str | None = None,
        data_file: str | None = None,
        commission_pct: float = 0.02,
        slippage_pct: float = 0.01,
        window_bars: int | None = None,
        window_mode: str = "tail",
        regime: str = "vol",
        chunks: int = 4,
        max_position_pct: float = 100.0,
        signal_threshold: float | None = None,
    ) -> MatrixJob:
        with self._lock:
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("已有全组合矩阵任务在运行")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"hold_matrix_web_{ts}.log"

            cmd = [sys.executable, "-u", "scripts/hold_matrix.py"]
            if strategy_file:
                cmd += ["--strategy-file", strategy_file]
            if data_file:
                cmd += ["--data-file", data_file]
            cmd += ["--commission", str(commission_pct), "--slippage", str(slippage_pct)]
            cmd += ["--max-position-pct", str(min(200.0, max(1.0, float(max_position_pct))))]
            if signal_threshold is not None:
                cmd += ["--threshold", str(signal_threshold)]
            if window_bars:
                if int(window_bars) < 800:
                    raise ValueError("样本外窗口需 ≥800 根（特征 warm-up）")
                cmd += ["--window-bars", str(int(window_bars))]
                mode = (window_mode or "tail").strip().lower()
                if mode not in ("tail", "spread"):
                    raise ValueError(f"未知窗口模式: {mode}")
                cmd += ["--window-mode", mode]
                if mode == "spread":
                    cmd += ["--regime", (regime or "vol").strip().lower()]
                    cmd += ["--chunks", str(max(2, int(chunks or 4)))]

            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["LOGURU_COLORIZE"] = "0"
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

            self._stopped_by_user = False
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            self._job = MatrixJob(
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
                strategy_file=strategy_file,
                data_file=data_file,
                window_bars=int(window_bars) if window_bars else None,
            )
            return self._job

    def stop(self) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
            self._stopped_by_user = True
            try:
                self._proc.terminate()
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            return True

    def tail_log(self, lines: int = 60) -> list[str]:
        with self._lock:
            if not self._job or not self._job.log_path:
                return []
            path = PROJECT_ROOT / self._job.log_path
            if not path.exists():
                return []
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return []
            return [strip_ansi(line) for line in content.splitlines()[-lines:]]

    def _refresh_state(self) -> None:
        if self._proc is None or self._job is None:
            return
        code = self._proc.poll()
        if code is None:
            return
        self._job.exit_code = code
        self._job.finished_at = datetime.now(timezone.utc).isoformat()
        was_running = self._job.state == MatrixJobState.RUNNING
        if was_running:
            if self._stopped_by_user:
                self._job.state = MatrixJobState.STOPPED
            elif code == 0:
                self._job.state = MatrixJobState.COMPLETED
            elif code < 0:
                self._job.state = MatrixJobState.STOPPED
            else:
                self._job.state = MatrixJobState.FAILED
        # 完成瞬间推一次飞书摘要（避免每次 status 轮询重复推送）
        if was_running and self._job.state == MatrixJobState.COMPLETED:
            self._notify_done()
        if self._job.state == MatrixJobState.FAILED and self._job.error is None:
            self._job.error = f"全组合矩阵进程异常退出 (exit_code={code})"
            try:
                if self._job.log_path:
                    path = PROJECT_ROOT / self._job.log_path
                    with path.open("a", encoding="utf-8") as fp:
                        fp.write(f"\n[Web] 全组合矩阵进程已结束，退出码: {code}\n")
            except OSError:
                pass
        if self._log_fp:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        self._proc = None

    # ── 完成通知：飞书优先，未启用/未配 webhook 时降级到 macOS 本地通知 ──
    def _notify_done(self) -> None:
        notify_log = PROJECT_ROOT / "logs" / "hold_matrix_notify.log"
        try:
            import json

            import web.feishu_notify as fn

            path = PROJECT_ROOT / "results" / "hold_matrix_latest.json"
            if not path.exists():
                self._log_notify(notify_log, "✗ 无 hold_matrix_latest.json")
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            ok, msg = fn.notify_hold_matrix_done(matrix=data)
            if ok:
                self._log_notify(notify_log, "✓ 组合矩阵完成摘要已推送（top5 vs 基线）")
            else:
                self._log_notify(notify_log, f"✗ 组合矩阵完成摘要: {msg} → macOS 本地通知已尝试")
                self._notify_macos(data, reason=msg)
        except Exception as exc:  # noqa: BLE001 通知失败绝不影响任务状态
            self._log_notify(notify_log, f"✗ 组合矩阵摘要异常: {exc}")
            self._notify_macos(None, reason=f"异常: {exc}")

    @staticmethod
    def _notify_macos(matrix: dict | None, reason: str = "") -> None:
        """macOS 本地通知降级通道（飞书不可用时避免整条链路静默失效）。"""
        if sys.platform != "darwin":
            return
        ranking = (matrix or {}).get("ranking") or []
        top = ranking[0] if ranking else {}
        ret = top.get("total_return")
        ret_s = f"{ret * 100:+.2f}%" if isinstance(ret, (int, float)) else "—"
        win = (matrix or {}).get("window_mode") or "全部历史"
        text = (f"全组合回测完成 · top {top.get('combo') or '—'} 收益 {ret_s} · "
                f"窗口 {win} · 详情见 results/hold_matrix_latest.json")
        if reason:
            text += f"（飞书: {reason}）"
        # 统一通知模块：osascript 投递 + 权限诊断 + logs/notifications.log 兜底
        try:
            from web.local_notify import notify as local_notify

            res = local_notify("AlphaMaster 矩阵完成", text)
            if not res.get("posted") and res.get("hint"):
                print(f"[本地通知被拦截] {res.get('hint')}", flush=True)
        except Exception:  # noqa: BLE001 本地通知失败也不影响任务
            pass

    @staticmethod
    def _log_notify(notify_log: Path, msg: str) -> None:
        try:
            with notify_log.open("a", encoding="utf-8") as fp:
                fp.write(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        except OSError:
            pass
        print(f"[飞书通知] {msg}", flush=True)
