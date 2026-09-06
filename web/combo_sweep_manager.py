"""web/combo_sweep_manager.py — 三轴联合回测（持仓方案 × 上限% × 阈值）子进程管理。

镜像 hold_matrix_manager 的设计：把 scripts/combo_sweep.py（全网格回放，写
results/combo_sweep_latest.json + hold_matrix_latest.json 基线切片）放进子进程执行，
stdout 写入 logs/combo_sweep_web_*.log；前端轮询 /api/backtest/combo-sweep/status
拿状态与进度（按日志里「已跑完的行数 / 预期总行数」估算，总行数 = 组合数 × 上限档 × 阈值档）。
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

# 与 scripts/hold_matrix.py 的 POLICIES 上三角去重组合数保持一致（7 方案 = 22 组合）
N_COMBOS = 22

# 日志里一行一个回放：  dd+be cap=25% t=0.30 收益 ... 夏普 ... 交易 ...
# 注意组合名按 24 宽右对齐补空格，pid 与 cap= 之间是 1+ 个空格，用 \s+ 匹配
_COMBO_LINE_RE = re.compile(r"^\s{2}\S+\s+cap=")


class ComboSweepJobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class ComboSweepJob:
    state: ComboSweepJobState = ComboSweepJobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    strategy_file: str | None = None
    data_file: str | None = None
    window_bars: int | None = None
    caps: list[float] | None = None
    thresholds: list[float] | None = None

    @property
    def combos_total(self) -> int:
        return N_COMBOS * len(self.caps or []) * len(self.thresholds or [])

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
            "caps": self.caps,
            "thresholds": self.thresholds,
            "combos_total": self.combos_total,
        }


class ComboSweepManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: ComboSweepJob | None = None
        self._log_fp = None
        self._stopped_by_user = False

    # ── 状态 ────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            job_dict = self._job.to_dict() if self._job else None
        # 全量日志统计已完成行数（tail 60 行会低估大网格进度：264 行 > 60）
        done = 0
        for line in self.read_log():  # 日志很小（一行一次回放），整读即可
            if _COMBO_LINE_RE.match(line):
                done += 1
        return {
            "active": self._job is not None and self._job.state == ComboSweepJobState.RUNNING,
            "job": job_dict,
            "combos_done": done,
            "combos_total": self._job.combos_total if self._job else 0,
            "log_tail": self.tail_log(60),
        }

    def read_log(self) -> list[str]:
        """读完整任务日志（去 ANSI）。"""
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
            return [strip_ansi(line) for line in content.splitlines()]

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
        caps: list[float] | None = None,
        thresholds: list[float] | None = None,
    ) -> ComboSweepJob:
        caps = sorted({min(200.0, max(1.0, float(c))) for c in (caps or [100.0]) if c > 0})
        thresholds = sorted({float(t) for t in (thresholds or [0.05]) if 0 < float(t) < 1})
        if not caps:
            caps = [100.0]
        if not thresholds:
            thresholds = [0.05]
        with self._lock:
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("已有三轴联合回测任务在运行")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"combo_sweep_web_{ts}.log"

            cmd = [sys.executable, "-u", "scripts/combo_sweep.py"]
            if strategy_file:
                cmd += ["--strategy-file", strategy_file]
            if data_file:
                cmd += ["--data-file", data_file]
            cmd += ["--commission", str(commission_pct), "--slippage", str(slippage_pct)]
            cmd += ["--caps", ",".join(str(c) for c in caps)]
            cmd += ["--thresholds", ",".join(str(t) for t in thresholds)]
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
            self._job = ComboSweepJob(
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
                strategy_file=strategy_file,
                data_file=data_file,
                window_bars=int(window_bars) if window_bars else None,
                caps=caps,
                thresholds=thresholds,
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
        was_running = self._job.state == ComboSweepJobState.RUNNING
        if was_running:
            if self._stopped_by_user:
                self._job.state = ComboSweepJobState.STOPPED
            elif code == 0:
                self._job.state = ComboSweepJobState.COMPLETED
            elif code < 0:
                self._job.state = ComboSweepJobState.STOPPED
            else:
                self._job.state = ComboSweepJobState.FAILED
        # 完成瞬间推一次飞书摘要（避免每次 status 轮询重复推送）
        if was_running and self._job.state == ComboSweepJobState.COMPLETED:
            self._notify_done()
        if self._job.state == ComboSweepJobState.FAILED and self._job.error is None:
            self._job.error = f"三轴联合回测进程异常退出 (exit_code={code})"
            try:
                if self._job.log_path:
                    path = PROJECT_ROOT / self._job.log_path
                    with path.open("a", encoding="utf-8") as fp:
                        fp.write(f"\n[Web] 三轴联合回测进程已结束，退出码: {code}\n")
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

    # ── 完成通知：飞书优先（读基线切片矩阵摘要），未启用时降级 macOS 本地通知 ──
    def _notify_done(self) -> None:
        notify_log = PROJECT_ROOT / "logs" / "combo_sweep_notify.log"
        try:
            import json

            import web.feishu_notify as fn

            path = PROJECT_ROOT / "results" / "hold_matrix_latest.json"
            grid_path = PROJECT_ROOT / "results" / "combo_sweep_latest.json"
            if not grid_path.exists():
                self._log_notify(notify_log, "✗ 无 combo_sweep_latest.json")
                return
            data = json.loads(grid_path.read_text(encoding="utf-8"))
            matrix = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            ok, msg = fn.notify_hold_matrix_done(matrix=matrix or data)
            if ok:
                self._log_notify(notify_log, "✓ 三轴联合回测完成摘要已推送（top vs 基线切片）")
            else:
                self._log_notify(notify_log, f"✗ 三轴联合回测完成摘要: {msg} → macOS 本地通知已尝试")
                self._notify_macos(data, reason=msg)
        except Exception as exc:  # noqa: BLE001 通知失败绝不影响任务状态
            self._log_notify(notify_log, f"✗ 三轴联合回测摘要异常: {exc}")
            self._notify_macos(None, reason=f"异常: {exc}")

    @staticmethod
    def _notify_macos(grid: dict | None, reason: str = "") -> None:
        """macOS 本地通知降级通道（飞书不可用时避免整条链路静默失效）。"""
        if sys.platform != "darwin":
            return
        best = (grid or {}).get("best_overall") or {}
        ret = best.get("total_return")
        ret_s = f"{ret * 100:+.2f}%" if isinstance(ret, (int, float)) else "—"
        cap = best.get("cap_pct")
        t = best.get("threshold")
        text = (f"三轴联合回测完成 · 最优 {best.get('combo') or '—'} @ "
                f"{cap if cap is not None else '—'}% × t={t if t is not None else '—'} · "
                f"收益 {ret_s} · 详情见 results/combo_sweep_latest.json")
        if reason:
            text += f"（飞书: {reason}）"
        try:
            from web.local_notify import notify as local_notify

            res = local_notify("AlphaMaster 三轴联合回测完成", text)
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


_get_combo_sweep_manager: ComboSweepManager | None = None


def get_manager() -> ComboSweepManager:
    """惰性单例（app.py 使用，与 hold_matrix_manager 同模式）。"""
    global _get_combo_sweep_manager
    if _get_combo_sweep_manager is None:
        _get_combo_sweep_manager = ComboSweepManager()
    return _get_combo_sweep_manager