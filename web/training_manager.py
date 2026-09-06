"""Subprocess manager for train_file.py jobs."""
from __future__ import annotations

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

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.train_logging import strip_ansi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainingJob:
    data_file: str
    symbol: str
    timeframe: str
    mode: str
    state: JobState = JobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_file": self.data_file,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "state": self.state.value,
            "pid": self.pid,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
        }


class TrainingManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: TrainingJob | None = None
        self._log_fp = None
        self._stopped_by_user = False
        self._recorded_log_paths: set[str] = set()
        # 训练巡检（内置 agent，无 API Key）：不定时读取日志输出产出诊断
        self._inspections: list[dict] = []
        self._inspector: threading.Thread | None = None
        self._inspector_stop = threading.Event()
        self._last_inspect_step = -1
        # 巡检结论回溯：champion_history.json 实时追加；training_history 终态 flush
        self._champion_path = PROJECT_ROOT / "strategies" / "champion_history.json"
        self._history_flushed = False
        # 「建议停止」自动停训：每轮只触发一次，避免重复 terminate/飞书轰炸
        self._auto_stop_done = False
        # ETA 估算：采集 (wall 时钟, step) 样本，按近期每步耗时外推
        self._eta_samples: list[tuple[float, int]] = []

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            return {
                "active": self._job is not None and self._job.state == JobState.RUNNING,
                "job": self._job.to_dict() if self._job else None,
            }

    def eta_status(self) -> dict[str, Any] | None:
        """status 用的 ETA 概要：补上 train_steps（来自 progress 口径）。"""
        eta = self.eta()
        if not eta:
            return None
        job = self._job
        if job and job.symbol:
            try:
                from web.progress import get_symbol_progress

                eta["train_steps"] = get_symbol_progress(job.symbol).train_steps
                remaining = max(0, eta["train_steps"] - eta["current_step"])
                pace = eta["seconds_per_step"] or 0
                eta["remaining_steps"] = remaining
                eta["remaining_seconds"] = round(remaining * pace)
                if pace > 0:
                    finish = datetime.now(timezone.utc).timestamp() + remaining * pace
                    eta["estimated_finish_local"] = datetime.fromtimestamp(
                        finish, tz=timezone.utc
                    ).astimezone().strftime("%Y-%m-%d %H:%M")
            except Exception:  # noqa: BLE001 ETA 已有基础值，train_steps 补不上就不补
                pass
        return eta

    def start(
        self,
        data_file: str,
        symbol: str,
        timeframe: str,
        mode: str = "ftmo",
        *,
        from_scratch: bool = False,
    ) -> TrainingJob:
        with self._lock:
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                sym = self._job.symbol if self._job else "unknown"
                raise RuntimeError(f"已有训练任务在运行: {sym}")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_sym = symbol.replace(".", "_")
            log_path = LOG_DIR / f"train_{safe_sym}_{ts}.log"

            hist_path = PROJECT_ROOT / f"training_history_{symbol}.json"
            try:
                hist_path.unlink(missing_ok=True)
            except OSError:
                pass

            cmd = [
                sys.executable,
                "-u",
                "train_file.py",
                "--data-file",
                data_file,
            ]
            if from_scratch:
                cmd.append("--from-scratch")

            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["LOGURU_COLORIZE"] = "0"

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._stopped_by_user = False
            self._proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            self._job = TrainingJob(
                data_file=data_file,
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            # 训练巡检守护：开始新一轮时清空旧条目
            self._inspections = []
            self._last_inspect_step = -1
            self._history_flushed = False
            self._auto_stop_done = False
            self._inspector_stop.clear()
            self._inspector = threading.Thread(
                target=self._inspector_loop, daemon=True, name="train-inspector"
            )
            self._inspector.start()
            return self._job

    def stop(self) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
            self._stopped_by_user = True
            try:
                self._proc.terminate()
            except Exception:
                self._proc.kill()
            return True

    # ── 训练巡检（内置 agent）───────────────────────────────────────────

    def inspections(self) -> list[dict]:
        with self._lock:
            return list(self._inspections)

    def _inspector_loop(self) -> None:
        import random

        from web.train_inspector import inspect_log_tail, read_champion_score

        strategies_dir = PROJECT_ROOT / "strategies"
        while not self._inspector_stop.is_set():
            # 不定时：基础间隔 + 抖动，避免与训练批次/UI 轮询完全同频
            wait = 25.0 + random.uniform(0, 20.0)
            self._inspector_stop.wait(wait)
            if self._inspector_stop.is_set():
                break
            with self._lock:
                proc = self._proc
                job = self._job
                if proc is None or job is None:
                    return
                alive = proc.poll() is None
                sym = job.symbol
            try:
                entry = self._run_inspection(strategies_dir=strategies_dir, symbol=sym)
            except Exception:  # noqa: BLE001 巡检失败不能拖垮训练
                entry = None
            if entry:
                self._append_inspection(entry, force=False)
            # ETA 采样：借巡检节拍记录 (wall, step)，供回归估算每步耗时
            try:
                self._record_eta_sample()
            except Exception:  # noqa: BLE001
                pass
            # 「建议停止」→ 自动停训 + 飞书（只在自动巡检节拍触发，手动巡检不越权）
            if entry and self._should_auto_stop(entry, alive):
                try:
                    self._auto_stop_for_entry(entry)
                except Exception:  # noqa: BLE001 自动停训失败不拖垮巡检线程
                    pass
            if not alive:
                # 进程已结束：追加一条“结束”巡检（若与上一条同 step，仍强制记录收尾状态）
                if entry and entry.get("metrics", {}).get("step"):
                    entry["title"] = f"训练进程已结束 · {entry['title']}"
                    self._append_inspection(entry, force=True)
                self._flush_terminal_history()
                return

    def _run_inspection(self, *, strategies_dir: Path, symbol: str | None = None) -> dict | None:
        from web.train_inspector import inspect_log_tail, read_champion_score

        lines = self.tail_log(300)
        if not lines:
            return None
        champ = read_champion_score(symbol or "", strategies_dir) if symbol else None
        entry = inspect_log_tail(lines, champion_score=champ, symbol=symbol)
        if entry and symbol:
            self._attach_formula_reading(entry, strategies_dir, symbol)
        return entry

    def _attach_formula_reading(self, entry: dict, strategies_dir: Path, symbol: str) -> None:
        """给巡检条目附上最新公式解读（解码 token + 算子组合人话）。

        读取失败/模型词表不可用都静默降级（巡检不能因解读问题失败）。
        """
        try:
            from web.formula_read import reading_for_symbol

            reading = reading_for_symbol(strategies_dir, symbol)
            if reading:
                entry["formula"] = reading
        except Exception:  # noqa: BLE001
            pass

    def _append_inspection(self, entry: dict, *, force: bool) -> None:
        step = entry.get("metrics", {}).get("step")
        with self._lock:
            if not force and step is not None and step - self._last_inspect_step < 25:
                return  # 步进太小不重复产报（自动巡检保持低频）
            if step is not None:
                self._last_inspect_step = int(step)
            self._inspections.append(entry)
            if len(self._inspections) > 30:
                self._inspections = self._inspections[-30:]
        # 结论落盘（champion_history 实时追加；引擎已退出时另写 training_history）
        try:
            from web.train_inspector import append_champion_inspect_event

            append_champion_inspect_event(self._champion_path, entry)
        except Exception:  # noqa: BLE001 巡检持久化失败不影响训练
            pass
        if not self._engine_alive_locked():
            self._flush_terminal_history()

    def _engine_alive_locked(self) -> bool:
        """引擎是否还活着（巡检线程/终态 flush 判断用，避免反复读日志）。"""
        with self._lock:
            proc = self._proc
            if proc is None:
                return False
            return proc.poll() is None

    def _flush_terminal_history(self) -> None:
        """终态把整轮巡检结论 flush 进 training_history_{symbol}.json。

        只在引擎退出后执行一次（引擎逐 step 整文件覆盖，活着时写会被顶掉）；
        与引擎最后写 holdout/曲线的时间错开，避免同文件双写。
        """
        with self._lock:
            if self._history_flushed:
                return
            job = self._job
            if job is None or not job.symbol:
                self._history_flushed = True
                return
            entries = list(self._inspections)
            symbol = job.symbol
        if not entries:
            with self._lock:
                self._history_flushed = True
            return
        try:
            from web.train_inspector import merge_inspections_into_history

            history_path = PROJECT_ROOT / f"training_history_{symbol}.json"
            merge_inspections_into_history(history_path, entries)
        except Exception:  # noqa: BLE001 写失败只影响回溯，不阻断
            pass
        with self._lock:
            self._history_flushed = True

    def _should_auto_stop(self, entry: dict | None, alive_snapshot: bool) -> bool:
        """巡检结论是否触发自动停训：危险级「建议停止」+ 进程还在 + 本轮未触发过。"""
        if not entry or not alive_snapshot:
            return False
        if self._auto_stop_done:
            return False
        from web.train_inspector import verdict_recommends_stop

        return verdict_recommends_stop(entry)

    def _auto_stop_for_entry(self, entry: dict) -> None:
        """按巡检建议自动停训：先飞书推结论，再 terminate，最后留一条停止记录。"""
        with self._lock:
            if self._auto_stop_done:
                return
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                return  # 进程已结束，无需再停（不标记 done，避免误报）
            self._auto_stop_done = True
            job = self._job
            symbol = job.symbol if job else entry.get("symbol")
        # 1) 飞书推送结论（未配 webhook 时优雅降级 + 落日志）
        self._notify_auto_stop(entry, symbol)

        # 2) 追加一条“已自动停止”巡检记录（force 保留）
        try:
            from datetime import datetime as _dt2, timezone as _tz2

            note = {
                "ts": _dt2.now(_tz2.utc).isoformat(),
                "level": "danger",
                "title": f"已自动停止：{entry.get('title') or '建议停止'}",
                "checks": entry.get("checks") or [],
                "recommendation": "训练已被巡检自动停止；可在修复停滞/塌缩后重开。",
                "metrics": entry.get("metrics") or {},
                "symbol": symbol,
                "auto_stopped": True,
            }
            self._append_inspection(note, force=True)
        except Exception:  # noqa: BLE001
            pass
        # 3) 停掉引擎（与用户手动点停同理）。
        #    注意不要立刻 flush training_history：terminate 是异步的，引擎可能仍在
        #    写曲线，此刻整文件覆盖会把我们刚加的 inspections 顶掉；下一拍巡检线程
        #    看到 alive=False 后会追加“结束”条目并终态 flush（≤45s 内完成）。
        self.stop()

    def _notify_auto_stop(self, entry: dict, symbol: str | None) -> None:
        """飞书推送自动停止结论；降级：macOS 本地通知 + logs 日志。"""
        from web.train_inspector import build_stop_notice

        text = build_stop_notice(entry, symbol) + "\n（已按巡检建议自动停止训练）"
        log_path = PROJECT_ROOT / "logs" / "train_autostop.log"
        try:
            import web.feishu_notify as fn

            ok, msg = fn.send_text(text)
            line = f"[auto-stop] ok={ok} msg={msg}\n{text}\n---\n"
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, str(exc)
            line = f"[auto-stop] feishu异常 {exc}\n{text}\n---\n"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(line)
        except OSError:
            pass
        if not ok:
            try:
                import web.local_notify as ln

                ln.notify("AlphaMaster · 训练自动停止", text[:400])
            except Exception:  # noqa: BLE001
                pass

    def _record_eta_sample(self) -> None:
        """记录一个 (wall_ts, step) 样本（由巡检线程按 25–45s 节拍调用）。"""
        import time as _time

        step = self.parse_step_from_log()
        if step is None:
            return
        with self._lock:
            samples = self._eta_samples
            # 只保留有推进的样本（重复步会稀释回归斜率；步回退说明换了新会话）
            if samples and step <= samples[-1][1]:
                return
            samples.append((_time.time(), step))
            if len(samples) > 240:  # ~2.5 小时 @40s 采样，足够近期窗口
                self._eta_samples = samples[-240:]

    def eta(self) -> dict[str, Any] | None:
        """训练 ETA：近期每步耗时回归 → 剩余步数 / 剩余秒 / 预计结束（本地时刻）。"""
        with self._lock:
            self._refresh_state()
            job = self._job
            active = job is not None and job.state == JobState.RUNNING
            samples = list(self._eta_samples)
            started_at = job.started_at if job else None
        if not active:
            return None
        step = self.parse_step_from_log()
        if step is None:
            return None
        elapsed = None
        if started_at:
            try:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
            except (ValueError, TypeError):
                elapsed = None
        from web.training_time import get_training_time_summary

        hist = None
        try:
            sym = job.symbol if job else None
            if sym:
                hist = get_training_time_summary(sym, job=None, active=False).history_total_seconds
        except Exception:  # noqa: BLE001
            hist = None
        from web.train_eta import eta_from_samples

        return eta_from_samples(
            step=step,
            elapsed_seconds=elapsed,
            train_steps=None,
            samples=samples,
            history_total_seconds=hist,
        )

    def inspect_now(self) -> dict:
        """手动「立即巡检一次」：无任务时返回说明条目。"""
        from datetime import datetime as _dt, timezone as _tz

        with self._lock:
            self._refresh_state()
            job = self._job
            proc = self._proc
        if job is None or proc is None:
            return {
                "ts": _dt.now(_tz.utc).isoformat(),
                "level": "info",
                "title": "当前没有训练任务",
                "checks": [],
                "recommendation": "开始一次训练后，内置巡检会自动不定期检查日志输出并给出诊断，无需 API Key。",
                "metrics": {},
            }
        entry = self._run_inspection(strategies_dir=PROJECT_ROOT / "strategies", symbol=job.symbol)
        if entry is None:
            return {
                "ts": _dt.now(_tz.utc).isoformat(),
                "level": "info",
                "title": f"暂无可解析的进度输出（{job.symbol}）",
                "checks": [],
                "recommendation": "训练刚启动，日志尚无进度行；稍候再试。",
                "metrics": {},
            }
        self._append_inspection(entry, force=True)
        return entry

    def parse_step_from_log(self) -> int | None:
        """从日志尾部解析当前步数，用于 checkpoint 写入前的进度展示。"""
        import re

        for line in reversed(self.tail_log(80)):
            m = re.search(r"\[(\d+)/(\d+)\]", line)
            if m:
                return int(m.group(1))
        return None

    def tail_log(self, lines: int = 200) -> list[str]:
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
        if self._job.state == JobState.RUNNING:
            if self._stopped_by_user:
                self._job.state = JobState.STOPPED
            elif code == 0:
                self._job.state = JobState.COMPLETED
            elif code in (-signal.SIGTERM, 1) and sys.platform != "win32":
                self._job.state = JobState.STOPPED
            elif code < 0:
                self._job.state = JobState.STOPPED
            else:
                self._job.state = JobState.FAILED
        if self._job.state == JobState.FAILED and self._job.error is None:
            self._job.error = f"训练进程异常退出 (exit_code={code})"
            try:
                if self._job.log_path:
                    path = PROJECT_ROOT / self._job.log_path
                    with path.open("a", encoding="utf-8") as fp:
                        fp.write(f"\n[Web] 训练进程已结束，退出码: {code}\n")
            except OSError:
                pass
        if self._log_fp:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        self._record_session_time()
        self._proc = None

    def _record_session_time(self) -> None:
        job = self._job
        if job is None or not job.log_path or not job.started_at:
            return
        rel = job.log_path.replace("\\", "/")
        if rel in self._recorded_log_paths:
            return
        if job.state == JobState.RUNNING:
            return
        from web.training_time import record_training_session

        record_training_session(
            symbol=job.symbol,
            started_at=job.started_at,
            finished_at=job.finished_at,
            log_path=rel,
        )
        self._recorded_log_paths.add(rel)


training_manager = TrainingManager()
