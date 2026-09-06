"""File logging for the training web UI.

日志策略:
- 运行日志 logs/web_server.log:INFO 及以上**常驻落盘**(不需要开调试模式),
  轮转 5MB × 5 份, 便于事后排查问题; 调试模式额外放行 DEBUG 并把控制台提到 INFO。
- 错误日志 logs/web_errors.log:log_error() 写入带完整 traceback 的错误块,
  轮转 2MB × 3 份。
- read_logs()/debug_snapshot(): 供 /api/debug/logs 读取, 支持按级别/关键词过滤。
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

SERVER_LOG = LOG_DIR / "web_server.log"
ERROR_LOG = LOG_DIR / "web_errors.log"

SERVER_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB
SERVER_LOG_BACKUPS = 5
ERROR_LOG_MAX_BYTES = 2 * 1024 * 1024  # 2MB
ERROR_LOG_BACKUPS = 3

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_logger: logging.Logger | None = None
_debug_mode: bool = False


def is_debug_mode() -> bool:
    return _debug_mode


def set_debug_mode(enabled: bool) -> None:
    """Toggle verbose logging. 关闭(默认): 文件 INFO / 控制台 WARNING; 开启: 文件 DEBUG / 控制台 INFO。"""
    global _debug_mode
    _debug_mode = bool(enabled)
    logger = get_logger()
    file_level = logging.DEBUG if _debug_mode else logging.INFO
    stream_level = logging.INFO if _debug_mode else logging.WARNING
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(file_level)
        else:
            handler.setLevel(stream_level)
    if _debug_mode:
        logger.info("Debug mode enabled")


def apply_debug_mode_from_settings() -> bool:
    from web.settings import load_settings

    enabled = bool(load_settings().get("debug_mode", False))
    set_debug_mode(enabled)
    return enabled


def setup_logging() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("alphamaster.web")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        SERVER_LOG,
        maxBytes=SERVER_LOG_MAX_BYTES,
        backupCount=SERVER_LOG_BACKUPS,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    _logger = logger
    apply_debug_mode_from_settings()
    return logger


def get_logger() -> logging.Logger:
    return _logger or setup_logging()


def _roll_error_log() -> None:
    """web_errors.log 超过阈值时轮转 (web_errors.1/.2/.3, 保留 3 份)。"""
    if not ERROR_LOG.exists():
        return
    try:
        if ERROR_LOG.stat().st_size < ERROR_LOG_MAX_BYTES:
            return
    except OSError:
        return
    for i in range(ERROR_LOG_BACKUPS - 1, 0, -1):
        src = ERROR_LOG.with_suffix(f".{i}")
        dst = ERROR_LOG.with_suffix(f".{i + 1}")
        try:
            if dst.exists():
                dst.unlink()
            if src.exists():
                src.rename(dst)
        except OSError:
            pass
    try:
        dst1 = ERROR_LOG.with_suffix(".1")
        if dst1.exists():
            dst1.unlink()
        ERROR_LOG.rename(dst1)
    except OSError:
        pass


def log_error(message: str, exc: BaseException | None = None) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"[{ts}] [ERROR] {message}"]
    if exc is not None:
        lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    block = "\n".join(lines) + "\n"

    _roll_error_log()
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(block)
    except OSError:
        pass

    get_logger().error(message, exc_info=exc)


def tail_file(path: Path, lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return content.splitlines()[-lines:]


def _matches(line: str, level: str | None, search: str | None) -> bool:
    if level:
        tag = f"[{level.upper()}]"
        if tag not in line:
            return False
    if search and search.lower() not in line.lower():
        return False
    return True


def read_logs(
    lines: int = 200,
    level: str | None = None,
    search: str | None = None,
) -> dict:
    """读取两个日志文件的尾部, 支持级别 ([ERROR] 前缀) 与关键词(大小写不敏感)过滤。

    过滤在更大的原始窗口上执行, 保证返回数量接近请求的 lines。
    """
    raw_level = (level or "").strip().lower() or None
    if raw_level not in _LEVELS:
        raw_level = None
    raw_search = (search or "").strip() or None

    def tail_filtered(path: Path) -> list[str]:
        window = max(lines * 4, 200)
        raw = tail_file(path, window)
        kept = [ln for ln in raw if _matches(ln, raw_level, raw_search)]
        return kept[-lines:]

    return {
        "debug_mode": is_debug_mode(),
        "server_log": str(SERVER_LOG),
        "error_log": str(ERROR_LOG),
        "server_tail": tail_filtered(SERVER_LOG),
        "error_tail": tail_filtered(ERROR_LOG),
    }


def debug_snapshot(lines: int = 200, level: str | None = None, search: str | None = None) -> dict:
    return read_logs(lines, level, search)