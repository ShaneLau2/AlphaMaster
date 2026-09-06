"""FastAPI application for AlphaMaster training UI."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import secrets
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.parquet_manager import inspect_parquet_file
from model_core.config import ModelConfig
from web.file_dialog import poll_picker, start_picker
from web.progress import (
    PROJECT_ROOT,
    checkpoint_glob,
    get_symbol_progress,
    get_strategy_for_export,
    invalidate_checkpoint_cache,
    list_strategies,
    build_strategy_export_filename,
)
from web.server_log import (
    debug_snapshot,
    get_logger,
    is_debug_mode,
    log_error,
    set_debug_mode,
    setup_logging,
)
from web.settings import STRATEGIES_DIR, load_settings, save_settings
from web.strategy_file import (
    inspect_strategy_file,
    resolve_strategy_file,
    sync_best_strategy_for_symbol,
)
from web.training_manager import training_manager
from web.training_time import get_training_time_summary
from web.training_package import build_training_export_zip, import_training_package
from web.backtest_manager import backtest_manager
from web.realtime_manager import realtime_manager
from web.data_sources.factory import list_sources
from web.paper_manager import paper_manager
from web.experiment_manager import compare_manager
from strategy_manager.live_signal import min_exposure

STATIC_DIR = Path(__file__).resolve().parent / "static"
BACKTEST_OUTPUT_DIR = ROOT / "backtest_output"

setup_logging()
logger = get_logger()

# ── API 令牌鉴权 ──────────────────────────────────────────────────────────
# 所有 /api/* 请求必须携带令牌 (Authorization: Bearer <token> 或 ?token=),
# 防止公网页面 (GitHub Pages 镜像 / 任意恶意网站) 读取本机控制台数据。
# 令牌首启生成后保存于 web_token.txt (与本机 web_settings.json 同级);
# 前端首次访问 GET /api/auth/token 自动获取并保存在浏览器 localStorage ——
# 该端点按来源放行: 同源(环回)/局域网控制台/已知镜像, 其它来源一律 403。
TOKEN_PATH = ROOT / "web_token.txt"
TRUSTED_MIRROR_ORIGINS = ("https://shanelau2.github.io",)


def _load_or_create_token() -> str:
    try:
        tok = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if len(tok) >= 16:
            return tok
    except OSError:
        pass
    tok = secrets.token_hex(32)
    try:
        TOKEN_PATH.write_text(tok, encoding="utf-8")
        TOKEN_PATH.chmod(0o600)
    except OSError:
        pass
    return tok


API_TOKEN = _load_or_create_token()


def _rotate_token() -> str:
    global API_TOKEN
    API_TOKEN = secrets.token_hex(32)
    try:
        TOKEN_PATH.write_text(API_TOKEN, encoding="utf-8")
        TOKEN_PATH.chmod(0o600)
    except OSError:
        pass
    return API_TOKEN


def _token_matches(supplied: str) -> bool:
    return bool(supplied) and hmac.compare_digest(supplied, API_TOKEN)


def _auth_from(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip()
    # SSE (EventSource) 无法带自定义头, 允许 ?token= 查询参数
    return request.query_params.get("token", "")


def _origin_is_trusted(request: Request, origin: str) -> bool:
    """令牌发放来源白名单: 同源 / 局域网控制台 / 已知镜像。"""
    if not origin:
        return True
    try:
        u = urlsplit(origin)
    except ValueError:
        return False
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    if origin in TRUSTED_MIRROR_ORIGINS:
        return True
    host = u.hostname.lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # 公网域名(除镜像白名单外)一律拒绝
    if not (ip.is_private or ip.is_loopback):
        return False
    # 局域网来源: 端口须与本机后端一致 (恶意站点的 Origin 对不上端口)
    rp, op = request.url.port, u.port
    if rp is None or op is None:
        return True
    return op == rp


logger.info("API token: %s (保存在 %s, 控制台页会自动获取)", API_TOKEN, TOKEN_PATH)

app = FastAPI(title="AlphaMaster Training", version="1.2.0")
# allow_private_network: 应答 Chrome 的 Private Network Access 预检 (公网 https 页
# → 本机 127.0.0.1)。GitHub Pages 镜像页连接本机后端属「公网→环回」私有网络请求,
# Chrome 会先发带 Access-Control-Request-Private-Network 的预检, 需要响应头
# Access-Control-Allow-Private-Network: true 才放行 (Starlette ≥1.4 原生支持)。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)


def _attach_data_meta(file_info: dict[str, Any], data_file: str) -> dict[str, Any]:
    """把下载元信息（数据来源/下载时间）合并到文件信息里，供卡片展示。"""
    try:
        from web.data_download import read_parquet_meta

        meta = read_parquet_meta(data_file)
        if meta:
            file_info = dict(file_info or {})
            file_info["download_source"] = meta.get("source")
            file_info["downloaded_at"] = meta.get("downloaded_at")
            # 旧版 sidecar 无计数字段时回填 1 次（首次日期=下载日），与
            # list_download_history 的推断一致；新 schema 由写入方逐次累计。
            file_info["download_count"] = (
                meta.get("download_count")
                if meta.get("download_count") is not None
                else (1 if meta.get("downloaded_at") else None)
            )
            file_info["first_downloaded_at"] = meta.get("first_downloaded_at") or meta.get("downloaded_at")
    except Exception:  # noqa: BLE001 元信息缺失不影响主流程
        pass
    return file_info


class StartTrainingRequest(BaseModel):
    data_file: str
    from_scratch: bool = False
    # 训练数据范围：full=全部 / tail=最近 n_bars 根 / spread=全历史分块 n_bars 根
    data_mode: str = "full"
    n_bars: int | None = None
    n_chunks: int | None = None


class SubsetPreviewRequest(BaseModel):
    data_file: str
    mode: str = "full"      # full / tail / spread
    n_bars: int | None = None
    n_chunks: int | None = None
    regime: str | None = "vol"  # vol / trend / equal（spread 用）


class DownloadDataRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    source: str = "tradingview"
    n_bars: int | None = None
    mode: str = "merge"  # merge=与已有文件合并追加（默认） / replace=覆盖为最近 n 根


class ClientLogRequest(BaseModel):
    level: str = "error"
    message: str
    context: dict[str, Any] | None = None


class SettingsRequest(BaseModel):
    last_data_file: str | None = None
    last_strategy_file: str | None = None
    debug_mode: bool | None = None
    bg_animation: bool | None = None
    ai_provider: str | None = None
    ai_api_key: str | None = None
    ai_base_url: str | None = None
    ai_model: str | None = None
    bt_commission_pct: float | None = None
    bt_slippage_pct: float | None = None
    bt_hold_policy: str | None = None
    bt_window_bars: int | None = None
    bt_prefs_symbol: str | None = None
    paper_starting_balance: float | None = None
    paper_notional: float | None = None
    paper_commission_pct: float | None = None
    paper_slippage_pct: float | None = None
    max_position_pct: float | None = None
    signal_threshold: float | None = None


class AnalyzeTrainingRequest(BaseModel):
    provider: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    symbol: str | None = None


class StartBacktestRequest(BaseModel):
    strategy_file: str
    data_file: str | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None
    hold_policy: str | None = None
    window_bars: int | None = None
    max_position_pct: float | None = None
    signal_threshold: float | None = None


class HoldMatrixRunRequest(BaseModel):
    strategy_file: str | None = None
    data_file: str | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None
    window_bars: int | None = None
    max_position_pct: float | None = None
    signal_threshold: float | None = None
    window_mode: str = "tail"
    regime: str = "vol"
    chunks: int = 4


class ComboSweepRunRequest(BaseModel):
    """三轴联合回测：持仓方案（22 正交组合）× 每笔投入上限% × 无信号阈值 全网格。"""
    strategy_file: str | None = None
    data_file: str | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None
    window_bars: int | None = None
    window_mode: str = "tail"
    regime: str = "vol"
    chunks: int = 4
    # 上限% 档位（如 [10,25,100]）；缺省回退 [10,25,100]
    caps: list[float] | None = None
    # 无信号阈值档位（如 [0.05,0.3,0.5,0.8]）；缺省回退四档
    thresholds: list[float] | None = None


class HoldMatrixPrefsRequest(BaseModel):
    symbol: str
    data_file: str | None = None
    window_bars: int | None = None
    window_mode: str = "tail"
    regime: str = "vol"
    chunks: int = 4


class StartCompareRequest(BaseModel):
    data_file: str
    n_bars: int = 60_000
    chunks: int | None = None
    steps: int = 12
    seeds: str = "42"
    regime: str = "vol"
    window_bars: int | None = None
    rep_criterion: str | None = None
    # 保证与 compare_ranges 的 choices 一致：in_sample / holdout_median / holdout_best


class AddWatchRequest(BaseModel):
    source: str
    symbol: str
    timeframe: str
    strategy_file: str
    policy_id: str | None = None


class RemoveWatchRequest(BaseModel):
    id: str


class FeishuSettingsRequest(BaseModel):
    enabled: bool | None = None
    webhook_url: str | None = None
    secret: str | None = None
    rt_alert_dev_pct: float | None = None  # 偏离入场参考告警阈值 %（0=关闭）
    rt_alert_stale_bars: int | None = None  # 因子硬钝化告警阈值（连续根数，0=关闭）


class FeishuTestRequest(BaseModel):
    webhook_url: str | None = None
    secret: str | None = None


class PaperWatchRequest(BaseModel):
    source: str
    symbol: str
    timeframe: str
    strategy_file: str
    policy_id: str | None = None


class PaperActionRequest(BaseModel):
    id: str


class PaperReplayRequest(BaseModel):
    data_file: str
    strategy_file: str
    policy_id: str | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None
    window_bars: int | None = None
    max_position_pct: float | None = None
    signal_threshold: float | None = None


class PaperResetRequest(BaseModel):
    starting_balance: float | None = None


@app.middleware("http")
async def api_token_middleware(request: Request, call_next):
    # CORS / PNA 预检由 CORSMiddleware 处理, 不参与鉴权
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        if path == "/api/auth/token":
            origin = request.headers.get("origin", "")
            if not _origin_is_trusted(request, origin):
                log_error(f"token fetch from untrusted origin: {origin!r}")
                return JSONResponse(status_code=403, content={"detail": "origin not allowed"})
        elif not _token_matches(_auth_from(request)):
            return JSONResponse(
                status_code=401,
                content={"detail": "missing or invalid API token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_error(f"{request.method} {request.url.path} unhandled", exc)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    # 访问日志常开: 文件日志轮转且有界, 排查问题时能还原请求时序
    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    if response.status_code >= 400:
        log_error(f"{request.method} {request.url.path} -> HTTP {response.status_code}")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_error(f"{request.method} {request.url.path} HTTP {exc.status_code}: {exc.detail}")
    detail = exc.detail
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_error(f"{request.method} {request.url.path} crashed", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()},
    )


def _inspect_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_parquet_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _browse_data_file(path: str | None = None) -> dict[str, Any]:
    """选择数据文件：带 path 时直接验证该路径（跳过原生对话框），否则启动非阻塞选择器。

    选择器默认打开 data/training/（下载数据的统一目录）；返回 session id，
    由 GET /api/data-file/browse-poll 轮询结果——用户翻目录多久都不会请求超时。
    """
    manual = str(path or "").strip()
    if manual:
        try:
            expanded = str(Path(manual).expanduser())
        except OSError:
            expanded = manual
        info = _inspect_or_http(expanded)
        save_settings({"last_data_file": info["data_file"]})
        return {"ok": True, "cancelled": False, **info}
    from web.data_download import DOWNLOAD_DIR  # 懒导入，避免启动时拖 pandas

    if is_debug_mode():
        logger.info("Opening native file picker (initialdir=%s)", DOWNLOAD_DIR)
    sid, created = start_picker(
        "选择 K 线 Parquet 文件",
        [("Parquet K线", "*.parquet"), ("所有文件", "*.*")],
        initialdir=str(DOWNLOAD_DIR),
    )
    return {"ok": True, "dialog": True, "session": sid, "already_open": not created}


def _settings_timeframe(symbol: str) -> str | None:
    """settings.last_data_file 里该品种的 timeframe（供 progress/export 按数据
    上下文选制品）；无匹配/读取失败 → None（回退旧式 best_{symbol}.json）。
    """
    try:
        data_file = load_settings().get("last_data_file") or ""
        if not data_file or not Path(data_file).exists():
            return None
        info = inspect_parquet_file(data_file)
        if str(info.get("symbol") or "") == symbol:
            return str(info.get("timeframe") or "") or None
    except Exception:  # noqa: BLE001 溯源失败不阻断默认解析
        pass
    return None


def _strategy_context() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    train_symbol = None
    train_tf = None
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            train_symbol = info.get("symbol")
            train_tf = str(info.get("timeframe") or "") or None
        except Exception:
            pass

    resolved = resolve_strategy_file(
        settings.get("last_strategy_file") or "",
        train_symbol,
        train_tf,
    )
    strategy_info = None
    if resolved:
        try:
            strategy_info = inspect_strategy_file(
                resolved,
                data_file_hint=settings.get("last_data_file") or None,
            )
        except Exception as e:
            strategy_info = {
                "strategy_file": resolved,
                "valid": False,
                "message": str(e),
            }
    return {
        "last_strategy_file": resolved,
        "strategy_file": strategy_info,
        "train_symbol": train_symbol,
    }


def _browse_strategy_file(path: str | None = None) -> dict[str, Any]:
    """选择策略 JSON 文件。

    带 path 时直接校验该路径并返回策略信息（下拉切换/手动路径用，
    不弹原生对话框）；否则启动非阻塞选择器（默认打开 strategies/ 目录）。
    """
    manual = str(path or "").strip()
    if manual:
        try:
            expanded = str(Path(manual).expanduser())
        except OSError:
            expanded = manual
        info = _inspect_strategy_or_http(expanded)
        save_settings({"last_strategy_file": info["strategy_file"]})
        return {"ok": True, "cancelled": False, **info}
    if is_debug_mode():
        logger.info("Opening strategy file picker")
    sid, created = start_picker(
        "选择策略 JSON 文件",
        [("策略 JSON", "*.json"), ("所有文件", "*.*")],
        initialdir=str(STRATEGIES_DIR),
    )
    return {"ok": True, "dialog": True, "session": sid, "already_open": not created}


def _inspect_strategy_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_strategy_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _resolve_train_symbol(symbol: str | None = None) -> str | None:
    if symbol:
        return symbol.strip() or None
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    if not data_file:
        return None
    try:
        return inspect_parquet_file(data_file).get("symbol")
    except Exception:
        return None


def _wait_training_idle(timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        training_manager.status()
        if not training_manager.status().get("active"):
            return
        time.sleep(0.2)


def _sync_and_persist_best_strategy(
    symbol: str,
    *,
    data_file_hint: str | None = None,
) -> dict[str, Any] | None:
    invalidate_checkpoint_cache()
    hint = data_file_hint
    if not hint:
        job = training_manager.status().get("job") or {}
        if str(job.get("symbol") or "") == symbol:
            hint = job.get("data_file") or None
    if not hint:
        hint = load_settings().get("last_data_file") or None
    info = sync_best_strategy_for_symbol(symbol, data_file_hint=hint)
    if info:
        save_settings({"last_strategy_file": info["strategy_file"]})
    return info


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.2.0"}


@app.get("/api/auth/token")
def api_get_token() -> dict[str, str]:
    """返回本机 API 令牌 (仅同源/局域网控制台/已知镜像来源可获取)。"""
    return {"token": API_TOKEN}


@app.post("/api/auth/rotate")
def api_rotate_token() -> dict[str, str]:
    """轮换 API 令牌 (需携带旧令牌); 新令牌立即生效, 其它设备需重新同步。"""
    new_token = _rotate_token()
    logger.info("API token rotated")
    return {"token": new_token}


@app.get("/api/routes")
def api_routes() -> dict[str, Any]:
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path and methods:
            routes.append({"path": path, "methods": sorted(methods)})
    return {"routes": sorted(routes, key=lambda r: r["path"])}


@app.get("/api/events")
async def api_events() -> StreamingResponse:
    """SSE 事件流：实时/模拟盘状态变更推送（事件名=域，data=与 REST 同构快照）。

    无订阅者时 watcher 自动停止；10s 无名 data 心跳保活（前端可见）。单 uvicorn worker 部署（本机控制台）。
    """
    from web.events import event_generator

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/debug/logs")
def api_debug_logs(
    lines: int = 200,
    level: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    """读取服务端日志尾部; level=error|warning|info|debug 按级别过滤, q=关键词(大小写不敏感)。"""
    return debug_snapshot(lines, level, q)


@app.post("/api/debug/client-log")
def api_client_log(req: ClientLogRequest) -> dict[str, bool]:
    msg = req.message
    if req.context:
        msg = f"{msg} | context={req.context}"
    if req.level == "error":
        log_error(f"[client] {msg}")
    elif is_debug_mode():
        logger.info("[client] %s", msg)
    return {"ok": True}


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return load_settings()


@app.put("/api/settings")
def api_put_settings(req: SettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.last_data_file is not None:
        payload["last_data_file"] = req.last_data_file
    if req.last_strategy_file is not None:
        payload["last_strategy_file"] = req.last_strategy_file
    if req.debug_mode is not None:
        payload["debug_mode"] = req.debug_mode
    if req.bg_animation is not None:
        payload["bg_animation"] = req.bg_animation
    if req.ai_provider is not None:
        payload["ai_provider"] = req.ai_provider
    if req.ai_api_key is not None:
        payload["ai_api_key"] = req.ai_api_key
    if req.ai_base_url is not None:
        payload["ai_base_url"] = req.ai_base_url
    if req.ai_model is not None:
        payload["ai_model"] = req.ai_model
    if req.bt_commission_pct is not None:
        payload["bt_commission_pct"] = req.bt_commission_pct
    if req.bt_slippage_pct is not None:
        payload["bt_slippage_pct"] = req.bt_slippage_pct
    if req.bt_hold_policy is not None:
        payload["bt_hold_policy"] = req.bt_hold_policy
    if req.bt_window_bars is not None:
        payload["bt_window_bars"] = req.bt_window_bars
    if req.paper_starting_balance is not None:
        payload["paper_starting_balance"] = req.paper_starting_balance
    if req.paper_notional is not None:
        payload["paper_notional"] = req.paper_notional
    if req.paper_commission_pct is not None:
        payload["paper_commission_pct"] = req.paper_commission_pct
    if req.paper_slippage_pct is not None:
        payload["paper_slippage_pct"] = req.paper_slippage_pct
    if req.max_position_pct is not None:
        payload["max_position_pct"] = req.max_position_pct
    if req.signal_threshold is not None:
        payload["signal_threshold"] = req.signal_threshold
    saved = save_settings(payload)
    if req.debug_mode is not None:
        set_debug_mode(req.debug_mode)
    return {"ok": True, **saved}


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    if data_file:
        try:
            file_info = _attach_data_meta(inspect_parquet_file(data_file), data_file)
        except Exception as e:
            file_info = {
                "data_file": data_file,
                "valid": False,
                "message": str(e),
            }
    snap = debug_snapshot(1)
    strat_ctx = _strategy_context()
    return {
        "train_steps": ModelConfig.TRAIN_STEPS,
        "batch_size": ModelConfig.BATCH_SIZE,
        "reward_mode": ModelConfig.REWARD_MODE,
        "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
        "device": str(ModelConfig.DEVICE),
        "last_data_file": data_file,
        "data_file": file_info,
        "last_strategy_file": strat_ctx["last_strategy_file"],
        "strategy_file": strat_ctx["strategy_file"],
        "debug_mode": load_settings().get("debug_mode", False),
        "bg_animation": settings.get("bg_animation", False),
        "ai_provider": settings.get("ai_provider", "deepseek"),
        "ai_api_key": settings.get("ai_api_key", ""),
        "ai_base_url": settings.get("ai_base_url", "https://api.deepseek.com"),
        "ai_model": settings.get("ai_model", "deepseek-v4-flash"),
        "bt_commission_pct": settings.get("bt_commission_pct", 0.02),
        "bt_slippage_pct": settings.get("bt_slippage_pct", 0.01),
        "paper_starting_balance": settings.get("paper_starting_balance", 100000.0),
        "paper_notional": settings.get("paper_notional", 10000.0),
        "paper_commission_pct": settings.get("paper_commission_pct", 0.02),
        "paper_slippage_pct": settings.get("paper_slippage_pct", 0.01),
        "max_position_pct": settings.get("max_position_pct", 100.0),
        "server_log": snap["server_log"],
        "error_log": snap["error_log"],
    }


@app.get("/api/ai/providers")
def api_ai_providers() -> dict[str, Any]:
    from web.ai_providers import provider_status

    status = provider_status()
    settings = load_settings()
    status["selected"] = settings.get("ai_provider", "deepseek")
    status["has_api_key"] = bool(settings.get("ai_api_key"))
    return status


@app.post("/api/ai/analyze-training")
def api_ai_analyze_training(req: AnalyzeTrainingRequest):
    from fastapi.responses import StreamingResponse

    from web.ai_analyze import analyze_training_stream

    settings = load_settings()
    raw_key = req.api_key if req.api_key is not None else settings.get("ai_api_key") or ""
    key_lower = str(raw_key).strip().lower()
    base_url = (
        req.base_url
        if req.base_url is not None
        else settings.get("ai_base_url") or ""
    )
    model = req.model if req.model is not None else settings.get("ai_model") or ""

    # openclaw_wb 必须先于 openclaw 判断
    if key_lower in ("openclaw_wb",) or key_lower.startswith("openclaw_wb/"):
        provider = "openclaw_wb"
    elif key_lower in ("openclaw",) or key_lower.startswith("openclaw/"):
        provider = "openclaw"
    else:
        provider = (req.provider or settings.get("ai_provider") or "deepseek").strip()

    save_settings({
        "ai_provider": provider,
        "ai_api_key": str(raw_key).strip(),
        "ai_base_url": str(base_url).strip(),
        "ai_model": str(model).strip(),
    })

    def event_gen():
        try:
            for event in analyze_training_stream(
                provider=provider,
                api_key=str(raw_key).strip() or None,
                base_url=str(base_url).strip() or None,
                model=str(model).strip() or None,
                symbol=req.symbol,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/data/files")
def api_data_files() -> dict[str, Any]:
    """本地 .parquet 数据文件清单（data/training + data/slices），供回放页下拉选择。"""
    from web.data_download import list_local_parquet_files

    return {"files": list_local_parquet_files()}


@app.post("/api/data-file/browse")
@app.get("/api/data-file/browse")
def api_browse_data_file(path: str | None = Query(None)) -> dict[str, Any]:
    return _browse_data_file(path)


@app.post("/api/data-file/browse-poll")
@app.get("/api/data-file/browse-poll")
def api_browse_data_file_poll(session: str = Query(...)) -> dict[str, Any]:
    """轮询文件选择会话：done 且选中时校验并返回文件信息（与选文件接口一致）。"""
    res = poll_picker(session)
    if res.get("error"):
        raise HTTPException(400, res["error"])
    if not res["done"]:
        return {"ok": True, "done": False}
    path = res.get("path")
    if not path:
        return {"ok": True, "done": True, "cancelled": True}
    info = _inspect_or_http(path)
    save_settings({"last_data_file": info["data_file"]})
    return {"ok": True, "done": True, "cancelled": False, **info}


@app.post("/api/data/download")
def api_data_download(
    req: DownloadDataRequest, sync: bool = Query(False)
) -> dict[str, Any]:
    """联网下载指定品种/周期的 K 线并保存为本地 parquet。

    默认异步：立即返回 job_id，进度轮询 GET /api/data/download-status?job_id=…；
    ?sync=1 保持旧的阻塞行为（CLI 脚本用），直接返回完整信息。
    """
    from web.data_download import (
        backfill_to_origin,
        download_symbol_bars,
        list_download_jobs,
        submit_download,
        validate_symbol,
        validate_timeframe,
    )

    # 参数校验先行，400 立即返回（不占用任务槽）
    try:
        sym = validate_symbol(req.symbol)
        tf = validate_timeframe(req.timeframe)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    source = (req.source or "tradingview").strip().lower()
    n_bars = req.n_bars
    mode = (req.mode or "merge").strip().lower()
    if mode not in ("merge", "replace", "backfill"):
        raise HTTPException(400, f"不支持的写入模式: {mode}；可选 merge / replace / backfill")

    if sync:
        try:
            if mode == "backfill":
                info = backfill_to_origin(symbol=sym, timeframe=tf, source=source,
                                          page_bars=n_bars)
            else:
                info = download_symbol_bars(
                    symbol=sym, timeframe=tf, source=source, n_bars=n_bars, mode=mode
                )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 网络/数据源错误
            log_error(f"download {sym} {tf} failed", exc)
            raise HTTPException(502, f"下载失败: {exc}") from exc
        save_settings({"last_data_file": info["data_file"]})
        return {"ok": True, "cancelled": False, "source": source, **info}

    job_id = submit_download(symbol=sym, timeframe=tf, source=source, n_bars=n_bars, mode=mode)
    mine = next((s for s in list_download_jobs() if s["job_id"] == job_id), {}) or {}
    position = mine.get("position") or 0
    status = mine.get("status") or "queued"
    return {
        "ok": True,
        "cancelled": False,
        "source": source,
        "job_id": job_id,
        "status": status,
        "position": position,
        "message": f"已加入下载队列（第 {position} 位）" if position else "下载任务已启动",
    }


@app.get("/api/data/download-status")
def api_data_download_status(job_id: str = Query(...)) -> dict[str, Any]:
    """查询单个后台下载任务进度；完成时附带完整结果。"""
    from web.data_download import get_download_job

    job = get_download_job(job_id)
    if job is None:
        raise HTTPException(404, f"任务不存在: {job_id}")
    snap = job.snapshot()
    snap["ok"] = snap["status"] != "error"
    return snap


@app.get("/api/data/download-queue")
def api_data_download_queue() -> dict[str, Any]:
    """下载队列总览：运行/排队在前，最近完成在后（含各自进度与排队位置）。"""
    from web.data_download import list_download_jobs

    return {"jobs": list_download_jobs()}


@app.get("/api/data/download-history")
def api_data_download_history(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    """下载历史面板：sidecar 驱动的 (source, symbol, bars, date) 行 + 按源统计。"""
    from web.data_download import list_download_history

    return list_download_history(limit=limit)


@app.post("/api/strategy-file/browse")
@app.get("/api/strategy-file/browse")
def api_browse_strategy_file(path: str | None = Query(None)) -> dict[str, Any]:
    """选择策略文件：带 path 时直接验证该路径（免对话框），否则启动非阻塞选择器。"""
    return _browse_strategy_file(path)


@app.post("/api/strategy-file/browse-poll")
@app.get("/api/strategy-file/browse-poll")
def api_browse_strategy_file_poll(session: str = Query(...)) -> dict[str, Any]:
    """轮询策略选择会话：done 且选中时校验并返回策略信息。"""
    res = poll_picker(session)
    if res.get("error"):
        raise HTTPException(400, res["error"])
    if not res["done"]:
        return {"ok": True, "done": False}
    path = res.get("path")
    if not path:
        return {"ok": True, "done": True, "cancelled": True}
    info = _inspect_strategy_or_http(path)
    save_settings({"last_strategy_file": info["strategy_file"]})
    return {"ok": True, "done": True, "cancelled": False, **info}


@app.post("/api/strategy-file/sync-best")
@app.get("/api/strategy-file/sync-best")
def api_sync_best_strategy(symbol: str | None = None) -> dict[str, Any]:
    sym = _resolve_train_symbol(symbol)
    if not sym:
        raise HTTPException(400, "请先选择训练数据文件或指定品种")
    info = _sync_and_persist_best_strategy(sym)
    if not info:
        raise HTTPException(404, f"未找到 {sym} 的可用策略")
    return {"ok": True, **info}


def _progress_with_live_step(symbol: str, active: bool,
                             timeframe: str | None = None) -> dict[str, Any]:
    p = get_symbol_progress(symbol, timeframe)
    current_step = p.current_step
    if active:
        live = training_manager.parse_step_from_log()
        if live is not None:
            current_step = max(current_step, live)
    train_steps = p.train_steps
    progress_pct = min(100.0, 100.0 * current_step / train_steps) if train_steps > 0 else 0.0
    val_score = None
    hist = p.history or {}
    vals = hist.get("val_score") or []
    if vals:
        try:
            val_score = float(vals[-1])
        except (TypeError, ValueError):
            val_score = None
    return {
        "symbol": p.symbol,
        "current_step": current_step,
        "train_steps": train_steps,
        "progress_pct": round(progress_pct, 1),
        "best_score": p.best_score,
        "val_score": val_score,
        "formula_decoded": p.formula_decoded,
        "status": p.status,
        "history": p.history,
        "has_checkpoint": bool(p.checkpoint_path),
        "has_strategy": p.has_strategy,
        "holdout_bars": p.holdout_bars,
        "holdout": p.holdout,
    }


def _attach_training_time(
    row: dict[str, Any] | None,
    *,
    symbol: str | None,
    job: dict[str, Any] | None,
    active: bool,
) -> dict[str, Any] | None:
    if not row or not symbol:
        return row
    summary = get_training_time_summary(symbol, job=job, active=active)
    row = dict(row)
    row["session_seconds"] = summary.session_seconds
    row["history_total_seconds"] = summary.history_total_seconds
    return row


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    progress = None

    training = training_manager.status()
    job = training.get("job")
    active = bool(training.get("active"))

    if data_file:
        try:
            file_info = _attach_data_meta(inspect_parquet_file(data_file), data_file)
            sym = file_info.get("symbol")
            row = _progress_with_live_step(sym, active=False,
                                           timeframe=file_info.get("timeframe"))
            progress = {
                "symbol": row["symbol"],
                "status": row["status"],
                "current_step": row["current_step"],
                "train_steps": row["train_steps"],
                "progress_pct": row["progress_pct"],
                "best_score": row["best_score"],
                "val_score": row.get("val_score"),
                "formula_decoded": row["formula_decoded"],
                "has_checkpoint": row.get("has_checkpoint", False),
                "has_strategy": row.get("has_strategy", False),
                "holdout_bars": row.get("holdout_bars"),
                "holdout": row.get("holdout"),
            }
            progress = _attach_training_time(
                progress, symbol=sym, job=job, active=active and job and job.get("symbol") == sym
            )
        except Exception as e:
            file_info = {"data_file": data_file, "valid": False, "message": str(e)}

    if job and job.get("symbol") and active:
        sym = job["symbol"]
        row = _progress_with_live_step(sym, active=True)
        progress = {
            "symbol": row["symbol"],
            "status": "running_job",
            "current_step": row["current_step"],
            "train_steps": row["train_steps"],
            "progress_pct": row["progress_pct"],
            "best_score": row["best_score"],
            "val_score": row.get("val_score"),
            "formula_decoded": row["formula_decoded"],
            "has_checkpoint": row.get("has_checkpoint", False),
            "has_strategy": row.get("has_strategy", False),
            "holdout_bars": row.get("holdout_bars"),
            "holdout": row.get("holdout"),
        }
        progress = _attach_training_time(progress, symbol=sym, job=job, active=True)

    return {
        "data_file": file_info,
        "progress": progress,
        "training": training,
    }


def _symbol_etag(symbol: str, timeframe: str | None = None) -> str:
    """响应签名：只依赖会改变 /api/symbols 载荷的文件统计信息。

    任一相关文件(mtime_ns/size)变化 → 新 ETag；都不变 → 浏览器 304，
    免去重复下载 ~200KB history 载荷。策略/实时 history/最新 checkpoint
    任一更新即换新。timeframe 给定时策略制品按 tf 限定（H1 数据上下文
    应加载 best_{symbol}_{tf}.json，etag 必须跟随实际读取的文件）。
    """
    parts: list[str] = []

    def _add(p: Path) -> None:
        try:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            pass

    _add(PROJECT_ROOT / f"training_history_{symbol}.json")
    _add(STRATEGIES_DIR / f"best_{symbol}.json")
    if timeframe:
        _add(STRATEGIES_DIR / f"best_{symbol}_{timeframe}.json")
    try:
        ck = checkpoint_glob(symbol)
        if ck:
            _add(ck[-1])          # 最新 checkpoint（决定展示的进度/history）
    except Exception:  # noqa: BLE001 glob 失败不阻断
        pass
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return f'"{digest}"'


@app.get("/api/symbols/{symbol}", response_model=None)
def api_symbol(symbol: str, request: Request, response: Response) -> Response | dict[str, Any]:
    tf = _settings_timeframe(symbol)
    etag = _symbol_etag(symbol, tf)
    # 浏览器带相同 ETag → 304 无 body（客户端缓存命中，跳过 ~200KB 下载）
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    p = get_symbol_progress(symbol, tf)
    payload = {
        "symbol": p.symbol,
        "status": p.status,
        "current_step": p.current_step,
        "train_steps": p.train_steps,
        "progress_pct": round(p.progress_pct, 1),
        "best_score": p.best_score,
        "best_formula": p.best_formula,
        "formula_decoded": p.formula_decoded,
        "has_strategy": p.has_strategy,
        "strategy_score": p.strategy_score,
        "checkpoint_path": p.checkpoint_path,
        "history": p.history,
        "holdout_bars": p.holdout_bars,
        "holdout": p.holdout,
    }
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "no-cache"   # 每次重验证（发 If-None-Match）
    return payload


@app.get("/api/strategies")
def api_strategies() -> dict[str, Any]:
    return {"strategies": list_strategies()}


@app.get("/api/strategies/{symbol}/export")
def api_export_strategy(symbol: str):
    import json

    from fastapi.responses import Response

    tf = _settings_timeframe(symbol)
    try:
        payload = get_strategy_for_export(symbol, tf)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    progress = get_symbol_progress(symbol, tf)
    step = progress.current_step
    score = payload.get("best_score")
    if score is None:
        score = progress.strategy_score if progress.strategy_score is not None else progress.best_score
    filename = build_strategy_export_filename(symbol, step, score)
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.get("/api/training/{symbol}/export")
def api_export_training(symbol: str):
    from fastapi.responses import Response

    try:
        body, zip_name = build_training_export_zip(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=body,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.post("/api/training/import")
async def api_import_training(
    file: UploadFile = File(...),
    symbol: str | None = Query(None, description="当前选择的品种，用于校验导入包是否一致"),
) -> dict[str, Any]:
    if training_manager.status().get("active"):
        raise HTTPException(409, "训练进行中，请先停止再导入")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "上传文件为空")

    try:
        return import_training_package(
            raw,
            file.filename or "upload.zip",
            expected_symbol=symbol or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/training/status")
def api_training_status() -> dict[str, Any]:
    status = training_manager.status()
    status["log_tail"] = training_manager.tail_log(150)
    status["inspections"] = training_manager.inspections()
    try:
        status["eta"] = training_manager.eta_status()
    except Exception:  # noqa: BLE001 ETA 是辅助信息，任何异常都不阻塞状态
        status["eta"] = None
    return status


@app.post("/api/training/inspect-now")
def api_training_inspect_now() -> dict[str, Any]:
    """内置训练巡检：立即对当前/最近训练日志做一次规则诊断（无需 API Key）。"""
    entry = training_manager.inspect_now()
    return {"ok": True, "entry": entry}


@app.post("/api/training/start")
def api_training_start(req: StartTrainingRequest) -> dict[str, Any]:
    info = _inspect_or_http(req.data_file)
    save_settings({"last_data_file": info["data_file"]})

    # 训练数据范围：tail / spread 时先在源文件上生成确定性子集，再训子集文件。
    # 子集是独立新文件（新数据指纹），walk-forward/holdout/冠军闸门按原逻辑生效；
    # 引擎与 train_file.py 无需任何改动。
    sampling = {"mode": "full", "n_bars": None, "n_chunks": None}
    train_file = info["data_file"]
    mode = (req.data_mode or "full").strip().lower()
    if mode in ("tail", "spread"):
        try:
            from data_pipeline.train_sampler import prepare_training_subset

            res = prepare_training_subset(
                train_file, mode=mode,
                n_bars=req.n_bars, n_chunks=req.n_chunks,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        sampling = res
        train_file = res["data_file"]

    # 训练管理器/引擎以子集文件为准（品种/周期由子集文件名解析，与源一致）
    job_info = _inspect_or_http(train_file)
    try:
        job = training_manager.start(
            data_file=train_file,
            symbol=job_info["symbol"],
            timeframe=job_info["timeframe"],
            mode="ftmo",
            from_scratch=bool(req.from_scratch),
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    if req.from_scratch:
        invalidate_checkpoint_cache()
    return {
        "ok": True,
        "job": job.to_dict(),
        "data_file": info,
        "from_scratch": bool(req.from_scratch),
        "sampling": sampling,
    }


@app.post("/api/training/subset-preview")
def api_training_subset_preview(req: SubsetPreviewRequest) -> dict[str, Any]:
    """训练子集取样预览：返回所选区间在整段历史上的位置与 regime 覆盖（不写文件）。"""
    try:
        info = _inspect_or_http(req.data_file)
        from data_pipeline.train_sampler import preview_training_subset

        return preview_training_subset(
            info["data_file"],
            mode=req.mode,
            n_bars=req.n_bars,
            n_chunks=req.n_chunks,
            regime=req.regime,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/training/stop")
def api_training_stop() -> dict[str, Any]:
    job = training_manager.status().get("job") or {}
    symbol = job.get("symbol")
    data_file_hint = job.get("data_file")
    stopped = training_manager.stop()
    strategy_file = None
    if symbol:
        _wait_training_idle()
        strategy_file = _sync_and_persist_best_strategy(
            symbol,
            data_file_hint=data_file_hint,
        )
    return {
        "ok": stopped,
        "training": training_manager.status(),
        "strategy_file": strategy_file,
    }


# ─────────────────────────────────────────────────────────────────────
# 范围对比实验 API（tail vs spread 短训对比）
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/experiment/compare")
def api_experiment_compare(req: StartCompareRequest) -> dict[str, Any]:
    info = _inspect_or_http(req.data_file)
    if training_manager.status().get("active"):
        raise HTTPException(409, "正式训练进行中，请先停止或等它结束再跑对比实验")
    if compare_manager.status().get("active"):
        raise HTTPException(409, "已有对比实验在运行")
    if req.n_bars < 5000:
        raise HTTPException(400, "n_bars 至少 5000 根")
    seeds = [int(s) for s in req.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = [42]
    try:
        job = compare_manager.start(
            data_file=info["data_file"],
            n_bars=int(req.n_bars),
            chunks=req.chunks,
            steps=max(1, int(req.steps)),
            seeds=seeds[:4],
            regime=(req.regime or "vol").strip().lower(),
            rep_criterion=req.rep_criterion,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict(), "data_file": info}


@app.get("/api/experiment/compare-status")
def api_experiment_compare_status() -> dict[str, Any]:
    return compare_manager.status()


@app.post("/api/experiment/compare-stop")
def api_experiment_compare_stop() -> dict[str, Any]:
    stopped = compare_manager.stop()
    return {"ok": stopped, "status": compare_manager.status()}


# ─────────────────────────────────────────────────────────────────────
# 回测 API
# ─────────────────────────────────────────────────────────────────────

_METRIC_KEYS = (
    "total_return", "sharpe", "sortino", "profit_loss_ratio",
    "n_trades", "win_rate", "avg_hold_bars",
)


def _file_etag(path: Path) -> str:
    """基于文件 mtime+size 的弱 ETag（304 重验证用，避免整读大 JSON）。"""
    try:
        st = path.stat()
        return f'"{path.name}:{st.st_mtime_ns}:{st.st_size}"'
    except OSError:
        return '""'


def _load_backtest_report() -> dict[str, Any] | None:
    import json

    report_path = BACKTEST_OUTPUT_DIR / "multi_factor_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _backtest_focus_symbol(symbol: str | None = None) -> str | None:
    """Resolve the symbol used to filter backtest charts/report for the web UI."""
    if symbol:
        return symbol.strip() or None

    job = backtest_manager.status().get("job") or {}
    if job.get("symbol"):
        return str(job["symbol"])

    strat = _strategy_context().get("strategy_file") or {}
    if strat.get("symbol"):
        return str(strat["symbol"])

    report = _load_backtest_report()
    if report:
        keys = list((report.get("symbols") or {}).keys())
        if len(keys) == 1:
            return keys[0]
    return None


def _filter_report_for_symbol(report: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbols = report.get("symbols") or {}
    if symbol not in symbols:
        return report

    sym_data = symbols[symbol]
    return {
        **report,
        "focus_symbol": symbol,
        "symbols": {symbol: sym_data},
        "portfolio": {
            "total_return": sym_data.get("total_return"),
            "sharpe": sym_data.get("sharpe"),
            "sortino": sym_data.get("sortino"),
            "max_drawdown": sym_data.get("max_drawdown"),
            "profit_loss_ratio": sym_data.get("profit_loss_ratio"),
            "n_trades": sym_data.get("n_trades"),
            "win_rate": sym_data.get("win_rate"),
        },
    }


def _list_backtest_charts(symbol: str | None = None) -> list[dict[str, str]]:
    """列出回测输出目录下的图表；单品种模式只返回该品种相关文件。"""
    if not BACKTEST_OUTPUT_DIR.exists():
        return []

    if symbol:
        charts: list[dict[str, str]] = []
        equity = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
        if equity.exists():
            charts.append(
                {"name": equity.name, "label": f"{symbol} 资金曲线", "kind": "equity"}
            )
        return charts

    charts = []
    portfolio = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
    if portfolio.exists():
        charts.append({"name": "portfolio_equity.png", "label": "组合资金曲线", "kind": "portfolio"})
    for path in sorted(BACKTEST_OUTPUT_DIR.glob("equity_*.png")):
        sym = path.stem.replace("equity_", "", 1)
        charts.append({"name": path.name, "label": f"{sym} 资金曲线", "kind": "symbol"})
    return charts


@app.get("/api/backtest/status")
def api_backtest_status() -> dict[str, Any]:
    status = backtest_manager.status()
    status["log_tail"] = backtest_manager.tail_log(200)
    return status


@app.post("/api/backtest/start")
def api_backtest_start(req: StartBacktestRequest) -> dict[str, Any]:
    info = _inspect_strategy_or_http(req.strategy_file)
    settings = load_settings()
    commission = (
        float(req.commission_pct)
        if req.commission_pct is not None
        else float(settings.get("bt_commission_pct", 0.02))
    )
    slippage = (
        float(req.slippage_pct)
        if req.slippage_pct is not None
        else float(settings.get("bt_slippage_pct", 0.01))
    )
    if commission < 0 or slippage < 0:
        raise HTTPException(400, "手续费和滑点不能为负数")

    save_settings({
        "last_strategy_file": info["strategy_file"],
        "bt_commission_pct": commission,
        "bt_slippage_pct": slippage,
    })

    data_file: str | None = None
    # 1) 显式指定的回测数据（页面数据下拉框；允许跨品种/跨周期/跨切片）
    if (req.data_file or "").strip():
        try:
            pf = inspect_parquet_file(req.data_file.strip())
            if pf.get("valid") is False:
                raise HTTPException(
                    400,
                    f"指定的回测数据无效: {pf.get('message') or req.data_file}",
                )
            data_file = pf["data_file"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                400,
                f"指定的回测数据无法加载: {req.data_file}\n{e}",
            ) from e
    elif not data_file:
        # 2) 未显式指定：优先用策略 JSON 里记录的训练数据路径
        strat_data = (info.get("data_file") or "").strip()
        if strat_data:
            try:
                pf = inspect_parquet_file(strat_data)
                if pf.get("valid") is False:
                    raise HTTPException(
                        400,
                        f"策略记录的数据文件无效: {pf.get('message') or strat_data}",
                    )
                data_file = pf["data_file"]
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    400,
                    f"策略记录的数据文件无法加载: {strat_data}\n{e}",
                ) from e
        else:
            # 3) 回退：训练页最近选择的、同品种 Parquet
            last_data = settings.get("last_data_file") or ""
            if last_data:
                try:
                    pf = inspect_parquet_file(last_data)
                    if pf.get("symbol") == info.get("symbol") and pf.get("valid") is not False:
                        data_file = pf["data_file"]
                except Exception:
                    pass

    if not data_file:
        raise HTTPException(
            400,
            "该策略未记录数据文件路径（data_file），且当前也没有同品种的 Parquet。"
            "请先在「模型训练」页选择对应品种的 Parquet 再回测；"
            "或使用本软件训练/导出、且包含 data_file 字段的策略文件。",
        )

    save_settings({"last_data_file": data_file})

    from web.hold_policy import combo_id

    hold_policy = combo_id(req.hold_policy)
    window_bars = None
    if req.window_bars is not None:
        window_bars = int(req.window_bars)
        if window_bars <= 0:
            window_bars = None
    pos_pct = _resolve_pos_pct(req.max_position_pct)
    save_settings({"max_position_pct": pos_pct})
    sig_thr = _resolve_signal_threshold(req.signal_threshold)
    save_settings({"signal_threshold": sig_thr})
    try:
        job = backtest_manager.start(
            strategy_file=info["strategy_file"],
            data_file=data_file,
            commission_pct=commission,
            slippage_pct=slippage,
            hold_policy=hold_policy,
            window_bars=window_bars,
            max_position_pct=pos_pct,
            signal_threshold=sig_thr,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict(), "strategy_file": info, "data_file": data_file,
            "hold_policy": hold_policy, "window_bars": window_bars,
            "signal_threshold": sig_thr}


def _resolve_signal_threshold(value: float | None) -> float:
    """统一无信号阈值：请求优先，缺省取设置/Config 默认（仅档位集合）。"""
    if value is None:
        value = load_settings().get("signal_threshold")
    from web.settings import sanitize_signal_threshold
    return sanitize_signal_threshold(value)


def _resolve_pos_pct(value: float | None) -> float:
    """每笔投入上限 %：请求优先，缺省取设置（1..200 钳制）。"""
    if value is None:
        value = float(load_settings().get("max_position_pct", 100.0))
    return min(200.0, max(1.0, float(value)))


def _normalize_ab_caps(caps: list[float] | None, fallback: float) -> list[float]:
    """A/B 混合上限档位归一化：钳到 1..200、去重、升序；缺省回退单档 fallback。"""
    if caps:
        out = sorted({min(200.0, max(1.0, float(c))) for c in caps if c and c > 0})
        if out:
            return out
    return [min(200.0, max(1.0, float(fallback)))]


def _sweep_pick_best(rows: list[dict]) -> tuple[dict | None, dict | None]:
    """阈值/成本类敏感性扫描的选档：夏普最高档（并列取交易更多者）；

    只认有交易且夏普非 None 的行；返回 (最优行, 次优行)，无可选行返回 (None, None)。
    """
    def _key(r_: dict):
        sr = float(r_["sharpe"]) if r_["sharpe"] is not None else float("-inf")
        nt = float(r_["n_trades"] or 0.0)
        return (sr, nt)

    cand = [r for r in rows if (r.get("n_trades") or 0) > 0 and r.get("sharpe") is not None]
    if not cand:
        return None, None
    best = max(cand, key=_key)
    second = max((r for r in cand if r is not best), key=_key, default=None)
    return best, second


@app.post("/api/backtest/stop")
def api_backtest_stop() -> dict[str, Any]:
    stopped = backtest_manager.stop()
    return {"ok": stopped, "backtest": backtest_manager.status()}


@app.get("/api/backtest/report")
def api_backtest_report(request: Request, response: Response, symbol: str | None = None) -> Any:
    from fastapi.responses import Response as _R304  # response 形参遮蔽了类名

    report_path = BACKTEST_OUTPUT_DIR / "multi_factor_report.json"
    if report_path.exists():
        etag = _file_etag(report_path)
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "no-cache"  # 每次重验证（304 无 body）
        if request.headers.get("if-none-match") == etag:
            return _R304(status_code=304, headers={"ETag": etag})
    report = _load_backtest_report()
    focus = _backtest_focus_symbol(symbol)
    if report and focus:
        report = _filter_report_for_symbol(report, focus)
    return {
        "available": report is not None,
        "report": report,
        "charts": _list_backtest_charts(focus),
        "focus_symbol": focus,
    }


@app.get("/api/backtest/hold-matrix")
def api_backtest_hold_matrix() -> dict[str, Any]:
    """持仓管理正交组合 N×N 矩阵（读 results/hold_matrix_latest.json）。"""
    import json as _json

    path = ROOT / "results" / "hold_matrix_latest.json"
    if not path.exists():
        return {"available": False, "matrix": None}
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError):
        return {"available": False, "matrix": None}
    # 样本外溯源感知：按训练真 holdout 边界给窗口打 真伪 OOS 标签（tail 窗口才可判）。
    # 注意 launchd 下 web 进程 CWD=/，相对路径必须先绝对化。
    if isinstance(data, dict) and data.get("window_bars") and data.get("window_mode") == "tail":
        try:
            from web.oos_provenance import classify_window

            def _abs(p: str | None) -> str | None:
                if not p:
                    return None
                pp = Path(p)
                return str((ROOT / pp) if not pp.is_absolute() else pp)

            data = dict(data)
            data["oos"] = classify_window(
                _abs(data.get("strategy_file")),
                _abs(data.get("data_file")),
                data.get("window_start"),
                data.get("window_bars"),
            )
        except Exception:  # noqa: BLE001 溯源失败不阻断矩阵展示
            pass
    # 成本敏感性（22 组合 × 成本档）：scripts/hold_matrix_cost.py 产物，缺省隐藏
    cost = None
    cost_path = ROOT / "results" / "hold_matrix_cost_latest.json"
    try:
        if cost_path.exists():
            cj = _json.loads(cost_path.read_text(encoding="utf-8"))
            if isinstance(cj, dict) and cj.get("combos"):
                cost = {
                    "meta": cj.get("meta"),
                    "combos": cj.get("combos"),
                    "summary": cj.get("summary"),
                }
    except Exception:  # noqa: BLE001 成本文件损坏不阻断矩阵
        cost = None
    return {"available": True, "matrix": data, "cost_sensitivity": cost}


_hold_matrix_manager: Any = None


def _get_hold_matrix_manager() -> Any:
    """惰性单例：跑 N×N 全组合矩阵的子进程管理器。"""
    global _hold_matrix_manager
    if _hold_matrix_manager is None:
        from web.hold_matrix_manager import HoldMatrixManager

        _hold_matrix_manager = HoldMatrixManager()
    return _hold_matrix_manager


def _resolve_matrix_defaults(req: HoldMatrixRunRequest) -> tuple[str | None, str | None]:
    """解析本次 N×N 全组合回测的 策略/数据 文件：优先请求体 → 上次回测设置。"""
    settings = load_settings()
    strategy_file = (req.strategy_file or "").strip() or (settings.get("last_strategy_file") or "").strip()
    sf = Path(strategy_file) if strategy_file else None
    if sf is not None and not sf.is_absolute():
        sf = ROOT / sf
    data_file = (req.data_file or "").strip()
    # 数据文件缺省：策略 JSON 里记录的 data_file → 上次回测用的数据
    if not data_file and sf is not None and sf.exists():
        try:
            import json as _json

            meta = _json.loads(sf.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and meta.get("data_file"):
                data_file = str(meta["data_file"])
        except Exception:
            pass
    if not data_file:
        data_file = (settings.get("last_data_file") or "").strip()
    df = Path(data_file) if data_file else None
    if df is not None and not df.is_absolute():
        df = ROOT / df
    return (str(sf) if sf is not None and sf.exists() else None,
            str(df) if df is not None and df.exists() else None)


@app.post("/api/backtest/hold-matrix/run")
def api_backtest_hold_matrix_run(req: HoldMatrixRunRequest) -> dict[str, Any]:
    """启动 N×N 全组合矩阵回测（脚本跑全部组合，写 results/hold_matrix_latest.json）。"""
    strategy_file, data_file = _resolve_matrix_defaults(req)
    window_bars = req.window_bars
    if window_bars is not None:
        window_bars = int(window_bars)
        if window_bars <= 0:
            window_bars = None
    try:
        job = _get_hold_matrix_manager().start(
            strategy_file=strategy_file,
            data_file=data_file,
            commission_pct=req.commission_pct if req.commission_pct is not None else 0.02,
            slippage_pct=req.slippage_pct if req.slippage_pct is not None else 0.01,
            window_bars=window_bars,
            window_mode=req.window_mode or "tail",
            regime=req.regime or "vol",
            chunks=req.chunks if req.chunks else 4,
            max_position_pct=_resolve_pos_pct(req.max_position_pct),
            signal_threshold=_resolve_signal_threshold(req.signal_threshold),
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "job": job.to_dict(),
            "strategy_file": strategy_file, "data_file": data_file,
            "window_bars": window_bars, "window_mode": req.window_mode or "tail"}


@app.get("/api/backtest/matrix-best")
def api_backtest_matrix_best(symbol: str | None = None) -> dict[str, Any]:
    """同品种最近一次全组合矩阵的「最优组合」（按夏普选非基线组合）+
    其最大回撤约束，供回测页默认持仓管理与标注。"""
    p = ROOT / "results" / "matrix_best_combo.json"
    if not p.exists():
        return {"exists": False, "best": None}
    try:
        best = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"exists": False, "best": None}
    if symbol and best.get("symbol") and best.get("symbol") != symbol:
        return {"exists": False, "best": None, "symbol_mismatch": True}
    return {"exists": True, "best": best}


class MatrixAppliedRequest(BaseModel):
    symbol: str
    combo: str = ""
    sharpe: float | None = None
    max_drawdown: float | None = None
    window_bars: int | None = None
    window_mode: str | None = None


@app.get("/api/backtest/matrix-applied")
def api_backtest_matrix_applied_get(symbol: str | None = None) -> dict[str, Any]:
    """读矩阵最优「用户接受记录」（settings.matrix_applied，区别于手动记忆）。

    给某品种返回其最近一次被明确接受的 N×N 最优组合；前端据此在重启后
    仍能标出「已应用（矩阵接受）」并自动恢复默认。"""
    rec = (load_settings().get("matrix_applied") or {})
    sym = (symbol or "").strip()
    if sym:
        return {"exists": bool(rec.get(sym)), "applied": rec.get(sym) or None, "symbol": sym}
    return {"applied": rec}


@app.put("/api/backtest/matrix-applied")
def api_backtest_matrix_applied_put(req: MatrixAppliedRequest) -> dict[str, Any]:
    """记录/清除某品种的矩阵最优接受。combo="" 表示清除该品种记录。"""
    from web.hold_policy import combo_id

    sym = (req.symbol or "").strip()
    if not sym:
        raise HTTPException(400, "缺少 symbol")
    cur = dict(load_settings().get("matrix_applied") or {})
    combo = combo_id((req.combo or "").strip()) if (req.combo or "").strip() else ""
    if combo and combo != "signal":
        row: dict[str, Any] = {"combo": combo, "at": time.time()}
        if req.sharpe is not None:
            row["sharpe"] = round(float(req.sharpe), 6)
        if req.max_drawdown is not None:
            row["max_drawdown"] = round(float(req.max_drawdown), 8)
        if req.window_bars:
            row["window_bars"] = int(req.window_bars)
        if req.window_mode:
            row["window_mode"] = str(req.window_mode)
        cur[sym] = row
    else:
        cur.pop(sym, None)
    save_settings({"matrix_applied": cur})
    return {"ok": True, "symbol": sym, "applied": cur.get(sym)}


class PolicyABRequest(BaseModel):
    strategy_file: str
    policies: list[str] = []
    data_file: str | None = None
    window_bars: int | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None
    max_position_pct: float | None = None
    signal_threshold: float | None = None
    # 混合上限对比：逗号分隔的多档每笔投入上限 %（如 [5, 25, 100]）→
    # 同一方案按每档各跑一行，比较风险调整后哪个上限最优；缺省单档（max_position_pct）
    caps: list[float] | None = None


class CostSweepRequest(BaseModel):
    strategy_file: str
    hold_policy: str = "signal"
    data_file: str | None = None
    window_bars: int | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None
    max_position_pct: float | None = None
    signal_threshold: float | None = None


class ThresholdSweepRequest(BaseModel):
    strategy_file: str
    hold_policy: str = "signal"
    data_file: str | None = None
    window_bars: int | None = None
    commission_pct: float | None = None
    slippage_pct: float | None = None
    max_position_pct: float | None = None
    # 缺省扫描 0.05/0.3/0.5/0.8 四档（与 btThresholdSelect 档位一致）；可传自定义档位
    thresholds: list[float] | None = None


@app.post("/api/backtest/policy-ab")
def api_backtest_policy_ab(req: PolicyABRequest) -> dict[str, Any]:
    """持仓方案 A/B：同一模型 × 多方案并排跑离散撮合回放，输出对比表指标。

    因子只算一次（特征/VM 大头），每个方案一次 run_replay。
    返回每方案 stats + 最大单笔亏损/最大盈利/平均持仓，供回测页 A/B 面板。
    """
    import numpy as np
    import torch
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.features import FeatureEngineer
    from model_core.vm import StackVM
    from web.hold_policy import HOLD_POLICIES, combo_id
    from web.paper_replay import run_replay

    strategy_file = (req.strategy_file or "").strip()
    data_file = (req.data_file or "").strip()
    sf_p = Path(strategy_file)
    if not sf_p.is_absolute():
        sf_p = ROOT / sf_p
    strategy_file = str(sf_p.resolve())
    if not sf_p.exists():
        raise HTTPException(400, f"策略文件不存在: {strategy_file}")
    if data_file:
        df_p = Path(data_file)
        if not df_p.is_absolute():
            df_p = ROOT / df_p
        data_file = str(df_p.resolve())
        if not Path(data_file).exists():
            raise HTTPException(400, f"数据文件不存在: {data_file}")
    else:
        meta = json.loads(sf_p.read_text(encoding="utf-8"))
        ds = meta.get("data_source") or {}
        data_file = str(ds.get("data_file") or meta.get("data_file") or "")
        if not data_file or not Path(data_file).exists():
            raise HTTPException(400, "无法定位策略训练数据文件，请显式选择回测数据")

    meta = json.loads(sf_p.read_text(encoding="utf-8"))
    formula = meta.get("formula")
    if not formula:
        raise HTTPException(400, "策略文件缺少 formula")
    sym = meta.get("symbol") or Path(strategy_file).stem.replace("best_", "", 1)

    pids = [combo_id(p) for p in (req.policies or [])]
    pids = [p for p in dict.fromkeys(pids) if p]
    if not pids:
        raise HTTPException(400, "至少选择一个持仓方案")

    pm = ParquetDataManager(data_file)
    pm.load()
    raw = pm.raw_dict
    T = int(raw["close"].shape[1])
    window = req.window_bars
    start = 0
    if window:
        window = int(window)
        if window < 800:
            raise HTTPException(400, f"窗口过小（{window} 根）：特征 warm-up 需 ≥800 根")
        if window < T:
            start = T - window
    if T - start < 800:
        raise HTTPException(400, f"数据不足（{T - start} 根）：特征 warm-up 需 ≥800 根")
    raw_s = {k: v[:, start:] for k, v in raw.items()}

    try:
        feats = FeatureEngineer.compute_features(raw_s)
        vm = StackVM()
        with torch.no_grad():
            factor = vm.execute([int(t) for t in formula], feats)
        if factor is None or factor.ndim != 2 or factor.shape[1] < 2:
            raise HTTPException(400, "公式在该数据上无有效输出")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"因子计算失败: {exc}") from exc

    comm = req.commission_pct if req.commission_pct is not None else 0.02
    slip = req.slippage_pct if req.slippage_pct is not None else 0.01
    pos_pct = _resolve_pos_pct(req.max_position_pct)
    sig_thr = _resolve_signal_threshold(req.signal_threshold)
    factor_np = factor[0].cpu().numpy().astype(float)
    # 混合上限对比：caps 缺省 = [当前 pos_pct] 单档；给出多档时 pid × cap 各一行
    caps = _normalize_ab_caps(req.caps, pos_pct)
    out = []
    for pid in pids:
        for cap in caps:
            r = run_replay(factor=factor_np,
                           open_p=raw_s["open"][0].numpy(),
                           high_p=raw_s["high"][0].numpy(),
                           low_p=raw_s["low"][0].numpy(),
                           close_p=raw_s["close"][0].numpy(),
                           commission_pct=comm, slippage_pct=slip, policy_id=pid,
                           max_position_pct=cap, threshold=sig_thr)
            pnls = [t["pnl"] for t in r["trades"]]
            st = r["stats"]
            out.append({
                "policy_id": pid,
                "name": HOLD_POLICIES[pid.split("+")[-1]]["name"],
                "desc": HOLD_POLICIES[pid.split("+")[-1]].get("desc", ""),
                "cap_pct": float(cap),
                "total_return": st.get("total_return"),
                "sharpe": st.get("sharpe"),
                "sortino": st.get("sortino"),
                "max_drawdown": st.get("max_drawdown"),
                "n_trades": st.get("n_trades"),
                "win_rate": st.get("win_rate"),
                "profit_loss_ratio": st.get("profit_loss_ratio"),
                "fees_total": st.get("fees_total"),
                "avg_hold_bars": st.get("avg_hold_bars"),
                "max_win": round(float(max(pnls)), 6) if pnls else None,
                "max_single_loss": round(float(min(pnls)), 6) if pnls else None,
            })
    # 最近一次 A/B 结果落盘 results/policy_ab_latest.json（回测页「最近 A/B 冠军方案」）
    try:
        from datetime import datetime as _dt_ab, timezone as _tz_ab

        ranked = sorted(out, key=lambda r: (r.get("sharpe") is not None,
                                            float(r.get("sharpe") or float("-inf"))),
                        reverse=True)
        best = ranked[0] if ranked else None
        latest = {
            "generated_at": _dt_ab.now(_tz_ab.utc).isoformat(),
            "symbol": sym,
            "data_file": data_file,
            "bars": T - start,
            "params": {
                "window_bars": window,
                "commission_pct": comm,
                "slippage_pct": slip,
                "max_position_pct": pos_pct,
                "signal_threshold": sig_thr,
                "caps": caps,
            },
            "policies": pids,
            "best": best,
            "results": ranked,
        }
        ab_path = ROOT / "results" / "policy_ab_latest.json"
        ab_path.parent.mkdir(parents=True, exist_ok=True)
        ab_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 持久化失败不影响本次返回
        pass
    return {
        "ok": True,
        "symbol": sym,
        "data_file": data_file,
        "bars": T - start,
        "results": out,
    }


@app.get("/api/backtest/policy-ab-latest")
def api_backtest_policy_ab_latest(symbol: str | None = None) -> dict[str, Any]:
    """最近一次 A/B 对比结果（results/policy_ab_latest.json），供回测页「最近 A/B 冠军方案」。"""
    ab_path = ROOT / "results" / "policy_ab_latest.json"
    if not ab_path.exists():
        return {"exists": False, "latest": None}
    try:
        latest = json.loads(ab_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"exists": False, "latest": None}
    if symbol:
        sym = str(symbol).strip()
        if latest.get("symbol") and latest["symbol"] != sym:
            return {"exists": False, "latest": None}
    return {"exists": True, "latest": latest}


class VerifyRollbackRequest(BaseModel):
    strategy_file: str
    retrain_steps: int = 0
    seed: int = 42


@app.post("/api/training/verify-rollback")
def api_training_verify_rollback(req: VerifyRollbackRequest) -> dict[str, Any]:
    """训练页冠军条目「回滚校验」：起子进程跑 scripts/verify_champion_rollback.py。

    --retrain-steps>0 时还会做短程重训对照（验证同种子/步数能否复现存档 best）。
    结果写入 results/verify_champion_rollback_<stamp>.json/.md，前端轮询 status 展示。
    """
    from web.verify_rollback_manager import get_manager

    sf = (req.strategy_file or "").strip()
    if not sf:
        raise HTTPException(400, "缺少 strategy_file")
    p = Path(sf)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise HTTPException(400, f"策略文件不存在: {p}")
    try:
        job = get_manager().start(str(sf), retrain_steps=int(req.retrain_steps or 0),
                                   seed=int(req.seed or 42))
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict()}


@app.get("/api/training/verify-rollback/status")
def api_training_verify_rollback_status() -> dict[str, Any]:
    from web.verify_rollback_manager import get_manager

    return get_manager().status()


@app.post("/api/backtest/cost-sweep")
def api_backtest_cost_sweep(req: CostSweepRequest) -> dict[str, Any]:
    """成本敏感性：同一模型×方案在 0→高 成本档位下重跑离散撮合，找盈亏平衡点。

    轴 = 每边总成本（手续费+滑点 %），手续费/滑点按当前基数的固定比例拆分；
    返回每档 total_return / sharpe / 盈亏比 / 交易数，以及 净收益由正转负 与
    盈亏比跌破 1 的两个 break-even（线性插值）——量化模型真实 edge。
    """
    import numpy as np
    import torch
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.features import FeatureEngineer
    from model_core.vm import StackVM
    from web.hold_policy import combo_id, combo_label
    from web.paper_replay import run_replay

    strategy_file = (req.strategy_file or "").strip()
    sf_p = Path(strategy_file)
    if not sf_p.is_absolute():
        sf_p = ROOT / sf_p
    if not sf_p.exists():
        raise HTTPException(400, f"策略文件不存在: {sf_p}")
    meta = json.loads(sf_p.read_text(encoding="utf-8"))
    formula = meta.get("formula")
    if not formula:
        raise HTTPException(400, "策略文件缺少 formula")
    data_file = (req.data_file or "").strip()
    if not data_file:
        data_file = str(meta.get("data_file") or (meta.get("data_source") or {}).get("data_file") or "")
    if data_file:
        df_p = Path(data_file)
        if not df_p.is_absolute():
            df_p = ROOT / df_p
        data_file = str(df_p.resolve())
    if not data_file or not Path(data_file).exists():
        raise HTTPException(400, "无法定位回测数据文件，请显式选择")

    pm = ParquetDataManager(data_file)
    pm.load()
    raw = pm.raw_dict
    T = int(raw["close"].shape[1])
    window = req.window_bars
    start = 0
    if window:
        window = int(window)
        if window < 800:
            raise HTTPException(400, f"窗口过小（{window} 根）：特征 warm-up 需 ≥800 根")
        start = max(0, T - window)
    if T - start < 800:
        raise HTTPException(400, f"数据不足（{T - start} 根）")
    raw_s = {k: v[:, start:] for k, v in raw.items()}
    try:
        feats = FeatureEngineer.compute_features(raw_s)
        vm = StackVM()
        with torch.no_grad():
            factor = vm.execute([int(t) for t in formula], feats)
        if factor is None or factor.ndim != 2:
            raise HTTPException(400, "公式在该数据上无有效输出")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"因子计算失败: {exc}") from exc

    pid = combo_id(req.hold_policy or "signal") or "signal"
    base_comm = float(req.commission_pct if req.commission_pct is not None else 0.02)
    base_slip = float(req.slippage_pct if req.slippage_pct is not None else 0.01)
    base_total = max(base_comm + base_slip, 1e-9)
    frac_c = base_comm / base_total
    frac_s = base_slip / base_total
    pos_pct = _resolve_pos_pct(req.max_position_pct)
    sig_thr = _resolve_signal_threshold(req.signal_threshold)
    factor_np = factor[0].cpu().numpy().astype(float)
    levels = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]

    rows = []
    for L in levels:
        comm = L * frac_c
        slip = L * frac_s
        r = run_replay(factor=factor_np, open_p=raw_s["open"][0].numpy(),
                       high_p=raw_s["high"][0].numpy(), low_p=raw_s["low"][0].numpy(),
                       close_p=raw_s["close"][0].numpy(),
                       commission_pct=comm, slippage_pct=slip, policy_id=pid,
                       max_position_pct=pos_pct, threshold=sig_thr)
        st = r["stats"]
        rows.append({
            "cost_pct": round(L, 4),
            "commission_pct": round(comm, 5),
            "slippage_pct": round(slip, 5),
            "total_return": st.get("total_return"),
            "sharpe": st.get("sharpe"),
            "profit_loss_ratio": st.get("profit_loss_ratio"),
            "n_trades": st.get("n_trades"),
            "fees_total": st.get("fees_total"),
        })

    def _cross(rows_: list[dict], key: str, ref: float) -> float | None:
        """找 key 从 >ref 翻到 <ref 的成本档（线性插值到 ref）。"""
        prev_v = None
        for i, row in enumerate(rows_):
            v = row[key]
            if v is None:
                prev_v = None
                continue
            if prev_v is not None and prev_v > ref >= v:
                denom = prev_v - v
                if denom > 1e-12:
                    c0, c1 = rows_[i - 1]["cost_pct"], row["cost_pct"]
                    return round(c0 + (c1 - c0) * (prev_v - ref) / denom, 4)
            prev_v = v
        return None

    # 净收益由正转负（0）与 盈亏比跌破 1（edge 消失）
    bre_ret = _cross(rows, "total_return", 0.0)
    bre_plr = _cross(rows, "profit_loss_ratio", 1.0)
    return {
        "ok": True,
        "strategy_file": str(sf_p),
        "data_file": data_file,
        "bars": T - start,
        "policy_id": pid,
        "policy_label": combo_label(pid),
        "current_cost_pct": round(base_total, 4),
        "break_even_return": bre_ret,
        "break_even_plr": bre_plr,
        "rows": rows,
    }


@app.post("/api/backtest/threshold-sweep")
def api_backtest_threshold_sweep(req: ThresholdSweepRequest) -> dict[str, Any]:
    """阈值敏感性：同一模型×方案在 0.05/0.3/0.5/0.8 无信号阈值下重跑离散撮合。

    轴 = 无信号阈值（|tanh(因子)| < t → FLAT 观望），成本/窗口/方案固定取当前设置；
    返回每档 收益/夏普/回撤/交易数/盈亏比/胜率 + 观望占比，并按夏普给一行决策建议。
    """
    import numpy as np
    import torch
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.features import FeatureEngineer
    from model_core.vm import StackVM
    from web.hold_policy import combo_id, combo_label
    from web.paper_replay import WARMUP_BARS, run_replay

    strategy_file = (req.strategy_file or "").strip()
    sf_p = Path(strategy_file)
    if not sf_p.is_absolute():
        sf_p = ROOT / sf_p
    if not sf_p.exists():
        raise HTTPException(400, f"策略文件不存在: {sf_p}")
    meta = json.loads(sf_p.read_text(encoding="utf-8"))
    formula = meta.get("formula")
    if not formula:
        raise HTTPException(400, "策略文件缺少 formula")
    data_file = (req.data_file or "").strip()
    if not data_file:
        data_file = str(meta.get("data_file") or (meta.get("data_source") or {}).get("data_file") or "")
    if data_file:
        df_p = Path(data_file)
        if not df_p.is_absolute():
            df_p = ROOT / df_p
        data_file = str(df_p.resolve())
    if not data_file or not Path(data_file).exists():
        raise HTTPException(400, "无法定位回测数据文件，请显式选择")

    pm = ParquetDataManager(data_file)
    pm.load()
    raw = pm.raw_dict
    T = int(raw["close"].shape[1])
    window = req.window_bars
    start = 0
    if window:
        window = int(window)
        if window < 800:
            raise HTTPException(400, f"窗口过小（{window} 根）：特征 warm-up 需 ≥800 根")
        start = max(0, T - window)
    if T - start < 800:
        raise HTTPException(400, f"数据不足（{T - start} 根）")
    raw_s = {k: v[:, start:] for k, v in raw.items()}
    try:
        feats = FeatureEngineer.compute_features(raw_s)
        vm = StackVM()
        with torch.no_grad():
            factor = vm.execute([int(t) for t in formula], feats)
        if factor is None or factor.ndim != 2:
            raise HTTPException(400, "公式在该数据上无有效输出")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"因子计算失败: {exc}") from exc

    pid = combo_id(req.hold_policy or "signal") or "signal"
    comm = float(req.commission_pct if req.commission_pct is not None else 0.02)
    slip = float(req.slippage_pct if req.slippage_pct is not None else 0.01)
    pos_pct = _resolve_pos_pct(req.max_position_pct)
    factor_np = factor[0].cpu().numpy().astype(float)
    # 缺省 0.05/0.3/0.5/0.8 四档（与页面阈值下拉一致）；自定义档位也钳到 (0, 1)
    levels = sorted({round(float(t), 4) for t in (req.thresholds or [0.05, 0.3, 0.5, 0.8])
                     if 0 < float(t) < 1})
    if not levels:
        levels = [0.05, 0.3, 0.5, 0.8]

    # 信号强度序列（引擎同口径：pos = tanh(因子)），观望占比/方向分布按 warm-up 之后统计
    pos_np = np.abs(np.tanh(factor_np))
    wu = max(800, int(WARMUP_BARS))
    seg = pos_np[wu:] if pos_np.size > wu else pos_np

    rows = []
    for t in levels:
        r = run_replay(factor=factor_np, open_p=raw_s["open"][0].numpy(),
                       high_p=raw_s["high"][0].numpy(), low_p=raw_s["low"][0].numpy(),
                       close_p=raw_s["close"][0].numpy(),
                       commission_pct=comm, slippage_pct=slip, policy_id=pid,
                       max_position_pct=pos_pct, threshold=t)
        st = r["stats"]
        flat_share = float(np.mean(seg < t)) if seg.size else 0.0
        # 方向分布（bar 收盘信号，含观望）：多/空/观望 各占多少
        n_long = int(np.sum(seg >= t)) if seg.size else 0
        n_flat = int(np.sum(seg < t)) if seg.size else 0
        pnls = [tr["pnl"] for tr in r["trades"]]
        rows.append({
            "threshold": float(t),
            "flat_share": round(flat_share, 4),
            "flat_bars": int(n_flat),
            "signal_bars": int(n_long),
            "total_return": st.get("total_return"),
            "sharpe": st.get("sharpe"),
            "sortino": st.get("sortino"),
            "max_drawdown": st.get("max_drawdown"),
            "n_trades": st.get("n_trades"),
            "win_rate": st.get("win_rate"),
            "profit_loss_ratio": st.get("profit_loss_ratio"),
            "fees_total": st.get("fees_total"),
            "avg_hold_bars": st.get("avg_hold_bars"),
            "max_win": round(float(max(pnls)), 6) if pnls else None,
            "max_single_loss": round(float(min(pnls)), 6) if pnls else None,
        })

    # 一行决策：夏普最高的档（交易数 ≥1）；并列时取交易数更足的一档
    best, second = _sweep_pick_best(rows)
    suggested = best["threshold"] if best else None
    suggested_reason = None
    if best:
        n_best = int(best["n_trades"] or 0)
        parts = [f"建议 t={best['threshold']:g}：夏普 {best['sharpe']:+.2f}、收益 {best['total_return'] * 100:+.1f}%、回撤 {best['max_drawdown'] * 100:.1f}%、交易 {n_best} 笔、观望占 {best['flat_share'] * 100:.0f}%"]
        if n_best <= 4:
            parts.append("⚠ 交易数过少，单笔主导风险高，建议用更长窗口复验后再定")
        elif second is not None:
            parts.append(f"次优 t={second['threshold']:g}（夏普 {second['sharpe']:+.2f}）可作稳健备选")
        suggested_reason = " ".join(parts)
    else:
        suggested_reason = "所有档位均无有效交易（该窗口/成本下无信号可成交），建议放宽窗口或降低成本再看"

    return {
        "ok": True,
        "strategy_file": str(sf_p),
        "data_file": data_file,
        "bars": T - start,
        "policy_id": pid,
        "policy_label": combo_label(pid),
        "commission_pct": round(comm, 5),
        "slippage_pct": round(slip, 5),
        "suggested_threshold": suggested,
        "suggested_reason": suggested_reason,
        "rows": rows,
    }


@app.get("/api/backtest/compare-summary")
def api_backtest_compare_summary(symbol: str | None = None) -> dict[str, Any]:
    """回测页默认模型卡片下方的 tail vs spread 对比摘要（同品种）。

    读取 results/compare_latest.json（compare_ranges 每次完成后写入），
    同时把 docs 报告 markdown 返回给前端展开查看。
    """
    from web.compare_report import light_summary, render_markdown, summary_for_symbol

    summary = summary_for_symbol(symbol)
    if summary is None:
        return {"exists": False, "summary": None, "markdown": None}
    return {
        "exists": True,
        "summary": light_summary(summary),
        "markdown": render_markdown(summary),
    }


@app.get("/api/backtest/hold-matrix/prefs")
def api_hold_matrix_prefs(symbol: str | None = None) -> dict[str, Any]:
    """全组合矩阵按品种记忆的工具栏设置（窗口模式/窗口/regime/块数/数据文件）。"""
    from web.hold_matrix_prefs import get_symbol, load_all

    if symbol:
        return {"symbol": symbol, "prefs": get_symbol(symbol)}
    return {"prefs": load_all()}


@app.get("/api/backtest/prefs")
def api_backtest_prefs(symbol: str | None = None) -> dict[str, Any]:
    """回测页持仓组合/窗口按品种记忆：优先品种记录，缺省回退全局设置。"""
    from web.bt_prefs import get_symbol

    settings = load_settings()
    if symbol:
        prefs = get_symbol(symbol)
        if prefs.get("hold_policy", "signal") != "signal" or prefs.get("window_bars"):
            return {"symbol": symbol, "prefs": prefs, "from_symbol": True}
        # 品种无记录 → 全局默认兜底（不落盘，避免污染品种记忆）
        return {
            "symbol": symbol,
            "prefs": {
                "hold_policy": settings.get("bt_hold_policy") or "signal",
                "window_bars": settings.get("bt_window_bars"),
            },
            "from_symbol": False,
        }
    from web.bt_prefs import load_all as _load_all

    return {"prefs": _load_all()}


@app.put("/api/backtest/prefs")
def api_backtest_prefs_put(req: SettingsRequest) -> dict[str, Any]:
    """保存某品种的回测持仓组合/窗口记忆（symbol 走 bt_prefs.json）。"""
    from web.bt_prefs import set_symbol

    symbol = str(getattr(req, "bt_prefs_symbol", None) or "").strip()
    hp = getattr(req, "bt_hold_policy", None)
    wb = getattr(req, "bt_window_bars", None)
    if not symbol:
        return {"ok": False, "error": "缺少 bt_prefs_symbol"}
    prefs = set_symbol(symbol, {"hold_policy": hp, "window_bars": wb})
    return {"ok": True, "symbol": symbol, "prefs": prefs}


@app.put("/api/backtest/hold-matrix/prefs")
def api_hold_matrix_prefs_put(req: HoldMatrixPrefsRequest) -> dict[str, Any]:
    from web.hold_matrix_prefs import set_symbol

    prefs = set_symbol(req.symbol, {
        "data_file": req.data_file,
        "window_bars": req.window_bars,
        "window_mode": req.window_mode or "tail",
        "regime": req.regime or "vol",
        "chunks": req.chunks if req.chunks else 4,
    })
    return {"ok": True, "symbol": req.symbol, "prefs": prefs}


@app.get("/api/backtest/hold-matrix/status")
def api_backtest_hold_matrix_status() -> dict[str, Any]:
    """全组合矩阵任务状态（含按日志估算的完成组合数）。"""
    return _get_hold_matrix_manager().status()


@app.get("/api/backtest/hold-matrix/curve")
def api_backtest_hold_matrix_curve(combo: str) -> dict[str, Any]:
    """单组合的资金曲线 + 滚动夏普（读 hold_matrix_curves_latest.npz 侧车）。

    页面点「按夏普排名」某一行时调用；数据来自最近一次全组合回测落盘的
    逐 bar equity/pnl，先裁掉开头的空仓平段再降采样到 ≤900 点。
    """
    from web.hold_matrix_curves import build_curve_response, load_annotations, load_curve
    from web.hold_policy import combo_id

    cid = combo_id(combo)
    loaded = load_curve(cid)
    if loaded is None:
        return {"available": False, "combo": cid, "error": "尚无该组合的曲线数据 — 先跑一次「三轴联合回测」生成侧车数据"}
    eq, pn, meta = loaded
    return build_curve_response(cid, eq, pn, meta, load_annotations(cid))


@app.get("/api/backtest/combo-sweep/curve")
def api_backtest_combo_sweep_curve(combo: str, cap_pct: float | None = None,
                                   threshold: float | None = None) -> dict[str, Any]:
    """三轴网格中任意一格 (组合 × 上限% × 无信号阈值) 的资金曲线 + 滚动夏普。

    基线切片（或未指定 上限/阈值）读 hold_matrix_curves_latest.npz 侧车 —— 与旧
    矩阵视图完全同构；非基线切片用同一离散撮合引擎（web.paper_replay.run_replay）
    对单格现场重放（与三轴回测同窗口/成本/因子，确定性复现网格行），再走与侧车
    相同的曲线装配逻辑。重放数据按数据窗口缓存，重复点击不重复加载。
    """
    from web.combo_sweep_curves import slice_curve

    result = slice_curve(combo, cap_pct, threshold)
    if result is None:
        # 无三轴网格 / 未指定或命中基线切片 → 走旧侧车路径（行为不变）
        return api_backtest_hold_matrix_curve(combo)
    return result
@app.post("/api/backtest/hold-matrix/stop")
def api_backtest_hold_matrix_stop() -> dict[str, Any]:
    stopped = _get_hold_matrix_manager().stop()
    return {"ok": stopped, "matrix": _get_hold_matrix_manager().status()}


_combo_sweep_manager: Any = None


def _get_combo_sweep_manager() -> Any:
    """惰性单例：跑 持仓×上限%×阈值 三轴全网格回测的子进程管理器。"""
    global _combo_sweep_manager
    if _combo_sweep_manager is None:
        from web.combo_sweep_manager import get_manager

        _combo_sweep_manager = get_manager()
    return _combo_sweep_manager


@app.get("/api/backtest/combo-sweep")
def api_backtest_combo_sweep() -> dict[str, Any]:
    """三轴联合回测结果（读 results/combo_sweep_latest.json，含基线切片说明）。"""
    import json as _json

    path = ROOT / "results" / "combo_sweep_latest.json"
    if not path.exists():
        return {"available": False, "grid": None}
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError):
        return {"available": False, "grid": None}
    # 样本外溯源感知：按训练真 holdout 边界给窗口打 真伪 OOS 标签（tail 窗口才可判）
    if isinstance(data, dict) and data.get("window_bars") and data.get("window_mode") == "tail":
        try:
            from web.oos_provenance import classify_window

            def _abs(p: str | None) -> str | None:
                if not p:
                    return None
                pp = Path(p)
                return str((ROOT / pp) if not pp.is_absolute() else pp)

            data = dict(data)
            data["oos"] = classify_window(
                _abs(data.get("strategy_file")),
                _abs(data.get("data_file")),
                data.get("window_start"),
                data.get("window_bars"),
            )
        except Exception:  # noqa: BLE001 溯源失败不阻断展示
            pass
    return {"available": True, "grid": data}


@app.post("/api/backtest/combo-sweep/run")
def api_backtest_combo_sweep_run(req: ComboSweepRunRequest) -> dict[str, Any]:
    """启动三轴联合回测：22 组合 × 上限%档 × 阈值档 全网格（子进程跑，可轮询/停止）。

    结果写 results/combo_sweep_latest.json + 基线切片（第一档上限×第一档阈值）
    同步写 hold_matrix_latest.json（矩阵视图/曲线/帕累托沿用旧格式）。
    """
    strategy_file, data_file = _resolve_matrix_defaults(req)
    window_bars = req.window_bars
    if window_bars is not None:
        window_bars = int(window_bars)
        if window_bars <= 0:
            window_bars = None
    try:
        job = _get_combo_sweep_manager().start(
            strategy_file=strategy_file,
            data_file=data_file,
            commission_pct=req.commission_pct if req.commission_pct is not None else 0.02,
            slippage_pct=req.slippage_pct if req.slippage_pct is not None else 0.01,
            window_bars=window_bars,
            window_mode=req.window_mode or "tail",
            regime=req.regime or "vol",
            chunks=req.chunks if req.chunks else 4,
            caps=_normalize_ab_caps(req.caps, 100.0) if req.caps is not None else [10.0, 25.0, 100.0],
            thresholds=[float(t) for t in (req.thresholds or [0.05, 0.3, 0.5, 0.8])],
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict()}


@app.get("/api/backtest/combo-sweep/status")
def api_backtest_combo_sweep_status() -> dict[str, Any]:
    """三轴联合回测任务状态（含按日志估算的完成行数 / 总行数）。"""
    return _get_combo_sweep_manager().status()


@app.post("/api/backtest/combo-sweep/stop")
def api_backtest_combo_sweep_stop() -> dict[str, Any]:
    stopped = _get_combo_sweep_manager().stop()
    return {"ok": stopped, "combo_sweep": _get_combo_sweep_manager().status()}


@app.get("/api/backtest/equity")
def api_backtest_equity(request: Request, response: Response, symbol: str | None = None) -> Any:
    """资金曲线原始数据（供前端渲染交互式 HTML 图表）。"""
    from fastapi.responses import Response as _R304  # response 形参遮蔽了类名
    import json

    path = BACKTEST_OUTPUT_DIR / "equity_curve.json"
    if path.exists():
        etag = _file_etag(path)
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "no-cache"  # 每次重验证（304 无 body）
        if request.headers.get("if-none-match") == etag:
            return _R304(status_code=304, headers={"ETag": etag})
    focus = _backtest_focus_symbol(symbol)
    if not path.exists():
        return {"available": False, "focus_symbol": focus, "data": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available": False, "focus_symbol": focus, "data": None}

    # 单品种模式：只保留聚焦品种，去掉无关序列
    if focus and isinstance(data.get("symbols"), dict) and focus in data["symbols"]:
        data = {
            **data,
            "symbols": {focus: data["symbols"][focus]},
        }
        data.pop("portfolio", None)

    return {"available": True, "focus_symbol": focus, "data": data}


@app.get("/api/backtest/chart/{name}")
def api_backtest_chart(name: str):
    # 防止路径穿越：仅允许输出目录内的 png 文件
    if "/" in name or "\\" in name or ".." in name or not name.lower().endswith(".png"):
        raise HTTPException(400, "非法文件名")
    path = (BACKTEST_OUTPUT_DIR / name).resolve()
    try:
        path.relative_to(BACKTEST_OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "非法路径") from None
    if not path.exists():
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────
# 实时行情分析 API
# ─────────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def _startup_events_bind() -> None:
    """把 SSE 事件中枢绑定到 uvicorn 事件循环（watcher/推送需要 loop 句柄）。"""
    import asyncio

    from web import events as _events

    try:
        _events.bind_loop(asyncio.get_running_loop())
    except RuntimeError:  # 兜底：无 loop 时首次订阅会懒绑定
        pass


@app.on_event("startup")
def _startup_realtime() -> None:
    try:
        realtime_manager.load_persisted()
    except Exception as exc:  # noqa: BLE001
        log_error("realtime load_persisted failed", exc)
    try:
        paper_manager.load_persisted()
    except Exception as exc:  # noqa: BLE001
        log_error("paper load_persisted failed", exc)


@app.get("/api/realtime/sources")
def api_realtime_sources() -> dict[str, Any]:
    return {
        "sources": list_sources(),
        "min_exposure": min_exposure(),  # 兼容旧前端
        "signal_threshold": _resolve_signal_threshold(None),
    }


@app.post("/api/realtime/tradingview/probe")
def api_realtime_tradingview_probe() -> dict[str, Any]:
    """Probe TradingView reachability (same behavior as PA_Agent before fetch)."""
    from web.data_sources.tradingview_connectivity import (
        TV_CLOUD_SERVER_WIKI_URL,
        TV_CONNECTIVITY_MESSAGE,
        check_tradingview_connectivity,
    )

    ok, detail = check_tradingview_connectivity(
        timeout_s=15.0, max_attempts=2, retry_delay_s=2.0
    )
    return {
        "ok": ok,
        "detail": detail,
        "blocked": not ok,
        "title": "无法使用 TradingView",
        "message": None if ok else TV_CONNECTIVITY_MESSAGE,
        "wiki_url": TV_CLOUD_SERVER_WIKI_URL,
    }


@app.get("/api/hold-policies")
def api_hold_policies() -> dict[str, Any]:
    from web.hold_policy import list_hold_policies

    return {"policies": list_hold_policies()}


@app.get("/api/realtime/strategies")
def api_realtime_strategies() -> dict[str, Any]:
    """已保存的 best_*.json 策略，供因子来源下拉。

    每个制品行必须指向它自己的文件（best_{symbol}_{tf}.json 与旧式
    best_{symbol}.json 并存时，不能按裸 symbol 重映射——否则 H1 行会
    指向 M5 文件，UI 显示与加载模型静默分叉）。
    """
    rows = []
    for s in list_strategies():
        path = s.get("strategy_file")
        if not path:
            continue
        rows.append(
            {
                "symbol": s.get("symbol"),
                "timeframe": s.get("timeframe"),
                "best_score": s.get("best_score"),
                "formula_decoded": s.get("formula_decoded"),
                "strategy_file": str(path),
            }
        )
    return {"strategies": rows}


@app.get("/api/realtime/status")
def api_realtime_status() -> dict[str, Any]:
    st = realtime_manager.status()
    # “两者结合”：若同一监控项在模拟盘正持仓，附上真实入场/止损/止盈（图表优先画实际线）
    try:
        paper = paper_manager.status()
        pos_by_watch = {p["watch_id"]: p for p in paper.get("positions") or []}
        for w in st.get("watches") or []:
            p = pos_by_watch.get(w.get("id"))
            if p:
                w["paper_position"] = {
                    "side": p.get("side"),
                    "entry_price": p.get("entry_price"),
                    "stop_price": p.get("stop_price"),
                    "target_price": p.get("target_price"),
                    "qty": p.get("qty"),
                    "mark_price": p.get("mark_price"),
                    "unrealized_pnl": p.get("unrealized_pnl"),
                }
    except Exception:  # noqa: BLE001 叠加失败不影响实时状态
        pass
    return st


@app.post("/api/realtime/watch")
def api_realtime_watch(req: AddWatchRequest) -> dict[str, Any]:
    try:
        watch = realtime_manager.add_watch(
            req.source, req.symbol, req.timeframe, req.strategy_file,
            policy_id=req.policy_id or "signal",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    logger.info("[watch] %s %s/%s 加入监控，加载策略 %s",
                req.source, req.symbol, req.timeframe, req.strategy_file)
    return {"ok": True, "watch": watch}


@app.post("/api/realtime/unwatch")
def api_realtime_unwatch(req: RemoveWatchRequest) -> dict[str, Any]:
    removed = realtime_manager.remove_watch(req.id)
    return {"ok": removed}


@app.post("/api/realtime/start")
def api_realtime_start() -> dict[str, Any]:
    realtime_manager.start()
    return {"ok": True, **realtime_manager.status()}


@app.post("/api/realtime/stop")
def api_realtime_stop() -> dict[str, Any]:
    realtime_manager.stop()
    return {"ok": True, "running": False}


@app.get("/api/realtime/feishu")
def api_realtime_feishu_get() -> dict[str, Any]:
    s = load_settings()
    return {
        "enabled": bool(s.get("feishu_enabled")),
        "webhook_url": s.get("feishu_webhook_url") or "",
        "secret": s.get("feishu_secret") or "",
        "rt_alert_dev_pct": s.get("rt_alert_dev_pct", 0.5),
        "rt_alert_stale_bars": int(s.get("rt_alert_stale_bars") or 0),
    }


@app.put("/api/realtime/feishu")
def api_realtime_feishu_put(req: FeishuSettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.enabled is not None:
        payload["feishu_enabled"] = bool(req.enabled)
    if req.webhook_url is not None:
        payload["feishu_webhook_url"] = req.webhook_url
    if req.secret is not None:
        payload["feishu_secret"] = req.secret
    if req.rt_alert_dev_pct is not None:
        payload["rt_alert_dev_pct"] = req.rt_alert_dev_pct
    if req.rt_alert_stale_bars is not None:
        payload["rt_alert_stale_bars"] = max(0, int(req.rt_alert_stale_bars))
    saved = save_settings(payload)
    return {
        "ok": True,
        "enabled": bool(saved.get("feishu_enabled")),
        "webhook_url": saved.get("feishu_webhook_url") or "",
        "secret": saved.get("feishu_secret") or "",
        "rt_alert_dev_pct": saved.get("rt_alert_dev_pct", 0.5),
        "rt_alert_stale_bars": int(saved.get("rt_alert_stale_bars") or 0),
    }


@app.post("/api/realtime/feishu/test")
def api_realtime_feishu_test(req: FeishuTestRequest) -> dict[str, Any]:
    from web.feishu_notify import send_text

    url = (req.webhook_url or "").strip()
    if not url:
        url = (load_settings().get("feishu_webhook_url") or "").strip()
    if not url:
        raise HTTPException(400, "请先填写 Webhook URL")
    secret = req.secret
    if secret is None:
        secret = load_settings().get("feishu_secret") or ""
    ok, msg = send_text(
        "✅ AlphaMaster 飞书通知测试：配置正常。信号方向转折时会推送提醒。",
        webhook_url=url,
        secret=secret or "",
    )
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


# ─────────────────────────────────────────────────────────────────────
# 模拟实盘（纸上交易）API
# ─────────────────────────────────────────────────────────────────────


@app.get("/api/paper/status")
def api_paper_status() -> dict[str, Any]:
    return paper_manager.status()


@app.post("/api/paper/dd-drill")
def api_paper_dd_drill() -> dict[str, Any]:
    """DD 演练：隔离子进程注入合成暴跌→收复，走真实生产 dd 熔断路径。

    返回完整转录（步骤/事件/流水/飞书结果）；演练在临时账户上进行，
    不影响正式模拟盘状态；飞书已配置时会真实推送「熔断」「收复」两条告警。
    """
    import subprocess

    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "dd_alert_drill.py"), "--json"],
            cwd=str(ROOT), timeout=180, capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "DD 演练超时（>180s，机器负载高时请稍后重试）") from None
    out_txt = (proc.stdout or "").strip()
    try:
        # stdout 可能带警告/日志行，取最后一个 JSON 块
        import json as _json

        idx = out_txt.find("{")
        data = _json.loads(out_txt[idx:]) if idx >= 0 else {}
    except Exception:  # noqa: BLE001
        data = {}
    if not data.get("ok"):
        raise HTTPException(500, f"DD 演练失败: {out_txt[:3000] or (proc.stderr or '')[:3000]}")
    return data


@app.get("/api/dd-events")
def api_dd_events(limit: int = 50) -> dict[str, Any]:
    """最近 N 条 DD 熔断/收复事件（回溯每次熔断/收复）。"""
    from web.dd_events import recent_events

    return {"ok": True, "events": recent_events(max(1, min(500, limit)))}


@app.post("/api/paper/start")
def api_paper_start() -> dict[str, Any]:
    paper_manager.start()
    st = paper_manager.status()
    return {"ok": True, **st}


@app.post("/api/paper/stop")
def api_paper_stop() -> dict[str, Any]:
    paper_manager.stop()
    return {"ok": True, "running": False}


def _replay_window_spec(window_bars: int | None, T: int) -> tuple[int, int]:
    """回放窗口解析：window_bars = 总切片长，前 WARMUP_BARS 根是 warm-up
    （run_replay 从第 WARMUP_BARS 根才开始交易）。

    返回 (start, span)：start 为切片起点，span 为实际参与 bar 数。
    契约（2026-09-06 修复）：
    - None → 整文件；T ≤ WARMUP_BARS 时无样本 → 400。
    - w ≤ WARMUP_BARS → 400：整片都是 warm-up，回放将返回空结果
      （0 笔交易/sharpe 0 的误导性 ok），必须 > WARMUP_BARS 才有可交易样本。
    - w 超出文件长度 → 从头起、span=T（与旧行为一致）。
    """
    from web.paper_replay import WARMUP_BARS

    if window_bars is None:
        if T <= WARMUP_BARS:
            raise HTTPException(
                400, f"数据不足（{T} 根）：前 {WARMUP_BARS} 根为 warm-up，无样本可回放")
        return 0, T
    w = int(window_bars)
    if w <= 0:
        raise HTTPException(400, f"窗口必须为正整数（收到 {window_bars}）")
    if w <= WARMUP_BARS:
        raise HTTPException(
            400, f"窗口过小（{w} 根）：前 {WARMUP_BARS} 根是 warm-up（回放自第 "
            f"{WARMUP_BARS} 根才开始交易），需 > {WARMUP_BARS} 才有可交易样本；"
            f"实际可交易 bar = 窗口 − {WARMUP_BARS}")
    start = max(0, T - w)
    span = min(w, T)
    if span <= WARMUP_BARS:
        raise HTTPException(400, f"数据不足（{T} 根）：不足 {WARMUP_BARS} 根 warm-up + 样本")
    return start, span


@app.post("/api/paper/replay")
def api_paper_replay(req: PaperReplayRequest) -> dict[str, Any]:
    """历史回放：本地 Parquet × 模型 × 持仓管理方案，离散撮合（与模拟盘同口径）。

    同步执行（FastAPI 线程池运行），大文件可能耗时数十秒；前端等待期间显示加载态。
    """
    import numpy as np
    import torch
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.backtest import estimate_periods_per_year
    from model_core.features import FeatureEngineer
    from model_core.vm import StackVM
    from web.paper_replay import run_replay
    from web.realtime_manager import load_strategy_meta
    from web.hold_policy import combo_id

    data_file = (req.data_file or "").strip()
    strategy_file = (req.strategy_file or "").strip()
    # launchd/web 服务的 cwd 不一定是项目根 → 相对路径按项目根解析
    df_p = Path(data_file)
    if not df_p.is_absolute():
        df_p = ROOT / df_p
    sf_p = Path(strategy_file)
    if not sf_p.is_absolute():
        sf_p = ROOT / sf_p
    data_file = str(df_p.resolve())
    strategy_file = str(sf_p.resolve())
    if not df_p.exists():
        raise HTTPException(400, f"数据文件不存在: {data_file}")
    if not sf_p.exists():
        raise HTTPException(400, f"策略文件不存在: {strategy_file}")
    meta = load_strategy_meta(strategy_file)
    formula = meta.get("formula")
    if not formula:
        raise HTTPException(400, "策略文件缺少 formula")
    logger.info("[replay] 加载策略 %s (symbol=%s tf=%s) 数据 %s",
                strategy_file, meta.get("symbol"), meta.get("timeframe"), data_file)

    pm = ParquetDataManager(str(Path(data_file).resolve()))
    pm.load()
    raw = pm.raw_dict
    T = int(raw["close"].shape[1])
    start, window = _replay_window_spec(req.window_bars, T)
    raw_s = {k: v[:, start:] for k, v in raw.items()}

    try:
        feats = FeatureEngineer.compute_features(raw_s)
        vm = StackVM()
        with torch.no_grad():
            factor = vm.execute([int(t) for t in formula], feats)
        if factor is None or factor.ndim != 2 or factor.shape[1] < 2:
            raise HTTPException(400, "公式在该数据上无有效输出")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"因子计算失败: {exc}") from exc

    comm = req.commission_pct if req.commission_pct is not None else 0.02
    slip = req.slippage_pct if req.slippage_pct is not None else 0.01
    pos_pct = _resolve_pos_pct(req.max_position_pct)
    sig_thr = _resolve_signal_threshold(req.signal_threshold)
    pid = combo_id(req.policy_id)
    times = raw_s.get("time")
    ppy = estimate_periods_per_year(times) if times is not None else 105195.0
    rep = run_replay(
        factor=factor[0].cpu().numpy().astype(float),
        open_p=raw_s["open"][0].numpy(),
        high_p=raw_s["high"][0].numpy(),
        low_p=raw_s["low"][0].numpy(),
        close_p=raw_s["close"][0].numpy(),
        commission_pct=comm,
        slippage_pct=slip,
        policy_id=pid,
        max_position_pct=pos_pct,
        threshold=sig_thr,
        periods_per_year=float(ppy),
    )
    eq = rep["equity"]
    n = eq.shape[0]
    from web.paper_replay import WARMUP_BARS as _WARMUP_BARS
    usable_bars = max(0, int(n) - _WARMUP_BARS)
    max_pts = 1500
    idx = np.unique(np.linspace(0, n - 1, min(max_pts, n)).astype(int))
    labels = []
    if times is not None:
        from datetime import datetime, timezone
        for i in idx:
            labels.append(datetime.fromtimestamp(
                int(times[0, int(i)].item()), tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"))
    else:
        labels = [str(int(i)) for i in idx]
    stats = rep["stats"]
    # ── 逐年度绩效拆解：按数据时间戳把窗口净值变化分桶到年，标出贡献/损失最多年 ──
    yearly: list[dict[str, Any]] = []
    best_year: dict[str, Any] | None = None
    worst_year: dict[str, Any] | None = None
    if times is not None and n > 1:
        import numpy as _np
        from collections import defaultdict

        ts_win = times[0].numpy().astype(float)
        pnl_win = rep["pnl"]
        # 真年份分桶（向量化 datetime64）：固定 365.25d 粗桶的边界会落在每年
        # 12 月下旬，把「2017-08..12」错拆成两个桶，这里按 UTC 真年份取 year。
        years_dt = ts_win.astype("datetime64[s]").astype("datetime64[Y]")
        year_int = years_dt.astype("int64") + 1970          # datetime64[Y] = years since 1970
        # 先按年聚合逐 bar pnl，再在年内序列上算最大回撤（权益相对滚动峰）
        by_year_pnl: dict[int, list[int]] = defaultdict(list)
        for i, yr in enumerate(year_int.tolist()):
            by_year_pnl[int(yr)].append(i)
        for yr, idxs in sorted(by_year_pnl.items()):
            idxs_arr = _np.asarray(idxs, dtype=int)
            seg_pnl = pnl_win[idxs_arr]
            seg_eq = _np.cumsum(seg_pnl)
            yr_ret = float(seg_eq[-1]) if seg_eq.size else 0.0
            peak = _np.maximum.accumulate(_np.concatenate([_np.zeros(1), seg_eq]))[1:]
            dd = float((seg_eq - peak).min()) if seg_eq.size else 0.0
            # 该年交易笔数（按出场 bar 落点）
            idx_set = set(int(i) for i in idxs_arr.tolist())
            n_tr = sum(1 for tr in rep["trades"] if int(tr.get("bar") or 0) in idx_set)
            row = {
                "year": str(int(yr)),
                "bars": int(idxs_arr.size),
                "return": round(yr_ret, 6),
                "max_drawdown": round(dd, 6),
                "n_trades": int(n_tr),
                "sharpe": (round(float(seg_pnl.mean() / seg_pnl.std() * _np.sqrt(float(ppy))), 3)
                           if seg_pnl.size > 2 and float(seg_pnl.std()) > 1e-12 else None),
            }
            yearly.append(row)
            if best_year is None or row["return"] > float(best_year["return"]):
                best_year = row
            if worst_year is None or row["return"] < float(worst_year["return"]):
                worst_year = row
    # ── 样本外溯源：窗口是否超出训练范围（供前端弹「样本外部署」警告 + 诚实 OOS 建议） ──
    oos_info: dict[str, Any] | None = None
    try:
        from web.oos_provenance import classify_window

        oos_info = classify_window(strategy_file, data_file, start, window or n)
        if oos_info and oos_info.get("status") == "unavailable":
            oos_info = None
    except Exception:  # noqa: BLE001 溯源失败不影响回放本身
        oos_info = None
    return {
        "ok": True,
        "policy_id": pid,
        "bars": int(n),
        "window_start": int(start),
        # 可交易样本口径：前 WARMUP_BARS 根是 warm-up（run_replay 自第 800 根
        # 才交易），stats 只覆盖该段 —— 不同 window 的可比性看 usable_bars
        "warmup_bars": _WARMUP_BARS,
        "usable_bars": usable_bars,
        "stats": stats,
        "equity": {"labels": labels, "equity": [round(float(eq[int(i)]), 6) for i in idx]},
        "trades_sample": rep["trades"][-20:],
        "yearly": yearly,
        "best_year": best_year,
        "worst_year": worst_year,
        "oos": oos_info,
    }


@app.post("/api/paper/replay-compare")
def api_paper_replay_compare(req: PaperReplayRequest) -> dict[str, Any]:
    """同参数回测对比：与历史回放完全相同的 数据/模型/持仓方案/窗口/成本，
    再跑一遍「回测口径」并返回两条资金曲线叠加。

    因子只算一次（特征/VM 是大头），两个引擎共享同一份因子序列：
    - 回放 = run_replay 离散撮合（模拟实盘同口径）；
    - 回测 = 持仓方案为 signal 时走训练连续口径 BacktestEngine，
            带出场模块（非 signal）时同样走离散撮合 → 两条曲线应重合，
            相当于一次引擎一致性校验。
    """
    import numpy as np
    import torch
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.backtest import estimate_periods_per_year
    from model_core.features import FeatureEngineer
    from model_core.vm import StackVM
    from web.paper_replay import run_replay
    from web.realtime_manager import load_strategy_meta
    from web.hold_policy import combo_id
    from run_backtest import calc_max_drawdown, calc_sharpe, calc_sortino

    data_file = (req.data_file or "").strip()
    strategy_file = (req.strategy_file or "").strip()
    df_p = Path(data_file)
    if not df_p.is_absolute():
        df_p = ROOT / df_p
    sf_p = Path(strategy_file)
    if not sf_p.is_absolute():
        sf_p = ROOT / sf_p
    data_file = str(df_p.resolve())
    strategy_file = str(sf_p.resolve())
    if not df_p.exists():
        raise HTTPException(400, f"数据文件不存在: {data_file}")
    if not sf_p.exists():
        raise HTTPException(400, f"策略文件不存在: {strategy_file}")
    meta = load_strategy_meta(strategy_file)
    formula = meta.get("formula")
    if not formula:
        raise HTTPException(400, "策略文件缺少 formula")
    sym = meta.get("symbol") or Path(strategy_file).stem.replace("best_", "", 1)

    pm = ParquetDataManager(str(Path(data_file).resolve()))
    pm.load()
    raw = pm.raw_dict
    T = int(raw["close"].shape[1])
    start, window = _replay_window_spec(req.window_bars, T)
    raw_s = {k: v[:, start:] for k, v in raw.items()}

    try:
        feats = FeatureEngineer.compute_features(raw_s)
        vm = StackVM()
        with torch.no_grad():
            factor = vm.execute([int(t) for t in formula], feats)
        if factor is None or factor.ndim != 2 or factor.shape[1] < 2:
            raise HTTPException(400, "公式在该数据上无有效输出")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"因子计算失败: {exc}") from exc

    comm = req.commission_pct if req.commission_pct is not None else 0.02
    slip = req.slippage_pct if req.slippage_pct is not None else 0.01
    pos_pct = _resolve_pos_pct(req.max_position_pct)
    sig_thr = _resolve_signal_threshold(req.signal_threshold)
    pid = combo_id(req.policy_id)
    times = raw_s.get("time")
    ppy = estimate_periods_per_year(times) if times is not None else 105195.0

    factor_np = factor[0].cpu().numpy().astype(float)
    open_np = raw_s["open"][0].numpy()
    high_np = raw_s["high"][0].numpy()
    low_np = raw_s["low"][0].numpy()
    close_np = raw_s["close"][0].numpy()

    # ① 回放：离散撮合（模拟实盘口径）
    rep = run_replay(
        factor=factor_np, open_p=open_np, high_p=high_np,
        low_p=low_np, close_p=close_np,
        commission_pct=comm, slippage_pct=slip,
        policy_id=pid, max_position_pct=pos_pct,
        threshold=sig_thr,
        periods_per_year=float(ppy),
    )
    eq_replay = np.asarray(rep["equity"], dtype=float)

    # ② 回测口径：signal → 训练连续引擎；非 signal → 同一离散引擎（一致性校验）
    if pid == "signal":
        from backtest_viz import BacktestEngine

        cost_rate = (comm + slip) / 100.0
        engine = BacktestEngine(formula=[int(t) for t in formula],
                                cost_rate=cost_rate, periods_per_year=int(ppy),
                                max_position_pct=pos_pct, min_abs=sig_thr)
        with torch.no_grad():
            sym_res = engine.run(raw_s, feats, [sym])
        r = sym_res[0]
        cum_bt = np.asarray(r.cum_pnl, dtype=float)  # 连续口径 = 累计对数收益
        eq_bt = np.exp(cum_bt)  # 净值轴（与回放侧 1.0 起步的净值同语义）
        bt_stats = {
            # 与回放侧统计同口径：真实简单收益率（log → expm1）
            "total_return": float(np.expm1(float(r.total_return))),
            "sharpe": calc_sharpe(np.asarray(r.pnl, dtype=float), ppy),
            "sortino": calc_sortino(np.asarray(r.pnl, dtype=float), ppy),
            "max_drawdown": calc_max_drawdown(np.expm1(cum_bt)),
            "n_trades": int(r.n_trades),
            "win_rate": float(r.win_rate),
            "profit_loss_ratio": float(r.profit_loss_ratio) if r.profit_loss_ratio is not None else None,
            "avg_hold_bars": float(r.avg_hold_bars),
            "engine": "连续（训练口径）",
        }
    else:
        eq_bt = eq_replay
        bt_stats = dict(rep["stats"])
        bt_stats["engine"] = "离散撮合（与回放同引擎，应为一致校验）"

    # 共享下采样与时间标签（两引擎曲线都与窗口 bar 对齐、同长度）
    n = eq_replay.shape[0]
    from web.paper_replay import WARMUP_BARS as _WARMUP_BARS
    usable_bars = max(0, int(n) - _WARMUP_BARS)
    if eq_bt.shape[0] != n:
        # 引擎长度意外不一致时按自身 0..end 对齐（保守回退）
        m = eq_bt.shape[0]
        idx_bt = np.unique(np.linspace(0, m - 1, n).astype(int))
        eq_bt = eq_bt[idx_bt]
    max_pts = 1500
    idx = np.unique(np.linspace(0, n - 1, min(max_pts, n)).astype(int))
    labels = []
    if times is not None:
        from datetime import datetime, timezone

        for i in idx:
            labels.append(datetime.fromtimestamp(
                int(times[0, int(i)].item()), tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"))
    else:
        labels = [str(int(i)) for i in idx]
    st_replay = rep["stats"]
    st_replay["engine"] = "离散撮合（模拟实盘口径）"
    return {
        "ok": True,
        "policy_id": pid,
        "bars": int(n),
        "window_start": int(start),
        "warmup_bars": _WARMUP_BARS,
        "usable_bars": usable_bars,
        "replay": {
            "stats": st_replay,
            "equity": [round(float(eq_replay[int(i)]), 6) for i in idx],
        },
        "backtest": {
            "stats": bt_stats,
            "equity": [round(float(eq_bt[int(i)]), 6) for i in idx],
        },
        "labels": labels,
        "policy_note": (
            "signal：回放=离散撮合（含成本），回测=训练连续口径；两者差异≈撮合方式/成本口径差异。"
            if pid == "signal"
            else "带出场模块：回测与回放同走离散撮合引擎，两条曲线重合 = 引擎一致性校验通过。"
        ),
    }


@app.post("/api/paper/watch")
def api_paper_watch(req: PaperWatchRequest) -> dict[str, Any]:
    try:
        watch = paper_manager.add_watch(
            req.source, req.symbol, req.timeframe, req.strategy_file,
            policy_id=req.policy_id or "signal",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    logger.info("[watch] %s %s/%s 加入模拟盘，加载策略 %s",
                req.source, req.symbol, req.timeframe, req.strategy_file)
    return {"ok": True, "watch": watch}


@app.post("/api/paper/unwatch")
def api_paper_unwatch(req: PaperActionRequest) -> dict[str, Any]:
    removed = paper_manager.remove_watch(req.id)
    return {"ok": removed}


@app.post("/api/paper/close")
def api_paper_close(req: PaperActionRequest) -> dict[str, Any]:
    closed = paper_manager.close_position(req.id)
    if not closed:
        raise HTTPException(400, "该监控项没有未平持仓（或监控项不存在）")
    return {"ok": True, **paper_manager.status()}


@app.post("/api/paper/close-all")
def api_paper_close_all() -> dict[str, Any]:
    count = paper_manager.close_all()
    return {"ok": True, "closed": count, **paper_manager.status()}


@app.post("/api/paper/reset")
def api_paper_reset(req: PaperResetRequest | None = None) -> dict[str, Any]:
    paper_manager.reset(
        starting_balance=req.starting_balance if req and req.starting_balance is not None else None
    )
    return {"ok": True, **paper_manager.status()}


@app.get("/favicon.ico")
def favicon_ico() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/docs/engines-compare")
def docs_engines_compare() -> FileResponse:
    """回测 vs 回放 vs 模拟实盘口径对照表（各页 hint 链接到此处）。"""
    p = ROOT / "docs" / "口径对照_回测回放模拟实盘.md"
    if not p.exists():
        raise HTTPException(404, "口径对照文档缺失")
    return FileResponse(p, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": "inline"})
