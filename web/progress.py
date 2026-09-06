"""Read training progress from checkpoints and strategy files."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"


def _safe_symbol_tag(symbol: str) -> str:
    return symbol.replace(".", "_")


def checkpoint_glob(symbol: str) -> list[Path]:
    tag = _safe_symbol_tag(symbol)
    patterns = [
        f"ckpt_{symbol}_step_*.pt",
        f"ckpt_{tag}_step_*.pt",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(CHECKPOINT_DIR.glob(pattern))
    return sorted(set(found), key=lambda p: p.stat().st_mtime)


def _step_from_name(path: Path) -> int:
    m = re.search(r"_step_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else 0


@dataclass
class SymbolProgress:
    symbol: str
    train_steps: int
    current_step: int
    best_score: float | None
    best_formula: list[int] | None
    formula_decoded: str | None
    has_strategy: bool
    strategy_score: float | None
    checkpoint_path: str | None
    checkpoint_mtime: float | None
    history: dict[str, Any] | None
    holdout_bars: int | None = None
    holdout: dict[str, Any] | None = None

    @property
    def progress_pct(self) -> float:
        if self.train_steps <= 0:
            return 0.0
        return min(100.0, 100.0 * self.current_step / self.train_steps)

    @property
    def status(self) -> str:
        if self.current_step >= self.train_steps and self.has_strategy:
            return "completed"
        if self.current_step > 0:
            return "in_progress"
        if self.has_strategy:
            return "strategy_only"
        return "idle"


_ckpt_cache: dict[str, tuple[float, dict[str, Any]]] = {}
# 实时 history JSON 解析缓存（按 mtime）；轮询 /api/symbols/{sym} 时避免每 ~4s
# 整份 json.loads（9000 步长跑该文件可达 ~MB 级）。配合 HISTORY_LIVE_EVERY_STEPS 节流。
_hist_cache: dict[str, tuple[float, dict[str, Any]]] = {}
# 策略/侧车 JSON 通用读缓存（按 mtime）：/api/strategies 每轮询重读全部
# best_*.json + *.live.json，文件不变时直接命中。
_json_read_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def invalidate_checkpoint_cache() -> None:
    _ckpt_cache.clear()
    _hist_cache.clear()
    _json_read_cache.clear()


def _read_json_cached(path: Path) -> dict[str, Any] | None:
    """按 (path, mtime) 缓存 JSON 读取；缺失/损坏/IO 错误 → None（与调用方语义一致）。"""
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _json_read_cache.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if len(_json_read_cache) >= 256:
        _json_read_cache.clear()
    _json_read_cache[key] = (mtime, data)
    return data


def _load_checkpoint_meta(path: Path) -> dict[str, Any]:
    mtime = path.stat().st_mtime
    key = str(path)
    cached = _ckpt_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = {
        "step": int(ckpt.get("step", _step_from_name(path))),
        "best_score": ckpt.get("best_score"),
        "best_formula": ckpt.get("best_formula"),
        "training_history": ckpt.get("training_history") or {},
    }
    _ckpt_cache[key] = (mtime, meta)
    return meta


def _fmt_mtime(path: Path) -> str:
    """文件修改时间的本地可读串（模型库侧车行展示用）。"""
    try:
        import datetime

        return datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


def _decode_formula(tokens: list[int] | None) -> str | None:
    if not tokens:
        return None
    names = FORMULA_VOCAB.token_names
    try:
        return " → ".join(names[t] for t in tokens)
    except (IndexError, TypeError):
        return str(tokens)


def _load_strategy(symbol: str, timeframe: str | None = None) -> dict[str, Any] | None:
    """读取该品种策略制品。timeframe 给定时优先 best_{symbol}_{tf}.json
    （timeframe 限定命名，如 H1 生产冠军），不存在则回退旧式
    best_{symbol}.json（M5 时代制品），保证旧部署仍可显示。
    """
    path: Path | None = None
    if timeframe:
        tf_path = STRATEGIES_DIR / f"best_{symbol}_{timeframe}.json"
        if tf_path.exists():
            path = tf_path
    if path is None:
        path = STRATEGIES_DIR / f"best_{symbol}.json"
    if not path.exists():
        return None
    return _read_json_cached(path)


def _pick_training_history(
    file_history: dict[str, Any] | None,
    ckpt_history: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """取步数更多的那份历史，避免旧 checkpoint 覆盖较新的 json 曲线。"""
    if not file_history and not ckpt_history:
        return None
    if not file_history:
        return ckpt_history
    if not ckpt_history:
        return file_history
    file_n = len(file_history.get("step") or [])
    ckpt_n = len(ckpt_history.get("step") or [])
    return file_history if file_n >= ckpt_n else ckpt_history


def get_symbol_progress(symbol: str, timeframe: str | None = None) -> SymbolProgress:
    train_steps = ModelConfig.TRAIN_STEPS
    strategy = _load_strategy(symbol, timeframe)
    ckpts = checkpoint_glob(symbol)

    current_step = 0
    best_score = None
    best_formula = None
    history: dict[str, Any] | None = None
    ckpt_path: str | None = None
    ckpt_mtime: float | None = None

    hist_file = PROJECT_ROOT / f"training_history_{symbol}.json"
    holdout: dict[str, Any] | None = None
    holdout_bars: int | None = None
    file_history: dict[str, Any] | None = None
    if hist_file.exists():
        try:
            _hk = str(hist_file)
            _hm = hist_file.stat().st_mtime
            _hit = _hist_cache.get(_hk)
            if _hit is not None and _hit[0] == _hm:
                file_history = _hit[1]
            else:
                file_history = json.loads(hist_file.read_text(encoding="utf-8"))
                _hist_cache[_hk] = (_hm, file_history)
            steps = file_history.get("step") or []
            if steps:
                # history 存的是 0 起算的训练步索引，展示与日志 [N/5000] 对齐用 N
                current_step = max(current_step, int(steps[-1]) + 1)
            bests = file_history.get("best_score") or []
            if bests:
                best_score = float(bests[-1])
            history = file_history
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    if ckpts:
        latest = ckpts[-1]
        ckpt_path = str(latest.relative_to(PROJECT_ROOT)).replace("\\", "/")
        ckpt_mtime = latest.stat().st_mtime
        try:
            meta = _load_checkpoint_meta(latest)
            current_step = max(current_step, int(meta["step"]))
            if meta.get("best_score") is not None:
                best_score = float(meta["best_score"])
            best_formula = meta.get("best_formula")
            history = _pick_training_history(file_history, meta.get("training_history"))
        except Exception:
            current_step = max(current_step, _step_from_name(latest))
    if history:
        if holdout is None and history.get("holdout"):
            holdout = history["holdout"]
        if holdout_bars is None and history.get("holdout_bars"):
            holdout_bars = history["holdout_bars"]

    if strategy:
        if best_score is None and strategy.get("best_score") is not None:
            best_score = float(strategy["best_score"])
        if best_formula is None and strategy.get("formula"):
            best_formula = strategy["formula"]
        if holdout is None and strategy.get("holdout"):
            holdout = strategy["holdout"]
        if (holdout_bars is None) and strategy.get("holdout_bars"):
            holdout_bars = strategy["holdout_bars"]

    return SymbolProgress(
        symbol=symbol,
        train_steps=train_steps,
        current_step=current_step,
        best_score=best_score,
        best_formula=best_formula,
        formula_decoded=_decode_formula(best_formula),
        has_strategy=strategy is not None,
        strategy_score=float(strategy["best_score"]) if strategy and strategy.get("best_score") is not None else None,
        checkpoint_path=ckpt_path,
        checkpoint_mtime=ckpt_mtime,
        history=history,
        holdout_bars=holdout_bars,
        holdout=holdout,
    )


def get_strategy_for_export(symbol: str, timeframe: str | None = None) -> dict[str, Any]:
    data = _load_strategy(symbol, timeframe)
    if not data:
        raise FileNotFoundError(f"未找到 {symbol} 的策略，请先完成训练")
    out = dict(data)
    formula = out.get("formula")
    if formula and not out.get("formula_decoded"):
        out["formula_decoded"] = _decode_formula(formula)
    return out


def build_strategy_export_filename(
    symbol: str,
    step: int,
    score: float | None,
) -> str:
    """e.g. strategy_ADAUSD_step0084_score2.4021.json"""
    safe = symbol.replace(".", "_")
    step_part = f"step{max(0, int(step)):04d}"
    if score is not None:
        return f"strategy_{safe}_{step_part}_score{float(score):.4f}.json"
    return f"strategy_{safe}_{step_part}.json"


def list_strategies() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not STRATEGIES_DIR.exists():
        return rows
    for path in sorted(STRATEGIES_DIR.glob("best_*.json")):
        # P0 加固：训练中实时保存的 *.live.json 侧车不是部署冠军，不进列表
        if path.name.endswith(".live.json"):
            continue
        data = _read_json_cached(path)
        if data is None:
            continue
        formula = data.get("formula")
        # data_source 溯源可能缺（老 schema 只有顶层 data_file）→ 归一化出 file
        # 供下拉/卡片展示训练数据文件；老文件只有顶层 data_file 时据此补全
        ds = data.get("data_source")
        if not isinstance(ds, dict):
            ds = None
        if ds is None and data.get("data_file"):
            ds = {"file": Path(str(data["data_file"])).name,
                  "data_file": data["data_file"]}
        # 同品种存在 *.live.json 侧车 → 该模型正被训练覆盖，标记“训练中·未验证”
        live = None
        live_path = STRATEGIES_DIR / f"{path.stem}.live.json"
        if live_path.exists():
            lv = _read_json_cached(live_path)
            if lv is not None:
                live = {
                    "best_score": lv.get("best_score"),
                    "formula": lv.get("formula"),
                    "formula_decoded": _decode_formula(lv.get("formula")),
                    "live_file": live_path.name,
                    "updated_at": _fmt_mtime(live_path),
                }
            else:
                live = None
        rows.append({
            "file": path.name,
            # 完整路径：供前端策略下拉 option 的 value 直接喂
            # strategy-file/browse?path=…（与 inspect_strategy_file 口径一致）
            "strategy_file": str(path.resolve()),
            "symbol": data.get("symbol") or path.stem.replace("best_", "", 1),
            "timeframe": data.get("timeframe"),
            "best_score": data.get("best_score"),
            "formula_decoded": data.get("formula_decoded") or _decode_formula(formula),
            "train_steps": data.get("train_steps"),
            "mode": data.get("mode"),
            "data_source": ds,
            "live": live,
            "train_range": data.get("train_range"),
        })

    # 仅有 *.live.json（尚无已部署冠军）→ 以“训练中·未验证”行展示，
    # 让首次训练中途的 best-so-far 也肉眼可见
    deployed_names = {r["file"].removesuffix(".json") for r in rows}  # best_ZZ
    for live_path in sorted(STRATEGIES_DIR.glob("best_*.live.json")):
        live_base = live_path.name.removesuffix(".live.json")  # best_ZZ.live.json → best_ZZ
        if live_base in deployed_names:
            continue
        stem = live_path.stem
        lv = _read_json_cached(live_path)
        if lv is None:
            continue
        formula = lv.get("formula")
        rows.append({
            "file": live_path.name,
            "strategy_file": str(live_path.resolve()),
            "symbol": lv.get("symbol") or stem.replace("best_", "", 1),
            "timeframe": lv.get("timeframe"),
            "best_score": lv.get("best_score"),
            "formula_decoded": _decode_formula(formula),
            "train_steps": lv.get("train_steps"),
            "mode": lv.get("mode"),
            "data_source": None,
            "live": {
                "best_score": lv.get("best_score"),
                "formula": formula,
                "formula_decoded": _decode_formula(formula),
                "live_file": live_path.name,
                "updated_at": _fmt_mtime(live_path),
            },
            "live_only": True,
            "train_range": None,
        })
    return rows
