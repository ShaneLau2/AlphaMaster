import copy
import datetime
import heapq
import json
import math
import os
import pathlib
import random
import sys
import zlib
from typing import Any

import torch

# ── 并行评估线程配置（必须在第一个张量操作前设置）──────────────────────────
# torch.set_num_interop_threads 一旦有任何张量 op 就锁定，所以紧跟 import torch。
# 策略：保持 intra_threads=full cores，让 PyTorch 内部
# 多线程处理大张量；同时开 8 workers 并行评估不同公式。
# PyTorch 的 intra-op 线程在执行期间会释放 GIL，允许多个 worker 真正并行。
_PHYS_CORES = os.cpu_count() or 4
_EVAL_WORKERS = min(_PHYS_CORES, 8)
_INTRA = _PHYS_CORES  # 保持满线程，不做 phys // workers
try:
    torch.set_num_interop_threads(_EVAL_WORKERS)
except RuntimeError:
    pass  # 已被锁定
torch.set_num_threads(_INTRA)

import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import ContinuousBacktest, estimate_periods_per_year
from .vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError  # task 12.2

# 模块级哨兵：区分「未预计算（走逐条 vm.execute）」vs「预计算=非法(None)」。
# 必须放在模块级而非类属性——默认参数在函数定义时求值，类属性此时不可引用；
# 且方法体内裸名 _EVAL_NO_RES 沿 local→module 查找，不会命中类属性。
_EVAL_NO_RES = object()

# P3：冠军在场时间稳健性校验所需
try:
    from strategy_manager.signal import compute_target_positions_stateless
except ImportError:
    # 兼容无 strategy_manager 的测试环境
    def compute_target_positions_stateless(factors):  # type: ignore
        import torch as _torch
        return _torch.sign(_torch.tanh(factors))

try:
    from config import Config as _RootConfig
    _STRATEGY_FILE  = _RootConfig.STRATEGY_FILE
    _CHECKPOINT_DIR = pathlib.Path(getattr(_RootConfig, 'CHECKPOINT_DIR', 'checkpoints'))
except ImportError:
    _STRATEGY_FILE  = "best_default.json"
    _CHECKPOINT_DIR = pathlib.Path("checkpoints")


def _strategy_file_for_symbol(symbol: str | None) -> str:
    """返回该品种对应的策略文件路径。

    单品种训练时使用 strategies/best_{symbol}.json，
    多品种/未指定品种时回退到默认路径。
    """
    if symbol:
        return str(pathlib.Path("strategies") / f"best_{symbol}.json")
    return _STRATEGY_FILE


def _live_strategy_file_for_symbol(symbol: str | None) -> str:
    """训练中实时保存的“best-so-far”侧车文件路径。

    P0 加固：训练中途的实时保存**绝不写部署路径** strategies/best_{symbol}.json，
    而是写 *.live.json 侧车；只有冠军闸门（holdout + 统计检验）通过后才
    os.replace 到部署路径。这样进程中途被杀/被停也不会用未验证公式覆盖旧冠军。
    """
    if symbol:
        return str(pathlib.Path("strategies") / f"best_{symbol}.live.json")
    return _STRATEGY_FILE.replace(".json", ".live.json")


def _clear_live_file(symbol: str | None) -> None:
    """删除 *.live.json 侧车（训练正式收尾/被抑制/被中止时清理）。"""
    try:
        p = pathlib.Path(_live_strategy_file_for_symbol(symbol))
        if p.exists():
            p.unlink()
    except OSError:
        pass


def _fallback_data_file_for_symbol(symbol: str) -> tuple[str | None, str | None]:
    """Read web_settings.json last_data_file when strategy JSON lacks data_file."""
    settings_path = pathlib.Path("web_settings.json")
    if not settings_path.exists():
        return None, None
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        last = str(settings.get("last_data_file") or "").strip()
    except (json.JSONDecodeError, OSError):
        return None, None
    if not last:
        return None, None
    p = pathlib.Path(last)
    if not p.exists():
        return None, None
    try:
        from data_pipeline.parquet_manager import inspect_parquet_file

        info = inspect_parquet_file(str(p.resolve()))
    except Exception:
        return str(p.resolve()), None
    if info.get("symbol") != symbol:
        return None, None
    return str(p.resolve()), info.get("timeframe")


def _engine_data_source_meta(engine: Any) -> dict[str, Any] | None:
    """由 engine 当前数据上下文构建 best_*.json 的 ``data_source`` 溯源标记。

    优先用已加载的 ParquetDataManager（symbol/timeframe/文件名/首末时间戳/
    数据指纹全量可用，零额外 IO）；管理器缺失时回退到 ``engine.data_file``
    属性（必要时查 web_settings.json），此时仅记录文件名级信息。
    """
    dm = getattr(engine, "data_manager", None)
    if dm is not None and hasattr(dm, "data_source_meta"):
        try:
            meta = dm.data_source_meta()
            if meta:
                return meta
        except Exception:
            pass
    data_file = getattr(engine, "data_file", None)
    if not data_file and getattr(engine, "target_symbol", None):
        data_file, _ = _fallback_data_file_for_symbol(engine.target_symbol)
    if not data_file:
        return None
    meta: dict[str, Any] = {"file": pathlib.Path(str(data_file)).name,
                            "data_file": str(data_file)}
    try:
        from data_pipeline.parquet_manager import inspect_parquet_file

        info = inspect_parquet_file(str(data_file)) or {}
    except Exception:
        info = {}
    if info.get("symbol"):
        meta["symbol"] = info["symbol"]
    if info.get("timeframe"):
        meta["timeframe"] = info["timeframe"]
    if info.get("bars"):
        meta["bars"] = int(info["bars"])
    return meta or None


def _merge_train_metadata(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """把训练元数据并入策略 payload，返回新 dict（不就地修改）。

    统一所有 best_*.json 写入口（训练中途 live 保存 / 最终部署），保证每条
    策略都携带「在哪个 parquet 上训的」（data_source：symbol/timeframe/文件/
    日期跨度）以及 timeframe/data_file/mode/train_steps；engine 未设置的字段
    回退到 payload 已保留的旧值（如旧文件里的 data_file）。
    """
    out = dict(payload)
    for key in ("timeframe", "data_file", "mode", "train_steps"):
        val = getattr(engine, key, None)
        if val is None:
            val = out.get(key)
        if val is not None:
            out[key] = val
    # 溯源保护：payload 里的 train_range（通常来自已读取的旧文件内容）原样保留，
    # 部署路径会用引擎本程的 _train_range_meta() 覆盖为更精确值；live 保存只保旧。
    if not out.get("data_file") and getattr(engine, "target_symbol", None):
        data_file, tf = _fallback_data_file_for_symbol(engine.target_symbol)
        if data_file:
            out["data_file"] = data_file
        if tf and not out.get("timeframe"):
            out["timeframe"] = tf
        if data_file and not out.get("mode"):
            out["mode"] = "parquet_file"
    ds = _engine_data_source_meta(engine)
    if ds:
        if ds.get("timeframe") and not out.get("timeframe"):
            out["timeframe"] = ds["timeframe"]
        out["data_source"] = ds
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 样本外（Holdout）预留
# ─────────────────────────────────────────────────────────────────────────────

def _load_holdout_state() -> dict:
    """读取数据指纹 → holdout 消费状态（损坏/缺失时返回空 dict）。"""
    try:
        p = pathlib.Path(ModelConfig.HOLDOUT_STATE_FILE)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _consume_holdout(fingerprint: str, formula: list[int]) -> None:
    """标记该数据版本已被当前公式消费（同版本后续公式不得再批准）。"""
    try:
        state = _load_holdout_state()
        state[fingerprint] = {
            "approved_formula": list(formula),
            "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        p = pathlib.Path(ModelConfig.HOLDOUT_STATE_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(state, fp, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[holdout] 消费状态写入失败: {exc}")


try:
    import fcntl as _fcntl
except Exception:  # noqa: BLE001 非 POSIX（如 Windows）降级为无锁
    _fcntl = None


def _champion_lock_ctx(path):
    """对 `<path>.lock` 加排他 flock（web 巡检与引擎部署写同一文件时互斥）。

    引擎在训练收尾写 deploy/reject，web 训练管理在运行中写 train_inspect
    事件，两个进程可能同时读改写 champion_history.json；os.replace 保证不会
    写出损坏的半文件，但并发读改写可能丢一条事件——这里用 flock 把
    「读→追加→替换」整段串行化。无 fcntl 的平台降级为无锁。
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        if _fcntl is None:
            yield
            return
        lock_path = pathlib.Path(str(path) + ".lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a+") as lf:
                _fcntl.flock(lf, _fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    _fcntl.flock(lf, _fcntl.LOCK_UN)
        except OSError:
            yield  # 锁文件不可用时退化为无锁，不阻断部署

    return _cm()


def _champion_history() -> list[dict]:
    try:
        p = pathlib.Path(ModelConfig.CHAMPION_HISTORY_FILE)
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _record_champion_event(entry: dict) -> None:
    """追加冠军部署/拒绝记录（append-only，供线上档案与回滚）。"""
    try:
        p = pathlib.Path(ModelConfig.CHAMPION_HISTORY_FILE)
        with _champion_lock_ctx(p):
            p.parent.mkdir(parents=True, exist_ok=True)
            hist = _champion_history()
            entry = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), **entry}
            hist.append(entry)
            tmp = str(p) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(hist, fp, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[champion_history] 写入失败: {exc}")


def _load_last_champion(symbol: str | None) -> dict | None:
    """返回该品种最后一次【部署】的冠军记录（无则 None）。"""
    if not symbol:
        return None
    hist = _champion_history()
    for entry in reversed(hist):
        if entry.get("symbol") == symbol and entry.get("event") == "deploy":
            return entry
    return None


def _holdout_bars_for(T_full: int) -> int:
    """训练尾部预留的样本外根数。

    规则：默认取 ModelConfig.HOLDOUT_BARS；小数据集按比例压缩
    （cap = T//8，至少留 300 根给训练窗口），保证确定性（同一份数据
    每次切分一致，续训/重训不会改变预留窗口）。
    """
    cap = T_full // 8
    if cap < 100:
        cap = max(0, T_full - 300)
    return max(0, min(ModelConfig.HOLDOUT_BARS, cap))


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward 折叠构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_walk_forward_folds(T: int, n_folds: int = 5, gap: int = 20) -> list[dict]:
    """构建 Walk-Forward 折叠。

    改为 rolling window（train_start = (k-1)*fold_size）以避免 expanding window
    导致早期折的 val 数据被后续折的 train 切片包含，造成 val_score 不再严格 OOS。

    同时修正最后一折 val_end = min(val_start + fold_size, T)，避免最后一折 val
    大小不均导致均值被该折主导。
    """
    fold_size = T // n_folds
    if fold_size < 2:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    total_required = fold_size * n_folds + gap * (n_folds - 1)
    if total_required > T:
        gap = max(0, (T - fold_size * n_folds) // n_folds)
    folds = []
    for k in range(1, n_folds):
        # rolling window：每折 train 起点前移，避免包含早期折的 val 切片
        train_start = (k - 1) * fold_size
        train_end   = k * fold_size
        val_start   = train_end + gap
        # 最后一折 val_end 用 min 避免超出 T，且与其他折大小一致
        val_end     = min(val_start + fold_size, T)
        if val_start >= T or val_end <= val_start:
            break
        folds.append({"train_start": train_start, "train_end": train_end,
                      "val_start": val_start, "val_end": val_end, "gap": gap})
    if not folds:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    return folds


def neutral_band_bonus(factors: torch.Tensor, w: float, half: float = 0.5) -> torch.Tensor:
    """中性带正则（观望区偏好）：|factor| < half 的时间占比 × 权重。

    纯函数（线程安全）：奖励 = w × mean(|f| < half)。只用于训练选优，
    不掺入 val_score——冠军闸门仍按纯绩效口径选冠军。
    """
    if w <= 0:
        return torch.zeros((), dtype=factors.dtype)
    return w * (factors.abs() < half).float().mean()


def _repetition_penalty(formula: list[int]) -> float:
    if not formula:
        return 0.0
    penalty, count = 0.0, 1
    for i in range(1, len(formula)):
        if formula[i] == formula[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# ConstrainedSampler — 保证 100% 合法公式
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedSampler:
    def __init__(self, vocab_size: int, feat_offset: int, arity_map: dict[int, int],
                 positive_only_ids: set[int] | None = None):
        self.vocab_size  = vocab_size
        self.feat_offset = feat_offset
        self.arity_map   = arity_map
        self.delta: dict[int, int] = {}
        for tid in range(vocab_size):
            if tid < feat_offset:
                self.delta[tid] = 1
            else:
                a = arity_map.get(tid, 1)
                self.delta[tid] = 1 - a
        # 恒正算子 token id 集合（用于算子链约束）
        self.positive_only_ids = positive_only_ids or set()
        # 构建感染传播/恢复算子 id 集合
        from .vm import INFECTED_PROPAGATING_OPS, SIGN_RESTORE_OPS
        from .ops import OPS_CONFIG as _ops
        self.infected_propagating_ids = set()
        self.sign_restore_ids = set()
        for i, cfg in enumerate(_ops):
            tid = i + feat_offset
            if cfg[0] in INFECTED_PROPAGATING_OPS:
                self.infected_propagating_ids.add(tid)
            if cfg[0] in SIGN_RESTORE_OPS:
                self.sign_restore_ids.add(tid)

    def valid_mask(self, stack_depth: int, step_idx: int,
                   total_steps: int, device: torch.device,
                   prev_token: int | None = None,
                   infected_chain_len: int = 0) -> torch.Tensor:
        remaining = total_steps - step_idx
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        for tid in range(self.vocab_size):
            d         = self.delta[tid]
            new_depth = stack_depth + d
            if new_depth < 1:
                mask[tid] = False;  continue
            min_future = new_depth + (remaining - 1) * (-2)
            max_future = new_depth + (remaining - 1) * 1
            if 1 < min_future or 1 > max_future:
                mask[tid] = False
            # ── 算子链约束（感染模型）──────────────────────────────
            # 如果已感染且感染链 >= 2，禁止再使用传播算子
            # （允许恢复算子和非传播算子如 ADD/SUB/MUL）
            if infected_chain_len >= 2 and tid in self.infected_propagating_ids:
                mask[tid] = False
            # 如果已感染且感染链 >= 3，禁止所有算子（强制恢复或结束）
            # 实际上不禁止恢复算子，只禁止传播和恒正算子
            if infected_chain_len >= 3:
                if tid in self.infected_propagating_ids or tid in self.positive_only_ids:
                    mask[tid] = False
        if not mask.any():
            for tid in range(self.vocab_size):
                if stack_depth + self.delta[tid] >= 1:
                    mask[tid] = True
        return mask

    def apply_mask_to_logits(self, logits: torch.Tensor, stack_depths: list[int],
                              step_idx: int, total_steps: int,
                              prev_tokens: list[int | None] | None = None,
                              infected_chain_lens: list[int] | None = None) -> torch.Tensor:
        masked = logits.clone()
        device = logits.device
        for b, depth in enumerate(stack_depths):
            prev_t = prev_tokens[b] if prev_tokens else None
            icl = infected_chain_lens[b] if infected_chain_lens else 0
            vmask = self.valid_mask(depth, step_idx, total_steps, device,
                                    prev_token=prev_t, infected_chain_len=icl)
            masked[b][~vmask] = -1e9
        return masked

    def update_infection(self, token: int, infected_chain_len: int) -> int:
        """更新感染链长度，返回新的感染链长度。"""
        if token in self.positive_only_ids:
            return infected_chain_len + 1
        elif token in self.sign_restore_ids:
            return 0
        elif token in self.infected_propagating_ids:
            if infected_chain_len > 0:
                return infected_chain_len + 1
            return 0
        return infected_chain_len  # 非传播/非恢复算子，不改变状态


# ─────────────────────────────────────────────────────────────────────────────
# AlphaEngine — __init__ 与静态辅助方法
# ─────────────────────────────────────────────────────────────────────────────

class AlphaEngine:
    def __init__(self, data_manager=None, use_lord_regularization=True,
                 lord_decay_rate=1e-3, lord_num_iterations=5, n_folds: int = 5,
                 target_symbol: str | None = None, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            try:
                import numpy as _np
                _np.random.seed(seed)
            except Exception:
                pass
            self.seed = int(seed)
        else:
            self.seed = None
        self.data_manager  = data_manager
        self.n_folds       = n_folds
        self.target_symbol = target_symbol   # None = 多品种模式，str = 单品种模式
        self.model   = AlphaGPT().to(ModelConfig.DEVICE)
        # 实验 ACC_CRITIC_GAE：在构造 optimizer 前挂 critic 价值头（默认关闭）
        if getattr(ModelConfig, "ACC_CRITIC_GAE", False):
            self.model.enable_critic()
        self.opt     = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["attention", "qk_norm"],
            )
            self.rank_monitor = StableRankMonitor(
                self.model, target_keywords=["in_proj", "out_proj", "qk_norm"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = ContinuousBacktest()

        from .vocab import FORMULA_VOCAB as _v
        self.sampler = ConstrainedSampler(
            vocab_size=_v.size, feat_offset=_v.operator_offset,
            arity_map=self.vm.arity_map,
            positive_only_ids=self.vm.positive_only_ids
        )

        self.best_score   = -float('inf')
        self.best_formula = None
        self._best_snapshot: dict | None = None

        # ── 冠军闸门（P0/P1）：val 只入围，holdout 真实回测选冠军 ────
        self._finalists: dict[tuple, dict] = {}   # formula-tuple -> {val, step, fml}
        self._champion_at_start: dict | None = None  # 训练开始时已部署的冠军内容
        self.champion_outcome: dict | None = None    # 最终部署/拒绝结果（train_file 读取）
        self._robustness_result: dict | None = None  # P3 稳健性复核结果（仅拒后触发）
        self.data_fingerprint: str | None = None     # 数据版本指纹（同版本 holdout 只消费一次）

        self.training_history = {
            'step': [], 'avg_reward': [], 'best_score': [], 'val_score': [], 'stable_rank': []
        }
        self._restart_count      = 0
        self.factor_pool: list[tuple[float, int, torch.Tensor]] = []
        self._factor_pool_counter = 0

        # Elite Replay pool: (val_score, counter, formula_tokens, birth_step)
        self._elite_pool: list[tuple[float, int, list[int], int]] = []
        self._elite_counter = 0

        # 自适应噪声：记录 best 刷新步数
        self._best_update_step = 0
        self._stagnation_steps = 0

        # Fix 3: EMA reward baseline
        self._reward_ema: float | None = None
        self._reward_ema_step: int = 0
        # 重启升级：记录上一次重启所在步（-1=尚未重启），用于判断“上次重启后
        # best 是否刷新过”（无刷新 → 下次重启强制 full reset 逃离吸引子）。
        self._last_restart_step: int = -1

        # 选优层 vol 覆盖（regime_ada 诊断落地）：训练窗口因果 vol 三段索引缓存，
        # 每次 train() 重建；None = 闸门失效（fail-open，不影响 reward）。
        self._vol_tier_idx: list | None = None
        self._vol_tier_meta: dict | None = None
        # 格级升级（VOL_COVERAGE_GRID=True 时使用）：vol×er 3×3 格索引缓存
        self._vol_grid_cells: list | None = None
        self._vol_grid_meta: dict | None = None

        # ── 并行评估线程池（CPU 利用率优化）──────────────────────────────
        self._eval_pool = None
        self._eval_workers = 1

        # ── 样本外（Holdout）预留与验证 ────────────────────────────────
        self.holdout_bars = 0          # 训练尾部预留的根数（训练/选优从不触碰）
        self.holdout_override: int | None = None  # CLI/实验可覆盖 ModelConfig.HOLDOUT_BARS
        self.holdout: dict[str, Any] | None = None
        self._holdout_verified = False
        # 部署前跨 ticker OOS 复核（P0.2）结果缓存（_finalize_champion 填充）
        self._oos_review: dict | None = None
        # 实验 run 隔离：非空时历史/检查点文件名带后缀、跳过 *.live.json 侧车
        # （仅供 scripts/ 下的对比实验使用；默认空字符串 = 与历史行为完全一致）
        self.run_tag = ""

    def _resolve_holdout_bars(self, T_full: int) -> int:
        """解析本次训练的样本外预留根数：engine.holdout_override > ModelConfig 默认。

        与模块级 _holdout_bars_for 同规则（小数据集按比例压缩），只是取值允许覆盖
        （--holdout-bars N / 实验引擎注入），大 holdout 下闸门分数更可信。
        """
        want = self.holdout_override
        if want is None:
            want = int(ModelConfig.HOLDOUT_BARS)
        cap = T_full // 8
        if cap < 100:
            cap = max(0, T_full - 300)
        return max(0, min(int(want), cap))

    def _run_tag_suffix(self) -> str:
        tag = getattr(self, "run_tag", "") or ""
        return f"_{tag}" if tag else ""
        if ModelConfig.PARALLEL_EVAL:
            self._init_parallel_eval()

    # ── 并行评估初始化 ──────────────────────────────────────────────────────

    def _init_parallel_eval(self):
        """初始化 ThreadPoolExecutor 用于并行公式评估。

        PyTorch CPU 算子会释放 GIL，多个 worker 线程可以真正并行执行
        vm.execute / bt.evaluate_fold 等纯张量计算。

        线程配置：
        - intra_threads = physical_cores（保持满，让大张量操作快）
        - inter_threads = workers（允许多 worker 并行调度）
        PyTorch 的 intra-op 线程池会自适应负载，不会真的 8×8=64 全跑满。
        """
        from concurrent.futures import ThreadPoolExecutor

        phys = _PHYS_CORES
        workers = ModelConfig.EVAL_WORKERS
        if workers <= 0:
            workers = min(phys, 8)
        intra = ModelConfig.EVAL_INTRA_THREADS
        if intra <= 0:
            intra = phys  # 保持满线程
        # 线程已在模块 import 时设置，这里只确认
        try:
            torch.set_num_threads(intra)
        except RuntimeError:
            pass

        self._eval_workers = workers
        self._eval_pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="alpha-eval",
        )
        print(f"[ParallelEval] workers={workers} intra_threads={intra} "
              f"(physical_cores={phys})  pool={'ON' if workers > 1 else 'OFF'}",
              flush=True)

    # ── 单条公式评估任务（线程安全）───────────────────────────────────────

    def _eval_formula_task(
        self,
        idx: int,
        fml: list[int],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
        precomputed_res=_EVAL_NO_RES,   # 模块级哨兵（未预计算）
    ) -> dict:
        """评估单条公式（线程安全，可被多个 worker 并发调用）。

        所有共享对象（self.vm, self.bt）在评估路径上都是只读的；
        factor_pool 通过 snapshot 传入只读快照。
        precomputed_res：批量路径（BATCHED_EVAL_ENABLED）传入已由
        execute_batched 算好的 [1,T] 结果（None=非法公式），跳过
        vm.execute；默认哨兵 = 未预计算，按逐条路径执行。
        """
        try:
            if precomputed_res is not _EVAL_NO_RES:
                res = precomputed_res
            else:
                with torch.no_grad():
                    res = self.vm.execute(fml, feat)

            if res is None:
                return {'idx': idx, 'status': 'none', 'reward': -5.0,
                        'val_score': -5.0, 'fml': fml}
            if res.std() < 1e-4:
                return {'idx': idx, 'status': 'const', 'reward': -2.0,
                        'val_score': -2.0, 'fml': fml}

            with torch.no_grad():
                if use_wf:
                    fold_tr, fold_vl, fold_ic = [], [], []
                    for fold in folds:
                        tr_sc, vl_sc = self.bt.evaluate_fold(
                            res, t_ret,
                            fold["train_start"], fold["train_end"],
                            fold["val_start"],   fold["val_end"],
                        )
                        ic_m, _ = AlphaEngine._compute_ic(
                            res[:, fold["train_start"]:fold["train_end"]],
                            t_ret[:, fold["train_start"]:fold["train_end"]],
                        )
                        tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
                        fold_tr.append(ModelConfig.REWARD_ALPHA * tr_adj)
                        ic_v, _ = AlphaEngine._compute_ic(
                            res[:, fold["val_start"]:fold["val_end"]],
                            t_ret[:, fold["val_start"]:fold["val_end"]],
                        )
                        vl_adj = AlphaEngine._apply_ic_gate(vl_sc, ic_v)
                        fold_vl.append(vl_adj)
                        fold_ic.append(ic_m.item())
                    train_score = torch.stack(fold_tr).mean()
                    val_score = torch.stack(fold_vl).mean()
                    ic_i = sum(fold_ic) / len(fold_ic)
                else:
                    T_total = res.shape[1]
                    split_pt = max(int(T_total * 0.8), T_total - 100)
                    train_score, _ = self.bt.evaluate(res, {}, t_ret)
                    ic_m0, _ = AlphaEngine._compute_ic(res, t_ret)
                    train_score = AlphaEngine._apply_ic_gate(
                        ModelConfig.REWARD_ALPHA * train_score, ic_m0
                    )
                    if split_pt < T_total - 1:
                        vl_sc, _ = self.bt.evaluate_fold(
                            res, t_ret, 0, split_pt, split_pt, T_total,
                        )
                        ic_v0, _ = AlphaEngine._compute_ic(
                            res[:, split_pt:], t_ret[:, split_pt:],
                        )
                        val_score = AlphaEngine._apply_ic_gate(vl_sc, ic_v0)
                    else:
                        val_score = train_score
                    ic_i = ic_m0.item()
                ic_full, ic_stab_full = AlphaEngine._compute_ic(res, t_ret)

            # 惩罚（纯函数，线程安全）
            reward = train_score
            val_score_out = val_score

            # 重复惩罚
            rp = _repetition_penalty(fml)
            if rp > 0:
                reward = reward - rp
                val_score_out = val_score_out - rp

            # 相关性惩罚（用 step 起始快照）
            if use_wf:
                _corr_slice = (folds[0]["train_start"], folds[0]["train_end"])
            else:
                _corr_slice = (0, max(int(res.shape[1] * 0.8), res.shape[1] - 100))
            reward = self._apply_corr_penalty(reward, res, _corr_slice)
            val_score_out = self._apply_corr_penalty(val_score_out, res, _corr_slice)

            # 中性带正则（观望区偏好，默认关闭）：|factor| < NEUTRAL_BAND_HALF
            # 的时间占比越高，额外奖励越大。只影响训练选优（reward），
            # 不掺入 val_score_out——冠军闸门仍按纯绩效口径选冠军。
            nb_w = float(getattr(ModelConfig, "NEUTRAL_BAND_W", 0.0))
            if nb_w > 0:
                _lo, _hi = _corr_slice
                reward = reward + neutral_band_bonus(
                    res[:, _lo:_hi], nb_w,
                    float(getattr(ModelConfig, "NEUTRAL_BAND_HALF", 0.5)),
                )

            # 选优层 vol 覆盖（regime 诊断落地）：fail-open，不影响 reward，
            # 只给 Part C 的 best/finalist 更新提供闸门位。
            # 格级（VOL_COVERAGE_GRID=True 默认）时缓存为 _vol_grid_cells；段级为
            # _vol_tier_idx。两者任一存在才判——否则 grid 口径在选优层静默失效
            # （2026-09-05 修正：grid 默认只作用于终局，选优层只认段级缓存）。
            vol_ok = True
            vol_info: dict | None = None
            if ModelConfig.VOL_COVERAGE_ENABLED and (
                    getattr(self, "_vol_tier_idx", None)
                    or getattr(self, "_vol_grid_cells", None)):
                try:
                    vol_ok, vol_info = self._vol_pnl_coverage(res, t_ret)
                except Exception:  # noqa: BLE001 覆盖检查失败 → 放行
                    vol_ok, vol_info = True, None

            return {
                'idx': idx, 'status': 'ok',
                'reward': reward.item() if isinstance(reward, torch.Tensor) else float(reward),
                'val_score': val_score_out.item() if isinstance(val_score_out, torch.Tensor) else float(val_score_out),
                'ic_full': ic_full.item(), 'ic_stab': ic_stab_full.item(),
                'ic_i': ic_i, 'res': res, 'fml': fml,
                'vol_ok': bool(vol_ok), 'vol_info': vol_info,
            }
        except Exception as e:
            return {'idx': idx, 'status': 'error', 'reward': -5.0,
                    'val_score': -5.0, 'fml': fml,
                    'error': f'{type(e).__name__}: {e}'}

    # ── IC computation ────────────────────────────────────────────────────────

    # ── 样本外（Holdout）验证与冠军闸门（P0/P1）──────────────────────────

    def _verify_holdout(self) -> dict[str, Any] | None:
        """训练结束后，用从未参与训练/选优的尾部窗口验证最优公式。

        与训练共用同一套评估路径（vm.execute + bt.evaluate_fold + IC 门控），
        结果写入 training_history（随历史 JSON / 检查点持久化），并落盘。

        P0 加固：结果带 passed/reasons（分数 > 0、保持率 ≥ 下限、Sharpe ≥ 下限、
        同数据版本单次消费）；P2 加固：评分窗口排除最后两根边界 bar（target=0）。

        返回值：holdout 结果 dict；无最优公式 / 无预留窗口时返回 None。
        幂等：同一引擎实例只验证一次。
        """
        if self._holdout_verified:
            return self.holdout or None
        self._holdout_verified = True

        h = self.holdout_bars
        if not self.best_formula or h <= 0 or self.data_manager is None:
            return None
        feat = self.data_manager.feat_tensor.to(ModelConfig.DEVICE)
        t_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)
        # 注意：特征张量是 [N, C, T]（通道维 C 在 shape[1]），时间维用 target_ret 的 [N, T]
        n_time = t_ret.shape[1]
        if n_time <= h + 2:
            return None
        start = n_time - h
        mature_end = n_time - 2   # 标签成熟度：最后两根 target 恒为 0（边界），不参与评分

        try:
            with torch.no_grad():
                res = self.vm.execute(self.best_formula, feat)
            if res is None or res.std() < 1e-4:
                return None
            with torch.no_grad():
                # 与训练同款评分：多目标 + OOS Sortino 门控，限定在 holdout 窗口
                _, ho_score = self.bt.evaluate_fold(
                    res, t_ret, start, mature_end, start, mature_end,
                )
                ic_ho, _ = AlphaEngine._compute_ic(
                    res[:, start:mature_end], t_ret[:, start:mature_end],
                )
                ho_adj = AlphaEngine._apply_ic_gate(ho_score, ic_ho)

                pos = compute_target_positions_stateless(res)
                prev = torch.roll(pos, 1, dims=1)
                prev[:, 0] = 0.0
                turnover = torch.abs(pos - prev)
                pnl = pos * t_ret - turnover * self.bt.cost_rate
                pnl_h = pnl[:, start:mature_end]
                total_return_pct = float(pnl_h.sum()) * 100.0
                sharpe = float(
                    pnl_h.mean()
                    / (pnl_h.std() + 1e-9)
                    * math.sqrt(max(1.0, float(self.bt.periods_per_year)))
                )
                sortino = float(self.bt._sortino(pnl_h))
        except Exception as exc:  # noqa: BLE001 验证失败不阻断训练收尾
            tqdm.write(f"[holdout] 验证失败: {exc}")
            return None

        best = self.best_score if self.best_score not in (None, -float('inf')) else None
        ratio = None
        if best is not None and best > 0:
            ratio = float(ho_adj) / float(best)

        passed, reasons = self._holdout_gate(ho_adj, ratio, sharpe)

        self.holdout = {
            "bars": h,
            "start": start,
            "end": mature_end,
            "mature_bars": mature_end - start,
            "val_score": float(ho_adj),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "total_return_pct": round(total_return_pct, 3),
            "score_ratio": round(ratio, 4) if ratio is not None else None,
            "passed": bool(passed),
            "gate": reasons,
        }
        # 实验/对比 run（run_tag 非空）额外存逐根 holdout 净值曲线，
        # 供 results/compare_*.json 携带 → 回测页「tail vs spread」摘要叠加两路曲线。
        # 正式训练 run_tag 为空：不写，避免 training_history 无限膨胀。
        if getattr(self, "run_tag", "") and pnl_h.numel() > 1:
            try:
                import numpy as _np

                add = pnl_h.cumsum(dim=1).squeeze(0).cpu().numpy()  # 累计收益(小数)
                labels = _np.arange(start, mature_end, dtype=int)
                # 降采样到 ≤ MAX_PTS 点（含首尾），保持对比曲线轻量
                max_pts = 200
                n = len(add)
                if n > max_pts:
                    idx = _np.linspace(0, n - 1, max_pts).round().astype(int)
                    idx[-1] = n - 1
                    idx = _np.unique(idx)
                    labels = labels[idx]
                    add = add[idx]
                self.holdout["equity_curve"] = {
                    "labels": [int(x) for x in labels],
                    # 以 bar0=0 为基准的累计收益曲线（乘 100 转 %），末点≈total_return_pct
                    "equity": [round(float(v) * 100.0, 4) for v in add],
                }
            except Exception:  # noqa: BLE001 曲线失败不影响 holdout 主结果
                pass
        self.training_history["holdout_bars"] = h
        self.training_history["holdout"] = self.holdout
        # 终局快照：holdout 判定必须落盘（不被节流跳过）
        self._save_training_history_live(force=True)
        ratio_str = f"{ratio:.2f}" if ratio is not None else "—"
        verdict = "✓ 通过" if passed else "✗ 未过"
        tqdm.write(
            f"[holdout] 样本外验证（尾部 {h} 根，未参与训练/选优）: "
            f"分={ho_adj:.4f} 收益={total_return_pct:+.2f}% "
            f"Sharpe={sharpe:.2f} 保持率={ratio_str} → {verdict}"
        )
        for r in reasons:
            tqdm.write(f"    [闸门] {r}")
        return self.holdout

    def _holdout_gate(
        self, ho_adj: float, ratio: float | None, sharpe: float,
    ) -> tuple[bool, list[str]]:
        """P0.1 闸门判定：分>0、保持率≥下限、Sharpe≥下限、同版本单次消费。"""
        reasons: list[str] = []
        if not ModelConfig.HOLDOUT_GATE_ENABLED:
            return True, ["holdout 闸门已关闭（配置）"]
        if ho_adj <= ModelConfig.HOLDOUT_MIN_SCORE:
            reasons.append(
                f"holdout 分 {ho_adj:.4f} ≤ 下限 {ModelConfig.HOLDOUT_MIN_SCORE}"
            )
        if ratio is not None and ratio < ModelConfig.HOLDOUT_MIN_RATIO:
            reasons.append(
                f"保持率 {ratio:.3f} < 下限 {ModelConfig.HOLDOUT_MIN_RATIO}"
            )
        if sharpe < ModelConfig.HOLDOUT_MIN_SHARPE:
            reasons.append(
                f"holdout Sharpe {sharpe:.3f} < 下限 {ModelConfig.HOLDOUT_MIN_SHARPE}"
            )
        if ModelConfig.HOLDOUT_SINGLE_USE and self.data_fingerprint:
            rec = _load_holdout_state().get(self.data_fingerprint) or {}
            approved = rec.get("approved_formula")
            if approved is not None and approved != self.best_formula:
                reasons.append(
                    "同数据版本 holdout 已被旧冠军消费（数据指纹未变，不得重复批准）"
                )
        return len(reasons) == 0, reasons

    # ── 入围表（P0）：val 只用于提名 top-K，真实回测选冠军 ────────────

    def _update_finalists(self, val: float, fml: list[int], step: int) -> None:
        if val <= -1.0 or not fml:
            return
        key = tuple(fml)
        k = ModelConfig.FINALIST_TOP_K
        if key in self._finalists:
            if val > self._finalists[key]["val"]:
                self._finalists[key] = {"val": float(val), "step": step, "fml": list(fml)}
            return
        if len(self._finalists) < k:
            self._finalists[key] = {"val": float(val), "step": step, "fml": list(fml)}
            return
        min_key = min(self._finalists, key=lambda kk: self._finalists[kk]["val"])
        if val > self._finalists[min_key]["val"]:
            del self._finalists[min_key]
            self._finalists[key] = {"val": float(val), "step": step, "fml": list(fml)}

    def _top_finalists(self, k: int | None = None) -> list[dict]:
        k = k or ModelConfig.FINALIST_TOP_K
        pool = sorted(self._finalists.values(), key=lambda d: d["val"], reverse=True)[:k]
        if self.best_formula is not None and all(
            tuple(d["fml"]) != tuple(self.best_formula) for d in pool
        ):
            pool.append({"val": float(self.best_score), "step": None, "fml": list(self.best_formula)})
            pool.sort(key=lambda d: d["val"], reverse=True)
        return pool

    def _rigorous_holdout_pnl(
        self, fml: list[int], feat: torch.Tensor, t_ret: torch.Tensor,
        start: int, end: int,
    ) -> dict | None:
        """在 holdout 窗口上按【生产口径】回测单条公式。

        与 run_backtest.py 对齐：position = tanh(factor)，
        pnl = pos × target_ret − |Δpos| × FINALIST_COST_RATE。
        同时返回训练同口径分（供保持率计算）。
        """
        try:
            with torch.no_grad():
                res = self.vm.execute(fml, feat)
            if res is None or res.std() < 1e-4:
                return None
            with torch.no_grad():
                _, ho_sc = self.bt.evaluate_fold(res, t_ret, start, end, start, end)
                ic_ho, _ = AlphaEngine._compute_ic(
                    res[:, start:end], t_ret[:, start:end],
                )
                ho_adj = AlphaEngine._apply_ic_gate(ho_sc, ic_ho)

                pos = compute_target_positions_stateless(res)
                prev = torch.roll(pos, 1, dims=1)
                prev[:, 0] = 0.0
                turnover = torch.abs(pos - prev)
                cost = float(ModelConfig.FINALIST_COST_RATE)
                pnl = pos * t_ret - turnover * cost
                w = pnl[:, start:end]
                ppy = max(1.0, float(self.bt.periods_per_year))
                return {
                    "fml": list(fml),
                    "ho_adj": float(ho_adj),
                    "sharpe": float(w.mean() / (w.std() + 1e-9) * math.sqrt(ppy)),
                    "sortino": float(self.bt._sortino(w)),
                    "ann_ret": float(w.mean() * ppy) * 100.0,
                    "total_return_pct": float(w.sum()) * 100.0,
                    "turnover": float(turnover[:, start:end].mean()),
                    "cost_rate": cost,
                }
        except Exception as exc:  # noqa: BLE001 单条失败不影响其余入围
            tqdm.write(f"[finalist] 回测失败: {type(exc).__name__}: {exc}")
            return None

    def _null_model_check(self, champ_val: float) -> tuple[bool, float | None]:
        """P1 空模型检验：冠军 val 必须超过 K 个随机公式的 99 分位。

        随机公式来自同一采样器（均匀合法采样），在同一批 walk-forward 折上
        评估——即 White reality-check 的廉价版。
        """
        if not ModelConfig.ENABLE_NULL_MODEL_CHECK:
            return True, None
        feat = getattr(self, "_train_feat", None)
        t_ret = getattr(self, "_train_t_ret", None)
        folds = getattr(self, "_train_folds", None)
        if feat is None or t_ret is None or not folds:
            return True, None
        n = max(4, int(ModelConfig.NULL_MODEL_N))
        seed = 0x5EED
        if self.data_fingerprint:
            seed ^= zlib.crc32(self.data_fingerprint.encode()) & 0xFFFFFFFF
        rng = torch.Generator(device="cpu").manual_seed(seed)
        null_vals: list[float] = []
        for _ in range(n):
            fml = self._sample_random_formula(rng)
            if fml is None:
                continue
            r = self._eval_formula_task(
                0, fml, feat, t_ret, folds, True, [],
            )
            v = float(r.get("val_score", -5.0))
            if v > -1.0:
                null_vals.append(v)
        if len(null_vals) < 4:
            return True, None
        null_vals.sort()
        q99 = null_vals[min(len(null_vals) - 1, int(len(null_vals) * 0.99))]
        return bool(champ_val > q99), float(q99)

    def _sample_random_formula(self, rng: torch.Generator) -> list[int] | None:
        """均匀合法采样一条公式（空模型基准用）。"""
        length = int(ModelConfig.MAX_FORMULA_LEN)
        depth = 0
        prev: int | None = None
        fml: list[int] = []
        for step in range(length):
            mask = self.sampler.valid_mask(depth, step, length, torch.device("cpu"),
                                           prev_token=prev)
            ids = mask.nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                return None
            idx = int(torch.randint(0, ids.numel(), (1,), generator=rng).item())
            tok = int(ids[idx].item())
            depth += self.sampler.delta[tok]
            prev = tok
            fml.append(tok)
        return fml if depth == 1 else None

    def _fold_se_check(
        self, champ: list[int] | None, runner: list[int] | None,
    ) -> tuple[bool, dict | None]:
        """P1 跨折 SE：冠军与次优在每折独立算 val，要求差 > k×SE(折间标准误)。"""
        if not ModelConfig.ENABLE_FOLD_SE_CHECK or champ is None or runner is None:
            return True, None
        feat = getattr(self, "_train_feat", None)
        t_ret = getattr(self, "_train_t_ret", None)
        folds = getattr(self, "_train_folds", None)
        if feat is None or t_ret is None or not folds or len(folds) < 3:
            return True, None
        vals: dict[tuple, list[float]] = {tuple(champ): [], tuple(runner): []}
        try:
            for fold in folds:
                for fml in (champ, runner):
                    with torch.no_grad():
                        res = self.vm.execute(fml, feat)
                    if res is None:
                        vals[tuple(fml)].append(-5.0)
                        continue
                    with torch.no_grad():
                        _, vl = self.bt.evaluate_fold(
                            res, t_ret,
                            fold["train_start"], fold["train_end"],
                            fold["val_start"], fold["val_end"],
                        )
                        ic_v, _ = AlphaEngine._compute_ic(
                            res[:, fold["val_start"]:fold["val_end"]],
                            t_ret[:, fold["val_start"]:fold["val_end"]],
                        )
                        vals[tuple(fml)].append(
                            float(AlphaEngine._apply_ic_gate(vl, ic_v))
                        )
        except Exception as exc:  # noqa: BLE001 统计检查失败不阻断部署主路径
            tqdm.write(f"[fold-SE] 检查失败: {type(exc).__name__}: {exc}")
            return True, None
        c = torch.tensor(vals[tuple(champ)], dtype=torch.float32)
        r = torch.tensor(vals[tuple(runner)], dtype=torch.float32)
        diff = c - r
        se = diff.std(unbiased=True) / math.sqrt(max(1, diff.numel()))
        k = float(ModelConfig.CHAMPION_SE_K)
        return bool(float(diff.mean()) > k * float(se)), {
            "champ_per_fold": [round(float(x), 4) for x in c.tolist()],
            "runner_per_fold": [round(float(x), 4) for x in r.tolist()],
            "mean_diff": round(float(diff.mean()), 4),
            "se": round(float(se), 4),
            "k": k,
        }

    def _wf_val_mean(self, res: torch.Tensor, t_ret: torch.Tensor,
                     folds: list[dict]) -> float | None:
        """训练口径 wf 均值 val：每折 evaluate_fold + IC 门控后求均值（None=非法）。

        与 `_eval_formula_task` / `_fold_se_check` 的验证段评分完全一致，
        供稳健性复核复用；任一折失败返回 None（不产出半截均值）。
        """
        vals: list[float] = []
        try:
            for fold in folds:
                with torch.no_grad():
                    _, vl = self.bt.evaluate_fold(
                        res, t_ret,
                        fold["train_start"], fold["train_end"],
                        fold["val_start"],   fold["val_end"],
                    )
                    ic_v, _ = AlphaEngine._compute_ic(
                        res[:, fold["val_start"]:fold["val_end"]],
                        t_ret[:, fold["val_start"]:fold["val_end"]],
                    )
                    vals.append(float(AlphaEngine._apply_ic_gate(vl, ic_v)))
        except Exception:  # noqa: BLE001 单折失败 → 整体判 None
            return None
        return (sum(vals) / len(vals)) if vals else None

    def _run_finalist_robustness(
        self, metrics: list[dict], feat: torch.Tensor, t_ret: torch.Tensor,
        ho_start: int, ho_end: int,
    ) -> dict[str, Any] | None:
        """冠军未过 holdout 闸门时自动运行的稳健性复核（P3）。

        对 top-K finalists 跑三轴敏感性（model_core/robustness.py 的纯计算 + 本
        方法的数据切分/评估编排）：
          - fold : n_folds±1 布局重算训练 wf val（冠军正收益 + 排名保持）
          - start: 丢弃训练区开头 5/10/20% 重建窗口重算 wf val（优势是否依赖早期段）
          - cost : holdout 窗口生产口径在 0/0.5/1/2/4×base 成本下的 Sharpe/年化/排名
        返回 JSON 友好 dict；条件不足/异常返回 None（不阻断训练收尾）。
        调用前提：metrics 已按 holdout 指标降序（metrics[0] 为冠军）。
        """
        try:
            from . import robustness as _rob
            if (not metrics or len(metrics) < 2
                    or feat is None or t_ret is None):
                return None
            top = metrics[: int(ModelConfig.ROBUSTNESS_TOP_K)]
            fmls = [m["fml"] for m in top]
            n_fml = len(fmls)
            n_time = int(t_ret.shape[1])
            h = int(self.holdout_bars)
            T = n_time - h
            if T < _rob.MIN_REMAIN_BARS or h <= 0:
                return None
            base_gap = int(getattr(ModelConfig, "WF_GAP", 20))
            base_cost = float(ModelConfig.FINALIST_COST_RATE)
            ppy = max(1.0, float(self.bt.periods_per_year))

            # 每条公式只执行一次 VM：fold/start 变体共享 res（切列复用），
            # cost 轴在 holdout 段用同一 res 做纯 pnl 重算。
            with torch.no_grad():
                res_map = {tuple(f): self.vm.execute(f, feat) for f in fmls}

            def _ok(res) -> bool:
                return res is not None and float(res.std()) >= 1e-4

            # ── fold / start 两轴（引擎训练口径 wf val）──────────────
            variants = _rob.build_variants(
                T, self.n_folds, base_gap, ModelConfig.ROBUSTNESS_SHIFT_FRACS
            )
            rows: list[dict[str, Any]] = []
            for var in variants:
                T_eff = T - var["shift"]
                folds = _build_walk_forward_folds(T_eff, var["n_folds"], var["gap"])
                vals: dict[str, Any] = {"label": var["label"], "axis": var["axis"]}
                for i, fml in enumerate(fmls):
                    res = res_map.get(tuple(fml))
                    if not _ok(res):
                        vals[f"f{i}"] = None
                        continue
                    # 丢弃头部 shift 根后重建窗口：与训练一致仅在训练区切分
                    vals[f"f{i}"] = self._wf_val_mean(
                        res[:, var["shift"]:T], t_ret[:, var["shift"]:T], folds
                    )
                rows.append(vals)
            _rob.annotate_rows(rows, n_fml)
            fold_verdict = _rob.axis_verdict(
                [r for r in rows if r["axis"] == "fold"], label="fold")
            start_verdict = _rob.axis_verdict(
                [r for r in rows if r["axis"] == "start"], label="start")

            # ── cost 轴（holdout 段生产口径 Sharpe）───────────────────
            cost_rows: list[dict[str, Any]] = []
            for mult in ModelConfig.ROBUSTNESS_COST_MULTS:
                cost = base_cost * float(mult)
                row: dict[str, Any] = {"label": f"cost={mult:.1f}x",
                                       "axis": "cost", "mult": float(mult)}
                for i, fml in enumerate(fmls):
                    res = res_map.get(tuple(fml))
                    if not _ok(res):
                        row[f"f{i}"] = None
                        row[f"f{i}_ann"] = None
                        continue
                    st = _rob.production_pnl_stats(res, t_ret, ho_start, ho_end,
                                                   cost, ppy)
                    row[f"f{i}"] = st["sharpe"] if st else None
                    row[f"f{i}_ann"] = st["ann_ret_pct"] if st else None
                cost_rows.append(row)

            def _champ_ok(r: dict, need_ann: bool = False) -> bool:
                c = r.get("f0")
                if c is None or c <= _rob.MIN_POSITIVE:
                    return False
                if need_ann:
                    a = r.get("f0_ann")
                    if a is None or a <= 0:
                        return False
                return True

            ok_1 = any(r["mult"] == 1.0 and _champ_ok(r) for r in cost_rows)
            ok_2 = any(r["mult"] == 2.0 and _champ_ok(r, need_ann=True)
                       for r in cost_rows)
            rank_rows = [r for r in cost_rows if r["mult"] in (1.0, 2.0)]
            rank1_n = 0
            for r in rank_rows:
                ri = _rob.rank_index([r.get(f"f{i}") for i in range(n_fml)])
                if ri == 0:
                    rank1_n += 1
            cost_rank_ok = bool(rank_rows) and rank1_n == len(rank_rows)
            cost_verdict: dict[str, Any] = {
                "axis": "cost",
                "passed": bool(ok_1 and ok_2 and cost_rank_ok),
                "n_valid": len(rank_rows),
                "reason": None if (ok_1 and ok_2 and cost_rank_ok) else (
                    "2×成本下冠军不再盈利" if not (ok_1 and ok_2)
                    else "成本扫描下冠军排名被超越"
                ),
            }

            axes = {"fold": fold_verdict, "start": start_verdict, "cost": cost_verdict}
            n_pass = sum(1 for v in axes.values() if v.get("passed"))
            report: dict[str, Any] = {
                "top_k": n_fml,
                "base_cost": base_cost,
                "cost_mults": list(ModelConfig.ROBUSTNESS_COST_MULTS),
                "fold_start_variants": rows,
                "cost_rows": cost_rows,
                "axes": axes,
                "verdict": "robust" if n_pass == 3 else "brittle",
                "passed_axes": n_pass,
            }
            tqdm.write(f"[robustness] 稳健性复核: {report['verdict']} "
                       f"（{n_pass}/3 轴通过，top-{n_fml}）")
            for a, v in axes.items():
                mark = "✓" if v.get("passed") else "✗"
                note = v.get("reason") or ""
                tqdm.write(f"    [{a}] {mark} {note}".rstrip())
            return report
        except Exception as exc:  # noqa: BLE001 复核失败不阻断训练收尾
            tqdm.write(f"[robustness] 复核失败: {type(exc).__name__}: {exc}")
            return None

    # ── 冠军部署（P0）：真实回测选冠军 + 闸门 + 部署/回滚 ───────────────

    def _train_range_meta(self) -> dict[str, Any] | None:
        """当前训练数据的溯源：优先子集 sidecar，旧文件无 sidecar 时按 data_source 推断。"""
        try:
            from data_pipeline.train_sampler import inferred_train_range, read_train_range

            dm = getattr(self, "data_manager", None)
            dm_file = getattr(dm, "file_path", None) or getattr(self, "data_file", None)
            if dm_file:
                tr = read_train_range(str(dm_file))
                if tr:
                    return tr
                # 旧版训练文件无 train_range.json → 用数据元信息推断可展示溯源
                return inferred_train_range(dm_file, getattr(dm, "data_source_meta", lambda: None)())
        except Exception:  # noqa: BLE001 溯源失败不影响部署决策
            pass
        return None

    def _finalize_champion(self) -> dict[str, Any]:
        """训练结束：入围 → holdout 真实回测选冠军 → 统计闸门 → 部署或恢复旧冠军。

        结果写入 self.champion_outcome（train_file 读取，避免二次覆盖）。
        action: deploy / reject_restore / no_champion
        """
        sym = self.target_symbol
        save_path = _strategy_file_for_symbol(sym)
        outcome: dict[str, Any] = {"action": "no_champion", "symbol": sym,
                                   "save_path": save_path}

        # 双 seed 模式：单次训练只评估不部署，全部 seed 通过后再统一部署
        if getattr(self, "_suppress_deploy", False):
            _clear_live_file(sym)
            self.champion_outcome = {"action": "suppressed", "symbol": sym,
                                     "save_path": save_path}
            return self.champion_outcome

        candidates = self._top_finalists(ModelConfig.FINALIST_TOP_K)
        if not candidates:
            _clear_live_file(sym)
            self.champion_outcome = outcome
            return outcome

        # 1) holdout 窗口生产回测（若配置了预留）
        metrics: list[dict] = []
        h = self.holdout_bars
        feat = t_ret = None
        start = end = 0
        if h > 0 and self.data_manager is not None:
            t_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)
            feat = self.data_manager.feat_tensor.to(ModelConfig.DEVICE)
            n_time = t_ret.shape[1]
            if n_time > h + 2:
                start, end = n_time - h, n_time - 2
                for cand in candidates:
                    m = self._rigorous_holdout_pnl(cand["fml"], feat, t_ret, start, end)
                    if m is not None:
                        m["val"] = cand["val"]
                        m["step"] = cand["step"]
                        metrics.append(m)
        if not metrics:
            # 无 holdout 窗口：退回 val 冠军（记录警告，不启用闸门）
            metrics = [{
                "fml": list(self.best_formula) if self.best_formula else [],
                "val": float(self.best_score),
                "step": None,
                "warn_no_holdout": True,
            }]

        def _rank_key(m: dict):
            if m.get("warn_no_holdout"):
                return (-99.0, -99.0, -99.0)
            return (float(m.get("sharpe", -99.0)), float(m.get("sortino", -99.0)),
                    float(m.get("ann_ret", -99.0)))

        metrics.sort(key=_rank_key, reverse=True)
        champ = metrics[0]
        runner = metrics[1] if len(metrics) > 1 else None

        # 2) P1 统计闸门
        null_ok, q99 = self._null_model_check(float(champ.get("val", self.best_score)))
        fold_ok, fold_se = self._fold_se_check(champ["fml"], runner["fml"] if runner else None)

        # 3) P0.1 holdout 闸门（对【冠军】自身指标判定）
        reasons: list[str] = []
        ho_adj = float(champ.get("ho_adj", 0.0))
        val_ref = float(champ.get("val", 0.0) or 0.0)
        ratio = (ho_adj / val_ref) if val_ref > 0 else None
        sharpe = float(champ.get("sharpe", 0.0) or 0.0)
        gate_ok = True
        if not champ.get("warn_no_holdout") and ModelConfig.HOLDOUT_GATE_ENABLED:
            if ho_adj <= ModelConfig.HOLDOUT_MIN_SCORE:
                reasons.append(f"holdout 分 {ho_adj:.4f} ≤ {ModelConfig.HOLDOUT_MIN_SCORE}")
            if ratio is not None and ratio < ModelConfig.HOLDOUT_MIN_RATIO:
                reasons.append(f"保持率 {ratio:.3f} < {ModelConfig.HOLDOUT_MIN_RATIO}")
            if sharpe < ModelConfig.HOLDOUT_MIN_SHARPE:
                reasons.append(f"holdout Sharpe {sharpe:.3f} < {ModelConfig.HOLDOUT_MIN_SHARPE}")
            if not null_ok:
                reasons.append(f"空模型 99 分位未过（val {val_ref:.4f} ≤ q99 {q99:.4f}）")
            if not fold_ok:
                reasons.append("冠军 vs 次优跨折 SE 裕度不足（单折碰运气风险）")
            if ModelConfig.HOLDOUT_SINGLE_USE and self.data_fingerprint:
                rec = _load_holdout_state().get(self.data_fingerprint) or {}
                approved = rec.get("approved_formula")
                if approved is not None and approved != champ["fml"]:
                    reasons.append(
                        "同数据版本 holdout 已被旧冠军消费（数据指纹未变，不得重复批准）"
                    )
            gate_ok = len(reasons) == 0

        # 3.75) P0.2 部署前跨 ticker OOS 复核：holdout 通过后，冠军还须在
        #       第二个标的（默认 ADAUSDT_H1）上通过 vol 覆盖才可部署。
        #       拦“训练文件干净、跨文件失血”的公式（e4_bc 型）。只跑真实
        #       部署（_suppress_deploy 已提前 return）；文件缺失/异常 → fail-open。
        self._oos_review = None
        if gate_ok and not champ.get("warn_no_holdout"):
            try:
                self._oos_review = self.cross_ticker_oos_review(champ["fml"])
            except Exception as exc:  # noqa: BLE001 复核失败放行，不误杀部署
                self._oos_review = {"passed": True,
                                    "error": f"{type(exc).__name__}: {exc}"}
            rv = self._oos_review
            if rv and rv.get("passed") is False:
                worst = rv.get("worst") or {}
                cell = worst.get("cell") or worst.get("tier") or "?"
                t = worst.get("t", float("nan"))
                reasons.append(
                    f"跨ticker OOS vol 覆盖未过（{rv.get('symbol')}/"
                    f"{rv.get('file')} {rv.get('mode')} worst {cell} "
                    f"t={t:.2f}，bps={worst.get('bps')}）"
                )
                gate_ok = False

        # 3.5) P3 稳健性复核（仅当冠军未过闸门）：成本/折叠/起点敏感性，
        #      结论随 champion_history 记录，不改部署决策。
        self._robustness_result = None
        if (not gate_ok and ModelConfig.ENABLE_FINALIST_ROBUSTNESS
                and not champ.get("warn_no_holdout")
                and feat is not None and len(metrics) >= 2):
            self._robustness_result = self._run_finalist_robustness(
                metrics, feat, t_ret, start, end,
            )
        robustness = self._robustness_result

        old_champ = _load_last_champion(sym) or self._champion_at_start
        has_old = bool(old_champ and old_champ.get("formula"))
        same_as_old = has_old and old_champ.get("formula") == champ["fml"]

        # 4) 部署 / 恢复
        if gate_ok or not has_old or same_as_old:
            if not has_old and not gate_ok:
                reasons.append("首冠军：无旧版本可保留，按部署处理（警告）")
            try:
                from .vocab import VOCAB_VERSION
                strategy_data: dict[str, Any] = {
                    "vocab_version": VOCAB_VERSION,
                    "symbol": sym,
                    "formula": champ["fml"],
                    "best_score": round(float(champ.get("val", self.best_score)), 6),
                    "formula_decoded": self._decode_formula(champ["fml"]),
                    "finalist_rank": 1,
                    "finalists": [
                        {"formula": m["fml"], "val": round(float(m.get("val", 0)), 4),
                         "sharpe": round(float(m.get("sharpe", 0)), 3),
                         "sortino": round(float(m.get("sortino", 0)), 3)}
                        for m in metrics[:10]
                    ],
                    "holdout_bars": int(h),
                    "holdout": {
                        "passed": bool(gate_ok),
                        "score": round(ho_adj, 4),
                        "sharpe": round(sharpe, 3),
                        "score_ratio": round(ratio, 4) if ratio is not None else None,
                        "gate": reasons,
                    },
                    "null_model_q99": round(q99, 4) if q99 is not None else None,
                    "fold_se": fold_se,
                    "data_fingerprint": self.data_fingerprint,
                    "seed": self.seed,
                    "cost_rate": float(ModelConfig.FINALIST_COST_RATE),
                }
                # 并入训练溯源元数据（data_source + train_range：抽样模式/参数/覆盖年代）
                strategy_data = _merge_train_metadata(self, strategy_data)
                _tr = self._train_range_meta()
                if _tr:
                    strategy_data["train_range"] = _tr
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                tmp_path = save_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as fp:
                    json.dump(strategy_data, fp, indent=2, ensure_ascii=False)
                os.replace(tmp_path, save_path)
                self._dump_finalists_json(metrics)
                _deploy_event: dict[str, Any] = {
                    "event": "deploy", "symbol": sym,
                    "formula": champ["fml"],
                    "best_score": strategy_data["best_score"],
                    "holdout": strategy_data["holdout"],
                    "data_fingerprint": self.data_fingerprint,
                    "seed": self.seed,
                    "train_range": self._train_range_meta(),
                }
                if robustness:
                    _deploy_event["robustness"] = robustness
                if self._oos_review is not None:
                    _deploy_event["oos_review"] = self._oos_review
                _record_champion_event(_deploy_event)
                if gate_ok and self.data_fingerprint and ModelConfig.HOLDOUT_SINGLE_USE:
                    _consume_holdout(self.data_fingerprint, champ["fml"])
                outcome = {
                    "action": "deploy", "symbol": sym, "save_path": save_path,
                    "formula": champ["fml"], "best_score": strategy_data["best_score"],
                    "reasons": reasons,
                }
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[冠军闸门] 部署失败: {exc}")
                _clear_live_file(sym)
                self.champion_outcome = outcome
                return outcome
        else:
            # 恢复旧冠军：_save_strategy_live 可能已把文件覆盖成未验证公式
            restored = False
            try:
                if old_champ:
                    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = save_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as fp:
                        json.dump(old_champ, fp, indent=2, ensure_ascii=False)
                    os.replace(tmp_path, save_path)
                    restored = True
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[冠军闸门] 旧冠军恢复失败: {exc}")
            _reject_event: dict[str, Any] = {
                "event": "reject", "symbol": sym,
                "candidate_formula": champ["fml"],
                "candidate_val": round(float(champ.get("val", 0)), 4),
                "candidate_sharpe": round(sharpe, 3),
                "reasons": reasons,
                "data_fingerprint": self.data_fingerprint,
                "seed": self.seed,
                "train_range": self._train_range_meta(),
            }
            if robustness:
                _reject_event["robustness"] = robustness
            if self._oos_review is not None:
                _reject_event["oos_review"] = self._oos_review
            _record_champion_event(_reject_event)
            outcome = {
                "action": "reject_restore", "symbol": sym, "save_path": save_path,
                "reasons": reasons,
                "candidate_formula": champ["fml"],
                "old_formula": old_champ.get("formula") if old_champ else None,
                "restored": restored,
            }
            for r in reasons:
                tqdm.write(f"[冠军闸门] 拒绝部署: {r}")

        _clear_live_file(sym)
        self.champion_outcome = outcome
        return outcome

    def _dump_finalists_json(self, metrics: list[dict]) -> None:
        """落盘入围 top-K 及其生产回测指标（供 P3 敏感性脚本使用）。"""
        try:
            if not self.target_symbol:
                return
            path = pathlib.Path(
                ModelConfig.FINALISTS_JSON.format(symbol=self.target_symbol)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump({
                    "symbol": self.target_symbol,
                    "data_fingerprint": self.data_fingerprint,
                    "seed": self.seed,
                    "cost_rate": float(ModelConfig.FINALIST_COST_RATE),
                    "holdout_bars": int(self.holdout_bars),
                    "train_range": self._train_range_meta(),
                    "finalists": [
                        {k: (m[k] if k != "fml" else list(m[k])) for k in m}
                        for m in metrics
                    ],
                }, fp, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"[finalists] 落盘失败: {exc}")

    @staticmethod
    def _compute_ic(factor: torch.Tensor, target_ret: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """时序 IC（每品种内部 factor[t] vs ret[t+1]）的均值与稳定性。

        对 5 品种宇宙，时序 IC 比横截面 IC 统计意义更强。
        """
        N, T = factor.shape
        if T < 2:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_list = []
        # P2 对齐修正：target_ret[t] = log(open[t+2]/open[t+1])，
        # position[t]（=tanh(factor[t])）产生 open[t+1]→open[t+2] 的收益，
        # 因此 IC 必须配对 factor[t]~target_ret[t]（原实现错配成
        # factor[t]~target_ret[t+1]，比 PnL 晚一根 bar）。
        # 尾部裁剪：最后两根 target 恒为 0（边界），退出配对。
        for n in range(N):
            x  = factor[n, :-2]
            y  = target_ret[n, :-2]
            if x.numel() < 2:
                continue
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic)

        if not ic_list:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_tensor = torch.stack(ic_list)
        ic_mean   = ic_tensor.mean()
        ic_stab   = (ic_mean / (ic_tensor.std(unbiased=False) + 1e-6)
                     if ic_tensor.numel() >= 2
                     else torch.zeros(1, device=factor.device))
        return ic_mean, ic_stab

    # ── IC gate: direction-based, dimension-agnostic ──────────────────────────

    @staticmethod
    def _apply_ic_gate(reward: torch.Tensor, ic_mean) -> torch.Tensor:
        """IC 门控：用 IC 符号而非量值调整 reward，完全规避量纲问题。
        IC > thresh  → reward × IC_GATE_MULT  (正向预测，奖励)
        IC < -thresh → reward × IC_NEG_MULT   (反向预测，惩罚)
        |IC| ≤ thresh→ 不修改                  (噪声区)
        """
        ic_val = ic_mean.item() if isinstance(ic_mean, torch.Tensor) else float(ic_mean)
        t = ModelConfig.IC_GATE_THRESH
        if ic_val > t:
            return reward * ModelConfig.IC_GATE_MULT
        elif ic_val < -t:
            return reward * ModelConfig.IC_NEG_MULT
        return reward


    # ── 选优层 vol 覆盖（regime_ada 诊断落地）─────────────────────────────

    def _init_vol_tiers(self, feat: torch.Tensor, t_ret: torch.Tensor) -> None:
        """训练窗口内按因果 realized-vol 把成熟 bar 分低/中/高三段并缓存索引。

        vol[t] = t_ret[:, t-W+1 .. t] 的滚动 std（跨品种取均值），
        W = VOL_COVERAGE_WINDOW（与 regime 诊断同口径）。分位边界在全体成熟
        样本上取 1/3、2/3：成熟样本 t ∈ [W-1, T-2]（尾部两 bar target=0 排除，
        与 IC/评分一致）。只覆盖训练区（holdout 已被 train() 切掉），不消费
        holdout、不参与 reward。数据过短/异常 → 清空缓存（闸门自动失效）。

        VOL_COVERAGE_GRID=True 时改用 _init_vol_grid（vol×er 9 格），语义不变。
        """
        self._vol_tier_idx = None
        self._vol_tier_meta = None
        self._vol_grid_cells = None
        self._vol_grid_meta = None
        if not ModelConfig.VOL_COVERAGE_ENABLED:
            return
        try:
            if ModelConfig.VOL_COVERAGE_GRID:
                self._init_vol_grid(t_ret)
                return
            T = int(t_ret.shape[1])
            W = int(ModelConfig.VOL_COVERAGE_WINDOW)
            min_bars = int(ModelConfig.VOL_COVERAGE_MIN_BARS)
            if T < W + 3 * min_bars:
                return
            N = t_ret.shape[0]
            pad = torch.zeros(N, W - 1, device=t_ret.device, dtype=t_ret.dtype)
            xp = torch.cat([pad, t_ret], dim=1)          # [N, T+W-1]
            rv = xp.unfold(1, W, 1).std(dim=2).mean(dim=0)   # [T]
            lo_bar, hi_bar = W - 1, T - 2                # 成熟区间 [lo_bar, hi_bar)
            if hi_bar - lo_bar < 3 * min_bars:
                return
            usable = rv[lo_bar:hi_bar]                   # [U]
            q1, q2 = torch.quantile(
                usable, torch.tensor([1/3, 2/3], device=usable.device))
            pos = torch.arange(lo_bar, hi_bar, device=t_ret.device)
            labels = ["低vol", "中vol", "高vol"]
            tiers: list[torch.Tensor] = []
            metas: list[dict] = []
            for ti in range(3):
                if ti == 0:
                    m = usable <= q1
                elif ti == 1:
                    m = (usable > q1) & (usable <= q2)
                else:
                    m = usable > q2
                idx = pos[m]
                tiers.append(idx)
                metas.append({"tier": labels[ti], "bars": int(idx.numel())})
            self._vol_tier_idx = tiers
            self._vol_tier_meta = {"window": int(W),
                                   "lo_bar": int(lo_bar), "hi_bar": int(hi_bar),
                                   "vol_q": [float(q1), float(q2)],
                                   "tiers": metas}
        except Exception as exc:  # noqa: BLE001 失败即闸门失效（fail-open）
            self._vol_tier_idx = None
            self._vol_tier_meta = {"error": f"{type(exc).__name__}: {exc}"}

    def _init_vol_grid(self, t_ret: torch.Tensor) -> None:
        """格级升级：按因果 realized-vol × 效率比(ER)把成熟 bar 分 3×3 九格。

        vol[t] = t_ret[:, t-48+1 .. t] 滚动 std（跨品种均值）；
        er[t]  = |Σ r| / Σ|r|（窗口 VOL_COVERAGE_ER_WINDOW=120，同 regime_health），
        0≈震荡、1≈混合、2≈趋势。成熟样本 t ∈ [max(W-1, W_ER-1), T-2]，两维
        三分位边界都在该集合上取。每格独立缓存索引，供 _vol_pnl_coverage 用
        任一格显著为负即拒。数据过短/异常 → 清空缓存（fail-open）。
        """
        self._vol_grid_cells = None
        self._vol_grid_meta = None
        try:
            T = int(t_ret.shape[1])
            W = int(ModelConfig.VOL_COVERAGE_WINDOW)
            W_ER = int(ModelConfig.VOL_COVERAGE_ER_WINDOW)
            min_bars = int(ModelConfig.VOL_COVERAGE_MIN_BARS)
            lo = max(W - 1, W_ER - 1)
            hi = T - 2
            if hi - lo < min_bars:
                return
            N = t_ret.shape[0]
            dev = t_ret.device
            dt = t_ret.dtype
            # vol（与段级同口径）
            pad = torch.zeros(N, W - 1, device=dev, dtype=dt)
            xp = torch.cat([pad, t_ret], dim=1)          # [N, T+W-1]
            rv = xp.unfold(1, W, 1).std(dim=2).mean(dim=0)   # [T]
            # er：窗口内 |净位移| / 总路程（跨品种均值）
            pad2 = torch.zeros(N, W_ER - 1, device=dev, dtype=dt)
            xe = torch.cat([pad2, t_ret], dim=1)         # [N, T+W_ER-1]
            win = xe.unfold(1, W_ER, 1)                  # [N, T, W_ER]
            er = (win.sum(-1).abs() / win.abs().sum(-1).clamp_min(1e-12)
                  ).mean(0)                              # [T]
            usable = slice(lo, hi)
            rv_u, er_u = rv[usable], er[usable]
            qv = torch.quantile(rv_u, torch.tensor([1/3, 2/3], device=dev))
            qe = torch.quantile(er_u, torch.tensor([1/3, 2/3], device=dev))
            pos = torch.arange(lo, hi, device=dev)
            vl = torch.full((T,), -1, device=dev, dtype=torch.long)
            el = torch.full((T,), -1, device=dev, dtype=torch.long)
            vl[pos] = torch.where(rv_u <= qv[0], 0, torch.where(rv_u <= qv[1], 1, 2))
            el[pos] = torch.where(er_u <= qe[0], 0, torch.where(er_u <= qe[1], 1, 2))
            cells: list[dict] = []
            metas: list[dict] = []
            for vi in range(3):
                for ei in range(3):
                    idx = pos[(vl[pos] == vi) & (el[pos] == ei)]
                    cells.append({"idx": idx, "vol": vi, "er": ei,
                                  "cell": f"v{vi}e{ei}"})
                    metas.append({"cell": f"v{vi}e{ei}",
                                  "vol": ["低vol", "中vol", "高vol"][vi],
                                  "er": ["震荡", "混合", "趋势"][ei],
                                  "bars": int(idx.numel())})
            self._vol_grid_cells = cells
            self._vol_grid_meta = {
                "window": int(W), "er_window": int(W_ER),
                "lo_bar": int(lo), "hi_bar": int(hi),
                "vol_q": [float(qv[0]), float(qv[1])],
                "er_q": [float(qe[0]), float(qe[1])],
                "cells": metas,
            }
        except Exception as exc:  # noqa: BLE001 失败即闸门失效（fail-open）
            self._vol_grid_cells = None
            self._vol_grid_meta = {"error": f"{type(exc).__name__}: {exc}"}

    def _vol_pnl_coverage(self, res: torch.Tensor, t_ret: torch.Tensor
                          ) -> tuple[bool, dict]:
        """候选的 vol 覆盖检查（生产口径，与 holdout 一致）。

        pos = tanh(res)；pnl = pos·t_ret − |Δpos|·cost。
        - 段级（VOL_COVERAGE_GRID=False）：对每个 vol 段取 pnl 单尾 t；任一段
          t < VOL_COVERAGE_MIN_T 即失败。
        - 格级（=True）：对 vol×er 每格取同样 t；任一格（样本足够）显著为负即
          失败，拦“段级聚合掩盖单格出血”。
        样本不足的段/格跳过（fail-open）。只读、无副作用，线程安全。
        """
        with torch.no_grad():
            pos = compute_target_positions_stateless(res)
            prev = torch.roll(pos, 1, dims=1)
            prev[:, 0] = 0.0
            turnover = torch.abs(pos - prev)
            pnl = pos * t_ret - turnover * self.bt.cost_rate   # [1, T]
            p0 = pnl[0]
            min_bars = int(ModelConfig.VOL_COVERAGE_MIN_BARS)
            min_t = float(ModelConfig.VOL_COVERAGE_MIN_T)

            if ModelConfig.VOL_COVERAGE_GRID:
                cells = getattr(self, "_vol_grid_cells", None)
                if not cells or len(cells) != 9:
                    return True, {"skipped": "no grid cells"}
                ok = True
                rows: list[dict] = []
                for c in cells:
                    idx = c["idx"]
                    nb = int(idx.numel())
                    if nb < min_bars:
                        rows.append({"cell": c["cell"], "bars": nb,
                                     "skipped": True})
                        continue
                    p = p0[idx]
                    n = int(p.numel())
                    mean = float(p.mean())
                    sd = float(p.std())
                    t_stat = mean / (sd + 1e-12) * math.sqrt(n)
                    if t_stat < min_t:
                        ok = False
                    rows.append({"cell": c["cell"],
                                 "vol": c["vol"], "er": c["er"],
                                 "bars": n, "bps": round(mean * 1e4, 3),
                                 "t": round(t_stat, 2),
                                 "pass": bool(t_stat >= min_t)})
                meta = getattr(self, "_vol_grid_meta", None) or {}
                return bool(ok), {
                    "grid": True,
                    "vol_q": meta.get("vol_q"), "er_q": meta.get("er_q"),
                    "cells": rows,
                }

            tiers = getattr(self, "_vol_tier_idx", None)
            if not tiers or len(tiers) != 3:
                return True, {"skipped": "no tiers"}
            ok = True
            rows = []
            for idx in tiers:
                if idx.numel() < min_bars:
                    rows.append({"bars": int(idx.numel()), "skipped": True})
                    continue
                p = p0[idx]
                n = int(p.numel())
                mean = float(p.mean())
                sd = float(p.std())
                t_stat = mean / (sd + 1e-12) * math.sqrt(n)
                if t_stat < min_t:
                    ok = False
                rows.append({"bars": n, "bps": round(mean * 1e4, 3),
                             "t": round(t_stat, 2),
                             "pass": bool(t_stat >= min_t)})
            meta = getattr(self, "_vol_tier_meta", None) or {}
            return bool(ok), {"vol_q": meta.get("vol_q"), "tiers": rows}

    # ── 部署前跨 ticker OOS 复核（P0.2）────────────────────────────────────

    def _oos_review_on_file(self, formula: list[int], fp: str) -> dict:
        """在第二个标的文件上对 formula 跑 vol 覆盖(与选优闸门同口径、同实现)。

        用一个轻量 scratch engine 复用 _init_vol_tiers/_vol_pnl_coverage，
        不碰 self 的训练缓存、不消费任何预留窗口。常量/非法公式 → fail-open。
        """
        from data_pipeline.parquet_manager import ParquetDataManager
        mgr = ParquetDataManager(fp)
        mgr.load()
        feat = mgr.feat_tensor.to(ModelConfig.DEVICE)
        t_ret = mgr.target_ret.to(ModelConfig.DEVICE)
        scratch = AlphaEngine(data_manager=mgr, target_symbol=mgr.symbol, seed=0)
        scratch._init_vol_tiers(feat, t_ret)
        base: dict = {
            "file": str(pathlib.Path(fp).name), "symbol": mgr.symbol,
            "mode": "grid" if ModelConfig.VOL_COVERAGE_GRID else "tier",
            "bars": int(t_ret.shape[1]),
        }
        res = scratch.vm.execute(list(formula), feat)
        if res is None or float(res.std()) < 1e-4:
            base.update({"passed": True, "skipped": "constant/invalid"})
            return base
        ok, info = scratch._vol_pnl_coverage(res, t_ret)
        base["info"] = info
        if not ok and info:
            segs = info.get("cells") if info.get("grid") else info.get("tiers")
            worst = None
            for r in segs or []:
                if "t" in r and (worst is None or r["t"] < worst["t"]):
                    worst = r
            if worst is not None:
                base["worst"] = worst
        base["passed"] = bool(ok)
        # 附带整体 bps(仅诊断，不作否决条件)
        try:
            meta = (scratch._vol_grid_meta if ModelConfig.VOL_COVERAGE_GRID
                    else scratch._vol_tier_meta) or {}
            lo = int(meta.get("lo_bar", 0))
            hi = int(meta.get("hi_bar", int(t_ret.shape[1])))
            with torch.no_grad():
                pos = compute_target_positions_stateless(res)
                prev = torch.roll(pos, 1, dims=1)
                prev[:, 0] = 0.0
                pnl = pos * t_ret - (pos - prev).abs() * scratch.bt.cost_rate
                base["overall_bps"] = round(float(pnl[0, lo:hi].mean()) * 1e4, 3)
        except Exception:  # noqa: BLE001 诊断字段失败不影响结论
            pass
        return base

    def cross_ticker_oos_review(self, formula: list[int]) -> dict | None:
        """部署前 P0.2: 冠军在第二标的上过 vol 覆盖才可部署。

        任一配置文件覆盖失败 → 整体失败(veto)。文件缺失(且不 SKIP_MISSING)/加载
        异常/公式常量 → fail-open。VOL_COVERAGE_ENABLED=False 或未配文件 → 跳过。
        只读，由 _finalize_champion 在真实部署时调用。
        """
        if not (ModelConfig.CROSS_TICKER_OOS_ENABLED
                and ModelConfig.VOL_COVERAGE_ENABLED):
            return None
        files = list(ModelConfig.CROSS_TICKER_OOS_FILES or ())
        if not files:
            return None
        last: dict | None = None
        for fp in files:
            if not pathlib.Path(fp).exists():
                if ModelConfig.CROSS_TICKER_OOS_SKIP_MISSING:
                    continue
                return {"passed": True, "file": fp, "skipped": "missing"}
            last = self._oos_review_on_file(list(formula), str(fp))
            if last and last.get("passed") is False:
                return last
        if last is None:
            return {"passed": True, "skipped": "no OOS review files available"}
        return last

    # ── Elite pool ────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_elite_pool(
        pool: list[tuple[float, int, list[int], int]]
    ) -> list[tuple[float, int, list[int], int]]:
        """对精英池去重：相同 tokens 只保留得分最高的一条，重建最小堆。"""
        best: dict[str, tuple[float, int, list[int], int]] = {}
        for sc, cnt, toks, birth in pool:
            key = str(toks)
            if key not in best or sc > best[key][0]:
                best[key] = (sc, cnt, toks, birth)
        deduped = list(best.values())
        heapq.heapify(deduped)
        return deduped

    def _update_elite_pool(self, val_score: float, formula: list[int], step: int = 0) -> None:
        """维护精英公式池（最小堆，Top-ELITE_POOL_SIZE 个历史最优公式，自动去重）。

        去重逻辑：若 formula 已在池中，只在新得分更高时原地更新，不插入重复副本。
        这防止了单一公式垄断 elite pool，保持多样性。
        新增：记录 birth_step 用于 elite decay。
        """
        k = ModelConfig.ELITE_POOL_SIZE

        # 检查是否已有相同公式
        for idx, (sc, cnt, toks, birth) in enumerate(self._elite_pool):
            if toks == formula:
                if val_score <= sc:
                    return  # 已有更高分的相同公式，不更新
                # 分数更高：从堆中移除旧条目，插入新条目
                self._elite_pool[idx] = self._elite_pool[-1]
                self._elite_pool.pop()
                heapq.heapify(self._elite_pool)  # O(k)，k≤20，可接受
                break

        entry = (val_score, self._elite_counter, list(formula), step)
        self._elite_counter += 1
        if len(self._elite_pool) < k:
            heapq.heappush(self._elite_pool, entry)
        elif val_score > self._elite_pool[0][0]:
            heapq.heapreplace(self._elite_pool, entry)

    # ── Factor pool ───────────────────────────────────────────────────────────

    def _update_factor_pool(self, val_score: float, factor: torch.Tensor) -> None:
        k     = ModelConfig.FACTOR_TOP_K
        f_gpu = factor.detach()
        entry = (val_score, self._factor_pool_counter, f_gpu)
        self._factor_pool_counter += 1
        if len(self.factor_pool) < k:
            heapq.heappush(self.factor_pool, entry)
        elif val_score > self.factor_pool[0][0]:
            heapq.heapreplace(self.factor_pool, entry)

    def _apply_corr_penalty(
        self,
        reward: torch.Tensor,
        factor: torch.Tensor,
        train_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """相关性惩罚：与因子池中已有因子的相关性超过阈值则惩罚 reward。

        P1-6 修复：相关性只在 train 切片上计算，避免含 val 段数据泄漏。
        train_slice=None 时回退到整段（向后兼容）。
        """
        if not self.factor_pool:
            return reward
        # P1-6: 相关性只在 train 切片上计算，避免 val 信息泄漏
        if train_slice is not None:
            s, e = train_slice
            f = factor.detach()[:, s:e]
        else:
            f = factor.detach()
        f_flat = f.reshape(-1).float()
        if f_flat.std() < 1e-4:
            return reward
        # 因子池中的历史因子也按相同切片取（若形状一致）
        pool_vecs_list = []
        for _, _cnt, pf in self.factor_pool:
            pf_t = pf.detach()
            if train_slice is not None and pf_t.shape[1] >= factor.shape[1]:
                pf_t = pf_t[:, s:e]
            pool_vecs_list.append(pf_t.reshape(-1).float())
        if not pool_vecs_list:
            return reward
        pool_vecs = torch.stack(pool_vecs_list, dim=0)
        f_c  = f_flat - f_flat.mean()
        p_c  = pool_vecs - pool_vecs.mean(dim=1, keepdim=True)
        cov  = (p_c * f_c).sum(dim=1)
        sx   = f_c.norm() + 1e-8
        sy   = p_c.norm(dim=1) + 1e-8
        corr = (cov / (sx * sy)).abs()
        if (corr > ModelConfig.CORR_THRESHOLD).any():
            reward = reward * ModelConfig.CORR_PENALTY
        return reward

    def _distribution_stats(self, prev_dist=None):
        """计算模型初始位置（zero prefix）token 分布的细化指标，用于判断 H 不变时
        分布是否真的在变化。
        """
        vocab_size = FORMULA_VOCAB.size
        with torch.no_grad():
            inp = torch.zeros((1, 1), dtype=torch.long,
                              device=ModelConfig.DEVICE)
            logits, _, _ = self.model(inp)
            logits = self.sampler.apply_mask_to_logits(
                logits, [0], 0, ModelConfig.MAX_FORMULA_LEN
            )
            dist = F.softmax(logits, dim=-1).squeeze(0)
            ent = -(dist * torch.log(dist + 1e-12)).sum().item()
            log_v = math.log(vocab_size)
            kl_uniform = log_v - ent
            top1 = dist.max().item()
            top5 = dist.topk(5, dim=-1).values.sum().item()
            eff_vocab = math.exp(ent)
            prob_std = dist.std(unbiased=False).item()
            kl_prev = 0.0
            if prev_dist is not None:
                kl_prev = (
                    dist * (torch.log(dist + 1e-12) -
                            torch.log(prev_dist.to(dist.device) + 1e-12))
                ).sum().item()
        return {
            'dist': dist.cpu(),
            'entropy': ent,
            'kl_uniform': kl_uniform,
            'top1_prob': top1,
            'top5_prob': top5,
            'eff_vocab': eff_vocab,
            'prob_std': prob_std,
            'kl_prev': kl_prev,
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, start_step: int = 0, end_step: int | None = None,
              migration_hook=None, verbose_header: bool = True):
        if self.data_manager is None:
            raise RuntimeError("AlphaEngine requires a data_manager.")

        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS

        if verbose_header:
            print("开始 Alpha 因子挖掘训练" +
                  ("（含 LoRD 正则化）..." if self.use_lord else "..."))
            print(f"   策略熵: 坍塌阈值={ModelConfig.ENTROPY_COLLAPSE_THRESH}  "
                  f"系数上限={ModelConfig.ENTROPY_COEFF_MAX}  "
                  f"连续坍塌步数={ModelConfig.ENTROPY_COLLAPSE_STEPS}")
            print(f"   精英回放: 比例={ModelConfig.ELITE_REPLAY_FRAC}  "
                  f"池大小={ModelConfig.ELITE_POOL_SIZE}")
            print(f"   IC门控: 阈值±{ModelConfig.IC_GATE_THRESH}  "
                  f"正向×{ModelConfig.IC_GATE_MULT}  负向×{ModelConfig.IC_NEG_MULT}")
            print(f"   最大重启: {ModelConfig.MAX_RESTARTS}  "
                  f"噪声={ModelConfig.RESTART_NOISE}")

        # ── 样本外预留：尾部 holdout 根数不进任何训练/折叠/选优 ────────
        T_full = self.data_manager.target_ret.shape[1]
        self.holdout_bars = self._resolve_holdout_bars(T_full)
        T = T_full - self.holdout_bars

        # ── 冠军闸门前置状态（P0）────────────────────────────────────
        # 数据指纹：同版本数据 holdout 只允许消费一次（防反复重训挑彩票）
        fp = getattr(self.data_manager, "fingerprint", None)
        if fp:
            self.data_fingerprint = fp
        # 训练开始时磁盘上的冠军快照：holdout 闸门拒绝时用它恢复。
        # （P0 加固后训练中途只写 *.live.json 侧车，部署路径始终是旧冠军，
        #  此快照仍保留以应对“训练前文件已被旧版本代码覆盖”的迁移场景。）
        if self.target_symbol and self._champion_at_start is None:
            p = pathlib.Path(_strategy_file_for_symbol(self.target_symbol))
            if p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict) and raw.get("formula"):
                        self._champion_at_start = raw
                except Exception:
                    self._champion_at_start = None

        # P0 加固：清掉上次遗留的 live 侧车；注册 SIGTERM/SIGINT 处理器，
        # 训练被中途终止（web 停止按钮 / Ctrl+C）时也能清理侧车、避免残留误导。
        try:
            p = pathlib.Path(_live_strategy_file_for_symbol(self.target_symbol))
            if p.exists():
                p.unlink()
        except OSError:
            pass
        try:
            import signal as _signal

            def _on_abort(_sig, _frame):
                # 防假中止：父进程已消失的孤儿 SIGTERM（终端/ssh/launchd 会话被回收）
                # 先原地重挂 launchd 再继续训练，不当作中止处理。
                try:
                    from model_core.supervise import guard_sigterm
                    if guard_sigterm(f"train_{self.target_symbol or 'multi'}"):
                        tqdm.write("[冠军闸门] SIGTERM 但父进程已消失（会话回收）："
                                   "已重挂 launchd，继续训练")
                        return
                except Exception:  # noqa: BLE001 守卫失败回退原中止语义
                    pass
                try:
                    live_p = pathlib.Path(_live_strategy_file_for_symbol(self.target_symbol))
                    if live_p.exists():
                        live_p.unlink()
                except OSError:
                    pass
                tqdm.write(f"[冠军闸门] 训练被中止：已清理 *.live.json 侧车，"
                           f"部署路径策略未被覆盖")
                raise SystemExit(128 + int(_sig))

            _signal.signal(_signal.SIGTERM, _on_abort)
            _signal.signal(_signal.SIGINT, _on_abort)
        except (ValueError, AttributeError, OSError):
            pass  # 非主线程 / 无信号环境（如部分测试宿主）下跳过

        folds = _build_walk_forward_folds(T, self.n_folds,
                                          gap=getattr(ModelConfig, 'WF_GAP', 20))
        use_wf = len(folds) > 1 and not (
            folds[0]["train_start"] == 0 and folds[0]["train_end"] == T
        )
        # P1 统计闸门复用同一份训练上下文（空模型基准 / 跨折 SE）
        self._train_folds = folds
        self._train_feat = None
        self._train_t_ret = None
        if verbose_header:
            print(f"   样本外预留: 最后 {self.holdout_bars} 根不参与训练/选优")
            if use_wf:
                print(f"   滚动验证: {len(folds)} 折  训练窗口共 {T} 根K线")
                for k, f in enumerate(folds):
                    print(f"  第{k+1}折: 训练[{f['train_start']},{f['train_end']}) "
                          f"间隔={f['gap']} 验证[{f['val_start']},{f['val_end']})")
            else:
                print(f"   退化为全量评估（训练窗口共 {T} 根K线）")

        # 因果安全：features.py 的 _robust_norm 已改为滚动因果实现
        # 每个 t 的归一化参数只用 [t-w+1..t]，walk-forward 折叠切片无泄露
        feat  = self.data_manager.feat_tensor.to(ModelConfig.DEVICE)
        t_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)
        # 截断到训练窗口：holdout 尾部对训练不可见
        # 注意 feat 是 [N, C, T]，必须用 [:, :, :T]——原实现写成 [:, :T]
        # 切的是通道维（65），时间维从未被截断，导致 position×t_ret 形状不匹配，
        # 训练评分每步全部 error（holdout 预留引入后训练分数恒为 -inf）。
        if self.holdout_bars > 0:
            feat = feat[:, :, :T]
            t_ret = t_ret[:, :T]
        self._train_feat = feat
        self._train_t_ret = t_ret

        # 选优层 vol 覆盖：按训练窗口（holdout 已切掉）缓存低/中/高 vol 三段索引
        self._init_vol_tiers(feat, t_ret)

        # 数据驱动年化因子：按训练数据的实际时间戳估计每年 bar 数，
        # 替代 ContinuousBacktest 默认的 H1=6240。A 股日线/15min、加密日线等
        # 非 H1 周期不再被按 H1 年化（否则 Sharpe/年化收益被放大数倍）。
        _dm_raw = getattr(self.data_manager, "raw_dict", None) or {}
        _dm_time = _dm_raw.get("time", None)
        if _dm_time is not None:
            try:
                _ppy = estimate_periods_per_year(_dm_time)
                if _ppy != self.bt.periods_per_year:
                    if verbose_header:
                        print(f"   年化因子: {_ppy} bar/年（按数据周期自动估计）")
                    self.bt.periods_per_year = _ppy
            except Exception:
                pass  # 估计失败则保留默认 6240

        bs      = ModelConfig.BATCH_SIZE
        n_elite = max(1, int(bs * ModelConfig.ELITE_REPLAY_FRAC))
        n_new   = bs - n_elite

        remaining = end_step - start_step
        if remaining <= 0:
            print(f"[训练] 起始步 {start_step} 已达目标步 {end_step}，无需继续训练。")
            self._verify_holdout()
            return

        # 非交互/重定向输出时关闭 tqdm 进度条，避免进度条刷屏把自定义日志淹掉。
        # tqdm.write 仍然可用，详细 step 日志会继续输出。
        pbar               = tqdm(range(start_step, end_step),
                                  total=end_step,
                                  initial=start_step,
                                  disable=not sys.stderr.isatty(),
                                  leave=False,
                                  mininterval=5.0)
        low_entropy_streak = 0
        prev_init_dist     = None  # 用于计算相邻步分布差异 KL

        for step in pbar:
            # ── Part A: Sample n_new new formulas ────────────────────
            inp_new = torch.zeros((n_new, 1), dtype=torch.long,
                                  device=ModelConfig.DEVICE)
            lp_new, tok_new, ent_new, v_new = [], [], [], []
            sd_new = [0] * n_new
            prev_tokens_new: list[int | None] = [None] * n_new
            infected_chain_new: list[int] = [0] * n_new

            for si in range(ModelConfig.MAX_FORMULA_LEN):
                lg, val_now, _ = self.model(inp_new)
                if val_now is not None:
                    v_new.append(val_now)
                lg = self.sampler.apply_mask_to_logits(lg, sd_new, si,
                                                       ModelConfig.MAX_FORMULA_LEN,
                                                       prev_tokens=prev_tokens_new,
                                                       infected_chain_lens=infected_chain_new)
                d  = Categorical(logits=lg)
                a  = d.sample()
                lp_new.append(d.log_prob(a))
                tok_new.append(a)
                ent_new.append(d.entropy())
                inp_new = torch.cat([inp_new, a.unsqueeze(1)], dim=1)
                for b in range(n_new):
                    sd_new[b] += self.sampler.delta[a[b].item()]
                    prev_tokens_new[b] = a[b].item()
                    infected_chain_new[b] = self.sampler.update_infection(
                        a[b].item(), infected_chain_new[b])

            seqs_new = torch.stack(tok_new, dim=1)


            # ── Part B: Elite Replay ─────────────────────────────────
            elite_formulas: list[list[int]] = []
            if self._elite_pool and n_elite > 0:
                ps = []
                pt = []
                weights = []
                for sc, cnt, toks, birth in self._elite_pool:
                    age = max(0, step - birth)
                    decay = 1.0
                    if ModelConfig.ELITE_DECAY:
                        half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                        decay = 0.5 ** (age / half)
                    ps.append(sc)
                    pt.append(toks)
                    weights.append(decay)
                ps_min  = min(ps)
                ps_max  = max(ps)
                # 软温度采样：避免最高分公式垄断
                # 先归一到 [0,1]，再除以温度 T=0.5 后做 softmax
                # T<1 → 高分公式仍被偏好，但不再独占
                if ps_max > ps_min:
                    normalized = [(s - ps_min) / (ps_max - ps_min + 1e-8) for s in ps]
                else:
                    normalized = [1.0] * len(ps)
                temp = 0.5
                exp_s = [weights[i] * (2.0 ** (normalized[i] / temp)) for i in range(len(ps))]
                exp_sum = sum(exp_s)
                probs = [e / exp_sum for e in exp_s]
                idx_e   = random.choices(range(len(self._elite_pool)),
                                         weights=probs, k=n_elite)
                elite_formulas = [pt[i] for i in idx_e]

                # 详细日志：Elite Replay 衰减状态（每 100 步打印一次）
                if step % 100 == 0:
                    avg_decay = sum(weights) / len(weights)
                    max_age = max(max(0, step - birth) for _, _, _, birth in self._elite_pool)
                    age_list = sorted([max(0, step - birth) for _, _, _, birth in self._elite_pool])
                    tqdm.write(
                        f"[精英回放 @ 第{step}步] 池大小={len(self._elite_pool)} "
                        f"平均衰减={avg_decay:.3f} 最大龄期={max_age} 龄期列表={age_list} "
                        f"抽样分数=[{', '.join(f'{ps[i]:.3f}' for i in idx_e[:3])}...]"
                    )
            else:
                elite_formulas = seqs_new[:n_elite].tolist()

            lp_elite, ent_elite, v_elite = [], [], []
            if elite_formulas:
                ne     = len(elite_formulas)
                inp_e  = torch.zeros((ne, 1), dtype=torch.long,
                                     device=ModelConfig.DEVICE)
                sd_e   = [0] * ne
                prev_tokens_elite: list[int | None] = [None] * ne
                infected_chain_elite: list[int] = [0] * ne
                tok_e_t = torch.tensor(elite_formulas, dtype=torch.long,
                                       device=ModelConfig.DEVICE)
                for si in range(ModelConfig.MAX_FORMULA_LEN):
                    lg_e, val_e, _ = self.model(inp_e)
                    if val_e is not None:
                        v_elite.append(val_e)
                    lg_e = self.sampler.apply_mask_to_logits(
                        lg_e, sd_e, si, ModelConfig.MAX_FORMULA_LEN,
                        prev_tokens=prev_tokens_elite,
                        infected_chain_lens=infected_chain_elite
                    )
                    d_e  = Categorical(logits=lg_e)
                    tk   = tok_e_t[:, si]
                    lp_elite.append(d_e.log_prob(tk))
                    ent_elite.append(d_e.entropy())
                    inp_e = torch.cat([inp_e, tk.unsqueeze(1)], dim=1)
                    for b in range(ne):
                        sd_e[b] += self.sampler.delta[tk[b].item()]
                        prev_tokens_elite[b] = tk[b].item()
                        infected_chain_elite[b] = self.sampler.update_infection(
                        tk[b].item(), infected_chain_elite[b])


            # ── Part C: Evaluate all formulas (并行评估) ────────────────
            all_fmls = seqs_new.tolist() + elite_formulas
            tot      = len(all_fmls)
            rewards    = torch.zeros(tot, device=ModelConfig.DEVICE)
            val_scores = torch.zeros(tot, device=ModelConfig.DEVICE)

            ok_cnt = none_cnt = const_cnt = 0
            step_max_val = -float('inf');  step_best_f = None
            bic, bis, bsor = [], [], []

            # factor_pool 快照：所有 worker 看到同一份只读视图
            factor_pool_snapshot = list(self.factor_pool)

            # ── 批量 VM 评估（BATCHED_EVAL_ENABLED，E2 可行性结论落地）──
            # 一次 execute_batched 产出全部 [1,T] 结果（按位置骨架分组、
            # idx 回填），再逐条走相同 fold 评分。N>1/异常形状时
            # execute_batched 内部自动退化为逐条，语义不变。
            # 线程池路径（PARALLEL_EVAL=ON）与批量互斥：有池时走池。
            use_batched = (bool(getattr(ModelConfig, "BATCHED_EVAL_ENABLED", False))
                           and not (self._eval_pool is not None
                                    and self._eval_workers > 1 and tot > 1))
            if use_batched:
                _vm_res_list = self.vm.execute_batched(all_fmls, feat)
                results = [
                    self._eval_formula_task(
                        i, fml, feat, t_ret,
                        folds, use_wf, factor_pool_snapshot,
                        precomputed_res=_vm_res_list[i],
                    )
                    for i, fml in enumerate(all_fmls)
                ]
            elif self._eval_pool is not None and self._eval_workers > 1 and tot > 1:
                # 并行提交所有公式评估任务
                from concurrent.futures import ThreadPoolExecutor
                futures = [
                    self._eval_pool.submit(
                        self._eval_formula_task, i, fml, feat, t_ret,
                        folds, use_wf, factor_pool_snapshot,
                    )
                    for i, fml in enumerate(all_fmls)
                ]
                results_by_idx: dict[int, dict] = {}
                for fut in futures:
                    r = fut.result()
                    results_by_idx[r['idx']] = r
                results = [results_by_idx[i] for i in range(tot)]
            else:
                # 串行回退（BATCHED_EVAL_ENABLED=False 或特征形状非 [1,C,T]）
                results = [
                    self._eval_formula_task(
                        i, fml, feat, t_ret,
                        folds, use_wf, factor_pool_snapshot,
                    )
                    for i, fml in enumerate(all_fmls)
                ]

            # ── 串行后处理：写入 rewards/val_scores，更新冠军/池 ─────────
            for r in results:
                i = r['idx']
                status = r.get('status', 'error')
                rewards[i] = r['reward']
                val_scores[i] = r['val_score']

                if status == 'none':
                    none_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue
                if status == 'const':
                    const_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue
                if status == 'error':
                    none_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue

                ok_cnt += 1
                bic.append(r['ic_full']); bis.append(r['ic_stab']); bsor.append(r['val_score'])
                fml = r['fml']
                res = r['res']
                ic_i = r.get('ic_i', 0.0)
                final_val = r['val_score']

                if final_val > step_max_val:
                    step_max_val = final_val; step_best_f = fml

                # P0：val 只入围——维护 top-K finalist 表（真实回测在训练结束后选冠军）
                # vol 覆盖闸门（regime 诊断落地）：候选须在低/中/高 vol 三段均
                # 不显著为负才可更新 best / 进入围表。拦“只有高 vol 能活”的
                # 公式族，与公式族无关；只拦选优，不改 reward。
                vol_ok = r.get('vol_ok', True)
                if vol_ok:
                    self._update_finalists(final_val, fml, step)
                elif final_val > self.best_score:
                    tqdm.write(
                        f"[vol覆盖跳过 @ 第{step}步] 验证={final_val:.3f} IC={ic_i:.4f}"
                        f" | vol 三段覆盖不足，不更新最优/不入围\n"
                        f"    {fml}   {self._decode_formula(fml)}"
                    )

                if vol_ok and final_val > self.best_score:
                    # OOS 泛化门控：val_score / train_score < 0.5 说明过拟合
                    train_val = r['reward']
                    if train_val > 0.5 and final_val < train_val * 0.5:
                        tqdm.write(
                            f"[过拟合跳过 @ 第{step}步] 验证={final_val:.3f} "
                            f"训练={train_val:.3f} 比值={final_val/train_val:.2f} | 样本外表现过差"
                        )
                        pass
                    else:
                        pos_check = compute_target_positions_stateless(res)
                        exposure = pos_check.abs().mean().item()
                        if exposure < 0.05:
                            tqdm.write(
                                f"[稀疏跳过 @ 第{step}步] 验证={final_val:.3f} "
                                f"IC={ic_i:.4f} 暴露度={exposure:.1%} | 仓位过稀疏，不更新最优"
                            )
                            pass
                        else:
                            old_best = self.best_score
                            self.best_score   = final_val
                            self.best_formula = fml
                            self._best_snapshot = copy.deepcopy(self.model.state_dict())
                            self._best_update_step = step
                            self._stagnation_steps = 0
                            self._update_factor_pool(final_val, res)
                            self._save_strategy_live()
                            tqdm.write(
                                f"[!] 新最优 @ 第{step}步: 验证={final_val:.3f} "
                                f"(原 {old_best:.3f}，+{final_val-old_best:.3f}) "
                                f"IC={ic_i:.4f} 暴露度={exposure:.1%} | "
                                f"{fml}\n    {self._decode_formula(fml)}"
                            )
                self._update_elite_pool(final_val, fml, step)


            # ── Part D: REINFORCE gradient update ────────────────────
            # Fix 3: EMA baseline 替代 batch mean，避免全负 batch 的相对优选问题
            batch_mean = rewards.mean().item()
            batch_std  = rewards.std().clamp(min=0.1)
            if ModelConfig.REWARD_EMA_BASELINE and self._reward_ema_step >= ModelConfig.REWARD_EMA_WARMUP:
                baseline = self._reward_ema
                adv = (rewards - baseline) / (batch_std + 1e-5)
            else:
                adv = (rewards - batch_mean) / (batch_std + 1e-5)
            # 更新 EMA
            if self._reward_ema is None:
                self._reward_ema = batch_mean
            else:
                self._reward_ema = ModelConfig.REWARD_EMA_DECAY * self._reward_ema + (1.0 - ModelConfig.REWARD_EMA_DECAY) * batch_mean
            self._reward_ema_step += 1
            adv_new   = adv[:n_new]
            adv_elite = adv[n_new:]

            # ── 实验分支 ACC_CRITIC_GAE：critic 价值头 + GAE(γ,λ) 逐 token 优势 ──
            # 默认关闭 → 走上方经典 EMA-baseline REINFORCE 路径，行为不变。
            critic_on = (getattr(ModelConfig, "ACC_CRITIC_GAE", False)
                         and bool(v_new) and len(v_new) == len(lp_new))
            loss_critic = torch.zeros(1, device=ModelConfig.DEVICE)
            if critic_on:
                gamma = float(getattr(ModelConfig, "ACC_CRITIC_GAMMA", 1.0))
                lam   = float(getattr(ModelConfig, "ACC_CRITIC_LAMBDA", 1.0))

                def _gae_adv(vs: list[torch.Tensor], r_term: torch.Tensor):
                    """逐 token GAE 优势。vs[t]=[B] 是采样 token t 前的状态价值，
                    终止价值=0；r_term=[B] 是该条公式的最终奖励。"""
                    T = len(vs)
                    with torch.no_grad():
                        vd = [v.detach() for v in vs]
                    advs: list[torch.Tensor] = []
                    acc: torch.Tensor | None = None
                    for t in range(T - 1, -1, -1):
                        r_t  = r_term if t == T - 1 else torch.zeros_like(vd[t])
                        v_n  = vd[t + 1] if t + 1 < T else torch.zeros_like(vd[t])
                        delta = r_t + gamma * v_n - vd[t]
                        acc = delta + gamma * lam * (
                            acc if acc is not None else torch.zeros_like(delta)
                        )
                        advs.append(acc)
                    advs.reverse()
                    return advs

                r_new = rewards[:n_new]
                adv_new_t = _gae_adv(v_new, r_new)
                policy_loss = torch.zeros(1, device=ModelConfig.DEVICE)
                for ti in range(len(lp_new)):
                    policy_loss += (-lp_new[ti] * adv_new_t[ti] / batch_std).mean()
                if (lp_elite and v_elite and len(v_elite) == len(lp_elite)
                        and rewards[n_new:].shape[0] == v_elite[0].shape[0]):
                    r_el = rewards[n_new:]
                    adv_elite_t = _gae_adv(v_elite, r_el)
                    for ti in range(len(lp_elite)):
                        policy_loss += (
                            -lp_elite[ti] * adv_elite_t[ti] / batch_std
                            * ModelConfig.ELITE_REWARD_SCALE
                        ).mean()
                # critic 目标：位置 t 的折扣回报 = γ^(T-1-t) × r_term
                # （终局奖励在最后一步 t=T-1 才到达；γ=1 时退化为原恒定 r_term
                #   目标，与 E1 的 γ=λ=1 版逐位置 baseline 完全一致，无回归）
                vstack = torch.stack(v_new, dim=1)          # [B, T]
                _Tt = vstack.shape[1]
                _disc_w = torch.tensor(
                    [gamma ** (_Tt - 1 - ti) for ti in range(_Tt)],
                    device=vstack.device, dtype=vstack.dtype,
                ).view(1, _Tt)
                tgt = r_new.unsqueeze(1) * _disc_w          # [B, T]
                loss_critic = ((vstack - tgt) ** 2).mean()
                adv_new_t_mag = sum(float(a.abs().mean()) for a in adv_new_t) / len(adv_new_t)
                self.training_history.setdefault("critic_loss", []).append(
                    float(loss_critic.detach()))
                self.training_history.setdefault("adv_mag", []).append(adv_new_t_mag)
            else:
                policy_loss = torch.zeros(1, device=ModelConfig.DEVICE)
                for ti in range(len(lp_new)):
                    policy_loss += (-lp_new[ti] * adv_new).mean()
                if lp_elite and adv_elite.shape[0] > 0:
                    for ti in range(len(lp_elite)):
                        lpe = lp_elite[ti]
                        if lpe.shape[0] == adv_elite.shape[0]:
                            policy_loss += (-lpe * adv_elite
                                            * ModelConfig.ELITE_REWARD_SCALE).mean()

            if ent_new:
                mean_ent_new = torch.stack(ent_new).mean()
            else:
                mean_ent_new = torch.zeros(1, device=ModelConfig.DEVICE)
            if ent_elite:
                mean_ent_elite = torch.stack(ent_elite).mean()
                mean_ent = (
                    mean_ent_new * n_new + mean_ent_elite * n_elite
                ) / (n_new + n_elite)
            else:
                mean_ent = mean_ent_new
            ent_val   = mean_ent.item()
            ent_coeff = ModelConfig.ENTROPY_COEFF_MAX / (
                (1.0 + ent_val) ** ModelConfig.ENTROPY_COEFF_POWER
            )
            # Fix 1: 熵下限惩罚——当 H < threshold 时加入固定惩罚，确保探索压力不归零
            ent_floor_loss = torch.zeros(1, device=ModelConfig.DEVICE)
            if ModelConfig.ENTROPY_FLOOR and ent_val < ModelConfig.ENTROPY_FLOOR_THRESH:
                floor_gap = ModelConfig.ENTROPY_FLOOR_THRESH - ent_val
                ent_floor_loss = ModelConfig.ENTROPY_FLOOR_LAMBDA * torch.tensor(
                    floor_gap, device=ModelConfig.DEVICE, dtype=mean_ent.dtype
                )
            loss = policy_loss - ent_coeff * mean_ent + ent_floor_loss
            if critic_on:
                loss = loss + float(getattr(ModelConfig, "ACC_CRITIC_LOSS_W", 1.0)) * loss_critic

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            if self.use_lord:
                self.lord_opt.step()

            # ── Part D2: 分布细化指标 ────────────────────────────────
            dst = self._distribution_stats(prev_init_dist)
            prev_init_dist = dst['dist']
            with torch.no_grad():
                uniq_tokens = seqs_new.unique().numel()
                uniq_fmls   = torch.unique(seqs_new, dim=0).shape[0]
                fml_div     = uniq_fmls / max(1, n_new)

            # ── Part E: Logging & history & checkpoint ───────────────
            avg_rew = rewards.mean().item()
            avg_val = val_scores.mean().item()
            bim  = sum(bic)  / len(bic)  if bic  else 0.0
            bis_ = sum(bis)  / len(bis)  if bis  else 0.0
            bsor_= sum(bsor) / len(bsor) if bsor else 0.0

            self._stagnation_steps = step - self._best_update_step
            tqdm.write(
                f"[{step+1}/{end_step}] "
                f"新公式={n_new} 精英={n_elite} | "
                f"有效={ok_cnt} 无效={none_cnt} 常数={const_cnt} | "
                f"奖励={avg_rew:.3f} 验证={avg_val:.3f} | "
                f"IC={bim:.4f} | 熵={ent_val:.3f}(系数={ent_coeff:.3f}) | "
                f"最优={self.best_score:.3f} 停滞={self._stagnation_steps} "
                f"精英池={len(self._elite_pool)} 重启={self._restart_count}"
            )
            tqdm.write(
                f"   分布: 初始熵={dst['entropy']:.3f} KL均匀={dst['kl_uniform']:.3f} "
                f"KL上步={dst['kl_prev']:.4f} 最高概率={dst['top1_prob']:.3f} "
                f"前五概率={dst['top5_prob']:.3f} 有效词汇={dst['eff_vocab']:.2f} "
                f"标准差={dst['prob_std']:.4f} | "
                f"本批: 唯一符号={uniq_tokens}/{FORMULA_VOCAB.size} "
                f"唯一公式={uniq_fmls}/{n_new} 多样性={fml_div:.2f}"
            )
            pbar.set_postfix({
                '验证': f"{avg_val:.3f}", '最优': f"{self.best_score:.3f}",
                '熵':   f"{ent_val:.2f}", 'IC':   f"{bim:.4f}",
                '停滞': f"{self._stagnation_steps}",
                '初始熵':  f"{dst['entropy']:.2f}",
                'KL上步': f"{dst['kl_prev']:.3f}",
            })

            if self.use_lord and step % 10 == 0:
                sr = self.rank_monitor.compute()
                self.training_history['stable_rank'].append(sr)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_rew)
            self.training_history['val_score'].append(avg_val)
            self.training_history['best_score'].append(self.best_score)
            self.training_history.setdefault('entropy', []).append(ent_val)
            self.training_history.setdefault('ic_mean', []).append(bim)
            self.training_history.setdefault('ic_stability', []).append(bis_)
            self.training_history.setdefault('sortino', []).append(bsor_)
            self.training_history.setdefault('elite_pool_size', []).append(
                len(self._elite_pool))
            self.training_history.setdefault('init_entropy', []).append(dst['entropy'])
            self.training_history.setdefault('kl_uniform', []).append(dst['kl_uniform'])
            self.training_history.setdefault('kl_prev', []).append(dst['kl_prev'])
            self.training_history.setdefault('top1_prob', []).append(dst['top1_prob'])
            self.training_history.setdefault('eff_vocab', []).append(dst['eff_vocab'])
            self.training_history.setdefault('batch_uniq_tokens', []).append(uniq_tokens)
            self.training_history.setdefault('batch_uniq_fmls', []).append(uniq_fmls)
            self.training_history.setdefault('batch_fml_div', []).append(fml_div)

            # P0 加固：训练中途只写 *.live.json 侧车，不碰部署路径（防覆盖旧冠军）
            if self.best_formula is not None:
                self._save_strategy_live()

            # 实时曲线按 HISTORY_LIVE_EVERY_STEPS 节流整份重写；run 终点强制落盘
            self._save_training_history_live(force=(step + 1) == end_step)

            if (step + 1) % 20 == 0 or (step + 1) == end_step:
                ckpt = self.save_checkpoint(step + 1)
                tqdm.write(f"[检查点] → {ckpt} (最优={self.best_score:.3f})")

            # ── Part F: Migration hook（多岛训练时交换精英）────────────
            if migration_hook is not None and (step + 1) % ModelConfig.MIGRATION_INTERVAL == 0:
                tqdm.write(f"[迁移钩子 @ 第{step+1}步] 调用已注册钩子")
                migration_hook(self, step + 1)

            # ── Part G: Entropy collapse detection & restart ─────────
            if ent_val < ModelConfig.ENTROPY_COLLAPSE_THRESH:
                low_entropy_streak += 1
            else:
                low_entropy_streak  = 0

            if low_entropy_streak >= ModelConfig.ENTROPY_COLLAPSE_STEPS:
                # ── 自适应噪声：根据 stagnation 调整 ─────────────────────
                self._stagnation_steps = step - self._best_update_step
                stagnation_ratio = self._stagnation_steps / max(1, ModelConfig.STAGNATION_WINDOW)
                base_noise = ModelConfig.RESTART_NOISE
                if ModelConfig.ADAPTIVE_NOISE:
                    raw_noise = base_noise + ModelConfig.NOISE_BOOST_FACTOR * 0.1 * min(stagnation_ratio, 3.0)
                    noise = max(ModelConfig.NOISE_MIN, min(ModelConfig.NOISE_MAX, raw_noise))
                else:
                    noise = base_noise

                max_r = ModelConfig.MAX_RESTARTS
                if self._restart_count < max_r:
                    # ── 重启升级（E1 critic 吸引子锁死诊断落地）──────────
                    # 上次重启后 best 无任何刷新 → 说明“部分层重启”没逃出
                    # best_snapshot 吸引子，本次强制完全重置（不只依赖熵<0.3
                    # 或 FULL_RESET_EVERY 固定周期——critic 臂 81 步封顶后两次
                    # 部分重启(熵0.337/0.463)都没回升的根因就是缺一次 full reset）。
                    improved_since_prev = (
                        self._last_restart_step < 0
                        or self._best_update_step > self._last_restart_step
                    )
                    escalate_full = bool(
                        ModelConfig.RESTART_ESCALATE_FULL
                    ) and self._restart_count > 0 and not improved_since_prev
                    self._restart_count  += 1
                    low_entropy_streak    = 0
                    self._last_restart_step = step

                    # Fix 2: 每 N 次重启做一次完全随机初始化，逃离 best_snapshot 吸引子
                    # 深度坍塌 (H < 0.3) 时强制 full reset，不给 best_snapshot 恢复的机会
                    do_full_reset = (
                        self._restart_count % ModelConfig.FULL_RESET_EVERY == 0
                        or ent_val < 0.3
                        or escalate_full
                    )

                    if do_full_reset:
                        # 完全重新初始化模型参数
                        for layer in self.model.modules():
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                        _why = ("升级:上次重启后无刷新" if escalate_full else
                                ("固定周期 full reset" if
                                 self._restart_count % ModelConfig.FULL_RESET_EVERY == 0
                                 else "深度坍塌 H<0.3"))
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式=完全重置（{_why}） "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f}"
                        )
                    elif self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                        with torch.no_grad():
                            if ModelConfig.PARTIAL_RESET:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if any(k in nm for k in ModelConfig.PARTIAL_RESET_LAYERS):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                            else:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if 'ffn' in nm or 'attention' in nm or nm.startswith('blocks'):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式={'部分层' if ModelConfig.PARTIAL_RESET else 'FFN/注意力'} "
                            f"噪声={noise:.4f}(基准={base_noise:.3f}，比率={stagnation_ratio:.2f}) "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f} "
                            f"扰动层数={len(perturbed_layers)}"
                        )
                    else:
                        with torch.no_grad():
                            for p in self.model.parameters():
                                p.add_(torch.randn_like(p) * noise)
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式=全参数 "
                            f"噪声={noise:.4f}(基准={base_noise:.3f}，比率={stagnation_ratio:.2f}) "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f} | 无最优快照"
                        )
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                else:
                    # 训练时间不敏感：超过重启上限后不再 Early Stop 终止，
                    # 改为「全参数强扰动 + 重置流计数」继续探索，直到跑满 TRAIN_STEPS。
                    # 从 best_snapshot 恢复（若有）以保住已发现的最优结构，再加大扰动。
                    low_entropy_streak = 0
                    hard_noise = min(ModelConfig.NOISE_MAX, noise * 2.0)
                    if self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                    with torch.no_grad():
                        for p in self.model.parameters():
                            p.add_(torch.randn_like(p) * hard_noise)
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                    tqdm.write(
                        f"[强重启 @ 第{step}步] 已达最大重启次数={max_r} "
                        f"熵={ent_val:.3f} 强噪声={hard_noise:.4f} "
                        f"继续训练，不提前停止"
                    )

        # ── End of training ──────────────────────────────────────────
        # 仅当跑满最终步时才保存最终 strategy 和历史
        if end_step == ModelConfig.TRAIN_STEPS:
            self._verify_holdout()
            # P0 冠军闸门：val 只入围，holdout 真实回测选冠军 + 多重比较/单次消费校验。
            # 失败时恢复旧冠军并记录 champion_history；成功才部署新冠军。
            self._finalize_champion()

            sym_tag = f"[{self.target_symbol}] " if self.target_symbol else ""
            self.training_history.pop('_low_entropy_streak', None)
            # 训练范围溯源：子集文件旁有 train_range.json 时并入历史 JSON
            try:
                dm_file = getattr(getattr(self, "data_manager", None), "file_path", None)
                if dm_file:
                    from data_pipeline.train_sampler import read_train_range

                    _tr = read_train_range(str(dm_file))
                    if _tr:
                        self.training_history["train_range"] = _tr
            except Exception:  # noqa: BLE001
                pass
            hist_path = (
                f"training_history_{self.target_symbol}{self._run_tag_suffix()}.json"
                if self.target_symbol else "training_history.json"
            )
            # P1-3: 原子写入
            tmp_hist = hist_path + ".tmp"
            with open(tmp_hist, "w", encoding="utf-8") as fp:
                json.dump(self.training_history, fp)
            os.replace(tmp_hist, hist_path)

            print(f"\n[完成] {sym_tag}训练结束！")
            print(f"  最优验证分数 : {self.best_score:.4f}")
            print(f"  最优公式令牌 : {self.best_formula}")
            print(f"  可读公式     : {self._decode_formula(self.best_formula)}")
            print(f"  精英池大小   : {len(self._elite_pool)}")
            print(f"  精英衰减     : 启用={ModelConfig.ELITE_DECAY}，半衰期={ModelConfig.ELITE_DECAY_HALF_LIFE}")
            print(f"  自适应噪声   : 启用={ModelConfig.ADAPTIVE_NOISE}，范围=[{ModelConfig.NOISE_MIN}, {ModelConfig.NOISE_MAX}]")
            print(f"  部分层重置   : 启用={ModelConfig.PARTIAL_RESET}，层={ModelConfig.PARTIAL_RESET_LAYERS}")
            print(f"  重启次数     : {self._restart_count}")
            oc = self.champion_outcome or {}
            action = oc.get("action", "no_champion")
            if action == "deploy":
                print(f"  [冠军闸门] 部署新冠军 ✓（holdout 真实回测选优 + 统计闸门通过）")
                print(f"  策略已保存   : {oc.get('save_path')}")
            elif action == "reject_restore":
                print(f"  [冠军闸门] 拒绝部署并恢复旧冠军 ✗")
                for r in (oc.get("reasons") or []):
                    print(f"    - {r}")
                print(f"  策略保留     : {oc.get('save_path')}（原冠军 formula={oc.get('old_formula')}）")
            elif action == "suppressed":
                print(f"  [多seed] 本次部署被抑制（由主流程在全部 seed 通过后统一部署）")
            else:
                print("  未发现有效公式，策略未保存")


    # ── 实时保存最优公式（防进程意外退出丢失）────────────────────────────────
    def _save_training_history_live(self, force: bool = False) -> None:
        """周期性写入训练曲线 JSON，供 Web UI 实时展示。

        P1-3 修复：原子写入（tmp + os.replace），避免 Ctrl+C / OOM 打断写入
        导致 history 文件损坏。异常打印告警而非静默吞掉。

        IO 节流（2026-09-05）：history 随步数线性增长，若每步整份重写，9000 步
        长跑累计 ~12GB 磁盘写。默认每 HISTORY_LIVE_EVERY_STEPS（50）步写一次，
        首步恒写、run 终点与终局判定（holdout）force=True 恒写。
        """
        if not self.target_symbol:
            return
        if not force:
            every = int(getattr(ModelConfig, "HISTORY_LIVE_EVERY_STEPS", 50) or 1)
            steps = self.training_history.get("step")
            n = len(steps) if steps else 0
            # 无步数信息 → 不节流（写一次保险）；否则 1 或整倍数步才写
            if n and n != 1 and n % every != 0:
                return
        try:
            hist_path = f"training_history_{self.target_symbol}{self._run_tag_suffix()}.json"
            payload = {
                k: v for k, v in self.training_history.items()
                if k != "_low_entropy_streak"
            }
            # 原子写入：先写 tmp，再 os.replace 覆盖（POSIX/Windows 均原子）
            tmp_path = hist_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp)
            os.replace(tmp_path, hist_path)
        except Exception as exc:  # noqa: BLE001
            # 静默吞掉会掩盖磁盘满/权限错误，至少打印告警
            try:
                tqdm.write(f"[警告] 训练历史保存失败: {exc}")
            except Exception:
                pass

    def _save_strategy_live(self) -> None:
        """每次 best_formula 更新时立即保存 strategy json（写到 *.live.json 侧车）。

        P0 加固：绝不写部署路径 strategies/best_{symbol}.json——训练中途的
        best-so-far 未经冠军闸门验证，写侧车由 UI 标记「训练中·未验证」；
        进程被杀/被停时旧冠军文件保持原样，闸门通过后才覆盖部署。
        P1-3 修复：原子写入（tmp + os.replace），避免写入中途被打断导致
        strategy JSON 截断损坏。异常打印告警。

        实验 run（run_tag 非空）跳过实时侧车，避免污染正式部署路径。
        """
        if getattr(self, "run_tag", ""):
            return
        if self.best_formula is None:
            return
        try:
            from .vocab import VOCAB_VERSION
            save_path = _live_strategy_file_for_symbol(self.target_symbol)
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            existing: dict = {}
            p = pathlib.Path(save_path)
            if p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        existing = raw
                except Exception:
                    existing = {}

            strategy_data = {
                "vocab_version": VOCAB_VERSION,
                "symbol": self.target_symbol,
                "formula": self.best_formula,
                "best_score": self.best_score,
                "formula_decoded": self._decode_formula(self.best_formula),
            }
            # 保留训练元数据（data_file/timeframe/... 与 data_source 溯源标记），
            # 避免 live 保存把这些字段冲掉
            strategy_data = _merge_train_metadata(self, {**existing, **strategy_data})

            # 原子写入：先写 tmp，再 os.replace 覆盖
            tmp_path = save_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(strategy_data, fp, indent=2, ensure_ascii=False)
            os.replace(tmp_path, save_path)
        except Exception as exc:  # noqa: BLE001
            # 静默吞掉会让用户误以为策略已保存，实则没有
            try:
                tqdm.write(f"[警告] 策略保存失败: {exc}")
            except Exception:
                pass

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            sym_tag = f"_{self.target_symbol}" if self.target_symbol else ""
            tag = self._run_tag_suffix()
            path = str(_CHECKPOINT_DIR / f"ckpt{sym_tag}{tag}_step_{step:04d}.pt")
        ckpt = {
            "step":                 step,
            "vocab_version":        VOCAB_VERSION,   # task 12.2: 版本校验所需
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_score":           self.best_score,
            "best_formula":         self.best_formula,
            "best_snapshot":        self._best_snapshot,
            "factor_pool":          self.factor_pool,
            "factor_pool_counter":  self._factor_pool_counter,
            "elite_pool":           self._elite_pool,
            "elite_counter":        self._elite_counter,
            "restart_count":        self._restart_count,
            "training_history":     {
                k: v for k, v in self.training_history.items()
                if k != '_low_entropy_streak'
            },
        }
        # P1-3: 原子写入（tmp + os.replace），避免 Ctrl+C / OOM 打断导致
        # checkpoint 文件截断损坏——既丢新最优也丢旧最优
        tmp_path = path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE)

        # ── Task 12.2：版本校验（R3.7）──────────────────────────────────────
        # 从 checkpoint 读取 vocab_version；若字段缺失（旧版 checkpoint），视为
        # 版本不匹配并抛错——拒绝加载、不消费任何 token。
        artifact_version = ckpt.get("vocab_version")
        if artifact_version is None:
            raise VocabVersionMismatchError(
                f"checkpoint '{path}' 不含 vocab_version 字段（旧版产物），"
                f"当前词表版本 {FORMULA_VOCAB.version!r}；需重新训练后加载"
            )
        # verify() 版本不匹配时抛 VocabVersionMismatchError，拒绝加载
        FORMULA_VOCAB.verify(artifact_version)
        # ── 版本校验通过，继续加载 ────────────────────────────────────────

        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_score          = ckpt.get("best_score",  -float('inf'))
        self.best_formula        = ckpt.get("best_formula", None)
        self._best_snapshot      = ckpt.get("best_snapshot", None)
        self.factor_pool         = ckpt.get("factor_pool", [])
        self._factor_pool_counter = ckpt.get("factor_pool_counter", 0)
        self._elite_pool         = ckpt.get("elite_pool", [])
        self._elite_counter      = ckpt.get("elite_counter", 0)
        self._restart_count      = ckpt.get("restart_count", 0)
        for k, v in ckpt.get("training_history", {}).items():
            self.training_history[k] = v

        # 清理 elite pool 中的重复条目（保留各公式的最高分版本）
        self._elite_pool = self._dedup_elite_pool(self._elite_pool)

        completed = ckpt.get("step", 0)
        tqdm.write(f"[检查点] 已从 {path} 恢复。"
                   f" 当前步={completed}  最优={self.best_score:.4f}"
                   f"  精英池={len(self._elite_pool)}（去重后）")
        return completed

    # ── Decode formula tokens to readable string ──────────────────────────────

    def _decode_formula(self, tokens: list[int] | None) -> str:
        if tokens is None:
            return "无"
        from .vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}"
                           for t in tokens)
