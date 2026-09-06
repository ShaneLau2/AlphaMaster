"""M1: MarketEnv —— 多市场训练/评估的市场环境。

每个 MarketEnv 包装一个 {sym}_{tf}.parquet:
  - 复用 AlphaEngine 的评估原语(_eval_formula_task / _rigorous_holdout_pnl /
    _vol_pnl_coverage / _init_vol_tiers),与单市场训练同口径、零行为分叉;
  - 自带训练窗口(feat/t_ret 切到 holdout 前)、walk-forward folds、ppy、
    格级 vol 覆盖缓存;min-bars 门槛决定 trainable(不足 → eval-only, 可作 OOS 腿);
  - eval_formulas() 支持批量 VM(execute_batched)+ 逐条 fold 评分(等价验证过)。

多市场训练器(MultiMarketEngine, model_core/multimarket.py)每步对抽中市场
调用 env.eval_formulas(), 聚合出共享奖励。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core.backtest import estimate_periods_per_year  # noqa: E402
from model_core.config import ModelConfig  # noqa: E402
from model_core.engine import (  # noqa: E402
    AlphaEngine,
    _build_walk_forward_folds,
    _holdout_bars_for,
)


class MarketEnv:
    def __init__(self, spec: dict, min_bars: int | None = None,
                 seed: int = 0) -> None:
        self.spec = dict(spec)
        self.symbol = str(spec["symbol"])
        self.timeframe = str(spec.get("timeframe", "H1"))
        self.group = str(spec.get("group", "default"))
        self.role = str(spec.get("role", "train"))
        self.file = str(spec["file"])
        self.name = f"{self.symbol}_{self.timeframe}"

        from data_pipeline.parquet_manager import ParquetDataManager
        self.mgr = ParquetDataManager(self.file)
        self.mgr.load()

        T_full = int(self.mgr.target_ret.shape[1])
        self.bars_full = T_full
        self.holdout_bars = int(_holdout_bars_for(T_full))
        T = T_full - self.holdout_bars
        self.bars_train = T

        self.min_bars = int(min_bars) if min_bars is not None else int(
            ModelConfig.MARKET_MIN_BARS if hasattr(ModelConfig, "MARKET_MIN_BARS")
            else 8000)
        self.trainable = T >= self.min_bars

        dev = ModelConfig.DEVICE
        self.feat_full = self.mgr.feat_tensor.to(dev)
        self.t_ret_full = self.mgr.target_ret.to(dev)
        self.feat = self.feat_full[:, :, :T]
        self.t_ret = self.t_ret_full[:, :T]

        # 轻量 scratch engine(只读评估, 不训练): 与单市场引擎同口径
        self.engine = AlphaEngine(data_manager=self.mgr,
                                  target_symbol=self.symbol, seed=seed)
        try:
            ppy = estimate_periods_per_year(self.mgr.raw_dict["time"].flatten())
            if ppy != self.engine.bt.periods_per_year:
                self.engine.bt.periods_per_year = ppy
        except Exception:  # noqa: BLE001 估计失败保留默认
            pass

        self.folds = _build_walk_forward_folds(
            T, self.engine.n_folds, gap=getattr(ModelConfig, "WF_GAP", 20))
        self.use_wf = len(self.folds) > 1 and not (
            self.folds[0]["train_start"] == 0 and self.folds[0]["train_end"] == T)

        # 格级 vol 覆盖缓存(与选优闸门同口径)
        self.engine._init_vol_tiers(self.feat, self.t_ret)
        self._vol_meta = (self.engine._vol_grid_meta
                          if getattr(ModelConfig, "VOL_COVERAGE_GRID", True)
                          else self.engine._vol_tier_meta)

    # ── 评估 ──────────────────────────────────────────────────────────────
    def eval_formulas(self, fmls: list[list[int]], batched: bool = True,
                      collect_res: bool = False) -> dict:
        """批量评估公式 → {rewards, vals, vol_ok, status, ics, res_list} (与引擎 Part C 同口径)。
        collect_res=True 时另返回每条公式的原始因子张量(多市场冠军晋升的暴露度检查用)。"""
        tot = len(fmls)
        rewards = torch.zeros(tot, device=ModelConfig.DEVICE)
        vals = torch.zeros(tot, device=ModelConfig.DEVICE)
        vol_ok = torch.ones(tot, dtype=torch.bool, device=ModelConfig.DEVICE)
        status = ["none"] * tot
        ic_full = torch.zeros(tot, device=ModelConfig.DEVICE)
        ic_stab = torch.zeros(tot, device=ModelConfig.DEVICE)

        use_batched = bool(batched) and bool(
            getattr(ModelConfig, "BATCHED_EVAL_ENABLED", True))
        vm_res = None
        if use_batched:
            try:
                vm_res = self.engine.vm.execute_batched(fmls, self.feat)
            except Exception:  # noqa: BLE001 批量退化到逐条
                vm_res = None
        # 覆盖诊断（格级口径汇总，供搜索健康监测；网格禁用/无 vol_info → None）
        cov_n = cov_ok = cov_partial = 0
        cov_eff_counts: list[int] = []
        cov_bars: list[float] = []

        res_list: list = []
        results = []
        for i, fml in enumerate(fmls):
            kw = {}
            if vm_res is not None and vm_res[i] is not None:
                kw["precomputed_res"] = vm_res[i]
            try:
                r = self.engine._eval_formula_task(
                    i, fml, self.feat, self.t_ret, self.folds,
                    self.use_wf, [], **kw)
            except Exception as exc:  # noqa: BLE001 单条失败不影响其余
                r = {"idx": i, "status": "error",
                     "reward": -5.0, "val_score": -5.0, "fml": fml,
                     "error": f"{type(exc).__name__}: {exc}"}
            results.append(r)

        for r in results:
            i = r["idx"]
            status[i] = r.get("status", "none")
            rewards[i] = r.get("reward", -5.0)
            vals[i] = r.get("val_score", -5.0)
            vol_ok[i] = bool(r.get("vol_ok", True))
            ic_full[i] = r.get("ic_full", 0.0)
            ic_stab[i] = r.get("ic_stab", 0.0)
            if collect_res:
                res_list.append(r.get("res"))
            # 覆盖诊断：只汇总 grid 口径（tier/skip/关闭不计入）
            vi = r.get("vol_info")
            if vi and vi.get("grid"):
                cov_n += 1
                if bool(r.get("vol_ok", True)):
                    cov_ok += 1
                cells = vi.get("cells") or []
                eff = [c for c in cells if not c.get("skipped")]
                if len(eff) < len(cells):
                    cov_partial += 1
                cov_eff_counts.append(len(eff))
                cov_bars.extend(float(c.get("bars", 0.0)) for c in eff)
        if cov_n:
            cov_stats = {
                "mode": "grid",
                "n": cov_n,
                "n_ok": cov_ok,
                "coverage_rate": round(cov_ok / cov_n, 4),
                "cand_partial_grid": cov_partial,          # 有格因样本不足被跳过的候选数
                "eff_cells_median": round(statistics.median(cov_eff_counts), 1)
                if cov_eff_counts else None,                # 有效(有样本)格数中位
                "cell_bars_median": int(statistics.median(cov_bars))
                if cov_bars else None,                      # 参与判定的格样本量中位
                "cell_bars_min": int(min(cov_bars)) if cov_bars else None,
            }
        else:
            cov_stats = None
        return {"rewards": rewards, "vals": vals, "vol_ok": vol_ok,
                "status": status, "ic_full": ic_full, "ic_stab": ic_stab,
                "res_list": res_list, "cov_stats": cov_stats}

    def holdout(self, fml: list[int]) -> dict | None:
        """冠军在【本市场】预留尾窗的生产口径回测(与 P0.1 同函数)。"""
        s = self.bars_full - self.holdout_bars
        e = self.bars_full - 2
        return self.engine._rigorous_holdout_pnl(
            list(fml), self.feat_full, self.t_ret_full, s, e)

    def coverage(self, fml: list[int]) -> tuple[bool, dict]:
        """本市场 vol 覆盖(格级/段级, 与选优闸门同实现)。"""
        try:
            res = self.engine.vm.execute(list(fml), self.feat)
            if res is None or float(res.std()) < 1e-4:
                return True, {"skipped": "const"}
            return self.engine._vol_pnl_coverage(res, self.t_ret)
        except Exception as exc:  # noqa: BLE001 fail-open
            return True, {"error": f"{type(exc).__name__}: {exc}"}

    def __repr__(self) -> str:  # pragma: no cover
        return (f"MarketEnv({self.name} bars={self.bars_train}"
                f" trainable={self.trainable} group={self.group})")
