"""verify_vol_grid.py — 验证 vol×er 格级闸门(VOL_COVERAGE_GRID=True)。

用真实引擎路径(_init_vol_tiers/_vol_pnl_coverage)在【训练窗口】(train() 同款
holdout 切除)上,对历次实验冠军跑 段级 vs 格级 两档覆盖,回答:
  1) e4_bc 冠军的低vol×震荡出血格在训练窗口内是否仍显著(t < -1.645)?
  2) 格级是否拦下 e4_bc(段级放行),且不误伤 e3_direct8 等健康冠军?

用法: .venv/bin/python scripts/verify_vol_grid.py [parquet...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.parquet_manager import ParquetDataManager  # noqa: E402
from model_core.config import ModelConfig  # noqa: E402
from model_core.engine import AlphaEngine, _holdout_bars_for  # noqa: E402
from model_core.vm import StackVM  # noqa: E402

BTC = "data/training/BINANCE_BTCUSDT_H1.parquet"
ADA = "data/training/ADAUSDT_H1.parquet"

CHAMPS = {
    "e1_base": json.load(open("results/exp1_base_200_s42.json"))["best_formula"],
    "e1_critic": json.load(open("results/exp1_critic_200_s42.json"))["best_formula"],
    "e3_direct8": json.load(open("results/exp3_direct8.json"))["best_formula"],
    "e3_direct14": json.load(open("results/exp3_direct14.json"))["best_formula"],
    "e3_curriculum_final": json.load(
        open("results/exp3_curriculum.json"))["best_formula"],
    "e4_bc": json.load(open("results/exp4_bc.json"))["best_formula"],
    "e4_scratch": json.load(open("results/exp4_scratch.json"))["best_formula"],
}


def coverage(file: str, fml: list[int], grid: bool, train_window: bool):
    mgr = ParquetDataManager(file)
    mgr.load()
    T_full = int(mgr.target_ret.shape[1])
    if train_window:
        h = _holdout_bars_for(T_full)
        T = T_full - h
    else:
        h, T = 0, T_full
    feat = mgr.feat_tensor[:, :, :T]
    t_ret = mgr.target_ret[:, :T]
    eng = AlphaEngine(data_manager=mgr, target_symbol=mgr.symbol, seed=1)
    ModelConfig.VOL_COVERAGE_GRID = bool(grid)
    eng._init_vol_tiers(feat, t_ret)
    res = StackVM().execute(fml, feat)
    if res is None:
        return None, None, None
    ok, info = eng._vol_pnl_coverage(res, t_ret)
    return ok, info, (T_full, h, T)


def summarize(ok: bool, info: dict) -> str:
    if not info:
        return "∅"
    if info.get("grid"):
        parts = []
        worst = None
        for r in info.get("cells", []):
            if "t" not in r:
                continue
            if worst is None or r["t"] < worst["t"]:
                worst = r
            mark = "" if r["pass"] else "✗"
            if r["pass"] is False:
                parts.append(f"{r['cell']} t={r['t']}{mark}")
        base = f"grid 9格 → {'✅通过' if ok else '❌拦截'} | worst {worst}"
        return base + (f" (失血格: {', '.join(parts)})" if parts else "")
    worst = None
    for r in info.get("tiers", []):
        if "t" in r and (worst is None or r["t"] < worst["t"]):
            worst = r
    return f"段级3段 → {'✅通过' if ok else '❌拦截'} | worst {worst}"


def main() -> None:
    files = sys.argv[1:] or [BTC, ADA]
    rows = []
    for file in files:
        print("=" * 88)
        print(f"文件 {file}")
        print("=" * 88)
        for lbl, fml in CHAMPS.items():
            print(f"\n[{lbl}] tokens={fml}")
            for grid in (False, True):
                ok, info, dims = coverage(file, fml, grid, train_window=True)
                T_full, h, T = dims
                tag = "格级" if grid else "段级"
                s = summarize(ok, info) if info else "∅ (常量/非法)"
                print(f"   训练窗(T={T_full}-{h}={T}) {tag}: {s}")
            # 全窗口(含 holdout 尾部)格级 — 与 regime_health 口径对照
            ok, info, dims = coverage(file, fml, True, train_window=False)
            print(f"   全窗(含holdout)  格级: {summarize(ok, info)}")
            rows.append((lbl, file, ok, info))


if __name__ == "__main__":
    main()
