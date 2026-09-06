"""中性带正则（NEUTRAL_BAND_W）开/关短训对比实验。

同一数据 × 同一 seed × 同一 steps：只在 ModelConfig.NEUTRAL_BAND_W 上开/关。
对训出的公式在同一数据上评估：
  - 全序列 FLAT 占比（|tanh(f)| < 0.05 / 0.5 / 0.8）；
  - 样本外 holdout 窗口在 0.5 阈值下的 收益/夏普/回撤/FLAT 占比（生产口径）；
  - 引擎自带 holdout 闸门分（0.05 口径）与闸门通过与否。

验证问题：中性带奖励能否训出「真正会输出观望」的因子（FLAT 占比明显更高），
且不牺牲 holdout 分。结果写 results/neutral_band_experiment.{json,REPORT.md}。

用法:
  python scripts/neutral_band_experiment.py [--steps 300] [--seed 42] [--band-w 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_variant import run_variant  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from backtest_viz import BacktestEngine  # noqa: E402
from model_core.features import FeatureEngineer  # noqa: E402
from model_core.vm import StackVM  # noqa: E402

DEFAULT_DATA = "data/slices/tail_15000/BTCUSDT_M5.parquet"


def _factor_series(formula: list[int], feat: torch.Tensor) -> np.ndarray:
    vm = StackVM()
    with torch.no_grad():
        out = vm.execute([int(t) for t in formula], feat)
    if out is None or out.ndim != 2 or out.shape[1] == 0:
        raise RuntimeError("公式无有效输出")
    return out[0].cpu().numpy().astype(np.float64)


def _flat_share(factor: np.ndarray, thr: float) -> float:
    pos = np.tanh(factor)
    return float(np.mean(np.abs(pos) < thr))


def _holdout_at_threshold(raw: dict, feat: torch.Tensor, formula: list[int],
                          s: int, e: int, thr: float, sym: str) -> dict:
    """样本外窗口按阈值 0.5 的「连续口径生产指标」：BacktestEngine 自带 min_abs。"""
    raw_s = {k: v[:, s:e] for k, v in raw.items()}
    feat_s = feat[:, :, s:e]
    eng = BacktestEngine(formula=[int(t) for t in formula], cost_rate=0.0003,
                         max_position_pct=100.0, min_abs=thr)
    from run_backtest import calc_sharpe

    res = eng.run(raw_s, feat_s, [sym])[0]
    return {
        "threshold": thr,
        "total_return": round(float(res.total_return), 6),
        "sharpe": round(float(calc_sharpe(np.asarray(res.pnl, dtype=float),
                                           periods_per_year=105195)), 4),
        "max_drawdown": round(float(res.max_drawdown), 4),
        "n_trades": int(res.n_trades),
    }


def _evaluate(res: dict, data_file: str) -> dict:
    from data_pipeline.parquet_manager import ParquetDataManager

    pm = ParquetDataManager(data_file)
    pm.load()
    raw = pm.raw_dict
    sym = pm.symbol
    formula = res.get("best_formula") or []
    if not formula:
        return {"error": "无 best_formula"}
    feats = FeatureEngineer.compute_features(raw)
    factor = _factor_series(formula, feats)

    ho = res.get("holdout") or {}
    s, e = int(ho.get("start") or 0), int(ho.get("end") or 0)
    out = {
        "flat_share": {str(t): round(_flat_share(factor, t), 4)
                       for t in (0.05, 0.5, 0.8)},
        "factor_mean_abs": round(float(np.mean(np.abs(factor))), 4),
        "factor_std": round(float(np.std(factor)), 4),
    }
    if e > s:
        f_win = factor[s:e]
        out["holdout_flat_share"] = {str(t): round(_flat_share(f_win, t), 4)
                                     for t in (0.05, 0.5, 0.8)}
        out["holdout_at_05"] = _holdout_at_threshold(raw, feats, formula, s, e, 0.05, sym)
        out["holdout_at_08"] = _holdout_at_threshold(raw, feats, formula, s, e, 0.8, sym)
    out["engine_holdout_val"] = ho.get("val_score")
    out["engine_holdout_passed"] = bool(ho.get("passed"))
    out["engine_best_score"] = res.get("best_score")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default=DEFAULT_DATA)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--band-w", type=float, default=0.5, help="开档 NEUTRAL_BAND_W（关档=0.0）")
    ap.add_argument("--batch", type=int, default=64, help="加速用缩小 batch（两档一致，保证公平）")
    ap.add_argument("--out", default="results/neutral_band_experiment.json")
    args = ap.parse_args()

    from model_core.config import ModelConfig

    old_batch = ModelConfig.BATCH_SIZE
    ModelConfig.BATCH_SIZE = max(16, int(args.batch))
    rows: dict[str, dict] = {}
    t0 = time.time()
    try:
        for w in (0.0, args.band_w):
            tag = f"nb_{'on' if w > 0 else 'off'}_s{args.seed}"
            print(f"[nb] w={w} tag={tag} steps={args.steps} …", flush=True)
            ModelConfig.NEUTRAL_BAND_W = w
            res = run_variant(args.data_file, tag=tag, steps=args.steps, seed=args.seed)
            ev = _evaluate(res, args.data_file)
            ev["neutral_w"] = w
            ev["best_formula"] = res.get("best_formula")
            rows[tag] = ev
            print(f"[nb] {tag} done: flat@0.5={ev.get('flat_share', {}).get('0.5')} "
                  f"holdout_val={ev.get('engine_holdout_val')} "
                  f"passed={ev.get('engine_holdout_passed')}", flush=True)
    finally:
        ModelConfig.NEUTRAL_BAND_W = 0.0
        ModelConfig.BATCH_SIZE = old_batch

    summary = {
        "kind": "neutral_band_experiment",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "data_file": str(Path(args.data_file).resolve()),
        "steps": args.steps,
        "seed": args.seed,
        "band_w": args.band_w,
        "batch_size": args.batch,
        "runs": rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # MD 报告
    md = [
        "# 中性带正则（NEUTRAL_BAND_W）开/关 短训对比\n",
        f"- 数据：`{summary['data_file']}` · steps={args.steps} · seed={args.seed} · "
        f"batch={args.batch}（两档一致）· 开档 w={args.band_w}",
        "- 判定：FLAT 占比（0.5 阈值下）更高 = 模型真的学会了观望；holdout 分不塌 = 代价可接受。\n",
        "| run | w | 全序列 FLAT@0.05/0.5/0.8 | holdout 窗口 FLAT@0.05/0.5/0.8 | "
        "holdout@0.05 收益/夏普/回撤 | holdout@0.8 收益/夏普/回撤 | 引擎闸门分 | 闸门 | 训练 best |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, ev in rows.items():
        if "error" in ev:
            md.append(f"| {tag} | {ev.get('neutral_w')} | 错误: {ev['error']} | | | | | | |")
            continue
        fs = ev["flat_share"]
        hf = ev.get("holdout_flat_share") or {}
        h05 = ev.get("holdout_at_05") or {}
        h08 = ev.get("holdout_at_08") or {}
        md.append(
            f"| {tag} | {ev.get('neutral_w')} | {fs.get('0.05')}/{fs.get('0.5')}/{fs.get('0.8')} | "
            f"{hf.get('0.05')}/{hf.get('0.5')}/{hf.get('0.8')} | "
            f"{h05.get('total_return', '—')}/{h05.get('sharpe', '—')}/{h05.get('max_drawdown', '—')} | "
            f"{h08.get('total_return', '—')}/{h08.get('sharpe', '—')}/{h08.get('max_drawdown', '—')} | "
            f"{ev.get('engine_holdout_val')} | {'✅' if ev.get('engine_holdout_passed') else '❌'} | "
            f"{ev.get('engine_best_score')} |"
        )
    md.append("")
    off, on = rows.get("nb_off_s%d" % args.seed), rows.get("nb_on_s%d" % args.seed)
    md.append("## 结论\n")
    if off and on and "error" not in off and "error" not in on:
        d = round((on["flat_share"]["0.5"] - off["flat_share"]["0.5"]) * 100, 1)
        vo, vn = off.get("engine_holdout_val"), on.get("engine_holdout_val")
        md.append(f"- 开档 FLAT@0.5 占比 {off['flat_share']['0.5'] * 100:.1f}% → "
                  f"{on['flat_share']['0.5'] * 100:.1f}%（Δ{d:+.1f}pp）")
        md.append(f"- 引擎 holdout 分：关 {vo} vs 开 {vn}"
                  f"{'；开档仍过闸门' if on.get('engine_holdout_passed') else '；开档未过闸门'}")
        md.append("- 解读：FLAT 占比显著提升且 holdout 分未塌 → 中性带正则有效；"
                  "若 FLAT 未提升 → 权重/半宽需要调参（本实验只验证机制）。\n")
    else:
        md.append("- 数据不完整，见 json。\n")
    md_path = out.with_suffix(".REPORT.md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[done] 用时 {time.time() - t0:.0f}s → {out} + {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())