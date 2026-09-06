"""M2/M2b/M3 统一入口：多市场共享 policy 训练（Mode C）。

用法示例：
  # 生产式训练（run_tag 空 → 终局全市场闸门通过后逐 best_{sym}.json 部署）
  .venv/bin/python -u scripts/train_multi.py --policy crypto --steps 9000 \
      --seed 42 --out-json results/multi_crypto.json

  # 实验 run（run_tag 非空 → 不部署，输出全进 out-json）
  .venv/bin/python -u scripts/train_multi.py --policy crypto --steps 150 \
      --seed 42 --tag expA --batch 2 --calib-every 20 --alpha 0.5 \
      --out-json results/multi_expA.json > logs/multi_expA.log 2>&1

  # 断点续训：--steps 为绝对目标总步数（含已训步）。
  # 例：已在 step_3000 存过断点 → 续训到 6000：
  #   .venv/bin/python -u scripts/train_multi.py --policy crypto --steps 6000 \
  #       --resume checkpoints/ckpt_MULTI_crypto_step_3000.pt --tag expB \
  #       --out-json results/multi_expB.json

关键参数（对应 ModelConfig.MARKET_*）：
  --batch B        普通步市场批次大小（默认 2）
  --calib-every K  每 K 步全市场 calibration（默认 20；冠军晋升只在 calib 步）
  --alpha A        P(market) ∝ N^A（默认 0.5；0=等权 1=按数据量）
  --min-sharpe S   任一市场聚合分低于 S 否决冠军晋升（min 门槛）
  --no-group-mean  平权 mean（默认组级等权聚合）
  --vol-gate       选优闸门口径: default|tier|grid|off（default=跟随 ModelConfig）
  --markets A,B,C  显式指定训练腿（逗号分隔 symbol；覆盖 registry role=train）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _apply_vol_gate(vg: str) -> None:
    from model_core.config import ModelConfig
    if vg == "default":
        return
    if vg == "off":
        ModelConfig.VOL_COVERAGE_ENABLED = False
        return
    ModelConfig.VOL_COVERAGE_ENABLED = True
    ModelConfig.VOL_COVERAGE_GRID = (vg == "grid")


def run_train(policy: str, steps: int, seed: int, tag: str,
              batch: int, calib_every: int, alpha: float, min_sharpe: float,
              group_mean: bool, vol_gate: str, markets: list[str] | None,
              resume: str | None, out_json: str, batch_size: int | None,
              max_len: int | None, n_folds: int) -> dict:
    from model_core.config import ModelConfig
    _apply_vol_gate(vol_gate)
    ModelConfig.MARKET_BATCH_SIZE = batch
    ModelConfig.MARKET_CALIB_EVERY = calib_every
    ModelConfig.MARKET_SAMPLE_ALPHA = alpha
    ModelConfig.MARKET_MIN_SHARPE = min_sharpe
    ModelConfig.MARKET_GROUP_MEAN = group_mean
    if batch_size is not None:
        ModelConfig.BATCH_SIZE = batch_size
    if max_len is not None:
        ModelConfig.MAX_FORMULA_LEN = max_len

    from data_pipeline.market_registry import load_registry
    from model_core.multimarket import MultiMarketEngine

    reg = load_registry()
    specs = None
    if markets:
        all_m = [m for pol in reg["policies"].values()
                 for m in pol.get("markets", [])] + reg.get("oos_review", [])
        by_sym = {m["symbol"]: m for m in all_m}
        specs = []
        for sym in markets:
            if sym not in by_sym:
                raise SystemExit(f"registry 中无 {sym}")
            specs.append(by_sym[sym])

    eng = MultiMarketEngine(policy=policy, seed=seed, markets=specs,
                            registry=None, n_folds=n_folds, run_tag=tag)
    start_step = 0
    if resume:
        # 语义：steps 为绝对目标步数；续训从 checkpoint 完成步继续，避免重放前缀
        start_step = eng.load_checkpoint(resume)
        if start_step >= steps:
            print(f"[multimarket] checkpoint 已完成 {start_step} 步 ≥ 目标 {steps} 步，无事可做。")
            raise SystemExit(0)
        print(f"[multimarket] 续训: 从第 {start_step} 步继续训练到第 {steps} 步"
              f"（{steps - start_step} 步增量）")

    t0 = time.time()
    eng.train(start_step, steps)
    wall = time.time() - t0

    # ── 结果汇总 ──────────────────────────────────────────────────────
    res: dict = {
        "engine": "MultiMarketEngine",
        "policy": policy,
        "tag": tag,
        "seed": seed,
        "steps": steps,
        "wall_seconds": round(wall, 2),
        "best_formula": eng.best_formula,
        "best_formula_decoded": eng._decode_formula(eng.best_formula),
        "best_score": eng.best_score,
        "restart_count": eng._restart_count,
        "config": {
            "batch": batch, "calib_every": calib_every, "alpha": alpha,
            "min_sharpe": min_sharpe, "group_mean": group_mean,
            "vol_gate": vol_gate, "vol_coverage_grid": bool(
                ModelConfig.VOL_COVERAGE_GRID),
            "vol_coverage_enabled": bool(ModelConfig.VOL_COVERAGE_ENABLED),
            "batch_size": ModelConfig.BATCH_SIZE,
            "max_len": ModelConfig.MAX_FORMULA_LEN,
            "resume_from": Path(resume).name if resume else None,
            "start_step": start_step,
        },
        "markets": [e.name for e in eng.envs],
        "market_weights": {e.name: round(w, 4)
                           for e, w in zip(eng.envs, eng._market_weights)},
        "history": {k: v for k, v in eng.training_history.items()
                    if k not in ("markets", "market_rewards", "is_calib")},
        "market_history": {k: eng.training_history.get(k)
                           for k in ("markets", "market_rewards", "is_calib")},
        "champion_outcome": eng.champion_outcome,
    }

    # 冠军逐市场表现（训练窗聚合 + holdout 生产口径）
    if eng.best_formula:
        per_mkt = {}
        for env in eng.envs:
            rr = env.eval_formulas([eng.best_formula], collect_res=False)
            ho = env.holdout(eng.best_formula)
            per_mkt[env.name] = {
                "train_reward": round(float(rr["rewards"][0].item()), 4),
                "val_score": round(float(rr["vals"][0].item()), 4),
                "vol_ok": bool(rr["vol_ok"][0]),
                "status": rr["status"][0],
                "holdout": ho,
            }
        res["best_per_market"] = per_mkt

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        tmp = out_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(res, fp, ensure_ascii=False, indent=2)
        os.replace(tmp, out_json)
        print(f"\n[结果] → {out_json}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="多市场共享 policy 训练（Mode C）")
    ap.add_argument("--policy", default="crypto")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="非空=实验 run（不部署）")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--calib-every", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--min-sharpe", type=float, default=0.0)
    ap.add_argument("--no-group-mean", action="store_true")
    ap.add_argument("--vol-gate", choices=["default", "tier", "grid", "off"],
                    default="default")
    ap.add_argument("--markets", default=None,
                    help="逗号分隔 symbol 列表，覆盖 registry role=train")
    ap.add_argument("--resume", default=None, help="checkpoint 路径续训")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 BATCH_SIZE（冒烟用）")
    ap.add_argument("--max-len", type=int, default=None, help="覆盖 MAX_FORMULA_LEN（冒烟用）")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    markets = [m.strip() for m in args.markets.split(",")] if args.markets else None
    run_train(args.policy, args.steps, args.seed, args.tag,
              args.batch, args.calib_every, args.alpha, args.min_sharpe,
              not args.no_group_mean, args.vol_gate, markets,
              args.resume, args.out_json, args.batch_size, args.max_len,
              args.n_folds)


if __name__ == "__main__":
    main()
