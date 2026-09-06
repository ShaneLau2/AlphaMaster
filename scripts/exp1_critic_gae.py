"""exp1_critic_gae.py — E1 实验: critic 价值头 + GAE vs 现有 EMA-baseline REINFORCE。

对比口径（与 train_variant 同款实验纪律）:
- run_tag 隔离: 历史/检查点文件名带后缀, 绝不写 *.live.json / best_*.json;
- engine._suppress_deploy=True: _finalize_champion 直接 suppressed, 不部署、不消费
  holdout 单次批准、不写 champion_history;
- 结束后清理本 run 的 tagged history/ckpt 残留。

用法:
  python scripts/exp1_critic_gae.py --data-file <parquet> --steps N --seed S \\
      --variant baseline|critic --out-json results/exp1_xxx.json
  # 可选: --gamma 1.0 --lambda 1.0 --loss-w 1.0  (critic 分支超参)

输出 JSON 字段含 best_score / best_formula / holdout / 各步曲线 / 墙钟时间,
供 compare 与报告使用。critic 分支额外记录 critic_loss / adv_mag 曲线。
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


def _cleanup_artifacts(tag: str) -> None:
    """删除本 run 生成的 tagged 产物（history/checkpoint），避免污染工作区。"""
    import glob
    import os

    patterns = [
        str(PROJECT_ROOT / f"training_history_*{tag}*.json"),
        str(PROJECT_ROOT / "checkpoints" / f"ckpt_*{tag}*_step_*.pt"),
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except OSError:
                pass


def run_arm(data_file: str, variant: str, tag: str, steps: int, seed: int,
            gamma: float, lam: float, loss_w: float) -> dict:
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.config import ModelConfig
    from model_core.engine import AlphaEngine

    assert variant in ("baseline", "critic")
    old_steps = ModelConfig.TRAIN_STEPS
    ModelConfig.TRAIN_STEPS = max(1, int(steps))
    # 显式设置实验开关（进程内生效; 默认关闭 = 行为不变）
    setattr(ModelConfig, "ACC_CRITIC_GAE", variant == "critic")
    setattr(ModelConfig, "ACC_CRITIC_GAMMA", float(gamma))
    setattr(ModelConfig, "ACC_CRITIC_LAMBDA", float(lam))
    setattr(ModelConfig, "ACC_CRITIC_LOSS_W", float(loss_w))
    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        symbol, timeframe = mgr.symbol, mgr.timeframe
        engine = AlphaEngine(data_manager=mgr, target_symbol=symbol, seed=int(seed))
        engine.timeframe = timeframe
        engine.data_file = str(Path(data_file).resolve())
        engine.run_tag = tag
        engine._suppress_deploy = True  # 实验永不部署

        t0 = time.time()
        engine.train(verbose_header=False)
        wall_s = time.time() - t0

        best = engine.best_score if engine.best_score not in (None, -float("inf")) else None
        hist = engine.training_history
        n_steps = len(hist.get("step", []))
        result = {
            "tag": tag,
            "variant": variant,
            "seed": int(seed),
            "steps_requested": max(1, int(steps)),
            "steps_done": n_steps,
            "wall_s": round(wall_s, 1),
            "s_per_step": round(wall_s / max(1, n_steps), 3),
            "symbol": symbol,
            "timeframe": timeframe,
            "data_file": str(Path(data_file).resolve()),
            "source_bars": int(mgr.raw_dict["close"].shape[1]),
            "holdout_bars": int(getattr(engine, "holdout_bars", 0) or 0),
            "restart_count": int(getattr(engine, "_restart_count", 0) or 0),
            "best_score": round(float(best), 6) if best is not None else None,
            "best_formula": engine.best_formula,
            "formula_decoded": engine._decode_formula(engine.best_formula),
            "holdout": (engine.holdout or None),
            "critic_hyper": {
                "gamma": float(gamma), "lambda": float(lam), "loss_w": float(loss_w),
            },
            "history": {
                "step": list(hist.get("step", [])),
                "val_score": list(hist.get("val_score", [])),
                "best_score": list(hist.get("best_score", [])),
                "avg_reward": list(hist.get("avg_reward", [])),
                "critic_loss": list(hist.get("critic_loss", [])),
                "adv_mag": list(hist.get("adv_mag", [])),
            },
        }
        return result
    finally:
        ModelConfig.TRAIN_STEPS = old_steps
        setattr(ModelConfig, "ACC_CRITIC_GAE", False)


def main() -> None:
    ap = argparse.ArgumentParser(description="E1: EMA-baseline REINFORCE vs critic+GAE")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--variant", required=True, choices=["baseline", "critic"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--loss-w", dest="loss_w", type=float, default=1.0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    res = run_arm(args.data_file, args.variant, args.tag, args.steps,
                  args.seed, args.gamma, args.lam, args.loss_w)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _cleanup_artifacts(args.tag)
    except Exception:  # noqa: BLE001
        pass
    print(f"[exp1:{args.tag}] done -> {out}")
    print(f"  variant={args.variant} steps={res['steps_done']} "
          f"wall={res['wall_s']}s ({res['s_per_step']}s/step) best_val={res['best_score']}")
    if res["best_formula"]:
        print(f"  best={res['best_formula']}  =>  {res['formula_decoded']}")
    if res.get("holdout"):
        h = res["holdout"]
        print(f"  holdout: val={h.get('val_score')} sharpe={h.get('sharpe')} "
              f"passed={h.get('passed')}")


if __name__ == "__main__":
    main()
