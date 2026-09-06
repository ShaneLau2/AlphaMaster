"""exp3_length_curriculum.py — E3 实验: 8→14 token 两阶段长度课程 vs 固定长度。

对比同一总预算(总步数相同)下三档:
  direct8      : 全程 MAX_FORMULA_LEN=8   (现状)
  direct14     : 全程 MAX_FORMULA_LEN=14  (直接长公式)
  curriculum   : 阶段A len=8 跑 S_A=round(split*N) 步 → 阶段B len=14 跑剩余
                 (复用 AlphaGPT 已预留的 pos_emb 上限 20,无需改模型)

实现: 同一 engine 实例上分两次 train() 调用(状态/最优/精英池/重启计数跨调用
持续),中途切 ModelConfig.MAX_FORMULA_LEN;阶段B 关精英回放(len-8 池与 len-14
解码不兼容:Part B 的 tok_e_t[:, si] 会越界)。训练结束在 holdout 上用生产口径
验证最优公式(长度 8 或 14 均可由 StackVM 执行)。

实验纪律: run_tag 隔离 + _suppress_deploy=True,不写 best_*.json / *.live.json,
不消费 holdout 单次批准,结束清理 tagged 残留。

用法:
  python scripts/exp3_length_curriculum.py --mode direct8    --data-file <p> --steps 200 --seed 42 --tag exp3_d8  --out-json results/exp3_d8.json
  python scripts/exp3_length_curriculum.py --mode direct14   --data-file <p> --steps 200 --seed 42 --tag exp3_d14 --out-json results/exp3_d14.json
  python scripts/exp3_length_curriculum.py --mode curriculum --data-file <p> --steps 200 --seed 42 --tag exp3_cur --out-json results/exp3_cur.json --split 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _cleanup_artifacts(tag: str) -> None:
    for pat in (str(PROJECT_ROOT / f"training_history_*{tag}*.json"),
                str(PROJECT_ROOT / "checkpoints" / f"ckpt_*{tag}*_step_*.pt")):
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except OSError:
                pass


def _apply_vol_gate(vg: str) -> None:
    """显式钉住选优层 vol 覆盖口径(default=跟随 ModelConfig)。"""
    from model_core.config import ModelConfig
    if vg == "default":
        return
    if vg == "off":
        ModelConfig.VOL_COVERAGE_ENABLED = False
        return
    ModelConfig.VOL_COVERAGE_ENABLED = True
    ModelConfig.VOL_COVERAGE_GRID = (vg == "grid")


def run_arm(data_file: str, mode: str, tag: str, steps: int, seed: int,
            split: float, vol_gate: str = "default") -> dict:
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.config import ModelConfig
    from model_core.engine import AlphaEngine

    assert mode in ("direct8", "direct14", "curriculum")
    assert 0.0 < split < 1.0
    _apply_vol_gate(vol_gate)
    old_steps = ModelConfig.TRAIN_STEPS
    old_max = ModelConfig.MAX_FORMULA_LEN
    old_elite = ModelConfig.ELITE_REPLAY_FRAC
    ModelConfig.TRAIN_STEPS = max(1, int(steps))
    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        symbol, timeframe = mgr.symbol, mgr.timeframe
        engine = AlphaEngine(data_manager=mgr, target_symbol=symbol, seed=int(seed))
        engine.timeframe = timeframe
        engine.data_file = str(Path(data_file).resolve())
        engine.run_tag = tag
        engine._suppress_deploy = True

        phases: list[dict] = []
        t0 = time.time()

        if mode == "direct8":
            ModelConfig.MAX_FORMULA_LEN = 8
            engine.train(verbose_header=False)
            phases.append({"len": 8, "start": 0, "end": steps, "elite_off": False})
        elif mode == "direct14":
            ModelConfig.MAX_FORMULA_LEN = 14
            engine.train(verbose_header=False)
            phases.append({"len": 14, "start": 0, "end": steps, "elite_off": False})
        else:  # curriculum
            s_a = max(1, int(round(split * steps)))
            ModelConfig.MAX_FORMULA_LEN = 8
            engine.train(start_step=0, end_step=s_a, verbose_header=False)
            phases.append({"len": 8, "start": 0, "end": s_a, "elite_off": False})
            # 阶段B: 切到 len=14。池内遗留 len-8 公式与 len-14 解码不兼容
            # (Part B 的 tok_e_t[:, si] 会越界), 先清空精英池; 之后池内重新
            # 累积 len-14 公式, 精英回放机制与 direct14 对齐。
            ModelConfig.MAX_FORMULA_LEN = 14
            engine._elite_pool = []
            engine.train(start_step=s_a, end_step=steps, verbose_header=False)
            phases.append({"len": 14, "start": s_a, "end": steps, "elite_off": False,
                           "note": "elite pool cleared at switch"})

        wall_s = time.time() - t0
        hist = engine.training_history
        n_steps = len(hist.get("step", []))
        best = engine.best_score if engine.best_score not in (None, -float("inf")) else None
        result = {
            "tag": tag, "mode": mode, "seed": int(seed),
            "steps_requested": max(1, int(steps)), "steps_done": n_steps,
            "phases": phases,
            "wall_s": round(wall_s, 1),
            "s_per_step": round(wall_s / max(1, n_steps), 3),
            "symbol": symbol, "timeframe": timeframe,
            "data_file": str(Path(data_file).resolve()),
            "source_bars": int(mgr.raw_dict["close"].shape[1]),
            "holdout_bars": int(getattr(engine, "holdout_bars", 0) or 0),
            "restart_count": int(getattr(engine, "_restart_count", 0) or 0),
            "best_score": round(float(best), 6) if best is not None else None,
            "best_formula": engine.best_formula,
            "best_formula_len": len(engine.best_formula) if engine.best_formula else 0,
            "formula_decoded": engine._decode_formula(engine.best_formula),
            "holdout": (engine.holdout or None),
            "history": {
                "step": list(hist.get("step", [])),
                "val_score": list(hist.get("val_score", [])),
                "best_score": list(hist.get("best_score", [])),
                "avg_reward": list(hist.get("avg_reward", [])),
            },
        }
        return result
    finally:
        ModelConfig.TRAIN_STEPS = old_steps
        ModelConfig.MAX_FORMULA_LEN = old_max
        ModelConfig.ELITE_REPLAY_FRAC = old_elite


def main() -> None:
    ap = argparse.ArgumentParser(description="E3: 8→14 token 长度课程 vs 固定长度")
    ap.add_argument("--mode", required=True, choices=["direct8", "direct14", "curriculum"])
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, default=0.5,
                    help="curriculum 阶段A(len=8)占总步数比例")
    ap.add_argument("--vol-gate", choices=["default", "tier", "grid", "off"],
                    default="default",
                    help="选优层 vol 覆盖口径钉住(default=跟随 ModelConfig; "
                         "tier=段级3段, grid=vol×er 9格, off=关)")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    res = run_arm(args.data_file, args.mode, args.tag, args.steps,
                  args.seed, args.split, args.vol_gate)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _cleanup_artifacts(args.tag)
    except Exception:  # noqa: BLE001
        pass
    ph = " + ".join(f"len{p['len']}[{p['start']}:{p['end']}]" for p in res["phases"])
    print(f"[exp3:{args.tag}] done -> {out}")
    print(f"  mode={res['mode']} phases={ph} steps={res['steps_done']} "
          f"wall={res['wall_s']}s ({res['s_per_step']}s/step) best_val={res['best_score']}")
    print(f"  best_len={res['best_formula_len']} best={res['best_formula']}  =>  "
          f"{res['formula_decoded']}")
    if res.get("holdout"):
        h = res["holdout"]
        print(f"  holdout: val={h.get('val_score')} sharpe={h.get('sharpe')} "
              f"passed={h.get('passed')}")


if __name__ == "__main__":
    main()
