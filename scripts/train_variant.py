"""实验用单次短训驱动：隔离运行一轮训练并输出结果 JSON。

与正式训练（train_file.py）的区别：
- 使用引擎 run_tag 隔离：历史/检查点文件名带后缀，绝不写 *.live.json /
  strategies/best_*.json / 不消费正式 holdout 指纹对应文件（部署被抑制）。
- 结束后清理本 run 产生的 checkpoint/history 残留。
- 输出 --out-json 汇总：{tag, seed, steps, symbol, timeframe, data_file,
  subset_bars(T_full), holdout_bars, best_score, best_formula,
  holdout(passed/gate/...), history:{step,val_score,best_score,avg_reward}}。

用法:
  python scripts/train_variant.py --data-file <parquet> --tag <run_tag> \
      --steps N --seed N --out-json path.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _cleanup_artifacts(symbol: str, tag: str) -> None:
    """删除本 run 生成的 tagged 产物（history/checkpoint），避免污染工作区。"""
    import glob
    import os

    patterns = [
        str(PROJECT_ROOT / f"training_history_{symbol}_{tag}.json"),
        str(PROJECT_ROOT / "checkpoints" / f"ckpt_{symbol}_{tag}_step_*.pt"),
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except OSError:
                pass


def run_variant(data_file: str, tag: str, steps: int, seed: int) -> dict:
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.config import ModelConfig
    from model_core.engine import AlphaEngine

    old_steps = ModelConfig.TRAIN_STEPS
    ModelConfig.TRAIN_STEPS = max(1, int(steps))
    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        symbol, timeframe = mgr.symbol, mgr.timeframe
        engine = AlphaEngine(data_manager=mgr, target_symbol=symbol, seed=int(seed))
        engine.timeframe = timeframe
        engine.data_file = str(Path(data_file).resolve())
        engine.run_tag = tag
        engine._suppress_deploy = True  # 实验永不部署
        engine.train(verbose_header=False)

        best = engine.best_score if engine.best_score not in (None, -float("inf")) else None
        hist = engine.training_history
        result = {
            "tag": tag,
            "seed": int(seed),
            "steps": len(hist.get("step", [])),
            "symbol": symbol,
            "timeframe": timeframe,
            "data_file": str(Path(data_file).resolve()),
            "source_bars": int(mgr.raw_dict["close"].shape[1]),
            "holdout_bars": int(getattr(engine, "holdout_bars", 0) or 0),
            "best_score": round(float(best), 6) if best is not None else None,
            "best_formula": engine.best_formula,
            "holdout": (engine.holdout or None),
            "history": {
                "step": list(hist.get("step", [])),
                "val_score": list(hist.get("val_score", [])),
                "best_score": list(hist.get("best_score", [])),
                "avg_reward": list(hist.get("avg_reward", [])),
            },
        }
        # 结果已落盘所需全部数据，清理本 run 的 tagged 残留（history/ckpt）
        try:
            _cleanup_artifacts(symbol, tag)
        except Exception:  # noqa: BLE001
            pass
        return result
    finally:
        ModelConfig.TRAIN_STEPS = old_steps


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated short training run (experiments)")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--holdout-bars", type=int, default=None,
                    help="样本外预留根数覆盖（默认 ModelConfig.HOLDOUT_BARS=500）")
    args = ap.parse_args()
    if args.holdout_bars is not None:
        ModelConfig.HOLDOUT_BARS = max(0, int(args.holdout_bars))

    res = run_variant(args.data_file, args.tag, args.steps, args.seed)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    # 清理残留
    try:
        _cleanup_artifacts(res.get("symbol") or "", args.tag)
    except Exception:  # noqa: BLE001
        pass
    print(f"[train_variant:{args.tag}] done -> {out} (best={res.get('best_score')})")


if __name__ == "__main__":
    main()
