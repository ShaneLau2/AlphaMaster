"""老数据主导 vs 新数据主导 —— 冠军闸门通过率实验。

研究问题：训练数据里「老年代 vs 新年代」占比如何影响泛化？
做法：同一源文件构建两个连续子集（前 70% = 老主导，后 70% = 新主导），
各自多 seed 短训；统计 holdout 闸门通过率、均值 best/val、未过原因分布。

注意：训练昂贵，默认是小规模 pilot（steps 较小）。正式结论建议加大步数
并固定 seed 列表过夜运行：
  python scripts/experiment_old_vs_new.py --data-file <parquet> \\
      --steps 400 --seeds 11,22,33,42

输出: results/old_vs_new_<symbol>_<ts>.json/.md + old_vs_new_latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_variant import run_variant  # noqa: E402


def _build_subsets(data_file: str, frac_old: float) -> dict[str, str]:
    """老主导 = 前 frac_old（按根数）；新主导 = 后 frac_old（连续尾部）。"""
    from data_pipeline.parquet_manager import inspect_parquet_file

    df = pd.read_parquet(data_file).sort_values("time")
    n = len(df)
    k = int(n * frac_old)
    info = inspect_parquet_file(data_file)
    name = Path(info["data_file"]).name  # {SYM}_{TF}.parquet
    slices = PROJECT_ROOT / "data" / "slices"
    paths: dict[str, str] = {}
    for key, sl in (("old", df.iloc[:k]), ("new", df.iloc[n - k:])):
        out = slices / f"oldvsnew_{key}_{int(frac_old * 100)}" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        sl.reset_index(drop=True).to_parquet(out, index=False)
        paths[key] = str(out)
    return paths


def _summarize(runs: list[dict]) -> dict:
    ho = [r.get("holdout") for r in runs]
    passed = sum(1 for h in ho if h and h.get("passed"))
    bests = [r["best_score"] for r in runs if r.get("best_score") is not None]
    vals = [h.get("val_score") for h in ho if h and h.get("val_score") is not None]
    reasons: dict[str, int] = {}
    for h in ho:
        if h and h.get("passed") is not True:
            for g in (h.get("gate") or []):
                reasons[str(g)] = reasons.get(str(g), 0) + 1
    return {
        "runs": len(runs),
        "holdout_passed": passed,
        "pass_rate": round(passed / len(runs), 3) if runs else None,
        "best_score_avg": round(float(np.mean(bests)), 4) if bests else None,
        "best_score_max": round(float(np.max(bests)), 4) if bests else None,
        "holdout_val_avg": round(float(np.mean(vals)), 4) if vals else None,
        "gate_fail_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def run_old_vs_new(data_file: str, frac_old: float, steps: int,
                   seeds: list[int], out_dir: Path) -> dict:
    subsets = _build_subsets(data_file, frac_old)
    variants: dict[str, list[dict]] = {}
    for key in ("old", "new"):
        print(f"\n== [{key}-dominant] {subsets[key]} ==", flush=True)
        variants[key] = []
        for seed in seeds:
            tag = f"{key}_s{seed}"
            print(f"  run tag={tag} seed={seed} steps={steps}", flush=True)
            res = run_variant(subsets[key], tag=tag, steps=steps, seed=seed)
            res["variant"] = key
            variants[key].append(res)
            ho = res.get("holdout") or {}
            print(f"  -> best={res.get('best_score')} "
                  f"holdout_passed={bool(ho.get('passed'))} holdout_val={ho.get('val_score')}",
                  flush=True)
    summary = {
        "kind": "old_vs_new",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(Path(data_file).resolve()),
        "frac_old": frac_old,
        "params": {"steps": steps, "seeds": seeds},
        "subsets": subsets,
        "variants": {"old": _summarize(variants["old"]),
                     "new": _summarize(variants["new"])},
        "conclusion": (
            ["新主导 holdout 通过率更高 → 训练越贴近近期，泛化越好（常见于行情漂移较快）。",
             "老主导通过率更高 → 覆盖更多老年代有助稳健（常见于周期性强品种）。",
             "两者相当/均未通过 → 短训噪声主导，需加大步数与 seed 数。"]
        ),
    }
    # 给出机器结论倾向
    po, pn = variants["old"], variants["new"]
    rate_o = summary["variants"]["old"]["pass_rate"] or 0
    rate_n = summary["variants"]["new"]["pass_rate"] or 0
    if rate_n > rate_o:
        summary["conclusion"] = [summary["conclusion"][0]]
    elif rate_o > rate_n:
        summary["conclusion"] = [summary["conclusion"][1]]
    else:
        summary["conclusion"] = [summary["conclusion"][2]]

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sym = Path(data_file).stem.split("_")[0]
    (out_dir / f"old_vs_new_{sym}_{ts}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "old_vs_new_latest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def row(key: str) -> str:
        v = summary["variants"][key]
        return (f"| {key}-dominant | {v['pass_rate']} ({v['holdout_passed']}/{v['runs']}) "
                f"| {v['best_score_avg']} | {v['holdout_val_avg']} |")
    md = [
        f"# 老 vs 新主导训练实验 {sym}",
        "",
        f"- 源: {Path(data_file).resolve()}",
        f"- 切分: 前 {int(frac_old*100)}% / 后 {int(frac_old*100)}%（连续）  "
        f"steps={steps} seeds={seeds}",
        "",
        "| 子集 | holdout 通过率 | best(均值) | holdout分(均值) |",
        "|---|---|---|---|",
        row("old"), row("new"),
        "",
        "## 结论倾向",
        *[f"- {c}" for c in summary["conclusion"]],
        "",
    ]
    (out_dir / f"old_vs_new_{sym}_{ts}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n[结论]", *summary["conclusion"], sep="\n  - ", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="old-dominant vs new-dominant training experiment")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--frac-old", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seeds", default="42,1")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    run_old_vs_new(args.data_file, args.frac_old, args.steps, seeds,
                   PROJECT_ROOT / args.out_dir)
    print(f"\n[完成] 用时 {time.time() - t0:.0f}s → results/old_vs_new_*")


if __name__ == "__main__":
    main()
