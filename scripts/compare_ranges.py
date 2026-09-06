"""范围对比实验：同一源文件，tail(最近N根) vs spread(全历史分块N根) 各跑一轮短训，
输出两路 val/best 曲线、holdout 闸门结果与结论建议（JSON + Markdown）。

串行执行（同机单 CPU，避免抢算力与 checkpoint 冲突）；每个变体都用引擎 run_tag
隔离，不写正式部署路径、不消费正式 holdout 指纹。

用法:
  python scripts/compare_ranges.py --data-file <源parquet> --n-bars 100000 \
      --steps 40 --seeds 42,7 --out-dir results --regime vol

输出:
  results/compare_<symbol>_<ts>.json/.md  + results/compare_latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_variant import run_variant  # noqa: E402


def _build_subsets(data_file: str, n_bars: int, chunks: int | None,
                   regime: str) -> dict[str, str]:
    from data_pipeline.train_sampler import prepare_training_subset

    tail = prepare_training_subset(data_file, mode="tail", n_bars=n_bars)
    spread = prepare_training_subset(data_file, mode="spread", n_bars=n_bars,
                                     n_chunks=chunks, regime=regime)
    return {"tail": tail["data_file"], "spread": spread["data_file"]}


def _pick_best(runs: list[dict]) -> dict | None:
    """跨 seed 挑 best_score 最高的那次（仅用于展示 in-sample best 曲线）。"""
    from web.compare_report import pick_best

    return pick_best(runs)


def _pick_representative(runs: list[dict]) -> dict | None:
    """结论代表：按【跨 seed 中位 holdout val_score】选那次运行（委托 web.compare_report）。"""
    from web.compare_report import pick_representative

    return pick_representative(runs)


def run_compare(data_file: str, n_bars: int, chunks: int | None, steps: int,
                seeds: list[int], regime: str, out_dir: Path,
                rep_criterion: str | None = None) -> dict:
    subsets = _build_subsets(data_file, n_bars, chunks, regime)
    variants: dict[str, list[dict]] = {"tail": [], "spread": []}
    for variant, sub in subsets.items():
        print(f"\n== [{variant}] {sub} ==", flush=True)
        for seed in seeds:
            tag = f"{variant}_s{seed}"
            print(f"  run tag={tag} seed={seed} steps={steps}", flush=True)
            res = run_variant(sub, tag=tag, steps=steps, seed=seed)
            res["variant"] = variant
            variants[variant].append(res)
            print(f"  -> best={res.get('best_score')} "
                  f"holdout_passed={bool((res.get('holdout') or {}).get('passed'))}",
                  flush=True)
    tail_best = _pick_best(variants["tail"])
    spread_best = _pick_best(variants["spread"])
    # 结论代表：跨 seed 中位 holdout（防 in-sample best 挑彩票）
    from web.compare_report import conclusion_for, median_holdout, pick_representative

    from web.compare_report import (CRITERION_DEFAULT, conclusions_by_criterion,
                                    pick_by_criterion)

    criterion = rep_criterion or CRITERION_DEFAULT
    tail_rep = pick_by_criterion(variants["tail"], criterion)
    spread_rep = pick_by_criterion(variants["spread"], criterion)
    conclusion = conclusion_for(tail_rep, spread_rep)

    summary = {
        "kind": "compare_ranges",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(Path(data_file).resolve()),
        "params": {"n_bars": n_bars, "chunks": chunks, "steps": steps,
                   "seeds": seeds, "regime": regime,
                   "rep_criterion": criterion},
        "variants": {
            "tail": {"runs": variants["tail"],
                     "best": tail_best,
                     "representative": tail_rep,
                     "median_holdout": median_holdout(variants["tail"]),
                     "holdout_passed": sum(
                         1 for r in variants["tail"]
                         if (r.get("holdout") or {}).get("passed"))},
            "spread": {"runs": variants["spread"],
                       "best": spread_best,
                       "representative": spread_rep,
                       "median_holdout": median_holdout(variants["spread"]),
                       "holdout_passed": sum(
                           1 for r in variants["spread"]
                           if (r.get("holdout") or {}).get("passed"))},
        },
        "conclusion": conclusion,
    }
    # 全部可代表口径各自给结论（in-sample best / holdout 中位 / holdout best）
    summary["conclusions_by_criterion"] = conclusions_by_criterion(summary)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sym = (tail_best or {}).get("symbol") or Path(data_file).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"compare_{sym}_{ts}.json"
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "compare_latest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # docs 报告：两路曲线 best/holdout 结论自动落盘（docs/compare_tail_vs_spread_REPORT.md）
    try:
        from web.compare_report import write_docs_report
        report_path = write_docs_report(summary)
        print(f"[compare] docs 报告已生成: {report_path}", flush=True)
    except Exception as exc:  # noqa: BLE001 报告失败不影响对比结果本身
        print(f"[compare] docs 报告生成失败(忽略): {exc}", flush=True)

    def cell(r: dict | None, key: str, fn=lambda v: v) -> str:
        if not r:
            return "—"
        ho = r.get("holdout") or {}
        if key == "holdout":
            return str(ho.get("val_score"))
        if key == "passed":
            return "✓" if ho.get("passed") else "✗"
        if key == "sharpe":
            return str(ho.get("sharpe"))
        return str(fn(r.get(key)))

    md = [f"# 范围对比实验 {sym}", "",
          f"- 源: {Path(data_file).resolve()}",
          f"- 参数: N={n_bars} 块数={chunks} steps={steps} seeds={seeds} regime={regime}",
          "",
          "| 变体 | best | holdout 分 | 通过 | Sharpe |",
          "|---|---|---|---|---|",
          f"| tail(最近N) | {cell(tail_best, 'best_score')} | "
          f"{cell(tail_best, 'holdout')} | {cell(tail_best, 'passed')} | "
          f"{cell(tail_best, 'sharpe')} |",
          f"| spread(全历史分块) | {cell(spread_best, 'best_score')} | "
          f"{cell(spread_best, 'holdout')} | {cell(spread_best, 'passed')} | "
          f"{cell(spread_best, 'sharpe')} |",
          "",
          "## 结论",
          *[f"- {l}" for l in conclusion],
          ""]
    (out_dir / f"compare_{sym}_{ts}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n[结论]", *conclusion, sep="\n  - ", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="tail vs spread range-compare experiment")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--n-bars", type=int, default=60_000)
    ap.add_argument("--chunks", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--regime", default="vol")
    ap.add_argument("--rep-criterion", default=None,
                    choices=["in_sample", "holdout_median", "holdout_best"],
                    help="结论代表口径：in_sample / holdout_median(默认) / holdout_best")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    # 防终端回收假中止：孤儿化 SIGTERM → 重挂 launchd 继续跑
    try:
        from model_core.supervise import install_orphan_sigterm_handler
        install_orphan_sigterm_handler("compare_ranges",
                                       log_path=str(PROJECT_ROOT / "logs" / "compare_ranges_guard.log"))
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    run_compare(args.data_file, args.n_bars, args.chunks, args.steps,
                seeds, args.regime, PROJECT_ROOT / args.out_dir,
                rep_criterion=args.rep_criterion)
    print(f"\n[完成] 用时 {time.time() - t0:.0f}s → results/compare_*")


if __name__ == "__main__":
    main()
