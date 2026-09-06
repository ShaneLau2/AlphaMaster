"""exp1g_compare_gae.py — E1-GAE 修正对比: 真折现 GAE (γ<1, λ<1) vs
原"逐位置 baseline"版 (γ=λ=1) critic 分支, 同 seed / 同数据 / 同步数。

背景: 原 E1 critic 分支的 GAE 用 γ=1, λ=1, 数学上塌缩为逐位置学习基线
(A_t = r_term − v_t), 且 critic 目标对所有 token 位置都回归未折扣的 r_term
(只在 γ=1 时自洽)。本实验把引擎改为折扣一致的目标
(位置 t 的目标 = γ^(T−1−t) × r_term), 并对比 γ=0.99/λ=0.95 与 γ=1/λ=1。

用法:
  python scripts/exp1g_compare_gae.py --data-file <parquet> --steps N --seed S \
      --tag-prefix exp1g_xxx --out-json results/exp1g_xxx.json

按顺序跑两条 arm (串行, 避免 CPU 争抢):
  1) flat  : ACC_CRITIC_GAE + γ=1.0 λ=1.0   (复现原"逐位置 baseline"语义)
  2) gae   : ACC_CRITIC_GAE + γ=0.99 λ=0.95 (真折现 GAE)
输出含两条 arm 的完整曲线 (best/val/ent 经日志, critic_loss/adv_mag 在 JSON)。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARMS = [
    {"name": "flat", "gamma": 1.0, "lam": 1.0},
    {"name": "gae", "gamma": 0.99, "lam": 0.95},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag-prefix", default="exp1g")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    results = {}
    for arm in ARMS:
        tag = f"{args.tag_prefix}_{arm['name']}_{args.steps}_s{args.seed}"
        out = PROJECT_ROOT / "results" / f"{tag}.json"
        cmd = [
            sys.executable, "-u", str(PROJECT_ROOT / "scripts/exp1_critic_gae.py"),
            "--data-file", args.data_file,
            "--variant", "critic",
            "--tag", tag,
            "--steps", str(args.steps),
            "--seed", str(args.seed),
            "--gamma", str(arm["gamma"]),
            "--lam", str(arm["lam"]),
            "--out-json", str(out),
        ]
        print(f"\n{'='*60}\n[{arm['name']}] γ={arm['gamma']} λ={arm['lam']} "
              f"steps={args.steps} seed={args.seed}\n{'='*60}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        wall = time.time() - t0
        print(f"[{arm['name']}] exit={proc.returncode} wall={wall:.1f}s", flush=True)
        if not out.exists():
            raise SystemExit(f"[{arm['name']}] 缺少输出 {out}")
        d = json.loads(out.read_text(encoding="utf-8"))
        results[arm["name"]] = {
            "json": str(out.relative_to(PROJECT_ROOT)),
            "wall_s": round(wall, 1),
            "best_score": d.get("best_score"),
            "holdout": d.get("holdout"),
        }
        print(f"[{arm['name']}] best_val={d.get('best_score')} "
              f"holdout_passed={bool(d.get('holdout') and d['holdout'].get('passed'))}",
              flush=True)

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 -> {out_path}")


if __name__ == "__main__":
    main()
