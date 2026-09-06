"""verify_batched_equiv.py — execute_batched 接入后的等价性对照。

在【同一份数据、同一 seed】下各起一个全新 AlphaEngine，分别以
BATCHED_EVAL_ENABLED=False（逐条串行）与 True（execute_batched 批量）跑
恰好 1 个训练步（192 新公式 + 精英回放，walk-forward 折内评分全走真实
Part C / _eval_formula_task 路径），对比：

  1. 两边评估的公式集合是否完全一致（seed 确定性的 sanity）；
  2. 每条公式的 status（ok/const/none/error）是否一致；
  3. ok 公式的 reward / val_score / ic_full / ic_i 是否一致；
  4. 批量路径产出的因子张量 res 与逐条路径是否逐元素一致。

方法不复制引擎内部逻辑：step-0 采样由 torch seed 决定，两个引擎在第一步
的采样、精英池、因子池初始状态完全相同，唯一差别就是 Part C 的求值路径。
所有 run 用 run_tag 隔离 + _suppress_deploy，不写冠军/holdout/指纹。

用法： .venv/bin/python scripts/verify_batched_equiv.py
      [--data-file data/training/BINANCE_BTCUSDT_H1.parquet] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine


def run_arm(data_file: str, batched: bool, seed: int, steps: int = 1,
            tag: str = "btv") -> dict:
    """起一个全新 engine，跑 steps 步，捕获每步每条公式的评估结果。"""
    old_flag = getattr(ModelConfig, "BATCHED_EVAL_ENABLED", False)
    ModelConfig.BATCHED_EVAL_ENABLED = batched
    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        eng = AlphaEngine(data_manager=mgr, target_symbol=None, seed=seed)
        eng.run_tag = tag
        eng._suppress_deploy = True          # 不写冠军/历史/消费 holdout
        eng.n_folds = 5                      # 与正式训练一致的 4~5 折口径

        captured: dict[tuple, dict] = {}
        order: list[tuple] = []
        orig_task = eng._eval_formula_task

        def wrapped(idx, fml, feat, t_ret, folds, use_wf, fps, **kw):
            r = orig_task(idx, fml, feat, t_ret, folds, use_wf, fps, **kw)
            key = tuple(r["fml"])
            rec = dict(r)
            res_t = rec.get("res")
            rec["res"] = None if res_t is None else res_t.detach().clone()
            if key not in captured:
                order.append(key)
            captured[key] = rec
            return r

        eng._eval_formula_task = wrapped          # type: ignore[method-assign]
        try:
            eng.train(start_step=0, end_step=steps, verbose_header=False)
        finally:
            eng._eval_formula_task = orig_task    # type: ignore[method-assign]
        return {"captured": captured, "order": order,
                "engine": eng, "flag": old_flag}
    finally:
        ModelConfig.BATCHED_EVAL_ENABLED = old_flag


def fmt_status(r: dict) -> str:
    if r.get("status") == "ok":
        return (f"ok  reward={r['reward']:+.4f} val={r['val_score']:+.4f} "
                f"ic={r['ic_full']:+.4f} ic_i={r['ic_i']:+.4f}")
    return r.get("status", "?") + (f" ({r.get('error')})"
                                   if r.get("error") else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="data/training/BINANCE_BTCUSDT_H1.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=1)
    args = ap.parse_args()

    print("=" * 74)
    print(f"[等价对照] file={args.data_file} seed={args.seed} steps={args.steps}")
    print("=" * 74)

    print("\n[arm 1/2] 逐条串行 (BATCHED_EVAL_ENABLED=False) ...", flush=True)
    s = run_arm(args.data_file, batched=False, seed=args.seed,
                steps=args.steps, tag="btv_serial")
    print("[arm 2/2] 批量 (BATCHED_EVAL_ENABLED=True) ...", flush=True)
    b = run_arm(args.data_file, batched=True, seed=args.seed,
                steps=args.steps, tag="btv_batched")

    sc, bc = s["captured"], b["captured"]
    sok = {k for k, v in sc.items() if v.get("status") == "ok"}
    bok = {k for k, v in bc.items() if v.get("status") == "ok"}
    sst = {k: v.get("status") for k, v in sc.items()}
    bst = {k: v.get("status") for k, v in bc.items()}

    print("\n" + "=" * 74)
    print("对比汇总")
    print("=" * 74)
    n = max(len(sc), len(bc))
    same_set = set(sc) == set(bc)
    same_st = sst == bst
    print(f"公式总数        : serial={len(sc)}  batched={len(bc)}  "
          f"集合一致={same_set}")
    if not same_set:
        print(f"  仅serial有: {sorted(set(sc) - set(bc))[:5]} ...")
        print(f"  仅batched有: {sorted(set(bc) - set(sc))[:5]} ...")
    print(f"status 一致     : {same_st}")
    if not same_st:
        for k in sorted(set(sst) | set(bst)):
            if sst.get(k) != bst.get(k):
                print(f"  {k}: serial={sst.get(k)} batched={bst.get(k)}")
    print(f"ok 公式         : serial={len(sok)}  batched={len(bok)}")
    print(f"非ok            : "
          f"serial={ {st: sum(1 for v in sst.values() if v==st) for st in set(sst.values()) - {'ok'}} }  "
          f"batched={ {st: sum(1 for v in bst.values() if v==st) for st in set(bst.values()) - {'ok'}} }")

    # ── 逐条数值对照 ─────────────────────────────────────────────
    mism = []           # reward/val/ic 不一致
    tens_mism = []      # res 张量不一致
    max_td = 0.0        # 张量最大绝对差
    n_cmp = 0
    for k in sorted(sok & bok, key=lambda x: str(x)):
        rs, rb = sc[k], bc[k]
        n_cmp += 1
        ok_num = (abs(rs["reward"] - rb["reward"]) < 1e-4
                  and abs(rs["val_score"] - rb["val_score"]) < 1e-4
                  and abs(rs["ic_full"] - rb["ic_full"]) < 1e-4
                  and abs(rs["ic_i"] - rb["ic_i"]) < 1e-4)
        if not ok_num:
            mism.append((k, rs, rb))
        a, bb = rs["res"], rb["res"]
        if a is not None and bb is not None and a.shape == bb.shape:
            d = (a - bb).abs().max().item()
            max_td = max(max_td, float(d))
            if not torch.allclose(a, bb, atol=1e-6, rtol=1e-5):
                tens_mism.append((k, float(d)))
        elif (a is None) != (bb is None):
            tens_mism.append((k, float("nan")))

    print(f"\n数值一致 ok 公式 : {n_cmp - len(mism)}/{n_cmp}  "
          f"(reward/val/ic 容差 1e-4)")
    if mism:
        print("  不一致样例:")
        for k, rs, rb in mism[:8]:
            print(f"  {list(k)}:\n    serial  : {fmt_status(rs)}\n    batched : {fmt_status(rb)}")
    print(f"res 张量一致     : {n_cmp - len(tens_mism)}/{n_cmp}  "
          f"(allclose atol=1e-6 rtol=1e-5)  最大|Δ|={max_td:.2e}")
    if tens_mism:
        print("  不一致样例:")
        for k, d in tens_mism[:8]:
            print(f"  {list(k)}: max|Δ|={d:.2e}")

    # ── 引擎外部守卫：批量路径是否真的没走逐条 execute ──────────
    # 方法被替换后无法直接计数；用最弱断言——两条 arm 的最终 best 相同
    # （1 步内通常无 best 更新，此处仅提示）。
    print("\n[注] 两边 best_score/best_formula：")
    for nm, arm in (("serial", s), ("batched", b)):
        e = arm["engine"]
        print(f"  {nm:8s}: best_score={getattr(e, 'best_score', None)} "
              f"best_formula={getattr(e, 'best_formula', None)}")

    ok = (same_set and same_st and not mism and not tens_mism)
    print("\n" + "=" * 74)
    print(f"结论: {'✅ 等价（批量与逐条在 192×4折 口径上无回归）' if ok
          else '❌ 存在不一致，需排查'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
