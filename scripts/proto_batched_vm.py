"""
proto_batched_vm.py — 批量 VM 求值原型（同算子序列公式合并成 [B,T] 张量一次跑）。

背景：StackVM.execute 逐公式逐 token 调度（192 公式 × 8 token × 每次
torch.isnan/isinf 全量归约），Python 调度与冗余归约是主要开销。本原型把
【算子序列相同】（特征 token 可不同）的公式合并成一个 [B, T] 张量管线：

  - 特征 push：feat[0, ids, :] 一次高级索引取 B 行（每行可不同特征）；
  - 算子：同一算子只调用一次，作用于 [B, T]（绝大多数算子按行独立，天然兼容；
    CS_* 在 N=1 单品种模式下按行逐条应用，保持与 serial 完全一致）；
  - nan/inf 清洗：每算子一次 [B,T] 归约，替代 B 次 [1,T] 归约；
  - 最终 _normalize_output 逐行调用（保持 N=1 滚动 zscore 语义）。

公式集构造模拟真实训练后期（熵收敛）：5 条真实高分公式 × 每个生成若干
仅改特征 token 的变体（同算子序列 → 成簇），再补随机公式到 n。

只读：不写任何文件、不 import 训练引擎。基准与正确性对照均在本脚本内完成。

用法：
    .venv/bin/python scripts/proto_batched_vm.py [--T 49500] [--n 192] [--reps 3]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_core.ops import OPS_CONFIG
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB

torch.set_num_threads(8)
torch.set_num_interop_threads(1)

_FEAT_OFFSET = FORMULA_VOCAB.operator_offset
_OP_MAP = {i + _FEAT_OFFSET: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
_ARITY_MAP = {i + _FEAT_OFFSET: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
_CS_NAMES = {"CS_RANK", "CS_SCALE", "CS_NEUTRALIZE"}
_VOCAB_NAMES = FORMULA_VOCAB.token_names


def sample_formulas(n: int, rng: random.Random, max_len: int = 8) -> list[list[int]]:
    """随机采样合法公式（特征开局 + 算子链，栈深恒 ≥1 且结尾 =1）。"""
    formulas = []
    for _ in range(n):
        fml: list[int] = []
        depth = 0
        for step in range(max_len):
            remaining = max_len - step
            cands: list[int] = []
            for tok in range(FORMULA_VOCAB.size):
                if tok < _FEAT_OFFSET:
                    nd = depth + 1
                else:
                    a = _ARITY_MAP.get(tok, 1)
                    nd = depth + 1 - a
                if nd < 1:
                    continue
                min_future = nd + (remaining - 1) * (-2)
                max_future = nd + (remaining - 1) * 1
                if 1 < min_future or 1 > max_future:
                    continue
                cands.append(tok)
            if not cands:
                break
            tok = rng.choice(cands)
            fml.append(tok)
            depth = depth + 1 if tok < _FEAT_OFFSET else depth + 1 - _ARITY_MAP.get(tok, 1)
        if depth == 1 and len(fml) == max_len:
            formulas.append(fml)
        if len(formulas) >= n:
            break
    return formulas


def op_sequence(fml: list[int]) -> tuple[int, ...]:
    """算子序列（去掉特征 token）——批量分组的 key。"""
    return tuple(t for t in fml if t >= _FEAT_OFFSET)


def feature_variants(fml: list[int], k: int, rng: random.Random) -> list[list[int]]:
    """同一算子序列下随机替换特征 token，生成 k 个变体（真实后期熵收敛场景）。"""
    feat_pos = [i for i, t in enumerate(fml) if t < _FEAT_OFFSET]
    variants = []
    for _ in range(k):
        v = list(fml)
        for p in feat_pos:
            v[p] = rng.randrange(_FEAT_OFFSET)
        variants.append(v)
    return variants


def execute_batched(formulas: list[list[int]], feat: torch.Tensor) -> list[torch.Tensor | None]:
    """按（长度, 算子序列）分组，每组一次 [B,T] 管线求值，结果与 serial 一一对应。"""
    groups: dict[tuple, list[int]] = {}
    for i, fml in enumerate(formulas):
        groups.setdefault((len(fml), op_sequence(fml)), []).append(i)

    results: dict[int, torch.Tensor | None] = {}
    for (L, _ops), idxs in groups.items():
        rows = [formulas[i] for i in idxs]
        B = len(idxs)
        try:
            stack: list[torch.Tensor] = []
            ok = True
            for step in range(L):
                toks = [int(r[step]) for r in rows]
                if all(t < _FEAT_OFFSET for t in toks):
                    # 特征位置：一次高级索引取 B 行（每行特征可不同）→ [B, T]
                    if any(t >= feat.shape[1] for t in toks):
                        ok = False
                        break
                    stack.append(feat[0, torch.tensor(toks), :])
                elif all(t == toks[0] for t in toks) and toks[0] in _OP_MAP:
                    # 算子位置：分组保证全部相同，一次作用于 [B, T]
                    tok = toks[0]
                    arity = _ARITY_MAP[tok]
                    if len(stack) < arity:
                        ok = False
                        break
                    args = [stack.pop() for _ in range(arity)][::-1]
                    fn = _OP_MAP[tok]
                    if _VOCAB_NAMES[tok] in _CS_NAMES and B > 1:
                        # N=1 单品种 CS 算子退化为恒等/逐行语义：逐行应用保持与 serial 一致
                        per_row = []
                        for b in range(B):
                            a_args = [a[b].unsqueeze(0) for a in args]
                            per_row.append(fn(*a_args).squeeze(0))
                        res = torch.stack(per_row)
                    else:
                        res = fn(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    ok = False
                    break
            if not ok or len(stack) != 1:
                for i in idxs:
                    results[i] = None
                continue
            final = stack[0]
            norm = torch.stack([
                StackVM._normalize_output(final[b].unsqueeze(0)).squeeze(0)
                for b in range(B)
            ])
            for b, i in enumerate(idxs):
                results[i] = norm[b].unsqueeze(0)  # [1, T]，与 serial 输出形状一致
        except Exception:
            for i in idxs:
                results[i] = None

    return [results[i] for i in range(len(formulas))]


def execute_serial(formulas: list[list[int]], feat: torch.Tensor) -> list[torch.Tensor | None]:
    vm = StackVM()
    return [vm.execute(f, feat) for f in formulas]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=49500)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    from data_pipeline.parquet_manager import ParquetDataManager
    mgr = ParquetDataManager("data/baseline/BTCUSDT_H1.parquet")
    mgr.load()
    feat = mgr.feat_tensor[:, :, : args.T]  # [1, 65, T]

    rng = random.Random(42)
    # 训练日志里真实出现的高分公式（含 EMA_20/DECAY 递归算子簇）
    real = [
        [56, 94, 109, 125, 4, 119, 87, 65],
        [56, 94, 110, 125, 4, 74, 87, 115],
        [33, 98, 94, 2, 74, 87, 125, 115],
        [56, 94, 2, 74, 87, 125, 117, 115],
        [56, 94, 2, 74, 89, 125, 83, 115],
    ]
    # 每个真实公式的算子序列下生成 feature 变体（模拟后期熵收敛成簇）
    clustered = real + [v for f in real for v in feature_variants(f, 30, rng)]
    random_fml = sample_formulas(max(0, args.n - len(clustered)), rng)
    formulas = (clustered + random_fml)[: args.n]
    print(f"公式数={len(formulas)}  特征 [1,{feat.shape[1]},{args.T}]  重复运行 {args.reps} 次")

    groups: dict[tuple, int] = {}
    for f in formulas:
        groups[(len(f), op_sequence(f))] = groups.get((len(f), op_sequence(f)), 0) + 1
    sizes = sorted(groups.values(), reverse=True)
    n_clustered = min(len(clustered), args.n)
    print(f"算子序列分组: {len(groups)} 组, 最大组 {sizes[0]}, 平均 {len(formulas)/len(groups):.1f}, "
          f"组大小分布前5: {sizes[:5]}  (含真实公式+变体 {n_clustered} 条)")

    # ── 正确性 ──
    serial_res = execute_serial(formulas, feat)
    batched_res = execute_batched(formulas, feat)
    mismatch = 0
    for i, (s, b) in enumerate(zip(serial_res, batched_res)):
        if s is None or b is None:
            if s is not None or b is not None:
                mismatch += 1
                if mismatch <= 5:
                    print(f"  [i={i}] None 不一致: serial={s is not None} batched={b is not None}")
            continue
        if s.shape != b.shape or not torch.allclose(s, b, atol=1e-5, rtol=1e-4):
            mismatch += 1
            if mismatch <= 5:
                print(f"  [i={i}] 数值不一致, max|Δ|={(s - b).abs().max().item():.2e}")
    print(f"正确性: {'✅ 全部一致' if mismatch == 0 else f'❌ {mismatch} 条不一致'}")

    # ── 基准 ──
    with torch.no_grad():
        t0 = time.perf_counter(); execute_serial(formulas, feat)
        warm_ser = time.perf_counter() - t0
        t0 = time.perf_counter(); execute_batched(formulas, feat)
        warm_bat = time.perf_counter() - t0
        print(f"warmup: serial={warm_ser*1e3:.1f}ms batched={warm_bat*1e3:.1f}ms")

        best_ser = best_bat = float("inf")
        for _ in range(args.reps):
            t0 = time.perf_counter(); execute_serial(formulas, feat)
            best_ser = min(best_ser, time.perf_counter() - t0)
            t0 = time.perf_counter(); execute_batched(formulas, feat)
            best_bat = min(best_bat, time.perf_counter() - t0)
        print(f"serial  : {best_ser*1e3:8.1f} ms 总   ({best_ser/len(formulas)*1e3:.2f} ms/公式)")
        print(f"batched : {best_bat*1e3:8.1f} ms 总   ({best_bat/len(formulas)*1e3:.2f} ms/公式)")
        print(f"加速比  : {best_ser/best_bat:.2f}x")

        # EMA/DECAY 密集子集（真实高分公式几乎全含 EMA_20/DECAY 递归算子）
        ema_fml = [f for f in formulas if 94 in f or 74 in f][: 64]
        if ema_fml:
            t0 = time.perf_counter(); execute_serial(ema_fml, feat)
            s_e = time.perf_counter() - t0
            t0 = time.perf_counter(); execute_batched(ema_fml, feat)
            b_e = time.perf_counter() - t0
            print(f"EMA/DECAY 密集子集({len(ema_fml)} 公式): serial={s_e*1e3:.1f}ms "
                  f"batched={b_e*1e3:.1f}ms 加速 {s_e/b_e:.2f}x")


if __name__ == "__main__":
    main()