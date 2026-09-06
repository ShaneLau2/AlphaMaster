"""diag_exp1_champions.py — E1 复盘诊断：critic 冠军 val 更高却 holdout 失败，
是「单尾噪声」还是「critic 系统性偏向折内拟合」？

方法（全部只读，不写文件、不训练）：
  1. 用训练同口径（_eval_formula_task 的逐折逻辑：evaluate_fold + IC gate +
     REWARD_ALPHA）在训练文件上重算两条冠军公式的 per-fold train/val；
  2. 复刻 _verify_holdout 的生产口径（tanh 仓位 + 成本 + 年化），验证能还原
     JSON 里的 holdout val_score/sharpe（口径校验）；
  3. 交叉验证：把两条公式拿到【未参与任何训练】的其他品种/周期文件上，
     在大量不重叠 500-bar OOS 窗口上打分 → 比较分布、配对差、正分率；
  4. 附加：同一日历尾部时间段在 ADA/BTC 上的同周期跨品种对照。

用法： .venv/bin/python scripts/diag_exp1_champions.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.backtest import ContinuousBacktest, estimate_periods_per_year
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine, _build_walk_forward_folds
from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless

RESULTS = Path("results")
TRAIN_FILE = Path(
    "data/training/BINANCE_BTCUSDT_H1.parquet")
X_FILES = [
    "data/training/ADAUSDT_H1.parquet",
    "data/training/BTCUSDT_H1.parquet",
    "data/training/600519_D1.parquet",
]

BASELINE_FML = [8, 114, 39, 121, 20, 97, 76, 126]
CRITIC_FML = [3, 41, 74, 76, 71, 8, 71, 72]


def load(name: str, path: str):
    mgr = ParquetDataManager(path)
    mgr.load()
    feat = mgr.feat_tensor.to(ModelConfig.DEVICE)
    t_ret = mgr.target_ret.to(ModelConfig.DEVICE)
    bt = ContinuousBacktest()
    time = mgr.raw_dict.get("time")
    if time is not None:
        try:
            bt.periods_per_year = estimate_periods_per_year(time)
        except Exception:
            pass
    print(f"  [{name}] bars={int(t_ret.shape[1])} ppy={bt.periods_per_year}")
    return mgr, feat, t_ret, bt


def prod_tail_stats(fml, feat, t_ret, bt, start, end):
    """复刻 _verify_holdout 生产口径：tanh 仓位 + 成本 + 年化 Sharpe。"""
    res = StackVM().execute(fml, feat)
    if res is None or float(res.std()) < 1e-4:
        return None
    pos = compute_target_positions_stateless(res)
    prev = torch.roll(pos, 1, dims=1)
    prev[:, 0] = 0.0
    turnover = torch.abs(pos - prev)
    pnl = pos * t_ret - turnover * bt.cost_rate
    pnl_h = pnl[:, start:end]
    _, ho_score = bt.evaluate_fold(res, t_ret, start, end, start, end)
    ic_ho, _ = AlphaEngine._compute_ic(res[:, start:end], t_ret[:, start:end])
    ho_adj = float(AlphaEngine._apply_ic_gate(ho_score, ic_ho))
    sharpe = float(
        pnl_h.mean() / (pnl_h.std() + 1e-9)
        * math.sqrt(max(1.0, float(bt.periods_per_year)))
    )
    return {"ho_adj": ho_adj, "sharpe": sharpe,
            "ret_pct": float(pnl_h.sum()) * 100.0}


def wf_per_fold(fml, feat, t_ret, folds, bt):
    """复刻 _eval_formula_task 逐折逻辑（train=REWARD_ALPHA×IC门控，val=IC门控）。"""
    res = StackVM().execute(fml, feat)
    if res is None or float(res.std()) < 1e-4:
        return None
    rows = []
    for fold in folds:
        tr_sc, vl_sc = bt.evaluate_fold(
            res, t_ret,
            fold["train_start"], fold["train_end"],
            fold["val_start"], fold["val_end"],
        )
        ic_m, _ = AlphaEngine._compute_ic(
            res[:, fold["train_start"]:fold["train_end"]],
            t_ret[:, fold["train_start"]:fold["train_end"]],
        )
        tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
        ic_v, _ = AlphaEngine._compute_ic(
            res[:, fold["val_start"]:fold["val_end"]],
            t_ret[:, fold["val_start"]:fold["val_end"]],
        )
        vl_adj = AlphaEngine._apply_ic_gate(vl_sc, ic_v)
        rows.append({
            "fold": f"{fold['train_start']}-{fold['train_end']}|"
                    f"{fold['val_start']}-{fold['val_end']}",
            "train": float(ModelConfig.REWARD_ALPHA * tr_adj),
            "val": float(vl_adj),
        })
    return rows


def main() -> None:
    b = json.load(open(RESULTS / "exp1_base_200_s42.json"))
    c = json.load(open(RESULTS / "exp1_critic_200_s42.json"))
    champs = {"baseline": b["best_formula"], "critic": c["best_formula"]}

    # ── 1) 训练文件逐折重算 ─────────────────────────────────────────
    print("=" * 70)
    print("[1] 训练文件 per-fold 重算 (walk-forward, 训练同口径)")
    print("=" * 70)
    _, feat, t_ret, bt = load("train", str(TRAIN_FILE))
    n_time = int(t_ret.shape[1])
    T = n_time - ModelConfig.HOLDOUT_BARS          # 5373
    folds = _build_walk_forward_folds(T, 5, gap=ModelConfig.WF_GAP)
    print(f"    T={T} 折数={len(folds)} gap={folds[0]['gap'] if folds else '-'}")

    for name, fml in champs.items():
        rows = wf_per_fold(fml, feat[:, :, :T], t_ret[:, :T], folds, bt)
        if rows is None:
            print(f"  {name}: 公式执行异常/常量"); continue
        vals = [r["val"] for r in rows]
        trs = [r["train"] for r in rows]
        mean, sd = sum(vals) / len(vals), torch.std(torch.tensor(vals)).item()
        print(f"  [{name}] 折内train 均值={sum(trs)/len(trs):+.3f} "
              f"逐折={[round(x,2) for x in trs]}")
        print(f"  [{name}] OOF val  均值={mean:+.3f} ± {sd:.3f} (SE={sd/math.sqrt(len(vals)):.3f}) "
              f"逐折={[round(x,2) for x in vals]}  最差折={min(vals):+.3f}")
        r = prod_tail_stats(fml, feat, t_ret, bt, n_time - 500, n_time - 2)
        if r:
            print(f"  [{name}] 复刻holdout: ho_adj={r['ho_adj']:+.4f} "
                  f"sharpe={r['sharpe']:+.3f}")

    # ── 2) 口径校验：与 JSON 记录对比 ───────────────────────────────
    print("=" * 70)
    print("[2] 口径校验 (复刻 vs JSON 记录)")
    print("=" * 70)
    for name, d in (("baseline", b), ("critic", c)):
        fml = d["best_formula"]
        r = prod_tail_stats(fml, feat, t_ret, bt, n_time - 500, n_time - 2)
        if r:
            print(f"  [{name}] 复刻 ho_adj={r['ho_adj']:+.4f} sharpe={r['sharpe']:+.3f}"
                  f" | JSON ho_adj={d['holdout']['val_score']:+.4f} "
                  f"sharpe={d['holdout']['sharpe']:+.3f}")

    # ── 3) 跨品种大量 OOS 窗口分布 ─────────────────────────────────
    print("=" * 70)
    print("[3] 跨品种 OOS 500-bar 窗口分布 (未参与训练的文件)")
    print("=" * 70)
    for xf in X_FILES:
        _, feat_x, t_ret_x, bt_x = load(xf.split("/")[-1], xf)
        n = int(t_ret_x.shape[1])
        w = 500
        n_max = min(n - 2, w)
        step = max(1, (n - 2 - w) // 24)          # ≤25 个不重叠窗口, 均匀铺开
        starts = list(range(0, n - 2 - w + 1, step))[:25]
        print(f"  {xf.split('/')[-1]} 窗口数={len(starts)} "
              f"区间=[{starts[0]},{starts[-1]+w})")
        per = {nm: [] for nm in champs}
        for s in starts:
            e = min(s + w, n - 2)
            for nm, fml in champs.items():
                r = prod_tail_stats(fml, feat_x, t_ret_x, bt_x, s, e)
                per[nm].append(r["sharpe"] if r else float("-inf"))
        for nm in champs:
            v = [x for x in per[nm] if x != float("-inf")]
            import statistics
            pos = sum(1 for x in v if x > 0) / max(1, len(v))
            print(f"    [{nm}] sharpe 均值={statistics.mean(v):+.3f} "
                  f"中位={statistics.median(v):+.3f} 最差={min(v):+.3f} "
                  f"最差窗口start={starts[v.index(min(v))] if v else '-'} "
                  f"正分率={pos:.2f}")
        # 配对差
        diffs = [per["baseline"][i] - per["critic"][i]
                 for i in range(len(starts))]
        import statistics
        n_pos = sum(1 for d in diffs if d > 0)
        print(f"    配对(baseline−critic) 差均值={statistics.mean(diffs):+.3f} "
              f"baseline更优窗口={n_pos}/{len(diffs)}")

    # ── 4) 同日历尾部、跨品种对照（与 E1 失败窗口同时段）─────────────
    print("=" * 70)
    print("[4] 同日历尾部跨品种 (训练文件最后500根的同一时间段)")
    print("=" * 70)
    _tm, _, _, _ = load("train-time", str(TRAIN_FILE))
    tt_full = _tm.raw_dict.get("time")
    tt_flat = tt_full.flatten() if tt_full is not None else None
    if tt_flat is None or tt_flat.numel() < 500:
        print("  [跳过] 训练文件无 time 轴")
        return
    t0 = int(tt_flat[-500])
    for xf in X_FILES:
        m = ParquetDataManager(xf)
        m.load()
        tt = m.raw_dict.get("time")
        tt_flat = tt.flatten() if tt is not None else None
        if tt_flat is None:
            print(f"  {xf.split('/')[-1]}: 无 time 轴"); continue
        sel = [i for i in range(tt_flat.numel()) if int(tt_flat[i]) >= t0]
        if not sel:
            print(f"  {xf.split('/')[-1]}: 无重叠时间"); continue
        s, e = sel[0], min(sel[-1], tt_flat.numel() - 2) + 1
        if e - s < 100:
            print(f"  {xf.split('/')[-1]}: 重叠过短({e-s}根), 跳过"); continue
        ts0 = int(tt_flat[s])
        print(f"  {xf.split('/')[-1]}: 重叠 {e-s} 根 ({s}..{e}) "
              f"起始ts={ts0}")
        feat_x = m.feat_tensor.to(ModelConfig.DEVICE)
        t_ret_x = m.target_ret.to(ModelConfig.DEVICE)
        bt_x = ContinuousBacktest()
        try:
            bt_x.periods_per_year = estimate_periods_per_year(tt)
        except Exception:
            pass
        for nm, fml in champs.items():
            r = prod_tail_stats(fml, feat_x, t_ret_x, bt_x, s, min(e, len(tt) - 2))
            if r:
                print(f"    [{nm}] ho_adj={r['ho_adj']:+.3f} sharpe={r['sharpe']:+.3f}")


if __name__ == "__main__":
    main()
