"""regime_health.py — 把 diag_regime_ada 的 regime 切片方法复用到任意公式集。

对每条公式在给定数据文件上输出:
  [A] 全样本生产口径 bps/sharpe/年化(pos=tanh, cost=FINALIST_COST_RATE);
  [B] vol 三段(低/中/高, 训练选优闸门同口径)的单尾 t 统计 → 覆盖是否通过;
  [C] vol×er 3×3 网格:每格 bps + 年化Sharpe + t 统计, t<-1.645 标 ✗(显著失血);
  [D] 趋势方向分层(er 上三分位内, 多/空/平)的 bps 与 t。

regime 标签与 diag_regime_ada 完全一致(因果, 无前视):
  lr = diff(log(close)); vol[t]=std(lr[t-48:t]); er[t]=|sum|/sum|.|(窗120);
  有效样本 t ∈ [W_ER+1, n-2]; 三分位边界在全体有效样本上取。

用法:
  .venv/bin/python scripts/regime_health.py \
     --files data/training/ADAUSDT_H1.parquet \
     --formulas '{"e3_direct8":[8,114,...],"e3_curriculum_p1":[...]}'
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.backtest import estimate_periods_per_year
from model_core.config import ModelConfig
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB
from strategy_manager.signal import compute_target_positions_stateless

W_VOL, W_ER = 48, 120
MIN_CELL = 200          # 网格/方向层最少样本(不足跳过该格)
MIN_T = -1.645          # "显著为负"的单尾阈值(与 vol 覆盖闸门一致)
COST = ModelConfig.FINALIST_COST_RATE


def decode(fml: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    out = []
    for t in fml:
        out.append(names[t] if 0 <= t < len(names) else f"T{t}")
    return " -> ".join(out)


def load_labels(file: str):
    mgr = ParquetDataManager(file)
    mgr.load()
    feat = mgr.feat_tensor.to(torch.device("cpu"))
    t_ret = mgr.target_ret.to(torch.device("cpu"))
    close = mgr.raw_dict.get("close")
    close = close.flatten().cpu().numpy().astype(np.float64) \
        if close is not None else None
    n = int(t_ret.shape[1])
    ppy = 8767.0
    try:
        ppy = float(estimate_periods_per_year(mgr.raw_dict["time"].flatten()))
    except Exception:
        pass
    return mgr, feat, t_ret, close, n, ppy


def regime_labels(close: np.ndarray | None, t_ret: torch.Tensor, n: int):
    """返回 rv/er/direction (因果) 与有效样本索引 t_idx。"""
    if close is not None and close.size >= n + 1:
        lr = np.diff(np.log(close))[:n]           # [n]
    else:                                          # 回退: t_ret 滚动
        r = t_ret[0].numpy().astype(np.float64)
        lr = r
    rv = np.full(n, np.nan)
    er = np.full(n, np.nan)
    direction = np.zeros(n)
    for t in range(W_VOL, n):
        w = lr[t - W_VOL:t]
        if not np.isnan(w).any():
            rv[t] = w.std()
    for t in range(W_ER, n):
        w = lr[t - W_ER:t]
        if np.isnan(w).any():
            continue
        s = w.sum()
        er[t] = abs(s) / (np.abs(w).sum() + 1e-12)
        direction[t] = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
    s0 = W_ER + 1
    s1 = n - 2
    idx = np.arange(n)
    mask = (idx >= s0) & (idx < s1) & ~np.isnan(rv) & ~np.isnan(er)
    t_idx = idx[mask]
    q = np.quantile(rv[t_idx], [1/3, 2/3])
    eq = np.quantile(er[t_idx], [1/3, 2/3])

    def tier(vals, qq):
        lo, hi = qq
        return np.where(vals <= lo, 0, np.where(vals <= hi, 1, 2))

    vt = np.full(n, -1); vt[t_idx] = tier(rv[t_idx], q)
    et = np.full(n, -1); et[t_idx] = tier(er[t_idx], eq)
    return {"rv": rv, "er": er, "direction": direction, "t_idx": t_idx,
            "vt": vt, "et": et, "vol_q": [float(q[0]), float(q[1])],
            "er_q": [float(eq[0]), float(eq[1])], "ppy": None}


def pnl_series(fml, feat, t_ret):
    res = StackVM().execute(fml, feat)
    if res is None or float(res.std()) < 1e-4:
        return None, None
    pos = compute_target_positions_stateless(res)
    prev = torch.roll(pos, 1, dims=1)
    prev[:, 0] = 0.0
    pnl = pos * t_ret - torch.abs(pos - prev) * COST
    return pnl, res


def tstat(p: np.ndarray) -> float:
    return float(p.mean() / (p.std(ddof=0) + 1e-12) * math.sqrt(max(1, p.size)))


def analyze(fml, feat, t_ret, R, ppy: float, min_t: float = -1.645) -> dict | None:
    pnl, _ = pnl_series(fml, feat, t_ret)
    if pnl is None:
        return None
    p = pnl[0].numpy()
    t_idx = R["t_idx"]
    pu = p[t_idx]
    out: dict = {
        "bps": round(float(pu.mean()) * 1e4, 3),
        "sharpe_ann": round(float(pu.mean() / (pu.std() + 1e-12))
                            * math.sqrt(ppy), 3),
        "ann_ret_pct": round(float(pu.mean()) * ppy * 100, 2),
        "vol_t": [], "vol_pass": True, "grid": [], "worst_t": 0.0,
        "worst_cell": None, "direction": [],
        "min_t": min_t,
    }
    # B: vol 三段 t(选优闸门口径, 但样本=全成熟段)
    for vi in range(3):
        sel = R["vt"][t_idx] == vi
        if sel.sum() < MIN_CELL:
            out["vol_t"].append({"tier": ["低vol", "中vol", "高vol"][vi],
                                 "bars": int(sel.sum()), "skip": True})
            continue
        q = pu[sel]
        tt = tstat(q)
        out["vol_t"].append({"tier": ["低vol", "中vol", "高vol"][vi],
                             "bars": int(sel.sum()),
                             "bps": round(float(q.mean()) * 1e4, 3),
                             "t": round(tt, 2), "pass": tt >= min_t})
        if tt < min_t:
            out["vol_pass"] = False
    # C: vol×er 9 格
    for vi in range(3):
        for ei in range(3):
            sel = (R["vt"][t_idx] == vi) & (R["et"][t_idx] == ei)
            nb = int(sel.sum())
            row = {"vol": vi, "er": ei, "bars": nb}
            if nb >= MIN_CELL:
                q = pu[sel]
                tt = tstat(q)
                row.update({"bps": round(float(q.mean()) * 1e4, 3),
                            "sharpe": round(float(q.mean() /
                                                   (q.std() + 1e-12)) *
                                            math.sqrt(ppy), 2),
                            "t": round(tt, 2), "bad": tt < min_t})
                if tt < out["worst_t"]:
                    out["worst_t"] = round(tt, 2)
                    out["worst_cell"] = f"{vi}-{ei}"
            out["grid"].append(row)
    # D: er 上三分位内的方向分层
    for d, lbl in ((1.0, "多头趋势"), (-1.0, "空头趋势"), (0.0, "无趋势")):
        sel = (R["et"][t_idx] == 2) & (R["direction"][t_idx] == d)
        nb = int(sel.sum())
        row = {"dir": lbl, "bars": nb}
        if nb >= MIN_CELL:
            q = pu[sel]
            tt = tstat(q)
            row.update({"bps": round(float(q.mean()) * 1e4, 3),
                        "t": round(tt, 2), "bad": tt < min_t})
        out["direction"].append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True,
                    help="逗号分隔 parquet 列表(每个都跑一遍)")
    ap.add_argument("--formulas", required=True,
                    help='JSON: {"label": [token,...], ...}')
    ap.add_argument("--out-json", default="results/regime_health.json")
    ap.add_argument("--min-t", type=float, default=-1.645)
    args = ap.parse_args()

    formulas: dict[str, list[int]] = json.loads(args.formulas)
    files = [f.strip() for f in args.files.split(",")]
    out: dict = {"min_t": MIN_T, "formulas": {
        lbl: {"tokens": f, "decoded": decode(f)} for lbl, f in formulas.items()},
        "files": {}}

    for file in files:
        print("=" * 84)
        print(f"文件 {Path(file).name}")
        print("=" * 84)
        mgr, feat, t_ret, close, n, ppy = load_labels(file)
        R = regime_labels(close, t_ret, n)
        print(f"  bars={n} ppy={ppy:.0f} vol_q={[round(x,5) for x in R['vol_q']]} "
              f"er_q={[round(x,3) for x in R['er_q']]} 有效样本={R['t_idx'].size}")
        fout = {}
        for lbl, fml in formulas.items():
            a = analyze(fml, feat, t_ret, R, ppy, min_t=args.min_t)
            if a is None:
                print(f"\n[{lbl}] 常量/非法公式")
                continue
            fout[lbl] = a
            print(f"\n[{lbl}] {decode(fml)}")
            print(f"  全样本: {a['bps']:+.1f}bps/bar  sharpe={a['sharpe_ann']:+.2f}  "
                  f"年化={a['ann_ret_pct']:+.1f}%")
            vp = "通过" if a["vol_pass"] else "❌拦截"
            vt = "  ".join(
                (f"{r['tier']}:t={r.get('t')}" + ("" if r.get("pass", True)
                 else "✗") if "t" in r else f"{r['tier']}:n<{MIN_CELL}")
                for r in a["vol_t"])
            print(f"  vol覆盖: {vp} | {vt}")
            wc = a["worst_cell"]
            print(f"  9格最差: t={a['worst_t']:+.2f} @格{wc} "
                  f"({'无显著失血格' if a['worst_t'] >= args.min_t else '存在显著失血格'})")
            # 9格紧凑矩阵
            print("  网格(vol\\er  bps):   震荡(0)    混合(1)    趋势(2)")
            for vi in range(3):
                cells = [r for r in a["grid"] if r["vol"] == vi]
                s = "  ".join(
                    f"{r.get('bps', float('nan')):+.1f}"
                    + ("✗" if r.get("bad") else "") for r in cells)
                print(f"   {['低vol','中vol','高vol'][vi]}        {s}")
            ds = "  ".join(
                f"{r['dir']}:{r.get('bps', float('nan')):+.1f}"
                + ("✗" if r.get("bad") else "") for r in a["direction"])
            print(f"  方向(er上段): {ds}")
        out["files"][Path(file).name] = {"ppy": ppy, "vol_q": R["vol_q"],
                                         "er_q": R["er_q"], "formulas": fout}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\n-> {args.out_json}")


if __name__ == "__main__":
    main()
