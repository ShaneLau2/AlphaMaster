"""diag_regime_ada.py — E1 两条冠军公式在 ADAUSDT_H1 长历史上的 regime 切片分析。

目的:定位 critic 冠军公式(MA_DIFF/AMIHUD_ILLIQ/DECAY/MAX3/SIGN/SIGN/GATE)
具体在哪种 regime(趋势/震荡 × vol 高低)失效,并做家族级验证——
固定算子序列、随机化特征 token,看失效是"单条公式"还是"整族结构性问题",
以回答"是否应在采样器里禁掉这类组合"。

口径(与生产一致):
  pos = tanh(factor)(min_abs=0.05 空仓带)
  pnl = pos × target_ret − |Δpos| × cost(0.0003)
Regime 标签用【到 t 为止】的 close 收益率滚动窗口(无前视):
  rv(t) = 近 48 根 close-to-close 对数收益的 std(vol)
  er(t) = |Σr|/Σ|r| 近 120 根(效率比,0=震荡 1=单边趋势)
  dir(t)= Σr 符号(趋势方向)
分档:vol 三分位;er 三分位(低=震荡 / 高=趋势)。

用法: .venv/bin/python scripts/diag_regime_ada.py
只读,输出 JSON 到 results/,表格打到 stdout。
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.backtest import ContinuousBacktest, estimate_periods_per_year
from model_core.vocab import FORMULA_VOCAB
from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless

ADA = "data/training/ADAUSDT_H1.parquet"
COST = 0.0003

BASELINE_FML = [8, 114, 39, 121, 20, 97, 76, 126]   # feats @ 0,2,4
CRITIC_FML = [3, 41, 74, 76, 71, 8, 71, 72]         # feats @ 0,1,5

FEAT_OFFSET = FORMULA_VOCAB.operator_offset         # 65
FEAT_IDS = list(range(FEAT_OFFSET))


def feat_positions(fml: list[int]) -> list[int]:
    return [i for i, t in enumerate(fml) if t < FEAT_OFFSET]


def feature_variants(fml: list[int], k: int, rng: random.Random,
                     seed_fml: bool = True) -> list[list[int]]:
    """固定算子序列,随机替换特征 token(带 seed_fml 即包含原公式)。"""
    out = [list(fml)] if seed_fml else []
    fp = feat_positions(fml)
    while len(out) < k:
        v = list(fml)
        for p in fp:
            v[p] = rng.randrange(FEAT_OFFSET)
        out.append(v)
    return out


def pnl_series(fml: list[int], feat: torch.Tensor, t_ret: torch.Tensor,
               cost: float = COST) -> tuple[torch.Tensor | None, dict]:
    res = StackVM().execute(fml, feat)
    if res is None or float(res.std()) < 1e-4:
        return None, {"status": "const/none"}
    pos = compute_target_positions_stateless(res)
    prev = torch.roll(pos, 1, dims=1)
    prev[:, 0] = 0.0
    turnover = torch.abs(pos - prev)
    pnl = pos * t_ret - turnover * cost
    return pnl, {"pos": pos, "turnover": turnover, "factor": res}


def main() -> None:
    mgr = ParquetDataManager(ADA)
    mgr.load()
    feat = mgr.feat_tensor.to(torch.device("cpu"))
    t_ret = mgr.target_ret.to(torch.device("cpu"))
    close = mgr.raw_dict["close"].flatten().numpy().astype(np.float64)
    n = int(t_ret.shape[1])
    ppy = 8767.0
    try:
        ppy = float(estimate_periods_per_year(mgr.raw_dict["time"].flatten()))
    except Exception:
        pass
    print(f"[data] ADAUSDT bars={n} ppy={ppy:.0f} cost={COST}")

    # ── regime 标签(到 t 为止,无前视)───────────────────────────────
    lr = np.diff(np.log(close))                       # [n-1]
    W_VOL, W_ER = 48, 120
    rv = np.full(n, np.nan)
    er = np.full(n, np.nan)
    direction = np.zeros(n)
    for t in range(W_VOL, n):
        w = lr[t - W_VOL:t]
        rv[t] = w.std()
    for t in range(W_ER, n):
        w = lr[t - W_ER:t]
        s = w.sum()
        er[t] = abs(s) / (np.abs(w).sum() + 1e-12)
        direction[t] = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)

    # 打分区间:两头去掉(首段缺 regime、末两 bar target=0)
    s0 = W_ER + 1
    s1 = n - 2
    idx = np.arange(n)
    mask = (idx >= s0) & (idx < s1) & ~np.isnan(rv) & ~np.isnan(er)
    t_idx = idx[mask]
    rv_v = rv[t_idx]
    # vol / er 三分位边界(全体有效样本)
    q33, q67 = np.quantile(rv_v, [1/3, 2/3])
    er_q33, er_q67 = np.quantile(er[t_idx], [1/3, 2/3])
    print(f"[regime] vol 分位边界={q33:.5f}/{q67:.5f}  "
          f"er 边界={er_q33:.3f}/{er_q67:.3f} 有效样本={mask.sum()}")

    def vol_tier(x):
        return 0 if x <= q33 else (1 if x <= q67 else 2)

    def er_tier(x):
        return 0 if x <= er_q33 else (1 if x <= er_q67 else 2)

    vol_lbl = ["低vol", "中vol", "高vol"]
    er_lbl = ["震荡", "混合", "趋势"]
    champs = {"baseline": BASELINE_FML, "critic": CRITIC_FML}

    # ── 1) 冠军公式逐 regime 表现 ───────────────────────────────────
    out: dict = {"data": "ADAUSDT_H1", "n": n, "ppy": ppy,
                 "vol_q": [float(q33), float(q67)],
                 "er_q": [float(er_q33), float(er_q67)],
                 "champs": {}, "family": {}}
    print("\n" + "=" * 78)
    print("[1] 冠军公式 × regime 单元格 (pnl 均值 bps/bar, 括号=年化Sharpe)")
    print("=" * 78)
    pnl_cache: dict[str, torch.Tensor] = {}
    for nm, fml in champs.items():
        pnl, _ = pnl_series(fml, feat, t_ret)
        assert pnl is not None
        pnl_cache[nm] = pnl
    # 表头: 每格打印 "baseline bps | critic bps | Δ"
    hdr = "regime(vol×er)".ljust(14)
    out["cells"] = []
    for vi in range(3):
        for ei in range(3):
            sel = [vol_tier(rv[t]) == vi and er_tier(er[t]) == ei for t in t_idx]
            cnt = int(sum(sel))
            row = {"vol": vol_lbl[vi], "er": er_lbl[ei], "bars": cnt}
            line = f"{vol_lbl[vi]}+{er_lbl[ei]}".ljust(14)
            for nm in ("baseline", "critic"):
                if cnt < 30:
                    line += f"  {nm[:3]}:n/a".ljust(24)
                    continue
                p = pnl_cache[nm][0, t_idx[sel]].numpy()
                mean_bps = float(p.mean()) * 1e4
                sharpe = float(p.mean() / (p.std() + 1e-12)) * math.sqrt(ppy)
                line += f"  {nm[:4]}={mean_bps:+.1f}bps({sharpe:+.2f})".ljust(26)
                row[nm] = {"bps": round(mean_bps, 2), "sharpe": round(sharpe, 2)}
            if cnt >= 30:
                pb = pnl_cache["baseline"][0, t_idx[sel]].numpy()
                pc = pnl_cache["critic"][0, t_idx[sel]].numpy()
                db = pb.mean() - pc.mean()
                line += f"| Δ={float(db)*1e4:+.1f}bps"
                row["delta_bps"] = round(float(db) * 1e4, 2)
            out["cells"].append(row)
            print(line)
    # 汇总行(全样本)
    for nm in ("baseline", "critic"):
        p = pnl_cache[nm][0, t_idx].numpy()
        print(f"  全样本 {nm}: {float(p.mean())*1e4:+.2f}bps/bar  "
              f"sharpe={float(p.mean()/(p.std()+1e-12))*math.sqrt(ppy):+.2f}  "
              f"年化收益={float(p.mean())*ppy*100:+.1f}%")

    # ── 2) 趋势方向分层(单独看多/空趋势)──────────────────────────
    print("\n" + "=" * 78)
    print("[2] 趋势方向分层 (只取 er 上三分位=趋势区, 按方向 split)")
    print("=" * 78)
    for d, dlbl in ((1.0, "多头趋势"), (-1.0, "空头趋势"), (0.0, "无趋势")):
        sel = [er_tier(er[t]) == 2 and direction[t] == d for t in t_idx]
        cnt = int(sum(sel))
        if cnt < 30:
            continue
        line = f"  {dlbl} ({cnt}bar)".ljust(16)
        for nm in ("baseline", "critic"):
            p = pnl_cache[nm][0, t_idx[sel]].numpy()
            line += f" {nm[:4]}={float(p.mean())*1e4:+.1f}bps"
        pb = pnl_cache["baseline"][0, t_idx[sel]].numpy()
        pc = pnl_cache["critic"][0, t_idx[sel]].numpy()
        line += f" | Δ={float(pb.mean()-pc.mean())*1e4:+.1f}bps"
        print(line)

    # ── 3) 家族级验证:固定算子序列 × 随机特征 ─────────────────────
    print("\n" + "=" * 78)
    print("[3] 家族级验证 (算子序列固定, 特征随机化, 每条在全样本评分)")
    print("=" * 78)
    rng = random.Random(20260905)
    N_VAR = 60
    fam_res: dict[str, list[dict]] = {}
    for nm, base in champs.items():
        variants = feature_variants(base, N_VAR, rng, seed_fml=True)
        rows = []
        for fml in variants:
            pnl, meta = pnl_series(fml, feat, t_ret)
            if pnl is None:
                continue
            p = pnl[0, t_idx].numpy()
            sharpe = float(p.mean() / (p.std() + 1e-12)) * math.sqrt(ppy)
            ann = float(p.mean()) * ppy * 100
            # 交易特征
            posm = float(meta["pos"][0, t_idx].abs().mean())
            tovm = float(meta["turnover"][0, t_idx].mean())
            rows.append({"formula": fml, "sharpe": sharpe,
                         "ann_ret_pct": round(ann, 2),
                         "|pos|": round(posm, 3), "turnover": round(tovm, 4)})
        sharpe_v = [r["sharpe"] for r in rows]
        sharpe_v.sort()
        def pct(x):
            i = min(len(sharpe_v) - 1, int(x * len(sharpe_v)))
            return sharpe_v[i]
        pos_rate = sum(1 for r in rows if r["sharpe"] > 0) / len(rows)
        print(f"  [{nm}] n={len(rows)} sharpe 中位={np.median(sharpe_v):+.2f} "
              f"p25={pct(0.25):+.2f} p75={pct(0.75):+.2f} 最差={sharpe_v[0]:+.2f} "
              f"最好={sharpe_v[-1]:+.2f} 正分率={pos_rate:.2f}")
        print(f"        |pos|均值={np.mean([r['|pos|'] for r in rows]):.2f} "
              f"换手均值={np.mean([r['turnover'] for r in rows]):.4f}")
        fam_res[nm] = rows
    out["family"] = {
        nm: {"n": len(rows),
             "sharpe_median": float(np.median([r["sharpe"] for r in rows])),
             "sharpe_p25": float(pct(0.25)), "sharpe_p75": float(pct(0.75)),
             "pos_rate": float(pos_rate),
             "pos_mean": float(np.mean([r["|pos|"] for r in rows])),
             "turnover_mean": float(np.mean([r["turnover"] for r in rows]))}
        for nm, rows in fam_res.items()
    }

    # ── 4) 家族级 × regime: 看失效是否在特定 regime 结构性复现 ─────
    print("\n" + "=" * 78)
    print("[4] 家族级 × regime (每格=家族内变体在该格的 pnl bps 均值)")
    print("=" * 78)
    # 每个家族取 ≤20 条变体, 累加各 regime 单元格的 pnl 总和
    fam_cell_sum: dict[str, np.ndarray] = {
        nm: np.zeros(9) for nm in fam_res}
    fam_cell_n: dict[str, np.ndarray] = {
        nm: np.zeros(9) for nm in fam_res}
    sel_cache: dict[int, np.ndarray] = {}
    for vi in range(3):
        for ei in range(3):
            ci = vi * 3 + ei
            sel_cache[ci] = np.array(
                [vol_tier(rv[t]) == vi and er_tier(er[t]) == ei
                 for t in t_idx])
    for nm, rows in fam_res.items():
        for row in rows[:20]:
            pnl, _ = pnl_series(row["formula"], feat, t_ret)
            if pnl is None:
                continue
            p = pnl[0, t_idx].numpy()
            for ci in range(9):
                sel = sel_cache[ci]
                if sel.sum() < 10:
                    continue
                fam_cell_sum[nm][ci] += float(p[sel].mean()) * 1e4
                fam_cell_n[nm][ci] += 1.0
    print("家族\\regime".ljust(14),
          "".join(f"{vol_lbl[vi]}+{er_lbl[ei]}".ljust(13)
                  for vi in range(3) for ei in range(3)))
    for nm in fam_res:
        colm = []
        for ci in range(9):
            nn = fam_cell_n[nm][ci]
            colm.append(f"{fam_cell_sum[nm][ci]/nn:+.1f}" if nn >= 5
                        else "n/a")
        print(nm.ljust(14), "".join(c.ljust(13) for c in colm))

    out_path = Path("results/regime_ada_analysis.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
