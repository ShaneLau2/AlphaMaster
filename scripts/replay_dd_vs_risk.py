"""整段历史回放：dd 回撤熔断 vs risk/be（含 signal 基线）。

对同一个真实长历史 Parquet（默认 data/training/BTCUSDT_M5.parquet 约 95 万根
M5，2017→2026 跨多轮牛熊）用同一策略因子，分别以离散撮合引擎回放
dd / risk / be / signal，量化「阶梯熔断」的：
  - 保护：最大回撤降了多少、避开了哪些大回撤段；
  - 代价：被熔断暂停期间踏空的收益、总收益/夏普差了多少；
并按日历年份 + 牛/熊 regime 分段统计，输出 json + markdown。

用法:
  python scripts/replay_dd_vs_risk.py \
      --data-file data/training/BTCUSDT_M5.parquet \
      --strategy-file strategies/best_BTCUSDT.json \
      --policies dd,risk,be,signal --out-dir results
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

import numpy as np
import torch


def _fmt_pct(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{100.0 * v:+.2f}%"


def _fmt_num(v: float | None, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def _compute_factor(data_file: Path, formula: list[int], max_seg: int = 200_000,
                    overlap: int = 1200) -> tuple[np.ndarray, np.ndarray, int]:
    """全宽特征一次算；内存不够时按重叠分块（特征最大因果回看 < overlap）。

    返回 (factor[total], bar_time_epoch[total], bars_total)。
    分块时各段因子取 [overlap:] 拼接 → 拼接后起点对应原始 bar overlap。
    """
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.features import FeatureEngineer
    from model_core.vm import StackVM

    pm = ParquetDataManager(data_file)
    pm.load()
    raw = pm.raw_dict
    close = raw["close"][0]
    T = int(close.shape[0])
    times = raw.get("time")
    t_arr = times[0].numpy().astype(np.int64) if times is not None else None

    vm = StackVM()
    seg_starts = list(range(0, T, max_seg))
    factors: list[np.ndarray] = []
    cut_each = 0
    try:
        feats = FeatureEngineer.compute_features(raw)
        with torch.no_grad():
            factor = vm.execute([int(t) for t in formula], feats)
        full = factor[0].cpu().numpy().astype(float)
        del feats, factor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return full, t_arr, T
    except RuntimeError as exc:
        if "memory" not in str(exc).lower() and "out of" not in str(exc).lower():
            raise
        print(f"[factor] 全宽内存不足（{exc}），改为重叠分块（seg={max_seg} overlap={overlap}）", flush=True)

    out = np.empty(T, dtype=np.float64)
    out[:] = np.nan
    for st in seg_starts:
        en = min(st + max_seg, T)
        seg = {k: v[:, st:en] for k, v in raw.items()}
        feats = FeatureEngineer.compute_features(seg)
        with torch.no_grad():
            factor = vm.execute([int(t) for t in formula], feats)
        f = factor[0].cpu().numpy().astype(float)
        out[st:en] = f
        del feats, factor
        print(f"[factor] segment {st}:{en} done ({len(f)} bars)", flush=True)
    # 扔掉每段前 overlap（分块边界处因果特征不完整）——第一段也丢，只保留稳定段
    valid = np.isfinite(out)
    first_keep = overlap
    keep = out[first_keep:]
    keep_t = t_arr[first_keep:] if t_arr is not None else None
    return keep, keep_t, int(keep.shape[0])


def _segments(close: np.ndarray, times: np.ndarray) -> tuple[list[tuple[str, int, int]], np.ndarray]:
    """按公历年份分段 + 每 bar 牛/熊标签（close vs 200 日均线）。"""
    years = np.array([datetime.fromtimestamp(int(t), tz=timezone.utc).year for t in times], dtype=np.int64)
    segs: list[tuple[str, int, int]] = []
    for y in sorted(set(int(v) for v in years)):
        idx = np.where(years == y)[0]
        if idx.size:
            segs.append((str(y), int(idx[0]), int(idx[-1])))
    # 200 日（约 5 个月）均线牛熊：按 M5≈288 根/日等比例折算回看根数
    dt = np.median(np.diff(times[: min(len(times), 20000)]))
    bpd = 86400.0 / dt if dt and dt > 0 else 288.0
    k = int(round(200 * bpd))
    ma = np.full(close.shape, np.nan)
    if close.shape[0] >= k:
        csum = np.cumsum(np.insert(close, 0, 0.0))
        ma[k - 1:] = (csum[k:] - csum[:-k]) / k
    bull = close >= ma
    bull = np.where(np.isnan(ma), True, bull)  # warmup 视为牛（不判）
    return segs, bull


def _regime_runs(bull: np.ndarray, times: np.ndarray, min_bars: int = 20000,
                 top: int = 8) -> list[dict]:
    """连续牛/熊段（MA200 方向连续不中断），供“多轮牛熊”逐段统计。"""
    runs: list[dict] = []
    n = bull.shape[0]
    a = 0
    while a < n:
        b = a
        while b + 1 < n and bull[b + 1] == bull[a]:
            b += 1
        if b - a + 1 >= min_bars:
            runs.append({"kind": "bull" if bool(bull[a]) else "bear",
                         "a": int(a), "b": int(b), "bars": int(b - a + 1)})
        a = b + 1
    runs.sort(key=lambda r: r["bars"], reverse=True)
    return runs[:top]


def _fmt_date(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def _trade_stats(trades: list[dict]) -> dict:
    pnls = [t["pnl"] for t in trades]
    if not pnls:
        return {"n": 0, "max_win": None, "max_single_loss": None}
    return {
        "n": len(pnls),
        "max_win": float(max(pnls)),
        "max_single_loss": float(min(pnls)),
        "avg_pnl": float(np.mean(pnls)),
    }


def _dd_gated_spans(dd_events: list[dict], T: int) -> list[dict]:
    """由 dd_events 转成“熔断区间”（gate → recover），含起止 bar/持续根/期间价格变化。"""
    spans: list[dict] = []
    open_bar: int | None = None
    for e in dd_events:
        if e["action"] in ("exit",) and open_bar is None:
            open_bar = e["bar"]
        elif e["action"] == "ok" and open_bar is not None:
            spans.append({"start": open_bar, "end": e["bar"], "bars": e["bar"] - open_bar + 1})
            open_bar = None
    if open_bar is not None:  # 熔断持续到结尾
        spans.append({"start": open_bar, "end": T - 1, "bars": T - open_bar})
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description="整段历史 dd vs risk/be 回放对比")
    ap.add_argument("--data-file", default="data/training/BTCUSDT_M5.parquet")
    ap.add_argument("--strategy-file", default="strategies/best_BTCUSDT.json")
    ap.add_argument("--policies", default="dd,risk,be,signal")
    ap.add_argument("--commission-pct", type=float, default=0.02)
    ap.add_argument("--slippage-pct", type=float, default=0.01)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    from web.hold_policy import HOLD_POLICIES, combo_id
    from web.paper_replay import run_replay

    data_file = Path(args.data_file).resolve()
    strategy_file = Path(args.strategy_file).resolve()
    policies = [combo_id(p.strip()) for p in args.policies.split(",") if p.strip()]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    meta = json.loads(strategy_file.read_text(encoding="utf-8"))
    formula = meta.get("formula")
    if not formula:
        raise SystemExit(f"策略缺少 formula: {strategy_file}")
    sym = meta.get("symbol") or strategy_file.stem.replace("best_", "", 1)

    cache_p = out_dir / "replay_dd_vs_risk_factor.npz"
    sig = f"{data_file.stat().st_mtime_ns}:{strategy_file.stat().st_mtime_ns}"
    sig_p = out_dir / "replay_dd_vs_risk_factor.sig"
    loaded_cache = False
    if cache_p.exists() and sig_p.exists() and sig_p.read_text(encoding="utf-8") == sig:
        try:
            z = np.load(cache_p)
            factor, times = z["factor"], z["times"]
            T = int(z["T"])
            print(f"[factor] 命中缓存 {T} bars", flush=True)
            loaded_cache = True
        except Exception:  # noqa: BLE001
            pass
    if not loaded_cache:
        t0 = time.time()
        factor, times, T = _compute_factor(data_file, [int(t) for t in formula])
        print(f"[factor] {T} bars 因子计算完成 ({time.time() - t0:.0f}s)", flush=True)
        try:
            np.savez_compressed(cache_p, factor=factor, times=times, T=T)
            sig_p.write_text(sig, encoding="utf-8")
        except OSError:
            pass

    from data_pipeline.parquet_manager import ParquetDataManager
    pm = ParquetDataManager(data_file)
    pm.load()
    raw = pm.raw_dict
    open_p = raw["open"][0].numpy()
    high_p = raw["high"][0].numpy()
    low_p = raw["low"][0].numpy()
    close_p = raw["close"][0].numpy()
    if times is None:
        times = raw.get("time")
        times = times[0].numpy().astype(np.int64) if times is not None else np.arange(T)

    # factor 若经分块拼接，前 overlap 根已丢弃；直接以因子为准做回放（对齐同一长度）
    L = factor.shape[0]
    if L < T:
        # 分块模式：close 等也截断到与因子对齐（取后 L 根，前 T-L 根为丢弃段）
        open_p = open_p[T - L:]
        high_p = high_p[T - L:]
        low_p = low_p[T - L:]
        close_p = close_p[T - L:]
        times = times[T - L:]
        T = L
    print(f"[replay] 对齐后 {T} 根 bar，策略 {policies} 开始回放…", flush=True)

    results: dict[str, dict] = {}
    for pid in policies:
        t1 = time.time()
        r = run_replay(factor=factor, open_p=open_p, high_p=high_p, low_p=low_p,
                       close_p=close_p, commission_pct=args.commission_pct,
                       slippage_pct=args.slippage_pct, policy_id=pid,
                       track_dd=(pid == "dd"))
        tr = _trade_stats(r["trades"])
        st = r["stats"]
        res = {
            "policy_id": pid,
            "policy_name": HOLD_POLICIES[pid.split("+")[-1]]["name"],
            "desc": HOLD_POLICIES[pid.split("+")[-1]].get("desc", ""),
            "stats": {k: (round(float(v), 6) if isinstance(v, float) else v)
                      for k, v in st.items()},
            "trades_summary": tr,
            "equity_tail": r["equity"][-5:].tolist(),  # 只存少量曲线锚点，避免 json 过大
        }
        results[pid] = res
        print(f"[replay] {pid}: ret={st['total_return']:.4f} sharpe={st['sharpe']} "
              f"dd={st['max_drawdown']:.4f} trades={st['n_trades']} "
              f"worst={tr['max_single_loss']} ({time.time() - t1:.0f}s)", flush=True)

    dd_res = results.get("dd")
    signal_res = results.get("signal")

    # ── 每策略整段 equity（内存可控 1×T 浮点）──
    segs, bull = _segments(close_p, times)
    equity_all: dict[str, np.ndarray] = {}
    for pid, res in results.items():
        t1 = time.time()
        r = run_replay(factor=factor, open_p=open_p, high_p=high_p, low_p=low_p,
                       close_p=close_p, commission_pct=args.commission_pct,
                       slippage_pct=args.slippage_pct, policy_id=pid,
                       track_dd=(pid == "dd"))
        equity_all[pid] = r["equity"]
        print(f"[equity] {pid} 曲线缓存完成 ({time.time() - t1:.0f}s)", flush=True)

    def _seg_metrics(eq: np.ndarray, a: int, b: int) -> dict:
        s = max(1, a)
        sub = eq[s:b + 1]
        if sub.size == 0:
            return {"ret": 0.0, "max_dd": 0.0}
        ret = float(sub[-1] - sub[0])
        pk = np.maximum.accumulate(sub)
        pk_safe = np.where(pk > 0.0, pk, 1.0)
        mdd = float((sub / pk_safe - 1.0).min())
        return {"ret": round(ret, 6), "max_dd": round(mdd, 6)}

    regime_seg: dict[str, dict] = {}
    for name, a, b in segs:
        regime_seg[name] = {}
        # 牛/熊按该段内 bull 占比
        seg_bull = float(bull[a:b + 1].mean())
        regime_seg[name]["bull_frac"] = round(seg_bull, 3)
        for pid, eq in equity_all.items():
            regime_seg[name][pid] = _seg_metrics(eq, a, b)

    regime_runs = _regime_runs(bull, times)
    regime_out: list[dict] = []
    for run in regime_runs:
        row = {"kind": run["kind"], "a": run["a"], "b": run["b"], "bars": run["bars"],
               "start_date": _fmt_date(int(times[run["a"]])),
               "end_date": _fmt_date(int(times[min(run["b"], len(times) - 1)]))}
        row["per_policy"] = {}
        for pid, eq in equity_all.items():
            row["per_policy"][pid] = _seg_metrics(eq, run["a"], run["b"])
        regime_out.append(row)

    # dd 熔断代价/保护
    dd_extra: dict = {}
    if "dd" in results:
        r_dd = run_replay(factor=factor, open_p=open_p, high_p=high_p, low_p=low_p,
                          close_p=close_p, commission_pct=args.commission_pct,
                          slippage_pct=args.slippage_pct, policy_id="dd",
                          track_dd=True)
        ev = r_dd.get("dd_events") or []
        spans = _dd_gated_spans(ev, T)
        n_gate = sum(1 for e in ev if e["action"] == "exit")
        gated_bars = sum(s["bars"] for s in spans)
        # 深档/普通档细分
        deep_gates = sum(1 for e in ev if e["action"] == "exit" and e["state"] == 2)
        dd_extra = {
            "n_gate_events": n_gate,
            "n_deep_gate_events": deep_gates,
            "gated_bars": gated_bars,
            "gated_frac": round(gated_bars / T, 5),
            "n_gated_spans": len(spans),
            "avg_span_bars": round(gated_bars / len(spans), 1) if spans else 0,
            "sample_spans": spans[:12],
        }
        # 保护 vs 代价（相对 signal）
        if "signal" in results:
            s_stats = results["signal"]["stats"]
            d_stats = results["dd"]["stats"]
            dd_extra["vs_signal"] = {
                "max_dd_improve_pp": round((s_stats["max_drawdown"] - d_stats["max_drawdown"]) * 100.0, 3),
                "total_return_diff_pp": round((d_stats["total_return"] - s_stats["total_return"]) * 100.0, 3),
                "sharpe_diff": round(d_stats["sharpe"] - s_stats["sharpe"], 3),
                "n_trades_diff": d_stats["n_trades"] - s_stats["n_trades"],
            }

    summary = {
        "kind": "replay_dd_vs_risk",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_file": str(data_file),
        "strategy_file": str(strategy_file),
        "symbol": sym,
        "bars": T,
        "params": {"commission_pct": args.commission_pct, "slippage_pct": args.slippage_pct},
        "results": {pid: {k: v for k, v in res.items() if k != "_equity"} for pid, res in results.items()},
        "per_year": regime_seg,
        "regime_runs": regime_out,
        "dd_extra": dd_extra,
    }
    jpath = out_dir / f"replay_dd_vs_risk_{ts}.json"
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "replay_dd_vs_risk_latest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Markdown ──────────────────────────────────────────────
    md = [f"# 整段历史回放：dd 回撤熔断 vs risk / be（{sym}）", "",
          f"- 数据：`{data_file}`（{T:,} 根）", f"- 策略：`{strategy_file}`",
          f"- 成本：手续费 {args.commission_pct}% / 滑点 {args.slippage_pct}%",
          f"- 生成：{datetime.now(timezone.utc).isoformat()}", ""]
    md += ["| 方案 | 总收益 | 夏普 | 最大回撤 | 交易数 | 胜率 | 盈亏比 | 最大单笔亏损 |", "|---|---|---|---|---|---|---|---|"]
    for pid, res in results.items():
        st = res["stats"]
        t = res["trades_summary"]
        win = st.get("win_rate")
        pl = st.get("profit_loss_ratio")
        md.append(f"| {res['policy_name']}（{pid}） | {_fmt_pct(st.get('total_return'))} | "
                  f"{_fmt_num(st.get('sharpe'), 2)} | {_fmt_pct(st.get('max_drawdown'))} | "
                  f"{st.get('n_trades')} | {_fmt_pct(win) if win is not None else '—'} | "
                  f"{_fmt_num(pl, 2) if pl is not None else '—'} | {_fmt_pct(t.get('max_single_loss'))} |")
    md += [""]
    if dd_extra:
        v = dd_extra.get("vs_signal") or {}
        md += ["## dd 阶梯熔断的代价与保护（相对 signal 基线）", "",
               f"- 熔断触发 **{dd_extra['n_gate_events']}** 次（深档 {dd_extra['n_deep_gate_events']} 次），"
               f"累计熔断暂停 **{dd_extra['gated_bars']:,}** 根 bar"
               f"（占回放期 {100.0 * dd_extra['gated_frac']:.2f}%），共 {dd_extra['n_gated_spans']} 段，"
               f"平均每段 {dd_extra['avg_span_bars']} 根。", ""]
        md += ["| 相对 signal | 最大回撤改善(pp) | 总收益差(pp) | 夏普差 | 交易数差 |",
               "|---|---|---|---|---|",
               f"| dd | {v.get('max_dd_improve_pp', '—')} | {v.get('total_return_diff_pp', '—')} | "
               f"{v.get('sharpe_diff', '—')} | {v.get('n_trades_diff', '—')} |", ""]
        if dd_extra.get("sample_spans"):
            md += ["最近几次熔断区间（bar 起止 · 持续根数）:", ""]
            for s in dd_extra["sample_spans"][-6:]:
                y0 = datetime.fromtimestamp(int(times[s["start"]]), tz=timezone.utc).strftime("%Y-%m-%d")
                y1 = datetime.fromtimestamp(int(times[min(s["end"], len(times) - 1)]), tz=timezone.utc).strftime("%Y-%m-%d")
                md.append(f"- `{s['start']}..{s['end']}`（{s['bars']:,} 根）≈ {y0} → {y1}")
            md.append("")

    hdr2 = "| 年份 | bull占比 |"
    sep2 = "|---|---|"
    for pid in policies:
        hdr2 += f" {results[pid]['policy_name']} 收益 | {results[pid]['policy_name']} 回撤 |"
        sep2 += "|---|"
    md += ["## 按年份分段（收益 / 最大回撤）", "", hdr2, sep2]
    for name in sorted(regime_seg.keys()):
        row = f"| {name} | {regime_seg[name]['bull_frac']} |"
        for pid in policies:
            m = regime_seg[name].get(pid) or {}
            row += f" {_fmt_pct(m.get('ret'))} | {_fmt_pct(m.get('max_dd'))} |"
        md.append(row)
    md += ["", "## 连续牛/熊段（MA200，按最长 8 段，每段 ≥2 万根 ≈ 2.5 个月）", "",
           "| 段 | 区间 | 方案 | 区间收益 | 最大回撤 |", "|---|---|---|---|---|"]
    for run in regime_out:
        row_kind = f"**{run['kind']}**"
        for pid in policies:
            m = run["per_policy"].get(pid) or {}
            md.append(f"| {row_kind} | {run['start_date']} → {run['end_date']} | "
                      f"{results[pid]['policy_name']} | {_fmt_pct(m.get('ret'))} | {_fmt_pct(m.get('max_dd'))} |")
            row_kind = run["kind"]
    md += ["", "> 口径：离散撮合（模拟实盘同引擎，固定名义本金 1.0/笔，累计可超 ±100%）；signal=无风控基线；"
              "dd 熔断不设冷却，收复后需方向翻转才重开。本回放把 2026 年窗训练出的因子整体回放回 2017-2026，"
              "年代跨域本身会主导盈亏；方案间相对差异看 最大回撤/单笔最大亏损 的保护层效果更有意义。"]
    mpath = out_dir / f"replay_dd_vs_risk_{ts}.md"
    mpath.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
