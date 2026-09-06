"""tail vs spread 全组合矩阵对比（真实 BTCUSDT，同总量 8000 根）。

三程矩阵：tail(尾部 8000) / spread vol·4 块 / spread vol·8 块（后两程作为
抽样差异的代理估计 spread 自带方差）。每程用 scripts/hold_matrix.py 跑
22 组合，产物复制到 results/matrix_8000_{tail,s4,s8}.json + 曲线侧车；
最后输出排名相关性/帕累托一致性的汇总 results/matrix_tail_vs_spread_8000.(json|md)。
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "training" / "BTCUSDT_M5.parquet"
STRAT = ROOT / "strategies" / "best_BTCUSDT.json"
LOG = ROOT / "logs" / "matrix_tail_vs_spread.log"


def run_one(tag: str, window_mode: str, chunks: int | None) -> None:
    cmd = [sys.executable, "-u", "scripts/hold_matrix.py",
           "--strategy-file", str(STRAT), "--data-file", str(DATA),
           "--window-bars", "8000", "--window-mode", window_mode,
           "--commission", "0.02", "--slippage", "0.01"]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    print(f"\n=== {tag}: {window_mode} chunks={chunks} ===", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    src = RESULTS / "hold_matrix_latest.json"
    dst = RESULTS / f"matrix_8000_{tag}.json"
    shutil.copyfile(src, dst)
    print(f"[ok] {tag} -> {dst.name}", flush=True)
    # 曲线侧车也留档，避免下一次运行覆盖（对比阶段各自留档）
    for ext in ("npz", "json"):
        csrc = RESULTS / f"hold_matrix_curves_latest.{ext}"
        if csrc.exists():
            shutil.copyfile(csrc, RESULTS / f"matrix_curves_8000_{tag}.{ext}")


def spearman(a: dict, b: dict) -> float:
    """按 combo 的 sharpe 排名计算 Spearman（并列取均值秩；缺数据跳过）。"""
    keys = [k for k in a if k in b]
    if len(keys) < 3:
        return 0.0

    def ranks(m):
        # 按 sharpe 升序排 -> 秩 1..n；并列取平均秩；None 视为最低
        order = sorted(keys, key=lambda k: (m[k].get("sharpe") is None, m[k].get("sharpe") or -9e9))
        r = {}
        i = 0
        while i < len(order):
            j = i
            v = m[order[i]].get("sharpe")
            while j + 1 < len(order) and m[order[j + 1]].get("sharpe") == v:
                j += 1
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(keys)
    d2 = sum((ra[k] - rb[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def top_n(m: dict, n: int = 3) -> list[str]:
    rank = sorted(m.get("ranking") or [], key=lambda r: -(r.get("sharpe") or -9e9))
    return [r["combo"] for r in rank if r.get("sharpe") is not None][:n]


def pareto(m: dict) -> set:
    return {(r.get("combo")) for r in (m.get("pareto_front") or [])}


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    runs = [("tail", "tail", None), ("s4", "spread", 4), ("s8", "spread", 8)]
    for tag, mode, chunks in runs:
        run_one(tag, mode, chunks)
    # ── 汇总对比 ──
    data = {tag: json.loads((RESULTS / f"matrix_8000_{tag}.json").read_text(encoding="utf-8")) for tag, _, _ in runs}
    cells = {tag: d.get("cells") or {} for tag, d in data.items()}
    combos = sorted(set().union(*[set(c.keys()) for c in cells.values()]))
    out = {"generated_at": None, "runs": {}, "per_combo": {}}
    for tag in cells:
        c = cells[tag]
        out["runs"][tag] = {
            "window_mode": data[tag].get("window_mode"),
            "window_bars": data[tag].get("window_bars"),
            "baseline_signal": data[tag].get("baseline_signal") or {},
            "top3": top_n(data[tag]),
            "pareto": sorted(pareto(data[tag])),
        }
    per = {}
    for c in combos:
        per[c] = {}
        for tag in cells:
            s = cells[tag].get(c) or {}
            per[c][tag] = {"sharpe": s.get("sharpe"), "total_return": s.get("total_return"),
                           "max_drawdown": s.get("max_drawdown"),
                           "n_trades": s.get("n_trades")}
    out["per_combo"] = per
    pairs = [("tail", "s4"), ("tail", "s8"), ("s4", "s8")]
    stats = {}
    for a, b in pairs:
        stats[f"{a}_vs_{b}"] = {
            "spearman_sharpe": round(spearman(cells[a], cells[b]), 4),
            "pareto_jaccard": round(jaccard(pareto(data[a]), pareto(data[b])), 4),
            "top3_hit": len(set(top_n(data[a])) & set(top_n(data[b]))),
            "best_combo_same": (top_n(data[a]) or [None])[0] == (top_n(data[b]) or [None])[0],
        }
    out["stats"] = stats
    out["conclusion"] = (
        "tail=单一年代(最近 8000 根)，spread=全年代按波动率分层取块(vol·4/vol·8)。"
        "若 spearman 高(>0.7)且帕累托交集大 → 组合优劣对抽样不敏感；若 tail 与 spread 分歧大而 "
        "s4/s8 彼此接近 → spread 更稳（tail 是被单一 regime 主导）；若 s4/s8 也分歧大 → spread 自身采样方差大。"
    )
    (RESULTS / "matrix_tail_vs_spread_8000.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # ── Markdown 报告 ──
    def fmt_sh(v):
        return "—" if v is None else f"{v:+.2f}"

    lines = ["# tail vs spread 全组合矩阵对比（8000 根 · 真实 BTCUSDT）", ""]
    for tag, d in data.items():
        lines += [f"## {tag}: {d.get('window_mode')} · {d.get('window_bars')} 根",
                  f"- baseline signal: 收益 {d['baseline_signal'].get('total_return')} 夏普 {d['baseline_signal'].get('sharpe')}",
                  f"- top3(夏普): {', '.join(out['runs'][tag]['top3'])}",
                  f"- 帕累托: {', '.join(out['runs'][tag]['pareto']) or '—'}", ""]
    lines += ["## 两两对比", "", "| 对 | Spearman(夏普排名) | 帕累托 Jaccard | top3 命中 | 最优组合一致 |",
              "|---|---|---|---|---|"]
    for k, v in stats.items():
        lines.append(f"| {k.replace('_vs_', ' vs ')} | {v['spearman_sharpe']} | {v['pareto_jaccard']} | "
                     f"{v['top3_hit']}/3 | {'✓' if v['best_combo_same'] else '✗'} |")
    lines += ["", "## 结论", "", out["conclusion"], ""]
    md = "\n".join(lines)
    (RESULTS / "matrix_tail_vs_spread_8000.md").write_text(md, encoding="utf-8")
    print("\n=== 完成 ===\n", md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
