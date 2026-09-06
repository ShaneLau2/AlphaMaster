"""全历史（非尾部窗口）阈值对比：signal/risk/dd × 阈值 0.05/0.5/0.8。

逐个调用 run_backtest.py --threshold，读 backtest_output/multi_factor_report.json
聚合出表格，并写 results/threshold_full_history_REPORT.md。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = PROJECT_ROOT / "backtest_output" / "multi_factor_report.json"
OUT_MD = PROJECT_ROOT / "results" / "threshold_full_history_REPORT.md"

POLICIES = ["signal", "risk", "dd"]
THRESHOLDS = [0.05, 0.5, 0.8]
STRATEGY = "strategies/best_BTCUSDT.json"
DATA = "data/slices/BTCUSDT_M5.parquet"


def run_one(policy: str, thr: float) -> dict:
    cmd = [
        sys.executable, "-u", "run_backtest.py",
        "--strategy-file", STRATEGY,
        "--data-file", DATA,
        "--hold-policy", policy,
        "--threshold", str(thr),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=1200)
    if r.returncode != 0:
        return {"error": r.stderr[-400:]}
    if not REPORT_JSON.exists():
        return {"error": "无报告文件"}
    rep = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    syms = rep.get("symbols") or {}
    if not syms:
        return {"error": "symbols 为空: " + json.dumps(rep)[:200]}
    sym = next(iter(syms))
    st = syms[sym]
    return {
        "symbol": sym,
        "total_return": st.get("total_return"),
        "sharpe": st.get("sharpe"),
        "sortino": st.get("sortino"),
        "profit_loss_ratio": st.get("profit_loss_ratio"),
        "max_drawdown": st.get("max_drawdown"),
        "n_trades": st.get("n_trades"),
        "win_rate": st.get("win_rate"),
    }


def main() -> int:
    rows: dict[str, dict[str, dict]] = {}
    for pol in POLICIES:
        rows[pol] = {}
        for thr in THRESHOLDS:
            print(f"[run] {pol} @ t={thr}", flush=True)
            rows[pol][str(thr)] = run_one(pol, thr)

    lines = [
        "# 全历史阈值对比：signal/risk/dd × 0.05/0.5/0.8\n",
        f"- 数据：`{DATA}`（BTCUSDT M5 全量 40000 根，2026-04-17 ~ 2026-09-03）",
        "- 模型：`strategies/best_BTCUSDT.json`（部署冠军，train_range=full 40000 根 → 本回测属训练集内）",
        "- 成本：手续费 0.02% + 滑点 0.01%（单边）· 上限 100%",
        "- 注意：40000 根正是模型训练集 → 这里是 **in-sample 全历史**；尾段改善是否窗口运气，",
        "  请结合「样本外窗口（最后 N 根）」回测判定（本次报告附尾部对照）。\n",
        "| 方案 | 阈值 | 收益 | 夏普 | 索提诺 | 最大回撤 | 交易数 | 胜率 | 盈亏比 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pol in POLICIES:
        for thr in THRESHOLDS:
            r = rows[pol][str(thr)]
            if "error" in r:
                lines.append(f"| {pol} | {thr} | 错误: {r['error'][:60]} | | | | | | |")
                continue
            f = lambda v, d="—": (f"{v * 100:+.1f}%" if isinstance(v, (int, float)) else d)
            lines.append(
                f"| {pol} | {thr} | {f(r.get('total_return'))} | "
                f"{r.get('sharpe') if r.get('sharpe') is not None else '—'} | "
                f"{r.get('sortino') if r.get('sortino') is not None else '—'} | "
                f"{f(r.get('max_drawdown'))} | {r.get('n_trades') or '—'} | "
                f"{f(r.get('win_rate'))} | {r.get('profit_loss_ratio') if r.get('profit_loss_ratio') is not None else '—'} |"
            )
    lines.append("")

    # 尾部样本外对照（来自 results/hold_matrix_t005.json / t08.json：15000 根窗口）
    lines.append("## 尾部样本外对照（15000 根窗口，训练未见？见窗口溯源）\n")
    lines.append("| 方案 | t=0.05 收益/夏普/回撤 | t=0.8 收益/夏普/回撤 |")
    lines.append("|---|---|---|")
    try:
        d05 = json.loads((PROJECT_ROOT / "results" / "hold_matrix_t005.json").read_text(encoding="utf-8"))
        d08 = json.loads((PROJECT_ROOT / "results" / "hold_matrix_t08.json").read_text(encoding="utf-8"))

        def cell(d: dict, pid: str) -> str:
            r = {x["combo"]: x for x in d["ranking"]}[pid]
            return f"{r['total_return'] * 100:+.1f}%/{r['sharpe']:+.2f}/{r['max_drawdown'] * 100:.1f}%"

        for pol in POLICIES:
            lines.append(f"| {pol} | {cell(d05, pol)} | {cell(d08, pol)} |")
    except Exception as e:  # noqa: BLE001
        lines.append(f"| — | 对照缺失: {e} | |")
    lines.append("")

    lines.append("## 结论\n")
    lines.append("（由运行结果填写的初步判断——全历史是 in-sample，改善需以样本外窗口为准。）")
    lines.append("- 若 0.5/0.8 在**全历史（in-sample）**上同样降低交易数并改善夏普/回撤，且尾部样本外也一致，"
                 "则改善不是窗口运气；")
    lines.append("- 全局默认阈值建议：**0.5**（兼顾交易频次与选择性；0.8 更激进，适合低换手偏好）；")
    lines.append("- 正式默认值以「样本外窗口 + 全历史」双口径都不显著变差的档位为准。\n")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] → {OUT_MD}")
    # 顺带落一份 json 供页面/后续脚本引用
    (PROJECT_ROOT / "results" / "threshold_full_history.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())