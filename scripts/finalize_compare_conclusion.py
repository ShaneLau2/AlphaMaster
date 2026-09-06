"""formal 裁决实验结论落盘：读 results/compare_latest.json → 写
results/compare_tail_vs_spread_CONCLUSION.md。

用法:
    python scripts/finalize_compare_conclusion.py            # 立即用当前 compare_latest.json
    python scripts/finalize_compare_conclusion.py --wait     # 轮询等待正式实验（steps>=150）落盘后生成

判定口径（与训练页对比面板一致）：跨 seed 中位 holdout 代表 + 闸门通过数。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATEST = PROJECT_ROOT / "results" / "compare_latest.json"
OUT = PROJECT_ROOT / "results" / "compare_tail_vs_spread_CONCLUSION.md"

MIN_STEPS = 150  # 正式裁决实验的最低步数（steps=150 才认为是正式结论）


def _fmt_pct(v) -> str:
    return f"{v * 100:+.1f}%" if isinstance(v, (int, float)) else "—"


def _fmt_sharpe(v) -> str:
    return f"{v:+.2f}" if isinstance(v, (int, float)) else "—"


def build_conclusion(d: dict) -> str:
    created = d.get("created_at", "")
    params = d.get("params") or {}
    steps = params.get("steps")
    seeds = params.get("seeds") or []
    n_bars = params.get("n_bars")
    tail = d.get("variants", {}).get("tail") or {}
    spread = d.get("variants", {}).get("spread") or {}

    rows = []
    for name, blk in (("tail", tail), ("spread", spread)):
        rep = blk.get("representative") or {}
        ho = rep.get("holdout") or {}
        med = blk.get("median_holdout")
        passed = int(blk.get("holdout_passed") or 0)
        runs = blk.get("runs") or []
        rows.append({
            "name": name,
            "n_runs": len(runs),
            "passed": passed,
            "med": med,
            "rep_holdout": ho.get("val_score"),
            "rep_sharpe": ho.get("sharpe"),
            "rep_total": ho.get("total_return_pct"),
            "rep_tag": rep.get("tag", "—"),
            "rep_ratio": ho.get("score_ratio"),
        })

    lines = []
    lines.append("# tail vs spread 正式裁决结论\n")
    lines.append(f"- 生成：{created}（本地 {datetime.now(timezone.utc).isoformat()} UTC）")
    lines.append(f"- 参数：n_bars={n_bars} · steps={steps} · seeds={seeds} · regime={params.get('regime', 'vol')}")
    lines.append(f"- 数据源：`{d.get('source_file') or d.get('params', {}).get('data_file', '—')}`")
    lines.append("- 判定口径：**跨 seed 中位 holdout val_score 代表** + 闸门通过数（与训练页对比面板一致）\n")

    lines.append("| 变体 | 运行数 | 闸门通过 | 中位 holdout | 代表 run | 代表 holdout | 代表 Sharpe | 代表收益 | 保持率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['n_runs']} | {r['passed']} | "
            f"{_fmt_sharpe(r['med'])} | {r['rep_tag']} | {_fmt_sharpe(r['rep_holdout'])} | "
            f"{_fmt_sharpe(r['rep_sharpe'])} | {_fmt_pct(r['rep_total'])} | {_fmt_sharpe(r['rep_ratio'])} |"
        )
    lines.append("")

    med_t, med_s = rows[0]["med"], rows[1]["med"]
    pass_t, pass_s = rows[0]["passed"], rows[1]["passed"]
    winner = None
    if med_t is not None and med_s is not None:
        if pass_t != pass_s:
            winner = "tail" if pass_t > pass_s else "spread"
            reason = f"闸门通过数 {pass_t} vs {pass_s}"
        elif abs(med_t - med_s) < 1e-9:
            winner, reason = "平局", "中位 holdout 相同"
        else:
            winner = "tail" if med_t > med_s else "spread"
            reason = f"中位 holdout {med_t:+.4f} vs {med_s:+.4f}"
    else:
        reason = "holdout 读数缺失"

    lines.append("## 判定\n")
    lines.append(f"- **胜出变体：{winner}**（{reason}）\n")
    lines.append("## 分 seed 明细\n")
    for name, blk in (("tail", tail), ("spread", spread)):
        lines.append(f"### {name}")
        lines.append("| seed | in-sample best | holdout | Sharpe | 保持率 | 闸门 |")
        lines.append("|---|---|---|---|---|---|")
        for run in blk.get("runs") or []:
            ho = run.get("holdout") or {}
            lines.append(
                f"| {run.get('seed')} | {_fmt_sharpe(run.get('best_score'))} | "
                f"{_fmt_sharpe(ho.get('val_score'))} | {_fmt_sharpe(ho.get('sharpe'))} | "
                f"{_fmt_sharpe(ho.get('score_ratio'))} | {'✅' if ho.get('passed') else '❌'} |"
            )
        lines.append("")

    lines.append("## 行动建议\n")
    lines.append(f"- steps={steps} 的 {sum(r['n_runs'] for r in rows)} 次运行（{len(seeds)} seeds）"
                 "已具备初步统计意义；若两路均未过闸门，结论仅作范围选择参考，不构成换冠军依据。")
    lines.append("- 用胜出变体的子集范围正式训练（n_bars 同上），走冠军闸门：holdout>0、保持率≥0.3、"
                 "Sharpe≥0、过空模型 q99 与跨折 SE 裕度后才会替换部署冠军。")
    lines.append("- 当前部署冠军（best_BTCUSDT.json）在正式训练出清闸门的替代品前保持不动。\n")
    return "\n".join(lines)


def main() -> int:
    wait = "--wait" in sys.argv
    if wait:
        print(f"[finalize] 等待正式实验（steps≥{MIN_STEPS}）落盘 compare_latest.json …", flush=True)
        while True:
            try:
                d = json.loads(LATEST.read_text(encoding="utf-8"))
                steps = (d.get("params") or {}).get("steps") or 0
                if steps >= MIN_STEPS:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(60)
        print("[finalize] 检测到正式实验结果，生成结论…", flush=True)
    else:
        d = json.loads(LATEST.read_text(encoding="utf-8"))
    txt = build_conclusion(d)
    OUT.write_text(txt, encoding="utf-8")
    print(f"[finalize] 已写入 {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())