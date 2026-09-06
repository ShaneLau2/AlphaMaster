"""tail vs spread 对比实验 → docs 报告 与 回测页摘要。

数据源：results/compare_latest.json（compare_ranges.py 每次跑完写入）。
- write_docs_report(summary)：把两路曲线 best/holdout 结论写成 docs 报告
  （docs/compare_tail_vs_spread_REPORT.md，另存时间戳副本）。
- load_latest() / summary_for_symbol()：供回测页 /api/backtest/compare-summary 读取。
- 诚实标注：steps<100 视为“迷你/演示口径”，不当作正式选型结论。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_NAME = "compare_tail_vs_spread_REPORT.md"
MINI_STEPS = 100  # <100 步的短训视为迷你口径


def _clean(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.loads(json.dumps(v, ensure_ascii=False, default=str))
    return v


def load_latest() -> tuple[dict | None, Path | None]:
    """读 compare_latest.json；缺失时回退到 results/ 里最新 compare_*.json。"""
    latest = RESULTS_DIR / "compare_latest.json"
    if latest.exists():
        try:
            return json.loads(latest.read_text(encoding="utf-8")), latest
        except (OSError, json.JSONDecodeError):
            pass
    cands = sorted(RESULTS_DIR.glob("compare_*.json")) if RESULTS_DIR.exists() else []
    cands = [p for p in cands if p.name != "compare_latest.json"]
    if not cands:
        return None, None
    path = cands[-1]
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (OSError, json.JSONDecodeError):
        return None, None


def summary_symbol(summary: dict | None) -> str | None:
    if not summary:
        return None
    variants = summary.get("variants") or {}
    for key in ("tail", "spread"):
        b = (variants.get(key) or {}).get("best") or {}
        if b.get("symbol"):
            return b["symbol"]
    src = str(summary.get("source_file") or "")
    if src:
        stem = Path(src).stem
        for tf in ("M1", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1"):
            if f"_{tf}" in src:
                return stem.split(f"_{tf}")[0]
    return None


def summary_for_symbol(symbol: str | None) -> dict | None:
    """回测页按品种取：symbol 给定时必须匹配；否则返回 None。"""
    summary, _ = load_latest()
    sym = summary_symbol(summary)
    if symbol and sym and sym != symbol:
        return None
    return summary


# ── 结论代表口径：跨 seed 中位 holdout ─────────────────────────────────
# 目的：不用「in-sample best」挑代表（那会挑中样本外彩票 seed），
# 而是按各 seed 的 holdout val_score 取中位数位置的那次 run 作代表。


def pick_best(runs: list[dict]) -> dict | None:
    """跨 seed 挑 best_score 最高的那次（仅用于展示 in-sample best 曲线）。"""
    best = None
    for r in runs:
        if r.get("best_score") is None:
            continue
        if best is None or r["best_score"] > best["best_score"]:
            best = r
    return best or (runs[0] if runs else None)


def _holdout_val(r: dict) -> float:
    ho = r.get("holdout") or {}
    v = ho.get("val_score")
    try:
        return float(v) if v is not None else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


def pick_representative(runs: list[dict]) -> dict | None:
    """结论代表：取 holdout val_score 处于中位数位置的那次 run。

    规则：
    - holdout 缺失（None）的 run 视为 -inf；全部缺失时回退 in-sample best 口径；
    -    奇数个：取正中间那次；
    - 偶数个：取「离真中位数（中间两值均值）最近」的那次 run，距离并列（即中位数
      恰为两点中点）时**不选无代表 run 的一侧**——若其中一侧为无 holdout run，
      取有读数的那次；两侧都有读数时取较低者（悲观侧，避免乐观偏置）。
    """
    if not runs:
        return None
    scored = sorted(((_holdout_val(r), i, r) for i, r in enumerate(runs)), key=lambda t: (t[0], t[1]))
    if all(v == float("-inf") for v, _, _ in scored):
        return pick_best(runs)
    n = len(scored)
    if n % 2 == 1:
        return scored[n // 2][2]
    lo, hi = scored[n // 2 - 1][0], scored[n // 2][0]
    if lo == float("-inf"):
        # 一侧是缺失读数的 run：无代表意义，取有读数的一侧
        return scored[n // 2][2]
    mid = (lo + hi) / 2.0
    # 离中位数最近；并列（中位数恰为两点中点）→ 取较低者
    if abs(hi - mid) < abs(mid - lo):
        return scored[n // 2][2]
    return scored[n // 2 - 1][2]


# 代表口径选择（可配置：in-sample best / 跨 seed 中位 holdout / holdout best）
CRITERION_DEFAULT = "holdout_median"  # 现行稳健默认：跨 seed 中位 holdout
CRITERION_IN_SAMPLE = "in_sample"
CRITERION_HOLDOUT_MEDIAN = "holdout_median"
CRITERION_HOLDOUT_BEST = "holdout_best"
CRITERIA = (CRITERION_IN_SAMPLE, CRITERION_HOLDOUT_MEDIAN, CRITERION_HOLDOUT_BEST)
CRITERION_LABELS = {
    CRITERION_IN_SAMPLE: "in-sample best（每 seed 训练最优）",
    CRITERION_HOLDOUT_MEDIAN: "跨 seed 中位 holdout（稳健默认）",
    CRITERION_HOLDOUT_BEST: "跨 seed 最优 holdout（谨慎，可能挑中彩票 seed）",
}
# 报告结论要“分别给两种口径”：in-sample best 与 holdout（中位）
REPORT_CRITERIA = (CRITERION_IN_SAMPLE, CRITERION_HOLDOUT_MEDIAN)


def pick_holdout_best(runs: list[dict]) -> dict | None:
    """跨 seed 挑 holdout val_score 最高那次（in-sample best 之外的“holdout best”口径）。"""
    best = None
    for r in runs:
        v = _holdout_val(r)
        if v == float("-inf"):
            continue
        if best is None or v > _holdout_val(best):
            best = r
    return best or pick_best(runs)


def pick_by_criterion(runs: list[dict], criterion: str) -> dict | None:
    """按可配置代表口径选代表 run。"""
    if criterion == CRITERION_HOLDOUT_MEDIAN:
        return pick_representative(runs)
    if criterion == CRITERION_HOLDOUT_BEST:
        return pick_holdout_best(runs)
    return pick_best(runs)  # in_sample 默认


def summary_rep_criterion(summary: dict | None) -> str:
    """落盘的口径；旧文件无该字段时回退 in_sample（当时结论即 in-sample best）。"""
    if not summary:
        return CRITERION_DEFAULT
    return str((summary.get("params") or {}).get("rep_criterion") or CRITERION_DEFAULT)


def run_holdout_curve(r: dict | None) -> dict | None:
    """run 携带的 holdout 逐根资金曲线（新 compare 跑时 engine 写入 run.holdout.equity_curve）。"""
    if not r:
        return None
    hc = (r.get("holdout") or {}).get("equity_curve")
    if isinstance(hc, dict) and hc.get("equity"):
        return hc
    # 兼容极早期产物：直接挂在 run.holdout_curve
    hc2 = r.get("holdout_curve")
    if isinstance(hc2, dict) and hc2.get("equity"):
        return hc2
    return None


def conclusion_for_variants(variants: dict, criterion: str) -> list[str]:
    """按指定口径给出结论（variants = summary['variants']）。"""
    tail = pick_by_criterion((variants.get("tail") or {}).get("runs") or [], criterion)
    spread = pick_by_criterion((variants.get("spread") or {}).get("runs") or [], criterion)
    return conclusion_for(tail, spread)


def conclusions_by_criterion(summary: dict | None) -> dict[str, list[str]]:
    """对所有可代表口径分别给结论（报告/摘要「两种口径各给结论」的数据源）。"""
    variants = (summary or {}).get("variants") or {}
    if not variants.get("tail") or not variants.get("spread"):
        return {}
    out: dict[str, list[str]] = {}
    for crit in CRITERIA:
        if not (variants.get("tail") or {}).get("runs") and not (variants.get("spread") or {}).get("runs"):
            continue
        out[crit] = conclusion_for_variants(variants, crit)
    return out


def variant_by_criterion(blk: dict | None, criterion: str) -> dict | None:
    """按口径取 variant 代表 run（带 runs 时现算，旧文件从落盘字段读）。"""
    if not blk:
        return None
    runs = blk.get("runs") or []
    if runs:
        return pick_by_criterion(runs, criterion)
    return variant_representative(blk)

def median_holdout(runs: list[dict]) -> float | None:
    """跨 seed holdout val_score 的中位数（无任何 holdout 读数 → None）。"""
    vals = sorted(_holdout_val(r) for r in runs if _holdout_val(r) != float("-inf"))
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def variant_representative(blk: dict | None) -> dict | None:
    """从 variants[key] 取代表 run：优先落盘的 representative，旧文件从 runs 现算。"""
    if not blk:
        return None
    rep = blk.get("representative")
    if rep:
        return rep
    return pick_representative(blk.get("runs") or [])


def conclusion_for(tail_rep: dict | None, spread_rep: dict | None) -> list[str]:
    """按代表口径（跨 seed 中位 holdout）给出结论行。"""
    def gate(r: dict | None) -> dict | None:
        if not r:
            return None
        ho = r.get("holdout")
        if not ho or ho.get("passed") is not True:
            return None
        return ho

    th, sh = gate(tail_rep), gate(spread_rep)
    lines: list[str] = []
    if th and sh:
        if th.get("val_score", 0) > sh.get("val_score", 0):
            lines.append("两路代表均通过闸门：tail 的样本外分更高，倾向 tail（最近N根）。")
        else:
            lines.append("两路代表均通过闸门：spread 的样本外分更高或相当，倾向 spread（全历史分块）。")
    elif th:
        lines.append("仅 tail 的代表通过 holdout 闸门：优先 tail（最近N根）。")
    elif sh:
        lines.append("仅 spread 的代表通过 holdout 闸门：优先 spread（全历史分块）。")
    else:
        lines.append("两路代表均未通过 holdout 闸门（短训期属正常）：对比仅作相对参考，建议加大步数后再判断。")
        t = tail_rep.get("holdout") if tail_rep else None
        s = spread_rep.get("holdout") if spread_rep else None
        tv = t.get("val_score") if t else None
        sv = s.get("val_score") if s else None
        if tv is not None and sv is not None:
            lines.append(f"相对参考：tail 代表样本外分 {tv:.4f} vs spread {sv:.4f}（选高者，注意未过闸门无统计意义）。")
    return lines


def _cell(r: dict | None, key: str) -> str:
    if not r:
        return "—"
    ho = r.get("holdout") or {}
    if key == "holdout_val":
        return f"{ho.get('val_score'):.4f}" if isinstance(ho.get("val_score"), (int, float)) else "—"
    if key == "passed":
        return "✓ 通过" if ho.get("passed") else "✗ 未过"
    if key == "sharpe":
        return f"{ho.get('sharpe'):+.2f}" if isinstance(ho.get("sharpe"), (int, float)) else "—"
    if key == "ratio":
        return f"{ho.get('score_ratio'):+.3f}" if isinstance(ho.get("score_ratio"), (int, float)) else "—"
    if key == "ret":
        return f"{ho.get('total_return_pct'):+.2f}%" if isinstance(ho.get("total_return_pct"), (int, float)) else "—"
    v = r.get(key)
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v or "—")


def render_markdown(summary: dict | None) -> str:
    if not summary:
        return "# tail vs spread 对比报告\n\n尚无对比实验结果（results/compare_latest.json 未生成）。\n"
    variants = summary.get("variants") or {}
    params = summary.get("params") or {}
    steps = params.get("steps")
    seeds = params.get("seeds") or []
    n_bars = params.get("n_bars")
    regime = params.get("regime")
    tail = variants.get("tail") or {}
    spread = variants.get("spread") or {}
    tb, sb = variant_representative(tail), variant_representative(spread)
    symbol = summary_symbol(summary) or "—"
    created = str(summary.get("created_at") or "")[:19].replace("T", " ")

    lines = [
        "# tail vs spread 范围对比 · 结论报告",
        "",
        f"- **品种**：{symbol}  \n- **生成时间**：{created}  \n"
        f"- **源数据**：`{summary.get('source_file') or '—'}`  \n"
        f"- **参数**：N={n_bars}（块）· regime={regime} · steps={steps} · seeds={seeds}",
        "",
    ]
    if isinstance(steps, int) and steps < MINI_STEPS:
        lines += [
            "> ⚠️ **迷你口径**：本组仅 steps={}（<{}）。可作为方法与管线演示/初步信号，"
            "**不可当作正式选型结论**；正式结论需 ≥{} 步 + 多 seed 全量跑。".format(steps, MINI_STEPS, MINI_STEPS),
            "",
        ]

    lines += [
        "## 两路曲线的 best / holdout 结论",
        "",
        "| 变体 | 代表(seed) | 代表 in-sample best | 代表 holdout 分 | 闸门 | holdout Sharpe | score_ratio | 收益 | 跨seed中位 holdout |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, blk in (("tail（最近 N 根）", tail), ("spread（全历史分层分块）", spread)):
        b = variant_representative(blk)
        seed = (b or {}).get("seed")
        seed_txt = str(seed) if seed is not None else "—"
        med = median_holdout(blk.get("runs") or [])
        med_txt = f"{med:.4f}" if med is not None else "—"
        lines.append(
            f"| {name} | {seed_txt} | {_cell(b, 'best_score')} | {_cell(b, 'holdout_val')} | "
            f"{_cell(b, 'passed')} | {_cell(b, 'sharpe')} | {_cell(b, 'ratio')} | {_cell(b, 'ret')} | {med_txt} |"
        )
    lines += [
        "",
        "> 代表口径：**跨 seed 中位 holdout** —— 取各 seed holdout val_score 处于中位数位置的那次 run（偶数个时取离中位数最近、并列取低者）。"
        "不用 in-sample best 选代表，避免挑中样本外彩票 seed；“跨seed中位 holdout”列 = 该变体全部 seed holdout 分的中位数。",
        "",
    ]

    # 全部 run（每 seed）明细
    runs = []
    for v in ("tail", "spread"):
        for r in (variants.get(v) or {}).get("runs") or []:
            r2 = dict(r)
            r2["_variant"] = v
            runs.append(r2)
    if runs:
        lines += ["## 分 seed 明细", "", "| 变体 | seed | in-sample best | holdout 分 | 闸门 | Sharpe |", "|---|---|---|---|---|---|"]
        for r in runs:
            lines.append(
                f"| {r['_variant']} | {r.get('seed', '—')} | {_cell(r, 'best_score')} | "
                f"{_cell(r, 'holdout_val')} | {_cell(r, 'passed')} | {_cell(r, 'sharpe')} |"
            )
        lines += [""]

    # 结论：优先用落盘 conclusion（当前口径），旧文件按代表口径现算补齐
    concl = summary.get("conclusion") or []
    if not concl and (tb or sb):
        concl = conclusion_for(tb, sb)
    lines += ["## 结论", ""]
    if concl:
        for c in concl if isinstance(concl, list) else [concl]:
            lines.append(f"- {c}")
        lines.append("")
    elif not tb and not sb:
        lines.append("两路均无有效代表 run，本次无法给出倾向。")
        lines.append("")

    # 两种口径各自给结论（in-sample best 与 holdout 中位）：口径不同 → 结论可能相反
    concl_by_crit = summary.get("conclusions_by_criterion") or \
        conclusions_by_criterion(summary)
    if concl_by_crit:
        lines += ["### 分口径结论对照", "",
                  "| 口径 | tail vs spread 结论 |",
                  "|---|---|"]
        for crit in (CRITERION_IN_SAMPLE, CRITERION_HOLDOUT_MEDIAN):
            cs = concl_by_crit.get(crit) or []
            txt = cs[0] if cs else "（该口径无可比 run）"
            lines.append(f"| {CRITERION_LABELS.get(crit, crit)} | {txt} |")
        lines += [""]
    lines += [
        "",
        "---",
        "",
        "### 使用说明",
        "",
        "- 本报告由每次 `scripts/compare_ranges.py` 完成后自动从 `results/compare_latest.json` 重新生成（见 `web/compare_report.py`）。",
        "- “闸门”= 与训练一致的 champion 风格 holdout 判定（val_score + 统计口径），passed=True 才建议作为正式选型依据。",
        "- 回测页“默认模型”卡片下方会自动展示最近一次对比摘要（同一品种）。",
        "",
    ]
    return "\n".join(lines)


def write_docs_report(summary: dict | None) -> Path:
    DOCS_DIR.mkdir(exist_ok=True)
    md = render_markdown(summary)
    main = DOCS_DIR / DOCS_NAME
    main.write_text(md, encoding="utf-8")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stamp = DOCS_DIR / DOCS_NAME.replace(".md", f"_{ts}.md")
    stamp.write_text(md, encoding="utf-8")
    return main


def light_summary(summary: dict | None) -> dict | None:
    """给回测页的轻量摘要（不含大 runs 明细）。代表 = 跨 seed 中位 holdout 口径。"""
    if not summary:
        return None
    variants = summary.get("variants") or {}

    def pick(name: str) -> dict:
        blk = variants.get(name) or {}
        b = variant_representative(blk) or {}
        ho = b.get("holdout") or {}
        med = median_holdout(blk.get("runs") or [])
        # 代表 run 的 holdout 逐根曲线（回测页叠加用；无则 None）
        hc = run_holdout_curve(b) if b else None
        return {
            "best_score": b.get("best_score"),
            "holdout_val": ho.get("val_score"),
            "passed": bool(ho.get("passed")),
            "sharpe": ho.get("sharpe"),
            "score_ratio": ho.get("score_ratio"),
            "total_return_pct": ho.get("total_return_pct"),
            "seed": b.get("seed"),
            "median_holdout": med,
            "runs": len(blk.get("runs") or []),
            "holdout_passed_count": blk.get("holdout_passed", 0),
            "holdout_curve": hc,
        }

    return {
        "kind": summary.get("kind"),
        "created_at": summary.get("created_at"),
        "source_file": summary.get("source_file"),
        "symbol": summary_symbol(summary),
        "params": {k: v for k, v in (summary.get("params") or {}).items()},
        "conclusion": summary.get("conclusion"),
        "conclusions_by_criterion": conclusions_by_criterion(summary),
        "tail": pick("tail"),
        "spread": pick("spread"),
        "mini": bool((summary.get("params") or {}).get("steps") and (summary["params"]["steps"] < MINI_STEPS)),
    }
