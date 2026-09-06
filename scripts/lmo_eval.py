"""M2b: Leave-Market-Out 评估 —— 把冠军公式拿到【未见过的市场】上验证。

回答 "AlphaGPT 学会了公式还是记住了市场"：训练时完全冻结 policy，
把 champion 放到 registry oos_review 腿（PAXG/XAUUSD/EURUSD/600519/AAPL…）
或任意 parquet 上跑生产口径 holdout + vol 覆盖，输出对比表。

用法：
  .venv/bin/python -u scripts/lmo_eval.py --formula results/multi_expA.json \
      --out results/lmo_expA.json
      # --formula 也可以是 strategies/best_BTCUSDT_H1.json 或 "[8,79,8,3,126,95,79,113]"

  # 只看某几个 OOS 腿
  .venv/bin/python -u scripts/lmo_eval.py --formula results/multi_expA.json \
      --legs PAXGUSDT,XAUUSD,EURUSD --out results/lmo_expA.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_formula(src: str) -> tuple[list[int], str]:
    """接受结果 JSON / 策略 JSON / 裸 token 列表，返回 (公式, 来源描述)。"""
    p = Path(src)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        for key in ("best_formula", "formula", "champ_fml"):
            if isinstance(raw, dict) and raw.get(key):
                return list(raw[key]), f"{src}#{key}"
        raise SystemExit(f"{src} 中找不到公式字段 (best_formula/formula)")
    try:
        return json.loads(src), "inline"
    except json.JSONDecodeError:
        raise SystemExit(f"无法解析公式来源: {src}") from None


def run(formula_src: str, legs: list[str] | None, out: str) -> dict:
    from data_pipeline.market_env import MarketEnv
    from data_pipeline.market_registry import load_registry, oos_review_markets
    from model_core.vocab import FORMULA_VOCAB

    fml, desc = _load_formula(formula_src)
    reg = load_registry()
    # 候选腿 = OOS 复核 + 全部 policy 市场（去重；含未达门槛的 eval-only 腿）
    seen: set[str] = set()
    candidates = []
    for m in oos_review_markets(reg) + [
        m for pol in reg.get("policies", {}).values()
        for m in pol.get("markets", [])]:
        if m["symbol"] not in seen:
            seen.add(m["symbol"])
            candidates.append(m)
    if legs:
        candidates = [m for m in candidates if m["symbol"] in legs]
    if not candidates:
        raise SystemExit("无 OOS 复核腿可用")

    print(f"公式: {fml}")
    print(f"解码: {' -> '.join(FORMULA_VOCAB.token_names[t] for t in fml)}")
    print(f"来源: {desc}\n")

    rows = []
    for m in candidates:
        env = MarketEnv(m, seed=0)
        ho = env.holdout(fml)
        ok_cov, info = env.coverage(fml)
        row = {
            "symbol": env.symbol, "timeframe": env.timeframe, "group": env.group,
            "bars": env.bars_full, "train_bars": env.bars_train,
            "holdout": ho,
            "coverage_ok": ok_cov, "coverage_info": info,
        }
        rows.append(row)
        if ho:
            print(f"{env.name:14s} bars={env.bars_train:6d} | holdout: "
                  f"ho_adj={ho.get('ho_adj', float('nan')):+.3f} "
                  f"sharpe={ho.get('sharpe', float('nan')):+.2f} "
                  f"ann_ret={ho.get('ann_ret', 0)*100:+.1f}% | "
                  f"vol覆盖={'✓' if ok_cov else '✗'} "
                  f"{('' if ok_cov else str(info))}")
        else:
            print(f"{env.name:14s} | holdout: 无结果（常数/错误）| "
                  f"vol覆盖={'✓' if ok_cov else '✗'}")

    res = {"formula": fml, "formula_decoded": [
        FORMULA_VOCAB.token_names[t] for t in fml], "source": desc,
        "legs": rows,
        "summary": {
            "n_legs": len(rows),
            "n_holdout_pass": sum(1 for r in rows
                                  if r["holdout"] and r["holdout"].get("passed", False)),
            "n_coverage_pass": sum(1 for r in rows if r["coverage_ok"]),
        }}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n[结果] → {out}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-Market-Out 评估")
    ap.add_argument("--formula", required=True, help="结果JSON/策略JSON/裸token列表")
    ap.add_argument("--legs", default=None, help="逗号分隔 symbol 过滤")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    legs = [s.strip() for s in args.legs.split(",")] if args.legs else None
    run(args.formula, legs, args.out)


if __name__ == "__main__":
    main()
