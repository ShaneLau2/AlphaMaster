"""vol_gate_diag.py — 把任意公式跑过 vol×er 9 格覆盖门,输出逐格 t 统计。

诊断「某公式被 vol 覆盖门拒,是真实 regime 失血还是数据假象」。与选优闸门
同引擎、同窗口(训练窗,holdout 已切)、同实现(_init_vol_tiers/_vol_pnl_coverage)。
默认重放当前 live best(strategies/best_{sym}.live.json),也可传入候选公式。

用法:
  # 只看当前 live best(训练中实时侧车):
  .venv/bin/python scripts/vol_gate_diag.py --symbol ETHUSDT

  # 额外传入候选公式(空格分隔 token id),对比哪格失血:
  .venv/bin/python scripts/vol_gate_diag.py --symbol ETHUSDT \
      --formula "50 98 34 31 98 93 98 126"

  # 指定数据文件 + 关闭 live best:
  .venv/bin/python scripts/vol_gate_diag.py --data-file data/training/ETHUSDT_H1.parquet \
      --no-live --formula "50 98 34 14 7 56 126 72"

退出码: 0=全部通过(或无可判公式)  1=至少一个公式被拦截  2=数据/参数错误
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline.parquet_manager import ParquetDataManager  # noqa: E402
from model_core.config import ModelConfig  # noqa: E402
from model_core.engine import AlphaEngine, _holdout_bars_for  # noqa: E402
from model_core.vm import StackVM  # noqa: E402


def _load_live_best(symbol: str) -> dict | None:
    """读取 strategies/best_{symbol}.live.json;缺失/损坏返回 None。"""
    path = ROOT / "strategies" / f"best_{symbol}.live.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fml = data.get("formula")
        if isinstance(fml, list) and fml:
            return {"label": f"live best (score={data.get('best_score', '?')})",
                    "formula": [int(t) for t in fml]}
    except Exception:  # noqa: BLE001 缺失/损坏 → None
        pass
    return None


def _parse_formula(s: str) -> list[int]:
    return [int(t) for t in s.replace(",", " ").split() if t.strip()]


def run(data_file: str, formulas: list[dict], use_live: bool,
        symbol: str | None = None) -> int:
    mgr = ParquetDataManager(data_file)
    mgr.load()
    T_full = int(mgr.target_ret.shape[1])
    h = _holdout_bars_for(T_full)
    T = T_full - h
    feat = mgr.feat_tensor[:, :, :T]
    t_ret = mgr.target_ret[:, :T]
    eng = AlphaEngine(data_manager=mgr, target_symbol=mgr.symbol, seed=1)
    eng._init_vol_tiers(feat, t_ret)
    meta = eng._vol_grid_meta or {}
    print(f"数据 {Path(data_file).name}: T_full={T_full} holdout={h} 训练窗={T}")
    print(f"vol_q={meta.get('vol_q')} er_q={meta.get('er_q')}")
    cells = meta.get("cells") or []
    print(f"9 格 bar 分布: " + ", ".join(f"{c['cell']}={c['bars']}" for c in cells))

    if use_live and symbol:
        lb = _load_live_best(symbol)
        if lb and all(tuple(f["formula"]) != tuple(lb["formula"]) for f in formulas):
            formulas = [lb] + formulas

    if not formulas:
        print("未提供任何公式(且无 live best),无可判。", file=sys.stderr)
        return 2

    any_blocked = False
    for item in formulas:
        fml = item["formula"]
        label = item.get("label", f"formula {fml}")
        try:
            res = StackVM().execute(fml, feat)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{label}] 执行失败: {type(exc).__name__}: {exc}")
            continue
        if res is None or float(res.std()) < 1e-4:
            print(f"\n[{label}] 常量/非法公式(无可判)")
            continue
        ok, info = eng._vol_pnl_coverage(res, t_ret)
        any_blocked = any_blocked or (not ok)
        verdict = "✅ 通过" if ok else "❌ 被拦"
        print(f"\n=== [{label}] tokens={fml} -> {verdict} ===")
        if info:
            for r in info.get("cells", []):
                if r.get("skipped"):
                    print(f"  {r['cell']}: SKIPPED bars={r['bars']}")
                else:
                    mark = "" if r["pass"] else "  <== 失血"
                    print(f"  {r['cell']} ({r.get('vol')}×{r.get('er')}): "
                          f"bars={r['bars']} bps={r['bps']} t={r['t']}{mark}")
            if info.get("tiers"):
                print("  (注: 本次为段级口径)")
    return 1 if any_blocked else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vol×er 9 格覆盖门逐格诊断")
    ap.add_argument("--data-file", default="data/training/ETHUSDT_H1.parquet")
    ap.add_argument("--symbol", default=None, help="读取 live best 的品种(默认由数据文件名推导)")
    ap.add_argument("--formula", action="append", default=[],
                    help="候选公式 token(空格/逗号分隔),可多次传")
    ap.add_argument("--no-live", action="store_true", help="不自动加载 live best")
    args = ap.parse_args(argv)

    stem = Path(args.data_file).stem  # 如 ETHUSDT_H1 / BTCUSDT
    parts = stem.split("_")
    symbol = args.symbol or parts[0]
    formulas = [{"label": f"formula #{i+1}", "formula": _parse_formula(s)}
                for i, s in enumerate(args.formula) if _parse_formula(s)]
    try:
        return run(args.data_file, formulas, use_live=not args.no_live, symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        print(f"运行失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())