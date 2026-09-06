"""train_gonogo.py — 训练中途 go/no-go 巡检。

读取 training_history_{SYMBOL}.json(引擎每 50 步原子刷新),在目标步(默认
1000)判定「继续 / 该停」,供人类或定时任务调用。判定只看已落盘的信号,
不触碰运行中的训练进程。

判定逻辑(2026-09-07 巡检口径,GO-on-climb 已改挂到门控系列):
  GO   = best_score 已突破覆盖门天花板(> threshold_best)
         或 best_score(逐公式验证奖励的 running max,即驱动选优的门控系列)
         在 improve_window 步内刷新过(出现新的更优公式)
  KILL = 到目标步后 best 仍冻结(≤ threshold_best 且超出 improve_window)且
         系数侧塌缩(top1_prob ≥ threshold_top1);原因指向公式族失血
         (见 scripts/vol_gate_diag.py),而非调门
  WAIT = 未到目标步(或目标步已到但证据不足以二分)

注意: 不用 batch-mean val_score 的爬升判活——它只是整批平均(~0.3),best 冻结
时也会随批次漂移,是假信号;判定只看真正驱动精英池的 best_score 是否刷新。

退出码: 0=GO(继续)  1=KILL(建议停)  2=WAIT(再等/再查)  3=数据缺失/异常
用法:
  .venv/bin/python scripts/train_gonogo.py --symbol ETHUSDT [--target-step 1000]
  # 定时任务示例(cron 每 30 分钟):
  .venv/bin/python scripts/train_gonogo.py --symbol ETHUSDT || echo "go/no-go 非 GO"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_history(symbol: str, history_dir: Path = ROOT) -> dict:
    """读取 history JSON;文件缺失/损坏返回空 dict。"""
    path = history_dir / f"training_history_{symbol}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001 缺失/损坏 → 空
        pass
    return {}


def _last_value(data: dict, key: str, default: float | None = None) -> float | None:
    vals = data.get(key)
    if not isinstance(vals, list) or not vals:
        return default
    try:
        return float(vals[-1])
    except (TypeError, ValueError):
        return default


def evaluate(
    symbol: str,
    target_step: int = 1000,
    threshold_best: float = 1.8,
    threshold_top1: float = 0.95,
    improve_window: int | None = None,
    history_dir: Path = ROOT,
) -> dict:
    """返回判定 dict;verdict ∈ {go, kill, wait, no-data}。"""
    data = load_history(symbol, history_dir)
    steps = data.get("step")
    if not isinstance(steps, list) or not steps:
        return {"verdict": "no-data", "reason": f"history 缺失/为空: training_history_{symbol}.json",
                "exit": 3}
    cur = int(steps[-1])
    best = _last_value(data, "best_score")
    top1 = _last_value(data, "top1_prob")
    val = _last_value(data, "val_score")
    eff_vocab = _last_value(data, "eff_vocab")

    # best 最后刷新步(冻结时长):找 best_score 序列最后一次变化的位置。
    # best_score 是逐公式验证奖励的 running max —— 也就是真正驱动精英池/选优的
    # 门控系列;它刷新 = 出现了新的更优公式(有真实信息增益),冻结 = 搜索未越过门。
    best_series = [float(v) for v in data.get("best_score", []) if isinstance(v, (int, float))]
    last_improve_step = 0
    if len(best_series) >= 2:
        for i in range(1, len(best_series)):
            if abs(best_series[i] - best_series[i - 1]) > 1e-9:
                last_improve_step = int(steps[i]) if i < len(steps) else last_improve_step
    frozen_steps = cur - last_improve_step
    # 「近期刷新」窗口:默认取目标步的 1/5(如 1000→200),至少 50 步。
    if improve_window is None:
        improve_window = max(50, target_step // 5)
    best_recent = frozen_steps <= improve_window

    out = {
        "symbol": symbol,
        "step": cur,
        "target_step": target_step,
        "best_score": best,
        "val_score": val,
        "top1_prob": top1,
        "eff_vocab": eff_vocab,
        "last_improve_step": last_improve_step,
        "frozen_steps": frozen_steps,
        "improve_window": improve_window,
        "best_recent": best_recent,
        "verdict": "wait",
        "reason": "",
        "exit": 2,
    }

    if cur < target_step:
        out["reason"] = (f"未到目标步 {target_step}(当前 {cur}),按巡检节奏复查")
        return out

    if best is None or top1 is None:
        out["verdict"] = "no-data"
        out["reason"] = "目标步已到但缺少 best_score/top1_prob 信号"
        out["exit"] = 3
        return out

    # GO: 突破覆盖门天花板(更新过且 > 阈值)
    if best > threshold_best:
        out["verdict"] = "go"
        out["reason"] = (f"best={best:.3f} > {threshold_best},已突破覆盖门天花板")
        out["exit"] = 0
        return out

    # GO: 门控系列(best_score)近期刷新 → 出现了新的更优公式,搜索仍有真实信息增益。
    # 不用 batch-mean val_score 爬升判活——它只是整批平均(~0.3),best 冻结时也会
    # 随批次漂移,是假信号;以 best_score 最近是否刷新为准。
    if best_recent:
        out["verdict"] = "go"
        out["reason"] = (f"best 最近 {frozen_steps} 步内刷新({last_improve_step}→{cur}),"
                         f"门控系列仍有信息增益;继续观察")
        out["exit"] = 0
        return out

    # KILL: best 冻结 + 系数塌缩 → 覆盖门平台收敛,继续 = 白烧 CPU。
    # 注意: 诊断已证明覆盖门行为正确(9 格样本充分,t 线卡在 -1.645),瓶颈是
    # 当前公式族在高vol×混合趋势格结构性失血,故建议不是「调门」而是「换族」——
    # 同族高分公式都会在失血格被拒,放松门只会放行真实失血的公式。
    if best <= threshold_best and top1 >= threshold_top1:
        out["verdict"] = "kill"
        out["reason"] = (
            f"best 冻结 {best:.3f} ≤ {threshold_best}({frozen_steps} 步未刷新)且系数塌缩 "
            f"(top1={top1:.3f} ≥ {threshold_top1}, eff_vocab={eff_vocab}),"
            f"已收敛到覆盖门平台。诊断(scripts/vol_gate_diag.py)显示门自身正确、"
            f"瓶颈是公式族在高vol×混合趋势格失血:建议停训后先查公式族(换族/换信号源)"
            f"而非调 vol 门阈值——调门只会放行真实失血的公式。"
        )
        out["exit"] = 1
        return out

    out["reason"] = f"证据不足以二分(best={best:.3f} 冻结 {frozen_steps} 步, top1={top1:.3f}),建议人工复核"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练 go/no-go 巡检")
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--target-step", type=int, default=1000)
    ap.add_argument("--threshold-best", type=float, default=1.8)
    ap.add_argument("--threshold-top1", type=float, default=0.95)
    ap.add_argument("--improve-window", type=int, default=None,
                    help="best_score 多久未刷新视为冻结(默认 max(50, target_step//5))")
    args = ap.parse_args(argv)

    res = evaluate(args.symbol, args.target_step, args.threshold_best,
                   args.threshold_top1, improve_window=args.improve_window)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return int(res["exit"])


if __name__ == "__main__":
    sys.exit(main())