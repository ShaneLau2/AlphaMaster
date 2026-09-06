"""M3: 多市场 A/B 实验 runner（对比报告在结果落地后另行撰写）。

变量（每次运行一个 arm，JSON 格式与 train_multi.py 完全一致，可横向比较）：
  --variant multi     多市场共享 policy（默认配置）        → 主臂
  --variant single    单市场控制臂（--markets BTCUSDT）
  --variant alpha     采样权重 α 扫描：0（等权）/ 0.5 / 1.0
  --variant reward    奖励聚合消融：group-mean / plain-mean / min-gate off
  --variant calib     calibration 周期扫描：10 / 20 / 40

用法：
  .venv/bin/python -u scripts/exp_multi_ab.py --variant multi --steps 150 \
      --seed 42 --out-json results/multi_ab_multi.json > logs/multi_ab_multi.log 2>&1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_multi import run_train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="多市场 A/B 实验")
    ap.add_argument("--variant", required=True,
                    choices=["multi", "single", "alpha", "reward", "calib"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--calib-every", type=int, default=20)
    ap.add_argument("--min-sharpe", type=float, default=0.0)
    ap.add_argument("--reward-mode", choices=["group", "plain", "nogate"],
                    default="group",
                    help="reward 消融: group=组级等权+min门槛 / plain=平权mean+min门槛 / nogate=组级等权无min门槛")
    ap.add_argument("--out-json", required=True)
    # 默认 = "default"（跟随 ModelConfig → VOL_COVERAGE_GRID=True 格级，与生产/部署口径一致）。
    # 早期 seed42 双臂误用 tier（段级）默认，属于 train/deploy 口径不一致，已修正。
    ap.add_argument("--vol-gate", choices=["default", "tier", "grid", "off"],
                    default="default")
    args = ap.parse_args()

    # tag 含 seed：不同 seed 的 checkpoint/滚动保留互相隔离，避免并发覆盖/误删
    tag = f"ab_{args.variant}_s{args.seed}"
    if args.variant == "multi":
        run_train("crypto", args.steps, args.seed, tag,
                  args.batch, args.calib_every, args.alpha, args.min_sharpe,
                  True, args.vol_gate, None, None, args.out_json, None, None, 5)
    elif args.variant == "single":
        # 同代码路径、同预算的单市场控制臂（共享 policy 在 1 个市场上退化）
        run_train("crypto", args.steps, args.seed, tag,
                  1, 1, 0.0, args.min_sharpe,
                  True, args.vol_gate, ["BTCUSDT"], None, args.out_json,
                  None, None, 5)
    elif args.variant == "alpha":
        run_train("crypto", args.steps, args.seed, tag,
                  args.batch, args.calib_every, args.alpha, args.min_sharpe,
                  True, args.vol_gate, None, None, args.out_json, None, None, 5)
    elif args.variant == "reward":
        group_mean = args.reward_mode != "plain"
        min_sharpe = args.min_sharpe if args.reward_mode != "nogate" else -1e9
        run_train("crypto", args.steps, args.seed, tag,
                  args.batch, args.calib_every, args.alpha, min_sharpe,
                  group_mean, args.vol_gate, None, None, args.out_json,
                  None, None, 5)
    else:  # calib
        run_train("crypto", args.steps, args.seed, tag,
                  args.batch, args.calib_every, args.alpha, args.min_sharpe,
                  True, args.vol_gate, None, None, args.out_json, None, None, 5)


if __name__ == "__main__":
    main()
