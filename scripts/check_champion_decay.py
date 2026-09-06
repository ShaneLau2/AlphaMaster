"""check_champion_decay.py — 线上冠军退化监控（P2）。

用法:
    python scripts/check_champion_decay.py --symbol BTCUSDT --data-file data/training/BTCUSDT_H1.parquet
    python scripts/check_champion_decay.py --symbol XAUUSD --data-file x.parquet --watch --interval 3600
    python scripts/check_champion_decay.py --symbol XAUUSD --data-file x.parquet --rollback

依赖：signal_archive 里的信号档案（data/signal_archive/{symbol}.csv）。
只有档案里积累了 min-bars（默认 60）根信号后才给出真实 IC/Sharpe 判定。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.parquet_manager import ParquetDataManager
from strategy_manager.signal_archive import (
    check_decay,
    realized_performance,
    rollback_champion,
)


def _load_opens(data_file: str):
    mgr = ParquetDataManager(data_file)
    mgr.load()
    return mgr.raw_dict["open"][0]


def main() -> None:
    ap = argparse.ArgumentParser(description="线上冠军退化监控/回滚")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--data-file", required=True, help="该品种的 parquet（提供已实现 open 序列）")
    ap.add_argument("--min-bars", type=int, default=60)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--max-declines", type=int, default=3)
    ap.add_argument("--watch", action="store_true", help="循环监控")
    ap.add_argument("--interval", type=int, default=3600, help="watch 轮询间隔（秒）")
    ap.add_argument("--rollback", action="store_true", help="直接回滚到上一版冠军")
    args = ap.parse_args()

    if args.rollback:
        res = rollback_champion(args.symbol)
        if res.get("ok"):
            print(f"[回滚] ✓ 已恢复上一版冠军 → {res['save_path']}")
        else:
            print(f"[回滚] ✗ {res.get('reason')}")
            sys.exit(1)
        return

    opens = _load_opens(args.data_file)
    while True:
        perf = realized_performance(args.symbol, opens, min_bars=args.min_bars)
        if perf is None:
            print(f"[{args.symbol}] 信号档案不足 {args.min_bars} 根，等待积累…")
        else:
            print(
                f"[{args.symbol}] 真实 IC={perf['realized_ic']:.4f} "
                f"Sharpe={perf['realized_sharpe']:.2f} "
                f"收益={perf['total_return_pct']:+.2f}% "
                f"（档案 {perf['bars_archived']} 根）"
            )
        decay = check_decay(
            args.symbol, opens,
            min_bars=args.min_bars, window=args.window,
            max_consecutive_declines=args.max_declines,
        )
        if decay.get("decayed"):
            print(f"⚠ [退化告警] {args.symbol} 连续 {args.max_declines} 期真实 IC 下滑: "
                  f"{decay.get('ic_series')}")
            print("  建议: python scripts/check_champion_decay.py "
                  f"--symbol {args.symbol} --data-file {args.data_file} --rollback")
        else:
            reason = decay.get("reason")
            if reason:
                print(f"[{args.symbol}] {reason}")
            elif decay.get("ic_series"):
                print(f"[{args.symbol}] IC 序列: {decay['ic_series']}（无连续下滑）")
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()