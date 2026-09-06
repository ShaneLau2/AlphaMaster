"""m0_resample_h1.py — 把仓库内已有细周期 parquet(M1/M5)重采样为 H1。

M0 数据补全的离线路径:当 Binance/OKX 等在线源被墙时,用已有细周期档案
(如 paxgusdt_M5, ~6 年)聚合出 H1,获得去相关的 PAXG 训练腿。

口径(与 Binance H1 一致):
  - time: 小时桶起点(epoch 秒, 桶 = floor(time/3600)*3600);
  - open 取桶内第一根 open, high/low 取桶内极值, close 取桶内最后一根 close,
    volume 求和;缺失小时段不补零(保持与在线下载的空缺语义一致)。
输出 schema: time/open/high/low/close/volume (与现有 parquet 完全一致),
时间升序去重。

用法:
  .venv/bin/python scripts/m0_resample_h1.py data/training/paxgusdt_M5.parquet \
      --out data/training/PAXGUSDT_H1.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def resample_to_h1(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    need = {"time", "open", "high", "low", "close", "volume"}
    if not need.issubset(set(cols)):
        raise SystemExit(f"缺列: 需要 {sorted(need)}, 实际 {cols}")
    d = df.copy()
    d["time"] = pd.to_numeric(d["time"])
    d = d.sort_values("time").drop_duplicates("time", keep="last")
    d["bucket"] = (d["time"] // 3600) * 3600
    g = d.groupby("bucket", sort=True)
    out = pd.DataFrame({
        "time": g["time"].first(),
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).reset_index(drop=True)
    return out[["time", "open", "high", "low", "close", "volume"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="细周期 → H1 重采样(M0 离线补数)")
    ap.add_argument("input", help="源 parquet(M1/M5)")
    ap.add_argument("--out", required=True, help="输出 H1 parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    out = resample_to_h1(df)
    out.to_parquet(args.out, index=False)
    t0, t1 = out["time"].iloc[0], out["time"].iloc[-1]
    import datetime as _dt
    s = _dt.datetime.fromtimestamp(t0, _dt.timezone.utc)
    e = _dt.datetime.fromtimestamp(t1, _dt.timezone.utc)
    print(f"-> {args.out}")
    print(f"   bars={len(out)}  {s:%Y-%m-%d} → {e:%Y-%m-%d} ({(e-s).days/365.25:.1f}y)")


if __name__ == "__main__":
    main()