"""Task B: fold 噪声分析 —— 按每市场每折 bar 数给 val 分数 CI, 定 min_bars。

做法: 对每个市场 env, 取 K 条已验证公式, 在【同一套 5 折】上逐折算 val_score
(bt.evaluate_fold, 与训练选优同函数), 得每条公式的 5 折 val 序列:
    SE_formula = std(5 折) / sqrt(5)
市场级 SE = 公式级 SE 的均值 → CI 半宽 ≈ 2×SE。
输出表: 市场 / bars_train / 每折 val 窗口 bar 数 / val 均值 / SE / CI 半宽。

结论判据: 若 CI 半宽 ≈ 训练中 val 差异的典型量级(±0.3~0.5), 则该市场的
选优信号被噪声主导 → min_bars 应上调; 若 CI 半宽 ≪ 差异量级, 现门槛够用。

用法: .venv/bin/python -u scripts/fold_noise_analysis.py [--out results/fold_noise.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 已验证(非常数、非报错)的公式: E3/E4 冠军 + 冒烟/早期实验结果
KNOWN_FORMULAS = [
    [8, 79, 8, 3, 126, 95, 79, 113],   # e3_curriculum phase-1
    [8, 79, 8, 3, 126, 95],            # 短版
    [31, 114, 93, 89],                 # multi 冒烟最优
    [4, 80, 61, 104],                  # 早期冒烟
    [8, 79, 8, 3],                     # 极短
    [31, 114, 93, 89, 126, 95, 79, 113],
    [4, 80, 61, 104, 126, 95],
    [126, 95, 79, 113, 4, 80, 61, 104],
]

# 分析的市场: 5 加密训练腿 + 2 个低于门槛的对照(XAUUSD 24h / MSFT 会话时段)
MARKETS = [
    {"symbol": "BTCUSDT", "timeframe": "H1", "group": "crypto", "role": "train",
     "file": "data/training/BTCUSDT_H1.parquet"},
    {"symbol": "ETHUSDT", "timeframe": "H1", "group": "crypto", "role": "train",
     "file": "data/training/ETHUSDT_H1.parquet"},
    {"symbol": "SOLUSDT", "timeframe": "H1", "group": "crypto", "role": "train",
     "file": "data/training/SOLUSDT_H1.parquet"},
    {"symbol": "ADAUSDT", "timeframe": "H1", "group": "crypto", "role": "train",
     "file": "data/training/ADAUSDT_H1.parquet"},
    {"symbol": "XRPUSDT", "timeframe": "H1", "group": "crypto", "role": "train",
     "file": "data/training/XRPUSDT_H1.parquet"},
    {"symbol": "XAUUSD", "timeframe": "H1", "group": "commodity", "role": "oos",
     "file": "data/training/XAUUSD_H1.parquet"},
    {"symbol": "MSFT", "timeframe": "H1", "group": "technology", "role": "train",
     "file": "data/training/MSFT_H1.parquet"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fold_noise.json")
    args = ap.parse_args()

    from data_pipeline.market_env import MarketEnv
    from model_core.config import ModelConfig

    dev = ModelConfig.DEVICE
    rows = []
    for spec in MARKETS:
        env = MarketEnv(spec, seed=0)
        feats = env.feat
        t_ret = env.t_ret
        per_formula_se = []
        n_ok = 0
        per_fold_bars = []
        for fml in KNOWN_FORMULAS:
            try:
                res = env.engine.vm.execute(fml, feats)
                if res is None or float(res.std()) < 1e-4:
                    continue
                vals = []
                for f in env.folds:
                    _, v = env.engine.bt.evaluate_fold(
                        res, t_ret, f["train_start"], f["train_end"],
                        f["val_start"], f["val_end"])
                    vals.append(float(v))
                if len(vals) >= 3:
                    import statistics
                    m = statistics.mean(vals)
                    sd = statistics.stdev(vals)
                    per_formula_se.append(sd / (len(vals) ** 0.5))
                    n_ok += 1
            except Exception:  # noqa: BLE001 单条公式失败跳过
                continue
        if not per_formula_se:
            print(f"{env.name:12s} 无有效公式")
            continue
        import statistics
        se_mean = statistics.mean(per_formula_se)
        ci_half = 2.0 * se_mean
        fold_bars = [f["val_end"] - f["val_start"] for f in env.folds]
        row = {"market": env.name, "bars_train": env.bars_train,
               "bars_per_fold_val": fold_bars,
               "n_formulas": n_ok, "val_se_mean": round(se_mean, 4),
               "ci_half_width": round(ci_half, 4)}
        rows.append(row)
        print(f"{env.name:12s} bars_train={env.bars_train:6d} "
              f"折val窗={fold_bars} 公式={n_ok:2d} "
              f"val SE={se_mean:.4f} CI半宽≈{ci_half:.3f}")

    res = {"method": "per-fold val SE over 5 folds, CI=2*SE",
           "min_bars_current": ModelConfig.MARKET_MIN_BARS,
           "markets": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n[结果] → {args.out}")


if __name__ == "__main__":
    main()
