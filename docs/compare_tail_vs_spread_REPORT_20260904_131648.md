# tail vs spread 范围对比 · 结论报告

- **品种**：BTCUSDT  
- **生成时间**：2026-09-04 10:33:45  
- **源数据**：`/Users/shanelau/Desktop/strategytrading/AlphaMaster-main/data/slices/BTCUSDT_M5.parquet`  
- **参数**：N=15000（块）· regime=vol · steps=10 · seeds=[42, 7]

> ⚠️ **迷你口径**：本组仅 steps=10（<100）。可作为方法与管线演示/初步信号，**不可当作正式选型结论**；正式结论需 ≥100 步 + 多 seed 全量跑。

## 两路曲线的 best / holdout 结论

| 变体 | in-sample best | holdout 分 | 闸门 | holdout Sharpe | score_ratio | 收益 |
|---|---|---|---|---|---|---|
| tail（最近 N 根） | 2.1634 | -0.3691 | ✗ 未过 | -6.42 | -0.171 | -0.67% |
| spread（全历史分层分块） | 0.8747 | 2.9620 | ✓ 通过 | +8.19 | +3.386 | +1.34% |

> 注：“best”按每变体 in-sample best_score 最高那次展示（`_pick_best` 口径，与结论文件一致）。

## 分 seed 明细

| 变体 | seed | in-sample best | holdout 分 | 闸门 | Sharpe |
|---|---|---|---|---|---|
| tail | 42 | 2.1634 | -0.3691 | ✗ 未过 | -6.42 |
| tail | 7 | 1.1040 | 6.9699 | ✓ 通过 | +11.36 |
| spread | 42 | 0.8747 | 2.9620 | ✓ 通过 | +8.19 |
| spread | 7 | 0.8317 | -0.2971 | ✗ 未过 | -8.80 |

## 结论

- 仅 spread 通过 holdout 闸门：优先 spread（全历史分块）。

**倾向：spread（全历史分层分块）** —— spread 的 best 通过 holdout 闸门，tail 未过。

---

### 使用说明

- 本报告由每次 `scripts/compare_ranges.py` 完成后自动从 `results/compare_latest.json` 重新生成（见 `web/compare_report.py`）。
- “闸门”= 与训练一致的 champion 风格 holdout 判定（val_score + 统计口径），passed=True 才建议作为正式选型依据。
- 回测页“默认模型”卡片下方会自动展示最近一次对比摘要（同一品种）。
