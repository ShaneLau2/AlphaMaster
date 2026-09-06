# 每日巡检：真·样本外自动累积 + 报告（daily_oos_watch）

把 `scripts/true_oos_backtest.py` 的一次性结论变成持续监控：
每次运行**增量下载** BTCUSDT M5 的尾部新 bar 到 `data/oos_daily/BTCUSDT_M5.parquet`
（只存 ts > 训练截止 的 bar，即「训练绝未见过」的数据），与训练切片
`data/slices/BTCUSDT_M5.parquet`（40k 根）共同构成回放文件，攒够阈值后自动重跑真 OOS
回放（5 方案 × 阈值 0.05/0.8，区域统计只在训练截止后的新段计收益/回撤），产出
`results/true_oos_daily_<时间戳>.json/.md` 并推送飞书（未配 webhook 时降级 macOS 本地
通知 + `logs/daily_oos_watch.log`，链路不静默失效）。

## 数据流
```
data/training/BTCUSDT_M5.parquet   ← 首跑种子化（一次）：导入 ts > cutoff 的既有 bar
        │
        ▼
data/oos_daily/BTCUSDT_M5.parquet  ← 每次运行：Binance 尾窗增量合并（只追加）
        │  距上次报告新增 ≥ min-new(默认 5000) 根？
        ▼ 是（或 --force）
data/oos_daily/run/BTCUSDT_M5.parquet  ← 切片(40k) + 累积新 bar 重建
        ▼
run_replay（同模拟盘/回测离散撮合引擎）→ results/true_oos_daily_*.md/json → 飞书
```

## 命令
```bash
# 每天一次（cron / launchd）：
.venv/bin/python scripts/daily_oos_watch.py

# 只看状态（不拉网不动盘）：
.venv/bin/python scripts/daily_oos_watch.py --dry-run --offline

# 攒够了立即出报告：
.venv/bin/python scripts/daily_oos_watch.py --force

# 手动控制阈值（测试/演示）：
.venv/bin/python scripts/daily_oos_watch.py --min-new 200 --offline
```

launchd 示例：`scripts/plists/com.alphamaster.daily-oos-watch.plist`（每天 09:05）。

## 语义与幂等
- 训练截止 = 冠军策略 `data_source.end`（缺省回退到训练切片最后一根 ts）；
- 首跑自动从全历史导入已有的 post-cutoff bar（2026-09-04 已下载 383 根），不重复下载；
- `reported_upto_ts` 记录上次报告推进到的 bar；距上次报告新增 ≥ min-new 才重跑，重复运行
  不会重复出报告；`--force` 跳过该检查；
- 状态在 `data/oos_daily/state.json`，全部在 `data/`（gitignore）下，不污染仓库。

## 结论怎么读
样本量提示：M5 每 24h ≈ 288 根。5000 根 ≈ 17 天。报告是方向性 sanity check，别用单次
下裁决；趋势看多次巡检的累积（可与 `results/true_oos_REPORT.md` 的早期 383 根对照：
signal t=0.05 −1.12% / −8.51 / 6 笔，dd t=0.05 +0.48% / +4.57 / 3 笔 —— dd 靠熔断压交易数
的「稳」在真 OOS 上依旧成立，但样本量太小）。
