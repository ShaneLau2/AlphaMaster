# AlphaMaster Project Notes

量化因子挖掘中心（训练 / 回测 / 实时分析 / 模拟实盘 四合一 Web 控制台）。中文 UI。

## 项目定位
- 用强化学习在特征 + 算子空间搜索可解释因子公式，训练 / 回测 / 实时共用同一信号逻辑（`position = tanh(factor)`）
- 仓库原始 README 面向 Windows；本机为 macOS，以下为 macOS 实测流程

## 启动（macOS，首次由 agent 补全）

```bash
# 1. 依赖（.venv 已存在，Python 3.13）
.venv/bin/python -m pip install -r requirements.txt


# 2. 启动 Web 控制台
.venv/bin/python run_web.py --host 127.0.0.1 --port 8765
# 浏览器打开 http://127.0.0.1:8765 （推荐入口，四步：01 训练 / 02 回测 / 03 实时 / 04 模拟实盘）

# 关键约定与踩坑
- **训练中途的实时保存只写 `strategies/best_{sym}.live.json` 侧车**，绝不覆盖部署路径
  `strategies/best_{sym}.json`；只有冠军闸门（holdout + 空模型/跨折 SE 检验）通过才覆盖。
  `list_strategies` / 数据文件兜底均已排除 `*.live.json`（避免把未验证公式当冠军）。
  训练被 SIGTERM/SIGINT 中止时处理器会清理侧车；UI 模型库对带侧车的行显示「训练中·未验证」。
- 持仓管理方案注册表在 `web/hold_policy.py`（signal/risk/hybrid/be/time/chandelier/dd，参数内置）；
  be=保本追踪（浮盈≥1.5% 止损抬成本上），time=时间止损（grace 48 未赚 1% 走 / 96 根强平），
  chandelier=吊灯 ATR 止损（引擎需传 ATR），dd=行情回撤熔断（自峰值回撤≥3% 平仓停开、收复≤1%
  恢复；≥6% 需收复到 2.5%，阶梯非对称）。time/chandelier 需要 bar 上下文（bars_held/atr/收盘），
  dd 由两引擎用 `dd_ladder_step` 按“行情自滚动峰值回撤”驱动（非账户现金净值——现金态无法自愈）。
  非 signal 的回测与模拟盘共用 `web/paper_replay.py` 离散撮合引擎（同口径），
  回测 CLI 支持 `--hold-policy` 与 `--window-bars`（只回测尾部最后 N 根，≥800）。
  run_backtest 的 --hold-policy 校验读注册表，新增方案无需改 CLI。
```

**代理工具环境下的启动姿势**（普通 `&` / nohup 后台进程会被工具的进程清理杀掉）：
用 launchd 托管：

```bash
CERT=$(.venv/bin/python -c 'import certifi; print(certifi.where())')
launchctl submit -l alphamaster-web -- /usr/bin/env SSL_CERT_FILE="$CERT" `pwd`/.venv/bin/python `pwd`/run_web.py --host 127.0.0.1 --port 8765
# 停止：launchctl remove alphamaster-web
```

## 关键约定与踩坑
- **macOS 文件对话框（重要）**：macOS 的 Tcl/Tk 非线程安全，在 uvicorn 线程池工作线程里调 `tk.Tk()` 会让**整个服务进程死锁**（表现为点击「选择数据文件」后全部请求无响应）。`web/file_dialog.py` 现在是**非阻塞会话制**：`POST /browse` 立即返回 `{dialog, session}`（macOS 子进程 Tk 弹窗 / Windows comdlg32 线程），前端轮询 `GET /browse-poll?session=` 拿结果。**别再给 browse 请求加客户端超时**——旧版 12s 超时会和用户翻目录的耗时赛跑，导致「网络错误 signal is aborted without reason」刷屏 + 对话框叠出多个；现在翻多久都行（服务端 10 分钟自动清理），重复点击返回同一会话。下载目录统一在 `data/training/`，选择器默认就打开它（`initialdir`），策略选择器默认打开 `strategies/`。手动输入 Parquet 路径兜底（`?path=` 直选接口）保留。改前端 JS/CSS 后记得 bump `index.html` 里的 `?v=` 版本号，否则浏览器吃缓存。
- **前端状态推送走 SSE（`web/events.py`，GET /api/events）**：实时分析/模拟实盘引擎在跑时，服务端 watcher（订阅者>0 才启动）按轻指纹比对推送变更快照，前端对应页可见时应用；前端连不上该路由（旧后端）自动回退 REST 轮询。**依赖单 uvicorn worker**（`run_web.py` 默认即单进程，launchd 托管亦单进程）；若以后多 worker/多进程部署，推送只会连到其中一个进程，前端 30s reconcile 仍能兜底自愈，但实时/模拟盘会退化为慢轮询。改前端 JS/CSS 后记得 bump `index.html` 里的 `?v=`。
- **SSL 证书（重要）**：python.org 的 Python 3.13 默认信任库为空（`cert_store_stats()` → 0 证书），一切外部 HTTPS（TradingView probes、飞书 webhook、AI 分析）都会报 `CERTIFICATE_VERIFY_FAILED`。
  必须设 `SSL_CERT_FILE` 指向 certifi 的 bundle，例如 `export SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi; print(certifi.where())')`。
  根治方案：运行 python.org 安装器附带的 `/Library/Frameworks/Python.framework/Versions/3.13/bin/Install Certificates.command`。
- 可选数据源：tvdatafeed（git 安装）、tushare、pytdx、playwright（截图脚本 `scripts/capture_readme_shots.py` 需要）均已装入 `.venv`
- TradingView 匿名拉数：`nologin` 模式可用；`TradingViewSource.available()` 只检查包可导入，不探网络；真实连通性看 POST `/api/realtime/tradingview/probe`
- 国内网络访问 TradingView 可能被墙，实时分析页有探测与提示（开启 VPN 全局 / 云服务器部署）
- Web 设置持久化在 `web_settings.json`（已 gitignore 之外；含 tqsdk 账号等本地配置）
- **训练数据范围与实验（train_sampler / 实验脚本）**：子集生成器 `data_pipeline/train_sampler.py`
  （tail / spread，spread 默认按滚动波动率分层选块 regime=vol，可 trend/equal）。**随机抽单根不可行**——
  特征全是滚动/因果（回看 ~800 根），只支持整块抽取，块前垫同年代 warm-up；子集确定性生成到
  `data/slices/{模式}_{根数}.../`，并写 `train_range.json` sidecar（溯源）。
  实验隔离纪律：引擎 `run_tag` 后缀历史/检查点文件名；实验脚本（scripts/train_variant.py /
  compare_ranges.py / experiment_old_vs_new.py）一律 `_suppress_deploy=True`，不写 best_*.json、
  不写 *.live.json、不消费正式 holdout 指纹，结束自清理 tagged 残留。跑完实验记得清 `data/slices/` 里的临时子集。
  策略 JSON 的 `train_range` 字段来自子集目录 sidecar；UI 徽标在老文件上按路径兜底解析。

## 常用命令
```bash
# Web 控制台
.venv/bin/python run_web.py --port 8765

# CLI 训练（自动续训；加 --from-scratch 重新训练；--seed N 固定随机种子）
.venv/bin/python train_file.py --data-file /path/to/BTCUSDT_H1.parquet
# 双 seed 升级纪律：每个 seed 完整训练一轮，全部 seed 的 val 都超过旧冠军 +
# CHAMPION_UPGRADE_MARGIN(0.02) 才部署最佳者，否则保留旧冠军
.venv/bin/python train_file.py --data-file x.parquet --seeds 17,23

## 冠军闸门（P0/P1/P2 加固，engine.py）
- **val 只入围**：训练中维护 top-20 finalist 表（val 排序）；训练结束在【从未参与选优的
  holdout 尾部窗口】上按生产配置（cost=0.0003=佣金0.02%+滑点0.01%，tanh 仓位）跑真实回测，
  冠军 = holdout Sharpe 最高者（次优者留作跨折 SE 参照）。finalists 落盘 strategies/finalists_{sym}.json
  （P3 脚本 scripts/check_finalist_robustness.py 用：cost×2 / folds±1 / 9 起点位移）。
- **holdout 闸门**：冠军的 holdout 分>0、保持率(分/val)≥0.30、Sharpe≥0、空模型 99 分位
  （64 条随机公式基准）、冠军 vs 次优跨折 SE 裕度——任一不过则【不覆盖】
  strategies/best_{sym}.json，恢复旧冠军并把拒绝原因写入 champion_history。
- **单次消费**：数据指纹（行数+首末时间+OHLC 内容 hash）存 data/holdout_state.json，
  同版本数据只允许批准一次（防反复重训挑 holdout 彩票）；数据一变（merge 新 K 线）才重置。
- 部署/拒绝事件 append 到 strategies/champion_history.json（date/symbol/formula/val/
  holdout/fingerprint/seed）——线上档案 + 退化回滚的基础：
  scripts/check_champion_decay.py --symbol X --data-file x.parquet [--watch] [--rollback]
  （实盘每 bar 用 strategy_manager/signal_archive.append_live_signal 记录信号，60 根后回填
  真实 IC/Sharpe，连续 3 期下滑告警并可回滚上一版冠军）。
- **P2 IC 对齐修正（重要）**：IC 现配对 factor[t]~target_ret[t]（target_ret[t]=log(open[t+2]/open[t+1])，
  position[t] 在 open[t+1] 成交），原实现错配成 factor[t]~target[t+1] 晚一根 bar；
  评分窗口排除最后两根边界 bar（target 恒 0）。tests/unit/test_ic_pnl_alignment.py 锁定对齐。
- **前视回归套件**：tests/property/test_lookahead.py 对全部 65 特征 × 62 算子做 shift-invariance
  属性测试（全序列 vs 截断序列前缀一致）+ 标签成熟度窗口断言。
- 修过的坑：feat 是 [N,C,T]，截断必须 feat[:,:,:T]——原 feat[:,:T] 切的是通道维(65)，
  holdout 预留后训练评分每步全部 error（分数恒 -inf 且无人察觉）；P3 脚本曾犯同错，已修。

# 健康检查
curl -s http://127.0.0.1:8765/api/health

# 联网下载 K 线（→data/training/{sym}_{tf}.parquet），UI 训练页可一键下载
# 1000 默认写入模式 mode=merge：与已有 {sym}_{tf}.parquet 按 time 去重合并，历史只增不减
#   （重复下载累积深度；旧文件损坏/格式不符自动回退覆盖）；mode=replace 恢复旧行为
# 默认异步：POST 立即返回 job_id（有任务在跑时自动排队，串行执行，不再 409）；
#   状态查询：GET /api/data/download-queue（全部任务+进度+排队位）或 ?job_id=…（单任务）
#   任务注册表持久化在 data/download_jobs.json：重启后 done 结果保留、queued 自动续跑、
#   running 标记中断重新提交（半途拉取无法恢复）；CLI 需要阻塞结果时加 ?sync=1
# source 可选 tradingview / binance / okx / tongdaxin（默认 tradingview）；n_bars 自选
# tradingview 走 web/tv_history.py 直连 WebSocket；tongdaxin 免费行情服务器（A股/指数）
# binance 走公开行情 API（endTime 翻页，全量历史，实测 5 万根 1h 约 27s）；okx 走 download_okx_klines 翻页（本机被墙时 502）
# Binance 周期完整性：1m/3m/5m/15m/30m/1h/2h/4h/6h/8h/12h/1d/3d/1w/1M 全部可用
#   （文件名 token：M3/H2/H6/H8/H12/D3，parse_parquet_filename 已认别名 3m/2h/6h/8h/12h/3d/120min/360min/480min/720min）
#   **30m 以下周期（1m/3m/5m/15m）可选全量历史，上限 100 万根**（bars_limit_for：binance+短周期→1M，
#   其余 10 万）；1m×100 万根 ≈ 23 个月，约 8-10 分钟（后台任务，进度实时可见）；更长周期不留种子
# 前端：周期下拉随数据源重建（各源支持列表不同）；Binance 选短周期时 K 线数量输入上限自动切到 1,000,000
curl -s -X POST http://127.0.0.1:8765/api/data/download -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","timeframe":"1d","source":"tongdaxin","n_bars":5000}'
curl -s "http://127.0.0.1:8765/api/data/download-status?job_id=<id>"
```

### TradingView 匿名会话深度实测（2026-09，单次 create_series 即可，翻页报文被拒）
- 1m ≈10,000 根（约 7 天）· 15m ≈6,200 根（约 2 个月）· 1h ≈9,900 根（约 1.7 年）·
  1d ≈3,200~14,800 根（2014 年起，部分节点到 1833）· 1w/1M 全量
- 请求量 >50,000 会被服务端静默丢弃（拉 D1 全量用 50,000 即可）；
  别用 request_more_data/get_series 翻页——服务端返回 invalid_method / 直接断连
- 文件名会转义 symbol 中的 `:`（如 BINANCE:BTCUSDT -> BINANCE_BTCUSDT，Windows 兼容）

## 数据位置
- **基准数据集（对比实验）**：`data/baseline/` —— 冻结快照 + manifest.json（行数/跨度/指纹/来源/
  STATUS 配方引用）。文件名保持 `{sym}_{tf}.parquet`（带 `_50k` 后缀会破坏 parse_parquet_filename）。
  当前：`data/baseline/BTCUSDT_H1.parquet` = 50,000 根 Binance H1（2020-12→2026-09，5.7 年，
  指纹 4913b295…，sha256 bb5ed0…）。
  用途：复现 strategies/STATUS_20250705.md 里 forex 组的 8 年训练配方（49,998 根 H1 / 年化+2.34% /
  Sharpe 0.64 / MDD 7.39% / 训练分 0.4851）。驱动脚本 scripts/run_baseline_experiment.py：
  有界步数训练（--steps，~55s/步 @50k 根；默认 150 ≈ 2.3h）+ 冠军生产口径回测（cost 0.0003）+
  对比报告（data/baseline/experiment_report.json）。完整 9000 步 ≈ 137h，不现实——有界预算跑法是
  正规姿势，步数写进报告与 champion_history，结论按“有界复现”表述。
- Parquet 数据：按 `{品种}_{周期}.parquet` 命名，如 `BTCUSDT_H1.parquet`；训练页可选用本地文件
- 下载文件附带 sidecar `{file}.meta.json`（source/source_label/downloaded_at），卡片显示数据来源；
  `web.data_download.read_parquet_meta` 读取（后端 app.py `_attach_data_meta` 挂到 overview/config）
- 策略输出：`strategies/best_{symbol}.json`
- 检查点：`checkpoints/`
- 日志：`logs/web_errors.log`（调试模式开关在界面右上「调试模式」）