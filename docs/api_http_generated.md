# HTTP 接口参考（自动生成）

> 本文件由 `scripts/gen_api_docs.py` 从 FastAPI 路由 + Pydantic 模型自动生成，请勿手改；路由/模型变化后重跑 `python scripts/gen_api_docs.py` 同步。

- 服务入口：`run_web.py`，默认 `http://127.0.0.1:8765`

- 返回均为 JSON（导出接口为二进制文件）；错误 `{"detail": "..."}`


## `health`

| `GET` | `/api/health` |
|:--|:--|


## `routes`

| `GET` | `/api/routes` |
|:--|:--|


## `settings`

| `GET` | `/api/settings` |
|:--|:--|

| `PUT` | `/api/settings` |
|:--|:--|
  - **请求体**：`last_data_file` str | None · 可选（默认 None）；`last_strategy_file` str | None · 可选（默认 None）；`debug_mode` bool | None · 可选（默认 None）；`bg_animation` bool | None · 可选（默认 None）；`ai_provider` str | None · 可选（默认 None）；`ai_api_key` str | None · 可选（默认 None）；`ai_base_url` str | None · 可选（默认 None）；`ai_model` str | None · 可选（默认 None）；`bt_commission_pct` float | None · 可选（默认 None）；`bt_slippage_pct` float | None · 可选（默认 None）；`bt_hold_policy` str | None · 可选（默认 None）；`bt_window_bars` int | None · 可选（默认 None）；`bt_prefs_symbol` str | None · 可选（默认 None）；`paper_starting_balance` float | None · 可选（默认 None）；`paper_notional` float | None · 可选（默认 None）；`paper_commission_pct` float | None · 可选（默认 None）；`paper_slippage_pct` float | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`signal_threshold` float | None · 可选（默认 None）


## `config`

| `GET` | `/api/config` |
|:--|:--|


## `overview`

| `GET` | `/api/overview` |
|:--|:--|


## `data-file`

| `GET` | `/api/data-file/browse` |
|:--|:--|
  - **查询参数**：`path`

| `POST` | `/api/data-file/browse` |
|:--|:--|
  - **查询参数**：`path`

| `GET` | `/api/data-file/browse-poll` |
|:--|:--|
  - **查询参数**：`session`
  - 轮询文件选择会话：done 且选中时校验并返回文件信息（与选文件接口一致）。

| `POST` | `/api/data-file/browse-poll` |
|:--|:--|
  - **查询参数**：`session`
  - 轮询文件选择会话：done 且选中时校验并返回文件信息（与选文件接口一致）。


## `data`

| `POST` | `/api/data/download` |
|:--|:--|
  - **请求体**：`symbol` str · 必填；`timeframe` str · 可选（默认 '1h'）；`source` str · 可选（默认 'tradingview'）；`n_bars` int | None · 可选（默认 None）；`mode` str · 可选（默认 'merge'）
  - **查询参数**：`sync`
  - 联网下载指定品种/周期的 K 线并保存为本地 parquet。

| `GET` | `/api/data/download-history` |
|:--|:--|
  - **查询参数**：`limit`
  - 下载历史面板：sidecar 驱动的 (source, symbol, bars, date) 行 + 按源统计。

| `GET` | `/api/data/download-queue` |
|:--|:--|
  - 下载队列总览：运行/排队在前，最近完成在后（含各自进度与排队位置）。

| `GET` | `/api/data/download-status` |
|:--|:--|
  - **查询参数**：`job_id`
  - 查询单个后台下载任务进度；完成时附带完整结果。

| `GET` | `/api/data/files` |
|:--|:--|
  - 本地 .parquet 数据文件清单（data/training + data/slices），供回放页下拉选择。


## `strategy-file`

| `GET` | `/api/strategy-file/browse` |
|:--|:--|
  - **查询参数**：`path`
  - 选择策略文件：带 path 时直接验证该路径（免对话框），否则启动非阻塞选择器。

| `POST` | `/api/strategy-file/browse` |
|:--|:--|
  - **查询参数**：`path`
  - 选择策略文件：带 path 时直接验证该路径（免对话框），否则启动非阻塞选择器。

| `GET` | `/api/strategy-file/browse-poll` |
|:--|:--|
  - **查询参数**：`session`
  - 轮询策略选择会话：done 且选中时校验并返回策略信息。

| `POST` | `/api/strategy-file/browse-poll` |
|:--|:--|
  - **查询参数**：`session`
  - 轮询策略选择会话：done 且选中时校验并返回策略信息。

| `GET` | `/api/strategy-file/sync-best` |
|:--|:--|
  - **查询参数**：`symbol`

| `POST` | `/api/strategy-file/sync-best` |
|:--|:--|
  - **查询参数**：`symbol`


## `training`

| `POST` | `/api/training/import` |
|:--|:--|
  - **查询参数**：`symbol`

| `POST` | `/api/training/inspect-now` |
|:--|:--|
  - 内置训练巡检：立即对当前/最近训练日志做一次规则诊断（无需 API Key）。

| `POST` | `/api/training/start` |
|:--|:--|
  - **请求体**：`data_file` str · 必填；`from_scratch` bool · 可选（默认 False）；`data_mode` str · 可选（默认 'full'）；`n_bars` int | None · 可选（默认 None）；`n_chunks` int | None · 可选（默认 None）

| `GET` | `/api/training/status` |
|:--|:--|

| `POST` | `/api/training/stop` |
|:--|:--|

| `POST` | `/api/training/subset-preview` |
|:--|:--|
  - **请求体**：`data_file` str · 必填；`mode` str · 可选（默认 'full'）；`n_bars` int | None · 可选（默认 None）；`n_chunks` int | None · 可选（默认 None）；`regime` str | None · 可选（默认 'vol'）
  - 训练子集取样预览：返回所选区间在整段历史上的位置与 regime 覆盖（不写文件）。

| `POST` | `/api/training/verify-rollback` |
|:--|:--|
  - **请求体**：`strategy_file` str · 必填；`retrain_steps` int · 可选（默认 0）；`seed` int · 可选（默认 42）
  - 训练页冠军条目「回滚校验」：起子进程跑 scripts/verify_champion_rollback.py。

| `GET` | `/api/training/verify-rollback/status` |
|:--|:--|

| `GET` | `/api/training/{symbol}/export` |
|:--|:--|


## `strategies`

| `GET` | `/api/strategies` |
|:--|:--|

| `GET` | `/api/strategies/{symbol}/export` |
|:--|:--|


## `backtest`

| `GET` | `/api/backtest/chart/{name}` |
|:--|:--|

| `GET` | `/api/backtest/combo-sweep` |
|:--|:--|
  - 三轴联合回测结果（读 results/combo_sweep_latest.json，含基线切片说明）。

| `POST` | `/api/backtest/combo-sweep/run` |
|:--|:--|
  - **请求体**：`strategy_file` str | None · 可选（默认 None）；`data_file` str | None · 可选（默认 None）；`commission_pct` float | None · 可选（默认 None）；`slippage_pct` float | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`window_mode` str · 可选（默认 'tail'）；`regime` str · 可选（默认 'vol'）；`chunks` int · 可选（默认 4）；`caps` list[float] | None · 可选（默认 None）；`thresholds` list[float] | None · 可选（默认 None）
  - 启动三轴联合回测：22 组合 × 上限%档 × 阈值档 全网格（子进程跑，可轮询/停止）。

| `GET` | `/api/backtest/combo-sweep/status` |
|:--|:--|
  - 三轴联合回测任务状态（含按日志估算的完成行数 / 总行数）。

| `POST` | `/api/backtest/combo-sweep/stop` |
|:--|:--|

| `GET` | `/api/backtest/compare-summary` |
|:--|:--|
  - **查询参数**：`symbol`
  - 回测页默认模型卡片下方的 tail vs spread 对比摘要（同品种）。

| `POST` | `/api/backtest/cost-sweep` |
|:--|:--|
  - **请求体**：`strategy_file` str · 必填；`hold_policy` str · 可选（默认 'signal'）；`data_file` str | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`commission_pct` float | None · 可选（默认 None）；`slippage_pct` float | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`signal_threshold` float | None · 可选（默认 None）
  - 成本敏感性：同一模型×方案在 0→高 成本档位下重跑离散撮合，找盈亏平衡点。

| `GET` | `/api/backtest/equity` |
|:--|:--|
  - **查询参数**：`symbol`
  - 资金曲线原始数据（供前端渲染交互式 HTML 图表）。

| `GET` | `/api/backtest/hold-matrix` |
|:--|:--|
  - 持仓管理正交组合 N×N 矩阵（读 results/hold_matrix_latest.json）。

| `GET` | `/api/backtest/hold-matrix/curve` |
|:--|:--|
  - **查询参数**：`combo`
  - 单组合的资金曲线 + 滚动夏普（读 hold_matrix_curves_latest.npz 侧车）。

| `GET` | `/api/backtest/hold-matrix/prefs` |
|:--|:--|
  - **查询参数**：`symbol`
  - 全组合矩阵按品种记忆的工具栏设置（窗口模式/窗口/regime/块数/数据文件）。

| `PUT` | `/api/backtest/hold-matrix/prefs` |
|:--|:--|
  - **请求体**：`symbol` str · 必填；`data_file` str | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`window_mode` str · 可选（默认 'tail'）；`regime` str · 可选（默认 'vol'）；`chunks` int · 可选（默认 4）

| `POST` | `/api/backtest/hold-matrix/run` |
|:--|:--|
  - **请求体**：`strategy_file` str | None · 可选（默认 None）；`data_file` str | None · 可选（默认 None）；`commission_pct` float | None · 可选（默认 None）；`slippage_pct` float | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`signal_threshold` float | None · 可选（默认 None）；`window_mode` str · 可选（默认 'tail'）；`regime` str · 可选（默认 'vol'）；`chunks` int · 可选（默认 4）
  - 启动 N×N 全组合矩阵回测（脚本跑全部组合，写 results/hold_matrix_latest.json）。

| `GET` | `/api/backtest/hold-matrix/status` |
|:--|:--|
  - 全组合矩阵任务状态（含按日志估算的完成组合数）。

| `POST` | `/api/backtest/hold-matrix/stop` |
|:--|:--|

| `GET` | `/api/backtest/matrix-applied` |
|:--|:--|
  - **查询参数**：`symbol`
  - 读矩阵最优「用户接受记录」（settings.matrix_applied，区别于手动记忆）。

| `PUT` | `/api/backtest/matrix-applied` |
|:--|:--|
  - **请求体**：`symbol` str · 必填；`combo` str · 可选（默认 ''）；`sharpe` float | None · 可选（默认 None）；`max_drawdown` float | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`window_mode` str | None · 可选（默认 None）
  - 记录/清除某品种的矩阵最优接受。combo="" 表示清除该品种记录。

| `GET` | `/api/backtest/matrix-best` |
|:--|:--|
  - **查询参数**：`symbol`
  - 同品种最近一次全组合矩阵的「最优组合」（按夏普选非基线组合）+

| `POST` | `/api/backtest/policy-ab` |
|:--|:--|
  - **请求体**：`strategy_file` str · 必填；`policies` list · 可选（默认 []）；`data_file` str | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`commission_pct` float | None · 可选（默认 None）；`slippage_pct` float | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`signal_threshold` float | None · 可选（默认 None）；`caps` list[float] | None · 可选（默认 None）
  - 持仓方案 A/B：同一模型 × 多方案并排跑离散撮合回放，输出对比表指标。

| `GET` | `/api/backtest/policy-ab-latest` |
|:--|:--|
  - **查询参数**：`symbol`
  - 最近一次 A/B 对比结果（results/policy_ab_latest.json），供回测页「最近 A/B 冠军方案」。

| `GET` | `/api/backtest/prefs` |
|:--|:--|
  - **查询参数**：`symbol`
  - 回测页持仓组合/窗口按品种记忆：优先品种记录，缺省回退全局设置。

| `PUT` | `/api/backtest/prefs` |
|:--|:--|
  - **请求体**：`last_data_file` str | None · 可选（默认 None）；`last_strategy_file` str | None · 可选（默认 None）；`debug_mode` bool | None · 可选（默认 None）；`bg_animation` bool | None · 可选（默认 None）；`ai_provider` str | None · 可选（默认 None）；`ai_api_key` str | None · 可选（默认 None）；`ai_base_url` str | None · 可选（默认 None）；`ai_model` str | None · 可选（默认 None）；`bt_commission_pct` float | None · 可选（默认 None）；`bt_slippage_pct` float | None · 可选（默认 None）；`bt_hold_policy` str | None · 可选（默认 None）；`bt_window_bars` int | None · 可选（默认 None）；`bt_prefs_symbol` str | None · 可选（默认 None）；`paper_starting_balance` float | None · 可选（默认 None）；`paper_notional` float | None · 可选（默认 None）；`paper_commission_pct` float | None · 可选（默认 None）；`paper_slippage_pct` float | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`signal_threshold` float | None · 可选（默认 None）
  - 保存某品种的回测持仓组合/窗口记忆（symbol 走 bt_prefs.json）。

| `GET` | `/api/backtest/report` |
|:--|:--|
  - **查询参数**：`symbol`

| `POST` | `/api/backtest/start` |
|:--|:--|
  - **请求体**：`strategy_file` str · 必填；`data_file` str | None · 可选（默认 None）；`commission_pct` float | None · 可选（默认 None）；`slippage_pct` float | None · 可选（默认 None）；`hold_policy` str | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`signal_threshold` float | None · 可选（默认 None）

| `GET` | `/api/backtest/status` |
|:--|:--|

| `POST` | `/api/backtest/stop` |
|:--|:--|

| `POST` | `/api/backtest/threshold-sweep` |
|:--|:--|
  - **请求体**：`strategy_file` str · 必填；`hold_policy` str · 可选（默认 'signal'）；`data_file` str | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`commission_pct` float | None · 可选（默认 None）；`slippage_pct` float | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`thresholds` list[float] | None · 可选（默认 None）
  - 阈值敏感性：同一模型×方案在 0.05/0.3/0.5/0.8 无信号阈值下重跑离散撮合。


## `realtime`

| `GET` | `/api/realtime/feishu` |
|:--|:--|

| `PUT` | `/api/realtime/feishu` |
|:--|:--|
  - **请求体**：`enabled` bool | None · 可选（默认 None）；`webhook_url` str | None · 可选（默认 None）；`secret` str | None · 可选（默认 None）；`rt_alert_dev_pct` float | None · 可选（默认 None）；`rt_alert_stale_bars` int | None · 可选（默认 None）

| `POST` | `/api/realtime/feishu/test` |
|:--|:--|
  - **请求体**：`webhook_url` str | None · 可选（默认 None）；`secret` str | None · 可选（默认 None）

| `GET` | `/api/realtime/sources` |
|:--|:--|

| `POST` | `/api/realtime/start` |
|:--|:--|

| `GET` | `/api/realtime/status` |
|:--|:--|

| `POST` | `/api/realtime/stop` |
|:--|:--|

| `GET` | `/api/realtime/strategies` |
|:--|:--|
  - 已保存的 best_*.json 策略，供因子来源下拉。

| `POST` | `/api/realtime/tradingview/probe` |
|:--|:--|
  - Probe TradingView reachability (same behavior as PA_Agent before fetch).

| `POST` | `/api/realtime/unwatch` |
|:--|:--|
  - **请求体**：`id` str · 必填

| `POST` | `/api/realtime/watch` |
|:--|:--|
  - **请求体**：`source` str · 必填；`symbol` str · 必填；`timeframe` str · 必填；`strategy_file` str · 必填；`policy_id` str | None · 可选（默认 None）


## `ai`

| `POST` | `/api/ai/analyze-training` |
|:--|:--|
  - **请求体**：`provider` str | None · 可选（默认 None）；`api_key` str | None · 可选（默认 None）；`base_url` str | None · 可选（默认 None）；`model` str | None · 可选（默认 None）；`symbol` str | None · 可选（默认 None）

| `GET` | `/api/ai/providers` |
|:--|:--|


## `debug`

| `POST` | `/api/debug/client-log` |
|:--|:--|
  - **请求体**：`level` str · 可选（默认 'error'）；`message` str · 必填；`context` dict[str, typing.Any] | None · 可选（默认 None）

| `GET` | `/api/debug/logs` |
|:--|:--|
  - **查询参数**：`lines`, `level`, `q`
  - 读取服务端日志尾部; level=error|warning|info|debug 按级别过滤, q=关键词(大小写不敏感)。


## `symbols`

| `GET` | `/api/symbols/{symbol}` |
|:--|:--|


## `other`

| `GET` | `/` |
|:--|:--|


## `auth`

| `POST` | `/api/auth/rotate` |
|:--|:--|
  - 轮换 API 令牌 (需携带旧令牌); 新令牌立即生效, 其它设备需重新同步。

| `GET` | `/api/auth/token` |
|:--|:--|
  - 返回本机 API 令牌 (仅同源/局域网控制台/已知镜像来源可获取)。


## `dd-events`

| `GET` | `/api/dd-events` |
|:--|:--|
  - **查询参数**：`limit`
  - 最近 N 条 DD 熔断/收复事件（回溯每次熔断/收复）。


## `events`

| `GET` | `/api/events` |
|:--|:--|
  - SSE 事件流：实时/模拟盘状态变更推送（事件名=域，data=与 REST 同构快照）。


## `experiment`

| `POST` | `/api/experiment/compare` |
|:--|:--|
  - **请求体**：`data_file` str · 必填；`n_bars` int · 可选（默认 60000）；`chunks` int | None · 可选（默认 None）；`steps` int · 可选（默认 12）；`seeds` str · 可选（默认 '42'）；`regime` str · 可选（默认 'vol'）；`window_bars` int | None · 可选（默认 None）；`rep_criterion` str | None · 可选（默认 None）

| `GET` | `/api/experiment/compare-status` |
|:--|:--|

| `POST` | `/api/experiment/compare-stop` |
|:--|:--|


## `hold-policies`

| `GET` | `/api/hold-policies` |
|:--|:--|


## `paper`

| `POST` | `/api/paper/close` |
|:--|:--|
  - **请求体**：`id` str · 必填

| `POST` | `/api/paper/close-all` |
|:--|:--|

| `POST` | `/api/paper/dd-drill` |
|:--|:--|
  - DD 演练：隔离子进程注入合成暴跌→收复，走真实生产 dd 熔断路径。

| `POST` | `/api/paper/replay` |
|:--|:--|
  - **请求体**：`data_file` str · 必填；`strategy_file` str · 必填；`policy_id` str | None · 可选（默认 None）；`commission_pct` float | None · 可选（默认 None）；`slippage_pct` float | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`signal_threshold` float | None · 可选（默认 None）
  - 历史回放：本地 Parquet × 模型 × 持仓管理方案，离散撮合（与模拟盘同口径）。

| `POST` | `/api/paper/replay-compare` |
|:--|:--|
  - **请求体**：`data_file` str · 必填；`strategy_file` str · 必填；`policy_id` str | None · 可选（默认 None）；`commission_pct` float | None · 可选（默认 None）；`slippage_pct` float | None · 可选（默认 None）；`window_bars` int | None · 可选（默认 None）；`max_position_pct` float | None · 可选（默认 None）；`signal_threshold` float | None · 可选（默认 None）
  - 同参数回测对比：与历史回放完全相同的 数据/模型/持仓方案/窗口/成本，

| `POST` | `/api/paper/reset` |
|:--|:--|

| `POST` | `/api/paper/start` |
|:--|:--|

| `GET` | `/api/paper/status` |
|:--|:--|

| `POST` | `/api/paper/stop` |
|:--|:--|

| `POST` | `/api/paper/unwatch` |
|:--|:--|
  - **请求体**：`id` str · 必填

| `POST` | `/api/paper/watch` |
|:--|:--|
  - **请求体**：`source` str · 必填；`symbol` str · 必填；`timeframe` str · 必填；`strategy_file` str · 必填；`policy_id` str | None · 可选（默认 None）


## `other`

| `GET` | `/docs` |
|:--|:--|

| `GET` | `/docs/engines-compare` |
|:--|:--|
  - 回测 vs 回放 vs 模拟实盘口径对照表（各页 hint 链接到此处）。

| `GET` | `/docs/oauth2-redirect` |
|:--|:--|

| `GET` | `/openapi.json` |
|:--|:--|

| `GET` | `/redoc` |
|:--|:--|
