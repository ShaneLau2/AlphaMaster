# AlphaMaster

基于深度神经网络强化学习的量化因子挖掘中心：从本地 Parquet K 线自动搜索可解释因子公式，支持 Web 端训练、回测与实时信号分析。

**QQ 交流群：1063897401**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

![Web 控制台总览](docs/images/00_hero.png)

仓库地址：[github.com/rosemarycox5334-debug/AlphaMaster](https://github.com/rosemarycox5334-debug/AlphaMaster)

---

## 它做什么

AlphaMaster 把「挖因子」做成一条可操作的流水线：

1. **训练**：用强化学习在特征 + 算子空间里搜索公式，按验证集表现选优  
2. **回测**：用 `tanh(因子)` 连续仓位在历史行情上模拟交易，看资金曲线与绩效  
3. **实时分析**：按周期收盘后重算信号，展示方向与把握；方向转折可推飞书提醒  

公式以 token 序列保存（如 `strategies/best_BTCUSDT.json`），可用 StackVM 解释执行，训练 / 回测 / 实时共用同一套信号逻辑。

---

## Web 控制台（推荐入口）

```bash
pip install -r requirements.txt
python run_web.py --port 8765
```

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。界面分三步：

| 步骤 | 作用 |
|------|------|
| **01 模型训练** | 选 Parquet、开始 / 重新训练、看曲线与日志、导出策略与检查点 |
| **02 策略回测** | 选策略 JSON，设手续费 / 滑点，看绩效与资金曲线 |
| **03 实时分析** | 多数据源监控，收盘后更新信号；可选飞书转折提醒 |
| **04 模拟实盘** | 纸上交易：收盘信号翻转时模拟开 / 平仓，记录成交流水与资金曲线（不真下单）。回测/模拟盘可自由组合「模型 × 持仓管理方案」（信号跟随 / 止盈止损保护 / 熔断止损 / 保本追踪 / 时间止损 / 吊灯止损(ATR) / 回撤熔断(DD)），实时分析每卡带价格走势图与入场/止损/止盈参考线 |

### 模型训练

![训练页](docs/images/01_train.png)

- Parquet 命名：`{品种}_{周期}.parquet`，例如 `BTCUSDT_H1.parquet`、`XAUUSD_H1.parquet`  
- **开始训练**：有检查点则断点续训  
- **重新训练**：清除检查点从头搜索；已有更优策略作为分数下限，不会被弱结果覆盖  
- **训练数据范围**：默认用选中的全部 Parquet；可选「最近 N 根」（连续尾部切片）或「全历史分块 N 根」（按年代抽连续块覆盖不同行情，末尾自动保留近期连续段作验证/holdout）。子集确定性生成到 `data/slices/{模式}_{根数}/` 下，独立数据指纹 → 各子集 holdout 单次消费互不干扰；随机抽单根会破坏滚动特征，故只支持整块抽取  
- **全历史分块的 regime 分层**：spread 默认按滚动波动率分位选块（`regime=vol`，low/mid/high 各年代尽量入样），可选按趋势强度（`regime=trend`）或纯等分（`regime=equal`）；覆盖率写入元信息供审计  
- **训练范围溯源**：每个子集目录写 `train_range.json` sidecar（模式/参数/覆盖年代/regime 覆盖），策略 JSON 存 `train_range` 字段，策略表与回测页显示溯源徽标——老文件（无字段）按子集路径自动兜底解析  
- 展示最优分数、验证分数、训练曲线与最优公式；可选 AI 分析当前训练情况  
- **范围对比实验**（tail vs spread）：训练页「范围对比实验」面板或 `python scripts/compare_ranges.py --data-file <parquet> --n-bars 100000 --steps 200`，同源两路短训并输出双路 holdout 对比结论（`results/compare_*`）；`python scripts/experiment_old_vs_new.py` 做老/新数据主导的闸门通过率实验。实验用引擎 `run_tag` 隔离（历史/检查点带后缀、不写 `best_*.json`、不消费正式 holdout 指纹），结束后自清理  

### 策略回测

![回测页](docs/images/02_backtest.png)

- 仓位：`position = tanh(factor)`，信号越强仓位越大  
- 成本：手续费 + 滑点（默认约 0.02% / 0.01%）  
- 输出：总收益、夏普、索提诺、盈亏比、滚动夏普与资金曲线  

![资金曲线示例](docs/images/04_equity.png)

### 实时分析

![实时分析页](docs/images/03_realtime.png)

- 数据源：TradingView / Binance / OKX / 通达信（以界面可用源为准）  
- **只在当前周期 K 线收盘后**重新判断；未收盘 bar 不参与信号  
- 卡片展示方向（看涨 / 看跌 / 不确定）与把握程度  
- 可选飞书 Webhook：仅在方向转折时推送文字提醒  

### 模拟实盘（纸上交易）

![模拟实盘页示意](docs/images/03_realtime.png)

- 信号口径与实时分析完全一致，但把信号变成**离散模拟订单**  
- 方向翻转才成交：信号 bar 收盘价 ± 滑点成交，开 / 平仓各收一次手续费  
- 下单名义金额 = 「满仓名义金额」× |tanh(因子)|，同方向不加仓  
- 展示账户净值 / 现金 / 盈亏、持仓、成交流水与资金曲线；状态落盘、重启可恢复  
- **不连任何券商，不会真下单**（真实下单请另行接入券商实盘通道）  

---

## 项目结构

```
AlphaMaster/
├── web/                 # FastAPI Web UI（训练 / 回测 / 实时 / 模拟实盘）
├── model_core/          # 特征、算子、StackVM、训练引擎、回测评分
├── data_pipeline/       # Parquet K 线加载与对齐
├── strategy_manager/    # 实盘信号与仓位逻辑（与回测口径一致）
├── web/paper_manager.py # 模拟实盘引擎（纸上交易，结算/成交流水/资金曲线）
├── backtest_viz/        # 回测引擎与图表
├── strategies/          # best_{symbol}.json 策略文件
├── checkpoints/         # 训练检查点
├── run_web.py           # 启动 Web 控制台
├── train_file.py        # CLI：从单个 Parquet 训练
└── requirements.txt
```

---

## 环境要求

- Python **3.10+**（建议 3.11）  
- PyTorch、pandas、FastAPI、uvicorn 等（见 `requirements.txt`）  
- 行情数据用本地 Parquet（`data/training` / `data/slices`），或在界面用数据源在线下载  
- 复制 `.env.example` 为 `.env` 按需填写凭证（`.env` 已 gitignore）  

```bash
python -m pip install -r requirements.txt
# TradingView 若上面 git 行失败，可单独装：
# python -m pip install git+https://github.com/rongardF/tvdatafeed.git
```

---

## 常用命令

```bash
# Web 控制台
python run_web.py --port 8765

# CLI 训练（自动续训；加 --from-scratch 则重新训练）
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet --from-scratch
```

策略输出默认在 `strategies/best_{symbol}.json`。

---

## 信号口径（训练 / 回测 / 实时一致）

- 因子经 StackVM 算出标量序列  
- `position = tanh(factor)` ∈ (-1, 1)  
- `|position|` 小于阈值时视为无信号（观望）  
- 实时侧只用**已收盘** K 线，避免盘中抖动与回测不一致  

---

## 截图更新

仓库内展示图由当前 Web UI 截取，可用：

```bash
python scripts/capture_readme_shots.py
```

（需本机已启动 `python run_web.py --port 8765`，并已安装 Playwright + Chromium。）

---

## License

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)。  
修改、分发或通过网络提供服务时，须按相同协议公开对应源代码。

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=rosemarycox5334-debug/AlphaMaster&type=date&legend=top-left)](https://www.star-history.com/#rosemarycox5334-debug/AlphaMaster&type=date&legend=top-left)
