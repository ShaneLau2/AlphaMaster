AlphaMaster 仓库速读

**开源协议**：GNU Affero General Public License v3.0 (AGPL-3.0)。修改、分发或通过网络提供服务时，须按相同协议公开源代码。详见根目录 `LICENSE`。

这是一套「因子挖掘 + 回测 + 模拟实盘/实时信号」量化系统。核心思路是：用 Transformer 自动生成可解释的因子公式，通过回测打分筛选，再把高分公式用于 Web 端的实时信号与模拟实盘。

代码组织（按功能划分）
- web/：FastAPI Web UI（训练 / 回测 / 实时分析 / 模拟实盘 / 数据下载）
- model_core/：策略挖掘。特征工程、算子 DSL、StackVM 执行、AlphaGPT 训练与回测评分
- data_pipeline/：本地 Parquet K 线加载与对齐（含训练子集 tail/spread 抽样）
- strategy_manager/：信号生成与持仓方案（live_signal / hold_policy）
- backtest_viz/：回测引擎、图表与报告
- scripts/：实验与运维脚本（hold_matrix、dd_alert_drill 等）

主流程（从数据到信号）
1) 数据：界面下载或本地 Parquet（data/training、data/slices）
2) model_core 训练生成最优公式（strategies/best_{symbol}.json，冠军闸门通过才部署）
3) strategy_manager / web 读取公式，计算因子方向信号
4) 回测/模拟实盘以离散撮合引擎结算（模拟实盘不真下单）
5) 持仓管理方案（dd 回撤熔断 / chandelier 吊灯等）叠加风控出场

核心思想
- 不是直接预测价格，而是「生成公式 → 解释执行 → 回测评分 → 优化生成器」
- 公式 = token 序列；token 由「特征 + 算子」组成，StackVM 执行成因子信号
- 交易层只消费最终信号分数，负责风控与执行

现状与依赖
- Python 3.10+、PyTorch；行情数据用本地 Parquet 或界面在线数据源
- 策略 JSON 需先训练生成，或使用仓库内已有的 strategies/best_*.json
- MT5 终端相关能力已移除（Windows 专属、本机不可用），旧代码已从工作区删除（存档于 git 历史提交）
