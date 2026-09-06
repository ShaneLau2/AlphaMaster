# 方案设计:多标的 × 多周期共享策略(Mode B/C)— 设计定稿 v2

日期: 2026-09-05 · 状态: 已批准,进入实现 · 前置事实全部来自本仓库实测
(parquet 逐文件计算 + AGENTS.md/engine 常量)

---

## 0. 结论摘要(TL;DR)

1. 同一套代码库内训练**多套 asset-class policy**:架构/词表/StackVM/RL 引擎/
   OOS-P0.2 闸门全共享,**policy 权重不共享**。
2. **Phase 1 = Crypto AlphaGPT**(H1:BTC/ETH/SOL/ADA/XRP),先证明多 ticker
   泛化;Phase 2 = Equity AlphaGPT(多行业多市值,单独训练);Phase 3(远期)=
   Meta-AlphaGPT(Crypto+Equity+XAU+FX + market/regime token)。
3. **5 分钟(M5)不进共享池**:成本 0.03%/边 占每 bar 平均波动 H1=6.8%、M5=22.7%、
   M1=73.6%;950k 根全文件 ≈ 600s/步。M1 结构性为负 + 仅 2.2y 历史。
4. 大周期不是"没数据",是**去相关深历史 H1 没下载**(M0 任务)。
5. 关键 RL 设计:√N 加权 market-batch + 每 K 步全市场 calibration;奖励按
   **市场族(group)等权聚合**防相关标的多计票;min-gate 默认开,λ 惩罚项默认 0
   待消融;consistency 先监控后入奖励;Leave-Market-Out 验证泛化。
6. 奖励统一**年化 Sharpe**(引擎 `estimate_periods_per_year` 已按时间戳估每年
   bar 数 → 跨 TF/跨市场同单位),不用原始 val。

---

## 1. 架构总览

```
                 AlphaGPT Architecture (d≈96, 单代码库)
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
 Crypto Policy          Equity Policy         [Phase3] Meta Policy
 (Phase 1)              (Phase 2)              (Crypto+Equity+XAU+FX,
    │                     │                     market/regime token)
 BTC ETH SOL ADA XRP  跨行业多市值股票池(H1)
    │                     │
 H1 数据               H1 数据(复权)
    │                     │
 Crypto reward        Stock reward
```

共享不复制:`FORMULA_VOCAB`、Transformer 结构、`StackVM`(含 `execute_batched`)、
65 特征核心框架、RL engine(REINFORCE + EMA baseline + restart 升级)、reward 框架、
OOS/holdout/P0.2 跨标的闸门。差异只在:**policy 权重** + **market registry(每
policy 一份)** + 少量市场结构特征(Phase 2:时段/隔夜 gap/复权)。

## 2. 数据盘存(实测)

| 文件 | bar 数 | 跨度 | 成本占每bar波动 | 角色 |
|---|---|---|---|---|
| BTCUSDT_H1 | 79,195 | 9.0y | 6.8% | crypto 池主腿 |
| ETHUSDT_H1 | 19,999 | 2.3y | 6.6% | crypto |
| SOLUSDT_H1 | 19,999 | ~3.3y | 5.3% | crypto |
| ADAUSDT_H1 | 29,999 | 3.4y | 5.0% | crypto |
| XRPUSDT_H1 | 29,999 | ~3.4y | 5.9% | crypto |
| XAUUSD_H1 | 4,999 | 0.8y | 12.5% | OOS 复核(eval-only, 待补深) |
| 600519_D1 / 300750_D1 | 6k / 2k | 25y / 8.2y | 2.1% / 1.4% | equity 占位 |
| BTCUSDT_M5/M1 | 950k/1.15M | 9.1y/2.2y | 22.7%/73.6% | 不进共享池 |

## 3. 多市场 RL 设计(v1 吸收评审)

### 3.1 采样:固定覆盖 + 加权随机
- 普通步:按 `P(market) ∝ N^α`(α 默认 0.5)加权随机抽 `MARKET_BATCH`(默认 2)个市场;
- 每 `MARKET_CALIB_EVERY`(默认 20)步:强制**全市场 calibration 步**(所有市场各评估
  一次,奖励聚合后单次 update)——防市场长期饿死,兼做 fold 同步锚。
- `MARKET_MIN_BARS`(默认 8k 或 ~1.5y)门槛:不足的标的不进训练腿(eval-only),
  可作 P0.2/LMO 复核。

### 3.2 奖励:市场族等权聚合 + min-gate + (默认关)λ 项
- 每市场:年化 Sharpe(IC 闸门后,生产口径 pnl)。
- 族内 mean → **族间等权 mean**:`reward = mean_group(mean_m∈g(Sharpe_m))`,
  防止 5 个相关 crypto ticker 重复计票(crypto 现仅一族,结构先就位)。
- **min-gate 默认开**:任一市场 Sharpe 低于门槛即该公式不更新 best/finalist。
- `λ1·max(0,−min)`、`λ2·std`、`λ3·consistency`(正 Sharpe 市场占比/离散度):
  默认 0 = 关,先跑基线再消融;consistency 指标从第一天**监控**。
- 采样概率与数据长度解耦(√N),但仍受 min-bars 门槛约束。

### 3.3 精英池与终局
- 精英池共享(key=公式,per-market 分向量);restart/熵逻辑沿用单市场机制。
- 冠军 = 聚合分最高;须**每个训练市场 holdout + P0.2 跨标的复核全过**才逐
  `best_{sym}.json` 部署同一公式;任一不过 → reject_restore,理由入
  champion_history(带 per-market 明细)。
- **Leave-Market-Out(LMO)**:训练留出 1–2 个成员 → 冻结 policy → 测留出成员;
  cross-family LMO(crypto 训、XAU/equity 测)回答"学会了公式 vs 记住了市场"。
  P0.2 = 部署期 LMO 的运行时实例,按数据指纹轮换 ticker。

### 3.4 周期(多 TF)
一期全部 H1(窗语义统一)。二期(可选):M5 用切片做独立 M5 policy(同代码路径、
不同 registry),不与 H1 混训——滚动窗参数语义随 TF 漂移,硬混不可比。M1 不做。

## 4. 里程碑与验收

| 步 | 内容 | 验收 |
|---|---|---|
| M0 | market registry + 去相关 H1 下载清单 | registry json + 数据就位 |
| M1 | MarketEnv 层(data_pipeline) | 单市场回归等价(全绿) |
| M2 | 共享 policy 多市场训练循环 + `train_multi.py` | 微型 2 市场训练跑通;单市场零变化 |
| M2b | 每市场终局闸门(holdout+P0.2)+ LMO 脚本 | 冠军全过才部署;LMO 报告 |
| M3 | 对照实验 runner(A vs B / α / reward 消融) | 命令就绪,用户后台跑 |
| M4 | 前端(Web 控制台):policy 选择/多市场曲线/跨市场徽标 | M2b 数据面定型后 |

## 5. 实现纪律(沿用全 session)
- 单市场默认路径零行为变化:多市场仅当 registry 显式配置才激活。
- 实验一律 `_suppress_deploy=True` + run_tag 隔离 + 不消费 holdout 指纹。
- 回归:既有单测 + `scripts/verify_batched_equiv.py` 等价冒烟。
