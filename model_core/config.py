"""
model_core/config.py — 模型层配置

仅保留模型训练所需的参数。
品种、数据、风控等全局配置统一由根目录 config.py 的 Config 类管理。
"""
import math

import torch
from .vocab import FORMULA_VOCAB


class ModelConfig:
    # ── 训练设备 ─────────────────────────────────────────────────────────
    # 注意：本任务 CPU 训练速度反而比 GPU 快（实测约 2.3 倍），故强制用 CPU。
    # 原因：
    #   1. 张量太小——forex 组仅 (2 品种 × 3508 × 20 特征)，单个算子的
    #      计算量小于 CUDA kernel 启动开销（数十微秒），GPU 算得快但启动慢。
    #   2. 训练循环是 Python 串行调度：每 step 逐条跑 128 条公式 × 8 个
    #      VM 步 × 4 个 walk-forward 折，GPU 被切成上万个碎片段，吃不满。
    #   3. host↔device 拷贝 + kernel 启动延迟主导总耗时，而非张量计算本身。
    #   4. 实测 GPU 利用率 ~51%，正是 GPU 一半时间在干等 Python 喂下一个
    #      kernel 的证据（不是“还能压榨”，而是“调度瓶颈”）。
    # 基准测试（forex 组, 50 步, RTX 4060, 2026-07-03）：
    #   cuda: 4.48 s/步  Best=4.875
    #   cpu : 1.91 s/步  Best=5.103
    #   加速比 = 0.43x（GPU 反而慢 2.3 倍）
    # 若后续改为批量并行公式评估（一次喂大批张量进 GPU），再切回 cuda。
    DEVICE = torch.device("cpu")

    # ── 训练参数（大搜索空间适配版，2026-07-04 重构）─────────────────────
    # 背景：特征库扩展到 65、算子库扩展到 66（vocab=131），8-token 搜索空间
    #   从旧版 ~7亿 暴增到 ~8.67×10^16（1.2 亿倍）。旧的采样预算（128×3000）
    #   覆盖率趋近于零，导致熵坍塌 Early Stop、公式退化。
    # 对策（训练时间不敏感场景）：
    #   1. 特征剪枝（active_features.json）把 vocab 降到 ~90，空间缩小约 20 倍
    #   2. 放大采样预算：BATCH_SIZE 128→256，TRAIN_STEPS 3000→8000
    #   3. 更大精英池（60）保留更多历史最优
    BATCH_SIZE      = 192   # 每步采样公式数（原 128，1.5x 提升覆盖率）
    TRAIN_STEPS     = 9000  # 每组训练步数（55次重启需要更多步数）
    # 实时训练曲线 JSON 的整份重写间隔（步）。9000 步长跑若每步重写会累计
    # ~12GB 磁盘写（history 不断变大）；默认每 50 步写一次 → ~0.2GB。
    # 终局/收尾快照（holdout 判定、run 结束）不受此限（force=True 恒写）。
    HISTORY_LIVE_EVERY_STEPS: int = 50
    MAX_FORMULA_LEN = 8     # 公式长度上限：保持 8（10 会导致 CPU 训练慢 3 倍）

    # ── 样本外（Holdout）预留与闸门（P0 加固）─────────────────────────
    # 训练尾部最后 N 根 K 线完全保留：不进 walk-forward 折叠、不参与任何
    # 评分与选优，只在训练结束后用最优公式验证一次（_verify_holdout），
    # 用于衡量真正的泛化保持率（holdout 分 / 最优验证分）。
    HOLDOUT_BARS: int = 500

    # holdout 闸门：验证不通过时【不覆盖】已部署的 strategies/best_{sym}.json，
    # 保留原冠军并记录到 champion_history（拒绝原因可追溯）。
    HOLDOUT_GATE_ENABLED: bool = True
    HOLDOUT_MIN_SCORE:    float = 0.0     # holdout 分（训练同口径）必须 > 0
    HOLDOUT_MIN_RATIO:    float = 0.30    # holdout 分 / 最优验证分（保持率）下限
    HOLDOUT_MIN_SHARPE:   float = 0.0     # holdout 生产口径 Sharpe 下限
    # 同数据版本 holdout 只可消费一次：指纹相同且已批准过其他公式时，
    # 新公式不得再用 holdout 批准（防止反复重训挑过拟合的 holdout 彩票）。
    HOLDOUT_SINGLE_USE:   bool = True
    HOLDOUT_STATE_FILE:   str = "data/holdout_state.json"
    CHAMPION_HISTORY_FILE: str = "strategies/champion_history.json"

    # ── 冠军：真实回测选优（P0）──────────────────────────────────────
    # val 只用于入围（top-K），最终冠军在从没参与选优的 holdout 尾部窗口上
    # 用生产配置（成本 + tanh 仓位）跑真实回测，赢了才当 champion。
    FINALIST_TOP_K:      int   = 20
    FINALIST_COST_RATE:  float = 0.0003   # 与 run_backtest.py 默认一致：佣金 0.02% + 滑点 0.01%
    FINALISTS_JSON:      str = "strategies/finalists_{symbol}.json"

    # ── 多重比较校正（P1，任意组合可选开）─────────────────────────────
    ENABLE_NULL_MODEL_CHECK:  bool  = True   # 空模型 99 分位：要求冠军 val 显著高于随机公式
    NULL_MODEL_N:             int   = 64     # 随机基准公式数（一次评估 ≈ 1 个训练 step 的量）
    NULL_MODEL_QUANTILE:      float = 0.99
    ENABLE_FOLD_SE_CHECK:     bool  = True   # 冠军 vs 次优：逐折 val 差 > k×SE（防单折碰运气）
    CHAMPION_SE_K:            float = 1.0

    # ── 选优层 vol 覆盖检查（regime_ada 诊断落地）──────────────────────
    # 候选须在低/中/高 vol 三段都【不显著为负】（生产口径 pnl 每段单尾 t 统计
    # ≥ 下限）才允许更新 best / 进 top-K 入围——拦下“只有高 vol 能活”的公式族
    # （如 critic 冠军 AMIHUD_ILLIQ+SIGN：中/低 vol 段持续失血），与公式族无关。
    # 作用层：选优（best/finalist），不改 REINFORCE reward，不消费 holdout。
    VOL_COVERAGE_ENABLED:  bool  = True     # 总开关（关 = 与旧行为完全一致）
    VOL_COVERAGE_WINDOW:   int   = 48       # 因果 realized-vol 窗口（与 regime 诊断一致）
    VOL_COVERAGE_MIN_BARS: int   = 200      # 段/格内最少成熟 bar 数，不足则跳过该判定
    VOL_COVERAGE_MIN_T:    float = -1.645   # 每段/格 pnl 单尾 t 下限（≈95% 单侧）

    # 档位：vol×er 3×3 格级检查（默认，2026-09-05 起从段级翻上）。
    # 任一格（样本≥VOL_COVERAGE_MIN_BARS）显著为负即拒，可拦下“段级聚合把单格
    # 出血抹平”的候选（验证见 results/vol_grid_gate_upgrade.md：段级放行的
    # e3_curriculum_final/e4_scratch 冠军在格级被样本内拦下，健康冠军误伤 0 例）。
    # 需要复刻旧段级口径（如与历史 tercile 结果对比）时，实验 runner 用
    # --vol-gate tier 显式钉住；本默认只影响未显式指定的运行。
    VOL_COVERAGE_GRID:      bool = True     # True = 格级(9格)；False = 段级(3段)
    VOL_COVERAGE_ER_WINDOW: int  = 120      # 效率比(趋势强度)窗口，与 regime_health 一致

    # ── 部署前跨 ticker OOS 复核（P0.2）──────────────────────────────
    # 冠军在同文件 holdout 通过后，还须在第二个标的（默认 ADAUSDT_H1）上通过
    # vol 覆盖才可部署——拦“训练文件干净、跨文件失血”的公式（如 e4_bc 冠军：
    # BTC 训练窗无出血格，但在 ADA 中/低vol 段显著为负）。口径与选优闸门共用
    # （grid/tier 随 VOL_COVERAGE_GRID；t 下限 VOL_COVERAGE_MIN_T）。
    # 只跑真实部署路径（_suppress_deploy 的实验不触发），不消费任何预留窗口；
    # 文件缺失/加载失败/公式常量 → fail-open 放行（不误杀部署）。
    CROSS_TICKER_OOS_ENABLED:      bool   = True
    CROSS_TICKER_OOS_FILES:        tuple  = ("data/training/ADAUSDT_H1.parquet",)
    CROSS_TICKER_OOS_SKIP_MISSING: bool   = True  # 文件不存在 → 跳过该文件

    # ── 冠军稳健性复核（P3，仅拒后触发）──────────────────────────────
    # 冠军未过 holdout 闸门时，对 top-K finalists 自动跑成本/折叠/起点三轴
    # 敏感性（model_core/robustness.py），结论随拒绝记录写入 champion_history，
    # 不改部署决策，只提供“这次拒绝是否稳健”的复核证据。
    ENABLE_FINALIST_ROBUSTNESS: bool = True
    ROBUSTNESS_TOP_K:           int  = 3        # 参与分析的 top finalists 数
    ROBUSTNESS_COST_MULTS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)  # 成本倍数
    ROBUSTNESS_SHIFT_FRACS: tuple[float, ...] = (0.05, 0.10, 0.20)        # 头部丢弃比例

    # ── 双 seed 升级纪律（P1）─────────────────────────────────────────
    # train_file --seeds A,B：每个 seed 完整训练一轮，全部 seed 的 val 都超过
    # 旧冠军 + CHAMPION_UPGRADE_MARGIN 才允许部署新冠军。
    CHAMPION_UPGRADE_MARGIN:  float = 0.02

    # ── 特征维度（由 vocab.py 自动派生，无需手动修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 为主，IC 做门控 ──────────────────────────────────
    # IC_NEG_MULT 0.30→0.50：0.30 对反向因子惩罚过重，可能误杀非线性高收益因子。
    # 收益优先模式下，只要年化收益是正的，适当负 IC 可以接受。
    # ── 中性带正则（观望区偏好，默认关闭）─────────────────────────────────
    # 让模型在 |factor| < NEUTRAL_BAND_HALF 的区间停留获得额外奖励：
    # 目标是训练出真正会输出 FLAT/观望 的因子，而不是永远单边满仓。
    # 权重调大后，中性带内的因子会比区间外同收益因子得分更高。
    NEUTRAL_BAND_HALF: float = 0.5    # |factor| 低于该值视为“中性/观望区”
    NEUTRAL_BAND_W:    float = 0.0    # 中性带奖励权重（0=关闭；建议 0.1~0.5 起步）
    REWARD_ALPHA:      float = 1.0
    IC_GATE_THRESH:    float = 0.01
    IC_GATE_MULT:      float = 1.15
    IC_NEG_MULT:       float = 0.75   # 收益优先：不过度误杀反向/非线性高收益因子

    # ── FTMO 专属奖励模式 ─────────────────────────────────────────────
    # "standard": 收益+风险平衡（默认，原权重）
    # "ftmo":     FTMO 考试盘专属——年化收益权重 0.60→0.75，Calmar 0.05→0.10
    #             （控制 MDD 贴近 10% Max Loss 上限），其余指标权重下调。
    #             目标：在 10% Max Loss 约束下最大化年化收益，快速达标。
    # "forex":    外汇均值回归专属（2026-07-08）——
    #             降年化收益权重(0.80→0.25)、提IC权重(0.03→0.25)、
    #             新增反转奖励(0.20，奖励低/负因子自相关)和多空对称检查(0.15)。
    #             原因：外汇H1以震荡为主，趋势算子效果差，需引导模型偏好
    #             均值回归信号而非追涨杀跌。
    REWARD_MODE:       str = "ftmo"

    # ── 熵保护（大空间加强版）──────────────────────────────────────────
    # ENTROPY_COEFF_MAX 0.5→1.0：加倍探索压力，对抗大 vocab 的过早收敛。
    # ENTROPY_COLLAPSE_THRESH 改为相对阈值 0.15×ln(vocab)：大 vocab 最大熵更高
    #   （ln(131)≈4.87 vs ln(54)≈3.99），绝对阈值 0.5 不再合理。
    # ENTROPY_COLLAPSE_STEPS 15→40：给模型更长的自我恢复窗口，不急于重启。
    ENTROPY_COEFF_MAX:   float = 1.0
    ENTROPY_COEFF_POWER: float = 1.0  # 降低幂次，让低熵时系数更激进（原1.3）
    ENTROPY_COLLAPSE_THRESH: float = 0.15 * math.log(FORMULA_VOCAB.size)
    ENTROPY_COLLAPSE_STEPS:  int   = 20  # 更快检测坍塌并重启

    # ── 熵下限惩罚（Fix 1: H→0 时熵项归零问题）──────────────────────────
    # 当 H < ENTROPY_FLOOR_THRESH 时，加入固定惩罚 λ×(thresh-H)。
    # 这确保即使 mean_ent→0，loss 中仍有非零探索压力。
    ENTROPY_FLOOR:        bool  = True
    ENTROPY_FLOOR_THRESH: float = 1.0   # 熵低于此值时触发固定惩罚（提高介入时机）
    ENTROPY_FLOOR_LAMBDA: float = 5.0   # 惩罚强度系数（加大力度对抗坍塌）

    # ── Elite Replay ──────────────────────────────────────────────────
    ELITE_REPLAY_FRAC:  float = 0.25
    ELITE_POOL_SIZE:    int   = 60    # 30→60：大空间需要更大的精英记忆
    ELITE_REWARD_SCALE: float = 1.2

    # ── 坍塌重启（大空间加强版）─────────────────────────────────────────
    # MAX_RESTARTS 8→25→55、RESTART_NOISE 0.05→0.1→0.25：时间不敏感，多给机会+更强扰动。
    # 配合 engine.py：超过 MAX_RESTARTS 后不再 Early Stop，改为强扰动继续训练。
    # 2026-07-09: US100 训练 24/25 重启仍有突破，扩到 55 次。
    MAX_RESTARTS:   int   = 55
    RESTART_NOISE:  float = 0.25

    # ── 自适应噪声：Best 停滞时自动增大扰动 ─────────────────────────────
    # stagnation_window: 判断停滞的步数窗口
    # noise_min / noise_max: 噪声下界和上界
    # noise_boost: 停滞时噪声提升倍率
    ADAPTIVE_NOISE:      bool  = True
    STAGNATION_WINDOW:   int   = 500
    NOISE_MIN:           float = 0.15
    NOISE_MAX:           float = 0.60
    NOISE_BOOST_FACTOR:  float = 2.0   # noise += 0.2 * (stagnation / window)

    # ── 重启多样性（Fix 2: best_snapshot 吸引子效应）─────────────────────
    # 每 FULL_RESET_EVERY 次重启中，做 1 次完全随机初始化而非从 best_snapshot 恢复。
    FULL_RESET_EVERY:    int   = 3     # 每 3 次重启中第 3 次做 full reset

    # ── 重启升级（E1 critic 吸引子锁死诊断落地）─────────────────────────
    # 若自上次重启以来 best 无任何刷新（说明上次“部分层重启”没逃出 best_snapshot
    # 吸引子），下次重启【强制完全重置】——不再只依赖 熵<0.3 或 FULL_RESET_EVERY
    # 的固定周期。E1 critic 臂 81 步封顶后两次部分重启（熵 0.337/0.463）都没回升的
    # 根因正是缺一次 full reset（熵差一点没触发、1&2 号重启都不是 3 的倍数）。
    RESTART_ESCALATE_FULL: bool = True  # 重启无收益→升级为完全重置

    # ── Reward baseline（Fix 3: 全负 batch 相对优选问题）──────────────────
    # 用 EMA baseline 替代 batch mean 计算 advantage，避免全负 batch 的问题。
    REWARD_EMA_BASELINE:     bool  = True
    REWARD_EMA_DECAY:        float = 0.95   # EMA 衰减系数
    REWARD_EMA_WARMUP:       int   = 10     # 前 N 步用 batch mean（EMA 未稳定）

    # ── 重启时部分重置参数：保留底层，扰动顶层 ───────────────────────────
    PARTIAL_RESET:       bool  = True
    PARTIAL_RESET_LAYERS: tuple = ("ln_f", "mtp_head", "head_critic", "blocks", "token_emb")

    # ── Elite Replay 衰减：旧 elite 采样权重随时间衰减 ──────────────────
    ELITE_DECAY:         bool  = True
    ELITE_DECAY_HALF_LIFE: int = 300   # 每 300 步旧 elite 权重减半

    # ── 多起点并行（Island）──────────────────────────────────────────────
    # 注意：Island 模式在 CPU 训练下会让总时间变成 N 倍（islands 串行），
    # 对于 index 这类大数组（T=32076）会变得极慢。当前默认关闭，保留配置开关。
    N_ISLANDS:              int   = 1
    MIGRATION_INTERVAL:     int   = 500
    MIGRATION_TOP_K:        int   = 5
    # island 默认关闭，避免用户误开导致速度爆炸

    # ── 因子去相关参数 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 25
    CORR_THRESHOLD:   float = 0.85
    CORR_PENALTY:     float = 0.8

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    # P2-6 修复说明：gap 必须按 target_horizon 标定，且与 CPCV purge gap 统一。
    # 当前 target_horizon=2（data_manager 用 log(open[t+2]/open[t+1]) 作 target_ret），
    # gap=20 相当于 10 个「真实预测步」——对 H1 数据足够，但若切换到 D1/15min
    # 需要重新评估。CPCV 模式（未实现）的 purge gap 应等于 WF_GAP，避免两套阈值。
    # 调整时同时检查：
    #   1. engine.py _build_walk_forward_folds 的 val_start = train_end + gap
    #   2. fold_size 应 >> gap（建议 fold_size >= 5*gap）以保证 val 有效性
    WF_GAP: int = 20

    # ── 公式结构约束（2026-07-05 新增）──────────────────────────────────
    # 背景：index 组因子因 TS_RANK 连续使用退化为 beta 因子（91.8% 做多），
    # 前半段市场跌亏钱、后半段市场涨赚钱，不是 alpha 而是 beta。
    # 对策：在采样阶段禁止恒正算子链，在评分阶段添加 beta 中性 + 前后一致性奖惩。
    ENABLE_FORMULA_STRUCTURE_CONSTRAINT: bool = True   # 总开关
    BETA_NEUTRAL_PENALTY:     bool  = True             # 多空比例失衡惩罚
    HALF_CONSISTENCY_BONUS:   bool  = True             # 前后一致性奖惩
    BETA_NEUTRAL_THRESH:      float = 0.85             # 超过此比例同方向触发重罚
    BETA_NEUTRAL_LIGHT_THRESH: float = 0.70            # 轻度失衡阈值

    # ── 并行评估（已实测关闭）──────────────────────────────────────────
    # 曾尝试将 Part C 公式评估并行化到 ThreadPoolExecutor，设想 PyTorch CPU
    # 算子释放 GIL 可多线程并行。但基准测试（8 逻辑核，5 品种，T=6000）证明
    # 并行反而更慢：
    #   串行(intra=8)          21.8 s/步   ← 最快
    #   并行 8 workers×8 intra  33.9 s/步   (+55%)
    #   并行 4 workers×2 intra  36.7 s/步   (+69%)
    #   并行 8 workers×1 intra  >60 s/步    (更差)
    # 根因：每条公式的 Python 编排（遍历 WF 折、.item()、IC、惩罚）全程持 GIL，
    # 释放 GIL 的张量核算子太小，收益抵不过线程池 + 线程超订阅开销；串行下
    # 每个算子用满 intra-op 线程池反而最高效。故保持 False。
    # 串行回退路径复用 _eval_formula_task，逻辑与并行完全一致，仅执行方式不同。
    PARALLEL_EVAL:        bool = False
    EVAL_WORKERS:         int  = 0    # 0=auto (physical cores, cap 8)
    EVAL_INTRA_THREADS:   int  = 0    # 0=auto (physical cores)

    # ── 批量 VM 评估（E2 可行性验证 → 生产接入）────────────────────────
    # StackVM.execute_batched 把同「位置骨架」的公式合并成 [B,T] 一次求值：
    # 特征槽一次高级索引取 B 行、同算子一次调用、nan/inf 一次 [B,T] 归约，
    # 实测（proto_batched_vm）≈8.65x（递归算子密集子集 7.7x）。
    # 语义等价：输出与逐条 execute 逐元素一致（CS_* 在 B>1 时逐行应用）；
    # 仅 N=1 单品种启用，N>1 / 异常形状自动退化为逐条。
    # 保留串行回退开关：BATCHED_EVAL_ENABLED=False 即回退原逐条路径。
    BATCHED_EVAL_ENABLED: bool = True

    # ── M2 多市场共享 policy 训练（Mode C, 2026-09-05）────────────────────
    # MultiMarketEngine(model_core/multimarket.py)：一份 policy 权重训练多市场。
    # 普通步随机采样 MARKET_BATCH_SIZE 个市场（P ∝ N^alpha，与数据量脱钩），
    # 每 MARKET_CALIB_EVERY 步做一次全市场 calibration（冠军晋升只在 calib 步）。
    # 奖励聚合：年化 Sharpe 按 group 等权（组内 mean 再组间 mean）；min 门槛
    # （任一市场聚合分 < MARKET_MIN_SHARPE）否决冠军晋升，不改 policy 梯度。
    # 单市场 AlphaEngine 完全不受影响（本组参数只被 train_multi.py 读取）。
    MULTI_MARKET_ENABLED: bool  = False    # 仅 train_multi.py 显式置 True
    MARKET_BATCH_SIZE:    int   = 2        # 普通步采样的市场数
    MARKET_CALIB_EVERY:   int   = 20       # 每 K 步全市场 calibration（冠军晋升步）
    MARKET_SAMPLE_ALPHA:  float = 0.5      # P(market) ∝ N^alpha；0=等权 1=按数据量
    MARKET_MIN_BARS:      int   = 8000     # 训练腿最低 bar 数（不足 → eval-only）
    MARKET_MIN_SHARPE:    float = 0.0      # min 门槛：任一市场聚合年化 Sharpe 低于此值不晋升
    MARKET_GROUP_MEAN:    bool  = True     # True=组级等权聚合；False=全市场平权 mean
    MARKET_POLICY:        str   = "crypto"
    MARKET_SEED:          int   = 42
